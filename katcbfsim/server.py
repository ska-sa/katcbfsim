from __future__ import print_function, division
import trollius
from trollius import From
import katcp
import katpoint
import tornado
import logging
import functools
import ipaddress
from katcp import Sensor
from katcp.kattypes import Str, Float, Int, Address, request, return_reply
from katsdptelstate.endpoint import Endpoint, endpoint_list_parser
import katcbfsim
from . import transport
from .stream import (Subarray, FXStream, BeamformerStream,
                     CaptureInProgressError, IncompleteConfigError, UnsupportedStreamError)
from .source import Source

logger = logging.getLogger(__name__)


def to_tornado_future(trollius_future, loop=None):
    """Modified version of :func:`tornado.platform.asyncio.to_tornado_future`
    that is a bit more robust: it allows taking a coroutine rather than a
    future, it passes through error tracebacks, and if a future is cancelled it
    properly propagates the CancelledError.
    """
    f = trollius.ensure_future(trollius_future, loop=loop)
    tf = tornado.concurrent.Future()
    def copy(future):
        assert future is f
        if f.cancelled():
            tf.set_exception(trollius.CancelledError())
        elif hasattr(f, '_get_exception_tb') and f._get_exception_tb() is not None:
            # Note: f.exception() clears the traceback, so must retrieve it first
            tb = f._get_exception_tb()
            exc = f.exception()
            tf.set_exc_info((type(exc), exc, tb))
        elif f.exception() is not None:
            tf.set_exception(f.exception())
        else:
            tf.set_result(f.result())
    f.add_done_callback(copy)
    return tf


def _stream_request(wrapped):
    """Decorator for per-stream commands. It looks up the stream and passes
    it to the wrapped function, or returns a failure message if it does not
    exist.
    """
    @functools.wraps(wrapped)
    def wrapper(self, sock, name, *args, **kwargs):
        try:
            stream = self._streams[name]
        except KeyError:
            return 'fail', 'requested stream name "{}" not found'.format(name)
        return wrapped(self, sock, stream, *args, **kwargs)
    return wrapper

def _stream_exceptions(wrapped):
    """Decorator used on requests that and turns exceptions defined in
    :py:mod:`stream` into katcp "fail" messages.
    """
    def wrapper(*args, **kwargs):
        try:
            return wrapped(*args, **kwargs)
        except (CaptureInProgressError,
                IncompleteConfigError,
                UnsupportedStreamError) as e:
            return 'fail', str(e)
    functools.update_wrapper(wrapper, wrapped)
    return wrapper


def endpoints_to_str(endpoints):
    """Convert a list of endpoints into a compact string that generates the
    same list. This is the inverse of
    :func:`katsdptelstate.endpoint.endpoint_list_parser`.
    """
    # Partition the endpoints by type
    ipv4 = []
    ipv6 = []
    other = []
    for endpoint in endpoints:
        # ipaddress module requires unicode, to convert if not already
        host = endpoint.host.decode('utf-8') if isinstance(endpoint.host, bytes) else endpoint.host
        try:
            ipv4.append(Endpoint(ipaddress.IPv4Address(host), endpoint.port))
        except ipaddress.AddressValueError:
            try:
                ipv6.append(Endpoint(ipaddress.IPv6Address(host), endpoint.port))
            except ipaddress.AddressValueError:
                other.append(endpoint)
    # We build a list of parts, each of which is either host:port, addr:port or
    # addr+n:port (where :port is omitted if None). These get comma-separated
    # at the end.
    parts = []
    for ip in (ipv4, ipv6):
        ip_parts = []    # lists of address, num, port (not tuples because mutated)
        # Group endpoints with the same port together, then order by IP address
        ip.sort(key=lambda endpoint: (endpoint.port is not None, endpoint.port, endpoint.host))
        for endpoint in ip:
            if (ip_parts and ip_parts[-1][2] == endpoint.port and
                    ip_parts[-1][0] + ip_parts[-1][1] == endpoint.host):
                ip_parts[-1][1] += 1
            else:
                ip_parts.append([endpoint.host, 1, endpoint.port])
        for (address, num, port) in ip_parts:
            if ip is ipv6:
                s = '[' + address.compressed + ']'
            else:
                s = address.compressed
            if num > 1:
                s += '+{}'.format(num - 1)
            if port is not None:
                s += ':{}'.format(port)
            parts.append(s)
    for endpoint in other:
        s = str(endpoint.host)
        if endpoint.port is not None:
            s += ':{}'.format(endpoint.port)
        parts.append(s)
    return ','.join(parts)


class SimulatorServer(katcp.DeviceServer):
    """katcp server for the simulator.

    Parameters
    ----------
    context : compute device context, optional
        Compute device context used for device-accelerated simulations
    subarray : :class:`katcbfsim.stream.Subarray`, optional
        Preconfigured subarray. If not specified, an unconfigured subarray is created.
    telstate : :class:`katsdptelstate.TelescopeState`, optional
        Telescope state used for the :samp:`?configure-subarray-from-telstate`
        and populated with sensors. If not provided, that request will fail,
        and no telstate sensors will be provided.
    *args, **kwargs :
        Passed to base class
    """

    VERSION_INFO = ('katcbfsim-api', 2, 0)
    BUILD_INFO = ('katcbfsim',) + tuple(katcbfsim.__version__.split('.', 1)) + ('',)

    def __init__(self, context=None, subarray=None, telstate=None, *args, **kwargs):
        super(SimulatorServer, self).__init__(*args, **kwargs)
        self._context = context
        self._streams = {}
        if subarray is None:
            self._subarray = Subarray()
        else:
            self._subarray = subarray
        self._telstate = telstate
        #: Dictionary of dictionaries, indexed by stream then sensor name
        self._stream_sensors = {}
        #: Set when halt is called
        self._halting = False

    def setup_sensors(self):
        self.add_sensor(Sensor.discrete('device-status',
            'Dummy device status sensor. The simulator is always ok.',
            '', ['ok', 'degraded', 'fail'], initial_status=Sensor.NOMINAL))

    def _add_stream(self, stream):
        assert stream.name not in self._streams
        self._streams[stream.name] = stream
        self._stream_sensors[stream] = {
            'bandwidth': Sensor.integer('{}.bandwidth'.format(stream.name),
                'The bandwidth currently configured for the data stream',
                'Hz', default=stream.bandwidth, initial_status=Sensor.NOMINAL),
            'channels': Sensor.integer('{}.channels'.format(stream.name),
                'The number of channels of the channelised data stream',
                '', default=stream.n_channels, initial_status=Sensor.NOMINAL),
            'centerfrequency': Sensor.integer('{}.centerfrequency'.format(stream.name),
                'The center frequency for the data stream', 'Hz')
        }
        for sensor in self._stream_sensors[stream].itervalues():
            self.add_sensor(sensor)

    def add_fx_stream(self, name, *args, **kwargs):
        stream = FXStream(self._context, self._subarray, name, *args, **kwargs)
        self._add_stream(stream)
        return stream

    def add_beamformer_stream(self, name, *args, **kwargs):
        stream = BeamformerStream(self._subarray, name, *args, **kwargs)
        self._add_stream(stream)
        return stream

    @request(Str(), Int(), Int(), Int(), Int())
    @return_reply()
    def request_stream_create_correlator(
            self, sock, name, adc_rate, center_frequency, bandwidth, n_channels):
        """Create a new simulated correlator stream

        Parameters
        ----------
        name : str
            Name for the new stream (must be unique)
        adc_rate : int
            Simulated ADC clock rate, in Hz
        center_frequency : int
            Sky frequency of the center of the band, in Hz
        bandwidth : int
            Bandwidth of all channels in the stream, in Hz
        n_channels : int
            Number of channels in the stream
        """
        if name in self._streams:
            return 'fail', 'stream {} already exists'.format(name)
        if self._halting:
            return 'fail', 'cannot add a stream while halting'
        if self._context is None:
            return 'fail', 'no device context available'
        self.add_fx_stream(name, adc_rate, center_frequency, bandwidth, n_channels)
        return 'ok',

    @request(Str(), Int(), Int(), Int(), Int(), Int(), Int())
    @return_reply()
    def request_stream_create_beamformer(
            self, sock, name, adc_rate, center_frequency, bandwidth, n_channels,
            timesteps, sample_bits):
        """Create a new simulated beamformer stream"""
        if name in self._streams:
            return 'fail', 'stream {} already exists'.format(name)
        if self._halting:
            return 'fail', 'cannot add a stream while halting'
        self.add_beamformer_stream(name, adc_rate, center_frequency, bandwidth,
                                   n_channels, timesteps, sample_bits)
        return 'ok',

    def set_destination(self, stream, endpoints, interface=None,
                        n_substreams=None, max_packet_size=None):
        if n_substreams is None:
            # Formula used by MeerKAT CBF
            n_substreams = 4
            while n_substreams < max(len(endpoints), stream.n_antennas * 4):
                n_substreams *= 2
        if isinstance(stream, FXStream):
            stream.transport_factories = [
                transport.FXSpeadTransport.factory(endpoints, interface, n_substreams, max_packet_size)
            ]
            if self._telstate is not None:
                stream.transport_factories.append(
                    transport.FXTelstateTransport.factory(
                        self._telstate, n_substreams))
        elif isinstance(stream, BeamformerStream):
            stream.transport_factories = [
                transport.BeamformerSpeadTransport.factory(endpoints, interface, n_substreams, max_packet_size)
            ]
            if self._telstate is not None:
                stream.transport_factories.append(
                    transport.BeamformerTelstateTransport.factory(
                        self._telstate, n_substreams, stream_name=stream.name))
        else:
            raise UnsupportedStreamError('unknown stream type')

    @request(Str(), Str(), Int(optional=True), Int(optional=True))
    @return_reply()
    @_stream_exceptions
    @_stream_request
    def request_capture_destination(self, sock, stream, destination, n_substreams=None,
                                    max_packet_size=None):
        """Set the destination endpoints for a stream"""
        endpoints = endpoint_list_parser(None)(destination)
        for e in endpoints:
            if e.port is None:
                return 'fail', 'no port specified'
        self.set_destination(stream, endpoints, None, n_substreams, max_packet_size)
        return 'ok',

    @request(Str(), Str())
    @return_reply()
    @_stream_exceptions
    @_stream_request
    def request_capture_destination_file(self, sock, stream, destination):
        """Set the destination to an HDF5 file"""
        if isinstance(stream, FXStream):
            stream.transport_factories = [transport.FXFileTransport.factory(destination)]
        else:
            return 'fail', 'file capture not supported for this stream type'
        return 'ok',

    @request(Str(default=''))
    @return_reply()
    def request_capture_list(self, sock, req_name):
        """List the destination endpoints for a stream, or all streams"""
        if req_name != '' and req_name not in self._streams:
            return 'fail', 'requested stream name "{}" not found'.format(req_name)
        for name, stream in self._streams.items():
            if req_name == '' or req_name == name:
                try:
                    endpoints = stream.transport_factories[0].endpoints
                except AttributeError:
                    endpoints = [Endpoint('0.0.0.0', 0)]
                endpoints_str = endpoints_to_str(endpoints)
                sock.inform(name, endpoints_str)
        return 'ok',

    def set_sync_time(self, timestamp):
        self._subarray.sync_time = katpoint.Timestamp(timestamp)

    @request(Int())
    @return_reply()
    @_stream_exceptions
    def request_sync_time(self, sock, timestamp):
        """Set the sync time, as seconds since the UNIX epoch. This will also
        be the timestamp associated with the first data dump."""
        self.set_sync_time(timestamp)
        return 'ok',

    def set_clock_ratio(self, clock_ratio):
        self._subarray.clock_ratio = clock_ratio

    @request(Float())
    @return_reply()
    @_stream_exceptions
    def request_clock_ratio(self, sock, clock_ratio):
        """Set the ratio between wall clock time and simulated time. Values
        less than 1 will cause simulated time to run faster than wall clock
        time. Setting it to 0 will cause simulation to be done as fast as
        possible.
        """
        self.set_clock_ratio(clock_ratio)
        return 'ok',

    def set_target(self, target):
        self._subarray.target = target

    @request(Str())
    @return_reply()
    def request_target(self, sock, target):
        """Set the simulated target, in the format used by katpoint. This also
        sets the phase center. If no position has been set with `position`, the
        position also defaults to the target. This can be set while capture is
        running."""
        self.set_target(katpoint.Target(target))
        return 'ok',

    def set_position(self, position):
        self._subarray.position = position

    @request(Str())
    @return_reply()
    def request_position(self, sock, position):
        """Set the simulated position (antenna pointing direction) for the
        first antenna, in the format used by katpoint. At present it is assumed
        that all antennas point in the same direction.
        """
        self.set_position(katpoint.Target(position))
        return 'ok',

    def set_accumulation_length(self, stream, period):
        if hasattr(stream, 'accumulation_length'):
            stream.accumulation_length = period
        else:
            raise UnsupportedStreamError('stream does not support setting accumulation length')

    @request(Str(), Float())
    @return_reply(Float())
    @_stream_exceptions
    @_stream_request
    def request_accumulation_length(self, sock, stream, period):
        """Set the accumulation interval for a stream.

        Note: this differs from the CAM-CBF ICD, in which this is subarray-wide."""
        self.set_accumulation_length(stream, period)
        # accumulation_length is a property, and the setter rounds the value.
        # We are thus returning the rounded value.
        return 'ok', stream.accumulation_length

    def set_center_frequency(self, stream, frequency):
        stream.center_frequency = frequency
        # TODO: get the simulated timestamp from the stream
        self._stream_sensors[stream]['centerfrequency'].set_value(frequency)

    @request(Str(), Int())
    @return_reply()
    @_stream_exceptions
    @_stream_request
    def request_frequency_select(self, sock, stream, frequency):
        """Set the center frequency for the band. Unlike the real CBF, an
        arbitrary frequency may be selected, and it will not be rounded.
        """
        self.set_center_frequency(stream, frequency)
        return 'ok',

    def set_n_dumps(self, stream, n_dumps):
        stream.n_dumps = n_dumps

    @request(Str(), Int())
    @return_reply()
    @_stream_exceptions
    @_stream_request
    def request_n_dumps(self, sock, stream, n_dumps):
        """Set a limited number of dumps for a stream, after which it will
        stop.
        """
        self.set_n_dumps(stream, n_dumps)
        return 'ok',

    def set_gain(self, gain):
        self._subarray.gain = gain

    @request(Float())
    @return_reply()
    @_stream_exceptions
    def request_gain(self, sock, gain):
        """Set the system-wide gain, as the expected visibility value per
        Hz of channel bandwidth per second of integration."""
        self.set_gain(gain)
        return 'ok',

    def add_antenna(self, antenna):
        self._subarray.add_antenna(antenna)

    @request(Str())
    @return_reply()
    @_stream_exceptions
    def request_antenna_add(self, sock, antenna_str):
        """Add an antenna to the simulated array, in the format accepted by katpoint."""
        self.add_antenna(katpoint.Antenna(antenna_str))
        return 'ok',

    @request()
    @return_reply()
    def request_antenna_list(self, sock):
        """Report all the antennas in the simulated array"""
        for antenna in self._subarray.antennas:
            sock.inform(antenna.description)
        return 'ok',

    def add_source(self, source):
        self._subarray.add_source(source)

    @request(Str())
    @return_reply()
    @_stream_exceptions
    def request_source_add(self, sock, source_str):
        """Add a source to the sky model, in the format accepted by
        :class:`katcbfsim.source.Source`."""
        self.add_source(Source(source_str))
        return 'ok',

    @request()
    @return_reply()
    def request_source_list(self, sock):
        """List the sources in the sky model."""
        for source in self._subarray.sources:
            sock.inform(source.description)
        return 'ok',

    def configure_subarray_from_telstate(self, telstate=None):
        """Configure subarray from sensors/attributes in telescope state."""
        if telstate is None:
            telstate = self._telstate
        antenna_names = telstate['config']['antenna_mask'].split(',')
        for name in antenna_names:
            attribute_name = name + '_observer'
            try:
                antenna = telstate[attribute_name]
            except KeyError:
                logger.warn('Antenna description for %s not found, skipping', name)
            else:
                # It might be either a string or an object at this point.
                # The constructor handles either case.
                self.add_antenna(katpoint.Antenna(antenna))

    @request()
    @return_reply()
    @_stream_exceptions
    def request_configure_subarray_from_telstate(self, sock):
        """Configure the subarray using sensors and attributes in the telescope
        state. This uses dynamic values, rather than the :samp:`config`
        dictionary used by the command-line parser.
        """
        if self._telstate is None:
            return 'fail', 'no telescope state was specified with --telstate'
        self.configure_subarray_from_telstate()
        return 'ok',

    @tornado.gen.coroutine
    def send_metadata(self, stream):
        yield to_tornado_future(trollius.ensure_future(stream.send_metadata(), loop=stream.loop))

    @request(Str())
    @return_reply()
    @_stream_exceptions
    @_stream_request
    @tornado.gen.coroutine
    def request_capture_meta(self, sock, stream):
        """Send metadata for a stream"""
        if self._halting:
            return 'fail', 'cannot send metadata while halting'
        self.send_metadata(stream)
        return 'ok',

    def capture_start(self, stream):
        stream.capture_start()

    @request(Str())
    @return_reply()
    @_stream_exceptions
    @_stream_request
    def request_capture_start(self, sock, stream):
        """Start the flow of data for a stream"""
        if self._halting:
            return 'fail', 'cannot start capture while halting'
        self.capture_start(stream)
        return 'ok',

    @request(Str())
    @return_reply()
    @tornado.gen.coroutine
    def request_capture_stop(self, sock, name):
        """Stop the flow of data for a stream"""
        try:
            stream = self._streams[name]
        except KeyError:
            raise tornado.gen.Return(('fail', 'requested stream name "{}" not found'.format(name)))
        stop = trollius.async(stream.capture_stop())
        yield to_tornado_future(stop)
        raise tornado.gen.Return(('ok',))

    @tornado.gen.coroutine
    def request_halt(self, req, msg):
        self._halting = True  # Prevents changes to _streams while we iterate
        for stream in self._streams.values():
            stop = trollius.async(stream.capture_stop())
            yield to_tornado_future(stop)
        yield tornado.gen.maybe_future(super(SimulatorServer, self).request_halt(req, msg))

    request_halt.__doc__ = katcp.DeviceServer.request_halt.__doc__
