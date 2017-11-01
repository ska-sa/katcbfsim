from __future__ import print_function, division
import logging
import functools
import enum
import asyncio

import aiokatcp
from aiokatcp import FailReply, Sensor, Timestamp, RequestContext

import katpoint
from katsdptelstate.endpoint import Endpoint, endpoint_list_parser, endpoints_to_str
import katsdpservices

import katcbfsim
from . import transport
from .stream import (Subarray, FXStream, BeamformerStream,
                     StreamError, UnsupportedStreamError, ConfigError)
from .source import Source


logger = logging.getLogger(__name__)


class DeviceStatus(enum.Enum):
    OK = 0
    DEGRADED = 1
    FAIL = 2


def _stream_request(wrapped):
    """Decorator for per-stream commands. It looks up the stream and passes
    it to the wrapped function, or returns a failure message if it does not
    exist.
    """
    @functools.wraps(wrapped)
    def wrapper(self, sock, name, *args, **kwargs):
        try:
            stream = self._streams[name]
        except KeyError as error:
            raise FailReply('requested stream name "{}" not found'.format(name)) from error
        return wrapped(self, sock, stream, *args, **kwargs)
    return wrapper


def _stream_exceptions(wrapped):
    """Decorator used on requests that and turns exceptions defined in
    :py:mod:`stream` into katcp "fail" messages.
    """
    def wrapper(*args, **kwargs):
        try:
            return wrapped(*args, **kwargs)
        except StreamError as error:
            raise FailReply(str(error)) from error
    functools.update_wrapper(wrapper, wrapped)
    return wrapper


class SimulatorServer(aiokatcp.DeviceServer):
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

    VERSION = 'katcbfsim-api-2.0'
    BUILD_STATE = 'katcbfsim-{}'.format(katcbfsim.__version__)

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
        self.sensors.add(Sensor(
            DeviceStatus, 'device-status',
            'Dummy device status sensor. The simulator is always ok.',
            '', default=DeviceStatus.OK,
            initial_status=Sensor.Status.NOMINAL))

    def _add_stream(self, stream):
        assert stream.name not in self._streams
        if self._telstate is not None:
            # TODO: this could block asyncio for a non-trivial amount of time,
            # but it can't be made asynchronous without creating race
            # conditions (where another stream could be added in parallel with
            # the same name). A lock may be required.
            stream.set_telstate(self._telstate)
        self._streams[stream.name] = stream
        self._stream_sensors[stream] = {
            'bandwidth': Sensor(float,
                '{}.bandwidth'.format(stream.name),
                'The bandwidth currently configured for the data stream',
                'Hz', default=stream.bandwidth,
                initial_status=Sensor.Status.NOMINAL),
            'channels': Sensor(int,
                '{}.channels'.format(stream.name),
                'The number of channels of the channelised data stream',
                '', default=stream.n_channels,
                initial_status=Sensor.Status.NOMINAL),
            'centerfrequency': Sensor(float,
                '{}.centerfrequency'.format(stream.name),
                'The center frequency for the data stream', 'Hz')
        }
        for sensor in self._stream_sensors[stream].values():
            self.sensors.add(sensor)

    def add_fx_stream(self, name, *args, **kwargs):
        stream = FXStream(self._context, self._subarray, name, *args, **kwargs)
        self._add_stream(stream)
        return stream

    def add_beamformer_stream(self, name, *args, **kwargs):
        stream = BeamformerStream(self._subarray, name, *args, **kwargs)
        self._add_stream(stream)
        return stream

    async def request_stream_create_correlator(
            self, ctx: RequestContext,
            name: str, adc_rate: float, center_frequency: float, bandwidth: float,
            n_channels: int, n_substreams: int = None) -> None:
        """Create a new simulated correlator stream

        Parameters
        ----------
        name : str
            Name for the new stream (must be unique)
        adc_rate : float
            Simulated ADC clock rate, in Hz
        center_frequency : float
            Sky frequency of the center of the band, in Hz
        bandwidth : float
            Bandwidth of all channels in the stream, in Hz
        n_channels : int
            Number of channels in the stream
        n_substreams : int, optional
            Number of substreams (X engines)
        """
        if name in self._streams:
            raise FailReply('stream {} already exists'.format(name))
        if self._halting:
            raise FailReply('cannot add a stream while halting')
        if self._context is None:
            raise FailReply('no device context available')
        self.add_fx_stream(name, adc_rate, center_frequency, bandwidth, n_channels, n_substreams)

    async def request_stream_create_beamformer(
            self, ctx: RequestContext,
            name: str, adc_rate: float, center_frequency: float, bandwidth: float,
            n_channels: int, n_substreams: int, timesteps: int, sample_bits: int) -> None:
        """Create a new simulated beamformer stream"""
        if name in self._streams:
            raise FailReply('stream {} already exists'.format(name))
        if self._halting:
            raise FailReply('cannot add a stream while halting')
        self.add_beamformer_stream(name, adc_rate, center_frequency, bandwidth,
                                   n_channels, n_substreams, timesteps, sample_bits)

    def set_destination(self, stream, endpoints, ifaddr=None, ibv=False,
                        max_packet_size=None):
        if isinstance(stream, FXStream):
            stream.transport_factories = [
                transport.FXSpeadTransport.factory(
                    endpoints, ifaddr, ibv, max_packet_size)
            ]
        elif isinstance(stream, BeamformerStream):
            stream.transport_factories = [
                transport.BeamformerSpeadTransport.factory(
                    endpoints, ifaddr, ibv, max_packet_size)
            ]
        else:
            raise UnsupportedStreamError('unknown stream type')

    @_stream_exceptions
    @_stream_request
    async def request_capture_destination(
            self, ctx: RequestContext, stream: str, destination: str,
            interface: str = None, ibv: bool = False, max_packet_size: int = None) -> None:
        """Set the destination endpoints for a stream"""
        endpoints = endpoint_list_parser(None)(destination)
        for e in endpoints:
            if e.port is None:
                raise FailReply('no port specified')
        try:
            ifaddr = katsdpservices.get_interface_address(interface)
        except ValueError as error:
            raise FailReply(str(error)) from error
        self.set_destination(stream, endpoints, ifaddr, ibv, max_packet_size)

    @_stream_exceptions
    @_stream_request
    async def request_capture_destination_file(
            self, ctx: RequestContext, stream: str, destination: str) -> None:
        """Set the destination to an HDF5 file"""
        if isinstance(stream, FXStream):
            stream.transport_factories = [transport.FXFileTransport.factory(destination)]
        else:
            raise FailReply('file capture not supported for this stream type')

    async def request_capture_list(self, ctx: RequestContext, req_name: str = '') -> None:
        """List the destination endpoints for a stream, or all streams"""
        if req_name != '' and req_name not in self._streams:
            raise FailReply('requested stream name "{}" not found'.format(req_name))
        informs = []
        for name, stream in self._streams.items():
            if req_name == '' or req_name == name:
                try:
                    endpoints = stream.transport_factories[0].endpoints
                except AttributeError:
                    endpoints = [Endpoint('0.0.0.0', 0)]
                endpoints_str = endpoints_to_str(endpoints)
                informs.append((name, endpoints_str))
        ctx.informs(informs)

    def set_sync_time(self, timestamp):
        self._subarray.sync_time = katpoint.Timestamp(timestamp)

    @_stream_exceptions
    async def request_sync_time(self, ctx: RequestContext, timestamp: Timestamp) -> None:
        """Set the sync time, as seconds since the UNIX epoch. This will also
        be the timestamp associated with the first data dump."""
        self.set_sync_time(float(timestamp))

    def set_clock_ratio(self, clock_ratio):
        self._subarray.clock_ratio = clock_ratio

    @_stream_exceptions
    async def request_clock_ratio(self, ctx: RequestContext, clock_ratio: float) -> None:
        """Set the ratio between wall clock time and simulated time. Values
        less than 1 will cause simulated time to run faster than wall clock
        time. Setting it to 0 will cause simulation to be done as fast as
        possible.
        """
        self.set_clock_ratio(clock_ratio)

    def set_target(self, target):
        self._subarray.target = target

    async def request_target(self, ctx: RequestContext, target: str) -> None:
        """Set the simulated target, in the format used by katpoint. This also
        sets the phase center. If no position has been set with `position`, the
        position also defaults to the target. This can be set while capture is
        running."""
        self.set_target(katpoint.Target(target))

    def set_position(self, position):
        self._subarray.position = position

    async def request_position(self, ctx: RequestContext, position: str):
        """Set the simulated position (antenna pointing direction) for the
        first antenna, in the format used by katpoint. At present it is assumed
        that all antennas point in the same direction.
        """
        self.set_position(katpoint.Target(position))

    def set_accumulation_length(self, stream, period):
        if hasattr(stream, 'accumulation_length'):
            stream.accumulation_length = period
        else:
            raise UnsupportedStreamError('stream does not support setting accumulation length')

    @_stream_exceptions
    @_stream_request
    async def request_accumulation_length(
            self, ctx: RequestContext, stream: str, period: float) -> float:
        """Set the accumulation interval for a stream.

        Note: this differs from the CAM-CBF ICD, in which this is subarray-wide."""
        self.set_accumulation_length(stream, period)
        # accumulation_length is a property, and the setter rounds the value.
        # We are thus returning the rounded value.
        return stream.accumulation_length

    def set_center_frequency(self, stream, frequency):
        stream.center_frequency = frequency
        # TODO: get the simulated timestamp from the stream
        self._stream_sensors[stream]['centerfrequency'].set_value(frequency)

    @_stream_exceptions
    @_stream_request
    async def request_frequency_select(
            self, ctx: RequestContext, stream: str, frequency: float) -> None:
        """Set the center frequency for the band. Unlike the real CBF, an
        arbitrary frequency may be selected, and it will not be rounded.
        """
        self.set_center_frequency(stream, frequency)
        return 'ok',

    def set_n_dumps(self, stream, n_dumps):
        stream.n_dumps = n_dumps

    @_stream_exceptions
    @_stream_request
    async def request_n_dumps(self, ctx: RequestContext, stream: str, n_dumps: int) -> None:
        """Set a limited number of dumps for a stream, after which it will
        stop.
        """
        self.set_n_dumps(stream, n_dumps)

    def set_gain(self, gain):
        self._subarray.gain = gain

    @_stream_exceptions
    async def request_gain(self, ctx: RequestContext, gain: float) -> None:
        """Set the system-wide gain, as the expected visibility value per
        Hz of channel bandwidth per second of integration."""
        self.set_gain(gain)

    def add_antenna(self, antenna):
        self._subarray.add_antenna(antenna)

    @_stream_exceptions
    async def request_antenna_add(self, ctx: RequestContext, antenna_str: str) -> None:
        """Add an antenna to the simulated array, in the format accepted by katpoint."""
        self.add_antenna(katpoint.Antenna(antenna_str))

    async def request_antenna_list(self, ctx: RequestContext) -> None:
        """Report all the antennas in the simulated array"""
        ctx.informs((antenna.description,) for antenna in self._subarray.antennas)

    def add_source(self, source):
        self._subarray.add_source(source)

    @_stream_exceptions
    async def request_source_add(self, ctx: RequestContext, source_str: str) -> None:
        """Add a source to the sky model, in the format accepted by
        :class:`katcbfsim.source.Source`."""
        self.add_source(Source(source_str))

    async def request_source_list(self, ctx: RequestContext) -> None:
        """List the sources in the sky model."""
        ctx.informs((source.description,) for source in self._subarray.sources)

    def configure_subarray_from_telstate(self, antenna_names=None, telstate=None):
        """Configure subarray from sensors/attributes in telescope state."""
        if telstate is None:
            telstate = self._telstate
        if antenna_names is None:
            antenna_names = [antenna.name for antenna in self._subarray.antennas]
        for name in antenna_names:
            attribute_name = name + '_observer'
            try:
                antenna = telstate[attribute_name]
            except KeyError:
                raise ConfigError('Antenna description for {} not found'.format(name))
            else:
                # It might be either a string or an object at this point.
                # The constructor handles either case.
                self.add_antenna(katpoint.Antenna(antenna))

    @_stream_exceptions
    async def request_configure_subarray_from_telstate(
            self, ctx: RequestContext, antenna_names: str = None) -> None:
        """Configure the subarray using sensors and attributes in the telescope
        state. This uses dynamic values, rather than the :samp:`config`
        dictionary used by the command-line parser.
        """
        if self._telstate is None:
            raise FailReply('no telescope state was specified with --telstate')
        if antenna_names is not None:
            antenna_names = antenna_names.split(',')
        self.configure_subarray_from_telstate(antenna_names=antenna_names)

    async def send_metadata(self, stream):
        await trollius.ensure_future(stream.send_metadata(), loop=stream.loop)

    @_stream_exceptions
    @_stream_request
    async def request_capture_meta(self, ctx: RequestContext, stream: str) -> None:
        """Send metadata for a stream"""
        if self._halting:
            raise FailReply('cannot send metadata while halting')
        await stream.send_metadata()

    def capture_start(self, stream):
        stream.capture_start()

    @_stream_exceptions
    @_stream_request
    async def request_capture_start(self, ctx: RequestContext, stream: str) -> None:
        """Start the flow of data for a stream"""
        if self._halting:
            raise FailReply('cannot start capture while halting')
        self.capture_start(stream)

    @_stream_exceptions
    @_stream_request
    async def request_capture_stop(self, ctx: RequestContext, stream: str) -> None:
        """Stop the flow of data for a stream"""
        await stream.capture_stop()

    async def stop(self, cancel=True):
        self._halting = True  # Prevents changes to _streams while we iterate
        for stream in self._streams.values():
            await stream.capture_stop()
        await super().stop(cancel)
