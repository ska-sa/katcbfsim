import logging
import collections
import math
import asyncio

import numpy as np
import scipy.interpolate

import katsdptelstate
from katsdpsigproc import accel
from katsdpsigproc.asyncio import resource
import katpoint

from . import rime


logger = logging.getLogger(__name__)


class StreamError(RuntimeError):
    """Base class for user errors relating to streams or subarrays."""
    pass


class CaptureInProgressError(StreamError):
    """Exception thrown when trying to modify some value that may not be
    modified while data capture is running.
    """
    pass


class StreamExistsError(StreamError):
    """Exception thrown when trying to modify some value that may not be
    modified after a stream has been created on the subarray.
    """
    pass


class ConfigError(StreamError):
    """Exception thrown when the stream or subarray has missing or invalid configuration."""
    pass


class UnsupportedStreamError(StreamError):
    """Exception thrown for operations not supported on this stream type."""
    pass


class _Property:
    """A property that wraps a class member, and applies checks and transformations when set.

    Parameters
    ----------
    attr : str
        Name of the underlying class member.
    transform : callable
        Method that transforms the value (and optionally raises exceptions)
    """
    def __init__(self, attr, transform=None):
        self.attr = attr
        self.transform = transform

    def __get__(self, instance, owner):
        if instance is None:
            raise AttributeError
        return getattr(instance, self.attr)

    def __set__(self, instance, value):
        if self.transform is not None:
            value = self.transform(instance, value)
        setattr(instance, self.attr, value)


class _CaptureImmutableProperty(_Property):
    def __set__(self, instance, value):
        if instance.capturing:
            msg = 'cannot set {} while capture is in progress'.format(self.attr[1:])
            raise CaptureInProgressError(msg)
        super().__set__(instance, value)


class _StreamImmutableProperty(_Property):
    def __set__(self, instance, value):
        if instance.streams:
            msg = 'cannot set {} after creating a stream'.format(self.attr[1:])
            raise StreamExistsError(msg)
        super().__set__(instance, value)


def _stream_immutable(transform):
    """Decorator for creating :class:`_StreamImmutableProperty`.

    It is given a transform method. The wrapped attribute must have the
    same name but with an underscore prefixed.
    """
    return _StreamImmutableProperty('_' + transform.__name__, transform)


class Subarray(object):
    """Model of an array, the sky, and possibly other simulation parameters,
    shared by several data streams. A subarray should first be configured
    before creating streams. Attributes fall into three classes (documented on
    each attribute):

      1. Mutable: can be safely changed at any time.
      2. Capture-immutable: can be changed provided no captures are in progress
         (otherwise a :exc:`CaptureInProgressError` is raised).
      3. Stream-immutable: can be changed only until a stream is created
         (otherwise a :exc:`StreamExistsError` is raised).

    This is partly enforced by exceptions thrown from setters, but the
    :attr:`antennas` and :attr:`sources` attributes are mutable lists and
    must only be manipulated by :meth:`add_antenna` and :meth:`add_source`.

    Attributes
    ----------
    antennas : list of :class:`katpoint.Antenna`
        The antennas in the simulated array. The values are
        capture-immutable, while the list size is stream-immutable.
    sources : list of :class:`katpoint.Target`
        The simulated sources. Only point sources are currently supported.
        The do not necessarily have to be radec targets, and the position
        and flux model are mutable, but the list size is capture-immutable.
    target : :class:`katpoint.Target`
        Target. This determines the phase center for the simulation (and
        eventually the center for the beam model as well). Mutable.
    sync_time : :class:`katpoint.Timestamp`
        Time base for timestamps. Stream-immutable.
    gain : float
        Expected output visibility value, per Jansky per Hz per second.
        Capture-immutable.
    clock_ratio : float
        Scale factor between virtual time in the simulation and wall clock
        time. Smaller values will run the simulation faster; setting it to 0
        will cause the simulation to run as fast as possible.
        Capture-immutable.
    n_servers : int
        Number of servers over which the simulation is split. Stream-immutable.
    server_id : int
        Number of this server amongst :attr:`n_servers` (0-based). Stream-immutable.
    """
    def __init__(self):
        self.antennas = []
        self.sources = []
        self.target = katpoint.Target('Zenith, azel, 0:00:00.0, 90:00:00.0')
        self.position = None
        self._sync_time = katpoint.Timestamp()
        self._gain = 1e-4
        self._clock_ratio = 1.0
        self.streams = 0       # Total number of streams created from this subarray
        self.capturing = 0     # Number of streams that are capturing
        self._n_servers = 1
        self._server_id = 0

    def add_antenna(self, antenna):
        """Add a new antenna to the simulation, or replace an existing one.

        If an antenna with the same name exists, it is replaced, otherwise the
        antenna is appended.

        Parameters
        ----------
        antenna : :class:`katpoint.Antenna`
            New antenna

        Raises
        ------
        CaptureInProgressError
            If any associated stream has a capture is in progress
        """
        if self.capturing:
            raise CaptureInProgressError('cannot modify antennas while capture is in progress')
        for i in range(len(self.antennas)):
            if self.antennas[i].name == antenna.name:
                self.antennas[i] = antenna
                return
        if self.streams:
            raise StreamExistsError('cannot add new antennas after creating a stream')
        self.antennas.append(antenna)

    def add_source(self, source):
        """Add a new source to the simulation.

        See :attr:`sources` for details on what types of source are supported.

        Parameters
        ----------
        source : :class:`katpoint.Target`
            New source

        Raises
        ------
        CaptureInProgressError
            If any associated stream has a capture is in progress
        """
        if self.capturing:
            raise CaptureInProgressError('cannot add source while capture is in progress')
        if source.flux_model is None:
            logging.warn('source has no flux model; it will be assumed to be 1 Jy')
            source.flux_model = katpoint.FluxDensityModel(0.0, np.inf, [])
        self.sources.append(source)

    def ensure_source(self, timestamp):
        """Ensure that at least one source exists. If no source exists, a 1 Jy
        source is placed at the phase center (which must exist at `timestamp`).

        Parameters
        ----------
        timestamp : :class:`katpoint.Timestamp`
            Time at which to look up the target
        """
        if not self.sources:
            self.sources.append(self.target_at(timestamp))

    sync_time = _StreamImmutableProperty('_sync_time')
    gain = _CaptureImmutableProperty('_gain')
    clock_ratio = _CaptureImmutableProperty('_clock_ratio')

    @_stream_immutable
    def n_servers(self, value):
        if value <= self.server_id:
            raise ValueError('cannot set n_servers <= server_id')
        return value

    @_stream_immutable
    def server_id(self, value):
        if value < 0:
            raise ValueError('cannot set server_id < 0')
        if value >= self.n_servers:
            raise ValueError('cannot set n_servers <= server_id')
        return value

    def target_at(self, timestamp):
        """Obtains the target at a given point in simulated time. In this base
        class this just returns :attr:`target`, but it can be overridden by
        subclasses to allow the target to be pulled rather than pushed."""
        return self.target

    def position_at(self, timestamp):
        """Obtains the position (pointing direction) at a given point in
        simulated time. In this base class this just returns :attr:`position`,
        but it can be overridden by subclasses to allow the position to be
        pulled rather than pushed."""
        return self.position


class Stream(object):
    """Base class for simulated streams. It provides the basic machinery to
    start and stop capture running in a separate asyncio task.

    Unless otherwise documented, changing any attributes while a capture is in
    progress has undefined behaviour. In some cases, but not all, this is
    enforced by a :exc:`CaptureInProgressError`.

    Parameters
    ----------
    subarray : :class:`Subarray`
        Subarray corresponding to this stream
    name : str
        Name for this stream (used by katcp)
    loop : :class:`asyncio.BaseEventLoop`, optional
        Event loop for coroutines

    Attributes
    ----------
    name : :class:`str`
        Name for this stream (used by katcp)
    name_norm : :class:`str`
        Name with dashes and dots changed to underscores (used in telstate)
    subarray : :class:`Subarray`
        Subarray corresponding to this stream
    n_antennas : int, read-only
        Number of antennas
    n_baselines : int, read-only
        Number of baselines (antenna pairs, not input pairs)
    transport_factories : list of callable
        Each factory is called with `self` to return a transport on which
        output will be sent.
    capturing : bool, read-only
        Whether a capture is in progress. This is true from
        :meth:`capture_start` until :meth:`capture_stop` completes, even if
        the capture coroutine terminates earlier.
    start_time : :class:`katpoint.Timestamp`
        Simulated time for the start of the capture.
    loop : :class:`asyncio.BaseEventLoop`
        Event loop for coroutines
    """
    def __init__(self, subarray, name, loop=None):
        self._capture_future = None
        self._stop_future = None
        self.name = name
        self.name_norm = self.name.replace('-', '_').replace('.', '_')
        self.subarray = subarray
        self.start_time = None
        self.transport_factories = []
        if loop is None:
            self.loop = asyncio.get_event_loop()
        else:
            self.loop = loop
        self._last_warn_behind = 0
        self.subarray.streams += 1

    def __del__(self):
        self.subarray.streams -= 1

    def __setattr__(self, key, value):
        # Prevent modifications while capture is in progress
        if not key.startswith('_') and self.capturing:
            raise CaptureInProgressError('cannot set {} while capture is in progress'.format(key))
        return super(Stream, self).__setattr__(key, value)

    @property
    def n_antennas(self):
        return len(self.subarray.antennas)

    @property
    def n_baselines(self):
        n = self.n_antennas
        return n * (n + 1) // 2

    @property
    def capturing(self):
        return self._capture_future is not None

    def capture_start(self, start_time):
        """Begin capturing data, if it has not been done already. The
        capturing is done on the asyncio event loop, which must thus be
        allowed to run frequently to ensure timeous delivery of results.

        Subclasses may override this to provide consistency checks on the
        state. They must also provide the :meth:`_capture` coroutine.

        Raises
        ------
        ConfigError
            if no destination is defined
        """
        if self.capturing:
            logger.warn('Ignoring attempt to start capture when already running')
            return
        if not self.transport_factories:
            raise ConfigError('no destination specified')
        if start_time < self.subarray.sync_time:
            raise ConfigError('start time is before sync time')
        self.subarray.capturing += 1
        self.start_time = start_time
        # Create a future that is set by capture_stop
        self._stop_future = asyncio.Future(loop=self.loop)
        # Start the capture coroutine on the event loop
        self._capture_future = asyncio.ensure_future(self._capture(), loop=self.loop)

    async def capture_stop(self):
        """Request an end to data capture, and wait for capture to complete.
        This is a coroutine.
        """
        if not self.capturing:
            logger.warn('Ignoring attempt to stop capture when not running')
            return
        # Need to check if a result has been set to protect against concurrent
        # stops.
        if not self._stop_future.done():
            self._stop_future.set_result(None)
        try:
            await self._capture_future
        except Exception:
            logger.warning('Exception in capture coroutine', exc_info=True)
        self._stop_future = None
        self._capture_future = None
        self.start_time = None
        self.subarray.capturing -= 1

    async def send_metadata(self):
        """Send metadata on all transports.

        This instantiates temporary copies of the transports, which are
        destroyed as soon as the metadata is sent. It is intended to be used
        when capturing is not in progress.
        """
        transports = []
        try:
            transports = [factory(self) for factory in self.transport_factories]
            futures = [t.send_metadata() for t in transports]
            await asyncio.gather(*futures, loop=self.loop)
        finally:
            for t in transports:
                await t.close()

    async def wait_for_next(self, wall_time):
        """Utility function for subclasses to managing timing. It will wait
        until either `wall_time` is reached, or :meth:`capture_stop` has been
        called. It also warns if `wall_time` has already passed.

        Parameters
        ----------
        wall_time : float
            Loop timestamp at which to stop

        Returns
        -------
        stopped : bool
            If true, then :meth:`capture_stop` was called.
        """
        try:
            now = self.loop.time()
            if now > wall_time:
                if now - self._last_warn_behind >= 1:
                    logger.warn('Falling behind the requested rate by %f seconds', now - wall_time)
                    self._last_warn_behind = now
            else:
                logger.debug('Sleeping for %f seconds', wall_time - now)
                self._last_warn_behind = 0
            await resource.wait_until(asyncio.shield(self._stop_future), wall_time, self.loop)
        except asyncio.TimeoutError:
            # This is the normal case: time for the next dump to be transmitted
            stopped = False
        else:
            # The _stop_future was triggered
            stopped = True
        return stopped


class CBFStream(Stream):
    """Parts that are shared between :class:`FXStream` and :class:`BeamformerStream`.

    Parameters
    ----------
    subarray : :class:`Subarray`
        Subarray corresponding to this stream
    name : str
        Name for this stream (used by katcp and in telstate attributes)
    adc_rate : float
        Simulated ADC clock rate, in Hz
    center_frequency : float
        Sky frequency of the center of the band, in Hz
    bandwidth : float
        Bandwidth of all channels in the stream, in Hz
    n_channels : int
        Number of channels in the stream
    n_substreams : int, optional
        Number of substreams (X/B-engines). If not specified, a default
        will be computed to match the MeerKAT CBF.
    loop : :class:`asyncio.BaseEventLoop`, optional
        Event loop for coroutines

    Attributes
    ----------
    adc_rate : float
        Simulated ADC clock rate, in Hz
    center_frequency : float
        Sky frequency of the center of the band, in Hz
    bandwidth : float
        Bandwidth of all channels in the stream, in Hz
    n_channels : int
        Number of channels in the stream
    n_substreams : int
        Number of substreams (X/B-engines)
    n_dumps : int
        If not ``None``, limits the number of dumps that will be done
    channel_range : slice
        Range out of n_channels for which this server is responsible
    scale_factor_timestamp : float
        Number of timestamp increments per second
    start_timestamp : int
        Timestamp for start time, as a raw CBF timestamp

    Raises
    ------
    ConfigError
        if
        - there are no antennas defined
        - the number of channels is not divisible by the number of substreams
        - the number of substreams is not divisible by the number of servers
    """
    def __init__(self, subarray, name, adc_rate, center_frequency, bandwidth,
                 n_channels, n_substreams=None, loop=None):
        if not subarray.antennas:
            raise ConfigError('no antennas defined')
        super(CBFStream, self).__init__(subarray, name, loop)
        self.adc_rate = adc_rate
        self.center_frequency = center_frequency
        self.bandwidth = bandwidth
        self.n_channels = n_channels
        if not n_substreams:
            # This is based on the MeerKAT CBF instruments, which support
            # power-of-two numbers of antennas (minimum 4), with 4 substreams
            # per antenna.
            n_substreams = 16
            while n_substreams < self.n_antennas * 4:
                n_substreams *= 2
        if n_channels % n_substreams:
            raise ConfigError('Number of channels not divisible by number of substreams')
        if n_substreams % subarray.n_servers:
            raise ConfigError('Number of substreams not divisible by number of servers')
        self.n_substreams = n_substreams
        self.n_dumps = None
        self.channel_range = slice(
            subarray.server_id * n_channels // subarray.n_servers,
            (subarray.server_id + 1) * n_channels // subarray.n_servers)
        # This is what real CBF uses, even though it wraps pretty quickly
        self.scale_factor_timestamp = adc_rate

    @property
    def start_timestamp(self):
        if self.start_time is None:
            return None
        return int(round((self.start_time - self.subarray.sync_time) * self.scale_factor_timestamp))

    def sensor(self, telstate, key, value, immutable=True):
        """Add an attribute or sensor to telescope state.

        Parameters
        ----------
        telstate : :class:`katsdptelstate.TelescopeState`
            Telescope state, scoped to the CBF stream/instrument
        key : str
            Name of the sensor within the stream or instrument. It is
            underscore-normalised before being used to form telstate keys.
        value : object
            Sensor/attribute value
        immutable : bool, optional
            Passed to :meth:`katsdptelstate.TelescopeState.add`
        """
        try:
            telstate.add(key, value, immutable=immutable)
        except katsdptelstate.ImmutableKeyError as error:
            logger.error('%s', error)

    def _instrument_sensors(self, view):
        # Only the sensors captured by cam2telstate are simulated
        self.sensor(view, 'adc_sample_rate', self.adc_rate)
        self.sensor(view, 'scale_factor_timestamp', self.scale_factor_timestamp)
        self.sensor(view, 'sync_time', self.subarray.sync_time.secs)
        self.sensor(view, 'n_inputs', 2 * self.n_antennas)

    def _antenna_channelised_voltage_sensors(self, view):
        for i in range(2 * self.n_antennas):
            input_pre = 'input{}_'.format(i)
            # These are all arbitrary dummy values
            self.sensor(view, input_pre + 'delay', (0, 0, 0, 0, 0), immutable=False)
            self.sensor(view, input_pre + 'delay_ok', True, immutable=False)
            self.sensor(view, input_pre + 'eq', [200 + 0j], immutable=False)
            self.sensor(view, input_pre + 'fft0_shift', 32767, immutable=False)
        self.sensor(view, 'bandwidth', self.bandwidth)
        self.sensor(view, 'center_freq', float(self.center_frequency))
        self.sensor(view, 'n_chans', self.n_channels)
        self.sensor(view, 'ticks_between_spectra',
                    int(round(self.n_channels * self.scale_factor_timestamp / self.bandwidth)))

    def set_telstate(self, telstate):
        """Populate telstate with simulated sensors for the stream.

        Subclasses overload this to set stream-specific sensors.
        """
        # Check if the upstreams have already been configured. If not, invent
        # names for them.
        stream_view = telstate.view(self.name_norm, exclusive=True)
        srcs = stream_view.get('src_streams')
        if srcs is None:
            src = 'katcbfsim_antenna_channelised_voltage_{}'.format(self.name_norm)
            logger.warning('No src_streams found for %s, so defining stream %s',
                           self.name_norm, src)
            stream_view.add('src_streams', [src], immutable=True)
        else:
            src = srcs[0]
        src_view = telstate.view(src, exclusive=True)
        instrument = src_view.get('instrument_dev_name')
        if instrument is None:
            instrument = 'katcbfsim_instrument_{}'.format(src)
            logger.warning('No instrument_dev_name found for %s, so defining %s',
                           src, instrument)
            src_view.add('instrument_dev_name', instrument, immutable=True)
        self._instrument_sensors(telstate.view(instrument, exclusive=True))
        self._antenna_channelised_voltage_sensors(src_view)


class FXStream(CBFStream):
    """Simulation of a correlation stream.

    Parameters
    ----------
    telstate : :class:`katsdptelstate.TelescopeState`
        Telescope state to populate with simulated sensors
    context : katsdpsigproc context
        Device context
    subarray : :class:`Subarray`
        Subarray corresponding to this stream
    name : str
        Name for this stream (used by katcp)
    adc_rate : float
        Simulated ADC clock rate, in Hz
    center_frequency : float
        Sky frequency of the center of the band, in Hz
    bandwidth : float
        Bandwidth of all channels in the stream, in Hz
    n_channels : int
        Number of channels in the stream
    n_substreams : int
        Number of substreams (X-engines)
    accumulation_length : float
        Approximate simulated integration time in seconds. It will be rounded
        to the nearest supported value.
    loop : :class:`asyncio.BaseEventLoop`, optional
        Event loop for coroutines

    Attributes
    ----------
    context : katsdpsigproc context
        Device context
    accumulation_length : float
        Simulated integration time in seconds. This is a property: setting the
        value will round it to the nearest supported value.
    wall_accumulation_length : float, read-only
        Minimum wall-clock time between emitting dumps. This is determined by
        :attr:`accumulation_length` and :attr:`Subarray.clock_ratio`.
    n_accs : int, read-only
        Number of simulated accumulations per output dump. This is set
        indirectly by writing to :attr:`accumulation_length`.
    """
    def __init__(self, context, subarray, name, adc_rate,
                 center_frequency, bandwidth, n_channels, n_substreams,
                 accumulation_length=None, loop=None):
        super(FXStream, self).__init__(subarray, name, adc_rate, center_frequency,
                                       bandwidth, n_channels, n_substreams, loop)
        self.context = context
        if accumulation_length is None:
            accumulation_length = 0.5
        self.accumulation_length = accumulation_length
        self.sefd = 400.0   # Jansky
        self.seed = 1

    @property
    def wall_accumulation_length(self):
        return self.subarray.clock_ratio * self.accumulation_length

    @property
    def accumulation_length(self):
        """Integration time in seconds. This is a property: setting the value
        will round it to the nearest supported value.
        """
        return self._accumulation_length

    @accumulation_length.setter
    def accumulation_length(self, value):
        # Round the accumulation length in the same way the real correlator
        # would. It requires a multiple of 256 accumulations.
        snap_length = self.n_channels / self.bandwidth
        self._n_accs = int(round(value / snap_length / 256)) * 256
        self._accumulation_length = snap_length * self.n_accs

    @property
    def n_accs(self):
        return self._n_accs

    def capture_start(self, start_time):
        """Begin capturing data, if it has not been done already.

        If no sources are defined, one is added at the phase center.

        Parameters
        ----------
        start_time : :class:`katpoint.Timestamp`
            Start time in simulated world

        Raises
        ------
        ConfigError
            if the subarray has no target
        ConfigError
            if no destination is defined
        """
        if not self.transport_factories:
            raise ConfigError('no destination specified')
        self.subarray.ensure_source(start_time)
        super(FXStream, self).capture_start(start_time)

    def _bandpass(self):
        """Create an approximate bandpass shape."""
        rs = np.random.RandomState(seed=self.seed)
        nx = 20
        x = np.linspace(0.0, self.n_channels, nx)
        # Set up a mostly flat bandpass with rolloff at the edges, in log-space.
        y = np.zeros(nx)
        y[-1] = y[0] = np.log(0.1)    # 10 dB rolloff
        y[:] += rs.normal(scale=0.01, size=y.shape)
        f = scipy.interpolate.interp1d(x, y, kind='cubic', assume_sorted=True)
        bp = f(np.arange(self.n_channels) + 0.5)
        # Make every 700th channel a spike. If power-of-two blocks of channels
        # get reordered, this should be very noticeable.
        bp[::700] += np.linspace(0.0, 2.5, len(bp[::700]))
        bp = np.exp(bp)
        return bp[self.channel_range]

    def _make_predict(self):
        """Compiles the kernel, allocates memory etc. This is potentially slow,
        so it is run in a separate thread to avoid blocking the event loop.

        Returns
        -------
        predict : :class:`~katcbfsim.rime.Rime`
            Visibility predictor
        data : list of :class:`~katsdpsigproc.accel.DeviceArray`
            Device storage for visibilities
        host : list of :class:`~katsdpsigproc.accel.HostArray`
            Pinned memory for transfers to the host
        """
        queue = self.context.create_command_queue()
        template = rime.RimeTemplate(self.context, len(self.subarray.antennas))
        my_n_channels = self.n_channels // self.subarray.n_servers
        my_bandwidth = self.bandwidth / self.subarray.n_servers
        my_mid_channel = (self.channel_range.start + self.channel_range.stop) / 2.0
        mid_channel = self.n_channels / 2.0
        my_center_frequency = self.center_frequency \
            + (my_mid_channel - mid_channel) * self.bandwidth / self.n_channels
        predict = template.instantiate(
            queue, my_center_frequency, my_bandwidth,
            my_n_channels, self.n_accs,
            self.subarray.sources, self.subarray.antennas,
            self.sefd, self.seed, async=True)
        predict.ensure_all_bound()
        # Initialise gains. Eventually this will need to be more sophisticated, but
        # for now it is just real and diagonal.
        predict.gain.fill(0)
        baseline_gain = (self.subarray.gain * self.bandwidth / self.n_channels
                         * self.accumulation_length / self.n_accs)
        antenna_gain = math.sqrt(baseline_gain)
        bandpass = self._bandpass() * antenna_gain
        predict.gain[:, :, 0, 0] = bandpass[:, np.newaxis]
        predict.gain[:, :, 1, 1] = bandpass[:, np.newaxis]
        data = [predict.buffer('out')]
        data.append(accel.DeviceArray(self.context, data[0].shape, data[0].dtype,
                                      data[0].padded_shape))
        host = [x.empty_like() for x in data]
        return predict, data, host

    async def _run_dump(self, index, predict_a, data_a, host_a, transports_a, io_queue_a):
        """Coroutine that does all the processing for a single dump. More than
        one of these will be active at a time, and they use
        :class:`katsdpsigproc.resource.Resource` to avoid data hazards and to
        prevent out-of-order execution.
        """
        try:
            with predict_a as predict, data_a as data, host_a as host, \
                    transports_a as transports, io_queue_a as io_queue:
                # Prepare the predictor object.
                # No need to wait for events on predict, because the object has its own
                # command queue to serialise use.
                await predict_a.wait()
                predict.bind(out=data)
                dump_start_time = self.start_time + index * self.accumulation_length
                # Set the timestamp for the center of the integration period
                dump_center_time = dump_start_time + 0.5 * self.accumulation_length
                predict.set_time(dump_center_time)
                predict.set_phase_center(self.subarray.target_at(dump_start_time))
                predict.set_position(self.subarray.position_at(dump_start_time))

                # Execute the predictor, updating data
                logger.debug('Dump %d: waiting for device memory event', index)
                events = await data_a.wait()
                logger.debug('Dump %d: device memory wait event found', index)
                predict.command_queue.enqueue_wait_for_events(events)
                logger.debug('Dump %d: device memory wait queued', index)
                predict_ready_event = predict()
                logger.debug('Dump %d: kernel queued', index)
                compute_event = predict.command_queue.enqueue_marker()
                predict.command_queue.flush()
                logger.debug('Dump %d: kernel flushed', index)
                predict_a.ready([predict_ready_event])   # Predict object can be reused now
                logger.debug('Dump %d: operation enqueued, waiting for host memory', index)

                # Transfer the data back to the host
                await io_queue_a.wait()   # Just to ensure ordering - no data hazards
                await host_a.wait()
                io_queue.enqueue_wait_for_events([compute_event])
                data.get_async(io_queue, host)
                io_queue.flush()
                logger.debug('Dump %d: transfer to host queued', index)
                transfer_event = io_queue.enqueue_marker()
                data_a.ready([transfer_event])
                io_queue_a.ready()
                # Wait for the transfer to complete
                await resource.async_wait_for_events([transfer_event], loop=self.loop)
                logger.debug('Dump %d: transfer to host complete, waiting for transport', index)

                # Send the data
                await transports_a.wait()        # Just to ensure ordering
                logger.debug('Dump %d: starting transmission', index)
                for transport in transports:
                    await transport.send(host, index)
                host_a.ready()
                transports_a.ready()
                logger.debug('Dump %d: complete', index)
        except Exception:
            logging.error('Dump %d: exception', index, exc_info=True)

    async def _capture(self):
        """Capture co-routine, started by :meth:`capture_start` and joined by
        :meth:`capture_stop`.
        """
        logger.info('_capture started')
        # Grab the start time here, rather than just before the loop. This is
        # necessary when running a distributed simulation, because the
        # initialisation takes a highly variable amount of time and can cause
        # the separate simulators to desynchronise. We instead estimate a
        # reasonable bound on the startup time and add that, so that we don't
        # start out with a flurry of "falling behind" messages.
        wall_time = self.loop.time() + 20
        # Futures corresponding to _run_dump coroutine calls
        dump_futures = collections.deque()
        transports = []
        try:
            transports = [factory(self) for factory in self.transport_factories]
            predict, data, host = await self.loop.run_in_executor(None, self._make_predict)
            index = 0
            predict_r = resource.Resource(predict, loop=self.loop)
            io_queue_r = resource.Resource(self.context.create_command_queue(), loop=self.loop)
            data_r = [resource.Resource(x, loop=self.loop) for x in data]
            host_r = [resource.Resource(x, loop=self.loop) for x in host]
            transports_r = resource.Resource(transports, loop=self.loop)
            if self.subarray.n_servers > 1:
                logger.info('waiting for start time')
                await self.wait_for_next(wall_time)
            else:
                # No need to wait, we don't have anyone to sync with
                wall_time = self.loop.time()
            logger.info('simulation starting')
            for transport in transports:
                await transport.send_metadata()
            while self.n_dumps is None or index < self.n_dumps:
                predict_a = predict_r.acquire()
                data_a = data_r[index % len(data_r)].acquire()
                host_a = host_r[index % len(host_r)].acquire()
                transports_a = transports_r.acquire()
                io_queue_a = io_queue_r.acquire()
                # Don't start the dump coroutine until the predictor is ready.
                # This ensures that if dumps can't keep up with the rate, then
                # we will block here rather than building an ever-growing set
                # of active coroutines.
                logger.debug('Dump %d: waiting for predictor', index)
                await predict_a.wait_events()
                logger.debug('Dump %d: predictor ready', index)
                future = self.loop.create_task(
                    self._run_dump(index, predict_a, data_a, host_a, transports_a, io_queue_a))
                dump_futures.append(future)
                # Sleep until either it is time to make the next dump, or we are asked
                # to stop.
                wall_time += self.wall_accumulation_length
                stopped = await self.wait_for_next(wall_time)
                if stopped:
                    break
                # Reap any previously completed coroutines
                while dump_futures and dump_futures[0].done():
                    dump_futures[0].result()  # Re-throws exceptions here
                    dump_futures.popleft()
                index += 1
            logging.info('Stop requested, waiting for in-flight dumps...')
            while dump_futures:
                await dump_futures[0]
                dump_futures.popleft()
            logging.info('Capture stopped by request')
            for transport in transports:
                await transport.close()
        except Exception:
            logger.error('Exception in capture coroutine', exc_info=True)
            for future in dump_futures:
                if not future.done():
                    future.cancel()
            for transport in transports:
                await transport.close()

    def _baseline_correlation_products_sensors(self, view):
        baselines = []
        for i in range(self.n_antennas):
            for j in range(i, self.n_antennas):
                for pol1 in ('v', 'h'):
                    for pol2 in ('v', 'h'):
                        name1 = self.subarray.antennas[i].name + pol1
                        name2 = self.subarray.antennas[j].name + pol2
                        baselines.append((name1, name2))
        self.sensor(view, 'bandwidth', self.bandwidth)
        self.sensor(view, 'bls_ordering', baselines)
        self.sensor(view, 'int_time', self.accumulation_length)
        self.sensor(view, 'n_accs', self.n_accs)
        self.sensor(view, 'n_chans_per_substream', self.n_channels // self.n_substreams)

    def set_telstate(self, telstate):
        super(FXStream, self).set_telstate(telstate)
        self._baseline_correlation_products_sensors(telstate.view(self.name_norm))


class BeamformerStream(CBFStream):
    """Simulated beam-former. The individual antennas are not simulated, but
    are still used to provide input labelling.

    Parameters
    ----------
    subarray : :class:`Subarray`
        Subarray corresponding to this stream
    name : str
        Name for this stream (used by katcp)
    adc_rate : float
        Simulated ADC clock rate, in Hz
    center_frequency : float
        Sky frequency of the center of the band, in Hz
    bandwidth : float
        Bandwidth of all channels in the stream, in Hz
    n_channels : int
        Number of channels in the stream
    n_substreams : int
        Number of substreams (B-engines)
    timesteps : int
        Number of samples in time accumulated into a single update. This is
        used by :class:`BeamformerSpeadTransport` to decide the data shape.
    sample_bits : int
        Number of bits per output sample (for each of real and imag). Currently
        this must be 8, 16 or 32.
    loop : :class:`asyncio.BaseEventLoop`, optional
        Event loop for coroutines

    Attributes
    ----------
    timesteps : int
        Number of samples in time accumulated into a single update. This is
        used by :class:`BeamformerSpeadTransport` to decide the data shape.
    sample_bits : int
        Number of bits per output sample (for each of real and imag). Currently
        this must be 8, 16 or 32.
    dtype : numpy data type
        Data type corresponding to :attr:`sample_bits`
    interval : float
        Seconds between output dumps, in simulation time
    wall_interval : float
        Equivalent to :attr:`interval`, but in wall-clock time
    """
    def __init__(self, subarray, name, adc_rate, center_frequency, bandwidth,
                 n_channels, n_substreams, timesteps, sample_bits, loop=None):
        super(BeamformerStream, self).__init__(
            subarray, name, adc_rate, center_frequency, bandwidth,
            n_channels, n_substreams, loop)
        self.timesteps = timesteps
        self.sample_bits = sample_bits
        if sample_bits == 8:
            self.dtype = np.int8
        elif sample_bits == 16:
            self.dtype = np.int16
        elif sample_bits == 32:
            self.dtype = np.int32
        else:
            raise ConfigError('sample_bits must be 8, 16 or 32')

    @property
    def interval(self):
        return self.timesteps * self.n_channels / self.bandwidth

    @property
    def wall_interval(self):
        return self.subarray.clock_ratio * self.interval

    async def _run_dump(self, transports, index):
        my_n_channels = self.n_channels // self.subarray.n_servers
        data = np.zeros((my_n_channels,
                         self.timesteps, 2), self.dtype)
        # Stuff in some patterned values to help test decoding
        for i in range(self.timesteps):
            value = index * self.timesteps + i
            data[value % my_n_channels, value % self.timesteps, 0] = value & 0x7f
            data[value % my_n_channels, value % self.timesteps, 1] = (value >> 15) & 0x7f
        for transport in transports:
            await transport.send(data, index)

    async def _capture(self):
        """Capture co-routine, started by :meth:`capture_start` and joined by
        :meth:`capture_stop`.
        """
        transports = None
        dump_futures = collections.deque()
        try:
            transports = [factory(self) for factory in self.transport_factories]
            for transport in transports:
                await transport.send_metadata()
            index = 0
            wall_time = self.loop.time()
            while self.n_dumps is None or index < self.n_dumps:
                while len(dump_futures) > 3:
                    await dump_futures[0]
                    dump_futures.popleft()
                future = asyncio.ensure_future(
                    self._run_dump(transports, index), loop=self.loop)
                dump_futures.append(future)
                wall_time += self.wall_interval
                stopped = await self.wait_for_next(wall_time)
                if stopped:
                    break
                # Reap any previously completed coroutines
                while dump_futures and dump_futures[0].done():
                    dump_futures[0].result()  # Re-throws exceptions here
                    dump_futures.popleft()
                index += 1
            logging.info('Stop requested, waiting for in-flight dumps...')
            while dump_futures:
                await dump_futures[0]
                dump_futures.popleft()
            logging.info('Capture stopped by request')
            for transport in transports:
                await transport.close()
        except Exception:
            logger.error('Exception in capture coroutine', exc_info=True)
            if transports is not None:
                for transport in transports:
                    await transport.close()

    def _tied_array_channelised_voltage_sensors(self, view):
        self.sensor(view, 'bandwidth', self.bandwidth)
        self.sensor(view, 'n_chans', self.n_channels, immutable=False)
        self.sensor(view, 'n_chans_per_substream', self.n_channels // self.n_substreams)
        self.sensor(view, 'spectra_per_heap', self.timesteps)
        for i in range(2 * self.n_antennas):
            self.sensor(view, 'input{}_weight'.format(i), 1.0, immutable=False)

    def set_telstate(self, telstate):
        super(BeamformerStream, self).set_telstate(telstate)
        self._tied_array_channelised_voltage_sensors(telstate.view(self.name_norm))
