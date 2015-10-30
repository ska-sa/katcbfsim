from __future__ import print_function, division
import trollius
import logging
import collections
from trollius import From, Return
from katsdptelstate import endpoint
from katsdpsigproc import accel
import katpoint
import numpy as np
from . import rime


logger = logging.getLogger(__name__)


class CaptureInProgressError(RuntimeError):
    """Exception thrown when trying to modify some value that may not be
    modified while data capture is running.
    """
    pass


class IncompleteConfigError(RuntimeError):
    """Exception thrown when requesting capture but the product or subarray
    is missing necessary configuration.
    """
    pass


class UnsupportedProductError(RuntimeError):
    """Exception thrown for operations not supported on this product type."""
    pass


@trollius.coroutine
def wait_until(future, when, loop=None):
    """Like :meth:`trollius.wait_for`, but with an absolute timeout."""
    def ready(*args):
        if not waiter.done():
            waiter.set_result(None)

    if loop is None:
        loop = trollius.get_event_loop()
    waiter = trollius.Future(loop=loop)
    timeout_handle = loop.call_at(when, ready)
    # Ensure the that future is really a future, not a coroutine object
    future = trollius.async(future, loop=loop)
    future.add_done_callback(ready)
    try:
        result = yield From(waiter)
        if future.done():
            raise trollius.Return(future.result())
        else:
            future.remove_done_callback(ready)
            future.cancel()
            raise trollius.TimeoutError()
    finally:
        timeout_handle.cancel()


@trollius.coroutine
def _async_wait_for_events(events, loop=None):
    def wait_for_events(events):
        for event in events:
            event.wait()
    if loop is None:
        loop = trollius.get_event_loop()
    if events:
        yield From(loop.run_in_executor(None, wait_for_events, events))


class ResourceAllocation(object):
    def __init__(self, start, end, value):
        self._start = start
        self._end = end
        self.value = value

    def wait(self):
        return self._start

    @trollius.coroutine
    def wait_events(self, loop=None):
        if loop is None:
            loop = trollius.get_event_loop()
        events = yield From(self._start)
        yield From(_async_wait_for_events(events, loop=loop))

    def ready(self, events=None):
        if events is None:
            events = []
        self._end.set_result(events)

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc_value, exc_tb):
        if not self._end.done():
            if exc_type is not None:
                self._end.cancel()
            else:
                logger.warn('Resource allocation was not explicitly made ready')
                self.ready()


class Resource(object):
    """Abstraction of a contended resource, which may exist on a device.

    Passing of ownership is done via futures. Acquiring a resource is a
    non-blocking operation that returns two futures: a future to wait for
    before use, and a future to be signalled with a result when done. The
    value of each of these futures is a (possibly empty) list of device
    events which must be waited on before more device work is scheduled.
    """
    def __init__(self, value):
        self._future = trollius.Future()
        self._future.set_result([])
        self.value = value

    def acquire(self):
        old = self._future
        self._future = trollius.Future()
        return ResourceAllocation(old, self._future, self.value)


class Subarray(object):
    """Model of an array, the sky, and possibly other simulation parameters,
    shared by several data products. A subarray should first be configured
    before starting capture on any of the corresponding products, and must
    remain immutable (except where noted) while any product is capturing.
    This is partly enforced by exceptions thrown from setters, but the
    :attr:`antennas` and :attr:`sources` attributes are mutable lists and so
    suitable care must be taken.

    Attributes
    ----------
    antennas : list of :class:`katpoint.Antenna`
        The antennas in the simulated array.
    sources : list of :class:`katpoint.Target`
        The simulated sources. Only point sources are currently supported.
        The do not necessarily have to be radec targets, and the position
        and flux model can safely be changed on the fly.
    target : :class:`katpoint.Target`
        Target. This determines the phase center for the simulation (and
        eventually the center for the beam model as well).
    sync_time : :class:`katpoint.Timestamp`
        Start time for the simulated capture. When set, it is truncated to a
        whole number of seconds.
    """
    def __init__(self):
        self.antennas = []
        self.sources = []
        self.target = None
        self._sync_time = katpoint.Timestamp()
        self.capturing = 0     # Number of products that are capturing

    def add_antenna(self, antenna):
        """Add a new antenna to the simulation.

        Parameters
        ----------
        antenna : :class:`katpoint.Antenna`
            New antenna

        Raises
        ------
        CaptureInProgressError
            If any associated product has a capture in progress
        """
        if self.capturing:
            raise CaptureInProgressError('cannot add antennas while capture in progress')
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
            If any associated product has a capture in progress
        """
        if self.capturing:
            raise CaptureInProgressError('cannot add antennas while capture in progress')
        if source.flux_model is None:
            logging.warn('source has no flux model; it will be assumed to be 1 Jy')
        self.sources.append(source)

    @property
    def sync_time(self):
        return self._sync_time

    @sync_time.setter
    def sync_time(self, value):
        if self.capturing:
            raise CaptureInProgressError('cannot change sync time while capture in progress')
        self._sync_time = katpoint.Timestamp(int(value.secs))

    def target_at(self, timestamp):
        """Obtains the target at a given point in simulated time. In this base
        class this just returns :attr:`target`, but it can be overridden by
        subclasses to allow the target to be pulled rather than pushed."""
        return self.target


class Product(object):
    """Base class for simulated products. It provides the basic machinery to
    start and stop capture running in a separate asyncio task.

    Unless otherwise documented, changing any attributes while a capture is in
    progress has undefined behaviour. In some cases, but not all, this is
    enforced by a :exc:`CaptureInProgressError`.

    Parameters
    ----------
    subarray : :class:`Subarray`
        Subarray corresponding to this product
    name : str
        Name for this product (used by katcp)
    loop : :class:`trollius.BaseEventLoop`, optional
        Event loop for coroutines

    Attributes
    ----------
    name : :class:`str`
        Name for this product (used by katcp)
    subarray : :class:`Subarray`
        Subarray corresponding to this product
    n_antennas : int, read-only
        Number of antennas
    n_baselines : int, read-only
        Number of baselines (antenna pairs, not input pairs)
    destination_factory : callable
        Called with `self` to return a stream on which output will be sent.
    time_scale : float
        Scale factor between virtual time in the simulation and wall clock
        time. Smaller values will run the simulation faster; setting it to 0
        will cause the simulation to run as fast as possible.
    capturing : bool, read-only
        Whether a capture is in progress. This is true from
        :meth:`capture_start` until :meth:`capture_stop` completes, even if
        the capture coroutine terminates earlier.
    """
    def __init__(self, subarray, name, loop=None):
        self._capture_future = None
        self._stop_future = None
        self.name = name
        self.subarray = subarray
        self.time_scale = 1.0
        self.destination_factory = None
        if loop is None:
            self._loop = trollius.get_event_loop()
        else:
            self._loop = loop

    def __setattr__(self, key, value):
        # Prevent modifications while capture is in progress
        if not key.startswith('_') and self.capturing:
            raise CaptureInProgressError('cannot set {} while capture is in progress'.format(key))
        return super(Product, self).__setattr__(key, value)

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

    def capture_start(self):
        """Begin capturing data, if it has not been done already. The
        capturing is done on the trollius event loop, which must thus be
        allowed to run frequency to ensure timeous delivery of results.

        Subclasses may override this to provide consistency checks on the
        state. They must also provide the :meth:`_capture` coroutine.

        Raises
        ------
        IncompleteConfigError
            if no destination is defined
        """
        if self.capturing:
            logger.warn('Ignoring attempt to start capture when already running')
            return
        if self.destination_factory is None:
            raise IncompleteConfigError('no destination specified')
        self.subarray.capturing += 1
        # Create a future that is set by capture_stop
        self._stop_future = trollius.Future(loop=self._loop)
        # Start the capture coroutine on the event loop
        self._capture_future = trollius.async(self._capture(), loop=self._loop)

    @trollius.coroutine
    def capture_stop(self):
        """Request an end to data capture, and wait for capture to complete.
        This is a coroutine.
        """
        if not self.capturing:
            logger.warn('Ignoring attempt to stop capture when not running')
            return
        self._stop_future.set_result(None)
        yield From(self._capture_future)
        self._stop_future = None
        self._capture_future = None
        self.subarray.capturing -= 1

    @trollius.coroutine
    def wait_for_next(self, wall_time):
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
            now = self._loop.time()
            if now > wall_time:
                logger.warn('Falling behind the requested rate by %f seconds', now - wall_time)
            else:
                logger.debug('Sleeping for %f seconds', wall_time - now)
            yield From(wait_until(trollius.shield(self._stop_future), wall_time, self._loop))
        except trollius.TimeoutError:
            # This is the normal case: time for the next dump to be transmitted
            stopped = False
        else:
            # The _stop_future was triggered
            stopped = True
        raise Return(stopped)


class CBFProduct(Product):
    """Parts that are shared between :class:`FXProduct` and :class:`BeamformerProduct`.

    Parameters
    ----------
    subarray : :class:`Subarray`
        Subarray corresponding to this product
    name : str
        Name for this product (used by katcp)
    adc_rate : int
        Simulated ADC clock rate, in Hz
    bandwidth : int
        Total bandwidth over all channels, in Hz
    n_channels : int
        Number of channels
    loop : :class:`trollius.BaseEventLoop`, optional
        Event loop for coroutines

    Attributes
    ----------
    adc_rate : int
        Simulated ADC clock rate, in Hz
    bandwidth : int
        Total bandwidth over all channels, in Hz
    n_channels : int
        Number of channels
    center_frequency : int
        Frequency of the center of the band, in Hz
    """
    def __init__(self, subarray, name, adc_rate, bandwidth, n_channels, loop=None):
        super(CBFProduct, self).__init__(subarray, name, loop)
        self.adc_rate = adc_rate
        self.bandwidth = bandwidth
        self.n_channels = n_channels
        self.center_frequency = 1284000000


class FXProduct(CBFProduct):
    """Simulation of a correlation product.

    Parameters
    ----------
    context : katsdpsigproc context
        Device context
    subarray : :class:`Subarray`
        Subarray corresponding to this product
    name : str
        Name for this product (used by katcp)
    adc_rate : int
        Simulated ADC clock rate, in Hz
    bandwidth : int
        Total bandwidth over all channels, in Hz
    n_channels : int
        Number of channels
    loop : :class:`trollius.BaseEventLoop`, optional
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
        :attr:`accumulation_length` and :attr:`time_scale`.
    n_accs : int, read-only
        Number of simulated accumulations per output dump. This is set
        indirectly by writing to :attr:`accumulation_length`.
    """
    def __init__(self, context, subarray, name, adc_rate, bandwidth, n_channels, loop=None):
        super(FXProduct, self).__init__(subarray, name, adc_rate, bandwidth, n_channels, loop)
        self.context = context
        self.accumulation_length = 0.5
        self.sefd = 20.0
        self.seed = 1

    @property
    def wall_accumulation_length(self):
        return self.time_scale * self.accumulation_length

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

    def capture_start(self):
        """Begin capturing data, if it has not been done already.

        Raises
        ------
        IncompleteConfigError
            if the subarray has no sources, no antennas, or no target
        IncompleteConfigError
            if no destination is defined
        """
        if not self.subarray.antennas:
            raise IncompleteConfigError('no antennas defined')
        if not self.subarray.sources:
            raise IncompleteConfigError('no sources defined')
        if self.destination_factory is None:
            raise IncompleteConfigError('no destination specified')
        if self.subarray.target_at(self.subarray.sync_time) is None:
            raise IncompleteConfigError('no target set')
        super(FXProduct, self).capture_start()

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
        predict = template.instantiate(
            queue, self.center_frequency, self.bandwidth,
            self.n_channels, self.n_accs,
            self.subarray.sources, self.subarray.antennas,
            self.sefd, self.seed, async=True)
        predict.ensure_all_bound()
        # Initialise gains. Eventually this will need to be more sophisticated, but
        # for now it is just real and diagonal.
        gain_host = predict.buffer('gain').empty_like()
        gain_host.fill(0)
        gain_host[:, :, 0, 0].fill(0.01)
        gain_host[:, :, 1, 1].fill(0.01)
        predict.buffer('gain').set(predict.command_queue, gain_host)
        data = [predict.buffer('out')]
        data.append(accel.DeviceArray(self.context, data[0].shape, data[0].dtype, data[0].padded_shape))
        host = [x.empty_like() for x in data]
        return predict, data, host

    @trollius.coroutine
    def _run_dump(self, index, predict_a, data_a, host_a, stream_a, io_queue_a):
        """Coroutine that does all the processing for a single dump. More than
        one of these will be active at a time, and they use :class:`Resource`
        avoid data hazards and to prevent out-of-order execution.
        """
        try:
            with predict_a as predict, data_a as data, host_a as host, \
                    stream_a as stream, io_queue_a as io_queue:
                # Prepare the predictor object.
                # No need to wait for events on predict, because the object has its own
                # command queue to serialise use.
                yield From(predict_a.wait())
                predict.bind(out=data)
                dump_start_time = self.subarray.sync_time + index * self.accumulation_length
                # Set the timestamp for the center of the integration period
                dump_center_time = dump_start_time + 0.5 * self.accumulation_length
                predict.set_time(dump_center_time)
                predict.set_phase_center(self.subarray.target_at(dump_start_time))

                # Execute the predictor, updating data
                logger.debug('Dump %d: waiting for device memory event', index)
                events = yield From(data_a.wait())
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
                yield From(io_queue_a.wait()) # Just to ensure ordering - no data hazards
                yield From(host_a.wait())
                io_queue.enqueue_wait_for_events([compute_event])
                data.get_async(io_queue, host)
                io_queue.flush()
                logger.debug('Dump %d: transfer to host queued', index)
                transfer_event = io_queue.enqueue_marker()
                data_a.ready([transfer_event])
                io_queue_a.ready()
                # Wait for the transfer to complete
                yield From(_async_wait_for_events([transfer_event], loop=self._loop))
                logger.debug('Dump %d: transfer to host complete, waiting for stream', index)

                # Send the data
                yield From(stream_a.wait())        # Just to ensure ordering
                logger.debug('Dump %d: starting transmission', index)
                yield From(stream.send(host, index))
                host_a.ready()
                stream_a.ready()
                logger.debug('Dump %d: complete', index)
        except Exception:
            logging.error('Dump %d: exception', index, exc_info=True)

    @trollius.coroutine
    def _capture(self):
        """Capture co-routine, started by :meth:`capture_start` and joined by
        :meth:`capture_stop`.
        """
        # Futures corresponding to _run_dump coroutine calls
        dump_futures = collections.deque()
        destination = None
        try:
            destination = self.destination_factory(self)
            yield From(destination.send_metadata())
            predict, data, host = yield From(self._loop.run_in_executor(None, self._make_predict))
            index = 0
            predict_r = Resource(predict)
            io_queue_r = Resource(self.context.create_command_queue())
            data_r = [Resource(x) for x in data]
            host_r = [Resource(x) for x in host]
            stream_r = Resource(destination)
            wall_time = self._loop.time()
            while True:
                predict_a = predict_r.acquire()
                data_a = data_r[index % len(data_r)].acquire()
                host_a = host_r[index % len(host_r)].acquire()
                stream_a = stream_r.acquire()
                io_queue_a = io_queue_r.acquire()
                # Don't start the dump coroutine until the predictor is ready.
                # This ensures that if dumps can't keep up with the rate, then
                # we will block here rather than building an ever-growing set
                # of active coroutines.
                logger.debug('Dump %d: waiting for predictor', index)
                yield From(predict_a.wait_events(self._loop))
                logger.debug('Dump %d: predictor ready', index)
                future = trollius.async(
                    self._run_dump(index, predict_a, data_a, host_a, stream_a, io_queue_a),
                    loop=self._loop)
                dump_futures.append(future)
                # Sleep until either it is time to make the next dump, or we are asked
                # to stop.
                wall_time += self.wall_accumulation_length
                stopped = yield From(self.wait_for_next(wall_time))
                if stopped:
                    break
                # Reap any previously completed coroutines
                while dump_futures and dump_futures[0].done():
                    dump_futures[0].result()  # Re-throws exceptions here
                    dump_futures.popleft()
                index += 1
            logging.info('Stop requested, waiting for in-flight dumps...')
            while dump_futures:
                yield From(dump_futures[0])
                dump_futures.popleft()
            logging.info('Capture stopped by request')
            yield From(destination.close())
        except Exception:
            logger.error('Exception in capture coroutine', exc_info=True)
            for future in dump_futures:
                if not future.done():
                    future.cancel()
            if destination is not None:
                yield From(destination.close())


class BeamformerProduct(CBFProduct):
    """Simulated beam-former. The individual antennas are not simulated, but
    are still used to provide input labelling.

    Parameters
    ----------
    subarray : :class:`Subarray`
        Subarray corresponding to this product
    name : str
        Name for this product (used by katcp)
    adc_rate : int
        Simulated ADC clock rate, in Hz
    bandwidth : int
        Total bandwidth over all channels, in Hz
    n_channels : int
        Number of channels
    timesteps : int
        Number of samples in time accumulated into a single update. This is
        used by :class:`BeamformerStreamSpead` to decide the data shape.
    sample_bits : int
        Number of bits per output sample (for each of real and imag). Currently
        this must be 8, 16 or 32.
    loop : :class:`trollius.BaseEventLoop`, optional
        Event loop for coroutines

    Attributes
    ----------
    timesteps : int
        Number of samples in time accumulated into a single update. This is
        used by :class:`BeamformerStreamSpead` to decide the data shape.
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
    def __init__(self, subarray, name, adc_rate, bandwidth, n_channels, timesteps, sample_bits, loop=None):
        super(BeamformerProduct, self).__init__(subarray, name, adc_rate, bandwidth, n_channels, loop)
        self.timesteps = timesteps
        self.sample_bits = sample_bits
        if sample_bits == 8:
            self.dtype = np.int8
        elif sample_bits == 16:
            self.dtype = np.int16
        elif sample_bits == 32:
            self.dtype = np.int32
        else:
            raise ValueError('sample_bits must be 8, 16 or 32')

    @property
    def interval(self):
        return self.timesteps * self.n_channels / self.bandwidth

    @property
    def wall_interval(self):
        return self.time_scale * self.interval

    @trollius.coroutine
    def _run_dump(self, destination, index):
        data = np.zeros((self.n_channels, self.timesteps, 2), self.dtype)
        yield From(destination.send(data, index))

    def _capture(self):
        destination = None
        dump_futures = collections.deque()
        try:
            destination = self.destination_factory(self)
            yield From(destination.send_metadata())
            index = 0
            wall_time = self._loop.time()
            while True:
                while len(dump_futures) > 3:
                    yield From(dump_futures[0])
                    dump_futures.popleft()
                future = trollius.async(
                    self._run_dump(destination, index), loop=self._loop)
                dump_futures.append(future)
                wall_time += self.wall_interval
                stopped = yield From(self.wait_for_next(wall_time))
                if stopped:
                    break
                # Reap any previously completed coroutines
                while dump_futures and dump_futures[0].done():
                    dump_futures[0].result()  # Re-throws exceptions here
                    dump_futures.popleft()
                index += 1
            logging.info('Stop requested, waiting for in-flight dumps...')
            while dump_futures:
                yield From(dump_futures[0])
                dump_futures.popleft()
            logging.info('Capture stopped by request')
            yield From(destination.close())
        except Exception:
            logger.error('Exception in capture coroutine', exc_info=True)
            if destination is not None:
                yield From(destination.close())
