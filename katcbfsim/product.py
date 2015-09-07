from __future__ import print_function, division
import trollius
import logging
import time
import collections
from trollius import From
from katsdptelstate import endpoint
from katsdpsigproc import accel
import katpoint
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


class ResourceAllocation(object):
    def __init__(self, start, end, value):
        self._start = start
        self._end = end
        self.value = value

    def wait(self):
        return self._start

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
    sync_time : :class:`katpoint.Timestamp`
        Start time for the simulated capture. When set, it is truncated to a
        whole number of seconds.
    """
    def __init__(self):
        self.antennas = []
        self.sources = []
        self.target = None
        self._sync_time = time.time()
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


class FXProduct(object):
    """Simulation of a correlation product.

    Unless otherwise documented, changing any attributes while a capture is in
    progress has undefined behaviour. In some cases, but not all, this is
    enforced by a :exc:`CaptureInProgressError`.

    Parameters
    ----------
    context : katsdpsigproc context
        Device context
    subarray : :class:`Subarray`
        Subarray corresponding to this product
    name : str
        Name for this product (used by katcp)
    bandwidth : int
        Total bandwidth over all channels, in Hz
    channels : int
        Number of channels
    loop : :class:`trollius.BaseEventLoop`, optional
        Event loop for coroutines

    Attributes
    ----------
    context : katsdpsigproc context
        Device context
    subarray : :class:`Subarray`
        Subarray corresponding to this product
    name : :class:`str`
        Name for this product (used by katcp)
    bandwidth : int
        Total bandwidth over all channels, in Hz
    channels : int
        Number of channels
    center_frequency : int
        Frequency of the center of the band, in Hz
    destination_factory : callable
        Called with `self` to return a stream on which output will be sent.
    accumulation_length : float
        Simulated integration time in seconds. This is a property: setting the
        value will round it to the nearest supported value.
    wall_accumulation_length : float, read-only
        Minimum wall-clock time between emitting dumps. This is determined by
        :attr:`accumulation_length` and :attr:`time_scale`.
    time_scale : float
        Scale factor between virtual time in the simulation and wall clock
        time. Smaller values will run the simulation faster; setting it to 0
        will cause the simulation to run as fast as possible.
    n_accs : int, read-only
        Number of simulated accumulations per output dump. This is set
        indirectly by writing to :attr:`accumulation_length`.
    capturing : bool, read-only
        Whether a capture is in progress. This is true from
        :meth:`capture_start` until :meth:`capture_stop` completes, even if
        the capture coroutine terminates earlier.
    """
    def __init__(self, context, subarray, name, bandwidth, channels, loop=None):
        self._capture_future = None
        self._stop_future = None
        self.context = context
        self.name = name
        self.subarray = subarray
        self.destination_factory = None
        self.bandwidth = bandwidth
        self.channels = channels
        self.center_frequency = 1412000000
        self.accumulation_length = 0.5
        self.time_scale = 1.0
        if loop is None:
            self._loop = trollius.get_event_loop()
        else:
            self._loop = loop

    def __setattr__(self, key, value):
        # Prevent modifications while capture is in progress
        if not key.startswith('_') and self.capturing:
            raise CaptureInProgressError('cannot set {} while capture is in progress'.format(key))
        return super(FXProduct, self).__setattr__(key, value)

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
        snap_length = self.channels / self.bandwidth
        self._n_accs = int(round(value / snap_length / 256)) * 256
        self._accumulation_length = snap_length * self.n_accs

    @property
    def n_accs(self):
        return self._n_accs

    @property
    def capturing(self):
        return self._capture_future is not None

    def capture_start(self):
        """Begin capturing data, if it has not been done already. The
        capturing is done on the trollius event loop, which must thus be
        allowed to run frequency to ensure timeous delivery of results.

        Raises
        ------
        IncompleteConfigError
            if the subarray has no sources
        IncompleteConfigError
            if the subarray has no antennas
        """
        if self.capturing:
            logger.warn('Ignoring attempt to start capture when already running')
            return
        if not self.subarray.antennas:
            raise IncompleteConfigError('no antennas defined')
        if not self.subarray.sources:
            raise IncompleteConfigError('no sources defined')
        if self.destination_factory is None:
            raise IncompleteConfigError('no destination specified')
        if self.subarray.target is None:
            raise IncompleteConfigError('no target set')
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
            self.channels, self.subarray.sources, self.subarray.antennas)
        predict.ensure_all_bound()
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
                # Set the timestamp for the center of the integration period
                predict.set_time(self.subarray.sync_time + (index + 0.5) * self.accumulation_length)
                predict.set_phase_center(self.subarray.target)

                # Execute the predictor, updating data
                logger.debug('Dump %d: waiting for device memory event', index)
                events = yield From(data_a.wait())
                logger.debug('Dump %d: device memory wait queued', index)
                predict.command_queue.enqueue_wait_for_events(events)
                predict()
                compute_event = predict.command_queue.enqueue_marker()
                predict.command_queue.flush()
                predict_a.ready()   # Predict object can be reused now
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
                yield From(self._loop.run_in_executor(None, transfer_event.wait))
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
                yield From(predict_a.wait())
                logger.debug('Dump %d: predictor ready', index)
                future = trollius.async(
                    self._run_dump(index, predict_a, data_a, host_a, stream_a, io_queue_a),
                    loop=self._loop)
                dump_futures.append(future)
                # Sleep until either it is time to make the next dump, or we are asked
                # to stop.
                wall_time += self.wall_accumulation_length
                try:
                    # TODO: support alternative rate of time
                    yield From(wait_until(trollius.shield(self._stop_future), wall_time, self._loop))
                except trollius.TimeoutError:
                    # This is the normal case: time for the next dump to be transmitted
                    pass
                else:
                    # The _stop_future was triggered
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
