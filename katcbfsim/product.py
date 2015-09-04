from __future__ import print_function, division
import trollius
import logging
import time
import collections
from trollius import From
from katsdptelstate import endpoint
from katsdpsigproc import accel
from . import rime


logger = logging.getLogger(__name__)


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
    def __init__(self):
        self.antennas = []
        self.sources = []
        self.sync_time = time.time()

    def clone(self):
        # copy.deepcopy doesn't work on Antenna objects
        other = Subarray()
        other.antennas = list(self.antennas)
        other.sources = list(self.sources)
        other.sync_time = self.sync_time
        return other


class FXProduct(object):
    def __init__(self, context, subarray, name, bandwidth, channels, loop=None):
        self.context = context
        self.name = name
        self.subarray = subarray
        self.destination_factory = None
        self.bandwidth = bandwidth
        self.channels = channels
        self.center_frequency = 1412000000
        self.accumulation_length = 0.5
        self.time_scale = 1.0
        self._active = False
        self._capture_future = None
        self._stop_future = None
        if loop is None:
            self._loop = trollius.get_event_loop()
        else:
            self._loop = loop

    @property
    def wall_accumulation_length(self):
        return self.time_scale * self.accumulation_length

    @property
    def capturing(self):
        return self._capture_future is not None

    def capture_start(self):
        if self.capturing:
            logger.warn('Ignoring attempt to start capture when already running')
            return
        if not self.subarray.antennas:
            raise ValueError('No antennas defined')
        if not self.subarray.sources:
            raise ValueError('No sources defined')
        # Freeze the subarray so that we're not affected by changes on the
        # master copy
        self.subarray = self.subarray.clone()
        # Create a future that is set by capture_stop
        self._stop_future = trollius.Future(loop=self._loop)
        # Start the capture coroutine on the event loop
        self._capture_future = trollius.async(self._capture(), loop=self._loop)

    @trollius.coroutine
    def capture_stop(self):
        if not self.capturing:
            logger.warn('Ignoring attempt to stop capture when not running')
            return
        self._stop_future.set_result(None)
        yield From(self._capture_future)
        self._stop_future = None
        self._capture_future = None

    def _make_predict(self):
        """Compiles the kernel, allocates memory etc. This is potentially slow,
        so it is run in a separate thread to avoid blocking asyncio.

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
        try:
            with predict_a as predict, data_a as data, host_a as host, \
                    stream_a as stream, io_queue_a as io_queue:
                # Prepare the predictor object.
                # No need to wait for events on predict, because the object has its own
                # command queue to serialise use.
                yield From(predict_a.wait())
                predict.bind(out=data)
                # TODO: set time, pointing

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
                yield From(stream.send(host))
                host_a.ready()
                stream_a.ready()
                logger.debug('Dump %d: complete', index)
        except Exception:
            logging.error('Dump %d: exception', index, exc_info=True)

    @trollius.coroutine
    def _capture(self):
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
            logging.info('Capture stopped by request')
            yield From(destination.close())
        except Exception:
            logger.error('Exception in capture coroutine', exc_info=True)
            for future in dump_futures:
                if not future.done():
                    future.cancel()
