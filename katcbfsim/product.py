from __future__ import print_function, division
import trollius
import logging
import time
from trollius import From
from katsdptelstate import endpoint
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
        self.destination = None
        self.bandwidth = bandwidth
        self.channels = channels
        self.center_frequency = 1412000000
        self.accumulation_length = 0.5
        self._active = False
        self._capture_future = None
        self._stop_future = None
        if loop is None:
            self._loop = trollius.get_event_loop()
        else:
            self._loop = loop

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
        """
        queue = self.context.create_command_queue()
        template = rime.RimeTemplate(self.context, len(self.subarray.antennas))
        predict = template.instantiate(
            queue, self.center_frequency, self.bandwidth,
            self.channels, self.subarray.sources, self.subarray.antennas)
        predict.ensure_all_bound()
        return predict

    @trollius.coroutine
    def _capture(self):
        try:
            predict = yield From(self._loop.run_in_executor(None, self._make_predict))
            wall_time = self._loop.time()
            index = 0
            while True:
                # TODO: set time and pointing for prediction
                predict()
                predict.command_queue.finish()
                logging.info('Prepared dump %d', index)
                # Sleep until either it is time to make the dump, or we are asked
                # to stop.
                wall_time += self.accumulation_length
                try:
                    # TODO: support alternative rate of time
                    yield From(wait_until(trollius.shield(self._stop_future), wall_time, self._loop))
                except trollius.TimeoutError:
                    # This is the normal case: time for the next dump to be transmitted
                    pass
                else:
                    # The _stop_future was triggered
                    break
                # TODO: emit the visibilities
                index += 1
            logging.info('Capture stopped by request')
        except Exception:
            logger.error('Exception in capture coroutine', exc_info=True)
