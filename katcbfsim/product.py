from __future__ import print_function, division
import trollius
import copy
import logging
import time
from trollius import From
from katsdptelstate import endpoint


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


class FXProduct(object):
    def __init__(self, subarray, name, bandwidth, channels, loop=None):
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
        # Freeze the subarray so that we're not affected by changes on the
        # master copy
        self.subarray = copy.deepcopy(self.subarray)
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

    @trollius.coroutine
    def _capture(self):
        try:
            wall_time = self._loop.time()
            index = 0
            while True:
                # TODO: compute visibilities
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
