from __future__ import print_function, division
import time
import copy
import trollius
from trollius import From
import katcp
import tornado
import logging
from katcp.kattypes import Str, Float, Int, Address, request, return_reply
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
        self.sync_time = time.time()
        self.accumulation_length = 0.5


class Product(object):
    def __init__(self, subarray, name, bandwidth, center_frequency, channels):
        self.name = name
        self.subarray = subarray
        self.destination = [endpoint.Endpoint('0.0.0.0', 0)]
        self.bandwidth = bandwidth
        self.center_frequency = center_frequency
        self.channels = channels
        self._capture_future = None
        self._stop_future = None

    def set_destination(self, destination):
        if isinstance(destination, str):
            destination = endpoint.endpoint_list_parser(0)(destination)
        self.destination = destination

    def capture_start(self):
        if self._capture_future is not None:
            logger.warn('Ignoring attempt to start capture when already running')
            return
        self._stop_future = trollius.Future()
        self._capture_future = trollius.async(self._capture())

    @trollius.coroutine
    def capture_stop(self):
        if self._capture_future is None:
            logger.warn('Ignoring attempt to stop capture when not running')
            return
        self._stop_future.set_result(None)
        yield From(self._capture_future)
        self._stop_future = None
        self._capture_future = None

    @trollius.coroutine
    def _capture(self):
        try:
            loop = trollius.get_event_loop()
            wall_time = loop.time()
            index = 0
            while True:
                # TODO: compute visibilities
                print('Prepared dump', index)
                # Sleep until either it is time to make the dump, or we are asked
                # to stop.
                wall_time += self.subarray.accumulation_length
                try:
                    yield From(wait_until(trollius.shield(self._stop_future), wall_time, loop))
                except trollius.TimeoutError:
                    # This is the normal case: time for the next dump to be transmitted
                    pass
                else:
                    # The _stop_future was triggered
                    break
                # TODO: emit the visibilities
                index += 1
            print('Capture stopped by request')
        except Exception:
            logger.error('Exception in capture coroutine', exc_info=True)


class SimulatorServer(katcp.DeviceServer):

    VERSION_INFO = ('katcbfsim-api', 1, 0)
    BUILD_INFO = ('katcbfsim', 0, 1, '')

    def __init__(self, *args, **kwargs):
        super(SimulatorServer, self).__init__(*args, **kwargs)
        self.products = {}
        self.subarray = Subarray()

    def setup_sensors(self):
        pass

    @request(Str(), Int(), Int(), Int())
    @return_reply()
    def request_instrument_create_correlator(self, sock, name, bandwidth, center_frequency, channels):
        """Create a new simulated correlator production"""
        if name in self.products:
            return 'fail', 'product {} already exists'.format(name)
        self.products[name] = Product(self.subarray, name, bandwidth, center_frequency, channels)
        return 'ok',

    @request(Str(), Address())
    @return_reply()
    def request_capture_destination(self, sock, stream, destination):
        """Set the destination endpoints for a stream"""
        if stream not in self.products:
            return 'fail', 'stream {} not in product list'.format(stream)
        self.products[stream].set_destination(destination)
        return 'ok',

    @request(Str(default=''))
    @return_reply()
    def request_capture_list(self, sock, stream):
        """List the destination endpoints for a stream, or all streams"""
        if stream != '' and stream not in self.products:
            return 'fail', 'requested product name not found'
        for name, product in self.products.items():
            if stream == '' or stream == name:
                # TODO: Add a formatter to katsdptelstate.endpoint that
                # reconstructs the a.b.c.d+N:port format.
                sock.inform(','.join([str(x) for x in product.destination]))
        return 'ok',

    @request(Int())
    @return_reply()
    def request_sync_time(self, sock, timestamp):
        """Set the sync time, as seconds since the UNIX epoch. This will also
        be the timestamp associated with the first data dump."""
        self.subarray.sync_time = timestamp
        return 'ok',

    @request(Float())
    @return_reply()
    def request_accumulation_length(self, sock, period):
        """Set the accumulation interval"""
        # TODO: apply rounding?
        self.subarray.accumulation_length = period
        return 'ok',

    @request(Str(), Float())
    @return_reply()
    def request_frequency_select(self, sock, stream, frequency):
        """Set the center frequency for the band. Unlike the real CBF, an
        arbitrary frequency may be selected, and it will not be rounded.
        """
        try:
            product = self.products[stream]
        except KeyError:
            return 'fail', 'requested product name not found'
        product.center_frequency = center_frequency
        return 'ok',

    @request(Str())
    @return_reply()
    def request_antenna_add(self, sock, antenna_str):
        """Add an antenna to the simulated array, in the format accepted by katpoint.

        All calls to this request must be performed before starting any correlators.
        """
        self.subarray.antennas.append(katpoint.Antenna(antenna_str))
        return 'ok',

    @request()
    @return_reply()
    def request_antenna_list(self, sock):
        """Report all the antennas in the simulated array"""
        for antenna in self.subarray.antennas:
            sock.inform(antenna.description)
        return 'ok',

    @request(Str())
    @return_reply()
    def request_capture_start(self, sock, stream):
        """Start the flow of data for a product"""
        try:
            product = self.products[stream]
        except KeyError:
            return 'fail', 'requested product name not found'
        product.capture_start()
        return 'ok',

    @request(Str())
    @return_reply()
    @tornado.gen.coroutine
    def request_capture_stop(self, sock, stream):
        """Stop the flow of data for a product"""
        try:
            product = self.products[stream]
        except KeyError:
            raise tornado.gen.Return(('fail', 'requested product name not found'))
        stop = trollius.async(product.capture_stop())
        yield tornado.platform.asyncio.to_tornado_future(stop)
        raise tornado.gen.Return(('ok',))

    @tornado.gen.coroutine
    def request_halt(self, req, msg):
        """Stop all products and halt the server."""
        for product in self.products.values():
            stop = trollius.async(product.capture_stop())
            yield tornado.platform.asyncio.to_tornado_future(stop)
        yield tornado.gen.maybe_future(super(SimulatorServer, self).request_halt(req, msg))
