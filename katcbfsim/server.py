from __future__ import print_function, division
import trollius
from trollius import From
import katcp
import katpoint
import tornado
import logging
from katcp.kattypes import Str, Float, Int, Address, request, return_reply
from katsdptelstate import endpoint
from .product import Subarray, FXProduct
from .stream import FXStreamSpead


logger = logging.getLogger(__name__)


class SimulatorServer(katcp.DeviceServer):

    VERSION_INFO = ('katcbfsim-api', 1, 0)
    BUILD_INFO = ('katcbfsim', 0, 1, '')

    def __init__(self, *args, **kwargs):
        super(SimulatorServer, self).__init__(*args, **kwargs)
        self.products = {}
        self.subarray = Subarray()

    def setup_sensors(self):
        pass

    @request(Str(), Int(), Int())
    @return_reply()
    def request_product_create_correlator(self, sock, name, bandwidth, channels):
        """Create a new simulated correlator productt"""
        if name in self.products:
            return 'fail', 'product {} already exists'.format(name)
        self.products[name] = FXProduct(self.subarray, name, bandwidth, channels)
        return 'ok',

    @request(Str(), Str())
    @return_reply()
    def request_capture_destination(self, sock, name, destination):
        """Set the destination endpoints for a product"""
        try:
            product = self.products[name]
        except KeyError:
            return 'fail', 'requested product name not found'
        endpoints = endpoint.endpoint_list_parser(None)(destination)
        for e in endpoints:
            if e.port is None:
                return 'fail', 'no port specified'
        product.destination = FXStreamSpead(endpoints)
        return 'ok',

    @request(Str(default=''))
    @return_reply()
    def request_capture_list(self, sock, req_name):
        """List the destination endpoints for a product, or all products"""
        if req_name != '' and req_name not in self.products:
            return 'fail', 'requested product name not found'
        for name, product in self.products.items():
            if req_name == '' or req_name == name:
                try:
                    endpoints = product.destination.endpoints
                except AttributeError:
                    endpoints = [endpoint.Endpoint('0.0.0.0', 0)]
                # TODO: Add a formatter to katsdptelstate.endpoint that
                # reconstructs the a.b.c.d+N:port format.
                sock.inform(','.join([str(x) for x in endpoints]))
        return 'ok',

    @request(Int())
    @return_reply()
    def request_sync_time(self, sock, timestamp):
        """Set the sync time, as seconds since the UNIX epoch. This will also
        be the timestamp associated with the first data dump."""
        self.subarray.sync_time = timestamp
        return 'ok',

    @request(Str(), Float())
    @return_reply()
    def request_accumulation_length(self, sock, name, period):
        """Set the accumulation interval for a product.

        Note: this differs from the CAM-CBF ICD, in which this is subarray-wide."""
        try:
            product = self.products[name]
        except KeyError:
            return 'fail', 'requested product name not found'
        product.accumulation_length = period
        return 'ok',

    @request(Str(), Float())
    @return_reply()
    def request_frequency_select(self, sock, name, frequency):
        """Set the center frequency for the band. Unlike the real CBF, an
        arbitrary frequency may be selected, and it will not be rounded.
        """
        try:
            product = self.products[name]
        except KeyError:
            return 'fail', 'requested product name not found'
        product.center_frequency = center_frequency
        return 'ok',

    @request(Str())
    @return_reply()
    def request_antenna_add(self, sock, antenna_str):
        """Add an antenna to the simulated array, in the format accepted by katpoint."""
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
    def request_source_add(self, sock, source_str):
        """Add a source to the sky model, in the format accepted by katpoint."""
        self.subarray.sources.append(katpoint.Target(source_str))
        return 'ok',

    @request()
    @return_reply()
    def request_source_list(self, sock):
        """List the sources in the sky model."""
        for source in self.subarray.sources:
            sock.inform(source.description)
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
        for product in self.products.values():
            stop = trollius.async(product.capture_stop())
            yield tornado.platform.asyncio.to_tornado_future(stop)
        yield tornado.gen.maybe_future(super(SimulatorServer, self).request_halt(req, msg))

    request_halt.__doc__ = katcp.DeviceServer.request_halt.__doc__
