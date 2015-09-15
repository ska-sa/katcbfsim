from __future__ import print_function, division
import trollius
from trollius import From
import katcp
import katpoint
import tornado
import logging
import functools
from katcp import Sensor
from katcp.kattypes import Str, Float, Int, Address, request, return_reply
from katsdptelstate import endpoint
from . import product
from .product import Subarray, FXProduct
from .stream import FXStreamSpeadFactory, FXStreamFileFactory


logger = logging.getLogger(__name__)


def _product_request(wrapped):
    """Decorator for per-product commands. It looks up the product and passes
    it to the wrapped function, or returns a failure message if it does not
    exist.
    """
    def wrapper(self, sock, name, *args, **kwargs):
        try:
            product = self.products[name]
        except KeyError:
            return 'fail', 'requested product name not found'
        return wrapped(self, sock, product, *args, **kwargs)
    functools.update_wrapper(wrapper, wrapped)
    return wrapper

def _product_exceptions(wrapped):
    """Decorator used on requests that and turns exceptions defined in
    :py:mod:`product` into katcp "fail" messages.
    """
    def wrapper(*args, **kwargs):
        try:
            return wrapped(*args, **kwargs)
        except (product.CaptureInProgressError, product.IncompleteConfigError) as e:
            return 'fail', str(e)
    functools.update_wrapper(wrapper, wrapped)
    return wrapper


class SimulatorServer(katcp.DeviceServer):

    VERSION_INFO = ('katcbfsim-api', 1, 0)
    BUILD_INFO = ('katcbfsim', 0, 1, '')

    def __init__(self, context, *args, **kwargs):
        super(SimulatorServer, self).__init__(*args, **kwargs)
        self.context = context
        self.products = {}
        self.subarray = Subarray()
        # Dictionary of dictionaries, indexed by product then sensor name
        self.product_sensors = {}

    def setup_sensors(self):
        self.add_sensor(Sensor.discrete('device-status',
            'Dummy device status sensor. The simulator is always ok.',
            '', ['ok', 'degraded', 'fail'], initial_status=Sensor.NOMINAL))

    @request(Str(), Int(), Int())
    @return_reply()
    def request_product_create_correlator(self, sock, name, bandwidth, n_channels):
        """Create a new simulated correlator product"""
        if name in self.products:
            return 'fail', 'product {} already exists'.format(name)
        product = FXProduct(self.context, self.subarray, name, bandwidth, n_channels)
        self.products[name] = product
        self.product_sensors[product] = {
            'bandwidth': Sensor.integer('{}.bandwidth'.format(name),
                'The bandwidth currently configured for the data product',
                'Hz', default=bandwidth, initial_status=Sensor.NOMINAL),
            'channels': Sensor.integer('{}.channels'.format(name),
                'The number of channels of the channelised data product',
                '', default=n_channels, initial_status=Sensor.NOMINAL),
            'centerfrequency': Sensor.integer('{}.centerfrequency'.format(name),
                'The center frequency for the data product', 'Hz')
        }
        for sensor in self.product_sensors[product].itervalues():
            self.add_sensor(sensor)
        return 'ok',

    @request(Str(), Str())
    @return_reply()
    @_product_exceptions
    @_product_request
    def request_capture_destination(self, sock, product, destination):
        """Set the destination endpoints for a product"""
        endpoints = endpoint.endpoint_list_parser(None)(destination)
        for e in endpoints:
            if e.port is None:
                return 'fail', 'no port specified'
        product.destination_factory = FXStreamSpeadFactory(endpoints)
        return 'ok',

    @request(Str(), Str())
    @return_reply()
    @_product_exceptions
    @_product_request
    def request_capture_destination_file(self, sock, product, destination):
        """Set the destination to an HDF5 file"""
        product.destination_factory = FXStreamFileFactory(destination)
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
    @_product_exceptions
    def request_sync_time(self, sock, timestamp):
        """Set the sync time, as seconds since the UNIX epoch. This will also
        be the timestamp associated with the first data dump."""
        self.subarray.sync_time = katpoint.Timestamp(timestamp)
        return 'ok',

    @request(Str())
    @return_reply()
    def request_target(self, sock, target):
        """Set the simulated target, in the format used by katpoint. This also
        sets the phase center. This can be set while capture is running."""
        self.subarray.target = katpoint.Target(target)
        return 'ok',

    @request(Str(), Float())
    @return_reply(Float())
    @_product_exceptions
    @_product_request
    def request_accumulation_length(self, sock, product, period):
        """Set the accumulation interval for a product.

        Note: this differs from the CAM-CBF ICD, in which this is subarray-wide."""
        product.accumulation_length = period
        # accumulation_length is a property, and the setter rounds the value.
        # We are thus returning the rounded value.
        return 'ok', product.accumulation_length

    @request(Str(), Int())
    @return_reply()
    @_product_exceptions
    @_product_request
    def request_frequency_select(self, sock, product, frequency):
        """Set the center frequency for the band. Unlike the real CBF, an
        arbitrary frequency may be selected, and it will not be rounded.
        """
        product.center_frequency = frequency
        # TODO: get the simulated timestamp from the product
        self.product_sensors[product]['centerfrequency'].set_value(frequency)
        return 'ok',

    @request(Str())
    @return_reply()
    @_product_exceptions
    def request_antenna_add(self, sock, antenna_str):
        """Add an antenna to the simulated array, in the format accepted by katpoint."""
        self.subarray.add_antenna(katpoint.Antenna(antenna_str))
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
    @_product_exceptions
    def request_source_add(self, sock, source_str):
        """Add a source to the sky model, in the format accepted by katpoint."""
        self.subarray.add_source(katpoint.Target(source_str))
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
    @_product_exceptions
    @_product_request
    def request_capture_start(self, sock, product):
        """Start the flow of data for a product"""
        product.capture_start()
        return 'ok',

    @request(Str())
    @return_reply()
    @tornado.gen.coroutine
    def request_capture_stop(self, sock, name):
        """Stop the flow of data for a product"""
        try:
            product = self.products[name]
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
