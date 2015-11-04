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
    @functools.wraps(wrapped)
    def wrapper(self, sock, name, *args, **kwargs):
        try:
            product = self._products[name]
        except KeyError:
            return 'fail', 'requested product name "{}" not found'.format(name)
        return wrapped(self, sock, product, *args, **kwargs)
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
    """katcp server for the simulator.

    Parameters
    ----------
    context : compute device context
        Compute device context used for device-accelerated simulations
    subarray : :class:`katcbfsim.product.Subarray`, optional
        Preconfigured subarray. If not specified, an unconfigured subarray is created.
    telstate : :class:`katsdptelstate.TelescopeState`, optional
        Telescope state used for the :samp:`?configure-subarray-from-telstate`.
        If not provided, that require will fail.
    *args, **kwargs :
        Passed to base class
    """

    VERSION_INFO = ('katcbfsim-api', 1, 0)
    BUILD_INFO = ('katcbfsim', 0, 1, '')

    def __init__(self, context, subarray=None, telstate=None, *args, **kwargs):
        super(SimulatorServer, self).__init__(*args, **kwargs)
        self._context = context
        self._products = {}
        if subarray is None:
            self._subarray = Subarray()
        else:
            self._subarray = subarray
        self._telstate = telstate
        # Dictionary of dictionaries, indexed by product then sensor name
        self._product_sensors = {}

    def setup_sensors(self):
        self.add_sensor(Sensor.discrete('device-status',
            'Dummy device status sensor. The simulator is always ok.',
            '', ['ok', 'degraded', 'fail'], initial_status=Sensor.NOMINAL))

    def add_fx_product(self, name, *args, **kwargs):
        assert name not in self._products
        product = FXProduct(self._context, self._subarray, name, *args, **kwargs)
        self._products[name] = product
        self._product_sensors[product] = {
            'bandwidth': Sensor.integer('{}.bandwidth'.format(name),
                'The bandwidth currently configured for the data product',
                'Hz', default=product.bandwidth, initial_status=Sensor.NOMINAL),
            'channels': Sensor.integer('{}.channels'.format(name),
                'The number of channels of the channelised data product',
                '', default=product.n_channels, initial_status=Sensor.NOMINAL),
            'centerfrequency': Sensor.integer('{}.centerfrequency'.format(name),
                'The center frequency for the data product', 'Hz')
        }
        for sensor in self._product_sensors[product].itervalues():
            self.add_sensor(sensor)
        return product

    @request(Str(), Int(), Int(), Int())
    @return_reply()
    def request_product_create_correlator(self, sock, name, adc_rate, bandwidth, n_channels):
        """Create a new simulated correlator product"""
        if name in self._products:
            return 'fail', 'product {} already exists'.format(name)
        self.add_fx_product(name, adc_rate, bandwidth, n_channels)
        return 'ok',

    def set_destination(self, product, endpoints):
        product.destination_factory = FXStreamSpeadFactory(endpoints)

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
        self.set_destination(product, endpoints)
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
        if req_name != '' and req_name not in self._products:
            return 'fail', 'requested product name not found'
        for name, product in self._products.items():
            if req_name == '' or req_name == name:
                try:
                    endpoints = product.destination.endpoints
                except AttributeError:
                    endpoints = [endpoint.Endpoint('0.0.0.0', 0)]
                # TODO: Add a formatter to katsdptelstate.endpoint that
                # reconstructs the a.b.c.d+N:port format.
                sock.inform(','.join([str(x) for x in endpoints]))
        return 'ok',

    def set_sync_time(self, timestamp):
        self._subarray.sync_time = katpoint.Timestamp(timestamp)

    @request(Int())
    @return_reply()
    @_product_exceptions
    def request_sync_time(self, sock, timestamp):
        """Set the sync time, as seconds since the UNIX epoch. This will also
        be the timestamp associated with the first data dump."""
        self.set_sync_time(timestamp)
        return 'ok',

    def set_target(self, target):
        self._subarray.target = target

    @request(Str())
    @return_reply()
    def request_target(self, sock, target):
        """Set the simulated target, in the format used by katpoint. This also
        sets the phase center. This can be set while capture is running."""
        self.set_target(katpoint.Target(target))
        return 'ok',

    def set_accumulation_length(self, product, period):
        product.accumulation_length = period

    @request(Str(), Float())
    @return_reply(Float())
    @_product_exceptions
    @_product_request
    def request_accumulation_length(self, sock, product, period):
        """Set the accumulation interval for a product.

        Note: this differs from the CAM-CBF ICD, in which this is subarray-wide."""
        self.set_accumulation_length(product, period)
        # accumulation_length is a property, and the setter rounds the value.
        # We are thus returning the rounded value.
        return 'ok', product.accumulation_length

    def set_center_frequency(self, product, frequency):
        product.center_frequency = frequency
        # TODO: get the simulated timestamp from the product
        self._product_sensors[product]['centerfrequency'].set_value(frequency)

    @request(Str(), Int())
    @return_reply()
    @_product_exceptions
    @_product_request
    def request_frequency_select(self, sock, product, frequency):
        """Set the center frequency for the band. Unlike the real CBF, an
        arbitrary frequency may be selected, and it will not be rounded.
        """
        self.set_center_frequency(product, frequency)
        return 'ok',

    def add_antenna(self, antenna):
        self._subarray.add_antenna(antenna)

    @request(Str())
    @return_reply()
    @_product_exceptions
    def request_antenna_add(self, sock, antenna_str):
        """Add an antenna to the simulated array, in the format accepted by katpoint."""
        self.add_antenna(katpoint.Antenna(antenna_str))
        return 'ok',

    @request()
    @return_reply()
    def request_antenna_list(self, sock):
        """Report all the antennas in the simulated array"""
        for antenna in self._subarray.antennas:
            sock.inform(antenna.description)
        return 'ok',

    def add_source(self, source):
        self._subarray.add_source(source)

    @request(Str())
    @return_reply()
    @_product_exceptions
    def request_source_add(self, sock, source_str):
        """Add a source to the sky model, in the format accepted by katpoint."""
        self.add_source(katpoint.Target(source_str))
        return 'ok',

    @request()
    @return_reply()
    def request_source_list(self, sock):
        """List the sources in the sky model."""
        for source in self._subarray.sources:
            sock.inform(source.description)
        return 'ok',

    def configure_subarray_from_telstate(self, telstate=None):
        """Configure subarray from sensors/attributes in telescope state."""
        if telstate is None:
            telstate = self._telstate
        antenna_names = telstate['config']['antenna_mask'].split(',')
        for name in antenna_names:
            attribute_name = name + '_observer'
            try:
                antenna = telstate[attribute_name]
            except KeyError:
                logger.warn('Antenna description for %s not found, skipping', name)
            else:
                # It might be either a string or an object at this point.
                # The constructor handles either case.
                self.add_antenna(katpoint.Antenna(antenna))

    @request()
    @return_reply()
    @_product_exceptions
    def request_configure_subarray_from_telstate(self, sock):
        """Configure the subarray using sensors and attributes in the telescope
        state. This uses dynamic values, rather than the :samp:`config`
        dictionary used by the command-line parser.
        """
        if self._telstate is None:
            return 'fail', 'no telescope state was specified with --telstate'
        self.configure_subarray_from_telstate()
        return 'ok',

    def configure_product_from_telstate(self, product, telstate=None):
        if telstate is None:
            telstate = self._telstate
        if isinstance(product, FXProduct):
            # Set accumulation length
            try:
                accumulation_length = 1.0 / telstate['sub_dump_rate']
            except KeyError:
                logger.warn('sub_dump_rate not found for %s, accumulation-length not set', product.name)
            else:
                self.set_accumulation_length(product, accumulation_length)

    @request(Str())
    @return_reply()
    @_product_request
    @_product_exceptions
    def request_configure_product_from_telstate(self, sock, product):
        """Configure product from sensors/attributes in telescope state.
        Currently only accumulation length is set; in particular it will
        **not** set a center frequency."""
        if self._telstate is None:
            return 'fail', 'no telescope state was specified with --telstate'
        self.configure_product_from_telstate(product)
        return 'ok',

    def capture_start(self, product):
        product.capture_start()

    @request(Str())
    @return_reply()
    @_product_exceptions
    @_product_request
    def request_capture_start(self, sock, product):
        """Start the flow of data for a product"""
        self.capture_start(product)
        return 'ok',

    @request(Str())
    @return_reply()
    @tornado.gen.coroutine
    def request_capture_stop(self, sock, name):
        """Stop the flow of data for a product"""
        try:
            product = self._products[name]
        except KeyError:
            raise tornado.gen.Return(('fail', 'requested product name not found'))
        stop = trollius.async(product.capture_stop())
        yield tornado.platform.asyncio.to_tornado_future(stop)
        raise tornado.gen.Return(('ok',))

    @tornado.gen.coroutine
    def request_halt(self, req, msg):
        for product in self._products.values():
            stop = trollius.async(product.capture_stop())
            yield tornado.platform.asyncio.to_tornado_future(stop)
        yield tornado.gen.maybe_future(super(SimulatorServer, self).request_halt(req, msg))

    request_halt.__doc__ = katcp.DeviceServer.request_halt.__doc__
