"""Coverage tests for katcp interface"""

import functools
import mock
import katcp
import trollius
import tornado.ioloop
import tornado.locks
import tornado.gen
import re
from tornado.gen import Return
from tornado.platform.asyncio import AsyncIOMainLoop
from katsdpsigproc import accel
from katsdpsigproc.test.test_accel import device_test, cuda_test, force_autotune
import katsdptelstate
from katsdptelstate.endpoint import Endpoint
from katcbfsim import server, stream, transport
from nose.tools import *


def test_endpoints_to_str():
    endpoints = [
        Endpoint('hostname', 1234),
        Endpoint('1.2.3.4', 7148),
        Endpoint('1.2.3.3', 7148),
        Endpoint('1.2.3.3', None),
        Endpoint('1.2.3.3', 7149),
        Endpoint('1.2.3.5', 7148),
        Endpoint('192.168.1.255', None),
        Endpoint('192.168.2.0', None),
        Endpoint('::10ff', 7148),
        Endpoint('::20ff', 7148),
        Endpoint('::0000:2100', 7148),
        Endpoint('hostname1', None)
    ]
    s = server.endpoints_to_str(endpoints)
    assert_equal('1.2.3.3,192.168.1.255+1,1.2.3.3+2:7148,1.2.3.3:7149,[::10ff]:7148,[::20ff]+1:7148,hostname:1234,hostname1', s)


@nottest
def async_test(func):
    """Decorator to run a test inside the Tornado event loop"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return tornado.ioloop.IOLoop.current().run_sync(lambda: func(*args, **kwargs))
    return wrapper


# Last created MockTransport, for tests to reach in and check state
_current_transport = None


class MockTransport(object):
    """Transport that throws away its data, for testing purposes."""
    @classmethod
    def factory(cls, endpoints, interface, n_substreams, max_packet_size):
        return transport.EndpointFactory(cls, endpoints, interface, n_substreams, max_packet_size)

    def __init__(self, endpoints, interface, n_substreams, max_packet_size, stream):
        global _current_transport
        self.endpoints = endpoints
        self.interface = interface
        self.stream = stream
        self.dumps = 0
        self.dumps_semaphore = tornado.locks.Semaphore(0)
        self.closed = False
        self.n_substreams = n_substreams
        _current_transport = self

    @trollius.coroutine
    def send_metadata(self):
        pass

    @trollius.coroutine
    def send(self, data, index):
        self.dumps += 1
        self.dumps_semaphore.release()

    @trollius.coroutine
    def close(self):
        assert_false(self.closed)
        self.closed = True


class FXMockTransport(MockTransport):
    pass


class BeamformerMockTransport(MockTransport):
    pass


class TestSimulationServer(object):
    def _patch(self, *args, **kwargs):
        self._patchers.append(mock.patch(*args, **kwargs))
        mock_obj = self._patchers[-1].start()
        return mock_obj

    @device_test
    def setup(self, context, queue):
        self._patchers = []
        self._patch('katcbfsim.transport.FXSpeadTransport', FXMockTransport)
        self._patch('katcbfsim.transport.BeamformerSpeadTransport', BeamformerMockTransport)
        self._telstate = mock.create_autospec(katsdptelstate.TelescopeState, instance=True)
        self._ioloop = AsyncIOMainLoop()
        self._ioloop.install()
        port = 7147
        self._server = server.SimulatorServer(
            context, None, telstate=self._telstate, host='localhost', port=port)
        self._server.set_concurrency_options(thread_safe=False, handler_thread=False)
        self._server.set_ioloop(self._ioloop)
        self._server.start()
        self._client = katcp.AsyncClient('localhost', port, timeout=15)
        self._client.set_ioloop(self._ioloop)
        self._client.start()
        self._ioloop.run_sync(self._client.until_protocol)

    @tornado.gen.coroutine
    def _teardown(self):
        self._client.disconnect()
        self._client.stop()
        self._server.stop()

    def teardown(self):
        self._ioloop.run_sync(self._teardown)
        for patcher in reversed(self._patchers):
            patcher.stop()
        tornado.ioloop.IOLoop.clear_instance()
        _current_transport = None

    @tornado.gen.coroutine
    def make_request(self, name, *args):
        """Issue a request to the server, and check that the result is an ok.

        Parameters
        ----------
        name : str
            Request name
        args : list
            Arguments to the request

        Returns
        -------
        informs : list
            Informs returned with the reply
        """
        reply, informs = yield self._client.future_request(katcp.Message.request(name, *args))
        assert_true(reply.reply_ok(), str(reply))
        raise Return(informs)

    @tornado.gen.coroutine
    def assert_request_fails(self, msg_re, name, *args):
        """Assert that a request fails, and test the error message against
        a regular expression."""
        reply, informs = yield self._client.future_request(katcp.Message.request(name, *args))
        assert_equal(2, len(reply.arguments))
        assert_equal('fail', reply.arguments[0])
        assert_regexp_matches(reply.arguments[1], msg_re)

    @async_test
    @tornado.gen.coroutine
    def test_sync_time(self):
        yield self.make_request('sync-time', 1446544133)

    @tornado.gen.coroutine
    def _configure_subarray(self, clock_ratio=None):
        """Sets up some values on a subarray. It is not part of setup because
        not all tests necessarily require it, and because setup isn't run
        inside the event loop.
        """
        if clock_ratio is None:
            clock_ratio = 0.5    # Run faster than real-time
        yield self.make_request('clock-ratio', clock_ratio)
        yield self.make_request('antenna-add', 'm062, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -1440.69968823 -2269.26759132 6.0, -0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 -0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22')
        yield self.make_request('antenna-add', 'm063, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -3419.58251626 -1606.01510973 2.0, -0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 -0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22')
        # One source with a flux model, one without
        yield self.make_request('source-add', 'test1, radec, 3:30:00.00, -35:00:00.0, (500.0 2000.0 1.0)')
        yield self.make_request('source-add', 'test2, radec, 3:33:00.00, -35:01:00.0')
        yield self.make_request('target', 'target, radec, 3:15:00.00, -36:00:00.0')

    def _check_common_telstate(self):
        self._telstate.add.assert_any_call('cbf_adc_sample_rate', 1712000000.0, immutable=True)
        self._telstate.add.assert_any_call('cbf_bandwidth', 856000000.0, immutable=True)
        self._telstate.add.assert_any_call('cbf_n_inputs', 4, immutable=True)
        self._telstate.add.assert_any_call('cbf_scale_factor_timestamp', 1712000000, immutable=True)
        self._telstate.add.assert_any_call('cbf_sync_time', mock.ANY, immutable=True)
        self._telstate.add.assert_any_call('cbf_ticks_between_spectra', 8192, immutable=True)
        self._telstate.add.assert_any_call('cbf_n_chans', 4096, immutable=True)
        # Baseband frequency
        self._telstate.add.assert_any_call('cbf_center_freq', 428000000.0, immutable=True)
        for i in range(4):   # inputs
            self._telstate.add.assert_any_call('cbf_input{}_fft0_shift'.format(i), mock.ANY, immutable=False)
            self._telstate.add.assert_any_call('cbf_input{}_delay'.format(i), (0, 0, 0, 0, 0), immutable=False)
            self._telstate.add.assert_any_call('cbf_input{}_delay_ok'.format(i), True, immutable=False)
            self._telstate.add.assert_any_call('cbf_input{}_eq'.format(i), [200 + 0j], immutable=False)

    @tornado.gen.coroutine
    def _test_fx_capture(self, clock_ratio=None, min_dumps=None):
        if min_dumps is None:
            min_dumps = 5    # Should be enough to demonstrate overlapping I/O
        yield self._configure_subarray()
        # Number of channels is kept small to avoid using too much memory
        yield self.make_request('stream-create-correlator', 'cross', 1712000000, 1284000000, 856000000, 4096)
        yield self.make_request('capture-destination', 'cross', 'localhost:7148')
        yield self.make_request('accumulation-length', 'cross', 0.5)
        yield self.make_request('frequency-select', 'cross', 1284000000)
        yield self.make_request('capture-start', 'cross')
        # Wait until we've received the minimum number of dumps
        for i in range(min_dumps):
            yield _current_transport.dumps_semaphore.acquire()
        yield self.make_request('capture-stop', 'cross')
        assert_is_not_none(_current_transport)
        assert_greater_equal(_current_transport.dumps, min_dumps)
        assert_true(_current_transport.closed)
        bls_ordering = []
        for a in ['m062', 'm063']:
            for b in ['m062', 'm063']:
                if a > b:
                    continue
                for ap in [a + 'v', a + 'h']:
                    for bp in [b + 'v', b + 'h']:
                        bls_ordering.append((ap, bp))
        self._check_common_telstate()
        n_accs = 408 * 256   # Gives nearest to 0.5s
        self._telstate.add.assert_any_call('cbf_bls_ordering', bls_ordering, immutable=True)
        # 0.5 rounded to nearest acceptable interval
        self._telstate.add.assert_any_call('cbf_int_time', n_accs * 2 * 4096 / 1712000000.0, immutable=True)
        self._telstate.add.assert_any_call('cbf_n_accs', n_accs, immutable=True)
        self._telstate.add.assert_any_call('cbf_n_chans_per_substream', 256, immutable=True)
        # Use assert_called_with here rather than assert_any_call, to ensure
        # that it is the *last* call.
        self._telstate.add.assert_called_with('sdp_cam2telstate_status', 'ready', immutable=False)

    @cuda_test
    @async_test
    @tornado.gen.coroutine
    def test_fx_capture(self):
        """Create an FX stream, start it, and stop it again"""
        yield self._test_fx_capture()

    @cuda_test
    @async_test
    @tornado.gen.coroutine
    def test_fx_capture_fast(self):
        """Run an FX capture as fast as possible"""
        yield self._test_fx_capture(clock_ratio=0.0, min_dumps=10)

    @tornado.gen.coroutine
    def _test_beamformer_capture(self, clock_ratio=None, min_dumps=None):
        if min_dumps is None:
            min_dumps = 5    # Should be enough to demonstrate overlapping I/O
        yield self._configure_subarray()
        name = 'i0.tied-array-channelised-voltage.0x'
        uname = 'i0_tied_array_channelised_voltage_0x'
        yield self.make_request('stream-create-beamformer', name, 1712000000, 1284000000, 856000000, 4096, 256, 8)
        yield self.make_request('capture-destination', name, 'localhost:7149', 4)
        yield self.make_request('capture-start', name)
        for i in range(min_dumps):
            yield _current_transport.dumps_semaphore.acquire()
        yield self.make_request('capture-stop', name)
        assert_is_not_none(_current_transport)
        assert_greater_equal(_current_transport.dumps, min_dumps)
        assert_true(_current_transport.closed)
        self._check_common_telstate()
        print(self._telstate.add.mock_calls)
        self._telstate.add.assert_any_call('cbf_{}_n_chans'.format(uname), 4096, immutable=False)
        self._telstate.add.assert_any_call(
            'cbf_{}_n_chans_per_substream'.format(uname), 1024, immutable=True)
        self._telstate.add.assert_any_call('cbf_{}_spectra_per_heap'.format(uname), 256, immutable=True)
        for i in range(4):   # input
            self._telstate.add.assert_any_call(
                'cbf_{}_input{}_weight'.format(uname, i), 1.0, immutable=False)
        # Use assert_called_with here rather than assert_any_call, to ensure
        # that it is the *last* call.
        self._telstate.add.assert_called_with('sdp_cam2telstate_status', 'ready', immutable=False)

    @async_test
    @tornado.gen.coroutine
    def test_beamformer_capture(self):
        """Create a beamformer target, start it, and stop it again"""
        yield self._test_beamformer_capture()

    @async_test
    @tornado.gen.coroutine
    def test_beamformer_capture_fast(self):
        """Run the beamformer simulation as fast as possible"""
        yield self._test_beamformer_capture(clock_ratio=0.0, min_dumps=20)

    @async_test
    @tornado.gen.coroutine
    def test_unknown_stream_name(self):
        """An appropriate error is returned when using an unknown stream name."""
        yield self.assert_request_fails('^requested stream name "unknown" not found$', 'capture-destination', 'unknown', '127.0.0.1:7147')
        yield self.assert_request_fails('^requested stream name "unknown" not found$', 'capture-destination-file', 'unknown', '/dev/null')
        yield self.assert_request_fails('^requested stream name "unknown" not found$', 'capture-start', 'unknown')
        yield self.assert_request_fails('^requested stream name "unknown" not found$', 'capture-stop', 'unknown')
        yield self.assert_request_fails('^requested stream name "unknown" not found$', 'accumulation-length', 'unknown', 0.5)
        yield self.assert_request_fails('^requested stream name "unknown" not found$', 'frequency-select', 'unknown', 1000000000)

    @async_test
    @tornado.gen.coroutine
    def test_change_while_capturing(self):
        """An appropriate error is returned when trying to change values while
        a capture is in progress."""
        yield self._configure_subarray()
        # Use lower bandwidth to reduce test time
        yield self.make_request('stream-create-beamformer', 'beam1', 1712000000, 1284000000, 856000000 / 4, 32768, 256, 8)
        yield self.make_request('capture-destination', 'beam1', 'localhost:7149')
        yield self.make_request('capture-start', 'beam1')
        yield self.assert_request_fails('^cannot add antennas while capture is in progress$', 'antenna-add', 'm062, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -1440.69968823 -2269.26759132 6.0, -0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 -0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22')
        yield self.assert_request_fails('^cannot add source while capture is in progress$', 'source-add', 'test3, radec, 3:30:00.00, -35:00:00.0, (500.0 2000.0 1.0)')
        yield self.assert_request_fails('^cannot set sync time while capture is in progress$', 'sync-time', 1446544133)
        yield self.assert_request_fails('^cannot set clock ratio while capture is in progress$', 'clock-ratio', 1.0)
        yield self.assert_request_fails('^cannot set center_frequency while capture is in progress', 'frequency-select', 'beam1', 10000000000)
        yield self.make_request('capture-stop', 'beam1')

    @tornado.gen.coroutine
    def _get_antenna_descriptions(self):
        informs = yield self.make_request('antenna-list')
        raise Return([inform.arguments[0] for inform in informs])

    @async_test
    @tornado.gen.coroutine
    def test_replace_antenna(self):
        """Adding an antenna with a duplicate name replaces the existing one."""
        yield self._configure_subarray()
        antennas = yield self._get_antenna_descriptions()
        assert_equal([
            'm062, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -1440.69968823 -2269.26759132 6.0, -0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 -0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22',
            'm063, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -3419.58251626 -1606.01510973 2.0, -0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 -0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22'],
            antennas)
        # Change the latitude, to check that the old value is replaced
        yield self.make_request('antenna-add', 'm062, -30:00:00.0, 21:26:38.0, 1035.0, 13.5, -1440.69968823 -2269.26759132 6.0, -0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 -0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22')
        antennas = yield self._get_antenna_descriptions()
        assert_equal([
            'm062, -30:00:00.0, 21:26:38.0, 1035.0, 13.5, -1440.69968823 -2269.26759132 6.0, -0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 -0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22',
            'm063, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -3419.58251626 -1606.01510973 2.0, -0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 -0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22'],
            antennas)
