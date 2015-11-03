"""Coverage tests for katcp interface"""

import functools
import mock
import katcp
import tornado.ioloop
from tornado.platform.asyncio import AsyncIOMainLoop
import tornado.gen
from tornado.gen import Return
from katsdpsigproc import accel
from katsdpsigproc.test.test_accel import device_test, force_autotune
from katcbfsim import server, product, stream
from nose.tools import *


def async_test(func):
    """Decorator to run a test inside the Tornado event loop"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return tornado.ioloop.IOLoop.current().run_sync(lambda: func(*args, **kwargs))
    return wrapper


# Last created MockStream, for tests to reach in and check state
_current_stream = None


class MockStream(object):
    """Stream that throws away its data, for testing purposes."""
    @classmethod
    def factory(cls, endpoints):
        return functools.partial(cls, endpoints)

    def __init__(self, endpoints, product):
        global _current_stream
        self.endpoints = endpoints
        self.product = product
        self.dumps = 0
        self.closed = False
        _current_stream = self

    def send_metadata(self):
        pass

    def close(self):
        assert_false(self.closed)
        self.closed = True


class FXStreamMock(MockStream):
    def send(self, vis, dump_index):
        self.dumps += 1


class BeamformerStreamMock(MockStream):
    def send(self, data, index):
        self.dumps += 1


class TestKatcp(object):
    @device_test
    def setup(self, context, queue):
        self._patchers = [
            mock.patch('katcbfsim.stream.FXStreamSpead', FXStreamMock),
            mock.patch('katcbfsim.stream.BeamformerStreamSpead', BeamformerStreamMock)
        ]
        for patcher in self._patchers:
            patcher.start()
        self._ioloop = AsyncIOMainLoop()
        self._ioloop.install()
        port = 7147
        self._server = server.SimulatorServer(context, None, host='localhost', port=port)
        self._server.set_concurrency_options(thread_safe=False, handler_thread=False)
        self._server.set_ioloop(self._ioloop)
        self._server.start()
        self._client = katcp.AsyncClient('localhost', port)
        self._client.set_ioloop(self._ioloop)
        self._client.start()

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
        _current_stream = None

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
        reply, informs = yield self._client.future_request(katcp.Message.request(name, *args), timeout=15)
        assert_true(reply.reply_ok(), str(reply))
        raise Return(informs)

    @async_test
    @tornado.gen.coroutine
    def test_sync_time(self):
        yield self._client.until_protocol()
        yield self.make_request('sync-time', 1446544133)

    @tornado.gen.coroutine
    def _configure_subarray(self):
        """Code used by several tests. It sets up some values on a subarray. It
        is not part of setup because not all tests necessarily require it, and
        because setup isn't run inside the event loop.
        """
        yield self.make_request('clock-ratio', 0.5)  # Run quickly
        yield self.make_request('antenna-add', 'm062, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -1440.69968823 -2269.26759132 6.0, -0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 -0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22')
        yield self.make_request('antenna-add', 'm063, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -3419.58251626 -1606.01510973 2.0, -0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 -0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22')
        yield self.make_request('source-add', 'test1, radec, 3:30:00.00, -35:00:00.0, (500.0 2000.0 1.0)')
        yield self.make_request('source-add', 'test2, radec, 3:33:00.00, -35:01:00.0, (500.0 2000.0 0.7)')
        yield self.make_request('target', 'target, radec, 3:15:00.00, -36:00:00.0')

    @async_test
    @tornado.gen.coroutine
    def test_fx_capture(self):
        """Create an FX product, start it, and stop it again"""
        yield self._client.until_protocol()
        yield self._configure_subarray()
        # Number of channels is kept small to avoid using too much memory
        yield self.make_request('product-create-correlator', 'cross', 1712000000, 856000000, 512)
        yield self.make_request('capture-destination', 'cross', 'localhost:7148')
        yield self.make_request('accumulation-length', 'cross', 1.0)
        yield self.make_request('frequency-select', 'cross', 1284000000)
        yield self.make_request('capture-start', 'cross')
        yield tornado.gen.sleep(2)
        yield self.make_request('capture-stop', 'cross')
        assert_is_not_none(_current_stream)
        assert_greater(_current_stream.dumps, 0)
        assert_true(_current_stream.closed)

    @async_test
    @tornado.gen.coroutine
    def test_beamformer_capture(self):
        """Create a beamformer target, start it, and stop it again"""
        yield self._client.until_protocol()
        yield self._configure_subarray()
        # Use lower bandwidth to reduce test time
        yield self.make_request('product-create-beamformer', 'beam1', 1712000000, 856000000 / 4, 32768, 256, 8)
        yield self.make_request('capture-destination', 'beam1', 'localhost:7149')
        yield self.make_request('frequency-select', 'beam1', 1284000000)
        yield self.make_request('capture-start', 'beam1')
        yield tornado.gen.sleep(2)
        yield self.make_request('capture-stop', 'beam1')
        assert_is_not_none(_current_stream)
        assert_greater(_current_stream.dumps, 0)
        assert_true(_current_stream.closed)
