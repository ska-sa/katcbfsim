"""Coverage tests for katcp interface"""

from unittest import mock
import asyncio

from nose.tools import (assert_equal, assert_is_not_none, assert_greater_equal,
                        assert_true, assert_false, assert_regexp_matches)
import asynctest

import aiokatcp

from katsdpsigproc.test.test_accel import device_test, cuda_test

import katsdptelstate

from katcbfsim import server, transport


M062_DESCRIPTION = 'm062, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -1440.69968823 -2269.26759132 6.0, -0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 -0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22'  # noqa: E501
M063_DESCRIPTION = 'm063, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -3419.58251626 -1606.01510973 2.0, -0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 -0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22'  # noqa: E501
# Modified version of m062, to test antenna replacement
M062_ALT_DESCRIPTION = 'm062, -30:00:00.0, 21:26:38.0, 1035.0, 13.5, -1440.69968823 -2269.26759132 6.0, -0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 -0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22'  # noqa: E501


# Last created MockTransport, for tests to reach in and check state
_current_transport = None


class MockTransport(object):
    """Transport that throws away its data, for testing purposes."""
    @classmethod
    def factory(cls, endpoints, interface, ibv, max_packet_size):
        return transport.EndpointFactory(cls, endpoints, interface, ibv,
                                         max_packet_size)

    def __init__(self, endpoints, interface, ibv, max_packet_size, stream):
        global _current_transport
        self.endpoints = endpoints
        self.interface = interface
        self.ibv = ibv
        self.stream = stream
        self.dumps = 0
        self.dumps_semaphore = asyncio.Semaphore(0)
        self.closed = False
        _current_transport = self

    async def send_metadata(self):
        pass

    async def send(self, data, index):
        self.dumps += 1
        self.dumps_semaphore.release()

    async def close(self):
        assert_false(self.closed)
        self.closed = True


class FXMockTransport(MockTransport):
    pass


class BeamformerMockTransport(MockTransport):
    pass


class TestSimulationServer(asynctest.TestCase):
    def _patch(self, *args, **kwargs):
        patcher = mock.patch(*args, **kwargs)
        mock_obj = patcher.start()
        self.addCleanup(patcher.stop)
        return mock_obj

    @device_test
    def _sync_setup(self, context, queue):
        self._patchers = []
        self._patch('katcbfsim.transport.FXSpeadTransport', FXMockTransport)
        self._patch('katcbfsim.transport.BeamformerSpeadTransport', BeamformerMockTransport)
        self._telstate = katsdptelstate.TelescopeState()
        self._telstate.clear()
        # telstate.get is rigged to return certain known values.
        # Note: this is fragile, and should maybe be replaced by the fakeredis
        # telstate in future.
        self._telstate['i0_antenna_channelised_voltage_instrument_dev_name'] = 'i0'
        for stream in ['i0_baseline_correlation_products',
                       'i0_tied_array_channelised_voltage_0x',
                       'i0_tied_array_channelised_voltage_0y']:
            self._telstate.view(stream)['src_streams'] = ['i0_antenna_channelised_voltage']

        port = 7147
        self._server = server.SimulatorServer(
            context, None, telstate=self._telstate, host='127.0.0.1', port=port)
        return port

    async def setUp(self):
        port = self._sync_setup()
        await self._server.start()
        self.addCleanup(self._server.stop)
        self._reader, self._writer = \
            await asyncio.open_connection('127.0.0.1', port)
        self._mid = 1

    async def tearDown(self):
        global _current_transport
        self._writer.close()
        _current_transport = None

    async def _make_request(self, name, *args):
        """Issue a request to the server, and return the reply message.

        This is a quick-n-dirty implementation until aiokatcp adds client
        support.

        Parameters
        ----------
        name : str
            Request name
        args : list
            Arguments to the request

        Returns
        -------
        reply : Message
            Reply message
        informs : list
            Informs returned with the reply
        """
        mid = self._mid
        self._mid += 1
        request = aiokatcp.Message(aiokatcp.Message.Type.REQUEST, name, *args, mid=mid)
        self._writer.write(bytes(request))
        informs = []
        while True:
            line = await self._reader.readline()
            msg = aiokatcp.Message.parse(line)
            if msg.mid == mid:
                if msg.mtype == aiokatcp.Message.Type.REPLY:
                    return msg, informs
                elif msg.mtype == aiokatcp.Message.Type.INFORM:
                    informs.append(msg)

    async def make_request(self, name, *args):
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
        reply, informs = await self._make_request(name, *args)
        assert_true(reply.reply_ok(), bytes(reply))
        return informs

    async def assert_request_fails(self, msg_re, name, *args):
        """Assert that a request fails, and test the error message against
        a regular expression."""
        reply, informs = await self._make_request(name, *args)
        assert_equal(2, len(reply.arguments))
        assert_equal(b'fail', reply.arguments[0])
        assert_regexp_matches(reply.arguments[1].decode('utf-8'), msg_re)

    async def test_sync_time(self):
        await self.make_request('sync-time', 1446544133)

    async def _configure_subarray(self, clock_ratio=None):
        """Sets up some values on a subarray. It is not part of setup because
        not all tests necessarily require it, and because setup isn't run
        inside the event loop.
        """
        if clock_ratio is None:
            clock_ratio = 0.5    # Run faster than real-time
        await self.make_request('clock-ratio', clock_ratio)
        await self.make_request('antenna-add', M062_DESCRIPTION)
        await self.make_request('antenna-add', M063_DESCRIPTION)
        # One source with a flux model, one without
        await self.make_request('source-add',
                                'test1, radec, 3:30:00.00, -35:00:00.0, (500.0 2000.0 1.0)')
        await self.make_request('source-add',
                                'test2, radec, 3:33:00.00, -35:01:00.0')
        await self.make_request('target', 'target, radec, 3:15:00.00, -36:00:00.0')

    def _check_attribute(self, telstate, key, value):
        assert_equal(telstate.get(key), value)
        assert_equal(telstate.key_type(key), katsdptelstate.KeyType.IMMUTABLE)

    def _check_sensor(self, telstate, key, value):
        assert_equal(telstate.get(key), value)
        assert_equal(telstate.key_type(key), katsdptelstate.KeyType.MUTABLE)

    def _check_common_telstate(self):
        instrument_view = self._telstate.view('i0', exclusive=True)
        self._check_attribute(instrument_view, 'adc_sample_rate', 1712000000.0)
        self._check_attribute(instrument_view, 'n_inputs', 4)
        self._check_attribute(instrument_view, 'scale_factor_timestamp', 1712000000)
        self._check_attribute(instrument_view, 'sync_time', mock.ANY)
        acv_view = self._telstate.view('i0_antenna_channelised_voltage')
        self._check_attribute(acv_view, 'ticks_between_spectra', 8192)
        self._check_attribute(acv_view, 'n_chans', 4096)
        self._check_attribute(acv_view, 'center_freq', 1284000000.0)
        for i in range(4):   # inputs
            input_view = self._telstate.view('i0_antenna_channelised_voltage_input{}'.format(i))
            self._check_sensor(input_view, 'fft0_shift', mock.ANY)
            self._check_sensor(input_view, 'delay', (0, 0, 0, 0, 0))
            self._check_sensor(input_view, 'delay_ok', True)
            self._check_sensor(input_view, 'eq', [200 + 0j])

    async def _test_fx_capture(self, clock_ratio=None, min_dumps=None):
        if min_dumps is None:
            min_dumps = 5    # Should be enough to demonstrate overlapping I/O
        await self._configure_subarray()
        # Number of channels is kept small to avoid using too much memory
        await self.make_request('stream-create-correlator', 'i0.baseline-correlation-products',
                                1712000000, 1284000000, 856000000, 4096)
        await self.make_request('capture-destination', 'i0.baseline-correlation-products',
                                '127.0.0.1:7148')
        await self.make_request('accumulation-length', 'i0.baseline-correlation-products', 0.5)
        await self.make_request('frequency-select', 'i0.baseline-correlation-products', 1284000000)
        await self.make_request('capture-start', 'i0.baseline-correlation-products')
        # Wait until we've received the minimum number of dumps
        for i in range(min_dumps):
            await _current_transport.dumps_semaphore.acquire()
        await self.make_request('capture-stop', 'i0.baseline-correlation-products')
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
        view = self._telstate.view('i0_baseline_correlation_products')
        self._check_attribute(view, 'bandwidth', 856000000.0)
        self._check_attribute(view, 'bls_ordering', bls_ordering)
        # 0.5 rounded to nearest acceptable interval
        self._check_attribute(view, 'int_time', n_accs * 2 * 4096 / 1712000000.0)
        self._check_attribute(view, 'n_accs', n_accs)
        self._check_attribute(view, 'n_chans_per_substream', 256)

    @cuda_test
    async def test_fx_capture(self):
        """Create an FX stream, start it, and stop it again"""
        await self._test_fx_capture()

    @cuda_test
    async def test_fx_capture_fast(self):
        """Run an FX capture as fast as possible"""
        await self._test_fx_capture(clock_ratio=0.0, min_dumps=10)

    async def _test_beamformer_capture(self, clock_ratio=None, min_dumps=None):
        if min_dumps is None:
            min_dumps = 5    # Should be enough to demonstrate overlapping I/O
        await self._configure_subarray()
        name = 'i0.tied-array-channelised-voltage.0x'
        uname = 'i0_tied_array_channelised_voltage_0x'
        await self.make_request('stream-create-beamformer', name,
                                1712000000, 1284000000, 856000000, 4096, 4, 256, 8)
        await self.make_request('capture-destination', name, '127.0.0.1:7149', 'lo', False)
        await self.make_request('capture-start', name)
        for i in range(min_dumps):
            await _current_transport.dumps_semaphore.acquire()
        await self.make_request('capture-stop', name)
        assert_is_not_none(_current_transport)
        assert_greater_equal(_current_transport.dumps, min_dumps)
        assert_true(_current_transport.closed)
        self._check_common_telstate()
        view = self._telstate.view(uname)
        self._check_attribute(view, 'bandwidth', 856000000.0)
        self._check_attribute(view, 'n_chans_per_substream', 1024)
        self._check_attribute(view, 'spectra_per_heap', 256)
        self._check_sensor(view, 'weight', '[1.0, 1.0, 1.0, 1.0]')

    async def test_beamformer_capture(self):
        """Create a beamformer target, start it, and stop it again"""
        await self._test_beamformer_capture()

    async def test_beamformer_capture_fast(self):
        """Run the beamformer simulation as fast as possible"""
        await self._test_beamformer_capture(clock_ratio=0.0, min_dumps=20)

    async def test_unknown_stream_name(self):
        """An appropriate error is returned when using an unknown stream name."""
        await self.assert_request_fails(
            '^requested stream name "unknown" not found$',
            'capture-destination', 'unknown', '127.0.0.1:7147')
        await self.assert_request_fails(
            '^requested stream name "unknown" not found$',
            'capture-destination-file', 'unknown', '/dev/null')
        await self.assert_request_fails(
            '^requested stream name "unknown" not found$',
            'capture-start', 'unknown')
        await self.assert_request_fails(
            '^requested stream name "unknown" not found$',
            'capture-stop', 'unknown')
        await self.assert_request_fails(
            '^requested stream name "unknown" not found$',
            'accumulation-length', 'unknown', 0.5)
        await self.assert_request_fails(
            '^requested stream name "unknown" not found$',
            'frequency-select', 'unknown', 1000000000)

    async def test_change_while_capturing(self):
        """An appropriate error is returned when trying to change values while
        a capture is in progress."""
        await self._configure_subarray()
        # Use lower bandwidth to reduce test time
        await self.make_request('stream-create-beamformer', 'beam1',
                                1712000000, 1284000000, 856000000 / 4, 32768, 16, 256, 8)
        await self.make_request('capture-destination', 'beam1', '127.0.0.1:7149')
        await self.make_request('capture-start', 'beam1')
        await self.assert_request_fails(
            '^cannot modify antennas while capture is in progress$',
            'antenna-add', M062_DESCRIPTION)
        await self.assert_request_fails(
            '^cannot add source while capture is in progress$',
            'source-add', 'test3, radec, 3:30:00.00, -35:00:00.0, (500.0 2000.0 1.0)')
        await self.assert_request_fails(
            '^cannot set clock_ratio while capture is in progress$',
            'clock-ratio', 1.0)
        await self.assert_request_fails(
            '^cannot set center_frequency while capture is in progress',
            'frequency-select', 'beam1', 10000000000)
        await self.make_request('capture-stop', 'beam1')

    async def test_change_while_streams_exist(self):
        """An appropriate error is returned when trying to change values while
        a stream exists."""
        await self._configure_subarray()
        await self.make_request(
            'stream-create-beamformer', 'beam1',
            1712000000, 1284000000, 856000000, 32768, 16, 256, 8)
        await self.assert_request_fails(
            '^cannot add new antennas after creating a stream$',
            'antenna-add',
            'm123, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -1440.69968823 -2269.26759132 6.0, -0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 -0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22')  # noqa: E501
        await self.assert_request_fails(
            '^cannot set sync_time after creating a stream$',
            'sync-time', 1446544133)

    async def _get_antenna_descriptions(self):
        informs = await self.make_request('antenna-list')
        return [inform.arguments[0].decode('ascii') for inform in informs]

    async def test_replace_antenna(self):
        """Adding an antenna with a duplicate name replaces the existing one."""
        await self._configure_subarray()
        antennas = await self._get_antenna_descriptions()
        assert_equal([M062_DESCRIPTION, M063_DESCRIPTION], antennas)
        # Change the latitude, to check that the old value is replaced
        await self.make_request('antenna-add', M062_ALT_DESCRIPTION)
        antennas = await self._get_antenna_descriptions()
        assert_equal([M062_ALT_DESCRIPTION, M063_DESCRIPTION], antennas)

    async def test_configure_subarray_from_telstate(self):
        """Success case for configure-subarray-from-telstate request"""
        # This is a somewhat fragile test because it doesn't fully
        # simulate telstate, but the fakeredis telstate is a singleton
        # and so leaks state across tests.
        telstate = {
            'm062_observer': M062_DESCRIPTION,
            'm063_observer': M063_DESCRIPTION
        }
        self._server._telstate = telstate
        await self.make_request('configure-subarray-from-telstate', 'm062,m063')
        antennas = await self._get_antenna_descriptions()
        assert_equal([M062_DESCRIPTION, M063_DESCRIPTION], antennas)

    async def test_configure_subarray_from_telstate_missing_antenna(self):
        telstate = {
            'm062_observer': M062_DESCRIPTION
        }
        self._server._telstate = telstate
        await self.assert_request_fails(
            '^Antenna description for m063 not found$',
            'configure-subarray-from-telstate', 'm062,m063')
