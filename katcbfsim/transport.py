"""Output stream abstraction"""

from __future__ import print_function, division
import trollius
from trollius import From
import numpy as np
import spead2
import spead2.send
import spead2.send.trollius
import katsdptelstate
import concurrent.futures
import h5py
import logging
from collections import deque


logger = logging.getLogger(__name__)


class EndpointFactory(object):
    def __init__(self, cls, endpoints, n_substreams, max_packet_size):
        self.cls = cls
        self.endpoints = endpoints
        self.n_substreams = n_substreams
        if max_packet_size is None:
            max_packet_size = 4096
        self.max_packet_size = max_packet_size

    def __call__(self, stream):
        return self.cls(self.endpoints, self.n_substreams, self.max_packet_size, stream)


class SpeadTransport(object):
    """Base class for SPEAD streams, providing a factory function."""
    @classmethod
    def factory(cls, endpoints, n_substreams, max_packet_size):
        return EndpointFactory(cls, endpoints, n_substreams, max_packet_size)

    @property
    def n_endpoints(self):
        return len(self.endpoints)

    def __init__(self, endpoints, n_substreams, max_packet_size, stream, in_rate):
        if not endpoints:
            raise ValueError('At least one endpoint is required')
        n = len(endpoints)
        if stream.n_channels % n_substreams:
            raise ValueError('Number of channels not divisible by number of substreams')
        if n_substreams % n:
            raise ValueError('Number of substreams not divisible by number of endpoints')
        self.endpoints = endpoints
        self.stream = stream
        self.n_substreams = n_substreams
        self._flavour = spead2.Flavour(4, 64, 48, 0)
        self._inline_format = [('u', self._flavour.heap_address_bits)]
        # Send at a slightly higher rate, to account for overheads, and so
        # that if the sender sends a burst we can catch up with it.
        out_rate = in_rate * 1.05 / n
        config = spead2.send.StreamConfig(rate=out_rate, max_packet_size=max_packet_size)
        self._substreams = []
        for i in range(n_substreams):
            e = endpoints[i * len(endpoints) // n_substreams]
            self._substreams.append(spead2.send.trollius.UdpStream(
                spead2.ThreadPool(), e.host, e.port, config))
            self._substreams[-1].set_cnt_sequence(i, n_substreams)

    @trollius.coroutine
    def close(self):
        # This is to ensure that the end packet won't be dropped for lack of
        # space in the sending buffer. In normal use it won't do anything
        # because we always asynchronously wait for transmission, but in an
        # exception case there might be pending sends.
        for i, substream in enumerate(self._substreams):
            heap = self.ig_data[i].get_end()
            yield From(substream.async_flush())
            yield From(substream.async_send_heap(heap))


class CBFSpeadTransport(SpeadTransport):
    """Common base class for correlator and beamformer streams, with utilities
    for shared items.
    """
    def __init__(self, *args, **kwargs):
        super(CBFSpeadTransport, self).__init__(*args, **kwargs)
        self.ig_data = [self.make_ig_data() for i in range(self.n_substreams)]

    def make_ig_data(self):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        ig.add_item(0x1600, 'timestamp', 'Timestamp of start of this integration. uint counting multiples of ADC samples since last sync (sync_time, id=0x1027). Divide this number by timestamp_scale (id=0x1046) to get back to seconds since last sync when this integration was actually started. Note that the receiver will need to figure out the centre timestamp of the accumulation (eg, by adding half of int_time, id 0x1016).',
            (), None, format=self._inline_format)
        ig.add_item(0x4103, 'frequency', 'Identifies the first channel in the band of frequencies in the SPEAD heap. Can be used to reconstruct the full spectrum.',
            (), None, format=self._inline_format)
        return ig

    @trollius.coroutine
    def _send_metadata_endpoint(self, endpoint_idx):
        """Reissue all the metadata on the stream (for one endpoint)."""
        substream_idx = endpoint_idx * self.n_substreams // self.n_endpoints
        substream = self._substreams[substream_idx]
        heap = self.ig_data[substream_idx].get_heap(descriptors='all', data='none')
        yield From(substream.async_send_heap(heap))
        yield From(substream.async_send_heap(self.ig_data[endpoint_idx].get_start()))

    @trollius.coroutine
    def send_metadata(self):
        """Reissue all the metadata on the stream."""
        futures = []
        # Send to all endpoints in parallel
        for i in range(self.n_endpoints):
            futures.append(trollius.ensure_future(self._send_metadata_endpoint(i),
                                                  loop=self.stream.loop))
        yield From(trollius.gather(*futures, loop=self.stream.loop))


class FXSpeadTransport(CBFSpeadTransport):
    """Data stream from an FX correlator, sent over SPEAD."""
    def __init__(self, endpoints, n_substreams, max_packet_size, stream):
        in_rate = stream.n_baselines * stream.n_channels * 4 * 8 / \
            stream.accumulation_length
        super(FXSpeadTransport, self).__init__(endpoints, n_substreams, max_packet_size, stream, in_rate)

    def make_ig_data(self):
        ig = super(FXSpeadTransport, self).make_ig_data()
        # flags_xeng_raw is still TBD in the ICD, so omitted for now
        ig.add_item(0x1800, 'xeng_raw', 'Raw data stream from all the X-engines in the system. For KAT-7, this item represents a full spectrum (all frequency channels) assembled from lowest frequency to highest frequency. Each frequency channel contains the data for all baselines (n_bls given by SPEAD Id=0x1008). Each value is a complex number - two (real and imaginary) signed integers.',
            (self.stream.n_channels // self.n_substreams, self.stream.n_baselines * 4, 2), np.int32)
        return ig

    @trollius.coroutine
    def send(self, vis, dump_index):
        assert vis.flags.c_contiguous, 'Visibility array must be contiguous'
        shape = (self.stream.n_channels, self.stream.n_baselines * 4, 2)
        substream_channels = self.stream.n_channels // self.n_substreams
        vis_view = vis.reshape(*shape)
        futures = []
        timestamp = dump_index * self.stream.n_accs * self.stream.n_channels * \
                self.stream.scale_factor_timestamp // self.stream.bandwidth
        # Truncate timestamp to the width of the field it is in
        timestamp = timestamp & ((1 << self._flavour.heap_address_bits) - 1)
        for i, substream in enumerate(self._substreams):
            channel0 = substream_channels * i
            channel1 = channel0 + substream_channels
            self.ig_data[i]['xeng_raw'].value = vis_view[channel0:channel1]
            self.ig_data[i]['timestamp'].value = timestamp
            self.ig_data[i]['frequency'].value = channel0
            heap = self.ig_data[i].get_heap()
            futures.append(trollius.ensure_future(substream.async_send_heap(heap),
                                                  loop=self.stream.loop))
        yield From(trollius.gather(*futures, loop=self.stream.loop))


class FileFactory(object):
    def __init__(self, cls, filename):
        self.cls = cls
        self.filename = filename

    def __call__(self, *args, **kwargs):
        return self.cls(filename, *args, **kwargs)


class FileTransport(object):
    """Base class for file-sink streams, providing a factory function."""
    @classmethod
    def factory(cls, filename):
        return FileFactory(cls, filename)

    def __init__(self, filename, stream):
        self._stream = stream
        self._file = h5py.File(filename, 'w')

    @trollius.coroutine
    def close(self):
        self._file.close()


class FXFileTransport(FileTransport):
    """Writes data to HDF5 file. This is just the raw visibilities, and is not
    katdal-compatible."""
    def __init__(self, filename, stream):
        super(FXFileTransport, self).__init__(filename, stream)
        self._dataset = self._file.create_dataset('correlator_data',
            (0, stream.n_channels, stream.n_baselines * 4, 2), dtype=np.int32,
            maxshape=(None, stream.n_channels, stream.n_baselines * 4, 2))
        self._flags = None

    @trollius.coroutine
    def send_metadata(self):
        pass

    @trollius.coroutine
    def send(self, vis, dump_index):
        vis = vis.reshape((vis.shape[0], vis.shape[1] * 4, 2))
        self._dataset.resize(dump_index + 1, axis=0)
        self._dataset[dump_index : dump_index + 1, ...] = vis[np.newaxis, ...]


#############################################################################

class BeamformerSpeadTransport(CBFSpeadTransport):
    """Data stream from a beamformer, sent over SPEAD."""
    def __init__(self, endpoints, n_substreams, max_packet_size, stream):
        if stream.wall_interval == 0:
            in_rate = 0
        else:
            in_rate = stream.n_channels * stream.timesteps * 2 * stream.sample_bits / stream.wall_interval / 8
        super(BeamformerSpeadTransport, self).__init__(
            endpoints, n_substreams, max_packet_size, stream, in_rate)

    def make_ig_data(self):
        ig = super(BeamformerSpeadTransport, self).make_ig_data()
        ig.add_item(0x5000, 'bf_raw', 'Beamformer output for frequency-domain beam. User-defined name (out of band control). Record length depending on number of frequency channels and F-X packet size (xeng_acc_len).',
            shape=(self.stream.n_channels // self.n_substreams, self.stream.timesteps, 2), dtype=self.stream.dtype)
        return ig

    @trollius.coroutine
    def send(self, beam_data, index):
        substream_channels = self.stream.n_channels // self.n_substreams
        timestamp = index * self.stream.timesteps * self.stream.n_channels * \
                self.stream.scale_factor_timestamp // self.stream.bandwidth
        # Truncate timestamp to the width of the field it is in
        timestamp = timestamp & ((1 << self._flavour.heap_address_bits) - 1)
        futures = []
        for i, substream in enumerate(self._substreams):
            channel0 = substream_channels * i
            channel1 = channel0 + substream_channels
            self.ig_data[i]['bf_raw'].value = beam_data[channel0:channel1]
            self.ig_data[i]['timestamp'].value = timestamp
            heap = self.ig_data[i].get_heap()
            futures.append(substream.async_send_heap(heap))
        yield From(trollius.gather(*futures, loop=self.stream.loop))


#############################################################################

class TelstateFactory(object):
    def __init__(self, cls, telstate, n_substreams, **kwargs):
        self.cls = cls
        self.telstate = telstate
        self.n_substreams = n_substreams
        self.kwargs = kwargs

    def __call__(self, stream):
        return self.cls(self.telstate, self.n_substreams, stream, **self.kwargs)


class TelstateTransport(object):
    """Transport that puts metadata into telstate. Sending data is a no-op."""
    @classmethod
    def factory(cls, telstate, n_substreams, **kwargs):
        return TelstateFactory(cls, telstate, n_substreams, **kwargs)

    def __init__(self, telstate, n_substreams, stream):
        if not telstate:
            raise ValueError('A telstate connection is required')
        self.telstate = telstate
        self.stream = stream
        self.n_substreams = n_substreams

    def sensor(self, key, value, immutable=True):
        try:
            self.telstate.add(key, value, immutable)
        except katsdptelstate.ImmutableKeyError:
            logger.error('Could not set %s to %r because it already has a different value',
                         key, value)

    def _send_metadata(self):
        """Blocking implementation of :meth:`send_metadata`, run in a separate thread."""
        pass

    @trollius.coroutine
    def send_metadata(self):
        yield From(self.stream.loop.run_in_executor(self._executor, self._send_metadata))

    @trollius.coroutine
    def send(self, data, index):
        pass

    @trollius.coroutine
    def close(self):
        self._executor.shutdown()


class CBFTelstateTransport(TelstateTransport):
    """:class:`TelstateTransport` specialisation for streams derived from
    antenna channelised voltages."""
    def __init__(self, telstate, n_substreams, stream,
                 instrument_name=None, stream_name=None,
                 antenna_channelised_voltage_stream_name=None):
        super(CBFTelstateTransport, self).__init__(telstate, n_substreams, stream)
        self.instrument_name = instrument_name
        self.stream_name = stream_name
        self.antenna_channelised_voltage_stream_name = antenna_channelised_voltage_stream_name
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    @classmethod
    def _make_prefix(cls, scope):
        """Creates a prefix for a sensor name. If `scope` is ``None``, returns
        :samp:`cbf_`, otherwise :samp:`cbf_{scope}_` where the scope is
        underscore-normalised.
        """
        if scope is None:
            return 'cbf_'
        else:
            scope = scope.replace('.', '_').replace('-', '_')
            return 'cbf_{}_'.format(scope)

    def _instrument_sensors(self):
        pre = self._make_prefix(self.instrument_name)
        # Only the sensors captured by cam2telstate are simulated
        self.sensor(pre + 'adc_sample_rate', float(self.stream.adc_rate))
        self.sensor(pre + 'bandwidth', float(self.stream.bandwidth))
        self.sensor(pre + 'scale_factor_timestamp', float(self.stream.scale_factor_timestamp))
        self.sensor(pre + 'sync_time', self.stream.subarray.sync_time.secs)
        self.sensor(pre + 'n_inputs', 2 * self.stream.n_antennas)

    def _antenna_channelised_voltage_sensors(self):
        pre = self._make_prefix(self.antenna_channelised_voltage_stream_name)
        for i in range(2 * self.stream.n_antennas):
            input_pre = pre + 'input{}_'.format(i)
            # These are all arbitrary dummy values
            self.sensor(input_pre + 'delay', (0, 0, 0, 0, 0), immutable=False)
            self.sensor(input_pre + 'delay_ok', True, immutable=False)
            self.sensor(input_pre + 'eq', [200 + 0j], immutable=False)
            self.sensor(input_pre + 'fft0_shift', 32767, immutable=False)
        # Need to report the baseband center frequency
        center_frequency = self.stream.center_frequency % self.stream.bandwidth
        self.sensor(pre + 'center_freq', float(center_frequency))
        self.sensor(pre + 'n_chans', self.stream.n_channels)
        self.sensor(pre + 'ticks_between_spectra',
                    self.stream.n_channels * self.stream.scale_factor_timestamp // self.stream.bandwidth)

    def _send_metadata(self):
        super(CBFTelstateTransport, self)._send_metadata()
        self._instrument_sensors()
        self._antenna_channelised_voltage_sensors()


class FXTelstateTransport(CBFTelstateTransport):
    def _baseline_correlation_products_sensors(self):
        pre = self._make_prefix(self.stream_name)
        baselines = []
        for i in range(self.stream.n_antennas):
            for j in range(i, self.stream.n_antennas):
                for pol1 in ('v', 'h'):
                    for pol2 in ('v', 'h'):
                        name1 = self.stream.subarray.antennas[i].name + pol1
                        name2 = self.stream.subarray.antennas[j].name + pol2
                        baselines.append((name1, name2))
        self.sensor(pre + 'bls_ordering', baselines)
        self.sensor(pre + 'int_time', self.stream.accumulation_length)
        self.sensor(pre + 'n_accs', self.stream.n_accs)

    def _send_metadata(self):
        super(FXTelstateTransport, self)._send_metadata()
        self._baseline_correlation_products_sensors()
        self.sensor('sdp_cam2telstate_status', 'ready', immutable=False)


class BeamformerTelstateTransport(CBFTelstateTransport):
    def _tied_array_channelised_voltage_sensors(self):
        pre = self._make_prefix(self.stream_name)
        self.sensor(pre + 'n_chans', self.stream.n_channels, immutable=False)
        self.sensor(pre + 'n_chans_per_substream', self.stream.n_channels // self.n_substreams,
            immutable=False)
        self.sensor(pre + 'spectra_per_heap', self.stream.timesteps)
        for i in range(2 * self.stream.n_antennas):
            self.sensor('{}input{}_weight'.format(pre, i), 1.0, immutable=False)

    def _send_metadata(self):
        super(BeamformerTelstateTransport, self)._send_metadata()
        self._tied_array_channelised_voltage_sensors()
        self.sensor('sdp_cam2telstate_status', 'ready', immutable=False)
