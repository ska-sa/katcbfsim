"""Output stream abstraction"""

import logging
import asyncio
from collections import deque
import concurrent.futures

import netifaces
import numpy as np
import h5py

import spead2
import spead2.send
import spead2.send.asyncio

import katsdpservices


logger = logging.getLogger(__name__)


class EndpointFactory(object):
    def __init__(self, cls, endpoints, ifaddr, ibv, max_packet_size):
        self.cls = cls
        self.endpoints = endpoints
        self.ifaddr = ifaddr
        self.ibv = ibv
        if max_packet_size is None:
            max_packet_size = 4096
        self.max_packet_size = max_packet_size

    def __call__(self, stream):
        return self.cls(self.endpoints, self.ifaddr, self.ibv,
                        self.max_packet_size, stream)


class SpeadTransport(object):
    """Base class for SPEAD streams, providing a factory function."""
    @classmethod
    def factory(cls, endpoints, ifaddr, ibv, max_packet_size):
        return EndpointFactory(cls, endpoints, ifaddr, ibv, max_packet_size)

    @property
    def n_endpoints(self):
        return len(self.endpoints)

    def __init__(self, endpoints, ifaddr, ibv, max_packet_size, stream, in_rate):
        if not endpoints:
            raise ValueError('At least one endpoint is required')
        n = len(endpoints)
        if stream.n_substreams % n:
            raise ValueError('Number of substreams not divisible by number of endpoints')
        self.endpoints = endpoints
        self.stream = stream
        self._flavour = spead2.Flavour(4, 64, 48, 0)
        self._inline_format = [('u', self._flavour.heap_address_bits)]
        # Send at a slightly higher rate, to account for overheads, and so
        # that if the sender sends a burst we can catch up with it.
        out_rate = in_rate * 1.05 / stream.n_substreams
        config = spead2.send.StreamConfig(rate=out_rate, max_packet_size=max_packet_size)
        self._substreams = []
        for i in range(stream.n_substreams):
            e = endpoints[i * len(endpoints) // stream.n_substreams]
            kwargs = {}
            if ifaddr is not None:
                kwargs['interface_address'] = ifaddr
                kwargs['ttl'] = 1
            if ibv:
                stream_cls = spead2.send.asyncio.UdpIbvStream
            else:
                stream_cls = spead2.send.asyncio.UdpStream
            self._substreams.append(stream_cls(
                spead2.ThreadPool(), e.host, e.port, config, **kwargs))
            self._substreams[-1].set_cnt_sequence(i, stream.n_substreams)

    async def close(self):
        # This is to ensure that the end packet won't be dropped for lack of
        # space in the sending buffer. In normal use it won't do anything
        # because we always asynchronously wait for transmission, but in an
        # exception case there might be pending sends.
        for i, substream in enumerate(self._substreams):
            heap = self.ig_data[i].get_end()
            await substream.async_flush()
            await substream.async_send_heap(heap)


class CBFSpeadTransport(SpeadTransport):
    """Common base class for correlator and beamformer streams, with utilities
    for shared items.
    """
    def __init__(self, *args, **kwargs):
        super(CBFSpeadTransport, self).__init__(*args, **kwargs)
        self.ig_data = [self.make_ig_data() for i in range(self.stream.n_substreams)]

    def make_ig_data(self):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        ig.add_item(0x1600, 'timestamp', 'Timestamp of start of this integration. uint counting multiples of ADC samples since last sync (sync_time, id=0x1027). Divide this number by timestamp_scale (id=0x1046) to get back to seconds since last sync when this integration was actually started. Note that the receiver will need to figure out the centre timestamp of the accumulation (eg, by adding half of int_time, id 0x1016).',
            (), None, format=self._inline_format)
        ig.add_item(0x4103, 'frequency', 'Identifies the first channel in the band of frequencies in the SPEAD heap. Can be used to reconstruct the full spectrum.',
            (), None, format=self._inline_format)
        return ig

    async def _send_metadata_endpoint(self, endpoint_idx):
        """Reissue all the metadata on the stream (for one endpoint)."""
        substream_idx = endpoint_idx * self.stream.n_substreams // self.n_endpoints
        substream = self._substreams[substream_idx]
        heap = self.ig_data[substream_idx].get_heap(descriptors='all', data='none')
        await substream.async_send_heap(heap)
        await substream.async_send_heap(self.ig_data[endpoint_idx].get_start())

    async def send_metadata(self):
        """Reissue all the metadata on the stream."""
        futures = []
        # Send to all endpoints in parallel
        for i in range(self.n_endpoints):
            futures.append(asyncio.ensure_future(self._send_metadata_endpoint(i),
                                                  loop=self.stream.loop))
        await asyncio.gather(*futures, loop=self.stream.loop)


class FXSpeadTransport(CBFSpeadTransport):
    """Data stream from an FX correlator, sent over SPEAD."""
    def __init__(self, endpoints, ifaddr, ibv, max_packet_size, stream):
        if stream.wall_accumulation_length == 0:
            in_rate = 0
        else:
            in_rate = stream.n_baselines * stream.n_channels * 4 * 8 / \
                stream.wall_accumulation_length
        super(FXSpeadTransport, self).__init__(
            endpoints, ifaddr, ibv, max_packet_size, stream, in_rate)
        self._last_metadata = 0   # Dump index of last periodic metadata

    def make_ig_data(self):
        ig = super(FXSpeadTransport, self).make_ig_data()
        # flags_xeng_raw is still TBD in the ICD, so omitted for now
        ig.add_item(0x1800, 'xeng_raw', 'Raw data stream from all the X-engines in the system. For KAT-7, this item represents a full spectrum (all frequency channels) assembled from lowest frequency to highest frequency. Each frequency channel contains the data for all baselines (n_bls given by SPEAD Id=0x1008). Each value is a complex number - two (real and imaginary) signed integers.',
            (self.stream.n_channels // self.stream.n_substreams, self.stream.n_baselines * 4, 2),
            np.int32)
        return ig

    async def send(self, vis, dump_index):
        if (dump_index - self._last_metadata) * self.stream.accumulation_length >= 5.0:
            self._last_metadata = dump_index
            await self.send_metadata()
        assert vis.flags.c_contiguous, 'Visibility array must be contiguous'
        shape = (self.stream.n_channels, self.stream.n_baselines * 4, 2)
        substream_channels = self.stream.n_channels // self.stream.n_substreams
        vis_view = vis.reshape(*shape)
        futures = []
        timestamp = int(dump_index * self.stream.n_accs * self.stream.n_channels
                        * self.stream.scale_factor_timestamp / self.stream.bandwidth)
        # Truncate timestamp to the width of the field it is in
        timestamp = timestamp & ((1 << self._flavour.heap_address_bits) - 1)
        for i, substream in enumerate(self._substreams):
            channel0 = substream_channels * i
            channel1 = channel0 + substream_channels
            self.ig_data[i]['xeng_raw'].value = vis_view[channel0:channel1]
            self.ig_data[i]['timestamp'].value = timestamp
            self.ig_data[i]['frequency'].value = channel0
            heap = self.ig_data[i].get_heap()
            futures.append(asyncio.ensure_future(substream.async_send_heap(heap),
                                                  loop=self.stream.loop))
        await asyncio.gather(*futures, loop=self.stream.loop)


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

    async def close(self):
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

    async def send_metadata(self):
        pass

    async def send(self, vis, dump_index):
        vis = vis.reshape((vis.shape[0], vis.shape[1] * 4, 2))
        self._dataset.resize(dump_index + 1, axis=0)
        self._dataset[dump_index : dump_index + 1, ...] = vis[np.newaxis, ...]


#############################################################################

class BeamformerSpeadTransport(CBFSpeadTransport):
    """Data stream from a beamformer, sent over SPEAD."""
    def __init__(self, endpoints, ifaddr, ibv, max_packet_size, stream):
        if stream.wall_interval == 0:
            in_rate = 0
        else:
            in_rate = stream.n_channels * stream.timesteps * 2 * stream.sample_bits / stream.wall_interval / 8
        super(BeamformerSpeadTransport, self).__init__(
            endpoints, ifaddr, ibv, max_packet_size, stream, in_rate)
        self._last_metadata = 0     # Dump index of last periodic metadata

    def make_ig_data(self):
        ig = super(BeamformerSpeadTransport, self).make_ig_data()
        ig.add_item(0x5000, 'bf_raw', 'Beamformer output for frequency-domain beam. User-defined name (out of band control). Record length depending on number of frequency channels and F-X packet size (xeng_acc_len).',
            shape=(self.stream.n_channels // self.stream.n_substreams, self.stream.timesteps, 2),
            dtype=self.stream.dtype)
        return ig

    async def send(self, beam_data, index):
        if (index - self._last_metadata) * self.stream.interval >= 5.0:
            self._last_metadata = index
            await self.send_metadata()
        substream_channels = self.stream.n_channels // self.stream.n_substreams
        timestamp = int(index * self.stream.timesteps * self.stream.n_channels * \
                self.stream.scale_factor_timestamp / self.stream.bandwidth)
        # Truncate timestamp to the width of the field it is in
        timestamp = timestamp & ((1 << self._flavour.heap_address_bits) - 1)
        futures = []
        for i, substream in enumerate(self._substreams):
            channel0 = substream_channels * i
            channel1 = channel0 + substream_channels
            self.ig_data[i]['bf_raw'].value = beam_data[channel0:channel1]
            self.ig_data[i]['timestamp'].value = timestamp
            self.ig_data[i]['frequency'].value = channel0
            heap = self.ig_data[i].get_heap()
            futures.append(substream.async_send_heap(heap))
        await asyncio.gather(*futures, loop=self.stream.loop)
