"""Output stream abstraction"""

import logging
import asyncio

import numpy as np
import h5py

import spead2
import spead2.send
import spead2.send.asyncio


# This is tuned for the beamformer output: each packet contains
# - SPEAD header (8 bytes)
# - standard SPEAD items (4 * 8 bytes)
# - beamformer-specific items (3 * 8 bytes)
DEFAULT_MAX_PACKET_SIZE = 4096 + 64
logger = logging.getLogger(__name__)


class EndpointFactory(object):
    def __init__(self, cls, endpoints, ifaddr, ibv, max_packet_size):
        self.cls = cls
        self.endpoints = endpoints
        self.ifaddr = ifaddr
        self.ibv = ibv
        if max_packet_size is None:
            max_packet_size = DEFAULT_MAX_PACKET_SIZE
        self.max_packet_size = max_packet_size

    def __call__(self, stream):
        return self.cls(self.endpoints, self.ifaddr, self.ibv,
                        self.max_packet_size, stream)


class Substream:
    def __init__(self, sender, endpoint, channel_range, index):
        self.sender = sender
        self.endpoint = endpoint
        self.channel_range = channel_range
        self.index = index

    async def async_send_heap(self, heap):
        return await self.sender.async_send_heap(heap, substream_index=self.index)


class SpeadTransport(object):
    """Base class for SPEAD streams, providing a factory function."""
    @classmethod
    def factory(cls, endpoints, ifaddr, ibv, max_packet_size):
        return EndpointFactory(cls, endpoints, ifaddr, ibv, max_packet_size)

    def __init__(self, endpoints, ifaddr, ibv, max_packet_size, stream, in_rate):
        if not endpoints:
            raise ValueError('At least one endpoint is required')
        if stream.n_substreams % len(endpoints):
            raise ValueError('Number of substreams not divisible by number of endpoints')
        server_id = stream.subarray.server_id
        n_servers = stream.subarray.n_servers
        first_substream = stream.n_substreams * server_id // n_servers
        last_substream = stream.n_substreams * (server_id + 1) // n_servers
        my_n_substreams = last_substream - first_substream
        self.stream = stream
        self._flavour = spead2.Flavour(4, 64, 48, 0)
        self._inline_format = [('u', self._flavour.heap_address_bits)]
        # Send at a slightly higher rate, to account for overheads, and so
        # that if the sender sends a burst we can catch up with it.
        out_rate = in_rate * 1.1 / stream.n_substreams * my_n_substreams
        config = spead2.send.StreamConfig(
            max_heaps=my_n_substreams,
            rate=out_rate,
            max_packet_size=max_packet_size)
        my_endpoints = [
            endpoints[i * len(endpoints) // stream.n_substreams]
            for i in range(first_substream, last_substream)
        ]
        # spead2 takes a tuple, not an Endpoint object
        spead2_endpoints = [
            (endpoint.host, endpoint.port)
            for endpoint in my_endpoints
        ]
        if ibv:
            sender = spead2.send.asyncio.UdpIbvStream(
                spead2.ThreadPool(),
                config,
                spead2.send.UdpIbvConfig(
                    endpoints=spead2_endpoints,
                    interface_address=ifaddr,
                    ttl=4           # For MeerKAT layer 3 switching
                )
            )
        else:
            kwargs = {}
            if ifaddr is not None:
                kwargs['interface_address'] = ifaddr
                kwargs['ttl'] = 4   # For MeerKAT layer 3 switching
            sender = spead2.send.asyncio.UdpStream(
                spead2.ThreadPool(), spead2_endpoints, config, **kwargs)
        sender.set_cnt_sequence(server_id, n_servers)
        self._substreams = [
            Substream(
                sender=sender,
                endpoint=endpoint,
                channel_range=slice(
                    i * stream.n_channels // stream.n_substreams,
                    (i + 1) * stream.n_channels // stream.n_substreams
                ),
                index=i - first_substream
            )
            for i, endpoint in enumerate(my_endpoints, start=first_substream)
        ]

    async def close(self):
        # This is to ensure that the end packet won't be dropped for lack of
        # space in the sending buffer. In normal use it won't do anything
        # because we always asynchronously wait for transmission, but in an
        # exception case there might be pending sends.
        prev_endpoint = None
        for substream in self._substreams:
            if substream.endpoint != prev_endpoint:
                heap = self.ig_data.get_end()
                await substream.sender.async_flush()
                await substream.async_send_heap(heap)
                prev_endpoint = substream.endpoint


class CBFSpeadTransport(SpeadTransport):
    """Common base class for correlator and beamformer streams, with utilities
    for shared items.
    """
    def __init__(self, *args, **kwargs):
        super(CBFSpeadTransport, self).__init__(*args, **kwargs)
        self.ig_data = spead2.send.ItemGroup(flavour=self._flavour)
        self.ig_data.add_item(
            0x1600, 'timestamp',
            'Timestamp of start of this integration. '
            'uint counting multiples of ADC samples since last sync '
            '(sync_time, id=0x1027). '
            'Divide this number by timestamp_scale (id=0x1046) '
            'to get back to seconds since last sync '
            'when this integration was actually started. '
            'Note that the receiver will need to figure out the centre timestamp '
            'of the accumulation (eg, by adding half of int_time, id 0x1016).',
            (), None, format=self._inline_format)
        self.ig_data.add_item(
            0x4103, 'frequency',
            'Identifies the first channel in the band of frequencies in the SPEAD heap. '
            'Can be used to reconstruct the full spectrum.',
            (), None, format=self._inline_format)

    async def _send_metadata_endpoint(self, substream, start):
        """Reissue all the metadata on the stream (for one endpoint)."""
        heap = self.ig_data.get_heap(descriptors='all', data='none')
        await substream.async_send_heap(heap)
        if start:
            await substream.async_send_heap(self.ig_data.get_start())

    async def send_metadata(self, start=True):
        """Reissue all the metadata on the stream."""
        futures = []
        # Send to all endpoints in parallel
        prev_endpoint = None
        for substream in self._substreams:
            if substream.endpoint != prev_endpoint:
                futures.append(asyncio.ensure_future(
                    self._send_metadata_endpoint(substream, start=start)))
                prev_endpoint = substream.endpoint
        await asyncio.gather(*futures)


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
        # flags_xeng_raw is still TBD in the ICD, so omitted for now
        self.ig_data.add_item(
            0x1800, 'xeng_raw',
            'Raw data stream from all the X-engines in the system. '
            'For KAT-7, this item represents a full spectrum '
            '(all frequency channels) assembled from lowest frequency to highest frequency. '
            'Each frequency channel contains the data for all baselines '
            '(n_bls given by SPEAD Id=0x1008). '
            'Each value is a complex number - two (real and imaginary) signed integers.',
            (self.stream.n_channels // self.stream.n_substreams, self.stream.n_baselines * 4, 2),
            np.dtype('>i4'))

    async def send(self, vis, dump_index):
        if (dump_index - self._last_metadata) * self.stream.accumulation_length >= 5.0:
            self._last_metadata = dump_index
            await self.send_metadata(start=False)
        assert vis.flags.c_contiguous, 'Visibility array must be contiguous'
        shape = (-1, self.stream.n_baselines * 4, 2)
        vis_view = vis.reshape(*shape)
        timestamp = int(dump_index * self.stream.n_accs * self.stream.n_channels
                        * self.stream.scale_factor_timestamp / self.stream.bandwidth)
        timestamp += self.stream.start_timestamp
        # Truncate timestamp to the width of the field it is in
        timestamp = timestamp & ((1 << self._flavour.heap_address_bits) - 1)
        offset = self._substreams[0].channel_range.start
        heap_refs = []
        for substream in self._substreams:
            c0 = substream.channel_range.start - offset
            c1 = substream.channel_range.stop - offset
            self.ig_data['xeng_raw'].value = vis_view[c0:c1]
            self.ig_data['timestamp'].value = timestamp
            self.ig_data['frequency'].value = substream.channel_range.start
            heap = self.ig_data.get_heap()
            heap.repeat_pointers = True
            heap_refs.append(spead2.send.HeapReference(heap=heap, substream_index=substream.index))
        await self._substreams[0].sender.async_send_heaps(
            heap_refs, mode=spead2.send.GroupMode.ROUND_ROBIN)


class FileFactory(object):
    def __init__(self, cls, filename):
        self.cls = cls
        self.filename = filename

    def __call__(self, *args, **kwargs):
        return self.cls(self.filename, *args, **kwargs)


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
        n_channels = stream.n_channels // stream.subarray.n_servers
        self._dataset = self._file.create_dataset(
            'correlator_data',
            (0, n_channels, stream.n_baselines * 4, 2), dtype=np.int32,
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
            in_rate = (stream.n_channels * stream.timesteps * 2 * stream.sample_bits
                       / stream.wall_interval / 8)
        super(BeamformerSpeadTransport, self).__init__(
            endpoints, ifaddr, ibv, max_packet_size, stream, in_rate)
        self._last_metadata = 0     # Dump index of last periodic metadata
        self.ig_data.add_item(
            0x5000, 'bf_raw',
            'Beamformer output for frequency-domain beam. '
            'User-defined name (out of band control). '
            'Record length depending on number of frequency channels '
            'and F-X packet size (xeng_acc_len).',
            shape=(self.stream.n_channels // self.stream.n_substreams, self.stream.timesteps, 2),
            dtype=self.stream.dtype)

    async def send(self, beam_data, index):
        if (index - self._last_metadata) * self.stream.interval >= 5.0:
            self._last_metadata = index
            await self.send_metadata(start=False)
        timestamp = int(index * self.stream.timesteps * self.stream.n_channels
                        * self.stream.scale_factor_timestamp / self.stream.bandwidth)
        timestamp += self.stream.start_timestamp
        # Truncate timestamp to the width of the field it is in
        timestamp = timestamp & ((1 << self._flavour.heap_address_bits) - 1)
        offset = self._substreams[0].channel_range.start
        heap_refs = []
        for substream in self._substreams:
            c0 = substream.channel_range.start - offset
            c1 = substream.channel_range.stop - offset
            self.ig_data['bf_raw'].value = beam_data[c0:c1]
            self.ig_data['timestamp'].value = timestamp
            self.ig_data['frequency'].value = substream.channel_range.start
            heap = self.ig_data.get_heap()
            heap.repeat_pointers = True
            heap_refs.append(spead2.send.HeapReference(heap=heap, substream_index=substream.index))
        await self._substreams[0].sender.async_send_heaps(
            heap_refs, mode=spead2.send.GroupMode.ROUND_ROBIN)
