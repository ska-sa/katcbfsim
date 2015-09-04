"""Output stream abstraction"""

from __future__ import print_function, division
import trollius
from trollius import From
import numpy as np
import spead2
import spead2.send
import spead2.send.trollius
import logging


logger = logging.getLogger(__name__)


class FXStreamSpeadFactory(object):
    """Data stream from an FX correlator, sent over SPEAD."""
    def __init__(self, endpoints):
        self._endpoints = endpoints

    @property
    def endpoints(self):
        return self._endpoints

    def __call__(self, *args, **kwargs):
        return FXStreamSpead(self._endpoints, *args, **kwargs)


class FXStreamSpead(object):
    def __init__(self, endpoints, product):
        if len(endpoints) != 1:
            raise ValueError('Only exactly one endpoint is currently supported')
        self.endpoint = endpoints[0]
        self.product = product
        tp = spead2.ThreadPool()
        n_antennas = len(self.product.subarray.antennas)
        n_baselines = n_antennas * (n_antennas + 1) // 2
        n_channels = self.product.channels
        in_rate = n_baselines * n_channels * 4 * 8 / self.product.accumulation_length
        # Send at a slightly higher rate, to account for overheads, and so
        # that if the sender sends a burst we can catch up with it.
        out_rate = in_rate * 1.05

        config = spead2.send.StreamConfig(rate=out_rate)
        self._flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
        self._stream = spead2.send.trollius.UdpStream(
            tp, endpoints[0].host, endpoints[0].port, config)
        self._static_ig = self._make_static_ig()
        self._data_ig = self._make_data_ig()
        # The ICD defines certain ways in which items need to be placed in
        # heaps, which makes it difficult to use HeapGenerator. So we need
        # to track IDs ourselves.
        self._next_cnt = 1

    def _make_static_ig(self):
        ig = spead2.ItemGroup()
        inline_fmt = [('u', self._flavour.heap_address_bits)]
        n_antennas = len(self.product.subarray.antennas)
        n_baselines = n_antennas * (n_antennas + 1) // 2
        n_accs = 0  # TODO!
        ig.add_item(0x1007, 'adc_sample_rate', 'Expected ADC sample rate (sampled/second)',
            (), np.uint64, value=self.product.bandwidth * 2)
        ig.add_item(0x1008, 'n_bls', 'The total number of baselines in the data product. Each pair of inputs (polarisation pairs) is considered a baseline.',
            (), None, format=inline_fmt, value=n_baselines * 4)
        ig.add_item(0x1009, 'n_chans', 'The total number of frequency channels present in any integration.',
            (), None, format=inline_fmt, value=self.product.channels)
        ig.add_item(0x100A, 'n_ants', 'The total number of dual-pol antennas in the system.',
            (), None, format=inline_fmt, value=n_antennas)
        ig.add_item(0x100B, 'n_xengs', 'The total number of X engines in a correlator system.',
            (), None, format=inline_fmt, value=1)
        ig.add_item(0x1011, 'center_freq', 'The center frequency of the DBE in Hz, 64-bit IEEE floating-point number.',
            (), np.float64, value=np.float64(self.product.center_frequency))
        ig.add_item(0x1013, 'bandwidth', 'The analogue bandwidth of the digitally processed signal in Hz.',
            (), np.float64, value=np.float64(self.product.bandwidth))
        ig.add_item(0x1015, 'n_accs', 'The number of spectra that are accumulated per integration.',
            (), None, format=inline_fmt, value=n_accs)
        ig.add_item(0x1016, 'int_time', "Approximate (it's a float!) time per accumulation in seconds. This is intended for reference only. Each accumulation has an associated timestamp which should be used to determine the time of the integration rather than incrementing the start time by this value for sequential integrations (which would allow errors to grow).",
            (), np.float64, value=self.product.accumulation_length)
        ig.add_item(0x1022, 'rx_udp_port', 'Destination UDP port for data output.',
            (), None, format=inline_fmt, value=self.endpoint.port)
        # TODO: this might need to be translated from hostname to IP address
        ig.add_item(0x1024, 'rx_udp_ip_str', 'Destination IP address for output UDP packets.',
            (None,), None, format=[('c', 8)], value=self.endpoint.host)
        ig.add_item(0x1027, 'sync_time', 'Time at which the system was last synchronised (armed and triggered by a 1PPS) in seconds since the Unix Epoch.',
            (), None, format=inline_fmt, value=self.product.subarray.sync_time)
        # TODO: do we need ddc_mix_freq, adc_bits?
        # TODO: what scaling factor should we use?
        ig.add_item(0x1046, 'scale_factor_timestamp', 'Timestamp scaling factor. Divide the SPEAD data packet timestamp by this number to get back to seconds since last sync.',
            (), np.float64, value=self.product.bandwidth * 2)
        ig.add_item(0x1048, 'xeng_out_bits_per_sample', 'The number of bits per value of the xeng accumulator output. Note this is for a single component value, not the combined complex size.',
            (), None, format=inline_fmt, value=32)
        return ig

    def _make_data_ig(self):
        ig = spead2.ItemGroup()
        inline_fmt = [('u', self._flavour.heap_address_bits)]
        # TODO: reduce code duplication here - make a property of product?
        n_antennas = len(self.product.subarray.antennas)
        n_baselines = n_antennas * (n_antennas + 1) // 2
        n_channels = self.product.channels
        ig.add_item(0x1600, 'timestamp', 'Timestamp of start of this integration. uint counting multiples of ADC samples since last sync (sync_time, id=0x1027). Divide this number by timestamp_scale (id=0x1046) to get back to seconds since last sync when this integration was actually started. Note that the receiver will need to figure out the centre timestamp of the accumulation (eg, by adding half of int_time, id 0x1016).',
            (), None, format=inline_fmt)
        # flags_xeng_raw is still TBD in the ICD, so omitted for now
        ig.add_item(0x1800, 'xeng_raw', 'Raw data stream from all the X-engines in the system. For KAT-7, this item represents a full spectrum (all frequency channels) assembled from lowest frequency to highest frequency. Each frequency channel contains the data for all baselines (n_bls given by SPEAD Id=0x1008). Each value is a complex number - two (real and imaginary) signed integers.',
            (n_channels, n_baselines * 4, 2), np.int32)
        return ig

    def _next_heap(self):
        heap = spead2.send.Heap(self._next_cnt, self._flavour)
        self._next_cnt += 1
        return heap

    @trollius.coroutine
    def send_metadata(self):
        """Reissue all the metadata on the stream."""
        heap = self._next_heap()
        for item in self._static_ig.values():
            heap.add_descriptor(item)
            heap.add_item(item)
        for item in self._data_ig.values():
            heap.add_descriptor(item)
        yield From(self._stream.async_send_heap(heap))

    @trollius.coroutine
    def send(self, vis):
        assert vis.flags.c_contiguous, 'Visibility array must be contiguous'
        vis_view = vis.view(np.int32).reshape(self._data_ig['xeng_raw'].shape)
        self._data_ig['xeng_raw'].value = vis_view
        self._data_ig['timestamp'].value = 0  # TODO: make a parameter, work out scaling
        heap = self._next_heap()
        for item in self._data_ig.values():
            heap.add_item(item)
        yield From(self._stream.async_send_heap(heap))

    @trollius.coroutine
    def close():
        # This is to ensure that the end packet won't be dropped for lack of
        # space in the sending buffer. In normal use it won't do anything
        # because we always asynchronously wait for transmission, but in an
        # exception case there might be pending sends.
        self._stream.flush()
        heap = self._next_heap()
        heap.add_end()
        yield From(self._stream.async_send_heap(heap))
