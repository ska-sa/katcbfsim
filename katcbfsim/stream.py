"""Output stream abstraction"""

from __future__ import print_function, division
import trollius
from trollius import From
import numpy as np
import spead2
import spead2.send
import spead2.send.trollius
import h5py
import logging
import functools


logger = logging.getLogger(__name__)


class FXStreamSpeadFactory(object):
    """Factory that generates instances of :class:`FXStreamSpead`."""
    def __init__(self, endpoints):
        self._endpoints = endpoints

    @property
    def endpoints(self):
        return self._endpoints

    def __call__(self, *args, **kwargs):
        return FXStreamSpead(self._endpoints, *args, **kwargs)


class FXStreamSpead(object):
    """Data stream from an FX correlator, sent over SPEAD."""
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
        self._inline_format = [('u', self._flavour.heap_address_bits)]
        self._stream = spead2.send.trollius.UdpStream(
            tp, endpoints[0].host, endpoints[0].port, config)
        self._ig_static = self._make_ig_static()
        self._ig_gain = self._make_ig_gain()
        self._ig_labels = self._make_ig_labels()
        self._ig_data = self._make_ig_data()

    def _make_ig_static(self):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        n_antennas = len(self.product.subarray.antennas)
        n_baselines = n_antennas * (n_antennas + 1) // 2
        ig.add_item(0x1007, 'adc_sample_rate', 'Expected ADC sample rate (sampled/second)',
            (), np.uint64, value=self.product.bandwidth * 2)
        ig.add_item(0x1008, 'n_bls', 'The total number of baselines in the data product. Each pair of inputs (polarisation pairs) is considered a baseline.',
            (), None, format=self._inline_format, value=n_baselines * 4)
        ig.add_item(0x1009, 'n_chans', 'The total number of frequency channels present in any integration.',
            (), None, format=self._inline_format, value=self.product.channels)
        ig.add_item(0x100A, 'n_ants', 'The total number of dual-pol antennas in the system.',
            (), None, format=self._inline_format, value=n_antennas)
        ig.add_item(0x100B, 'n_xengs', 'The total number of X engines in a correlator system.',
            (), None, format=self._inline_format, value=1)
        ig.add_item(0x1011, 'center_freq', 'The center frequency of the DBE in Hz, 64-bit IEEE floating-point number.',
            (), np.float64, value=np.float64(self.product.center_frequency))
        ig.add_item(0x1013, 'bandwidth', 'The analogue bandwidth of the digitally processed signal in Hz.',
            (), np.float64, value=np.float64(self.product.bandwidth))
        ig.add_item(0x1015, 'n_accs', 'The number of spectra that are accumulated per integration.',
            (), None, format=self._inline_format, value=self.product.n_accs)
        ig.add_item(0x1016, 'int_time', "Approximate (it's a float!) time per accumulation in seconds. This is intended for reference only. Each accumulation has an associated timestamp which should be used to determine the time of the integration rather than incrementing the start time by this value for sequential integrations (which would allow errors to grow).",
            (), np.float64, value=self.product.accumulation_length)
        ig.add_item(0x1022, 'rx_udp_port', 'Destination UDP port for data output.',
            (), None, format=self._inline_format, value=self.endpoint.port)
        # TODO: this might need to be translated from hostname to IP address
        ig.add_item(0x1024, 'rx_udp_ip_str', 'Destination IP address for output UDP packets.',
            (None,), None, format=[('c', 8)], value=self.endpoint.host)
        ig.add_item(0x1027, 'sync_time', 'Time at which the system was last synchronised (armed and triggered by a 1PPS) in seconds since the Unix Epoch.',
            (), None, format=self._inline_format, value=self.product.subarray.sync_time.secs)
        # TODO: do we need ddc_mix_freq, adc_bits?
        # TODO: what scaling factor should we use?
        ig.add_item(0x1046, 'scale_factor_timestamp', 'Timestamp scaling factor. Divide the SPEAD data packet timestamp by this number to get back to seconds since last sync.',
            (), np.float64, value=self.product.bandwidth / self.product.channels)
        ig.add_item(0x1048, 'xeng_out_bits_per_sample', 'The number of bits per value of the xeng accumulator output. Note this is for a single component value, not the combined complex size.',
            (), None, format=self._inline_format, value=32)
        return ig

    def _make_ig_gain(self):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        initial_gain = np.zeros((self.product.channels, 2), np.uint32)
        initial_gain[:, 0].fill(200)    # Arbitrary value for now (200 + 0j)
        input_number = 0
        for i, antenna in enumerate(self.product.subarray.antennas):
            for pol in ('v', 'h'):
                ig.add_item(0x1400 + input_number, 'eq_coef_{}{}'.format(antenna.name, pol), 'The unitless, per-channel, digital scaling factors implemented prior to requantisation, for inputN. Complex number (real,imag) 32 bit integers.',
                    initial_gain.shape, initial_gain.dtype, value=initial_gain)
                input_number += 1
        return ig

    def _make_ig_labels(self):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        n_antennas = len(self.product.subarray.antennas)
        baselines = []
        for i in range(n_antennas):
            for j in range(i, n_antennas):
                for pol1 in ('v', 'h'):
                    for pol2 in ('v', 'h'):
                        name1 = self.product.subarray.antennas[i].name + pol1
                        name2 = self.product.subarray.antennas[j].name + pol2
                        baselines.append([name1, name2])
        baselines = np.array(baselines)
        ig.add_item(0x100C, 'bls_ordering', "The X-engine baseline output ordering. The form is a list of arrays of strings of user-defined antenna names ('input1','input2'). For example [('antC23x','antC23y'), ('antB12y','antA29y')]",
            baselines.shape, baselines.dtype, value=baselines)

        labelling = []
        input_number = 0
        for i, antenna in enumerate(self.product.subarray.antennas):
            for pol in ('v', 'h'):
                name = antenna.name + pol
                labelling.append((name, input_number, 'simulator', input_number))
                input_number += 1
        labelling = np.rec.fromrecords(labelling)
        ig.add_item(0x100E, 'input_labelling', "The physical location of each antenna's connection. It is an array of structures, each with (str,int,str,int) the following form in the case of KAT-7: (user-assigned_antenna_name, systemwide_unique_input_number, LRU, input_number_on_this_LRU). An example entry might be: ('antC23y',12,'roach030267',3)",
            labelling.shape, labelling.dtype, value=labelling)
        return ig

    def _make_ig_data(self):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        # TODO: reduce code duplication here - make a property of product?
        n_antennas = len(self.product.subarray.antennas)
        n_baselines = n_antennas * (n_antennas + 1) // 2
        n_channels = self.product.channels
        ig.add_item(0x1600, 'timestamp', 'Timestamp of start of this integration. uint counting multiples of ADC samples since last sync (sync_time, id=0x1027). Divide this number by timestamp_scale (id=0x1046) to get back to seconds since last sync when this integration was actually started. Note that the receiver will need to figure out the centre timestamp of the accumulation (eg, by adding half of int_time, id 0x1016).',
            (), None, format=self._inline_format)
        # flags_xeng_raw is still TBD in the ICD, so omitted for now
        ig.add_item(0x1800, 'xeng_raw', 'Raw data stream from all the X-engines in the system. For KAT-7, this item represents a full spectrum (all frequency channels) assembled from lowest frequency to highest frequency. Each frequency channel contains the data for all baselines (n_bls given by SPEAD Id=0x1008). Each value is a complex number - two (real and imaginary) signed integers.',
            (n_channels, n_baselines * 4, 2), np.int32)
        return ig

    @trollius.coroutine
    def send_metadata(self):
        """Reissue all the metadata on the stream."""
        heap = self._ig_static.get_heap(descriptors='all', data='all')
        yield From(self._stream.async_send_heap(heap))
        heap = self._ig_gain.get_heap(descriptors='all', data='all')
        yield From(self._stream.async_send_heap(heap))
        heap = self._ig_labels.get_heap(descriptors='all', data='all')
        yield From(self._stream.async_send_heap(heap))
        heap = self._ig_data.get_heap(descriptors='all', data='none')
        yield From(self._stream.async_send_heap(heap))

    @trollius.coroutine
    def send(self, vis, dump_index):
        assert vis.flags.c_contiguous, 'Visibility array must be contiguous'
        vis_view = vis.reshape(self._ig_data['xeng_raw'].shape)
        self._ig_data['xeng_raw'].value = vis_view
        timestamp = dump_index * self.product.n_accs
        # Truncate timestamp to the width of the field it is in
        timestamp = timestamp & ((1 << self._flavour.heap_address_bits) - 1)
        self._ig_data['timestamp'].value = timestamp
        heap = self._ig_data.get_heap()
        yield From(self._stream.async_send_heap(heap))

    @trollius.coroutine
    def close(self):
        # This is to ensure that the end packet won't be dropped for lack of
        # space in the sending buffer. In normal use it won't do anything
        # because we always asynchronously wait for transmission, but in an
        # exception case there might be pending sends.
        self._stream.flush()
        heap = self._ig_data.get_end()
        yield From(self._stream.async_send_heap(heap))


class FXStreamFileFactory(object):
    """Factory which generates instances of :class:`FXStreamFile`."""
    def __init__(self, filename):
        self._filename = filename

    @property
    def filename(self):
        return self._filename

    def __call__(self, *args, **kwargs):
        return FXStreamFile(self._filename, *args, **kwargs)


def make_coroutine(fn):
    """Decorator to turn a normal function into a coroutine."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if False:
            yield None
        raise trollius.Return(fn(*args, **kwargs))
    return wrapper


class FXStreamFile(object):
    """Writes data to HDF5 file. This is just the raw visibilities, and is not
    katdal-compatible."""
    def __init__(self, filename, product):
        self._product = product
        self._file = h5py.File(filename, 'w')
        n_antennas = len(product.subarray.antennas)
        n_baselines = n_antennas * (n_antennas + 1) // 2
        self._dataset = self._file.create_dataset('correlator_data',
            (0, product.channels, n_baselines * 4, 2), dtype=np.int32,
            maxshape=(None, product.channels, n_baselines * 4, 2))
        self._flags = None

    @trollius.coroutine
    @make_coroutine
    def send_metadata(self):
        pass

    @trollius.coroutine
    @make_coroutine
    def send(self, vis, dump_index):
        vis = vis.reshape((vis.shape[0], vis.shape[1] * 4, 2))
        self._dataset.resize(dump_index + 1, axis=0)
        self._dataset[dump_index : dump_index + 1, ...] = vis[np.newaxis, ...]

    @trollius.coroutine
    @make_coroutine
    def close(self):
        self._file.close()
