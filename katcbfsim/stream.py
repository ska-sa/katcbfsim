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


class SpeadStream(object):
    """Base class for SPEAD streams, providing a factory function."""
    @classmethod
    def factory(cls, endpoints):
        return functools.partial(cls, endpoints)

    @property
    def n_streams(self):
        return len(self._streams)

    def __init__(self, endpoints, product, in_rate):
        if not endpoints:
            raise ValueError('At least one endpoint is required')
        n = len(endpoints)
        if product.n_channels % n:
            raise ValueError('Number of channels not divisible by number of streams')
        self.endpoints = endpoints
        self.product = product
        self._flavour = spead2.Flavour(4, 64, 48, 0)
        self._inline_format = [('u', self._flavour.heap_address_bits)]
        # Send at a slightly higher rate, to account for overheads, and so
        # that if the sender sends a burst we can catch up with it.
        out_rate = in_rate * 1.05 / n
        config = spead2.send.StreamConfig(rate=out_rate, max_packet_size=9172)
        self._streams = [
            spead2.send.trollius.UdpStream(
                spead2.ThreadPool(), e.host, e.port, config)
            for e in endpoints]

    @trollius.coroutine
    def close(self):
        # This is to ensure that the end packet won't be dropped for lack of
        # space in the sending buffer. In normal use it won't do anything
        # because we always asynchronously wait for transmission, but in an
        # exception case there might be pending sends.
        for i, stream in enumerate(self._streams):
            heap = self._ig_data[i].get_end()
            yield From(stream.async_flush())
            yield From(stream.async_send_heap(heap))


class CBFSpeadStream(SpeadStream):
    """Common base class for correlator and beamformer streams, with utilities
    for shared items.
    """
    def __init__(self, *args, **kwargs):
        super(CBFSpeadStream, self).__init__(*args, **kwargs)
        # Hard-coded values assumed from the ICD or the real CBF
        self.fft_shift_pattern = 32767
        self.xeng_acc_len = 256
        self.requant_bits = 8
        self.adc_bits = 10
        # CBF apparently use this, even though it wraps pretty quickly
        self.scale_factor_timestamp = self.product.adc_rate

    def add_adc_sample_rate_item(self, ig):
        ig.add_item(0x1007, 'adc_sample_rate', 'Expected ADC sample rate (sampled/second)',
            (), format=[('u', 64)], value=self.product.adc_rate)

    def add_n_chans_item(self, ig):
        ig.add_item(0x1009, 'n_chans', 'The total number of frequency channels present in any integration.',
            (), format=self._inline_format, value=self.product.n_channels // self.n_streams)

    def add_n_ants_item(self, ig):
        ig.add_item(0x100A, 'n_ants', 'The total number of dual-pol antennas in the system.',
            (), format=self._inline_format, value=self.product.n_antennas)

    def add_center_freq_item(self, ig, stream_idx):
        # Find per-stream center frequency
        center_frequency = (
            self.product.center_frequency
            + self.product.bandwidth * (2 * stream_idx + 1 - self.n_streams) / (2 * self.n_streams))
        # Convert sky frequency to DBE frequency
        digitised_bandwidth = self.product.adc_rate // 2
        center_frequency = self.product.center_frequency % digitised_bandwidth
        ig.add_item(0x1011, 'center_freq', 'The center frequency of the DBE in Hz, 64-bit IEEE floating-point number.',
            (), format=[('f', 64)], value=np.float64(center_frequency))

    def add_bandwidth_item(self, ig):
        ig.add_item(0x1013, 'bandwidth', 'The analogue bandwidth of the digitally processed signal in Hz.',
            (), format=[('f', 64)], value=np.float64(self.product.bandwidth / self.n_streams))

    def add_fft_shift_item(self, ig):
        ig.add_item(0x101E, 'fft_shift', 'The FFT bitshift pattern. F-engine correlator internals.',
            (), format=self._inline_format, value=self.fft_shift_pattern)

    def add_xeng_acc_len_item(self, ig):
        ig.add_item(0x101F, 'xeng_acc_len', 'Number of spectra accumulated inside X engine. Determines minimum integration time and user-configurable integration time step-size. X-engine correlator internals',
            (), format=self._inline_format, value=self.xeng_acc_len)

    def add_requant_bits_item(self, ig):
        ig.add_item(0x1020, 'requant_bits', 'Number of bits per sample after requantisation. For FX correlators, this represents the number of bits after requantisation in the F engines (post FFT and any phasing stages) and is the actual number of bits used in X-engine processing. For time-domain systems, this is requantisation in the time domain before any subsequent processing.',
            (), format=self._inline_format, value=self.requant_bits)

    def add_rx_udp_items(self, ig, stream_idx):
        ig.add_item(0x1022, 'rx_udp_port', 'Destination UDP port for data output.',
            (), format=self._inline_format, value=self.endpoints[stream_idx].port)
        # TODO: this might need to be translated from hostname to IP address
        ig.add_item(0x1024, 'rx_udp_ip_str', 'Destination IP address for output UDP packets.',
            (None,), format=[('c', 8)], value=self.endpoints[stream_idx].host)

    def add_sync_time_item(self, ig):
        ig.add_item(0x1027, 'sync_time', 'Time at which the system was last synchronised (armed and triggered by a 1PPS) in seconds since the Unix Epoch.',
            (), format=self._inline_format, value=self.product.subarray.sync_time.secs)

    def add_adc_bits_item(self, ig):
        ig.add_item(0x1045, 'adc_bits', 'ADC resolution (number of bits).',
            (), format=self._inline_format, value=self.adc_bits)

    def add_scale_factor_timestamp_item(self, ig):
        # TODO: what scaling factor should we use?
        ig.add_item(0x1046, 'scale_factor_timestamp', 'Timestamp scaling factor. Divide the SPEAD data packet timestamp by this number to get back to seconds since last sync.',
            (), format=[('f', 64)], value=self.scale_factor_timestamp)

    def add_eq_coef_items(self, ig):
        initial_gain = np.zeros((self.product.n_channels // self.n_streams, 2), np.uint32)
        initial_gain[:, 0].fill(200)    # Arbitrary value for now (200 + 0j)
        input_number = 0
        for antenna in self.product.subarray.antennas:
            for pol in ('v', 'h'):
                ig.add_item(0x1400 + input_number, 'eq_coef_{}{}'.format(antenna.name, pol), 'The unitless, per-channel, digital scaling factors implemented prior to requantisation, for inputN. Complex number (real,imag) 32 bit integers.',
                    initial_gain.shape, initial_gain.dtype, value=initial_gain)
                input_number += 1

    def add_input_labelling_item(self, ig):
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

    def add_timestamp_item(self, ig):
        ig.add_item(0x1600, 'timestamp', 'Timestamp of start of this integration. uint counting multiples of ADC samples since last sync (sync_time, id=0x1027). Divide this number by timestamp_scale (id=0x1046) to get back to seconds since last sync when this integration was actually started. Note that the receiver will need to figure out the centre timestamp of the accumulation (eg, by adding half of int_time, id 0x1016).',
            (), None, format=self._inline_format)


class FXStreamSpead(CBFSpeadStream):
    """Data stream from an FX correlator, sent over SPEAD."""
    def __init__(self, endpoints, product):
        in_rate = product.n_baselines * product.n_channels * 4 * 8 / \
            product.accumulation_length
        super(FXStreamSpead, self).__init__(endpoints, product, in_rate)
        self._ig_static = [self._make_ig_static(i) for i in range(self.n_streams)]
        self._ig_gain = self._make_ig_gain()
        self._ig_labels = self._make_ig_labels()
        self._ig_data = [self._make_ig_data() for i in range(self.n_streams)]

    def _make_ig_static(self, stream_idx):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        self.add_adc_sample_rate_item(ig)
        ig.add_item(0x1008, 'n_bls', 'The total number of baselines in the data product. Each pair of inputs (polarisation pairs) is considered a baseline.',
            (), format=self._inline_format, value=self.product.n_baselines * 4)
        self.add_n_chans_item(ig)
        self.add_n_ants_item(ig)
        ig.add_item(0x100B, 'n_xengs', 'The total number of X engines in a correlator system.',
            (), format=self._inline_format, value=1)
        self.add_center_freq_item(ig, stream_idx)
        self.add_bandwidth_item(ig)
        ig.add_item(0x1015, 'n_accs', 'The number of spectra that are accumulated per integration.',
            (), format=self._inline_format, value=self.product.n_accs)
        ig.add_item(0x1016, 'int_time', "Approximate (it's a float!) time per accumulation in seconds. This is intended for reference only. Each accumulation has an associated timestamp which should be used to determine the time of the integration rather than incrementing the start time by this value for sequential integrations (which would allow errors to grow).",
            (), format=[('f', 64)], value=self.product.accumulation_length)
        self.add_fft_shift_item(ig)
        self.add_xeng_acc_len_item(ig)
        self.add_requant_bits_item(ig)
        self.add_rx_udp_items(ig, stream_idx)
        self.add_sync_time_item(ig)
        self.add_adc_bits_item(ig)
        self.add_scale_factor_timestamp_item(ig)
        ig.add_item(0x1048, 'xeng_out_bits_per_sample', 'The number of bits per value of the xeng accumulator output. Note this is for a single component value, not the combined complex size.',
            (), format=self._inline_format, value=32)
        # TODO: missing the following (shouldn't be needed by anything?)
        # - feng_udp_port
        # - eng_rate
        # - x_per_fpga
        # - ddc_mix_freq
        # - f_per_fpga
        return ig

    def _make_ig_gain(self):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        self.add_eq_coef_items(ig)
        return ig

    def _make_ig_labels(self):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        baselines = []
        for i in range(self.product.n_antennas):
            for j in range(i, self.product.n_antennas):
                for pol1 in ('v', 'h'):
                    for pol2 in ('v', 'h'):
                        name1 = self.product.subarray.antennas[i].name + pol1
                        name2 = self.product.subarray.antennas[j].name + pol2
                        baselines.append([name1, name2])
        baselines = np.array(baselines)
        ig.add_item(0x100C, 'bls_ordering', "The X-engine baseline output ordering. The form is a list of arrays of strings of user-defined antenna names ('input1','input2'). For example [('antC23x','antC23y'), ('antB12y','antA29y')]",
            baselines.shape, baselines.dtype, value=baselines)

        self.add_input_labelling_item(ig)
        return ig

    def _make_ig_data(self):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        self.add_timestamp_item(ig)
        # flags_xeng_raw is still TBD in the ICD, so omitted for now
        ig.add_item(0x1800, 'xeng_raw', 'Raw data stream from all the X-engines in the system. For KAT-7, this item represents a full spectrum (all frequency channels) assembled from lowest frequency to highest frequency. Each frequency channel contains the data for all baselines (n_bls given by SPEAD Id=0x1008). Each value is a complex number - two (real and imaginary) signed integers.',
            (self.product.n_channels // self.n_streams, self.product.n_baselines * 4, 2), np.int32)
        return ig

    @trollius.coroutine
    def _send_metadata_stream(self, stream_idx):
        stream = self._streams[stream_idx]
        heap = self._ig_static[stream_idx].get_heap(descriptors='all', data='all')
        yield From(stream.async_send_heap(heap))
        heap = self._ig_gain.get_heap(descriptors='all', data='all')
        yield From(stream.async_send_heap(heap))
        heap = self._ig_labels.get_heap(descriptors='all', data='all')
        yield From(stream.async_send_heap(heap))
        heap = self._ig_data[stream_idx].get_heap(descriptors='all', data='none')
        yield From(stream.async_send_heap(heap))

    @trollius.coroutine
    def send_metadata(self):
        """Reissue all the metadata on the streams."""
        futures = []
        # Send on all streams in parallel
        for i in range(self.n_streams):
            futures.append(trollius.async(self._send_metadata_stream(i)))
        for future in futures:
            yield From(future)

    @trollius.coroutine
    def send(self, vis, dump_index):
        assert vis.flags.c_contiguous, 'Visibility array must be contiguous'
        shape = (self.product.n_channels, self.product.n_baselines * 4, 2)
        stream_channels = self.product.n_channels // self.n_streams
        vis_view = vis.reshape(self.product.n_channels, self.product.n_baselines * 4, 2)
        futures = []
        timestamp = dump_index * self.product.n_accs * self.product.n_channels * self.scale_factor_timestamp // self.product.bandwidth
        # Truncate timestamp to the width of the field it is in
        timestamp = timestamp & ((1 << self._flavour.heap_address_bits) - 1)
        for i, stream in enumerate(self._streams):
            channel0 = stream_channels * i
            channel1 = channel0 + stream_channels
            self._ig_data[i]['xeng_raw'].value = vis_view[channel0:channel1]
            self._ig_data[i]['timestamp'].value = timestamp
            heap = self._ig_data[i].get_heap()
            futures.append(trollius.async(stream.async_send_heap(heap)))
        for future in futures:
            yield From(future)


class FileStream(object):
    """Base class for file-sink streams, providing a factory function."""
    @classmethod
    def factory(cls, filename):
        return functools.partial(cls, filename)

    def __init__(self, filename, product):
        self._product = product
        self._file = h5py.File(filename, 'w')

    @trollius.coroutine
    def close(self):
        self._file.close()


class FXStreamFile(FileStream):
    """Writes data to HDF5 file. This is just the raw visibilities, and is not
    katdal-compatible."""
    def __init__(self, filename, product):
        super(FXStreamFile, self).__init__(filename, product)
        self._dataset = self._file.create_dataset('correlator_data',
            (0, product.n_channels, product.n_baselines * 4, 2), dtype=np.int32,
            maxshape=(None, product.n_channels, product.n_baselines * 4, 2))
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

class BeamformerStreamSpead(CBFSpeadStream):
    """Data stream from a beamformer, sent over SPEAD."""
    def __init__(self, endpoints, product):
        if product.wall_interval == 0:
            in_rate = 0
        else:
            in_rate = product.n_channels * product.timesteps * 2 * product.sample_bits / product.wall_interval / 8
        super(BeamformerStreamSpead, self).__init__(endpoints, product, in_rate)
        self.xeng_acc_len = self.product.timesteps
        self._ig_static = [self._make_ig_static(i) for i in range(self.n_streams)]
        self._ig_weights = self._make_ig_weights()
        self._ig_labels = self._make_ig_labels()
        self._ig_timing = self._make_ig_timing()
        self._ig_data = [self._make_ig_data() for i in range(self.n_streams)]

    def _make_ig_static(self, stream_idx):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        self.add_adc_sample_rate_item(ig)
        self.add_n_chans_item(ig)
        self.add_n_ants_item(ig)
        ig.add_item(0x100F, 'n_bengs', 'The total number of B engines in a beamformer system.',
            (), format=self._inline_format, value=1)
        self.add_center_freq_item(ig, stream_idx)
        self.add_bandwidth_item(ig)
        self.add_xeng_acc_len_item(ig)
        self.add_requant_bits_item(ig)
        self.add_fft_shift_item(ig)
        # ICD strongly implies that feng_pkt_len is the same as xeng_acc_len
        ig.add_item(0x1021, 'feng_pkt_len', 'Payload size of packet exchange between F and X engines in 64 bit words. Usually equal to the number of spectra accumulated inside X engine. F-engine correlator internals.',
            (), format=self._inline_format, value=self.xeng_acc_len)
        # ddc_mix_freq omitted, since the correct value isn't known
        # ig.add_item(0x1043, 'ddc_mix_freq', 'Digital downconverter mixing frequency as a fraction of the ADC sampling frequency. eg: 0.25. Set to zero if no DDC is present.',
        #     (), format=[('f', 64)], value=0.0)
        self.add_adc_bits_item(ig)
        self.add_rx_udp_items(ig, stream_idx)
        ig.add_item(0x1050, 'beng_out_bits_per_sample', 'The number of bits per value of the beng accumulator output. Note this is for a single component value, not the combined complex size.',
            (), format=self._inline_format, value=self.product.sample_bits)
        return ig

    def _make_ig_weights(self):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        self.add_eq_coef_items(ig)
        beamweight = np.zeros((2 * self.product.n_antennas,), np.int32)
        ig.add_item(0x2000, 'beamweight', 'The unitless digital scaling factors implemented prior to combining signals for this beam. See 0x100E (input_labelling) to get mapping from inputN to user defined input string.',
            beamweight.shape, beamweight.dtype, value=beamweight)
        return ig

    def _make_ig_labels(self):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        self.add_input_labelling_item(ig)
        return ig

    def _make_ig_timing(self):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        self.add_scale_factor_timestamp_item(ig)
        self.add_sync_time_item(ig)
        return ig

    def _make_ig_data(self):
        ig = spead2.send.ItemGroup(flavour=self._flavour)
        self.add_timestamp_item(ig)
        # frequency is omitted because we have no way to set a per-packet value. It's
        # useless anyway.
        # bf_raw is not in v3 of the ICD, but this is according to MKAT-ECP-157
        ig.add_item(0x5000, 'bf_raw', 'Beamformer output for frequency-domain beam. User-defined name (out of band control). Record length depending on number of frequency channels and F-X packet size (xeng_acc_len).',
            shape=(self.product.n_channels // self.n_streams, self.product.timesteps, 2), dtype=self.product.dtype)
        return ig

    @trollius.coroutine
    def _send_metadata_stream(self, stream_idx):
        """Reissue all the metadata on the stream."""
        stream = self._streams[stream_idx]
        heap = self._ig_timing.get_heap(descriptors='all', data='all')
        yield From(stream.async_send_heap(heap))
        heap = self._ig_weights.get_heap(descriptors='all', data='all')
        yield From(stream.async_send_heap(heap))
        heap = self._ig_static[stream_idx].get_heap(descriptors='all', data='all')
        yield From(stream.async_send_heap(heap))
        heap = self._ig_labels.get_heap(descriptors='all', data='all')
        yield From(stream.async_send_heap(heap))
        heap = self._ig_data[stream_idx].get_heap(descriptors='all', data='none')
        yield From(stream.async_send_heap(heap))

    @trollius.coroutine
    def send_metadata(self):
        """Reissue all the metadata on the streams."""
        futures = []
        # Send on all streams in parallel
        for i in range(self.n_streams):
            futures.append(trollius.async(self._send_metadata_stream(i)))
        for future in futures:
            yield From(future)

    @trollius.coroutine
    def send(self, beam_data, index):
        stream_channels = self.product.n_channels // self.n_streams
        timestamp = index * self.product.timesteps * self.product.n_channels * self.scale_factor_timestamp // self.product.bandwidth
        # Truncate timestamp to the width of the field it is in
        timestamp = timestamp & ((1 << self._flavour.heap_address_bits) - 1)
        futures = []
        for i, stream in enumerate(self._streams):
            channel0 = stream_channels * i
            channel1 = channel0 + stream_channels
            self._ig_data[i]['bf_raw'].value = beam_data[channel0:channel1]
            self._ig_data[i]['timestamp'].value = timestamp
            heap = self._ig_data[i].get_heap()
            futures.append(stream.async_send_heap(heap))
        for future in futures:
            yield From(future)
