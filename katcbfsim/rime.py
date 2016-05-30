"""GPU implementation the Radio Interferometry Measurement Equation"""

from __future__ import print_function, division
import pkg_resources
import numpy as np
import katpoint
import logging
import scipy.special
from katsdpsigproc import accel, tune


logger = logging.getLogger(__name__)


class RimeTemplate(object):
    autotune_version = 3

    def __init__(self, context, max_antennas, tuning=None):
        self.context = context
        self.max_antennas = max_antennas
        if tuning is None:
            tuning = self.autotune(context, max_antennas)
        self.predict_wgs = tuning['predict_wgs']
        self.predict_program = accel.build(context, 'predict.mako',
            {'max_antennas': max_antennas},
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])
        self.sample_wgs = tuning['sample_wgs']
        self.sample_rows = tuning['sample_rows']
        self.sample_program = accel.build(context, 'sample.mako',
            {},
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    @classmethod
    @tune.autotuner(test={'predict_wgs': 64, 'sample_wgs': 256, 'sample_rows': 64})
    def autotune(cls, context, max_antennas):
        queue = context.create_tuning_command_queue()
        n_antennas = max_antennas
        n_baselines = n_antennas * (n_antennas + 1) // 2
        n_channels = 2048
        n_sources = 32
        n_accs = 1024
        sefd = 20.0
        sources = [katpoint.construct_radec_target(0, 0)] * n_sources
        antennas = [katpoint.Antenna('Antenna {}'.format(i),
                                     '-34:00:00', '20:00:00', '0.0', 10.0, (i, 0, 0))
                    for i in range(n_antennas)]
        # The values are irrelevant, we just need the memory to be there.
        out = accel.DeviceArray(context, (n_channels, n_baselines, 2, 2, 2), np.int32)
        gain = accel.DeviceArray(context, (n_channels, n_antennas, 2, 2), np.complex64)
        def generate_predict(predict_wgs):
            tuning = dict(predict_wgs=predict_wgs, sample_wgs=256, sample_rows=64)
            fn = cls(context, max_antennas, tuning).instantiate(
                queue, 1284000000.0, 856000000.0, n_channels, n_accs, sources, antennas, sefd)
            fn.bind(out=out, gain=gain)
            fn._update_scaled_phase()
            queue.finish()  # _update_scaled_phase is asynchronous
            return tune.make_measure(queue, fn._run_predict)
        def generate_sample(sample_wgs, sample_rows):
            tuning = dict(predict_wgs=64, sample_wgs=sample_wgs, sample_rows=sample_rows)
            fn = cls(context, max_antennas, tuning).instantiate(
                queue, 1284000000.0, 856000000.0, n_channels, n_accs, sources, antennas, sefd)
            fn.bind(out=out, gain=gain)
            fn._update_scaled_phase()
            fn._run_predict()
            queue.finish()  # _update_scaled_phase is asynchronous
            return tune.make_measure(queue, fn._run_sample)
        wgss = [x for x in [64, 128, 256, 512, 1024] if x >= max_antennas]
        tuning = {}
        tuning.update(tune.autotune(generate_predict, predict_wgs=wgss))
        tuning.update(tune.autotune(generate_sample, sample_wgs=wgss,
            sample_rows=[16, 64, 256]))
        return tuning

    def instantiate(self, *args, **kwargs):
        return Rime(self, *args, **kwargs)


class Rime(accel.Operation):
    def __init__(
            self, template, command_queue,
            center_frequency, bandwidth, n_channels, n_accs,
            sources, antennas, sefd, seed=None, async=False, allocator=None):
        if len(antennas) > template.max_antennas:
            raise ValueError('Too many antennas for the template')
        super(Rime, self).__init__(command_queue, allocator)
        self.template = template
        self.n_channels = n_channels
        self.n_accs = n_accs
        self.sources = sources
        self.antennas = antennas
        self.sefd = sefd
        self.seed = seed if seed is not None else np.random.randint(0, 2**63 - 1)
        self._sequence = 0
        self.async = async
        n_antennas = len(antennas)
        n_sources = len(sources)
        n_baselines = n_antennas * (n_antennas + 1) // 2
        n_padded_baselines = max(
            accel.roundup(n_baselines, template.predict_wgs),
            accel.roundup(n_baselines, template.sample_wgs))
        self.time = katpoint.Timestamp()
        self.phase_center = katpoint.construct_radec_target(0, 0)
        self.position = None
        _2 = accel.Dimension(2, exact=True)
        self.slots['out'] = accel.IOSlot((n_channels, n_baselines, _2, _2, _2), np.int32)
        self.slots['gain'] = accel.IOSlot((n_channels, n_antennas, _2, _2), np.complex64)
        self.predict_kernel = template.predict_program.get_kernel('predict')
        self.sample_kernel = template.sample_program.get_kernel('sample')
        # Set up the inverse wavelength lookup table
        self._inv_wavelength = accel.DeviceArray(command_queue.context, (n_channels,), np.float32)
        channel_width = bandwidth / n_channels
        min_frequency = center_frequency - bandwidth / 2
        # Note: min_frequency is the centre of the first bin, not the bottom
        # end.
        self.frequencies = np.arange(n_channels) * channel_width + min_frequency
        logging.info("Frequencies %s", self.frequencies)
        inv_wavelength = self._inv_wavelength.empty_like()
        inv_wavelength[:] = (self.frequencies / katpoint.lightspeed).astype(np.float32)
        self._inv_wavelength.set(command_queue, inv_wavelength)
        # Set up internal arrays
        self._scaled_phase = accel.DeviceArray(command_queue.context, (n_sources, n_antennas), np.float32)
        self._scaled_phase_host = self._scaled_phase.empty_like()
        self._flux_models = np.empty((n_channels, n_sources), np.float32)
        self._flux_density = accel.DeviceArray(command_queue.context, (n_channels, n_sources, 2, 2), np.complex64)
        self._flux_density_host = self._flux_density.empty_like()
        self._flux_sum = accel.DeviceArray(command_queue.context, (n_channels, 2), np.float32)
        self._flux_sum_host = self._flux_sum.empty_like()
        # Set up the internal baseline mapping
        # TODO: this could live in the template
        self._baselines = accel.DeviceArray(command_queue.context, (n_padded_baselines, 2), np.int16)
        baselines_host = self._baselines.empty_like()
        next_baseline = 0
        baselines_host.fill(0)
        for i in range(n_antennas):
            for j in range(i, n_antennas):
                baselines_host[next_baseline, 0] = i
                baselines_host[next_baseline, 1] = j
                next_baseline += 1
        self._baselines.set(command_queue, baselines_host)
        self._sample_flux_models()
        self.command_queue.finish()  # _update_flush_density is asynchronous

    def set_time(self, time):
        """Set the time for which the visibilities are simulated.

        Parameters
        ----------
        time : :class:`katpoint.Timestamp`
            Time
        """
        self.time = time

    def set_phase_center(self, direction):
        """Set the direction of the phase center for the simulation.

        Parameters
        ----------
        direction : :class:`katpoint.Target`
        """
        self.phase_center = direction

    def set_position(self, direction):
        """Set the dish pointing direction for the simulation. If this is never
        set, it defaults to the phase center.
        """
        self.position = direction

    def _sample_flux_models(self):
        """Sample the flux models of the sources."""
        logger.debug('Starting _sample_flux_models')
        for channel, freq in enumerate(self.frequencies):
            freq_MHz = freq / 1e6  # katpoint takes freq in MHz
            for i, source in enumerate(self.sources):
                if source.flux_model is None:
                    fd = 1.0
                else:
                    fd = source.flux_model.flux_density(freq_MHz)
                    # Assume zero emission outside the defined frequency range.
                    if np.isnan(fd):
                        fd = 0.0
                self._flux_models[channel, i] = fd
        logger.debug('Flux models updated')

    def _update_flux_density(self):
        """Set the per-channel perceived flux density from the flux models of
        the sources and a simple beam model.

        This performs an **asynchronous** transfer to the GPU, and the caller
        must wait for it to complete before calling the function again.
        """
        logger.debug('Starting _update_flex_density')
        self._flux_sum_host.fill(self.sefd)
        position = self.position or self.phase_center

        # Determine scaling factor for Airy function
        # 1.029 is the beam width factor for an ideal Airy disk
        airy_scale = np.pi * self.antennas[0].diameter / katpoint.lightspeed * (1.029 / self.antennas[0].beamwidth)
        angles = np.empty((len(self.sources),), np.float64)
        # Find angles between sources and the pointing direction
        for i, source in enumerate(self.sources):
            # Evaluate Airy beam model. It uses azel internally, so it needs an antenna.
            angles[i] = source.separation(position, timestamp=self.time, antenna=self.antennas[0])
            # There is a pole at 0, so adjust it away if it is zero
            if abs(angles[i]) < 1e-30:
                angles[i] = 1e-30

        for channel, freq in enumerate(self.frequencies):
            x = np.sin(angles) * airy_scale * freq
            beam = (2 * scipy.special.j1(x) / x)**2
            fd = self._flux_models[channel, :] * beam
            self._flux_density_host[channel, :, 0, 0] = fd
            self._flux_density_host[channel, :, 0, 1] = 0.0
            self._flux_density_host[channel, :, 1, 0] = 0.0
            self._flux_density_host[channel, :, 1, 1] = fd
            fd_sum = fd.sum()
            self._flux_sum_host[channel, 0] += fd_sum
            self._flux_sum_host[channel, 1] += fd_sum
        logger.debug('Host flux densities updated')
        self._flux_density.set_async(self.command_queue, self._flux_density_host)
        self._flux_sum.set_async(self.command_queue, self._flux_sum_host)

    def _update_scaled_phase(self):
        """Compute the propagation delay phase for each source and antenna.

        This performs an **asynchronous** transfer to the GPU, and the caller
        must wait for it to complete before calling the function again.
        """
        # Create a reference antenna from the reference position of the first
        # antenna. If all the antennas use the same reference point, this
        # allows for a faster path through katpoint.Target.uvw.
        ref_antenna = katpoint.Antenna('', *self.antennas[0].ref_position_wgs84)
        u, v, w = zip(*[self.phase_center.uvw(antenna, self.time, ref_antenna)
                        for antenna in self.antennas])
        ra, dec = zip(*[source.radec(self.time, ref_antenna) for source in self.sources])
        ra = np.array(ra)
        dec = np.array(dec)
        l, m, n = self.phase_center.lmn(ra, dec, self.time, ref_antenna)
        self._scaled_phase_host[:] = -2 * (np.outer(l, u) + np.outer(m, v) + np.outer(n - 1, w))
        self._scaled_phase.set_async(self.command_queue, self._scaled_phase_host)

    def _run_predict(self):
        out = self.buffer('out')
        n_antennas = len(self.antennas)
        n_sources = len(self.sources)
        n_baselines = n_antennas * (n_antennas + 1) // 2
        self.command_queue.enqueue_kernel(
            self.predict_kernel,
            [
                out.buffer,
                np.int32(out.padded_shape[1]),
                self._flux_density.buffer,
                np.int32(self._flux_density.padded_shape[1]),
                self._inv_wavelength.buffer,
                self._scaled_phase.buffer,
                self._baselines.buffer,
                np.float32(self.sefd),
                np.int32(n_sources),
                np.int32(n_antennas),
                np.int32(n_baselines),
            ],
            global_size=(accel.roundup(n_baselines, self.template.predict_wgs), self.n_channels),
            local_size=(self.template.predict_wgs, 1)
        )

    def _run_sample(self):
        out = self.buffer('out')
        gain = self.buffer('gain')
        n_antennas = len(self.antennas)
        n_baselines = n_antennas * (n_antennas + 1) // 2
        self.command_queue.enqueue_kernel(
            self.sample_kernel,
            [
                out.buffer,
                np.int32(out.padded_shape[1]),
                self._flux_sum.buffer,
                gain.buffer,
                np.int32(gain.padded_shape[1]),
                self._baselines.buffer,
                np.int32(self.n_channels),
                np.int32(n_baselines),
                np.float32(self.n_accs),
                np.uint64(self.seed),
                np.uint64(self._sequence)
            ],
            global_size=(accel.roundup(n_baselines, self.template.sample_wgs),
                         min(self.template.sample_rows, self.n_channels)),
            local_size=(self.template.sample_wgs, 1)
        )
        self._sequence += 1

    def _run(self):
        logger.debug('Updating scaled_phase')
        self._update_scaled_phase()
        logger.debug('Updating perceived flux densities')
        self._update_flux_density()
        transfer_event = self.command_queue.enqueue_marker()
        logger.debug('Marker enqueued')

        self._run_predict()
        self._run_sample()
        logger.debug('Kernels queued')
        if self.async:
            return transfer_event
        else:
            # Make sure that the host->device transfers have completed, so that we
            # are ready for another call to this function.
            transfer_event.wait()
