"""GPU implementation the Radio Interferometry Measurement Equation"""

from __future__ import print_function, division
import pkg_resources
import numpy as np
import katpoint
import logging
from katsdpsigproc import accel, tune


logger = logging.getLogger(__name__)


class RimeTemplate(object):
    def __init__(self, context, max_antennas, tuning=None):
        self.context = context
        self.max_antennas = max_antennas
        if tuning is None:
            tuning = self.autotune(context, max_antennas)
        self.wgs = tuning['wgs']
        self.program = accel.build(context, 'rime.mako',
            {'max_antennas': max_antennas},
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    @classmethod
    @tune.autotuner(test={'wgs': 64})
    def autotune(cls, context, max_antennas):
        queue = context.create_tuning_command_queue()
        n_antennas = max_antennas
        n_baselines = n_antennas * (n_antennas + 1) // 2
        n_channels = 2048
        n_sources = 32
        sources = [katpoint.construct_radec_target(0, 0)] * n_sources
        antennas = [katpoint.Antenna('Antenna {}'.format(i),
                                     '-34:00:00', '20:00:00', '0.0', 10.0, (i, 0, 0))
                    for i in range(n_antennas)]
        # The values are irrelevant, we just need the memory to be there.
        out = accel.DeviceArray(context, (n_channels, n_baselines, 2, 2, 2), np.int32)
        gain = accel.DeviceArray(context, (n_channels, n_antennas, 2, 2), np.complex64)
        def generate(wgs):
            fn = cls(context, max_antennas, dict(wgs=wgs)).instantiate(
                queue, 1412000000.0, 856000000.0, n_channels, sources, antennas)
            fn.bind(out=out, gain=gain)
            return tune.make_measure(queue, fn)
        wgss = [x for x in [32, 64, 128, 256, 512, 1024] if x >= max_antennas]
        return tune.autotune(generate, wgs=wgss)

    def instantiate(self, *args, **kwargs):
        return Rime(self, *args, **kwargs)


class Rime(accel.Operation):
    def __init__(
            self, template, command_queue,
            center_frequency, bandwidth, n_channels,
            sources, antennas, async=False, allocator=None):
        if len(antennas) > template.max_antennas:
            raise ValueError('Too many antennas for the template')
        super(Rime, self).__init__(command_queue, allocator)
        self.template = template
        self.n_channels = n_channels
        self.sources = sources
        self.antennas = antennas
        self.async = async
        n_antennas = len(antennas)
        n_sources = len(sources)
        n_baselines = n_antennas * (n_antennas + 1) // 2
        n_padded_baselines = accel.roundup(n_baselines, template.wgs)
        self.time = katpoint.Timestamp()
        self.phase_center = katpoint.construct_radec_target(0, 0)
        _2 = accel.Dimension(2, exact=True)
        self.slots['out'] = accel.IOSlot((n_channels, n_baselines, _2, _2, _2), np.int32)
        self.slots['gain'] = accel.IOSlot((n_channels, n_antennas, _2, _2), np.complex64)
        self.kernel = template.program.get_kernel('predict')
        # Set up the inverse wavelength lookup table
        self._inv_wavelength = accel.DeviceArray(command_queue.context, (n_channels,), np.float32)
        channel_width = bandwidth / n_channels
        min_frequency = center_frequency - bandwidth / 2
        self.frequencies = np.arange(0.5, n_channels) * channel_width + min_frequency
        inv_wavelength = self._inv_wavelength.empty_like()
        inv_wavelength[:] = (self.frequencies / katpoint.lightspeed).astype(np.float32)
        self._inv_wavelength.set(command_queue, inv_wavelength)
        # Set up internal arrays
        self._scaled_phase = accel.DeviceArray(command_queue.context, (n_sources, n_antennas), np.float32)
        self._scaled_phase_host = self._scaled_phase.empty_like()
        self._flux_density = accel.DeviceArray(command_queue.context, (n_channels, n_sources, 2, 2), np.complex64)
        self._flux_density_host = self._flux_density.empty_like()
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

    def set_time(self, time):
        """Set the time for which the visibilities are simulated.

        Parameters
        ----------
        time : :class:`katpoint.Timestamp`
            Time
        """
        self.time = time

    def set_phase_center(self, direction):
        """Set the direction of the phase center for the simulation

        Parameters
        ----------
        direction : :class:`katpoint.Target`
        """
        self.phase_center = direction

    def _update_flux_density(self):
        """Set the per-channel flux density from the flux models of the
        sources.

        This performs an **asynchronous** transfer to the GPU, and the caller
        must wait for it to complete before calling the function again.
        """
        logger.debug('Starting update_flex_density')
        for channel, freq in enumerate(self.frequencies):
            freq_Mhz = freq / 1e6  # katpoint takes freq in MHz
            for i, source in enumerate(self.sources):
                if source.flux_model is None:
                    fd = 1.0
                else:
                    fd = source.flux_model.flux_density(freq_MHz)
                    # Assume zero emission outside the defined frequency range.
                    if np.isnan(fd):
                        fd = 0.0
                # katpoint currently doesn't model polarised sources, so
                # set up a diagonal brightness matrix
                self._flux_density_host[channel, i, 0, 0] = fd
                self._flux_density_host[channel, i, 0, 1] = 0.0
                self._flux_density_host[channel, i, 1, 0] = 0.0
                self._flux_density_host[channel, i, 1, 1] = fd
        logger.debug('Host flux densities updated')
        self._flux_density.set_async(self.command_queue, self._flux_density_host)

    def _update_scaled_phase(self):
        """Compute the propagation delay phase for each source and antenna.

        This performs an **asynchronous** transfer to the GPU, and the caller
        must wait for it to complete before calling the function again.
        """
        ref_antenna = self.antennas[0]
        u, v, w = zip(*[self.phase_center.uvw(antenna, self.time, ref_antenna)
                        for antenna in self.antennas])
        ra, dec = zip(*[source.radec(self.time, ref_antenna) for source in self.sources])
        ra = np.array(ra)
        dec = np.array(dec)
        l, m = self.phase_center.sphere_to_plane(ra, dec, self.time, ref_antenna, 'SIN', 'radec')
        n = np.sqrt(1 - l**2 - m**2)
        self._scaled_phase_host[:] = -2 * (np.outer(l, u) + np.outer(m, v) + np.outer(n - 1, w))
        self._scaled_phase.set_async(self.command_queue, self._scaled_phase_host)

    def _run(self):
        logger.debug('Updating flux_density')
        self._update_flux_density()
        logger.debug('Updating scaled_phase')
        self._update_scaled_phase()
        transfer_event = self.command_queue.enqueue_marker()
        logger.debug('Marker enqueued')

        # Locate the buffers
        out = self.buffer('out')
        gain = self.buffer('gain')
        n_antennas = len(self.antennas)
        n_sources = len(self.sources)
        n_baselines = n_antennas * (n_antennas + 1) // 2
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                out.buffer,
                np.int32(out.padded_shape[1]),
                self._flux_density.buffer,
                np.int32(self._flux_density.padded_shape[1]),
                gain.buffer,
                np.int32(gain.padded_shape[1]),
                self._inv_wavelength.buffer,
                self._scaled_phase.buffer,
                self._baselines.buffer,
                np.int32(n_sources),
                np.int32(n_antennas),
                np.int32(n_baselines)
            ],
            global_size=(accel.roundup(n_baselines, self.template.wgs), self.n_channels),
            local_size=(self.template.wgs, 1)
        )
        logger.debug('Kernel queued')
        if self.async:
            return transfer_event
        else:
            # Make sure that the host->device transfers have completed, so that we
            # are ready for another call to this function.
            transfer_event.wait()
