"""Tests for :mod:`katcbfsim.rime`."""

from __future__ import print_function, division
from katsdpsigproc import accel
from katsdpsigproc.test.test_accel import device_test, force_autotune
from katcbfsim import rime
import katpoint
import numpy as np
import scipy.stats


class TestRime(object):
    """Tests for :class:`katcbfsim.rime.Rime`."""

    @device_test
    def test_sample_stats(self, context, queue):
        """Run a dumb CPU simulation for a small array, and check that the
        statistics are the same as the GPU simulation. For simplicity, we
        simulate only a single polarised source with a flat spectrum.
        """
        rs = np.random.RandomState(1)
        n_antennas = 3
        n_baselines = n_antennas * (n_antennas + 1) // 2
        n_channels = 4
        n_accs = 400
        n_samples = 100000
        allocator = accel.DeviceAllocator(context)
        raw = allocator.allocate_raw(n_channels * n_baselines * 4 * 8)
        # in_data and out_data both alias raw, since _run_sample runs in-place
        in_data = allocator.allocate((n_channels, n_baselines, 2, 2), np.complex64, raw=raw)
        out_data = allocator.allocate((n_channels, n_baselines, 2, 2, 2), np.int32, raw=raw)
        delays = np.array([3.0, -4.0, 6.0])           # In m, after phasing
        stokes = (10.0, 7.0, 5.0, 4.0)                # In Jansky
        B = np.matrix([
            [stokes[0] + stokes[1], stokes[2] + 1j * stokes[3]],
            [stokes[2] - 1j * stokes[3], stokes[0] - stokes[1]]])
        sefd = 20.0   # Jansky

        ### Prepare the operation
        template = rime.RimeTemplate(context, n_antennas)
        # The actual antennas are irrelevant for this test
        antennas = [None] * n_antennas
        fn = template.instantiate(queue, 1284e6, 856e6, n_channels,
                n_accs, [], antennas, sefd, seed=1)
        fn.bind(out=out_data)
        fn.ensure_all_bound()
        # Set gains to all identity
        gain = np.tile(np.identity(2, np.complex64), (n_channels, n_antennas, 1, 1))
        fn.buffer('gain').set(queue, gain)
        # We didn't specify the sources, so we need to fudge the total flux
        fn._flux_sum[0] = B[0, 0].real + sefd
        fn._flux_sum[1] = B[1, 1].real + sefd

        ### Predict expected visibilities
        predict = []
        in_data_host = in_data.empty_like()
        for channel in range(n_channels):
            inv_wavelength = fn.frequencies[channel] / katpoint.lightspeed
            K = np.exp(-2j * np.pi * delays * inv_wavelength)
            # Convert from scalars to Jones matrices
            K = np.matrix(np.kron(np.row_stack(K), np.identity(2)))
            # Compute the expected visibility matrix (a grid of 2x2
            # Jones matrices).
            V = K * B * K.getH()
            # Add in system equivalent flux density on diagonal
            V += sefd * np.identity(len(V))
            predict.append(V)
            # Load the predicted visibilities to the staging area, without _run_predict
            baseline_index = 0
            for i in range(n_antennas):
                for j in range(i, n_antennas):
                    in_data_host[channel, baseline_index, ...] = \
                        predict[channel][2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
                    baseline_index += 1
        # Run n_samples times
        out_data_host = out_data.empty_like()
        device_samples = np.empty((n_channels, n_baselines, 2, 2, n_samples), np.complex64)
        for i in range(n_samples):
            in_data.set(queue, in_data_host)
            fn._run_sample()
            out_data.get(queue, out_data_host)
            device_samples[..., i] = out_data_host[..., 0] + 1j * out_data_host[..., 1]

        ### Compare to a simple CPU simulation
        for channel in range(n_channels):
            V = predict[channel]
            # Generate random voltage samples, which are circularly
            # symmetric and have covariance V/2.
            L = np.linalg.cholesky(V / 2)
            samples = np.empty((2 * n_antennas, 2 * n_antennas, n_samples), np.complex64)
            for i in range(n_samples):
                z = np.matrix(rs.randn(len(L), n_accs) + 1j * rs.randn(len(L), n_accs))
                # This gives E[z * z.H] = 2I, but we want it to be I.
                z /= np.sqrt(2)
                v = L * z
                # Each column of v is now a sample. v * v.H sums the visibilities
                # over all samples.
                samples[..., i] = 2 * (v * v.getH())
            # Test each complex visibility individually, since the simulation
            # does not model correlations between visibilities.
            baseline_index = 0
            for i in range(n_antennas):
                for j in range(i, n_antennas):
                    for k in range(2):
                        for l in range(2):
                            cur = samples[2 * i + k, 2 * j + l, ...]
                            ds = device_samples[channel, baseline_index, k, l, ...]
                            cur = np.c_[cur.real, cur.imag]
                            ds = np.c_[ds.real, ds.imag]
                            expected = V[2 * i + k, 2 * j + l] * n_accs
                            expected = np.array([expected.real, expected.imag])
                            print(i, j, k, l)
                            print("mean (h)", np.mean(cur, axis=0))
                            print("mean (d)", np.mean(ds, axis=0))
                            print("expected", expected)
                            print("cov (h)", np.cov(cur, rowvar=0))
                            print("cov (d)", np.cov(ds, rowvar=0))
                    baseline_index += 1

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        """Test that autotuner runs successfully"""
        rime.RimeTemplate(context, 64)
