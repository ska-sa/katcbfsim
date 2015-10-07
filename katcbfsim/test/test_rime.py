"""Tests for :mod:`katcbfsim.rime`."""

from __future__ import print_function, division
from katsdpsigproc import accel
from katsdpsigproc.test.test_accel import device_test, force_autotune
from katcbfsim import rime
from nose.tools import *
import katpoint
import numpy as np
import scipy.stats
from collections import namedtuple


@nottest
def boxm_test(*args):
    """Box M Test for equal covariance matrices. This is a generalisation of
    :func:`scipy.stats.bartlett` for multivariate variables. The formulae are
    taken from here__.

    __ http://www.real-statistics.com/multivariate-statistics/boxs-test-equality-covariance-matrices/boxs-test-basic-concepts/

    Parameters
    ----------
    sample1, sample2, ... : array-like
        arrays of sample data. Each row represents a variable, with
        observations in the columns, as for :func:`numpy.cov`. The samples need
        not have the same number of observations.

    Returns
    -------
    T : float
        The test statistic
    p-value : float
        The p-value of the test
    """
    Si = [np.atleast_2d(np.cov(x)) for x in args]
    ni = [x.shape[1] for x in args]
    k = len(Si[0])
    m = len(Si)
    n = sum(ni)
    if n <= m:
        raise ValueError('Too few samples')
    S = sum((nj - 1) * Sj for nj, Sj in zip(ni, Si)) / (n - m)
    M = (n - m) * np.log(np.linalg.det(S))
    M -= sum((nj - 1) * np.log(np.linalg.det(Sj)) for nj, Sj in zip(ni, Si))
    scale = (2 * k**2 + 3 * k - 1) / ((6 * (k + 1) * (m - 1)))
    c = scale * sum(1 / (nj - 1) - 1 / (n - m) for nj in ni)
    T = M * (1 - c)
    df = k * (k + 1) * (m - 1) / 2
    pval = scipy.stats.chi2.sf(T, df)
    BoxmResult = namedtuple('BoxmResult', ('statistic', 'pvalue'))
    return BoxmResult(T, pval)


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
        n_samples = 1000
        threshold = 0.01  # Overall probability that the test will randomly fail
        # Threshold for each p test. There are n_channels * n_baselines * 4
        # visibilities, and we compute 3 p-values for each, except for autos
        # where we only compute two.
        tests = n_channels * (n_baselines * 4 * 3 - n_antennas * 2)
        threshold1 = 1 - (1 - threshold)**(1 / tests)
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
                    if i == j:
                        np.testing.assert_array_equal(
                            device_samples[channel, baseline_index, 0, 1, ...],
                            np.conj(device_samples[channel, baseline_index, 1, 0, ...]),
                            "Autocorrelation is not Hermitian")
                    for k in range(2):
                        for l in range(2):
                            cur = samples[2 * i + k, 2 * j + l, ...]
                            ds = device_samples[channel, baseline_index, k, l, ...]
                            cur = np.c_[cur.real, cur.imag].T
                            ds = np.c_[ds.real, ds.imag].T
                            expected = V[2 * i + k, 2 * j + l] * n_accs
                            expected = np.array([expected.real, expected.imag])
                            # Special case for autocorrelations: the value will be
                            # purely real, making the covariance matrix singular,
                            # where boxm_test and ttest_1samp break down.
                            if i == j and k == l:
                                assert_true(np.all(cur[1, ...] == 0))
                                assert_true(np.all(ds[1, ...] == 0))
                                cur = cur[0:1]
                                ds = ds[0:1]
                                expected = expected[0:1]
                            result = boxm_test(cur, ds)
                            assert_greater(result.pvalue, threshold1)
                            result = scipy.stats.ttest_1samp(ds, expected, axis=1)
                            for p in np.atleast_1d(result.pvalue):
                                assert_greater(p, threshold1)
                    baseline_index += 1

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        """Test that autotuner runs successfully"""
        rime.RimeTemplate(context, 64)
