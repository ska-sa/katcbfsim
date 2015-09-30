<%include file="/port.mako"/>
<%include file="/jones.mako"/>

} // extern C
#include <curand_kernel.h>
extern "C" {

DEVICE_FN float sqr(float x)
{
    return x * x;
}

DEVICE_FN int2 to_int(cplex a)
{
#ifdef __OPENCL_VERSION__
    return convert_int2_rte(a);
#else
    int2 out;
    out.x = __float2int_rn(a.x);
    out.y = __float2int_rn(a.y);
    return out;
#endif
}

typedef union
{
    jones in;
    ijones out;
} in_out;

/**
 * This kernel combined several operations:
 * - Sample visibilities from a statistical distribution.
 * - Apply per-antenna gains.
 * - Quantise to integer.
 *
 * The sampling ignores correlation between visibilities, and generates each
 * visibility independently. It also assumes that there are enough
 * accumulations to allow for a Gaussian approximation. There is thus some
 * non-zero probability than an autocorrelation could turn out negative, for
 * example.
 *
 * @param [in,out] data   On input, expected value for a single correlation product. On output, a sample for the accumulated visibility. Frequency-major.
 * @param flux_sum        Diagonal elements of sum of all the brightness matrices and the system equivalent flux density
 * @param gain            Jones matrix per antenna, frequency-major
 * @param baselines       Indices of the two antennas for each baseline
 * @param n_channels      Number of frequencies
 * @param n_antennas      Number of antennas
 * @param n_accs          Number of correlations being summed (in simulation)
 * @param seed            Random generator seed (keep fixed for a simulation)
 * @param sequence        Random generator sequence (increment for each call)
 */
KERNEL void sample(
    GLOBAL in_out * RESTRICT data,
    int data_stride,
    float flux_sum_x,
    float flux_sum_y,
    const GLOBAL jones * RESTRICT gain,
    int gain_stride,
    const GLOBAL short2 * RESTRICT baselines,
    int n_channels,
    int n_baselines,
    float n_accs,
    unsigned long long seed,
    unsigned long long sequence)
{
    int b = get_global_id(0);
    if (b >= n_baselines)
        return;
    int f = get_global_id(1);
    int f_step = get_local_size(1) * get_num_groups(1);
    short2 pq = baselines[b];
    int p = pq.x;
    int q = pq.y;

    /* Using sequence seems to be unspeakably slow, even for small numbers of
     * threads. We'll just ensure that every thread has a unique seed and not
     * worry that this might lead to correlations between threads.
     *
     * The scale factor is to ensure that incrementing seed by 1 gives an
     * entirely different random sequence, rather than shifting it by one
     * baseline. It is the prime using in the FNV hash function.
     */
    seed += ((sequence * f_step + f) * n_baselines + b) * 1099511628211ULL;
    curandState_t state;
    curand_init(seed, 0, 0, &state);
    float flux_sum[2] = {flux_sum_x, flux_sum_y};

    for (; f < n_channels; f += f_step)
    {
        int data_idx = f * data_stride + b;
        jones predict = data[data_idx].in;
        jones sample;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                // TODO: could precompute
                float a = 0.5f * n_accs * flux_sum[i] * flux_sum[j];
                float b = 0.5f * n_accs * (sqr(predict.m[i][j].x) - sqr(predict.m[i][j].y));
                float rr = a + b;
                float ii = a - b;
                float ri = n_accs * predict.m[i][j].x * predict.m[i][j].y;
                /* Compute Cholesky factorisation of
                 * [ rr ri ]
                 * [ ri ii ].
                 */
                float l_rr = sqrt(rr);
                float l_ri = ri / l_rr;
                float l_ii = sqrt(ii - l_ri * l_ri);
                /* Compute the random sample by transforming a pair of standard
                 * normal variables by L and adding the mean.
                 */
                float2 norm = curand_normal2(&state);
                sample.m[i][j].x = n_accs * predict.m[i][j].x + l_rr * norm.x;
                sample.m[i][j].y = n_accs * predict.m[i][j].y + l_ri * norm.x + l_ii * norm.y;
            }

        // Apply gains
        int offset = f * gain_stride;
        sample = jones_mul_h(jones_mul(gain[offset + p], sample), gain[offset + q]);
        // Quantise to integer
        ijones quant;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                quant.m[i][j] = to_int(sample.m[i][j]);
        data[data_idx].out = quant;
    }
}
