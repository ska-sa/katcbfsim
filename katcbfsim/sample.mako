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

DEVICE_FN int2 to_big_endian(int2 a)
{
#ifdef __OPENCL_VERSION__
# error "Not yet implemented for OpenCL"
#else
    /* The CUDA programming guide says "The NVIDIA GPU architecture uses a
     * little-endian representation," so there is no need to check the
     * endianness.
     */
    int2 out;
    out.x = __byte_perm(a.x, 0, 0x0123);
    out.y = __byte_perm(a.y, 0, 0x0123);
    return out;
#endif
}

typedef union
{
    jones in;
    ijones out;
} in_out;

/**
 * This kernel combines several operations:
 * - Sample visibilities from a statistical distribution.
 * - Apply per-antenna direction-independent effects.
 * - Quantise to integer.
 *
 * The sampling ignores correlation between visibilities, and generates each
 * visibility independently. It also assumes that there are enough
 * accumulations to allow for a Gaussian approximation. There is thus some
 * non-zero probability than an autocorrelation could turn out negative, for
 * example.
 *
 * @param [in,out] data   On input, expected value for a single correlation product. On output, a sample for the accumulated visibility. Frequency-major.
 * @param data_stride     Element stride for @a data
 * @param baselines       Indices of the two antennas for each baseline
 * @param autocorrs       Baseline indices for autocorrelations
 * @param n_channels      Number of frequencies
 * @param n_antennas      Number of antennas
 * @param n_accs          Number of correlations being summed (in simulation)
 * @param seed            Random generator seed (keep fixed for a simulation)
 * @param sequence        Random generator sequence (increment for each call)
 */
KERNEL void sample(
    GLOBAL in_out * RESTRICT data,
    int data_stride,
    const GLOBAL short2 * RESTRICT baselines,
    const GLOBAL int * RESTRICT autocorrs,
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
    int p_bl = autocorrs[p];
    int q_bl = autocorrs[q];

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

    for (; f < n_channels; f += f_step)
    {
        int data_offset = f * data_stride;
        float diag_p[2], diag_q[2];
        // TODO: this isn't the best access pattern. It may be better to use a
        // separate kernel to extract the autocorrelations.
        diag_p[0] = data[data_offset + p_bl].in.m[0][0].x;
        diag_p[1] = data[data_offset + p_bl].in.m[1][1].x;
        diag_q[0] = data[data_offset + q_bl].in.m[0][0].x;
        diag_q[1] = data[data_offset + q_bl].in.m[1][1].x;
        int data_idx = data_offset + b;
        // TODO: moving the load of 'predict' inside the loop would reduce
        // register pressure (but also reduce latency hiding).
        jones predict = data[data_idx].in;
        jones sample;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                // TODO: could precompute A.
                float A = 0.5f * n_accs * diag_p[i] * diag_q[j];
                float B = 0.5f * n_accs * (sqr(predict.m[i][j].x) - sqr(predict.m[i][j].y));
                float rr = A + B;
                float ii = A - B;
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

        /* Autocorrelations must be Hermitian. This needs to be done before
         * applying direction-independent effects.
         * TODO: probably more efficient to have a separate kernel
         * to fix up the autocorrelations afterwards.
         */
        if (p == q)
        {
            sample.m[0][0].y = 0.0;
            sample.m[1][1].y = 0.0;
            sample.m[1][0].x = sample.m[0][1].x;
            sample.m[1][0].y = -sample.m[0][1].y;
        }

        // Quantise to integer
        ijones quant;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                quant.m[i][j] = to_big_endian(to_int(sample.m[i][j]));
        data[data_idx].out = quant;
    }
}
