/**
 * @file
 *
 * Radio Interferometry Measurement Equation
 */

} // extern C
#include <curand_kernel.h>
extern "C" {

<%include file="/port.mako"/>

#define MAX_ANTENNAS ${max_antennas}

typedef float2 cplex;

typedef struct
{
    cplex m[2][2]; // row-major order
} jones;

typedef struct
{
    int2 m[2][2]; // row-major order
} ijones;

// Compute a * b
DEVICE_FN cplex cmul(cplex a, cplex b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Compute a * conj(b)
DEVICE_FN cplex cmul_conj(cplex a, cplex b)
{
    return make_float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}

// Compute a * b
DEVICE_FN cplex cmul_real(float a, cplex b)
{
    return make_float2(a * b.x, a * b.y);
}

// Compute a + b
DEVICE_FN cplex cadd(cplex a, cplex b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

// Compute conj(a)
DEVICE_FN cplex cconj(cplex a)
{
    return make_float2(a.x, -a.y);
}

// Compute exp(pi * i * phase)
DEVICE_FN cplex exp_pi_i(float phase)
{
    float s, c;
#ifdef __OPENCL_VERSION__
    s = sinpi(phase);
    c = cospi(phase);
#else
    sincospi(phase, &s, &c);
#endif
    return make_float2(c, s);
}

// Returns an identity matrix scaled by a real scalar
DEVICE_FN jones jones_init(float diag)
{
    jones out;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            out.m[i][j].x = (i == j) ? diag : 0;
            out.m[i][j].y = 0;
        }
    return out;
}

// Computes a * b
DEVICE_FN jones jones_mul(jones a, jones b)
{
    jones out;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            out.m[i][j] = cadd(cmul(a.m[i][0], b.m[0][j]), cmul(a.m[i][1], b.m[1][j]));
    return out;
}

// Compute a * b^H
DEVICE_FN jones jones_mul_h(jones a, jones b)
{
    jones out;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            out.m[i][j] = cadd(cmul_conj(a.m[i][0], b.m[j][0]), cmul_conj(a.m[i][1], b.m[j][i]));
    return out;
}

// Compute a + b
DEVICE_FN jones jones_add(jones a, jones b)
{
    jones out;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            out.m[i][j] = cadd(a.m[i][j], b.m[i][j]);
    return out;
}

// Compute a * b
DEVICE_FN jones scalar_jones_mul(cplex a, jones b)
{
    jones out;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            out.m[i][j] = cmul(a, b.m[i][j]);
    return out;
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

DEVICE_FN float sqr(float x)
{
    return x * x;
}

DEVICE_FN float index_float2(float2 f, int idx)
{
    return idx == 0 ? f.x : f.y;
}

/**
 * Compute predicted visibilities. The work division is that axis 0 selects the
 * baseline, axis 1 selects the frequency, and each workgroup must only handle
 * one frequency (otherwise the local arrays need to be scaled up to be
 * per-frequency). The work group size must also be at least as large as the
 * number of antennas.
 *
 * The baselines array must be padded out to a multiple of the work group size,
 * with valid antenna indices in the padding positions.
 *
 * @param out             Output predicted visibilities, channel-major, int32, xx xy yx yy
 * @param flux_density    Jones brightness matrices for point source apparent brightness, frequency-major
 * @param flux_sum        Diagonal elements of sum of all the brightness matrices and the system equivalent flux density
 * @param gain            Jones matrix per antenna, frequency-major
 * @param inv_wavelength  Inverse of wavelength per channel, in per-metre
 * @param scaled_phase    -2(ul + vm + w(n-1)) per antenna per source (source-major), in metres
 * @param baselines       Indices of the two antennas for each baseline
 * @param sefd            System Equivalent Flux Density (assumed to be unpolarised)
 * @param n_sources       Number of point sources
 * @param n_antennas      Number of antennas
 * @param n_baselines     Number of baselines
 * @param n_accs          Number of correlations being summed (in simulation)
 * @param seed            Random generator seed (keep fixed for a simulation)
 * @param sequence        Random generator sequence (increment for each call)
 */
KERNEL void predict(
    GLOBAL ijones * RESTRICT out,
    int out_stride,
    const GLOBAL jones * RESTRICT flux_density,
    int flux_stride,
    float2 flux_sum,
    const GLOBAL jones * RESTRICT gain,
    int gain_stride,
    const GLOBAL float * RESTRICT inv_wavelength,
    const GLOBAL float * RESTRICT scaled_phase,
    const GLOBAL short2 * RESTRICT baselines,
    float sefd,
    int n_sources,
    int n_antennas,
    int n_baselines,
    float n_accs,
    unsigned long long seed,
    unsigned long long sequence)
{
    LOCAL_DECL cplex k[MAX_ANTENNAS];
    LOCAL_DECL jones kb[MAX_ANTENNAS];

    int b = get_global_id(0);
    int f = get_global_id(1);
    short2 pq = baselines[b];
    float inv_wavelength_private = inv_wavelength[f];
    int p = pq.x;
    int q = pq.y;
    int lid = get_local_id(0);
    jones sum = jones_init(p == q ? sefd : 0);
    for (int source = 0; source < n_sources; source++)
    {
        if (lid < n_antennas)
        {
            int idx = source * n_antennas + lid;
            float ph = scaled_phase[idx] * inv_wavelength_private;
            cplex k_private = exp_pi_i(ph);
            k[lid] = k_private;
            kb[lid] = scalar_jones_mul(k_private, flux_density[f * flux_stride + source]);
        }
        BARRIER(); // TODO: could batch several sources, to reduce barrier costs
        jones kbk = scalar_jones_mul(cconj(k[q]), kb[p]);
        sum = jones_add(sum, kbk);
        BARRIER();
    }

    // Done with barriers, so dead lanes can safely exit
    if (b >= n_baselines)
        return;

    curandState_t state;
    curand_init(seed + (sequence << 32) + (f * n_baselines + b) * 8LL, 0, 0, &state);
    jones v;
    /* sum gives the mean / expected visibility matrix. We now generate samples
     * for each of the four visibilities. We ignore correlation between
     * visibilities, and only consider correlation between the real and
     * imaginary parts.
     */
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            // TODO: could precompute
            float a = 0.5f * n_accs
                * index_float2(flux_sum, i) * index_float2(flux_sum, j);
            float b = 0.5f * n_accs * (sqr(sum.m[i][j].x) - sqr(sum.m[i][j].y));
            float rr = a + b;
            float ii = a - b;
            float ri = n_accs * sum.m[i][j].x * sum.m[i][j].y;
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
            v.m[i][j].x = n_accs * sum.m[i][j].x + l_rr * norm.x;
            v.m[i][j].y = n_accs * sum.m[i][j].y + l_ri * norm.x + l_ii * norm.y;
        }

    // Apply gains
    int offset = f * gain_stride;
    v = jones_mul_h(jones_mul(gain[offset + p], v), gain[offset + q]);
    // Quantise to integer
    ijones vi;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            vi.m[i][j] = to_int(v.m[i][j]);
    out[f * out_stride + b] = vi;
}
