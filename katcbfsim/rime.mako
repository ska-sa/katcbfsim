/**
 * @file
 *
 * Radio Interferometry Measurement Equation
 */

<%include file="/port.mako"/>

#define MAX_ANTENNAS ${max_antennas}

typedef float2 cplex;

typedef struct
{
    cplex m[2][2];
} jones;

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

// Returns a matrix of zeros
DEVICE_FN jones jones_zero()
{
    jones out;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            out.m[i][j].x = 0;
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
 * @param out             Output predicted visibilities, channel-major
 * @param flux_density    Jones matrices for point source apparent brightness, frequency-major
 * @param gain            Jones matrix per antenna, frequency-major
 * @param inv_wavelength  Inverse of wavelength per channel, in per-metre
 * @param scaled_phase    -2(ul + vm + w(n-1)) per antenna per source (source-major), in metres
 * @param baselines       Indices of the two antennas for each baseline
 * @param n_sources       Number of point sources
 * @param n_antennas      Number of antennas
 * @param n_baselines     Number of baselines
 */
KERNEL void predict(
    GLOBAL jones * RESTRICT out,
    int out_stride,
    const GLOBAL jones * RESTRICT flux_density,
    int flux_stride,
    const GLOBAL jones * RESTRICT gain,
    int gain_stride,
    const GLOBAL float * RESTRICT inv_wavelength,
    const GLOBAL float * RESTRICT scaled_phase,
    const GLOBAL short2 * RESTRICT baselines,
    int n_sources,
    int n_antennas,
    int n_baselines)
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
    jones sum = jones_zero();
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
        jones kbk = scalar_jones_mul(k[q], kb[p]);
        sum = jones_add(sum, kbk);
        BARRIER();
    }

    if (b < n_baselines)
    {
        int offset = f * gain_stride;
        jones v = jones_mul_h(jones_mul(gain[offset + p], sum), gain[offset + q]);
        out[f * out_stride + b] = v;
    }
}
