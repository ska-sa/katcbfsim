/**
 * @file
 *
 * Radio Interferometry Measurement Equation
 */

<%include file="/port.mako"/>
<%include file="jones.mako"/>

#define MAX_ANTENNAS ${max_antennas}

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
 * @param out             Output predicted visibilities, channel-major, float32, xx xy yx yy
 * @param flux_density    Jones brightness matrices for point source apparent brightness, frequency-major
 * @param inv_wavelength  Inverse of wavelength per channel, in per-metre
 * @param scaled_phase    -2(ul + vm + w(n-1)) per antenna per source (source-major), in metres
 * @param baselines       Indices of the two antennas for each baseline
 * @param sefd            System Equivalent Flux Density (assumed to be unpolarised)
 * @param n_sources       Number of point sources
 * @param n_antennas      Number of antennas
 * @param n_baselines     Number of baselines
 */
KERNEL void predict(
    GLOBAL jones * RESTRICT out,
    int out_stride,
    const GLOBAL jones * RESTRICT flux_density,
    int flux_stride,
    const GLOBAL float * RESTRICT inv_wavelength,
    const GLOBAL float * RESTRICT scaled_phase,
    const GLOBAL short2 * RESTRICT baselines,
    float sefd,
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

    if (b < n_baselines)
        out[f * out_stride + b] = sum;
}
