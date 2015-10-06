/**
 * @file
 *
 * Common utilities for complex numbers and Jones matrices.
 */

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
            out.m[i][j] = cadd(cmul_conj(a.m[i][0], b.m[j][0]), cmul_conj(a.m[i][1], b.m[j][1]));
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


