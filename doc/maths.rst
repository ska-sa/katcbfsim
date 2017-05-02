Maths background
================

Statistical distribution of visibilities
----------------------------------------
We will consider only a single channel. Let :math:`e^k` be the electric field
at source :math:`k` and :math:`J_p^k` be the Jones matrix representing all
effects (direction-dependent and -independent) for antenna :math:`p` and source
:math:`k` (in general, we will use superscripts for sources and subscripts for
antennas). Each antenna will also experience thermal noise, which we'll
denote as :math:`n_p`. The channelised voltage :math:`v_p` then satisfies

.. math::
    v_p = n_p + \sum_k J_p^k e^k.

We can break these terms into scalars with separate variables for the real and
imaginary components. Let

.. math::
    n_p &=       \begin{pmatrix}a_p^0 + b_p^0 i\\c_p^0 + d_p^0 i\end{pmatrix}\\
    J_p^k e^k &= \begin{pmatrix}a_p^k + b_p^k i\\c_p^k + d_p^k i\end{pmatrix}\\
    v_p &= \begin{pmatrix}a_p + b_pi\\c_p + d_pi\end{pmatrix} =
    \begin{pmatrix}
    (a_p^0 + a_p^1 + \dots + a_p^k) + (b_p^0 + b_p^1 + \dots + b_p^k)i\\
    (c_p^0 + c_p^1 + \dots + c_p^k) + (d_p^0 + d_p^1 + \dots + d_p^k)i
    \end{pmatrix}

The covariance matrix of these elements has some important properties:

 - Different sources are assumed to be uncorrelated. This in turn means that
   :math:`J_p^k e^k` and :math:`J_q^l e^l` are also uncorrelated.

 - Noise terms are assumed to be uncorrelated with everything.

 - The statistical distribution is stationary (time-independent), which means
   that expectations are invariant under a global phase shift. From this is can
   be deduced that :math:`E[a_p] = E[b_p] = 0`, :math:`E[(a_p^j)^2] =
   E[(b_p^j)^2]` and :math:`E[a_p^jb_p^j] = 0` (and similarly for :math:`c` and
   :math:`d`); and :math:`E[a_pa_q] = E[b_pb_q]`, :math:`E[a_pb_q] =
   -E[b_pa_q]`, and similarly for cross-hand terms.

The visibility matrix for antennas :math:`p` and :math:`q` is

.. math::
  2\begin{pmatrix}
  (a_pa_q+b_pb_q) + (b_pa_q-a_pb_q)i & (a_pc_q+b_pd_q) + (b_pc_q-a_pd_q)i\\
  (c_pa_q+d_pb_q) + (d_pa_q-c_pb_q)i & (c_pc_q+d_pd_q) + (d_pc_q-c_pd_q)i
  \end{pmatrix}.

This matrix is then accumulated :math:`N` times. Assuming :math:`N` is large,
the sum will behave like a multi-variate Gaussian distribution, and thus to
generate samples it is sufficient to know the mean and covariance matrix for
the 8 terms.

Expanding to a sum of sources, most terms disappear under the assumptions of
independence, giving an expected value of

.. math::
  2\sum_j E\begin{pmatrix}
    (a_p^ja_q^j+b_p^jb_q^j) + (b_p^ja_q^j-a_p^jb_q^j)i
    & (a_p^jc_q^j+b_p^jd_q^j) + (b_p^jc_q^j-a_p^jd_q^j)i\\
    (c_p^ja_q^j+d_p^jb_q^j) + (d_p^ja_q^j-c_p^jb_q^j)i
    & (c_p^jc_q^j+d_p^jd_q^j) + (d_p^jc_q^j-c_p^jd_q^j)i
  \end{pmatrix}.

The terms in this sum are exactly the source coherencies, and can be
predicted from the RIME [rime]_, with the exception of the noise power that
appears in the autocorrelations. Using the expectation identities listed
earlier, we can also simplify this to

.. math::
  4\sum_j E\begin{pmatrix}
    a_p^ja_q^j + b_p^ja_q^j i & a_p^jc_q^j + b_p^jc_q^j i\\
    c_p^ja_q^j + d_p^ja_q^j i & c_p^jc_q^j + d_p^jc_q^j i
  \end{pmatrix} = 4E\begin{pmatrix}
    a_pa_q + b_pa_q i & a_pc_q + b_pc_q i\\
    c_pa_q + d_pa_q i & c_pc_q + d_pc_q i
  \end{pmatrix}.

It's important to note that it is only the *expectations* that are equivalent
to the previous formula; the values inside the expectations are not the same
for particular samples of the random variables.

Next, let us compute variance for the visibility matrix, using Isserlis'
Theorem:

.. math::
    \begin{align}
    & E[(2a_pa_q + 2b_pb_q)^2] - E[2a_pa_q + 2b_pb_q]^2\\
    &= 4E[a_p^2a_q^2] + 8E[a_pa_qb_pb_q] + 4E[b_p^2b_q^2] - 4(E[a_pa_q]^2 +
       2E[a_pa_q]E[b_pb_q] + E[b_pb_q]^2)\\
    &= 4E[a_p^2]E[a_q^2] + 8E[a_pa_q]^2 + 8(
        E[a_pa_q]E[b_pb_q] + E[a_pb_p]E[a_qb_q] + E[a_pb_q]E[a_qb_p])
     + 4E[b_p^2]E[b_q^2] + 8E[b_pb_q]^2
     - 4(E[a_pa_q]^2 + 2E[a_pa_q]E[b_pb_q] + E[b_pb_q]^2)\\
    &= 4E[a_p^2]E[a_q^2] + 4E[b_p^2]E[b_q^2] + 4E[a_pa_q]^2 + 4E[b_pb_q]^2 + 8E[a_pb_q]E[a_qb_p]\\
    &= 8E[a_p^2]E[a_q^2] + 8E[a_pa_q]^2 - 8E[a_pb_q]^2.
    \end{align}

The expectations in this expression all appear in the visibility matrix
itself. Variances for the other seven terms can be computed similarly by
substituting :math:`(b_p, -a_p)`, :math:`(c_p, d_p)` or :math:`(d_p, -c_p)`
in place of :math:`(a_p, b_p)` and the same for :math:`q`. The covariance
between real and imaginary parts can be computed similarly:

.. math::
    \begin{align}
    &E[4(a_pa_q + b_pb_q)(b_pa_q-a_pb_q)] - E[2(a_pa_q+b_pb_q)]E[2(b_pa_q-a_pb_q)]\\
    &= 4E[a_q^2a_pb_p - a_p^2a_qb_q + b_p^2a_qb_q - b_q^2a_pb_p] - 16E[a_pa_q]E[b_pa_q]\\
    &= 8(E[a_pa_q]E[b_pa_q] - E[a_pa_q]E[a_pb_q] + E[b_pa_q]E[b_pb_q] -
         E[b_pb_q]E[a_pb_q]) - 16E[a_pa_q]E[b_pa_q]\\
    &= 16E[a_pa_q]E[b_pa_q].
    \end{align}

For now we omit covariances between polarizations and between baselines and
treat them as zero. While incorrect, these covariances currently have no
effect on ingest (it may eventually have an effect if flagging is done jointly
across polarisations).

.. [rime] Smirnov, O.M. Revisiting the radio interferometer measurement
   equation. I. A full-sky Jones formalism. A&A 527 A106 (2011).
