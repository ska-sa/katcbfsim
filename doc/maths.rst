Maths background
================

Statistical distribution of visibilities
----------------------------------------
We will consider only a single channel. The set of F engine outputs for all
antennas is assumed to be a multi-variate Gaussian distribution, where real
and imaginary components are treated as separate variables. It can further be
broken into a sum from each source, plus an input-specific noise term, which we
label with index 0.

.. math::
    v_p = \begin{pmatrix}a_p + b_pi\\c_p + d_pi\end{pmatrix} =
    \begin{pmatrix}
    (a_p^0 + a_p^1 + \dots + a_p^k) + (b_p^0 + b_p^1 + \dots + b_p^k)i\\
    (c_p^0 + c_p^1 + \dots + c_p^k) + (d_p^0 + d_p^1 + \dots + d_p^k)i
    \end{pmatrix}

Here subscripts index antennas and superscripts index sources. We will assume
that antenna gains are applied after statistical sampling, so that for the
purposes of this section we can assume that G Jones terms are all the
identity.

The covariance matrix of these elements has some important properties:

 - Different sources are assumed to be uncorrelated.

 - Noise terms are assumed to be uncorrelated with everything.

 - Each polarisation of each source has independant magnitude and phase, so
   :math:`E[a_p] = E[b_p] = 0`, :math:`E[(a_p^j)^2] = E[(b_p^j)^2]` and
   :math:`E[a_p^jb_p^j] = 0`, and similarly for c, d.

 - More generally, the distribution is invariant under a phase shift applied
   to all antennas (since this just corresponds to sampling at a different
   point in time). Applying a :math:`\pi/2` shift gives
   :math:`E[a_pa_q] = E[b_pb_q]`, :math:`E[a_pb_q] = -E[b_pa_q]`, and
   similarly for cross-hand terms.

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

Firstly let us consider the mean for cross-correlations. Expanding to a sum of
sources, most terms disappear under the assumptions of independence, giving an
expected value of

.. math::
  2\sum_j E\begin{pmatrix}
    (a_p^ja_q^j+b_p^jb_q^j) + (b_p^ja_q^j-a_p^jb_q^j)i
    & (a_p^jc_q^j+b_p^jd_q^j) + (b_p^jc_q^j-a_p^jd_q^j)i\\
    (c_p^ja_q^j+d_p^jb_q^j) + (d_p^ja_q^j-c_p^jb_q^j)i
    & (c_p^jc_q^j+d_p^jd_q^j) + (d_p^jc_q^j-c_p^jd_q^j)i
  \end{pmatrix}.

The terms in this sum are exactly the source coherencies, and can be
predicted from the RIME [rime]_.

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

Variances for the other seven terms can be computed similarly by substituting
:math:`(b_p, -a_p)`, :math:`(c_p, d_p)` or :math:`(d_p, -c_p)` in place of
:math:`(a_p, b_p)` and the same for :math:`q`. For now we omit covariances and
treat them as zero. While incorrect, the covariances currently have no effect
on ingest (it may eventually have an effect if flagging is done jointly across
polarisations).

We still need a method to evaluate this expectation. The final two terms can
be obtained from the predicted visibility. To evaluate :math:`E[a_p^2]` we
need to expand it:

.. math::
    E[a_p^2] = E[\sum_j (a_p^j)^2]
    = \tfrac{1}{2} \sum_j E[\lvert a_p^j + b_p^j i\rvert^2]
    = \tfrac{1}{4} \sum_j (I^j + Q^j)

where :math:`I^j` and :math:`Q^j` are Stokes parameters for source :math:`j`
(or the system equivalent flux density).
Similarly, :math:`E[c_p^2] = E[d_p^2] = \tfrac{1}{4} \sum_j (I^j - Q^j)`. Note
that this does not depend on :math:`p`, assuming that the noise term is the
same for all antennas after gain correction.

For autocorrelations the result is almost the same, except that when computing
the mean, the noise term must be included in the sum for the non-crosshand
terms.

Let us return to covariance, and consider just the covariance between real and
imaginary parts:

.. math::
    \begin{align}
    &E[4(a_pa_q + b_pb_q)(b_pa_q-a_pb_q)] - E[2(a_pa_q+b_pb_q)]E[2(b_pa_q-a_pb_q)]\\
    &= 4E[a_q^2a_pb_p - a_p^2a_qb_q + b_p^2a_qb_q - b_q^2a_pb_p] - 16E[a_pa_q]E[b_pa_q]\\
    &= 8(E[a_pa_q]E[b_pa_q] - E[a_pa_q]E[a_pb_q] + E[b_pa_q]E[b_pb_q] -
         E[b_pb_q]E[a_pb_q]) - 16E[a_pa_q]E[b_pa_q]\\
    &= 16E[a_pa_q]E[b_pa_q].
    \end{align}

.. [rime] Smirnov, O.M. Revisiting the radio interferometer measurement
   equation. I. A full-sky Jones formalism. A&A 527 A106 (2011).
