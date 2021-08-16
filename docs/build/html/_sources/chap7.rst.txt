Chapter 7 Dimensionality Reduction
==================================

7.1 Background
--------------

Let the data :math:`\D` consist of :math:`n` points over :math:`d` attributes, 
that is, it is an :math:`n\times d` matrix, given as

.. math::

    \D=\left(\begin{array}{c|cccc}&X_1&X_2&\cds&X_d\\ \hline 
    \x_1&x_{11}&x_{12}&\cds&x_{1d}\\\x_2&x_{21}&x_{22}&\cds&x_{2d}\\ 
    \vds&\vds&\vds&\dds&\vds\\\x_n&x_{n1}&x_{n2}&\cds&x_{nd}\end{array}\right)

Each point :math:`\x_i=(x_{i1},x_{i2},\cds,x_{id})^T` is a vector in the ambient 
:math:`d`-dimensional vector space spanned by the :math:`d` standard basis 
vectors :math:`\e_1,\e_2,\cds,\e_d`, where :math:`\e_i` corresponds to the
:math:`i`th attribute :math:`X_i`.

Given any other set of :math:`d` orthonormal vectors :math:`\u_1,\u_2,\cds,\u_d`,
with :math:`\u_i^T\u_j=0` and :math:`\lv\u_i\rv=1` (or :math:`\u_i^T\u_i=1`), we
can re-express each point :math:`x` as the linear combination

.. note::

    :math:`\x=a_1\u_1+a_2\u_2+\cds+a_d\u_d`

where the vector :math:`\a=(a_1,a_2,\cds,a_d)^T` represents the coordinates of
:math:`\x` in the new basis.
The above linear combination can also be expressed as a matrix multiplication:

.. note::

    :math:`\x=\U\a`.

where :math:`\U` is an *orthonormal* matrix whose :math:`i`\ th column comprises 
the :math:`i`\ th basis vector :math:`\u_i`.

Because :math:`\U` is orthogonal, we have

.. math::

    \U\im=\U^T

which implies that :math:`\U^T\U=\I`.

.. math::

    \U^T\x=\U^T\U\a

.. note::

    :math:`\a=\U^T\x`

Becuase there are potentially infinite choices for the set of orthonormal basis
vectors, one natural question is whether ther exists an *optimal* basis, for a
suitable notion of optimality.
We are interested in finding the optimal :math:`r`-dimensional representation of :math:`\D` with :math:`r\ll d`.
Projection of :math:`\x` onto the first :math:`r` basis vectors is given as

.. math::

    \x\pr=a_1\u_1+a_2\u_2+\cds+a_r\u_r+\sum_{i=1}^ra_i\u_i

which can be written in matrix notaion as follows

.. math::

    \x\pr=\bp|&|&&|\\\u_1&\u_2&\cds&\u_r\\|&|&&|\ep\bp a_1\\a_2\\\vds\\a_r \ep=\U_r\a_r

where :math:`\U_r` is the matrix comprising the first :math:`r` basis vectors, 
and :math:`\a_r` is a vectgor comprising the first :math:`r` coordinates.
Because :math:`\a=\U^T\x`, restricting it to the first :math:`r` terms, we get

.. math::

    \a_r=\U_r^T\x

The projection of :math:`\x` onto the first :math:`r` basis vectors can be compactly written as

.. note::

    :math:`\x\pr=\U_r\U_r^T\x=\P_r\x`

where :math:`\P_r=\U_r\U_r^T` is the *orthogonal projection matrix* for the 
subspace spanned by the first :math:`r` basis vectors.
The projection matrix :math:`\P_r` can also be written as the decomposition

.. math::

    P_r=\U_r\U_r^T=\sum_{i=1}^r\u_i\u_i^T

The projection of :math:`\x` onto the remaining dimensions comprises the *error vector*

.. note::

    :math:`\dp\bs\epsilon=\sum_{i=r+1}^da_i\u_i=\x-\x\pr`

It is worth noting that :math:`\x\pr` and :math:`\bs\epsilon` are orthogonal vectors:

.. math::

    {\x\pr}^T\bs\epsilon=\sum_{i=1}^r\sum_{j=r+1}^da_ia_j\u_i^T\u_j=0

The subspace spanned by the first :math:`r` basis vectors and the subspace 
spanned by the remaining basis vectors are *orthogonal subspaces*.
They are *orthogonal complement* of each other.

The goal of dimensionality reduction is to seek an :math:`r`-dimensional basis 
that gives the best possible approximation :math:`\x_i\pr` over all the points
:math:`\x_i\in\D`.
Alternatively, we may seek to minimize the error :math:`\bs\epsilon_i=\x_i-\x_i\pr` over all the points.

7.2 Principal Component Analysis
--------------------------------
































7.3 Kernel Principal Component Analysis
---------------------------------------


























7.4 Singular Value Decomposition
--------------------------------