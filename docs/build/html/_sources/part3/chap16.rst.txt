Chapter 16 Spectral and Graph Clustering
========================================

16.1 Graphs and Matrices
------------------------

Given a dataset :math:`\D` comprising :math:`n` points 
:math:`\x_i\in\R^d\ (i=1,2,\cds,n)`, let :math:`\A` denote the :math:`n\times n`
symmetric *similarity matrix* between the points, given as

.. math::

    \A = \left(\begin{array}{cccc}a_{11}&a_{12}&\cds&a_{1n}\\ 
    \a_{21}&a_{22}&\cds&a_{1n}\\\vds&\vds&\dds&\vds\\
    \a_{n1}&a_{n2}&\cds&a_{nn}\end{array}\right)

where :math:`\A(i,j)=a_{ij}` denotes the similarity or affinity between points :math:`\x_i` and :math:`\x_j`.
We require the similarity to be symmetric and non-negative, that is, :math:`a_{ij}=a_{ji}` and :math:`a_{ji}\geq 0`.
The matrix :math:`\A` may be considered to be a *weighted adjacency matrix* of 
the weighted (undirected) graph :math:`G=(V,E)`, where each vertex is a point 
and each edge joins a pair of points, that is,

.. math::

    V&=\{\x_i|i=1,\cds,n\}

    E&=\{(\x_i,\x_j)|1\leq i,j\leq n\}

Further, the similarity matrix :math:`\A` gives the weight on each edge, that 
is, :math:`a_{ij}` denotes the weight of the edge :math:`(\x_i,\x_j)`.
If all affinities are 0 or 1, then :math:`\A` represents the regular adjacency relationship between the vertices.

For a vertex :math:`\x_i`, let :math:`d_j` denote the *degree* of the vertex, defined as

.. math::

    d_i=\sum_{j=1}^n a_{ij}

We define the *degree matrix* :math:`\Delta` of graph :math:`G` as the :math:`n\times n` diagonal matrix:

.. note::

    :math:`\Delta=\left(\begin{array}{cccc}d_1&0&\cds&0\\0&d_2&\cds&0\\\vds&\vds&\dds&\vds\\0&0&\cds&d_n\end{array}\right)`
    :math:`=\left(\begin{array}{cccc}\sum_{j=1}^na_{1j}&0&\cds&0\\0&\sum_{j=1}^na_{2j}&\cds&0\\\vds&\vds&\dds&\vds\\0&0&\cds&\sum_{j=1}^na_{nj}\end{array}\right)`

:math:`\Delta` can be compactly written as :math:`\Delta(i,i)=d_i` for all :math:`1\leq i\leq n`.

**Normalized Adjacency Matrix**

The normalized adjacency matrix is obtained by dividing each row of the 
adjacency matrix by the degree of the corresponding node.

.. note::

    :math:`\bs{\rm{M}}=\Delta\im\A=`
    :math:`\left(\begin{array}{cccc}\frac{a_{11}}{d_1}&\frac{a_{12}}{d_1}&\cds&\frac{a_{1n}}{d_1}\\\frac{a_{21}}{d_2}&\frac{a_{22}}{d_2}&\cds&\frac{a_{2n}}{d_2}\\\vds&\vds&\dds&\vds\\\frac{a_{n1}}{d_n}&\frac{a_{n2}}{d_n}&\cds&\frac{a_{nn}}{d_n}\end{array}\right)`

Each element of :math:`\bs{\rm{M}}`, namely :math:`m_{ij}` is also non-negative, 
as :math:`m_{ij}=\frac{a_{ij}}{d_i}\geq 0`.
Consider the sum of the :math:`i`\ th row in :math:`\bs{\rm{M}}`, we have

.. math::

    \sum_{j=1}^nm_{ij}=\sum_{j=1}^n\frac{a_{ij}}{d_i}=\frac{d_i}{d_i}=1

Thus, each row in :math:`\bs{\rm{M}}` sums to 1.
This implies that 1 is an eigenvalue of :math:`\bs{\rm{M}}`.
In fact, :math:`\ld=1` is the largest eigenvalue of :math:`\bs{\rm{M}}`, and the
other eigenvalues satisfy the property that :math:`|\ld_i|\leq 1`.
Also, if :math:`G` is connected then the eigenvector corresponding to 
:math:`\ld_1` is 
:math:`\u_1=\frac{1}{\sqrt{n}}=(1,1,\cds,1)^T=\frac{1}{\sqrt{n}}\1`.
Because :math:`\bs{\rm{M}}` is not symmetric, its eigenvectors are not necessarily orthogonal.

**Graph Laplacian Matrices**

The *Laplacian matrix* of a graph is defined as

.. math::

    \bs{\rm{L}}=\Delta-\A=\left(\begin{array}{cccc}\sum_{j=1}^na_{1j}&0&\cds&0\\
    0&\sum_{j=1}^na_{2j}&\cds&0\\\vds&\vds&\dds&\vds\\
    0&0&\cds&\sum_{j=1}^na_{nj}\end{array}\right)
    -\left(\begin{array}{cccc}a_{11}&a_{12}&\cds&a_{1n}\\ 
    \a_{21}&a_{22}&\cds&a_{1n}\\\vds&\vds&\dds&\vds\\
    \a_{n1}&a_{n2}&\cds&a_{nn}\end{array}\right)

.. note::

    :math:`=\left(\begin{array}{cccc}\sum_{j\ne 1}^na_{1j}&-a_{12}&\cds&-a_{1n}\\-a{21}&\sum_{j\ne 2}^na_{2j}&\cds&-a_{2n}\\\vds&\vds&\dds&\vds\\-a_{n1}&-a_{n2}&\cds&\sum_{j\ne n}^na_{nj}\end{array}\right)`

:math:`\bs{\rm{L}}` is a symmetric, positive semidefinite matrix, as for any :math:`\c\in\R^n`, we have

.. math::

    \c^T\bs{\rm{L}}\c&=\c^T(\Delta-\A)\c=\c^T\Delta\c-\c^T\A\c

    &=\sum_{i=1}^nd_ic_i^2-\sum_{i=1}^n\sum_{j=1}^nc_ic_ja_{ij}

    &=\frac{1}{2}\bigg(\sum_{i=1}^nd_ic_i^2-2\sum_{i=1}^n\sum_{j=1}^nc_ic_ja_{ij}+\sum_{j=1}^nd_jc_j^2\bigg)

    &=\frac{1}{2}\bigg(\sum_{i=1}^n\sum_{j=1}^na_{ij}c_i^2-2\sum_{i=1}^n
    \sum_{j=1}^nc_ic_ja_{ij}+\sum_{i=j}^n\sum_{i=1}^na_{ij}c_j^2\bigg)

    &=\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^na_{ij}(c_i-c_j)^2

    &\geq 0

This means that :math:`\bs{\rm{L}}` has :math:`n` real, non-negative 
eigenvalues, which can be arranged in decreasing order as follows:
:math:`\ld_1\geq\ld_2\geq\cds\geq\ld_n\geq 0`.
Because :math:`\bs{\rm{L}}` is symmetric, its eigenvectors are orthonormal.
We can observe that the first column (and the first row) is a linear combination of the remaining columns (rows).
This implies that the rank of :math:`\bs{\rm{L}}` is at most :math:`n-1`, and 
the smallest eigenvalue is :math:`\ld_n=0`, with the corresponding eigenvector
given as :math:`\u_n=\frac{1}{\sqrt{n}}=(1,1,\cds,1)^T=\frac{1}{\sqrt{n}}\1`,
provided the graph is connected.
If the graph is disconnected, then the number of eigenvalues equal to zero
specifies the number of connected components in the graph.

The *normalized symmetric Laplacian matrix* of a graph is defined as

.. math::

    \bs{\rm{L}}^S&=\Delta^{-1/2}\bs{\rm{L}}\Delta^{-1/2}

    &=\bs{\rm{L}}^{-1/2}(\Delta-\A)\Delta^{-1/2}=\Delta^{-1/2}\Delta\Delta^{-1/2}-\Delta^{-1/2}\A\Delta^{-1/2}

    &=\I-\Delta^{-1/2}\A\Delta^{-1/2}

.. note::

    :math:`\bs{\rm{L}}^S=\Delta^{-1/2}\bs{\rm{L}}\Delta^{-1/2}`
    :math:`\left(\begin{array}{cccc} \frac{\sum_{j\ne 1}a_{1j}}{\sqrt{d_1d_1}}&-\frac{a_{12}}{\sqrt{d_1d_2}}&\cds&-\frac{a_{1n}}{\sqrt{d_1d_n}}\\-\frac{a_{21}}{\sqrt{d_2d_1}}&\frac{\sum_{j\ne 2}a_{2j}}{\sqrt{d_2d_2}}&\cds&-\frac{a_{2n}}{\sqrt{d_2d_n}}\\\vds&\vds&\dds&\vds\\-\frac{a_{n1}}{\sqrt{d_nd_1}}&-\frac{a_{n2}}{\sqrt{d_nd_2}}&\cds&\frac{\sum_{j\ne n}a_{nj}}{\sqrt{d_nd_n}}\end{array}\right)`

We can hsow that :math:`\bs{\rm{L}}^S` is also positive semidefinite because for any :math:`\c\in\R^d`, we get

.. math::

    \c^T\bs{\rm{L}}^s\c=\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^na_{ij}
    \bigg(\frac{c_i}{\sqrt{d_i}}-\frac{c_j}{\sqrt{d_j}}\bigg)^2\geq 0

The first column is also a linear combination of the other columns, which means 
that :math:`\bs{\rm{L}}^S` has rank at most :math:`n-1`, with the smallest 
eigenvalue :math:`\ld_n=0`, and the corresponding eigenvector 
:math:`\frac{1}{\sqrt{\sum_id_i}}(\sqrt{d_1},\sqrt{d_2},\cds,\sqrt{d_n})^T=\frac{1}{\sqrt{\sum_id_i}}\Delta^{1/2}\1`.
Combined with the fact that :math:`\bs{\rm{L}}^S` is positive semidefinite, we 
conclude that :math:`\bs{\rm{L}}^S` has :math:`n` (not necessarily distinct) 
real, positive eigenvalues :math:`\ld_1\geq\ld_2\geq\cds\geq\ld_n=0`.

The *normalized asymmetric Laplacian* matrix is defined as

.. note::

    :math:`\bs{\rm{L}}^a=\Delta\im\bs{\rm{L}}=\Delta\im(\Delta-\A)=\I-\Delta\im\A`

    :math:``