Chapter 5 Kernel Methods
========================

Given a data instance :math:`\x`, we need to find a mapping :math:`\phi`, so 
that :math:`\phi(\x)` is the vector representation of :math:`\x`.
Even when the input data is a numeric data matrix, if we wish to discover 
nonlinear relationships among the attributes, then a nonlinear mapping 
:math:`\phi` may be used, so that :math:`\phi(\x)` represents a vector in the 
corresponding high-dimensional space comprising nonlinear attributes.
We use the *input space* to reefer to the data space for the input data 
:math:`\x` and *feature space* to refer to the space of mapped vectors 
:math:`\phi(\x)`.

Kernel methods avoid explicitly transforming each point :math:`\x` in the input
space into the mapped point :math:`\phi(\x)` in the feature space.
Instead, the input objects are represented via their :math:`n\times n` pairwise similarity values.
The similarity function, called a *kernel*, is chosen so that it represents a 
dot product in some high-dimensional feature space, yet it can be computed 
without directly constructing :math:`\phi(\x)`.
Let :math:`\cl{I}` denote the input space, and let :math:`\D\subset\cl{I}` be a 
dataset comprising :math:`n` objects :math:`\x_i\ (i=1,2,\cds,n)` in the input 
space.
We can represent the pairwise similarity values between points in :math:`\D` via the :math:`n\times n` *kernel matrix*

.. math::

    \K=\bp K(\x_1,\x_1)&K(\x_1,\x_2)&\cds&K(\x_1,\x_n)\\
    K(\x_2,\x_1)&K(\x_2,\x_2)&\cds&K(\x_2,\x_n)\\\vds&\vds&\dds&\vds\\
    K(\x_m,\x_1)&K(\x_n,\x_2)&\cds&K(\x_n,\x_n) \ep

where :math:`K:\cl{I}\times\cl{I}\ra\R` is a *kernel function* on any two points in input space.
For any :math:`\x_i,\x_j\in\cl{I}`, the kernel function should satisfy the condition

.. note::

    :math:`K(\x_i,\x_j)=\phi(\x_i)^T\phi(\x_j)`

Intuitively, this means that we should be able to compute the value of the dot 
product using the original input representation :math:`\x`, without having 
recourse to the mapping :math:`\phi(\x)`.

many data mining methods can be *kernelized*, that is, instead of mapping the
input points into feature space, the data can be represented via the 
:math:`n\times n` kernel matrix :math:`\K`, and all relevant analysis can be
performed over :math:`\K`.
This is usually done via the so-called *kernel trick*, that is, show that the
analysis task requires only dot products :math:`\phi(\x_i)^T\phi(\x_j)` that can
be computed efficiently in input space.
Once the kernel matrix has been computed, we no longer even need the input 
points :math:`\x_i`, as all operations involving only dot products inthe feature
space can be performed over the :math:`n\times n` kernel matrix :math:`\K`.

5.1 Kernel Matrix
-----------------

Let :math:`\cl{I}` denote the input space, which can be any arbitrary set of
data objects, and let :math:`\D\subset\cl{I}` denote a subset of :math:`n` 
objects :math:`\x_i` in the input space.
Let :math:`\phi:\cl{I}\ra\cl{F}` be a mapping from the input space into the 
feature space :math:`\cl{F}`, which is endowed with a dot product and norm.
Let :math:`K:\cl{I}\times\cl{I}\ra\R` be a function that maps pairs of input 
objects to their dot product value in feature space, that is, 
:math:`K(\x_i,\x_j)=\phi(\x_i)^T\phi(\x_j)`, and let :math:`\K` be the 
:math:`n\times n` kernel matrix corresponding to the subset :math:`\D`.

The function :math:`K` is called a **positive semidefinite kernel** if and only if it is symmetric:

.. math::

    K(\x_i,\x_j)=K(\x_j,\x_i)

and the corresponding kernel matrix :math:`\K` for any subset :math:`\D\subset\cl{I}` is positive semidefinite, that is,

.. math::

    \a^T\K\a\geq 0,\rm{\ for\ all\ vectors\ }\a\in\R^n

which implies that

.. math::

    \sum_{i=1}^n\sum_{j=1}^na_ia_jK(\x_i,\x_j)\geq 0,\rm{\ for\ all\ }a_i\in\R,i\in[1,n]

:math:`K` is symmetric since the dot product is symmetric, which also implies that :math:`\K` is symmetric.
:math:`\K` is positive semidefinite because

.. math::

    \a^T\K\a&=\sum_{i=1}^n\sum_{j=1}^na_ia_jK(\x_i,\x_j)

    &=\sum_{i=1}^n\sum_{j=1}^na_ia_j\phi(\x_i)^T\phi(\x_j)

    &=\bigg(\sum_{i=1}^na_i\phi(\x_i)\bigg)^T\bigg(\sum_{j=1}^na_j\phi(\x_j)\bigg)

    &=\lv\sum_{i=1}^na_i\phi(\x_i)\rv^2\geq 0

5.1.1 Reproducing Kernel Map
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the reproducing kernel map :math:`\phi`, we map each point 
:math:`\x\in\cl{I}` into a function in a *function space* 
:math:`\{f:\cl{I}\ra\R\}` comprising functions that map points in :math:`\cl{I}`
into :math:`\R`.
Any :math:`\x\in\R` in the input space is mapped to the following function:

.. math::

    \phi(\x)=K(\x,\cd)

where the :math:`\cd` stands for any argument in :math:`\cl{I}`.
That is, each object :math:`\x` in the input space gets mapped to a 
*feature point* :math:`\phi(\x)`, which is in fact a function :math:`K(\x,\cd)`
that represents its similarity to all other points in the input space 
:math:`\cl{I}`.

Let :math:`\cl{F}` be the set of all functions or points that can be obtained as
a linear combination of any subset of feature points, defined as

.. math::

    \cl{F}&=span\{K(\x,\cd)|\x\in\cl{I}\}

    &=\bigg\{\bs{\rm{f}}=f(\cd)=\sum_{i=1}^m\alpha_iK(\x_i,\cd)|m\in\mathbb{N},
    \alpha_i\in\R,\{\x_1,\cds,\x_m\}\subseteq\cl{I}\bigg\}

Let :math:`\f,\g\in\cl{F}` be any two points in feature space:

.. math::

    \f=f(\cd)=\sum_{i=1}^{m_a}\alpha_iK(\x_i,\cd)\quad\g=g(\cd)=\sum_{j=1}^{m_b}\beta_jK(\x_j,\cd)

Define the dot product between two points as

.. math::

    \f^T\g=f(\cd)^T g(\cd)=\sum_{i=1}^{m_a}\sum_{j=1}^{m_b}\alpha_i\beta_jK(\x_i,\x_j)

The dot product is *bilinear*, that is, linear in both arguments, because

.. math::

    \f^T\g=\sum_{i=1}^{m_a}\sum_{j=1}^{m_b}\alpha_i\beta_jK(\x_i,\x_j)=
    \sum_{i=1}^{m_a}\alpha_ig(\x_i)=\sum_{j=1}^{m_b}\beta_jf(\x_j)

The fact that :math:`K` is positive semidefinite implies that

.. math::

    \lv\f\rv^2=\f^T\f=\sum_{i=1}^{m_a}\sum_{j=1}^{m_a}\alpha_i\alpha_jK(\x_i,\x_j)\geq 0

Thus, the space :math:`\cl{F}` is a *pre-Hilbert space*,

The space :math:`\cl{F}` has the so-called *reproducing property*, that is, we
can evaluate a function :math:`f(\cd)=\f` at a point :math:`\x\in\cl{I}` by
taking the dot product of :math:`\f` with :math:`\phi(\x)`, that is

.. math::

    \f^T\phi(\x)=f(\cd)^TK(\x,\cd)=\sum_{i=1}^{m_a}\alpha_iK(\x_i,\x)=f(\x)

For this reason, the space :math:`\cl{F}` is also called a *reproducing kernel Hilbert space*.

.. math::

    \phi(\x_i)^T\phi(\x_j)=K(\x_i,\cd)^TK(\x_j,\cd)=K(\x_i,\x_j)

The reproducing kernel map shows that any positive semidefinite kernel 
corresponds to a dot product in some feature space. 
This means we can apply well known algebraic and geometric methods to understand and analyze the data in these spaces.

**Empirical Kernel Map**

Define the map :math:`\phi` as follows:

.. math::

    \phi(\x)=(K(\x_1,\x),K(\x_2,\x),\cds,K(\x_n,\x))^T\in\R^n

Define the dot product in feature space as

.. math::

    \phi(\x_i)^T\phi(\x_j)=\sum_{k=1}^nK(\x_k,\x_i)K(\x_k,\x_j)=\K_i^T\K_j

For :math:`\phi` to be a valid map, we require that 
:math:`\phi(\x_i)^T\phi(\x_j)=K(\x_i,\x_j)`, which is clearly not satisfied.
One solution is to replace :math:`\K_i^T\K_j` with :math`\K_i^T\A\K_j` for some 
positive semidefinite matrix :math:`\A` such that

.. math::

    \K_i^T\A\K_j=\K(\x_i,\x_j)

If we can find such an :math:`\A`, it would imply that over all pairs of mapped points we have

.. math::

    \{\K_i^T\A\K_j\}_{i,j=1}^n=\{K(\x_i,\x_j)\}_{i,j=1}^N

which can be written compactly as

.. math::

    \K\A\K=\K

This immediately suggests that we take :math:`\A=\K\im`, the (pseudo)inverse of the kernel matrix :math:`K`.
The modified map :math:`\phi`, called the *empirical kernel map*, is then defined as

.. math::

    \phi(\x)=\K^{-1/2}\cd(K(\x_1,\x),K(\x_2,\x),\cds,K(\x_n,\x))^T\in\R^n

so that the dot product yields

.. math::

    \phi(\x_i)^T\phi(\x_j)&=(\K^{-1/2}\K_i)^T(\K^{-1/2}\K_j)

    &=\K_i^T(\K^{-1/2}\K^{-1/2})\K_j

    &=\K_i^T\K\im\K_j

Over all pairs of mapped points, we have

.. math::

    \{\K_i^T\K\im\K_j\}_{i,j=1}^n=\K\K\im\K=\K

It is important to note that this empirical feature representation is valid only for the :math:`n` points in :math:`\D`.
If points are added to or removed from :math:`\D`, the kernel map will have to be updated for all points.

5.1.2 Mercer Kernel Map
^^^^^^^^^^^^^^^^^^^^^^^

**Data-specific Kernel Map**

Because :math:`\K` is a symmetric positive semidefinite matrix, it has real and 
non-negative eigenvalues, and it can be decomposed as follows:

.. math::

    \K=\U\Ld\U^T

where :math:`\U` is the orthonormal matrix of eigenvectors 
:math:`\u_i=(u_{i1},u_{i2},\cds,u_{in})^T\in\R^n`, and :math:`\Ld` is the 
diagonal matrix of eigenvalues, with both arranged in non-increasing order of
the eigenvalues :math:`\ld_1\geq\ld_2\geq\cds\geq\ld_n\geq 0`:

.. math::

    \U=\bp |&|&&|\\\u_1&\u_2&\cds&\u_n\\|&|&&| \ep\quad
    \Ld=\bp\ld_1&0&\cds&0\\0&\ld_2&\cds&0\\\vds&\vds&\dds&\vds\\0&0&\cds&\ld_n\ep

The kernel matrix :math:`\K` can therefore be rewritten as the spectral sum

.. math::

    \K=\ld_1\u_1\u_1^T+\ld_2\u_2\u_2^T+\cds+\ld_n\u_n\u_n^T

In particular the kernel function between :math:`\x_i` and :math:`\x_j` is given as

.. math::

    \K(\x_i,\x_j)=\ld_1u_{1i}u_{1j}+\ld_2u_{2i}u_{2j}+\cds+\ld_nu_{ni}u_{nj}=\sum_{k=1}^n\ld_ku_{ki}u_{kj}

The Mercer map :math:`\phi` is defined as follows:

.. math::

    \phi(\x_i)=(\sqrt{\ld_1}u_{1i},\sqrt{\ld_2}u_{2i},\cds,\sqrt{\ld_n}u_{ni})^T

then :math:`\K(\x_i,\x_j)` is a dot product in feature space between the mapped 
points :math:`\phi(\x_i)` and :math:`\phi(\x_j)` because

.. math::

    \phi(\x_i)^T\phi(\x_j)&=(\sqrt{\ld_1}u_{1i},\cds,\sqrt{\ld_n}u_{ni})(\sqrt{\ld_1}u_{1i},\cds,\sqrt{\ld_n}u_{ni})^T

    &=\ld_1u_{1i}u_{1j}+\cds+\ld_nu_{ni}u_{nj}=\K(\x_i,\x_j)

We can rewrite the Mercer map :math:`\phi` as 

.. math::

    \phi(\x_i)=\sqrt\Ld\U_i

The kernel value is simply the dot product between scaled rows of :math:`\U`:

.. math::

    \phi(\x_i)^T\phi(\x_j)=(\sqrt\Ld\U_i)^T(\sqrt\Ld\U_j)=\U_i^T\Ld\U_j

The Mercer map restricted to the input dataset :math:`\D` is called the *data-specific Mercer kernel Map*.

**Mercer Kernel Map**

For compact continuous spaces, the kernel value between any two points can be 
written as the infinite spectral decomposition

.. math::

    K(\x_i,\x_j)=\sum_{k=1}^\infty\ld_k\u_k(\x_i)\u_k(\x_j)

where each normalized *eigenfunction* :math:`\u_i(\cd)` is a solution to the integral equation

.. math::

    \int K(\x,\y)\u_i(\y)d\y=\ld_i\u_i(\x)

and :math:`K` is a continuous positive semidefinite kernel, that is, for all 
functions :math:`a(\cd)` with a finite square integral :math:`K` satisfies the
condition

.. math::

    \iint K(\x_1,\x_2)a(\x_1)a(\x_2)d\x_1d\x_2\geq 0

The general Mercer kernel map is given as

.. math::

    \phi(\x_i)=(\sqrt{\ld_1}\u_1(\x_i),\sqrt{\ld_2}\u_2(\x_i),\cds)^T

with the kernel value being equivalent to the dot product between two mapped points:

.. math::

    K(\x_i,\x_j)=\phi(\x_i)^T\phi(\x_j)

5.2 Vector Kernels
------------------




















5.3 Basic Kernel Operations in Feature Space
--------------------------------------------




















5.4 Kernels for Complex Objects
-------------------------------