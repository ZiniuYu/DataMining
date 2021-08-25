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
    \alpha_i\in\R,\{\x_1,\cds,\x_m\}\seq\cl{I}\bigg\}

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

**Polynomial Kernel**

Polynomial kernels are of two types: homogeneous or inhomogeneous.
Let :math:`\x,\y\in\R^d`.
The *homogeneous polynomial kernel* is defined as

.. note::

    :math:`K_q(\x,\y)=\phi(\x)^T\phi(\y)=(\x^T\y)^q`

where :math:`q` is the degree of the polynomial.

The most typical cases are the *linear* (with :math:`q=1`) and *quadratic* (with :math:`q=2`) kernels, given as

.. math::

    K_1(\x,\y)=\x^T\y

    K_2(\x,\y)=(\x^T\y)^2

The *inhomogeneous polynomial kernel* is defined as

.. note::

    :math:`k_q(\x,\y)=\phi(\x)^T\phi(\y)=(c+\x^T\y)^q`

where :math:`q` is the degree of the polynomial, and :math:`c\geq 0` is some constant.

.. math::

    K_q(\x,\y)=(c+\x^T\y)^q=\sum_{k=0}^q\bp q\\k \ep c^{q-k}(\x^T\y)^k

Let :math:`n_0,n_1,\cds,n_d` denote non-negative integers, such that :math:`\sum_{i=0}^dn_i=q`.
Further, let :math:`\n=(n_0,n_1,\cds,n_d)`, and let :math:`|\n|=\sum_{i=0}^dn_i=q`.
Also, let :math:`\bp q\\\n \ep` denote the multinomial coefficient

.. math::

    \bp q\\\n \ep=\bp q\\n_0,n_1,\cds,n_d \ep=\frac{q!}{n_0!n_1!\cds n_d!}

The multinomial expansion of the inhomogeneous kernel is then given as

.. math::

    K_q(\x,\y)=(c+\x^T\y)^q=\bigg(c+\sum_{k=1}^dx_ky_k\bigg)^q=(c+x_1y_1+\cds+x_dy_d)^q

    =\sum_{|\n|=q}\bp q\\\n \ep c^{n_0}(x_1y_1)^{n_1}(x_2y_2)^{n_2}\cds(x_dy_d)^{n_d}

    =\sum_{|\n|=q}\bp q\\\n \ep c^{n_0}(x_1^{n_1}x_2^{n_2}\cds x_d^{n_d})(y_1^{n_1}y_2^{n_2}\cds y_d^{n_d})

    =\sum_{|\n|=q}\bigg(\sqrt{a_\n}\prod_{k=1}^dx_k^{n_k}\bigg)\bigg(\sqrt{a_\n}\prod_{k=1}^dy_k^{n_k}\bigg)

    =\phi(\x)^T\phi(\y)

Using the notation :math:`\x^\n=\prod_{k=1}^dx_k^{n_k}`, the mapping :math:`\phi:\R^d\ra\R^m` is given as the vector

.. math::

    \phi(\x)=(\cds,a_\n\x^\n,\cds)^T=\left(\cds,\sqrt{\bp q\\\n \ep c^{n_0}}\prod_{k=1}^dx_k^{n_k},\cds\right)^T

It can be shown that the dimensionality of the feature space is given as

.. math::

    m=\bp d+q\\q \ep

**Gaussian Kernel**

The Gaussian kernel, also called the Gaussian radial basis function (RBF) kernel, is defined as

.. note::

    :math:`\dp K(\x,\y)=\exp\bigg\{-\frac{\lv\x-\y\rv^2}{2\sg^2}\bigg\}`

where :math:`\sg>0` is the spread parameter that plays the same role as the 
standard deviation in a normal density function.
Note that :math:`K(\x,\x)=1`, and further that the kernel value is inversely 
related to the distance between the two points :math:`\x` and :math:`\y`.

A feature space for the Gaussian kernal has infinite dimensionality.

.. math::

    \exp\{a\}=\sum_{n=0}^\infty\frac{a^n}{n!}=1+a+\frac{1}{2!}a^2+\frac{1}{3!}a^3+\cds

Further, using :math:`\gamma=\frac{1}{2\sg^2}`, and noting that 
:math:`\lv\x-\y\rv^2=\lv\x\rv^2+\lv\y\rv^2-2\x^T\y`, we can rewrite the Gaussian
kernel as follows:

.. math::

    K(\x,\y)&=\exp\{-\gamma\lv\x-\y\rv^2\}

    &=\exp\{-\gamma\lv\x\rv^2\}\cd\exp\{-\gamma\lv\y\rv^2\}\cd\exp\{2\gamma\x^T\y\}

In particular, the last term is given as the infinite expansion

.. math::

    \exp\{2\gamma\x^T\y\}=\sum_{q=0}^\infty\frac{(2\gamma)^q}{q!}(\x^T\y)^q=
    1+(2\gamma)\x^T\y+\frac{(2\gamma)^2}{2!}(\x^T\y)^2+\cds

Using the multinomial expansion of :math:`(\x^T\y)^q`, we can write the Gaussian kernel as

.. math::

    K(\x,\y)=\exp\{-\gamma\lv\x\rv^2\}\exp\{-\gamma\lv\y\rv^2\}
    \sum_{q=0}^\infty\frac{(2\gamma)^q}{q!}\bigg(\sum_{|\n|=q}\bp q\\\n \ep
    \prod_{k=1}^d(x_ky_k)^{n_k}\bigg)

    =\sum_{q=0}^\infty\sum_{|\n|=q}
    \bigg(\sqrt{a_{q,\n}}\exp\{-\gamma\lv\x\rv^2\}\prod_{k=1}^dx_k^{n_k}\bigg)
    \bigg(\sqrt{a_{q,\n}}\exp\{-\gamma\lv\y\rv^2\}\prod_{k=1}^dy_k^{n_k}\bigg)

    =\phi(\x)^T\phi(\y)

where

.. math::
    
    \dp a_{q,\n}=\frac{(2\gamma)^q}{q!}\bp q\\\n \ep

    \n=(n_1,n_2,\cds,n_d)

    |\n|=n_1+n_2+\cds+n_d=q

The mapping into feature space corresponds to the function :math:`\phi:\R^d\ra\R^\infty`

.. math::

    \phi(\x)=\left(\cds,\sqrt{\frac{(2\gamma)^q}{q!}\bp q\\\n \ep}
    \exp\{-\gamma\lv\x\rv^2\}\prod_{k=1}^dx_k^{n_k},\cds\right)^T

5.3 Basic Kernel Operations in Feature Space
--------------------------------------------

**Norm of a Point**

.. note::

    :math:`\lv\phi(\x)\rv^2=\phi(\x)^T\phi(\x)=K(\x,\x)`

**Distance between Points**

The squared distance between two points :math:`\phi(\x_i)` and :math:`\phi(\x_j)` can be computed as

.. note::

    :math:`\lv\phi(\x_i)-\phi(\x_j)\rv^2=\lv\phi(\x_i)\rv^2+\lv\phi(\x_j)\rv^2-2\phi(\x_i)^T\phi(\x_j)`
    :math:`=K(\x_i,\x_i)+K(\x_j,\x_j)-2K(\x_i,\x_j)`

which implies that the distance is

.. math::

    \lv\phi(\x_i)-\phi(\x_j)\rv=\sqrt{K(\x_i,\x_i)+K(\x_j,\x_j)-2K(\x_i,\x_j)}

The kernel value can be considered as a measure of the similarity between two points, as

.. math::

    \frac{1}{2}(\lv\phi(\x_i)\rv^2+\lv\phi(\x_j)\rv^2-\lv\phi(\x_i)-\phi(\x_j)\rv^2)=K(\x_i,\x_j)=\phi(\x_i)^T\phi(\x_j)

Thus, the more the distance :math:`\lv\phi(\x_i)-\phi(\x_j)\rv` between the two 
points in feature space, the less the kernel value, that is, the less the 
similarity.

**Mean in Feature Space**

.. math::

    \mmu_\phi=\frac{1}{n}\sum_{i=1}^n\phi(\x_i)

Because we do not, in general, have access to :math:`\phi(\x_i)`, we cannot 
explicity compute the mean point in feature space

.. math::

    \lv\mmu_\phi\rv^2=\mmu_\phi^T\mmu_\phi=\bigg(\frac{1}{n}\sum_{i=1}^n
    \phi(\x_i)\bigg)^T\bigg(\frac{1}{n}\sum_{j=1}^n\phi(\x_j)\bigg)=
    \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n\phi(\x_i)^T\phi(\x_j)

which implies that

.. note::

    :math:`\dp\lv\mmu_\phi\rv^2=\frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^nK(\x_i,\x_j)`

**Total Variance in Feature Space**

.. note::

    :math:`\lv\phi(\x_i)-\mmu_\phi\rv^2=\lv\phi(\x_i)\rv^2-2\phi(\x_i)^T\mmu_\phi+\lv\mmu_\phi\rv^2`

    :math:`\dp=K(\x_i,\x_i)-\frac{2}{n}\sum_{j=1}^nK(\x_i,\x_j)+\frac{1}{n^2}\sum_{a=1}^n\sum_{b=1}^nK(\x_a,\x_b)`

.. math::

    \sg_\phi^2&=\frac{1}{n}\sum_{i=1}^n\lv\phi(\x_i)-\mmu_\phi\rv^2

    &=\frac{1}{n}\sum_{i=1}^n\bigg(K(\x_i,\x_i)-\frac{2}{n}\sum_{j=1}^n
    K(\x_i,\x_j)+\frac{1}{n^2}\sum_{a=1}^n\sum_{b=1}^nK(\x_a,\x_b)\bigg)

    &=\frac{1}{n}\sum_{i=1}^nK(\x_i,\x_i)-\frac{2}{n^2}\sum_{i=1}^n\sum_{j=1}^n
    K(\x_i,\x_j)+\frac{n}{n^3}\sum_{a=1}^n\sum_{b=1}^nK(\x_a,\x_b)

That is

.. note::

    :math:`\dp\sg_\phi^2=\frac{1}{n}\sum_{i=1}^nK(\x_i,\x_i)-\frac{2}{n^2}\sum_{i=1}^n\sum_{j=1}^nK(\x_i,\x_j)`

**Centering in Feature Space**

We can center each point in feature space by subtracting the mean from it, as follows:

.. math::

    \bar\phi(\x_i)=\phi(\x_i)-\mmu_\phi

The *centered kernel matrix* is given as

.. math::

    \bar\K=\{\bar{K}(\x_i,\x_j)\}_{i,j=1}^n

where each cell corresponds to the kernel between centered points, that is

.. math::

    \bar{K}(\x_i,\x_j)&=\bar\phi(\x_i)^T\bar\phi(\x_j)
    
    &=(\phi(\x_i)-\mmu_\phi)^T(\phi(\x_j)-\mmu_\phi)

    &=\phi(\x_i)^T\phi(\x_j)-\phi(\x_i)^T\mmu_\phi-\phi(\x_j)^T\mmu_\phi+\mmu_\phi^T\mmu_\phi

    &=K(\x_i,\x_j)-\frac{1}{n}\sum_{k=1}^n\phi(\x_i)^T\phi(\x_k)-
    \frac{1}{n}\sum_{k=1}^n\phi(\x_j)^T\phi(\x_k)+\lv\mmu_\phi\rv^2

    &=K(\x_i,\x_j)-\frac{1}{n}\sum_{k=1}^nK(\x_i,\x_k)-\frac{1}{n}\sum_{k=1}^n
    K(\x_j,\x_k)+\frac{1}{n^2}\sum_{a=1}^n\sum_{b=1}^nK(\x_a,\X_b)

The centered kernel matrix can be written campactly as follows:

.. note::

    :math:`\dp\bar\K=\K-\frac{1}{n}\1_{n\times n}\K-\frac{1}{n}\K\1_{n\times n}`
    :math:`\dp+\frac{1}{n^2}\1_{n\times n}\K\1_{n\times n}=`
    :math:`\dp\bigg(\bs{\rm{I}}-\frac{1}{n}\1_{n\times n}\bigg)\K\bigg(\bs{\rm{I}}-\frac{1}{n}\1_{n\times n}\bigg)`

**Normalizing in Feature Space**

The dot product in feature space corresponds to the cosine of the angle between the two mapped points, because

.. math::

    \phi_n(\x_i)^T\phi_n(\x_j)=\frac{\phi(\x_i)^T\phi(\x_j)}{\lv\phi(\x_i)\rv\cd\lv\phi(\x_j)\rv}=\cos\th

The normalized kernel matrix :math:`\K_n` can be computed using only the kernel function :math:`K`, as

.. note::

    :math:`\dp\K_n(\x_i,\x_j)=\frac{\phi(\x_i)^T\phi(\x_j)}{\lv\phi(\x_i)\rv\cd\lv\phi(\x_j)\rv}=`
    :math:`\dp\frac{K(\x_i,\x_j)}{\sqrt{K(\x_i,\x_i)\cd K(\x_j,\x_j)}}`

Let :math:`\W` denote the diagonal matrix comprising the diagonal elements of :math:`\K`:

.. math::

    \W=\rm{diag}(\K)=\bp K(\x_1,\x_1)&0&\cds&0\\0&K(\x_2,\x_2)&\cds&0\\\vds&\vds&\dds&\vds\\0&0&\cds&K(\x_n,\x_n) \ep

The normalized kernel matrix can then be expressed compactly as

.. math::

    \K_n=\W^{-1/2}\cds\K\cd\W^{-1/2}

5.4 Kernels for Complex Objects
-------------------------------

5.4.1 Spectrum Kernel for Strings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider text or sequence data defined over an alphabet :math:`\Sg`.
The :math:`l`-spectrum feature map is the mapping 
:math:`\phi:\Sg^*\ra\R^{|\Sg|^l}` from the set of substrings over :math:`\Sg` to 
the :math:`|\Sg|^l`-dimensional space representing the number of occurrences of 
all possible substrings of length :math:`l`, defined as

.. math::

    \phi(\x)=(\cds,\#(\alpha),\cds)_{\alpha\in\Sg^l}^T

where :math:`\#(\alpha)` is the number of occurrences of the :math:`l`-length string :math:`\alpha` in :math:`\x`.

The (full) spectrum map is an extension of the :math:`l`-spectrum map, obtained 
by considering all lengths from :math:`l=0` to :math:`l=\infty`, leading to an
infinite dimensional feature map :math:`\phi:\Sg^*\ra\R^\infty`:

.. math::

    \phi(\x)=(\cds,\#(\alpha),\cds)_{\alpha\in\Sg^*}^T

where :math:`\#(\alpha)` is the number of occurrences of the string :math:`\alpha` in :math:`\x`.

The (:math:`l`-)spectrum kernel between two string :math:`\x_i,\x_j` is simply 
the dot product between their (:math:`l`-)spectrum maps:

.. math::

    K(\x_i,\x_j)=\phi(\x_i)^T\phi(\x_j)

5.4.2 Diffusion Kernels on Graph Nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math`\S` be some symmetric similarity matrix between nodes of a graph :math:`G=(V,E)`.
Consider the similarity between any two nodes obtained by summing the product of the similarities over walks of length 2:

.. math::

    S^{(2)}(\x_i,\x_j)=\sum_{a=1}^nS(\x_i,\x_a)S(\x_a,\x_j)=\S_i^T\S_j

where

.. math::

    \S_i=(S(\x_i,\x_1),S(\x_i,\x_2),\cds,S(\x_i,\x_n))^T

denotes the (column) vector representing the :math:`i`th row of :math:`S`.
Over all pairs of nodes the similarity matrix over walks of length 2, denoted 
:math:`\S^{(2)}`, is thus given as the square of the base similarity matrix 
:math:`\S`:

.. math::

    \S^{(2)}=\S\times\S=\S^2

In general, if we sum up the product of the base similarities over all :math:`l`
-length walks between two nodes, we obtain the :math:`l`-length similarity
matrix :math:`\S^{(l)}`, which is simply the :math:`l`th power of :math:`\S`,
that is,

.. math::

    \S^{(l)}=\S^l

**Power Kernels**

The kernel value between any two points is then a dot product in feature space:

.. math::

    K(\x_i,\x_j)=S^{(2)}(\x_i,\x_j)=\S_i^T\S_j=\phi(\x_i)^T\phi(\x_j)

For a general walk length :math:`l`, let :math:`\K=\S^l`.
Consider the eigen-decomposition of :math:`\S`:

.. math::

    \S=\U\Ld\U^T=\sum_{i=1}^n\u_i\ld_i\u_i^T

The eigen-decomposition of :math:`\K` can be obtained as follows:

.. math::

    \K=\S^l=(\U\Ld\U^T)^l=\U(\Ld^l)\U^T

**Exponential Diffusion Kernel**

Instead of fixing the walk length *a priori*, we can obtain a new kernel between 
nodes of a graph by considering walks of all possible lengths, but by damping
the contribution of longer walks, which leads to the 
*exponential diffusion kernel*, defined as

.. note::

    :math:`\dp\K=\sum_{l=0}^\infty\frac{1}{l!}\beta^l\S^l=`
    :math:`\dp\I+\beta\S+\frac{1}{2!}\beta^2\S^2+\frac{1}{3!}\beta^3\S^3+\cds=\exp\{\beta\S\}`

where :math:`\beta` is a damping factor, and :math:`\exp\{\beta\S\}` is the matrix exponential.

.. math::

    \K&=\I+\beta\S+\frac{1}{2!}\beta^2\S^2+\cds

    &=\bigg(\sum_{i=1}^n\u_i\u_i^T\bigg)+\bigg(\sum_{i=1}^n\u_i\beta\ld_i\u_i^T
    \bigg)+\bigg(\sum_{i=1}^n\u_i\frac{1}{2!}\beta^2\ld_i^2\u_i^T\bigg)+\cds

    &=\sum_{i=1}^n\u_i(1+\beta\ld_i+\frac{1}{2!}\beta^2\ld_i^2+\cds)+\u_i^T

    &=\sum_{i=1}^n\u_i\exp\{\beta\ld_i\}\u_i^T

.. math::

    =\U\bp\exp\{\beta\ld_1\}&0&\cds&0\\0&\exp\{\beta\ld_2\}&\cds&0\\
    \vds&\vds&\dds&\vds\\0&0&\cds&\exp\{\beta\ld_n\}\ep\U^T\quad\quad\quad\quad

**Von Neumann Diffusion Kernel**

A related kernel based on powers of :math:`\S` is the *von Neumann diffusion kernel*, defined as

.. note::

    :math:`\dp\K=\sum_{l=0}^\infty\beta^l\S^l`

.. math::

    \K&=\I+\beta\S+\beta^2\S^2+\beta^3\S^3+\cds

    &=\I+\beta\S(\I+\beta\S+\beta^2\S^2+\cds)

    &=\I+\beta\S\K

Rearranging the terms to obtain a closed form expression for the von Neumann kernel:

.. math::

    \K-\beta\S\K&=\I

    (\I-\beta\S)\K&=\I

    \K&=(\I-\beta\S)\im

    \K&=(\U\U^T-\U(\beta\Ld)\U^T)\im

    &=(\U(\I-\beta\Ld)\U^T)\im

    &=\U(\I-\beta\Ld)\im\U^T

For :math:`\K` to be a positive semidefinite kernel, all its eigenvalues should 
be non-negative, which in turn implies that

.. math::

    (1-\beta\ld_i)\im&\geq 0

    1-\beta\ld_i&\geq 0

    \beta&\leq 1/\ld_i

Further, the inverse matrix :math:`(\I-\beta\Ld)\im` exists only if

.. math::

    \det(\I-\beta\Ld)=\prod_{i=1}^n(1-\beta\ld_i)\neq 0