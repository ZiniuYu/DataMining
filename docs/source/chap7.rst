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

Principal Component Analysis (PCA) is a technique that seeks a :math:`r`
-dimensional basis that best captures the variance in the data.

7.2.1 Best Line approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assume that :math:`\u` is a unit vector, and the data matrix :math:`\D` has been 
centered by subtracting the mean :math:`\mu`.

.. math::

    \bar\D=\D-\1\cd\mmu^T

The projection of the centered point :math:`\bar\x_i\in\bar\D` on the vector :math:`\u` is given as

.. math::

    \x_i\pr=\bigg(\frac{\u^T\bar\x_i}{\u^T\u}\bigg)\u=(\u^T\bar\x_i)\u=a_i\u

where

.. note::

    :math:`a_i=\u^T\bar\x_i`

is the offset or scalar projection of :math:`\x_i` on :math:`\u`.
We also call :math:`a_i` a *projected point*.
Note that the scalar projection of the mean :math:`\bar\mmu` is 0.
Therefore, the mean of the projected points :math:`a_i` is also zero, since

.. math::

    \mu_a=\frac{1}{n}\sum_{i=1}^na_i=\frac{1}{n}\sum_{i=1}^n\u^T(\bar\x_i)=\u^T\bar\mmu=0

We have to choose the direction :math:`\u` such that the variance of the projected points is maximized.
The projected variance along :math:`\u` is given as

.. math::

    \sg_\u^2=\frac{1}{n}\sum_{i=1}^n(a_i-\mu_a)^2=\frac{1}{n}\sum_{i=1}^n
    (\u^T\bar\x_i)^2=\frac{1}{n}\sum_{i=1}^n\u^T(\bar\x_i\bar\x_i^T)\u=\u^T
    \bigg(\frac{1}{n}\sum_{i=1}^n\bar\x_i\bar\x_i^T\bigg)\u

Thus, we get

.. note::

    :math:`\sg_\u^2=\u^T\Sg\u`

where :math:`\Sg` is the sample covariance matrix for the centered data :math:`\bar\D`.

We have to find the optimal basis vector :math:`\u` that maximizes the projected
variance :math:`\sg_\u^2` subject to the constraint that :math:`\u^T\u=1`.
This can be solved by introducing a Lagrangian multiplier :math:`\alpha` for the
constraint, to obtain the unconstrained maximization problem

.. math::

    \max_\u J(\u)=\u^T\Sg\u-\alpha(\u^T\u-1)

Setting the derivative of :math:`J(\u)` with respect to :math:`\u` to the zero vector, we obtain

.. math::

    \frac{\pd}{\pd\u}J(\u)&=\0

    \frac{\pd}{\pd\u}(\u^T\Sg\u-\alpha(\u^T\u-1))&=\0

    2\Sg\u-2\alpha\u&=\0

.. note::

    :math:`\Sg\u=\alpha\u`

.. math::

    \u^T\Sg\u=\u^T\alpha\u=\alpha\u^T\u=\alpha

The dominant eigenvector :math:`\u_1` specifies the direction of most variance,
also called the *first principal component*, that is, :math:`\u=\u_1`.
Further, the largest eigenvalue :math:`\ld_1` specifies the projected variance, that is, :math:`\sg_\u^2=\alpha=\ld_1`.

**Minimum Squared Error Approach**

The direction that maximizes the projected variance is also the one that minimizes the average squared error.
The mean squared error (MSE) optimization condition is defined as

.. math::

    MSE(\u)&=\frac{1}{n}\sum_{i=1}^n\lv\epsilon_i\rv^2=
    \frac{1}{n}\sum_{i=1}^n\lv\bar\x_i-\x_i\pr\rv^2=
    \frac{1}{n}\sum_{i=1}^n(\bar\x_i-\x_i\pr)^T(\bar\x_i-\x_i\pr)

    &=\frac{1}{n}\sum_{i=1}^n(\lv\bar\x_i\rv^2-2\bar\x_i^T\x_i\pr+(\x_i\pr)^T\x_i\pr)

    &=\frac{1}{n}\sum_{i=1}^n(\lv\bar\x_i\rv^2-2\bar\x_i^T(\u^T\bar\x_i)\u+
    ((\u^T\bar\x_i)\u)^T(\u^T\bar\x_i)\u),\rm{since\ }\x_i\pr=(\u^T\bar\x_i)\u

    &=\frac{1}{n}\sum_{i=1}^n(\lv\bar\x_i\rv^2-2(\u^T\bar\x_i)(\bar\x_i^T\u)+(\u^T\bar\x_i)(\bar\x_i^T\u)\u^T\u)

    &=\frac{1}{n}\sum_{i=1}^n(\lv\bar\rv^2-(\u^T\bar\x_i)\bar\x_i^T\u))

    &=\frac{1}{n}\sum_{i=1}^n\lv\bar\x_i\rv^2-\frac{1}{n}\sum_{i=1}^n\u^T(\bar\x_i\bar\x_i^T)\u

    &=\frac{1}{n}\sum_{i=1}^n\lv\bar\x_i\rv^2-\u^T\bigg(\frac{1}{n}\sum_{i=1}^n\bar\x_i\bar\x_i^T\bigg)\u

which implies

.. note::

    :math:`\dp MSE=\sum_{i=1}^n\frac{\lv\bar\x_i\rv^2}{n}-\u^T\Sg\u`

Further, we have

.. note::

    :math:`\dp\rm{var}(\D)=tr(\Sg)=\sum_{i=1}^d\sg_i^2`

.. note::

    :math:`\dp MSE(\u)=\rm{var}(\D)-\u^T\Sg\u=\sum_{i=1}^d\sg_i^2-\u^T\Sg\u`

The principal component :math:`\u_1`, which is the direction that maximizes the 
projected variance, is also the direction that minimizes the mean squared error.

.. math::

    MSE(\u_1)=\rm{var}(\D)-\u_1^T\Sg\u_1=\rm{var}(\D)=\u_1^T\ld_1\u_1=\rm{var}(\D)-\ld_1

7.2.2 Best 2-dimensional Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are now interested in the best two-dimensional approximation to :math:`\D`.
We now want to find another direction :math:`\v`, which also maximizes the 
projected variance, but is orthogonal to :math:`\u_1`.
The projected variance along :math:`\v` is given as

.. math::

    \sg_\v^2=\v^T\Sg\v

We further require that :math:`\v` be a unit vector orthogonal to :math:`\u_1`.
The optimization condition then becomes

.. math::

    \max_\v J(\v)=\v^T\Sg\v-\alpha(\v^T\v-1)-\beta(\v^T\u_1-0)

Taking the derivative of :math:`J(\v)` with respect to :math:`\v`, and setting 
it to the zero vector, finally gives that :math:`\v` is the second largest
eigenvector of :math:`\Sg`.

**Total Projected Variance**

Let :math:`\U_2` be the matrix whose columns correspond to the two principal components.
Given the point :math:`\bar\x_i\in\bar\D` its coordinates in the two-dimensional 
subspace spanned by :math:`\u_1` and :math:`\u_2` can be computed as follows:

.. math::

    \a_i=\U_2^T\bar\x_i

Assume that each point :math:`\bar\x_i\in\R^d` in :math:`\bar\D` has been 
projected to obtain its coordinates :math:`\a_i\in\R^2`, yielding the new 
dataset :math:`\A`.
The total variance for :math:`\A` is given as

.. math::

    \rm{var}(\A)&=\frac{1}{n}\sum_{i=1}^n\lv\a_i-\0\rv^2=
    \frac{1}{n}\sum_{i=1}^n(\U_2^T\bar\x_i)^T(\U_2^T\bar\x_i)=
    \frac{1}{n}\sum_{i=1}^n\bar\x_i^T(\U_2\U_2^T)\bar\x_i

    &=\frac{1}{n}\sum_{i=1}^n\bar\x_i^T\P_2\bar\x_i

where :math:`\P_2` is the orthogonal projection matrix given as

.. math::

    \P_2=\U_2\U_2^T=\u_1\u_1^T+\u_2\u_2^T

The projected total variance is then given as

.. math::

    \rm{var}(\A)&=\frac{1}{n}\sum_{i=1}^n\bar\x_i^T\P_2\bar\x_i

    &=\u_1^T\Sg\u_1+\u_2^T\Sg\u_2=\u_1^T\ld_1\u_1+\u_2^T\ld_2\u_2=\ld_1+\ld_2

**Mean Squared Error**

.. math::

    MSE&=\frac{1}{n}\sum_{i=1}^n\lv\bar\x_i-\x_i\pr\rv^2

    &=\frac{1}{n}\sum_{i=1}^n(\lv\bar\x_i\rv^2-2\bar\x_i^T\x_i\pr+(\x_i\pr)^T\x_i\pr)

    &=\rm{var}(\D)+\frac{1}{n}\sum_{i=1}^n(-2\bar\x_i^T\P_2\bar\x_i+(\P_2\bar\x_i)^T\P_2\bar\x_i)

    &=\rm{var}(\D)-\frac{1}{n}\sum_{i=1}^n(\bar\x_i^T\P_2\bar\x_i)

    &=\rm{var}(\D)-\rm{var}(\A)

    &=\rm{var}(\D)-\ld_1-\ld_2

7.2.3 Best :math:`r`-dimensional Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To find the best :math:`r`-dimensional approximation to :math:`\D`, we compute the eigenvalue of :math:`\Sg`.
Because :math:`\Sg` is positive semidefinite, its eigenvalues are non-negative and can be sorted in decreasing order

.. math::

    \ld_1\geq\ld_2\geq\cds\ld_r\geq\ld_{r+1}\cds\geq\ld_d\geq 0

We then select the :math:`r` largest eigenvalues, and their corresponding 
eigenvectors to form the best :math:`r`-dimensional approximation.

**Total Projected Variance**

.. math::

    \rm{var}(\A)=\frac{1}{n}\sum_{i=1}^n\bar\x_i^T\P_r\bar\x_i=\sum_{i=1}^r\u_i^T\Sg\u_i=\sum_{i=1}^r\ld_i

**Mean Squared Error**

.. math::

    MSE&=\frac{1}{n}\sum_{i=1}^n\lv\bar\x_i-\x_i\pr\rv^2=\rm{var}(\D)-\rm{var}(\A)

    &=\rm{var}(\D)-\sum_{i=1}^r\u_i^T\Sg\u_i=\rm{var}(\D)-\sum_{i=1}^r\ld_i

**Total Variance**

.. note::

    :math:`\dp\rm{var}(\D)=\sum_{i=1}^d\sg_i^2=\sum_{i=1}^d\ld_i`

**Choosing the Dimensionality**

One criteria for choosing :math:`r` is to compute the fraction of the total 
variance captured by the first :math:`r` principal components, computed as

.. note::

    :math:`\dp f(r)=\frac{\ld_1+\ld_2+\cds+\ld_r}{\ld_1+\ld_2+\cds+\ld_d}=`
    :math:`\dp\frac{\sum_{i=1}^r\ld_i}{\sum_{i=1}^d\ld_i}=\frac{\sum_{i=1}^r\ld_i}{\rm{var}(\D)}`

Given a certain desired variance threshold, say :math:`\alpha`, starting from 
the first principal component, we keep on adding additional components, and stop
at the smallest value :math:`r` for which :math:`f(r)\geq\alpha`, given as

.. note::

    :math:`r=\min\{r\pr|f(r\pr)\geq\alpha\}`

7.2.4 Geometry of PCA
^^^^^^^^^^^^^^^^^^^^^

Geometrically, when :math:`r=d`, PCA corresponds to a orthogonal change of basis,
so that the total variance is captured by the sum of the variances along each of
the principal direction :math:`\u_1,\u_2,\cds,\u_d`, and further, all 
covariances are zero.
This can be seen by looking at the collective action of the full set of 
principal components, which can be arranged in the :math:`d\times d` orthogonal
matrix with :math:`\U\im=\U^T`.

Each principal component :math:`\u_i` corresponds to an eigenvector of the 
covariance matrix :math:`\Sg`, which can be written compactly as

.. math::

    \Sg\U=\U\Ld

Multiply above equation on the left by :math:`\U\im=\U^T` we obtain

.. math::

    \U^T\Sg\U=\U^T\U\Ld=\Ld

This means that if we change the basis to :math:`\U`, we change the covariance 
matrix :math:`\Sg` to a similar matrix :math:`\Ld`, which in fact is the 
covariance matrix in the new basis.

It is worth noting that in the new basis, the equation

.. math::

    \x^T\Sg\im\x=1

defines a :math:`d`-dimensional ellipsoid (or hyper-ellipse).
The eigenvectors :math:`\u_i` of :math:`\Sg`, that is, the principal components,
are the directions for the principal axes of the ellipsoid.
The square roots of the eigenvalues, that is, :math:`\sqrt{\ld_i}`, give the lengths of the semi-axes.

The *eigen-decomposition* of :math:`\Sg` is

.. note::

    :math:`\dp\Sg=\U\Ld\U^T=\ld_1\u_1\u_1^T+\ld_2\u_2\u_2^T+\cds+\ld_d\u_d\u_d^T=\sum_{i=1}^d\ld_i\u_i\u_i^T`

Assuming that :math:`\Sg` is invertible or nonsingular, we have

.. math::

    \Sg\im=(\U\Ld\U^T)\im=(\U\im)^T\Ld\im\U\im=\U\Ld\im\U^T

Using the fact that :math:`\x=\U\a`, we get

.. math::

    \x^T\Sg\im\x&=1

    (\a^T\U^T)\U\Ld\im\U^T(\U\a)&=1

    \a^T\Ld\im\a&=1

    \sum_{i=1}^d\frac{a_i^2}{\ld_i}&=1

which is precisely the equation for an ellipse centered at :math:`\0`, with semi-axes lengths :math:`\sqrt{\ld_i}`.
Thus :math:`\x^T\Sg\im\x=1`, or equivalently :math:`\a^T\Ld\im\a=1` in the new
principal components basis, defines an ellipsoid in :math:`d`-dimensions, where
the semi-axes lengths equal the standard deviations along each axis.
Likewise, the equation :math:`\x^T\Sg\im\x=s`, or equivalently 
:math:`\a^T\Ld\im\a=s`, for different values of the scalar :math:`s`, represents
concentric ellipsoids.

7.3 Kernel Principal Component Analysis
---------------------------------------

Principal component analysis can be extended to find nonlinear "directions" in the data using kernel methods.
Kernel PCA finds the directions of most variance in the feature space instead of the input space.

In feature space, we can find the first kernel principal component :math:`\u_1`, 
by solving for the eigenvector corresponding to the largest eigenvalue of the
covariance matrix in feature space:

.. math::

    \Sg_\phi\u_1=\ld_1\u_1

where :math:`\Sg_\phi`, the covariance matrix in feature space, is given as

.. math::

    \Sg_\phi=\frac{1}{n}\sum_{i=1}^n(\phi(\x_i)-\mmu_\phi)(\phi(\x_i)-
    \mmu_\phi)^T=\frac{1}{n}\sum_{i=1}^n\bar\phi(\x_i)\bar\phi(\x_i)^T

Plugging the expansion of :math:`\Sg_\phi`, we get

.. math::

    \bigg(\frac{1}{n}\sum_{i=1}^n\bar\phi(\x_i)\bar\phi(\x_i)^T\bigg)\u_1&=\ld_1\u_1

    \frac{1}{n}\sum_{i=1}^n\bar\phi(\x_i)(\bar\phi(\x_i)^T\u_1)&=\ld_1\u_1

    \sum_{i=1}^n\bigg(\frac{\bar\phi(\x_i)^T\u_1}{n\ld_1}\bigg)\bar\phi(\x_i)&=\u_1

    \sum_{i=1}^nc_i\bar\phi(\x_i)&=\u_1

where :math:`c_i=\frac{\bar\phi(\x_i)^T\u_1}{n\ld_1}` is a scalar value.

.. math::

    \bigg(\frac{1}{n}\sum_{i=1}^n\bar\phi(\x_i)\bar\phi(\x_i)^T\bigg)\bigg(
    \sum_{j=1}^nc_j\bar\phi(\x_j)\bigg)&=\ld_1\sum_{i=1}^nc_i\bar\phi(\x_i)

    \frac{1}{n}\sum_{i=1}^n\sum_{j=1}^nc_i\bar\phi(\x_i)\bar\phi(\x_i)^T\bar\phi
    (\x_j)&=\ld_1\sum_{i=1}^nc_i\bar\phi(\x_i)

    \sum_{i=1}^n\bigg(\bar\phi(\x_i)\sum_{j=1}^nc_j\bar\phi(\x_i)^T\bar\phi
    (\x_j)\bigg)&=n\ld_1\sum_{i=1}^nc_i\bar\phi(\x_i)

    \sum_{i=1}^n\bigg(\bar\phi(\x_i)\sum_{j=1}^nc_j\bar{K}(\x_i,\x_j)\bigg)&=n\ld_1\sum_{i=1}^nc_i\bar\phi(\x_i)

We assume that the kernel matrix :math:`\K` has already been centered using

.. math::

    \bar\K=\bigg(\I-\frac{1}{n}\1_{n\times n}\bigg)\K\bigg(\I-\frac{1}{n}\1_{n\times n}\bigg)

Take any point, say :math:`\bar\phi(\x_k)` and multiply by :math:`\bar\phi(\x_k)^T` on both sides to obtain

.. math::

    \sum_{i=1}^n\bigg(\bar\phi(\x_k)^T\bar\phi(\x_i)\sum_{j=1}^nc_j\bar{K}
    (\x_i,\x_j)\bigg)&=n\ld_1\sum_{i=1}^nc_i\bar\phi(\x_k)^T\bar\phi(\x_i)

    \sum_{i=1}^n\bigg(\bar{K}(\x_k,\x_i)\sum_{j=1}^nc_j\bar{K}
    (\x_i,\x_j)\bigg)&=n\ld_1\sum_{i=1}^nc_i\bar{K}(\x_k,\x_i)

We can compactly represent it as follows:

.. math::

    \bar\K^2\c=n\ld_1\bar\K\c














7.4 Singular Value Decomposition
--------------------------------