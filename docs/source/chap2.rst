Chapter 2 Numeric Attributes
============================

2.1 Univariate Analysis
-----------------------

Univariate analysis focuses on a single attribute at a time; thus the data
matrix :math:`\D` can be thought of as an :math:`n\times 1` matrix, or simply a
column vector, given as

.. math::

    \D=\bp X\\x_1\\x_2\\\vds\\x_n \ep

where :math:`X` is the numeric attribute of interest, with :math:`x+i\in\R`.
:math:`X` is assumed to be a random variable, with each point 
:math:`x_i(1\leq i\leq b)` itself treated as an identity random variable.

**Empirical Cumulative Distribution Function**

The *empirical cumulative distribution function (CDF)* of :math:`X` is given as

.. math::

    \hat{F}(x)=\frac{1}{n}\sum_{i=1}^nI(x_i\leq x)

where

.. math::

    I(x_i\leq x)=\left\{\begin{array}{lr}1\quad\rm{if\ }x_i\leq x\\0\quad\rm{if\ }x_i>x\end{array}\right.

is a binary *indicator variable* that indicates whether the given condition is satisfied or not.
Note that we use the notation :math:`\hat{F}` to denote the fact that the 
empirical CDF is an estimate for the unknown population CDF :math:`F`.

**Inverse Cumulative Distribution Function**

Define the *inverse cumulative distribution function* or *quantile function* for a random variable :math:`X` as follows:

.. math::

    F\im(q)=\min\{x|F(x)\geq q\}\quad\rm{for\ }q\in[0,1]

**Empirical Probability Mass Function**

The *empirical probability mass function (PMF)* of :math:`X` is given as

.. math::

    \hat{f}(x)=P(X=x)=\frac{1}{n}\sum_{i=1}^nI(x_i=x)

where

.. math::

    I(x_i=x)=\left\{\begin{array}{lr}1\quad\rm{if\ }x_i=x\\0\quad\rm{if\ }x_i\neq x\end{array}\right.

2.1.1 Measures of Central Tendency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Mean**
The *mean*, also called the *expected value*, of a random variable :math:`X` is 
the arithmetic average of the values of :math:`X`.
It provides a one-number summary of the *location* or *central tendency* for the distribution of :math:`X`.

The mean or expected value of a discrete random variable :math:`X` is definede as

.. note::

    :math:`\dp\mu=E[X]=\sum_xx\cd f(x)`

where :math:`f(x)` is the probability mass function of :math:`X`.

The expected vvalue of a continuous random variable :math:`X` is defined as

.. note::

    :math:`\dp\mu=E[x]=\int_{-\infty}^\infty x\cd f(x)dx`

where :math:`f(x)` is the probability density function of :math:`X`.

**Sample Mean**

The *sample mean* is a statistic, that is, a function 
:math:`\hat\mu:\{x_1,x_2,\cds,x_n\}\ra\R`, defined as the average value of 
:math:`x_i`\ 's:

.. note::

    :math:`\dp\hat\mu=\frac{1}{n}\sum_{i=1}^nx_i`

It serves as an estimator for the unknown mean value :math:`\mu` of :math:`X`.

.. math::

    \hat\mu=\sum_xx\cd\hat{f}(x)=\sum_xx\bigg(\frac{1}{n}\sum_{i=1}^nI(x_i=x)\bigg)=\frac{1}{n}\sum_{i=1}^nx_i

**Sample Mean Is Unbiased**

An estimator :math:`\hat\th` is called an *unbiased estimator* for parameter 
:math:`\th` if :math:`E[\hat\th]=\th` forr every possible value of :math:`\th`.
The sample mean :math:`\hat\mu` is an unbiased estimator for the population mean :math:`\mu`, as

.. math::

    E[\hat\mu]=E\bigg[\frac{1}{n}\sum_{i=1}^nx_i\bigg]=\frac{1}{n}\sum_{i=1}^nE[x_i]=\frac{1}{n}\sum_{i=1}^n\mu=\mu

**Robustness**

We say that a statistic is *robust* if it is not affected by extreme values (such as outliers) in the data.
The sample mean is not robust because a single large value (an outlier) can skew the average.
A more robust measure is the *trimmed mean* obtained after discarding a small 
fraction of extreme values on one or both ends.

**Geometric Interpretation of Sample Mean**

Consider the projection of :math:`X` onto the vector :math:`\bs{1}`, we have

.. math::

    \p=\bigg(\frac{X^T\bs{1}}{\bs{1}^T\bs{1}}\bigg)\cd\bs{1}=
    \bigg(\frac{\sum_{i=1}^nx_i}{n}\bigg)\cd\bs{1}=\hat\mu\cd\bs{1}

Thus, the sample mean is simply the offset or the scalar projection of :math:`X` on the vector :math:`\bs{1}`:

.. note::

    :math:`\dp\hat\mu=\rm{proj}_{\bs{1}}(X)=\bigg(\frac{X^T\bs{1}}{\bs{1}^T\bs{1}}\bigg)`

The sample mean can be used to center the attribute :math:`X`.
Define the *centered attribute vector*, :math:`\bar{X}`, as follows:

.. math::

    \bar{X}=X-\hat\mu\cd\bs{1}=\bp x_1-\hat\mu\\x_2-\hat\mu\\\vds\\x_n-\hat\mu \ep

We can see that :math:`\bs{1}` and :math:`\bar{X}` are orthogonal to each other, since

.. math::

    \bs{1}^T\bar{X}=\bs{1}^T(X-\hat\mu\cd\bs{1})=\bs{1}^TX-
    \bigg(\frac{X^T\bs{1}}{\bs{1}^T\bs{1}}\bigg)\cd\bs{1}^T\bs{1}=0

If fact, the subspace containing :math:`\bar{X}` is an *orthogonal complement* of the space spanned by :math:`bs{1}`.

**Median**

The *median* of a random variable is defined as the value :math:`m` such that

.. math::

    P(X\leq m)\geq\frac{1}{2}\quad\rm{and}\quad P(X\geq m)\geq\frac{1}{2}

In terms of the (inverse) cumulative distribution function, the median is therefore the value :math:`m` for which

.. math::

    F(m)=0.5\quad\rm{or}\quad m=F\im(0.5)

The *sample median* can be obtained from the empirical CDF or the empirical inverse CDF by computing

.. math::

    \hat{F}(m)=0.5\quad\rm{or}\quad m=\hat{F}\im(0.5)

Median is robust, as it is not affected very much by extreme values.

**Mode**

The *mode* of a random variable :math:`X` is the value at which the probability 
mass function or the probability density function attains its maximum value,
depending on whether :math:`X` is discrete or continuous, respectively.

2.1.2 Measures of Dispersion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Range**

The *value range* or simply *range* of a random variable :math:`X` is the 
difference between the maximum and minimum values of :math:`X`, given as

.. math::

    r=\max\{X\}-\min\{X\}

The *sample range* is a statistic, given as

.. math::

    \hat{r}=\max_{i=1}^n\{x_i\}-\min_{i=1}^n\{x_i\}

By definition, range is sensitive to extreme values, and thus is not robust.

**Interquartile Range**

*Quartiles* are special values of the quantile function that divide the data into four equal parts.
A more robust measure of the dispersion of :math:`X` is the *interquartile range (IQR)*, defined as

.. math::

    IQR=q_3-q_1=F\im(0.75)-F\im(0.25)

The *sample* IQR can be obtained by plugging in the empirical inverse CDF:

.. math::

    \wh{IQR}=\hat{q}_3-\hat{q}_1=\hat{F}\im(0.75)-\hat{F}\im(0.25)

**Variance and Standard Deviation**

The *variance* of a random variable :math:`X` provides a measure of how much the 
values :math:`X` deviate from the mean or expected value of :math:`X`

.. note::

    :math:`\dp\sg^2=\rm{var}(X)=E[(X-\mu)^2]=`
    :math:`\dp\left\{\begin{array}{lr}\dp\sum_x(x-\mu)^2f(x)\quad\rm{if\ }X\rm{\ is\ discrete}\\\dp\int_{-\infty}^\infty(x-\mu)^2f(x)dx\quad\rm{if\ }X\rm{\ is\ continuous}\end{array}\right.`

The *standard deviation*, :math:`\sg`, is defined as the positive square root of the variance, :math:`\sg^2`.

.. math::

    \sg^2&=\rm{var}(X)=E[(X-\mu)^2]=E[X^2-2\mu X+\mu^2]

    &=E[X^2]-2\mu E[X]+\mu^2=E[X^2]-2\mu^2+\mu^2

    &=E[X^2]-(E[X])^2

It is worth noting that variance is in fact the *second moment about the mean*,
corresponding to :math:`r=2`, which is a special case of the :math:`r`\ 
*th moment about the mean* for a random variable :math:`X`, defined as
:math:`E[(X-\mu)^r]`.

**Sample Variance**

The *sample variance* is defined as

.. note::

    :math:`\dp\sg^2=\frac{1}{n}\sum_{i=1}^n(x_i-\mu)^2`

.. math::

    \hat\sg^2=\sum_x(x-\hat\mu)^2\hat{f}(x)=\sum_x(x-\hat\mu)^2\bigg(\frac{1}{n}
    \sum_{i=1}^nI(x_i=x)\bigg)=\frac{1}{n}\sum_{i=1}^n(x_i-\hat\mu)^2

The *sample standard deviation* is given as the positive square root of the sample variance:

.. math::

    \hat\sg=\sqrt{\frac{1}{n}\sum_{i=1}^n(x_u-\hat\mu)^2}

The *standard score*, also called the :math:`z`\ *-score*, of a sample value 
:math:`x_i` is the number of standard deviations the value is away from the mean:

.. note::

    :math:`\dp z_i=\frac{x_i-\hat\mu}{\hat\sg}`

**Variance of the Sample Mean**

The expected value of the sample mean is simply :math:`\mu`.

.. math::

    \rm{var}\bigg(\sum_{i=1}^nx_i\bigg)=\sum_{i=1}^n\rm{var}(x_i)=\sum_{i=1}^n\sg^2=n\sg^2

Further, note that

.. math::

    E\bigg[\sum_{i=1}^nx_i\bigg]=n\mu

The variance of the sample mean :math:`\hat\mu` can be computed as

.. math::

    \rm{var}(\hat\mu)&=E[(\hat\mu-\mu)^2]=E[\hat\mu^2]-\mu^2-
    E\bigg[\bigg(\frac{1}{n}\sum_{i=1}^nx_i\bigg)^2\bigg]-
    \frac{1}{n^2}E\bigg[\sum_{i=1}^nx_i\bigg]^2

    &=\frac{1}{n^2}\bigg(E\bigg[\bigg(\sum_{i=1}^nx_i\bigg)^2\bigg]-
    E\bigg[\sum_{i=1}^nx_i\bigg]^2\bigg)-\frac{1}{n^2}\rm{var}
    \bigg(\sum_{i=1}^nx_i\bigg)

    &=\frac{\sg^2}{n}

**Bias of Sample Variance**

The sample variance is a *biased estimator* for the true population variance, 
:math:`\sg^2`, that is, :math:`E[\hat\sg^2]\neq\sg^2`.

.. math::

    \sum_{i=1}^n(x_i-\mu)^2=n(\hat\mu-\mu)^2+\sum_{i=1}^n(x_i-\hat\mu)^2

.. math::

    E[\hat\sg]^2&=E\bigg[\frac{1}{n}\sum_{i=1}^n(x_i-\hat\mu)^2\bigg]=
    E\bigg[\frac{1}{n}\sum_{i=1}^n(x_i-\mu)^2\bigg]-E[(\hat\mu-\mu)^2]

    &=\frac{1}{n}n\sg^2-\frac{\sg^2}{n}=\bigg(\frac{n-1}{n}\bigg)\sg^2

The sample variance :math:`\hat\sg^2` is a biased estimator of :math:`\sg^2`,
as its expected value differs from the population variance by a factor of
:math:`\frac{n-1}{n}`.
However, it is *asymptotically unbiased*, that is, the bias vanishes as :math:`n\ra\infty` because

.. math::

    \lim_{n\ra\infty}\frac{n-1}{n}=\lim_{n\ra\infty}1-\frac{1}{n}=1

Put differently, as the sample size increases, we have

.. math::

    E[\hat\sg^2]\ra\sg^2\quad\rm{as\ }n\ra\infty

If we eant an unbiased estimate of the sample variance, denoted 
:math:`\hat\sg_u^2`, we must divide by :math:`n-1` instead of :math:`n`:

.. math::

    \hat\sg_u^2=\frac{1}{n-1}\sum_{i=1}^n(x_i-\hat\mu)^2

.. math::

    E[\hat\sg_u^2]&=E\bigg[\frac{1}{n-1}\sum_{i=1}^n(x_i-\hat\mu)^2\bigg]=
    \frac{1}{n-1}\cd E\bigg[\sum_{i=1}^n(x_i-\mu)^2\bigg]-\frac{n}{n-1}\cd
    E[(\hat\mu-\mu)^2]

    &=\frac{n}{n-1}\sg^2-\frac{n}{n-1}\cd\frac{\sg^2}{n}

    &=\frac{n}{n-1}\sg^2-\frac{1}{n-1}\sg^2=\sg^2

**Geometric Interpretation of Sample Variance**

Let :math:`\bar{X}` denote the centered attribute vector

.. math::

    \bar{X}=X-\hat\mu\cd\bs{1}=\bp x_1-\hat\mu\\x_2-\hat\mu\\\vds\\x_n-\hat\mu \ep

.. note::

    :math:`\dp\hat\sg^2=\frac{1}{n}\lv\bar{X}\rv^2=\frac{1}{n}\bar{X}^T\bar{X}=\frac{1}{n}\sum_{i=1}^n(x_i-\bar\mu)^2`

Define the *degress of freedom* (dof) of a statistical vector as the 
dimensionality of the subspace that contains the vector.
Notice that the centered attribute vector :math:`\bar{X}=X-\hat\mu\cd\bs{1}`
lies in a :math:`n-1` dimensional subspace that is an orthogonal complement of 
the 1 dimensional subspace spanned by the ones vector :math:`\bs{1}`.
Thus, the vector :math:`\bar{X}` has only :math:`n-1` degrees of freedom, and
the unbiased sample variance is simply the mean or expected squared length of
:math:`\bar{X}` per dimension

.. math::

    \sg_u^2=\frac{\lv X\rv^2}{n-1}=\frac{\bar{X}^T\bar{X}}{n-1}=\frac{1}{n-1}\cd\sum_{i=1}^n(x_i-\hat\mu)^2

2.2 Bivariate Analysis
----------------------

In bivariate analysis, we consider two attributes at the same time.

.. math::

    \D=\bp X_1&X_2\\x_{11}&x_{12}\\x_{21}&x_{22}\\\vds&\vds\\x_{n1}&x_{n2} \ep

It can be viewed as :math:`n` points or vectors in 2-dimensional space over the 
attributes :math:`X_1` and :math:`X_2`, that is, 
:math:`\x_i=(x_{i1},x_{i2})^T\in\R^2`.
Alternatively, it can be viewed as two points or vectors in an :math:`n`\
-dimensional space comprising the points, that is, each column is a vector in
:math:`\R`, as follows:

.. math::

    X_1=(x_{11},x_{21},\cds,x_{n1})^T

    X_2=(x_{12},x_{22},\cds,x_{n2})^T

In the probabilistic view, the column vector :math:`\X=(X_1,X_2)^T` is 
considered a bivariate vector random variable, and the points 
:math:`\x_i (1\leq i\leq n)` are treated as a random sample drawn from 
:math:`\X`, that is, :math:`\x_i`'s are considered independent and identically
distributed as :math:`\X`.

**Empirical Joint Probability Mass Function**

The *empirical joint probability mass function* for :math:`\X` is given as

.. math::

    \hat{f}(\x)=P(\X=\x)=\frac{1}{n}\sum_{i=1}^nI(\x_i=\x)

.. math::

    \hat{f}(x_1,x_2)=P(X_1=x_1,X_2=x_2)=\frac{1}{n}\sum_{i=1}^nI(x_{i1}=x_1,x_{i2}=x_2)

where

.. math::

    I(\x_i=\x)=\left\{\begin{array}{lr}1\quad\rm{if\ }x_{i1}=x_1\rm{\ and\ }
    x_{i2}=x_2\\0\quad\rm{otherwise}\end{array}\right.

2.2.1 Measures of Location and Dispersion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Mean**

The bivariate mean is defined as the expected value of the vector random variable :math:`\X`, defined as follows:

.. math::

    \mmu=E[\X]=E\bigg[\bp X_1\\X_2 \ep\bigg]=\bp E[X_1]\\E[X_2] \ep=\bp \mu_1\\\mu_2 \ep

The sample mean vector can be computed from the joint empirical PMF

.. note::

    :math:`\dp\hat\mmu=\sum_\x\x\hat{f}(\x)=\sum_\x\x\bigg(\frac{1}{n}\sum_{i=1}^nI(\x_i=\x)\bigg)=\frac{1}{n}\sum_{i=1}^n\x_i`

**Variance**

The *total variance* is given as

.. math::

    \sg_1^2+\sg_2^2

The *sample total variance* is simply

.. math::

    \rm{var}(\D)=\hat\sg_1^2+\hat\sg_2^2

2.2.2 Measures of Association
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Covariance**

The *covariance* between two attributes :math:`X_1` and :math:`X_2` provides a 
measure of the association or linear dependence between them, and is defined as

.. note::

    :math:`\sg_{12}=E[(X_1-\mu_1)(X_2-\mu_2)]`

By linearity of expectation, we have

.. math::

    \sg_{12}&=E[(X_1-\mu_1)(X_2-\mu_2)]=E[X_1X_2-X_1\mu_2-X_2\mu_1+\mu_1\mu_2]

    &=E[X_1X_2]-\mu_2E[X_1]-\mu_1E[X_2]+\mu_1\mu_2=E[X_1X_2]-\mu_1\mu_2

which implies

.. note::

    :math:`\sg_{12}=E[X_1X_2]-E[X_1]E[X_2]`

If :math:`X_1` and :math:`X_2` are independent random variables, then we conclude that their covariance is zero.
This is because if :math:`X_1` and :math:`X_2` are independent, then we have

.. math::

    E[X_1X_2]=E[X_1]\cd E[X_2]

which in turn implies that

.. math::

    \sg_{12}=0

The converse is not true.

The *sample covariance* between :math:`X_1` and :math:`X_2` is given as

.. note::

    :math:`\dp\hat\sg_{12}=\frac{1}{n}\sum_{i=1}^n(x_{i1}-\hat\mu_1)(x_{i2}-\hat\mu_2)`

.. math::

    \hat\sg_{12}&=E[(X_1-\hat\mu_1)(X_2-\hat\mu_2)]

    &=\sum_{\x=(x_1,x_2)^T}(x_1-\hat\mu_1)(x_2-\hat\mu_2)\hat{f}(x_1,x_2)

    &=\frac{1}{n}\sum_{\x=(x_1,x_2)^T}\sum_{i=1}^n(x_1-\hat\mu_1)\cd(x_2-\hat\mu_2)\cd I(x_{i1}=x_1,x_{i2}=x_2)

    &=\frac{1}{n}\sum_{i=1}^n(x_{i1}-\hat\mu_1)(x_{i2}-\hat\mu_2)

**Correlation**

The *correlation* between variables :math:`X_1` and :math:`X_2` is the 
*standardized covariance*, obatained by normalizing the covariance with the
standard deviation of each variable, given as

.. math::

    \rho_{12}=\frac{\sg_{12}}{\sg_1\sg_2}=\frac{\sg_{12}}{\sqrt{\sg_1^2\sg_2^2}}

The *sample correlation* for attributes :math:`X_1` and :math:`X_2` is given as

.. note::

    :math:`\dp\hat\rho_{12}=\frac{\hat\sg_{12}}{\hat\sg_1\sg_2}=`
    :math:`\dp\frac{\sum_{i=1}^n(x_{i1}-\hat\mu_1)(x_{i2}-\hat\mu_2)}{\sqrt{\sum_{i=1}^n(x_{i1}-\hat\mu_1)^2}\sqrt{\sum_{i=1}^n(x_{i2}-\hat\mu_2)^2}}`

**Geometric Interpretation of Sample Covariance and Correlation**

Let :math:`\bar{X}_1` and :math:`\bar{X}_2` denote the centered attribute vectors in :math:`\R^n`, given as follows:

.. math::

    \bar{X}_1=X_1-\hat\mu_1\cd\bs{1}=\bp x_{11}-\hat\mu_1\\x_{21}-\hat\mu_1\\
    \vds\\x_{n1}-\hat\mu_1 \ep\quad\bar{X}_2=X_2-\hat\mu_2\cd\bs{1}=
    \bp x_{12}-\hat\mu_2\\x_{22}-\hat\mu_2\\\vds\\x_{n2}-\hat\mu_2 \ep

The sample covariance can then be written as

.. note::

    :math:`\dp\hat\sg_{12}=\frac{\hat{X}_1^T\hat{X}_2}{n}`

The sample correlation can be written as

.. note::

    :math:`\dp\hat\rho_{12}=\frac{\bar{X}_1^T\bar{X}_2}{\sqrt{\bar{X}_1^T\bar{X}_1}\sqrt{\bar{X}_2^T\bar{X}_2}}=`
    :math:`\dp\frac{\bar{X}_1^T\bar{X}_2}{\lv\bar{X}_1\rv\lv\bar{X}_2\rv}=`
    :math:`\dp\left(\frac{\bar{X}_1}{\lv\bar{X}_1\rv}\right)^T\left(\frac{\bar{X}_2}{\lv\bar{X}_2\rv}\right)=\cos\th`

**Covariance Matrix**

The variance-covariance information for the two attributes :math:`X_1` and 
:math:`X_2` can be summarized in the square :math:`2\times 2` 
*covariance matrix*, given as

.. math::

    \Sg=E[(\X-\mmu)(\X-\mmu)^T]

.. math::

    =E\bigg[\bp X_1-\mu_1\\X_2-\mu_2 \ep\bp X_1-\mu&X_2-\mu_2 \ep\bigg]

.. math::

    =\bp E[(X_1-\mu_1)(X_1-\mu_1)]&E[(X_1-\mu_1)(X_2-\mu_2)]\\E[(X_2-\mu_2)(X_1-\mu_1)]&E[(X_2-\mu_2)(X_2-\mu_2)] \ep

.. math::
    
    =\bp \sg_1^2&\sg_{12}\\\sg_{21}&\sg_2^2 \ep

Because :math:`\sg_{12}=\sg_{21}`, :math:`\Sg` is a *symmetric* matrix.

The *total variance* of the two attributes is given as the sum of the diagonal 
elements of :math:`\Sg`, which is also called the *trace* of :math:`\Sg`, given
as

.. math::

    tr(\Sg)=\sg_1^2+\sg_2^2

We immediately have :math:`tr(\Sg)\leq 0`.

2.3 Multivariate Analysis
-------------------------

In multivariate analysis, we consider all the :math:`d` numeric attributes :math:`X_1,X_2,\cds,X_d`.
The full data is an :math:`n\times d` matrix, given as

.. math::

    \D=\bp X_1&X_2&\cds&X_d\\x_{11}&x_{12}&\cds&x_{1d}\\
    x_{21}&x_{22}&\cds&x_{2d}\\\vds&\vds&\dds&\vds\\x_{n1}&x_{n2}&\cds&x_{nd}\ep
    =\bp|&|&&|\\X_1&X_2&\cds&X_d\\|&|&&|\ep=\bp -&\x_1^T&-\\-&\x_2^T&-\\&\vds\\
    -&\x_n^T&\ep

In the row view, the data can be considered as a set of :math:`n` points or
vectors in the :math:`d`-dimensional attribute space

.. math::

    \x_i=(x_{i1},x_{i2},\cds,x_{id})^T\in\R^d

In the column view, the data can be considered as a set of :math:`d` points or
vectors in the :math:`n`-dimensional space spanned by the data points

.. math::

    X_j=(x_{1j},x_{2j},\cds,x_{nj})^T\in\R^n

In the probabilistic view, the :math:`d` attributes are modeled as a vector 
random variable, :math:`\X=(X_1,X_2,\cds,X_d)^T`, and the points :math:`\x_i`
are considered to be a random sample drawn from :math:`\X`, that is, they are
independent and identically distributed as :math:`\X`.

**Mean**

The *multivariate mean vector* is obtained by taking the mean of each attribute, given as

.. math::

    \mmu=E[\X]=\bp E[X_1]\\E[X_2]\\\vds\\E[X_d] \ep=\bp \mu_1\\\mu_2\\\vds\mu_d \ep

The *sample mean* is given as

.. note::

    :math:`\dp\hat\mmu=\frac{1}{n}\sum_{i=1}^n\x_i`

.. math::

    \hat\mmu=\frac{1}{n}\D^T\bs{1}

**Covariance Matrix**

The multivariate covariance information is captured by the :math:`d\times d` symmetric *covariance matrix*

.. math::

    \Sg=E[\X-\mmu)(\X-\mmu)^T]=\bp \sg_1^2&\sg_{12}&\cds&\sg_{1d}\\
    \sg_{21}&\sg_{2}^2&\cds&\sg_{2d}\\\cds&\cds&\cds&\cds\\
    \sg_{d1}&\sg_{d2}&\cds&\sg_d^2 \ep

**Covariance Matrix Is Positive Semidefinite**

:math:`\Sg` is a *positive semidefinite* matrix, that is,

.. math::

    \a^T\Sg\a\geq 0\rm{\ for\ any\ }d\rm{-dimensional\ vector\ }\a

Too see this, observe that

.. math::

    \a^t\Sg\a&=\a^TE[(\X-\mmu)(\X-\mmu)^T]\a

    &=E[\a^T(\X-\mmu)(\X-\mmu)^T\a]

    &=E[Y^2]

    &\geq 0

where :math:`Y` is the random variable :math:`Y=\a^t(\X-\mmu)=\sum_{i=1}^da_i(X_i-\mu_i)`.

The :math:`d` eigenvalues of :math:`\Sg` can be arranged from the largest to the 
smallest as follows: :math:`\ld_1\geq\ld_2\geq\cds\geq\ld_d\geq 0`.

**Total and Generalized Variance**

The total variacne is given as the trace of the covariance matrix:

.. note::

    :math:`tr(\Sg)=\sg_1^2+\sg_2^2+\cds+\sg_d^2`

The generalized variacne is defined as the determinant of the covariance matrix,
:math:`\det(\Sg)`, also denoted as :math:`|\Sg|`; it gives a single value for
the overall multivariate scatter:

.. note::

    :math:`\dp\det(\Sg)=|\Sg|=\prod_{i=1}^d\ld_i`

Since all the eigenvalues of :math:`\Sg` are non-negative (:math:`\ld_i\geq 0`), it follows that :math:`\det(\Sg)\geq 0`.

**Sample Covariance Matrix**

The *sample covariance matrix* is given as

.. note::

    :math:`\dp\hat\Sg=E[(\X-\hat\mmu)(\X-\hat\mmu)^T]=`
    :math:`\dp\bp\hat\sg_1^2&\hat\sg_{12}&\cds&\hat\sg_{1d}\\\hat\sg_{21}&\hat\sg_{2}^2&\cds&\hat\sg_{2d}\\\cds&\cds&\cds&\cds\\\hat\sg_{d1}&\hat\sg_{d2}&\cds&\hat\sg_d^2\ep`

Let :math:`\bar{D}` represent the centered data matrix, given as the matrix of 
centered attribute vectors :math:`\bar{X}_i-X_i-\hat\mu_i\cd\bs{1}`, where
:math:`\bs{1}\in\R^n`:

.. math::

    \bar{\D}=\D-\bs{1}\cd\hat\mmu^T=\bp |&|&&|\\\bar{X}_1&\bar{X}_2&\cds&\bar{X}_d\\|&|&&|\ep

    =\bp \x_1^T-\hat\mmu^T\\\x_2^T-\hat\mmu^T\\\vds\\\x_n^T-\hat\mmu^T \ep=
    \bp -&\bar\x_1^T&-\\-&\bar\x_2^T&-\\&\vds\\-&\bar\x_n^T&- \ep

In matrix notation, the sample covariance matrix can be written as

.. note::

    :math:`\dp\hat\Sg=\frac{1}{n}(\bar\D^T\bar\D)=\frac{1}{n}`
    :math:`\dp\bp\bar{X}_1^T\bar{X}_1&\bar{X}_1^T\bar{X}_2&\cds&\bar{X}_1^T\bar{X}_d\\\bar{X}_2^T\bar{X}_1&\bar{X}_2^T\bar{X}_2&\cds&\bar{X}_2^T\bar{X}_d\\\vds&\vds&\dds&\vds\\\bar{X}_d^T\bar{X}_1&\bar{X}_d^T\bar{X}_2&\cds&\bar{X}_d^T\bar{X}_d\ep`

The sample covariance matrix can also be written as a sum of rank-one matrices 
obtained as the *outer product* of each centered point:

.. note::

    :math:`\dp\hat\Sg=\frac{1}{n}\sum_{i=1}^n\bar\x_i\cd\bar\x_i^T`

Also the sample total variance is given as

.. math::

    \rm{var}(\D)=tr(\hat\Sg)=\hat\sg_1^2=\hat\sg_2^2+\cds+\hat\sg_d^2
    
**Sample Scatter Matrix**

The *sample scatter matrix* is the :math:`d\times d` positive semi-denifite matrix defined as

.. math::

    \bs{\rm{S}}=\bar\D^T\bar\D=\sum_{i=1}^n\bar\x_i\cd\bar\x_i^T

It is simply the un-normalized sample covariance matrix, since :math:`\bs{\rm{S}}=n\cd\hat\Sg`.

2.4 Data Normalization
----------------------

**Range Normalization** 

Let :math:`X` be an attribute and let :math:`x_1,x_2,\cds,x_n` be a random sample drawn from :math:`X`.
In *range normalization* each value is caled by the sample range :math:`\hat{r}` of :math:`X`:

.. math::

    x_i\pr=\frac{x_i-\min_i\{x_i\}}{\hat{r}}=\frac{x_i-\min_i\{x_i\}}{\max_i\{x_i\}-\min_i\{x_i\}}

After transformation the new attribute takes on values in the range [0, 1].

**Standard Score Normalization**

In *standard score normalization*, also called :math:`z`\ -normalization, each 
value is replaced by its :math:`z`\ -score:

.. math::

    x_i\pr=\frac{x_i-\hat\mu}{\hat\sg}

2.5 Normal Distribution
-----------------------

2.5.1 Univariate Normal Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    :math:`\dp f(x|\mu,\sg^2)=\frac{1}{\sqrt{2\pi \sg^2}}\exp\bigg\{-\frac{(x-\mu)^2}{2\sg^2}\bigg\}`

**Probability Mass**

Given an interval :math:`[a,b]` the probability mass of the normal distribution within that interval is given as

.. math::

    P(a\leq x\leq b)=\int_a^bf(x|\mu,\sg^2)dx

The probability mass concentrated with :math:`k` standard deviations from the
mean, that is, for the interval :math:`[\mu-k\sg,\mu+k\sg]`, can be computed as

.. math::

    P(\mu-k\sg\leq x\leq\mu+k\sg)=\frac{1}{\sqrt{2\pi}\sg}
    \int_{\mu-k\sg}^{\mu+k\sg}\exp\bigg\{-\frac{(x-\mu)^2}{2\sg^2}\bigg\}

Via a change of variable :math:`z=\frac{x-\mu}{\sg}`, we get

.. math::

    P(-k\leq z\leq k)=\frac{1}{\sqrt{2\pi}}=\int_{-k}^ke^{-\frac{1}{2}z^2}dz=
    \frac{2}{\sqrt{2\pi}}\int_0^ke^{-\frac{1}{2}z^2}dz

Via another change of variable :math:`t=\frac{z}{\sqrt{2}}`, we get

.. math::

    P(-k\leq z\leq k)=2\cd P(0\leq t\leq k/\sqrt{2})=\frac{2}{\sqrt{\pi}}
    \int_0^{k/\sqrt{2}}e^{-t^2}dt=\rm{erf}(k/\sqrt{2})

where erf is the *Gauss error function*, defined as

.. math::

    \rm{erf}(x)=\frac{2}{\sqrt{\pi}}\int_0^xe^{-t^2}dt

2.5.2 Multivariate Normal Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    :math:`\dp f(\x|\mmu,\Sg)=\frac{1}{(\sqrt{2\pi})^d\sqrt{|\Sg|}}`
    :math:`\dp\exp\bigg\{-\frac{(\x-\mmu)^T\Sg\im(\x-\mmu)}{2}\bigg\}`

As in the univariate case, the term

.. math::
    
    (\x-\mmu)^T\Sg\im(\x-\mmu)

measures the distance, called the *Mahalanobis distance*, of the point 
:math:`\x` from the mean :math:`\mmu` of the distribution, taking into account 
all of the variance-covariance information between the attributes.

The *standard multivariate normal distribution* has parameters :math:`\mu=\0` and :math:`\Sg=\bs{\rm{I}}`.

**Geometry of the Multivariate Normal**

Compared to the standard normal distribution, we can expect the density contours to be shifted, scaled, and rotated.
The shape or geometry of the normal distribution becomes clear by considering 
the eigen-decomposition of the covariance matrix.
The eigenvector equation for :math:`\Sg` is given as

.. math::

    \Sg\u_i=\ld_i\u_i

The diagonal matrix :math:`\Ld` is used to record the eigenvalues:

.. math::

    \Ld=\bp \ld_1&0&\cds&0\\0&\ld_2&\cds&0\\\vds&\vds&\dds&\vds\\0&0&\cds&\ld_d \ep

The eigenvectors are orthonormal, and can be put together into an orthogonal matrix :math:`\bs{\rm{U}}`:

.. math::

    \bs{\rm{U}}=\bp |&|&&|\\\u_1&\u_2&\cds&\u_d\\|&|&&| \ep

The eigen-decomposition of :math:`\Sg` can then be expressed compactly as follows:

.. math::

    \Sg=\bs{\rm{U}}\Ld\bs{\rm{U}}^T

This equation can be interpreted geometrically as a change in basis vectors.

**Total and Generalized Variance**

.. math::

    \rm{var}(\D)=tr(\D)=\sum_{i=1}^d\sg_i^2=\sum_{i=1}^d\ld_i=tr(\Ld)

In other words :math:`\sg_1^2+\cds+\sg_d^2=\ld_1+\cds+\ld_d`.