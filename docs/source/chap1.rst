Chapter 1 Data Matrix
=====================

1.1 Data Matrix
---------------

The :math:`n\times d` data matrix is given as

.. math::

    \D=\bp &X_1&X_2&\cds&X_d\\ \x_1&x_{11}&x_{12}&\cds&x_{1d}\\
    \x_2&x_{21}&x_{22}&\cds&x_{2d}\\ \vds&\vds&\vds&\dds&\vds\\
    \x_n&x_{n1}&x_{n2}&\cds&x_{nd} \ep

where :math:`\x_i` donotes the :math:`i`\ th row, which is a :math:`d`-tuple given as

.. math::

    \x_i=(x_{i1},x_{i2},\cds,x_{id})

and :math:`X_j` denotes the :math:`j`\ th column, which is an :math:`n`-tuple given as

.. math::

    X_j=(x_{1j},x_{2j},\cds,x_{nj})

The number of instances :math:`n` is referred to as the *size* of the data, 
whereas the number of attributes :math:`d` is called the *dimensionality* of the
data.

1.2 Attributes
--------------

**Numeric Attributes**

A *numeric* attribute is one that has a real-valued or integer-valued domain.
Numeric attributes that take on a finite or countably infinite set of values are 
called *discrete*, whereas those that can take on any real value are called
*continuous*.
If an attribute has as its domain the set :math:`\{0, 1\}`, it is called a *binary* attribute.

* *Interval-scaled*: For these kinds of attributes only differences (addtion or subtraction) make sense.

* *Ratio-scaled*: Here one can compute both differences as well as ratios between values.

**Categorical Attributes**

A *categorical* attribute is one that has a set-valued domain composed of a set of symbols.

* *Nominal*: The attribute values in the domain are unordered, and thus only equality comparisons are meaningful.

* *Ordinal*: The attribute values are ordered, and thus both equality 
  comparisons and inequality comparisons are allowed, though it may not be 
  possible to quantify the difference between values.

1.3 Data: Algebraic and Geometric View
--------------------------------------

If the :math:`d` attributes or dimensions in the data matrix :math:`\D` are all 
numeric, then each row can be considered as a :math:`d`-dimensional point:

.. math::

    \x_i=(x_{i1},x_{i2},\cds,x_{id})\in\R^d

or equivalently, each row may be considered as a :math:`d`-dimensional column vector:

.. math::

    \x_i=\bp x_{i1}\\x_{i2}\\\vds\\x_{id} \ep=\bp x_{i1}&x_{i2}&\cds&x_{id} \ep^T\in\R^d

The :math:`j`\ th *standard basis vector* :math:`\e_j` of Cartesian coordinate
space is the :math:`d`-dimensional unit vector whose :math:`j`\ th component is
1 and the rest of thecomponents are 0:

.. math::

    \e_j=(0,\cds,1_j,\cds,0)^T

Any other vector in :math:`\R^d` can be written as a *linear combination* of the standard basis vectors:

.. math::

    \x_i=x_{i1}\e_1+\x_{i2}\e_2+\cds+x_{id}\e_d=\sum_{j=1}^dx_{ij}\e_j

where the scalar value :math:`x_{ij}` is the coordinate value along the :math:`j`\ th axis or attribute.

Each numeric column or attribute can also be treated as a vector in an :math:`n`-dimensional space :math:`\R^n`:

.. math::

    X_j=\bp x_{1j}\\x_{2j}\\\vds\\x_{nj} \ep

If all attributes are numeric, then the data matrix :math:`\D` is in the fact an 
:math:`n\times d` matrix, also written as :math:`\D\in\R^{n\times d}`, given as

.. math::

    \D=\bp x_{11}&x_{12}&\cds&x_{1d}\\x_{21}&x_{22}&\cds&x_{2d}\\
    \vds&\vds&\dds&\vds\\x_{n1}&x_{n2}&\cds&x_{nd} \ep=
    \bp -&\x_1^T&-\\-&\x_2^T&-\\&\vds\\-&\x_n^T&- \ep=
    \bp |&|&&|\\X_1&X_2&\cds&X_d\\|&|&&| \ep

1.3.1 Distance and Angle
^^^^^^^^^^^^^^^^^^^^^^^^
Let :math:`\a,\b\in\R^m` be two :math:`m`-dimensional vectors given as

.. math::

    \a=\bp a_1\\a_2\\\vds\\a_m \ep\quad\b=\bp b_1\\b_2\\\vds\\b_m \ep

**Dot Product**

.. note::

    :math:`\dp\a^T\b=\bp a_1&a_2&\cds&a_m\ep\times\bp b_1\\b_2\\\vds\\b_m\ep=a_1b_1+a_2b_2+\cds+a_mb_m=\sum_{i=1}^ma_ib_i`

**Length**

The *Euclidean norm* or *length* of a vector :math:`\a\in\R^m` is defined as

.. note::

    :math:`\dp\lv\a\rv=\sqrt{\a^T\a}=\sqrt{a_1^2+a_2^2+\cds+a_m^2}=\sqrt{\sum_{i=1}^ma_i^2}`

The *unit vector* in the direction of :math:`\a` is given as

.. math::

    \u=\frac{\a}{\lv\a\rv}=\bigg(\frac{1}{\lv\a\rv}\bigg)\a

By definition :math:`\u` has length :math:`\lv\u\rv=1`, and it is also called a *normalized* vector.

The Euclidean norm is a special case of a general class of norms, known as :math:`L_p`\ *-norm*, defined as

.. note::

    :math:`\dp\lv\a\rv_p=(|a_1|^p+|a_2|^p+\cds+|a_m|^p)^{\frac{1}{\p}}=\bigg(\sum_{i=1}^m|a_i|^p\bigg)^{\frac{1}{p}}`

for any :math:`p\neq 0`.

**Distance**

The *Eclidean distance* between :math:`\a` and :math:`\b`, as follows

.. note::

    :math:`\dp\lv\a-\b\rv=\sqrt{(\a-\b)^T(\a-\b)}=\sqrt{\sum_{i=1}^m(a_i-b_i)^2}`

The general :math:`L_p`-distance function is geven as follows

.. math::

    \lv\a-\b\rv_p=\bigg(\sum_{i=1}^m|a_i-b_i|^p\bigg)^{\frac{1}{p}}

**Angle**

The cosine of the smallest angle between vectors :math:`\a` and :math:`\b`, also 
called the *cosine similarity* is given as

.. note::

    :math:`\dp\cos\th=\frac{\a^T\b}{\lv\a\rv\lv\b\rv}=\bigg(\frac{\a}{\lv\a\rv}\bigg)^T\bigg(\frac{\b}{\lv\b\rv}\bigg)`

The *Cauchy-Schwartz* inequality states that for any vectors :math:`\a` and :math:`\b` in :math:`\R^ma_i

.. math::

    |\a^T\b|\leq\lv\a\rv\cd\lv\b\rv

It follows immediately from the Cauchy-Schwartz inequality that

.. math::

    -1\leq\cos\th\leq 1

**Orthogonality**

Two vectors :math:`\a` and :math:`\b` are said to be *orthogonal* if and only if 
:math:`\a^T\b=0`, which in turn implies that :math:`\cos\th=0`.
In this case, we say that they have no similarity.

1.3.2 Mean and Total Variance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Mean**

.. math::

    mean(\D)=\mmu=\frac{1}{n}\sum_{i=1}^n\x_i

**Total Variance**

.. math::

    \rm{var}(\D)=\frac{1}{n}\sum_{i=1}^n\lv\x_i-\mmu\rv^2

Simplifying the equation we obtain

.. math::

    \rm{var}(\D)&=\frac{1}{n}\sum_{i=1}^n(\lv\x_i\rv^2-2\x_i^T\mmu+\lv\mmu\rv^2)
    
    &=\frac{1}{n}\bigg(\sum_{i=1}^n\lv\x_i\rv^2-2n\mmu^T\bigg(\frac{1}{n}\sum_{i=1}^n\x_i\bigg)+n\lv\mmu\rv^2\bigg)

    &=\frac{1}{n}\bigg(\sum_{i=1}^n\lv\x_i\rv^2-2n\mmu^T\mmu+n\lv\mmu\rv^2\bigg)

    &=\frac{1}{n}\bigg(\sum_{i=1}^n\lv\x_i\rv^2\bigg)-\lv\mmu\rv^2

**Centered Data Matrix**

.. math::

    \ol\D=\D-\bs{1}\cd\mmu^T=\bp\x_1^T\\\x_2^T\\\vds\\\x_n^T\ep-
    \bp\mmu^T\\\mmu^T\\\vds\\\mmu^T\ep=\bp\x_1^T-
    \mmu^T\\\x_2^T-\mmu^T\\\vds\\\x_n^T-\mmu^T\ep=
    \bp\ol\x_1^T\\\ol\x_2^T\\\vds\\\ol\x_n^T\ep

The mean of the centered data matrix :math:`\ol\D` is :math:`\0\in\R^d`, because 
we have subtracted the mean :math:`\mmu` from all the points :math:`\x_i`.

1.3.3 Orthogonal Projection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\a,\b\in\R^m` be two :math:`m`-dimensional vectors.
An *orthogonal decomposition* of the vector :math:`\b` in the direction of another vector :math:`\a` is given as

.. math::

    \b=\b_\parallel+\b_\perp=\p+\r

where :math:`\p=\b_\parallel` is parallel to :math:`\a`, and :math:`\r=\b_\perp` 
is perpendicular or orthogonal to :math:`\a`.
The vector :math:`\p` is called the *orthogonal projection* or simply projection of :math:`\b` on the vector :math:`\a`.
The magnitude of the vector :math:`\r=\b-\p` gives the *perpendicular distance* 
between :math:`\b` and :math:`\a`, which is often interpreted as the residual or
error between the points :math:`\b` and :math:`\p`.
The vector :math:`\r` is also called the *error vector*.

We can derive an expression for :math:`\p` by noting that :math:`\p=c\a` for 
some scalar :math:`c`, as :math:`p` is parallel to :math:`\a`.
Thus :math:`\r=\b-\p=\b-c\a`.
Because :math:`\p` and :math:`\r` are orthogonal, we have

.. math::

    \p^T\r=(c\a)^T(\b-c\a)=c\a^T\b-c^2\a^T\a=0

which implies that

.. math::

    c=\frac{\a^T\b}{\a^T\a}

Therefore, the projection of :math:`\b` on :math:`\a` is given as

.. note::

    :math:`\dp\p=c\a=\bigg(\frac{\a^T\b}{\a^T\a}\bigg)\a`

The scalar offset :math:`c` along :math:`\a` is also called the 
*scalar projection* of :math:`\b` on :math:`\a`, denoted as

.. note::

    :math:`\dp\rm{proj}_\a(\b)=\bigg(\frac{\b^T\a}{\a^T\a}\bigg)`

Therefore, the projection of :math:`\b` on :math:`\a` can also be written as

.. math::

    \p=\rm{proj}_\a(\b)\cd\a

1.3.4 Linear Independence and Dimensionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the data matrix

.. math::

    \D=\bp \x_1&\x_2&\cds&\x_n \ep^T=\bp X_1&X_2&\cds&X_d \ep

we are often interested in the linear combinations of the rows (points) or the columns (attributes).

Given any set of vectors :math:`\v_1,\v_2,\cds,\v_k` in an :math:`m`-dimensional 
vector space :math:`\R^m`, their *linear combination* is given as 

.. math::

    c_1\v_1+c_2\v_2+\cds+c_k\v_k

where :math:`c_i\in\R` are scalar values.
The set of all possible linear combinations of the :math:`k` vectors is called 
the *span*, denoted as :math:`span(\v_1,\cds,\v_k)`, which is itself a vector
space being a *subspace* of :math:`\R^m`.
If :math:`span(\v_1,\cds,\v_k)=\R^m`, then we say that :math:`\v_1,\cds,\v_k`
is a *spanning set* for :math:`\R^m`.

**Row and Column space**

The *column space* of :math:`\D`, denoted :math:`col(\D)`, is the set of all 
linear combinations of the :math:`d` attributes :math:`X_j\in\R^n`

.. math::

    col(\D)=span(X_1,X_2,\cds,X_d)

By definition :math:`col(\D)` is a subsapce of :math:`\R^n`.
The *row space* of :math:`\D`, denoted :math:`row(\D)`, is the setof all linear
combinations of the :math:`n` points :math:`\x_i\in\R^d`

.. math::

    row(\D)=span(\x_1,\x_2,\cds,\x_n)

By definition :math:`row(\D)` is a subspace of :math:`\R^d`.

.. math::

    row(\D)=col(\D^T)

**Linear Independence**

The :math:`k` vectors are linearly dependent if there are scalars 
:math:`c_1,c_2,\cds,c_k`, at least one of which is not zero, such that

.. math::

    c_1\v_1+c_2\v_2+\cds+c_k\v_k=\0

On the other hand, :math:`\v_1,\cds,\v_k` are *linearly independent* if and only if

.. math::

    c_1\v_1+c_2\v_2+\cds+c_k\v_k=\0\rm{\ implies\ }c_1=c_2=\cds=c_k=0

**Dimension and Rank**

Let :math:`S` be a subspace of :math:`\R^m`.
A *basis* for :math:`S` is a set of vectors in :math:`S`, say 
:math:`\v_1,\cds,\v_k`, that are linearly independent and they span :math:`S`, 
that is, :math:`span(\v_1,\cds,\v_k)=S`.
A basis is a minimal spanning set.
If the vectors in the basis are pairwise orthogonal, they are said to form an *orthogonal basis* for :math:`S`.
If they are also normalized to be unit vectors, then they make up an *orthonormal basis* for :math:`S`.
The *standard basis* for :math:`\R^m` is an orthonormal basis consisting of the vectors

.. math::

    \e_1=\bp 1\\0\\\vds\\0 \ep\quad\e_2=\bp 0\\1\\\vds\\0 \ep\quad\cds\quad\e_m=\bp 0\\0\\\vds\\1 \ep

The number of vectors in a basis for :math:`S` is called the *dimension* of :math:`S`, denoted as :math:`dim(S)`.
Because :math:`S` is a subspace of :math:`\R^m`, we must have :math:`dim(S)\leq m`.

For any matrix, the dimension of its row and column space is the same, and this 
dimension is also called the *rank* of the matrix.
For the data matrix :math:`\D\in\R^{n\times d}`, we have 
:math:`rank(\D)\leq\min(n,d)`, which follows from the fact that the column space 
can have dimension at most :math:`d`, the row space can have dimension at most
:math:`n`.
With dimensionality reduction methods it is often possible to approximate 
:math:`\D\in\R^{n\times d}` with a derived data matrix 
:math:`\D\pr\in\R^{n\times k}`, which has much lower dimensionality, that is,
:math:`k\ll d`.

1.4 Data: Probabilistic View
----------------------------

A random variable :math:`X` is called a *discrete random variable* if it takes 
on only a finite or countably infinite number of values in its range, wehreas 
:math:`X` is called a *continuous random variable* if it can take on any value
in its range.

**Probability mass Function**

If :math:`X` is discrete, the *probability mass function* of :math:`X` is defined as

.. note::

    :math:`f(x)=P(X=x)` for all :math:`x\in\R`.

:math:`f` must obey the basi rules of probability, that is, :math:`f` must be non-negative:

.. math::

    f(x)\geq 0

and the sum of all probabilities should add to 1:

.. math::

    \sum_xf(x)=1

**Probability Density Function**

We define the *probability density function*, which specifies the probability 
that the variable :math:`X` takes on values in any interval 
:math:`[a,b]\subset\R`:

.. note::

    :math:`\dp P(X\in[a,b])=\int_a^b f(x)dx`

The density function :math:`f` must satisfy the basic laws of probability:

.. math::

    f(x) \geq 0,\quad\rm{for\ all\ }x\in\R

and

.. math::

    \int_{-\infty}^\infty f(x)dx=1

We can get an intuitive understanding of the density function :math:`f` by 
considering the probability density over a small interval of width 
:math:`2\epsilon >0`, centered at :math:`x`, namely 
:math:`[x-\epsilon,x+\epsilon]`:

.. math::

    P(X\in[x-\epsilon,x+\epsilon])=\int_{x-\epsilon}^{x+\epsilon}f(x)dx\simeq 2\epsilon\cd f(x)

    f(x)\simeq\frac{P(X\in[x-\epsilon,x+\epsilon])}{2\epsilon}

It is important to note that :math:`P(X=x)\neq f(x)`.

We can use the PDF to obtain the relative probability of one value :math:`x_1` 
over another :math:`x_2` because for a given :math:`\epsilon>0`, we have

.. math::

    \frac{P(X\in[x_1-\epsilon,x_1+\epsilon])}{P(X\in[x_2-\epsilon,x_2+\epsilon])}
    \simeq\frac{2\epsilon\cd f(x_1)}{2\epsilon\cd f(x_2)}=\frac{f(x_1)}{f(x_2)}

If :math:`f(x_1)` is larger than :math:`f(x_2)`, then values of :math:`X` close
to :math:`x_1` are more probable than values cloes to :math:`x_2` and vice versa.

**Cumulative Distribution Function**

The *cumulative distribution function (CDF)* :math:`F:\R\rightarrow[0,1]` which 
gives the probability of observing a value at most some given value :math:`x`:

.. note::

    :math:`F(x)=P(X\leq x)` for all :math:`-\infty<x<\infty`

When :math:`X` is discrete, :math:`F` is given as

.. math::

    F(x)=P(X\leq x)=\sum_{u\leq x}f(u)

and when :math:`X` is continuous, :math:`F` is given as

.. math::

    F(x)=P(X\leq x)=\int_{-\infty}^xf(u)du

1.4.1 Bivariate Random Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can also perform pair-wise analysis by considering a pair of attributes, 
:math:`X_1` and :math:`X_2`, as a *bivariate random variable*:

.. math::

    \X=\bp X_1\\X_2 \ep

**Joint Probability Mass Function**

If :math:`X_1` and :math:`X_2` are both discrete random variables then 
:math:`\X` has a *joint probability mass function* given as follows:

.. math::

    f(\x)=f(x_1,x_2)=P(X_1=x_1,X_2=x_2)=P(\X=\x)

:math:`f` must satisfy the following two conditions:

.. math::

    f(\x)=f(x_1,x_2)\geq 0\quad\rm{for\ all\ }-\infty<x_1,x_2<\infty

    \sum_\x f(\X)=\sum_{x_1}\sum_{x_2}f(x_1,x_2)=1

**Joint Probability Density Function**

If :math:`X_1` and :math:`X_2` are both continuous random variables then 
:math:`\X` has a *joint probability density function* :math:`f` given as follows:

.. math::

    P(\X\in W)=\iint_{\x\in W}f(\X)d\X=\iint_{(x_1,x_2)^T\in W}f(x_1,x_2)dx_1dx_2

where :math:`W\subset\R^2` is some subset of the 2-dimensional space of reals.
:math:`f` must also satisfy the following two conditions:

.. math::

    f(\x)=f(x_1,x_2)\geq 0\quad\rm{for\ all\ }-\infty<x_1,x_2<\infty

    \int_{\R^2}f(\x)d\x=\int_{-\infty}^\infty\int_{-\infty}^\infty f(x_1,x_2)dx_1dx_2=1

The probability density at :math:`\x` can be approximated using a 2-dimensional 
window of width :math:`2\epsilon` centered at :math:`\x=(x_1,x_2)^T` as

.. math::

    P(\X\in W)&=P(\X\in([x_1-\epsilon,x_1+\epsilon],[x_2-\epsilon,x_2+\epsilon]))

    &=\int_{x_1-\epsilon}^{x_1+\epsilon}\int_{x_2-\epsilon}^{x_2+\epsilon}
    f(x_1,x_2)dx_1dx_2\simeq 2\epsilon\cd 2\epsilon\cd f(x_1,x_2)

which implies that

.. math::

    f(x_1,x_2)=\frac{P(\X\in W)}{(2\epsilon)^2}

The relative probability of one value :math:`(a_1,a_2)` versus another 
:math:`(b_1,b_2)` can therefore be computed via the probability density function:

.. math::

    \frac{P(\X\in([a_1-\epsilon,a_1+\epsilon],[a_2-\epsilon,a_2+\epsilon]))}
    {P(\X\in([b_1-\epsilon,b_1+\epsilon],[b_2-\epsilon,b_2+\epsilon]))}\simeq
    \frac{(2\epsilon)^2\cd f(a_1,a_2)}{(2\epsilon)^2\cd f(b_1,b_2)}=
    \frac{f(a_1,a_2)}{f(b_1,b_2)}

**Joint Cumulative Distribution Function**

The *joint cumulative distribution function* for two random variables 
:math:`X_1` and :math:`X_2` is defined as the function :math:`F`, such that for
all values :math:`x_1,x_2\in(-\infty,\infty)`,

.. math::

    F(\x)=F(x_1,x_2)=P(X_1\leq x_1\rm{\ and\ }X_2\leq x_2)=P(\X\leq\x)

**Statistical Independence**

Two random variables :math:`X_1` and :math:`X_2` are said to be (statistically)
*independent* if, for every :math:`W_1\subset\R` and :math:`W_2\subset\R`, 

.. math::

    P(X_1\in W_1\rm{\ and\ }X_2\in W_2)=P(X_1\in W_1)\cd P(X_2\in W_2)

Furthermore, if :math:`X_1` and :math:`X_2` are independent, then the following two conditions are also satisfied:

.. math::

    F(\x)=F(x_1,x_2)=F_1(x_1)\cd F_2(x_2)

    f(\x)=f(x_1,x_2)=f_1(x_1)\cd f_2(x_2)

1.4.2 Multivariate Random Variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A :math:`d`-dimensional *multivariate random variable* 
:math:`\X=(X_1,X_2,\cds,X_d)^T`, also called a *vector random variable*, is
defined as a function that assigns a vector of real numbers to each outcome in
the sample space, that is :math:`\X:\cl{O}\ra\R^d`.

If all :math:`X_j` are discrete, then :math:`\X` is jointly discrete and its
joint probability mass function :math:`f` is given as

.. math::

    f(\x)&=P(\X=\x)

    f(x_1,x_2,\cds,x_d)&=P(X_1=x_1,X_2=x_2,\cds,X_d=x_d)

If all :math:`X_j` are continuous, then :math:`\X` is jointly continuous and its
joint probability density function is given as

.. math::

    P(\X\in W)&=\underset{\x\in W}{\int\cds\int}f(\x)d\x

    P((X_1,X_2,\cds,X_d)^T\in W)&=\underset{(x_1,x_2,\cds,x_d)^T\in W}{\int\cds\int}f(x_1,x_2,\cds,x_d)dx_1dx_2\cds dx_d

for any :math:`d`-dimensional region :math:`W\subseteq\R^d`.

We say that :math:`X_1,X_2,\cds,X_d` are independent random variables if any only if, for every region :math:`W_i\in\R`:

.. math::

    P(X_1\in W_1\rm{\ and\ }X_2\in W_2\cds\rm{\ and\ }X_d\in W_d)
    
    =P(X_1\in W_1)\cd P(X_2\in W_2)\cds P(X_d\in W_d)

If :math:`X_1,X_2,\cds,X_d` are independent then the following conditions are also satisfied

.. math::

    F(\x)=F(x_1,\cds,x_d)=F_1(x_1)\cd F_2(x_2)\cd \cds \cd F_d(x_d)

    f(\x)=f(x_1,\cds,x_d)=f_1(x_1)\cd f_2(x_2)\cd \cds \cd f_d(x_d)

1.4.3 Random Sample and Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In statistics, the word *population* is used to refer to the set or universe of all entieis under study.
We try to make inferences about the population parameters by drawing a random 
sample from the population, and by computing appropriate *statistics* from the
sample that give estimates of the corresponding population parameters of 
interest.

**Univariate Sample**

Given a random variable :math:`X`, a *random sample* of size :math:`n` from 
:math:`X` is defined as a set of :math:`n` *independent and identically* 
*distributed (IID)* random variables :math:`S_1,S_2,\cds,S_n`, that is, all of
the :math:`S_i`'s are statistically independent of each other, and follow the 
same probability mass or density function as :math:`X`.

Their joint probability function is given as


.. note::

    :math:`\dp f(x_1,\cds,x_n)=\prod_{i=1}^nf_X(x_i)`

where :math:`f_X` is the probability mass or density function for :math:`X`.

**Multivariate Sample**

:math:`\x_i` are assumed to be independent and identically distributed, and thus their joint distirbution is given as

.. note::

    :math:`\dp f(\x_1,\x_2,\cds,\x_n)=\prod_{i=1}^n f_{\X}(\x_i)`

where :math:`f_{\X}` is the probability mass or density function for :math:`\X`.

Under the attribute independence assumption the above equation can be rewritten as

.. math::

    f(\x_1,\x_2,\cds,\x_n)=\prod_{i=1}^n f(\x_i)=\prod_{i=1}^n\prod_{j=1}^df_{X_j}(x_{ij})

**Statistics**

Let :math:`\{ \bs{\rm{S}}_i\}_{i=1}^m` denote a random sample of size :math:`m` 
drawn from a (multivariate) random variable :math:`\X`.
A statistic :math:`\hat{\th}` is some function over the random sample, given as

.. math::

    \hat{\th}:(\bs{\rm{S}}_1,\bs{\rm{S}}_2,\cds,\bs{\rm{S}}_m)\ra\R

If we use the value of a statistic to estimate a population parameter, this 
value is called a *point estimate* of the parameter, and the statistic is called 
an *estimator* of the parameter.