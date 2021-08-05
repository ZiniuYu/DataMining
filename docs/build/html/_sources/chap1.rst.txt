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



1.4 Data: Probabilistic View
----------------------------


1.4.1 Bivariate Random Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



1.4.2 Multivariate Random Variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



1.4.3 Random Sample and Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^