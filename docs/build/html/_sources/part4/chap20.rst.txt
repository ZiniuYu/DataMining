Chapter 20 Linear Discriminant Analysis
=======================================

Given labeled data consisting of :math:`d`-dimensional points :math:`\x_i` along 
with their classes :math:`y_i`, the goal of linear discriminant analysis (LDA) 
is to find a vector :math:`\w` that maximizes the separation between the classes
after projection onto :math:`\w`.

20.1 Optimal Linear Discriminant
--------------------------------

Let us assume that the dataset :math:`\D` consists of :math:`n` points 
:math:`\x_i\in\R^d`, with the corresponding class label 
:math:`y_i\in\{c_1,c_2,\cds,c_k\}`.
Let :math:`\D_i` denote the subset of points labeled with class :math:`c_i`,
i.e., :math:`\D_i=\{\x_j^T|y_j=c_i\}`, and let :math:`|\D_i|=n_i` denote the 
number of points with class :math:`c_i`.
We assume that there are only :math:`k=2` classes.
Thus, the dataset :math:`\D` can be partitioned into :math:`\D_1` and :math:`\D_2`.

Let :math:`\w` be a unit vector, that is, :math:`\w^T\w=1`.
The projection of any :math:`d`-dimensional point :math:`\x_i` onto the vector :math:`\w` is given as

.. math::

    \x_i\pr=\bigg(\frac{\w^T\x_i}{\w^T\w}\bigg)\w=(\w^T\x_i)\w=a_i\w

where :math:`a_i` is the offset or scalar projection of :math:`\x_i` on the line :math:`\w`:

.. math::

    a_i=\w^T\x_i

We also call :math:`a_i` a *projected point*.
Thus the set of :math:`n` projected points :math:`\{a_1,a_2,\cds,a_n\}` 
represents a mapping from :math:`\R^d` to :math:`\R`, that is, from the original
:math:`d`-dimensional space to a 1-dimensional space of offsets along 
:math:`\w`.

Each projected point :math:`a_i` has associated with it the original class label 
:math:`y_i`, and thus we can compute, for each of the two classes, the mean of
the projected points ,called the *projected mean*, as follows:

.. math::

    m_1&=\frac{1}{n_1}\sum_{\x_i\in\D_1}a_i

    &=\frac{1}{n_1}\sum_{\x_i\in\D_1}\w^T\x_i

    &=\w^T\bigg(\frac{1}{n_1}\sum_{\x_i\in\D_1}\x_i\bigg)

    &=\w^T\mmu_1

where :math:`\mmu_1` is the mean of all point in :math:`\D_1`.
Likewise, we can obtain

.. math::

    m_2=\w^T\mmu_2

To maximize the separation between the classes, it seems reasonable to maximize 
the difference between the projected means, :math:`|m_1-m_2|`.
For good separation, the variance of the projected points for each class should also not be too large.
LDA maximizes the separation by ensuring that the *scatter* :math:`s_i^2` for 
the projected points within each class is small

.. math::

    s_i^2=\sum_{\x_j\in\D_i}(a_j-m_i)^2=n_i\sg_i^2

We can incorporate the two LDA criteria, namely, maximizing the distance between 
projected means and minimizing the sum of projected scatter, into a single
maximization criterion called the *Fisher LDA objective*:

.. note::

    :math:`\dp\max_\w J(\w)\frac{(m_1-m_2)^2}{s_1^2+s_2^2}`

The vector :math:`\w` is also called the *optimal linear discriminant (LD)*.

We can rewrite :math:`(m_1-m_2)^2` as follows:

.. math::

    (m_1-m_2)^2&=(\w^T(\mmu_1-\mmu_2))^2
    
    &=\w^T((\mmu_1-\mmu_2)(\mmu_1-\mmu_2)^T)\w
    
    &=\w^T\B\w

where :math:`\B=(\mmu_1-\mmu_2)(\mmu_1-\mmu_2)^T` is a :math:`d\times d` 
rank-one matrix called the *between-class scatter matrix*.