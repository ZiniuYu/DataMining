Chapter 15 Density-based Clustering
===================================

15.1 The DBSCAN Algorithm
-------------------------

Density-based clustering uses the local density of points to determine the 
clusters, rather than using only the distance between points.
We define a ball of radius :math:`\epsilon` around a point :math:`\x\in\R^d`,
called the :math:`\epsilon`-*neighborhood* of :math:`\x`, as follows:

.. note::

    :math:`N_\epsilon(\x)=B_d(\x, \epsilon)=\{\y|\lv\x-\y\rv\leq\epsilon\}`

Here :math:`\lv\x-\y\rv` is the Euclidean distance between points :math:`\x` and :math:`\y`.
However, other distance metrics can also be used.

For any point :math:`\x\in\D`, we say that :math:`x` is a *core point* if there 
are at least *minpts* points in its :math:`\epsilon`-neighborhood.
A *border point* is defined as a point that does not meet the *minpts* 
threshold, but it belongs to the :math:`\epsilon`-neighborhood of some core
point :math:`\bs{\rm{z}}`.
If a point is neight a core nor a border point, then it is called a *noise point* or an outlier.

We say that a point :math:`\x` is *directly density reachable* from another 
point :math:`y` if :math:`\x\in N_\epsilon(\y)` and :math:`\y` is a core point.
We say that :math:`\x` is *density reachable* from :math:`y` if there exists a
chain of points, :math:`\x_0,\x_1,\cds,\x_l`, such that :math:`\x=\x_0` and 
:math:`y=\x_l` and :math:`\x_i` is directly density reachable from 
:math:`\x_{i-1}` for all :math:`i=1,\cds,l`.
Define any two points :math:`\x` and :math:`\y` to be *density connected* if
there exists a core point :math:`\bs{rm{z}`, such that both :math:`\x` and 
:math:`\y` are density reachable from :math:`\bs{\rm{z}}`.
A *density-based cluster* is defined as a maximal set of density connected points.

One limitation of DBSCAN is that it is sensitive to the choice of 
:math:`\epsilon`, in particular if clusters have different densities.
If :math:`\epsilon` is too small, sparser clusters will be categorized as noise.
If :math:`\epsilon` is too large, denser clusters may be merged together.

.. image:: ../_static/Algo15.1.png

**Computational Complexity**

The overall complexity of DBSCAN is :math:`O(n^2)` is the worst-case.

15.2 Kernel Density Estimation
------------------------------

15.2.1 Univariate Density Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can directly estimate the cumulative distribution function from the data by 
counting how many points are less than or equal to :math:`x`:

.. math::

    \hat{F}(x)=\frac{1}{n}\sum_{i=1}^nI(x_i\leq x)

We can estimate the density funciton by taking the derivative of 
:math:`\hat{F}(x)`, by considering a window of small width :math:`h` centered at 
:math:`x`, that is,

.. math::

    \hat{f}(x)=\frac{\hat{F}(x+\frac{h}{2})-\hat{F}(x-\frac{h}{2})}{h}=\frac{k/n}{h}=\frac{k}{nh}

where :math:`k` is the number of points that lie in the window of width 
:math:`h` centered at :math:`x`, that is, with the closed interval
:math:`[x-\frac{h}{2},x+\frac{h}{2}]`

**Kernel Estimator**

Kernel density estimation relies on a *density kernel function* :math:`K` that 
is non-negative, symmetric, and integrates to 1, that is, 
:math:`K(x)\geq 0, K(-x)=K(x)` for all values :math:`x`, and :math:`\int K(x)dx=1`.
Thus, :math:`K` is essentially a probability density function.