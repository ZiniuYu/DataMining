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

**Discrete Kernel**

The density estimate :math:`\hat{f}(x)` can also be rewritten in terms of the kernel function as follows:

.. note::

    :math:`\dp\hat{f}(x)=\frac{1}{nh}\sum_{i=1}^nK\bigg(\frac{x-x_i}{h}\bigg)`

where the **discrete kernel** function :math:`K` computes the number of points 
in a window of width :math:`h`, and is defined as

.. note::

    :math:`K(z)=\left\{\begin{array}{lr}1\quad\rm{if\ }|z|\leq\frac{1}{2}\\0\quad\rm{Otherwise}\end{array}\right.`

**Gaussian Kernel**

Instead of the discrete kernel, we can define a more smooth transition of influence via a Gaussian kernel:

.. math::

    K(z)=\frac{1}{\sqrt{2\pi}}\exp\bigg\{-\frac{z^2}{2}\bigg\}

Thus, we have

.. note::

    :math:`\dp K\bigg(\frac{x-x_i}{h}\bigg)=\frac{1}{\sqrt{2\pi}}\exp\bigg\{-\frac{(x-x_i)^2}{2h^2}\bigg\}`

Here :math:`x`, which is at the center of the window, plays the role of the 
mean, and :math:`h` acts as the standard deviation.

15.2.2 Multivariate Density Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The volume of a :math:`d`-dimensional hypercube is given as

.. math::

    \rm{vol}(H_d(h))=h^d

The density is then estimated as the fraction of the point weight lying within 
the :math:`d`-dimensional window centered at :math:`\x`, divided by the volume
of the hypercube:

.. math::

    \hat{f}(\x)=\frac{1}{nh^d}\sum_{i=1}^nK\bigg(\frac{\x-\x_i}{h}\bigg)

where the multivariate kernel function :math:`K` satisfies the condition :math:`\int K(\z)d\z=1`.

**Discrete Kernel**

.. note::

    :math:`K(\z)=\left\{\begin{array}{lr}1\quad\rm{if\ }|z_j|\leq\frac{1}{2},\rm{for\ all\ dimensions\ }j=1,\cds,d\\0\quad\rm{Otherwise}\end{array}\right.`

**Gaussian Kernel**

The :math:`d`-dimensional Gaussian kernel is given as

.. math::

    K(\z)=\frac{1}{(2\pi)^{d/2}}\exp\bigg\{-\frac{\z^T\z}{2}\bigg\}

where we assume that the covariance matrix is the :math:`d\times d` identity matrix, that is, :math:`\Sg=\I_d`.

.. note::

    :math:`\dp K\bigg(\frac{\x-\x_i}{h}\bigg)=\frac{1}{(2\pi)^{d/2}}\exp\bigg\{-\frac{(\x-\x_i)^T(\x-\x_i)}{2h^2}\bigg\}`

Each point contributes a weight to the density estimate inversely proportional 
to its distance from :math:`\x` termpered by the width parameter :math:`h`.

15.2.3 Nearest Neighbor Density Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An alternative approach to density estimation is to fix :math:`k`, the number of 
points required to estimate the density, and allow the volume of the enclosing 
region to vary to accomodate those :math:`k` points.
This apporach is called the :math:`k` nearest neighbors (KNN) approach to density estimation.

Given :math:`k`, the number of neighbors, we estimate the density at :math:`\x` as follows:

.. math::

    \hat{f}(\x)=\frac{k}{n\rm{\ vol}(S_d(h_\x))}

where :math:`h_\x` is the distance from :math:`\x` to its :math:`k`\ th nearest 
neighbor, and :math:`\rm{vol}(S_d(h_\x))` is the volume of the 
:math:`d`-dimensional hypersphere :math:`S_d(h\x)` centered at :math:`\x`, with
radius :math:`h_\x`.

15.3 Density-Based Clustering: DENCLUE
--------------------------------------

**Density Attractors and Gradient**

A point :math:`\x^*` is called a *density attractor* if it is a local maxima of 
the probability density funciton :math:`f`.

The gradient at a point :math:`\x` can be computed as the multivariate 
derivative of the probability density, given as

.. math::

    \nabla\hat{f}(\x)=\frac{\pd}{\pd\x}\hat{f}(\x)=\frac{1}{nh^d}\sum_{i=1}^n\frac{\pd}{\pd\x}K\bigg(\frac{\x-\x_i}{h}\bigg)

For the Gaussian kernel, we have

.. math::

    \frac{\pd}{\pd\x}K(\z)&=\bigg(\frac{1}{(2\pi)^{d/2}}\exp
    \bigg\{-\frac{\z^T\z}{2}\bigg\}\bigg)\cd-\z\cd\frac{\pd\z}{\pd\x}

    &=K(\z)\cd-\z\cd\frac{\pd\z}{\pd\x}

Setting :math:`\z=\frac{\x-\x_i}{h}` above, we get

.. math::

    \frac{\pd}{\pd\x}K\bigg(\frac{\x-\x_i}{h}\bigg)=K\bigg(\frac{\x-\x_i}{h}
    \bigg)\cd\bigg(\frac{\x_i-\x}{h}\bigg)\cd\bigg(\frac{1}{h}\bigg)

The gradient at a point :math:`\x` is given as

.. note::

    :math:`\dp\nabla\hat{f}{\x}=\frac{1}{nh^{d+2}}\sum_{i=1}^nK\bigg(\frac{\x-\x_i}{h}\bigg)\cd(\x_i-\x)`

We say that :math:`\x^*` is a *density attractor* for :math:`\x`, 
or alternatively that :math:`\x` is *density attracted* to :math:`\x^*`, if a
hill climbing process started at :math:`\x` converges to :math:`\x^*`.

The typical approach is to use the graident-ascent method to compute 
:math:`\x^*`, that is, starting from :math:`\x`, we iteratively update it at
each step :math:`t` via the update rule:

.. math::

    \x_{t+1}=\x_t+\eta\cd\nabla\hat{f}(\x_t)

where :math:`\eta>0` is the step size.
One can directly optimize the move direction by setting the gradient to the zero vector:

.. math::

    \nabla\hat{f}(\x)&=\0

    \frac{1}{nh^{d+2}}\sum_{i=1}^nK\bigg(\frac{\x-\x_i}{h}\bigg)\cd(\x_i-\x)&=\0

    \x\cd\sum_{i=1}^nK\bigg(\frac{\x-\x_i}{h}\bigg)&=\sum_{i=1}^nK\bigg(\frac{\x-\x_i}{h}\bigg)\x_i

    \x&=\frac{\sum_{i=1}^nK(\frac{\x-\x_i}{h})\x_i}{\sum_{i=1}^nK(\frac{\x-\x_i}{h})}

The point :math:`\x` is involved on both the left- and right-hand sides above;
however, it can be used to obtain the following iterative update rule:

.. note::

    :math:`\dp\x_{t+1}=\frac{\sum_{i=1}^nK(\frac{\x_t-\x_i}{h})\x_i}{\sum_{i=1}^nK(\frac{\x_t-\x_i}{h})}`

**Center-defined Cluster**

A cluster :math:`C\subseteq\D`, is called a *Center-defined cluster* if all the 
points :math:`\x\in C` are density attracted to a unique density attractor 
:math:`\x^*`, such that :math:`\hat{f}(\x^*)\geq\xi`, where :math:`\xi` is a 
user-defined minimum density threshold.
In other words,

.. math::

    \hat{f}(\x^*)=\frac{1}{nh^d}\sum_{i=1}^nK\bigg(\frac{\x^*-\x_i}{h}\bigg)\geq\xi

**Density-defined Cluster**

A cluster :math:`C\subseteq\D` is called a *density-based cluster* if there 
exists a set of density attractors :math:`\x_1^*,\x_2^*,\cds,\x_m^*`, such that

#. Each point :math:`\x\in C` is attracted to some attractor :math:`\x_i^*`.

#. Each density attractor has density above :math:`\xi`.
   That is, :math:`\hat{f}(\x_i^*)\geq\xi`.

#. Any two density attractors :math:`\x_i^*` and :math:`\x_j^*` are 
   *density reachable*, that is, there exists a path from :math:`\x_i^*` to
   :math:`\x_j^*`, such that for all points :math:`\y` on the path, 
   :math:`\hat{f}(\y)\geq\xi`.

**DENCLUE Algorithm**

.. image:: ../_static/Algo15.2.png

**DENCLUE: Special Cases**

If we let :math:`h=\epsilon` and :math:`\xi=minpts`, then using a discrete 
kernel DENCLUE yields exactly the same clusters as DBSCAN.

**Computational Complexity**

The time for DENCLUE is dominated by the cost of the hill-climbing process.