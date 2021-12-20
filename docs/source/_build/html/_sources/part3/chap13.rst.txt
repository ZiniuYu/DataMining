Chapter 13 Representative-based Clustering
==========================================

Given a dataset :math:`\D` with :math:`n` points :math:`\x_i` in a 
:math:`d`-dimensional space, and given the number of desired clusters :math:`k`,
the goal of representative-based clustering is to partition the dataset into 
:math:`k` groups or clusters, which is called a *clustering* and is denoted as
:math:`\cl{C}=\{C_1,C_2,\cds,C_k\}`.
Further, for each cluster :math:`C_i` there exists a representative point that 
summarizes the cluster, a common choice being the mean (also called the
*centroid*) :math:`\mmu_i` of all points in the cluster, that is,

.. math::

    \mmu_i=\frac{1}{n_i}\sum_{x_j\in C_i}\x_j

where :math:`n_i=|C_i|` is the number of points in cluster :math:`C_i`.

The *exact* number of ways of partitioning :math:`n` points into :math:`k` 
nonempty and disjoint parts is given by the *Stirling numbers of second kind*,
given as

.. math::

    S(n,k)=\frac{1}{k!}\sum_{t=0}^k(-1)^t\bp k\\t \ep(k-t)^n

13.1 K-Means Algorithm
----------------------

Given a clustering :math:`\cl{C}=\{C_1,C_2,\cds,C_k\}` we need some scoring 
function that evaluates its quality or goodness.
This *sum of squared errors* scoring function is defined as

.. note::

    :math:`SSE(\cl{c})=\sum_{i=1}^k\sum_{\x_j\in C_i}\lv\x_j-\mmu_i\rv^2`

The goal is to find the clustering that minimizes the SSE scores:

.. math::

    \cl{C}^*=\arg\min_{\cl{C}}\{SSE(\cl{C})\}

K-means employs a greedy iterative approach to find a clustering that minimizes the SSE objective.

.. image:: ../_static/Algo13.1.png

The cluster assignment step take :math:`O(nkd)` time because for each of the 
:math:`n` points we have to compute its distance to each of the :math:`k` 
clusters, which takes :math:`d` operations in :math:`d` dimensions. 
The centroid re-computation step takes :math:`O(nd)`` time because we have to 
add at total of :math:`n` :math:`d`-dimensional points. 
Assuming that there are :math:`t` iterations, the total time for K-means is given as :math:`O(tnkd)`. 
In terms of the I/O cost it requires :math:`O(t)` full database scans, because 
we have to read the entire database in each iteration.

13.2 Kernel K-Means
-------------------

Assume for the moment that all points :math:`\x_i\in\D` have been mapped to 
their corresponding images :math:`\phi(\x_i)` in feature space.
Let :math:`\K=\{K(\x_i,\x_j)\}_{i,j=1,\cds,n}` denote the :math:`n\times n` 
matrix, where :math:`K(\x_i,\x_j)=\phi(\x_i)^T\phi(\x_j)`.
Let :math:`\{C_1,\cds,C_k\}` specify the partitioning of the :math:`n` points 
into :math:`k` clusters, and let the corresponding cluster means in feature
space be given as :math:`\{\mmu_1^\phi,\cds,\mmu_k^\phi\}`, where

.. math::

    \mmu_i^\phi=\frac{1}{n_i}\sum_{\x_j\in C_i}\phi(\x_j)

is the mean of cluster :math:`C_i` in feature space, with :math:`n_i=|C_i|`.

In feature space, the kernel K-means sum of squared errors objective can be written as

.. math::

    \min_{\cl{C}}SSE(\cl{C})&=\sum_{i=1}^k\sum_{\x_j\in C_i}\lv\phi(\x_j)-\mmu_i^\phi\rv^2

    &=\sum_{i=1}^k\sum_{\x_j\in C_i}\lv\phi(\x_j)\rv^2-2\phi(\x_j)^T\mmu_i^\phi+\lv\mmu_i\rv^2

    &=\sum_{i=1}^k\bigg(\bigg(\sum_{\x_j\in C_i}\lv\phi(\x_j)\rv^2\bigg)-2n_i
    \bigg(\frac{1}{n_i}\sum_{\x_j\in C_i}\phi(\x_j)\bigg)^T\mmu_i^\phi+
    n_i\lv\mmu_i^\phi\rv^2\bigg)

    &=\bigg(\sum_{i=1}^k\sum_{\x_j\in C_i}\phi(\x_j)^T\phi(\x_j)\bigg)-\bigg(\sum_{i=1}^k n_i\lv\mmu_i^\phi\rv^2\bigg)

    &=\sum_{i=1}^k\sum_{\x_j\in C_i}K(\x_j,\x_j)-\sum_{i=1}^k\frac{1}{n_i}
    \sum_{\x_a\in C_i}\sum_{\x_b\in C_i}K(\x_a,\x_b)

    &=\sum_{j=1}^nK(\x_j,\x_j)-\sum_{i=1}^k\frac{1}{n_i}\sum_{\x_a\in C_i}\sum_{\x_b\in C_i}K(\x_a,\x_b)

Consider the distance of a point :math:`\phi(\x_j)` to the mean 
:math:`\mmu_i^\phi` in feature space, which can be computed as

.. math::

    \lv\phi(\x_j)-\mmu_i^\phi\rv^2&=\lv\phi(\x_j)\rv^2-2\phi(\x_j)^T\mmu_i^\phi+\lv\mmu_i^\phi\rv^2

    &=\phi(\x_j)^T\phi(\x_j)-\frac{2}{n_i}\sum_{\x_a\in C_i}\phi(\x_j)^T
    \phi(\x_a)+\frac{1}{n^2}\sum_{\x_a\in C_i}\sum_{\x_b\in C_i}
    \phi(\x_a)^T\phi(\x_b)

    &=K(\x_j,\x_j)-\frac{2}{n_i}\sum_{\x_a\in C_i}K(\x_a,\x_j)+\frac{1}{n_i^2}
    \sum_{\x_a\in C_i}\sum_{\x_b\in C_i}K(\x_a,\x_b)

In the cluster assignment step of kernel K-means, we assign a point to the closest cluster mean as follows:

.. math::

    C^*(\x_j)&=\arg\min_i\{\lv\phi(\x_j)-\mmu_i^\phi\rv^2\}

    &=\arg\min_i\bigg\{K(\x_j,\x_j)-\frac{2}{n_i}\sum_{\x_a\in C_i}K(\x_a,\x_j)+
    \frac{1}{n^2}\sum_{\x_a\in C_i}\sum_{\x_b\in C_i}K(\x_a,\x_b)\bigg\}

    &=\arg\min_i\bigg\{\frac{1}{n_i^2}\sum_{\x_a\in C_i}\sum_{\x_b\in C_i}
    K(\x_a,\x_b)-\frac{2}{n_i}\sum_{\x_a\in C_i}K(\x_a,\x_j)\bigg\}

.. image:: ../_static/Algo13.2.png

The fraction of points reassigned to a different cluster in the current iteration is given as

.. math::

    \frac{n-\sum_{i=1}^k|C_i^T\cap C_i^{t-1}|}{n}=1-\frac{1}{n}\sum_{i=1}^k|C_i^T\cap C_i^{t-1}|

**Computational Complexity**

The total computational complexity of kernel K-means is :math:`O(tn^2)`, where 
:math:`t` is the number of iterations until convergence.
The I/O complexity is :math:`O(t)` scans of the kernel matrix :math:`\K`.

13.3 Expectation-Maximization Clustering
----------------------------------------

Let :math:`\D` consist of :math:`n` points :math:`\x_j` in :math:`d`-dimensional space :math:`\R^d`.
Let :math:`X_a` denote the random variable corresponding to the :math:`a`\ th attribute.
Let :math:`\X=(X_1,X_2,\cds,X_d)` denote the vector random variable across the :math:`d`-attributes, with :math:`\x_j` 
being a data sample from :math:`\X`.

**Gaussian Mixture Model**

Assume that each cluster :math:`C_i` is characterized by a multivariate normal distribution, that is,

.. note::

    :math:`\dp f_i(\x)=f(\x|\mmu_i,\Sg_i)=\frac{1}{(2\pi)^{\frac{d}{2}}|\Sg_i|^{\frac{1}{2}}}`
    :math:`\dp\exp\bigg\{-\frac{(\x-\mmu_i)^T\Sg_i\im(\x-\mmu_i)}{2}\bigg\}`

where the cluster mean :math:`\mmu_i\in\R^d` and covariance matrix 
:math:`\Sg_i\in\R^{d\times d}` are both unknown parameters.
:math:`f_i(\x)` is the probability density at :math:`\x` attributable to cluster :math:`C_i`.
We assume that the probability density function of :math:`\X` is given as a 
*Gaussian mixture model* over all the :math:`k` cluster normals, defined as

.. note::

    :math:`\dp f(\x)=\sum_{i=1}^kf_i(\x)P(C_i)=\sum_{i=1}^kf(\x|\mmu_i,\Sg_i)P(C_i)`

where the prior probabilities :math:`P(C_i)` are called the *mixture parameters*, which must satisfy the condition

.. math::

    \sum_{i=1}^kP(C_i)=1

We write the set of all the model parameters compactly as 

.. math::

    \bs\theta=\{\mmu_1,\Sg_1,P(C_1),\cds,\mmu_k,\Sg_k,P(C_k)\}

**Maximum Likelihood Estimation**

Given the dataset :math:`\D`, we define the *likelihood* of :math:`\bs\th` as 
the conditional probability of the data :math:`\D` given the model parameters
:math:`\bs\th`, denoted as :math:`P(\D|\bs\th)`.

.. math::

    p(\D|\bs\th)=\prod_{j=1}^nf(\x_j)

The goal of maximum likelihood estimation (MLE) is to choose the parameters :math:`\bs\th` that maximize the likelihood

.. math::

    \bs\th^*=\arg\max_{\bs\th}\{P(\D|\bs\th)\}

It is typical the maximize the log of the likelihood function

.. math::

    \bs\th^*=\arg\max_{\bs\th}\{\ln P(\D|\bs\th)\}

where the *log-likelihood* function is given as

.. math::
    
    \ln P(\D|\bs\th)=\sum_{j=1}^n\ln f(\x_j)=\sum_{j=1}^n\ln\bigg(\sum_{i=1}^kf(\x_j|\mmu_i,\Sg_i)P(C_i)\bigg)

We can use the expectation-maximization (EM) approach for finding the maximum 
likelihood estimates for the parameters :math:`\bs\th`.
EM is a two-step iterative approach that starts from an initial guess for the parameters :math:`\bs\th`.
Given the current estimates for :math:`\bs\th`, in the *expectation step* EM 
computes the cluster posterior probabilities :math:`P(C_i|\x_j)` via the Bayes
theorem:

.. math::

    P(C_i|\x_j)=\frac{P(C_i\rm{\ and\ }\x_j)}{P(\x_j)}=\frac{P(\x_j|C_i)P(C_i)}{\sum_{a=1}^kP(\x_j|C_a)P(C_a)}

Because each cluster is modeled as a multivariate normal distribution, the 
probability of :math:`\x_j` given cluster :math:`C_i` can be obtained by 
considering a small interval :math:`\epsilon>0` centered at :math:`\x_j`, as
follows:

.. math::

    P(\x_j|C_i)\simeq 2\epsilon\cd f(\x_j|\mmu_i,\Sg_i)=2\epsilon\cd f_i(\x_j)

The posterior probability of :math:`C_i` given :math:`\x_j` is thus given as

.. note::

    :math:`\dp P(C_i|\x_j)=\frac{f_i(\x_j)\cd P(C_i)}{\sum_{a=1}^kf_a(\x_j)\cd P(C_a)}`

and :math:`P(C_i|\x_j)` can be considered as the weight or contribution of the point :math:`\x_j` to cluster :math:`C_i`.
Next, in the *maximization step*, using the weights :math:`P(C_i|\x_j)` EM 
re-estimates :math:`\bs\th`, for each cluster :math:`C_i`.
The re-estimated mean is given as the weighted average of all the points, the
re-estimated covariance matrix is given as the weighted covariance over all 
pairs of dimensions, and the re0estimated prior probability for each cluster is 
given as the fraction of weights that contribute to that cluster.

13.3.1 EM in One Dimension
^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider a dataset :math:`\D` consisting of a single attribute :math:`X`, where 
each point :math:`x_j\in\R` (:math:`j=1,\cds,n`) is a random sample from 
:math:`X`.
For the mixture model, we use univariate normals for each cluster:

.. math::

    f_i(x)=f(x|\mu_i,\sg_i^2)=\frac{1}{\sqrt{2\pi}\sg_i}\exp\bigg\{-\frac{(x-\mu_i)^2}{2\sg_i^2}\bigg\}

with the cluster parameters :math:`\mu_i,\sg_i^2`, and :math:`P(C_i)`.

**Initialization**

For each cluster :math:`C_i`, with :math:`i=1,2,\cds,k`, we can randomly 
initialize the cluster parameters :math:`\mu,\sg_i^2`, and :math:`P(C_i)`.

**Expectation Step**

The posterior probabilities are computed as

.. math::

    P(C_i|x_j)=\frac{f(x_j|\mu_i,\sg_i^2)\cd P(C_i)}{\sum_{a=1}^kf(x_j|\mu_a,\sg_a^2)\cd P(C_a)}

For convenience, we use the notation :math:`w_{ij}=P(C_i|x_j)`, and let

.. math::

    \w_i=(w_{i1},\cds,w_{in})^T

denote the weight vector for cluster :math:`C_i` across all the :math:`n` points.

**Maximization Step**

The re-estimated value for the cluster mean, :math:`\mu_i`, is computed as the weighted mean of all the points:

.. math::

    \mu_i=\frac{\sum_{j=1}^nw_{ij}\cd x_j}{\sum_{j=1}^nw_{ij}}

In terms of the weight vector :math:`\w_i` and the attribute vector :math:`X=(x_1,x_2,\cds,x_n)^T`, we can write as

.. math::

    \mu_i=\frac{\w_i^TX}{\w_i^T\1}

The re-estimated value of the cluster variance is computed as the weighted variance across all the points:

.. math::

    \sg_i^2=\frac{\sum_{j=1}^nw_{ij}(x_j-\mu_i)^2}{\sum_{j=1}^nw_{ij}}

Let :math:`\bar{X}_i=X-\mu_i\1=(x_1-\mu_i,x_2-\mu_i,\cds,x_n-\mu_i)^T=`
:math:`(\bar{x}_{i1},\bar{x}_{i2},\cds,\bar{x}_{in})^T` be the centered 
attribute vector for cluster :math:`C_i`, and let :math:`\bar{X}_i^s` be the
squared vector given as 
:math:`\bar{X}_i^s=(\bar{x}_{i1}^2,\cds,\bar{x}_{in}^2)^T`.
The variance can be expressed compactly as

.. math::

    \sg_i^2=\frac{\w_i^T\bar{X}_i^s}{\w_i^T\1}

The prior probability of cluster :math:`C_i` is re-estimated as the fraction of 
the total weight belonging to :math:`C_i`, computed as

.. math::

    P(C_i)=\frac{\sum_{j=1}^nw_{ij}}{\sum_{a=1}^k\sum_{j=1}^nw_{aj}}=
    \frac{\sum_{j=1}^nw_{ij}}{\sum_{j=1}^n1}=\frac{\sum_{j=1}^nw_{ij}}{n}

where we made use of the fact that

.. math::

    \sum_{i=1}^kw_{ij}=\sum_{i=1}^kP(C_i|x_j)=1

In vector notation the prior probability can be written as

.. math::

    P(C_i)=\frac{\w_i^T\1}{n}

**Iteration**

Starting from an initial set of values for the cluster parameters 
:math:`\mu_i,\sg_i^2`, and :math:`P(C_i)` for all :math:`i=1,\cds,k`, the EM
algorithm applies the expectation step to compute the weights 
:math:`w_{ij}=P(C_i|x_j)`.

13.3.2 EM in :math:`d` Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each cluster :math:`C_i`, we now need to estimate the :math:`d`-dimensional mean vector:

.. math::

    \mmu_i=(\mu_{i1},\mu_{i2},\cds,\mu_{id})^T

and the :math:`d\times d` covariance matrix:

.. math::

    \Sg_i=\bp (\sg_1^i)^2&\sg_{12}^i&\cds&\sg_{id}^i\\
    \sg_{21}^i&(\sg_2^i)^2&\cds&\sg_{2d}^i\\\vds&\vds&\dds&\vds\\
    \sg_{d1}^i&\sg_{d2}^i&\cds&(\sg_d^i)^2 \ep

One simplification is to assume that all dimensions are independent, which leads to a diagonal covariance matrix:

.. math::

    \Sg_i=\bp (\sg_1^i)^2&0&\cds&0\\0&(\sg_2^i)^2&\cds&0\\\vds&\vds&\dds&\vds\\0&0&\cds&(\sg_d^i)^2 \ep

**Initialization**

For each cluster :math:`C_i`, with :math:`i=1,2,\cds,k`, we can randomly 
initialize the cluster parameters :math:`\mmu,\Sg_i`, and :math:`P(C_i)`.

**Expectation Step**

.. math:: 
    
    w_{ij}=P(C_i|\x_j)=\frac{f_i(\x_j)\cd P(C_i)}{\sum_{a=1}^kf_a(\x_j)\cd P(C_a)}

**Maximization Step**

The mena :math:`\mmu_i` for cluster :math:`C_i` can be estimated as

.. math::

    \mmu_i=\frac{\sum_{j-1}^nw_{ij}\cd\x_j}{\sum_{j=1}^nw_{ij}}=\frac{\D^T\w_i}{\w_i^T\1}

Let :math:`\bar\D_i=\D-\1\cd\mmu_i^T` be the centered data matrix for cluster :math:`C_i`.
Let :math:`\bar\x_{ji}=\x_j-\mmu_i\in\R^d` denote the :math:`j`\ th centered point in :math:`\bar\D_i`.
We can express :math:`\Sg_i` as 

.. math::

    \Sg_i=\frac{\sum_{j=1}^nw_{ij}\bar\x_{ji}\bar\x_{ji}^T}{\w_i^T\1}

The covariance between dimensions :math:`X_a` and :math:`X_b` is estimated as

.. math::

    \sg_{ab}^i=\frac{\sum_{j=1}^nw_{ji}(x_{ja}-\mu_{ia})(x_{jb}-\mu_{ib})}{\sum_{j=1}^nw_{ij}}

The prior probability :math:`P(C_i)` for each cluster is the same as in the one-dimensional case, given as

.. math::

    P(C_i)=\frac{\sum_{j=1}^nw_{ij}}{n}=\frac{\w_i^T\1}{n}

**EM Clustering Algorithm**

.. image:: ../_static/Algo13.3.png

**Computational Complexity**

The computational complexity of the EM method is :math:`O(t(kd^3+nkd^2))`, where :math:`t` is the number of iterations.
If we use a diagonal covariance matrix, then the complexity is therefore :math:`O(tnkd)`.
The I/O complexity for the EM algorithm is :math:`O(t)` complete databases scans 
because we read the entire set of points in each iteration.

**K-means as Specialization of EM**

K-menas can be considered as a special case of the EM algorithm, obtained as follows:

.. math::

    P(C_i|\x_j)=\left\{\begin{array}{lr}1\quad\rm{if\ }C_i=\arg\min_{C_a}
    \{\lv\x_j-\mmu_a\rv^2\}\\0\quad\rm{otherwise}\end{array}\right.

The posterior probability :math:`P(C_i|\x_j)` is given as

.. math::

    P(C_i|\x_j)=\frac{P(\x_j|C_i)P(C_i)}{\sum_{a=1}^kP(\x_j|C_a)P(C_a)}

.. note::

    :math:`P(C_i|\x_j)=\left\{\begin{array}{lr}1\quad\rm{if\ }\x_j\in C_i,\rm{\ i.e.,\ if\ }C_i=\arg\min_{C_a}\{\lv\x_j-\mmu_a\rv^2\}\\0\quad\rm{otherwise}\end{array}\right.`

13.3.3 Maximum Likelihood Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Estimation of Mean**

**Estimation of Covariance Matrix**

**Estimating the Prior Probability: Mixture Parameters**

13.3.4 EM Approach
^^^^^^^^^^^^^^^^^^

**Expectation Step**

**Maximization Step**