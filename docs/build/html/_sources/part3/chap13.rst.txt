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