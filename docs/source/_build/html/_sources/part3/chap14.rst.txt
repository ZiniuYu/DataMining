Chapter 14 Hierarchical Clustering
==================================

Given :math:`n` points in a :math:`d`-dimensional space, the goal of 
hierarchical clustering is to create a sequence of nested partitions, which can
be conveniently visualized via a tree or hierarchy of clusters, also called the 
cluster *dendrogram*.

There are two main algorithmic approaches to mine hierarchical clusters: agglomerative and divisive.
Agglomerative strategies work in a bottom-up manner.
That is, starting with each of the :math:`n` points in a separate cluster, they 
repeatedly merge the most similar pair of clusters until all points are members 
of the same cluster.
Divisive strategies do just the opposite, working in a top-down manner. 
Starting with all the points in the same cluster, they recursively split the 
clusters until all points are in separate clusters.

14.1 Preliminaries
------------------

Given a dataset :math:`\D` comprising :math:`n` points 
:math:`\x_i\in\R^D(i=1,2,\cds,n)`, a clustering :math:`\cl{C}=\{C_1,\cds,C_k\}`
is a partition of :math:`\D`, that is, each cluster is a set of points 
:math:`C_i\subseteq\D`, such that the clusters are pairwise disjoint
:math:`C_i\cap C_j=\emptyset` (for all :math:`i\neq j`), and 
:math:`\cup_{i=1}^kC_i=\D`.
A clustering :math:`\cl{A}=\{A_1,\cds,A_r\}` is said to be nested in another
clustering :math:`\cl{B}=\{B_1,\cds,\B_s\}` if and only if :math:`r>s`, and for
each cluster :math:`A_i\in\cl{A}`, there exists a cluster :math:`B_j\in\cl{B}`,
such that :math:`A_i\subseteq B_j`.
Hierarchical clustering yields a sequence of :math:`n` nested partitions :math:`\cl{C}_1,\cds,\cl{C}_n`.
The clustering :math:`\cl{C}_{t-1}` is nested in the clustering :math:`\cl{C}_t`.
The cluster dendrogram is a rooted binary tree that captures this nesting 
structure, with edges between cluster :math:`C_i\in\cl{C}_{i-1}` and cluster
:math:`C_j\in\cl{C}_t` if :math:`C_i` is nested in :math:`C_j`, that is, if
:math:`C_i\subset C_j`.

**Number of Hierarchical Clusterings**

The number of different nested or hierarchical clusterings corresponds to the 
number of different binary rooted trees or dendrograms with :math:`n` leaves 
with distinct labels.
Any tree with :math:`t` nodes has :math:`t−1` edges. 
Also, any rooted binary tree with :math:`m` leaves has :math:`m−1` internal nodes.
Thus, a dendrogram with :math:`m` leaf nodes has a total of :math:`t=m+m−1=2m−1` 
nodes, and consequently :math:`t−1=2m−2` edges.
The total number of different dendrograms with :math:`n` leaves is obtained by the following product:

.. math::

    \prod_{m=1}^{n-1}(2m-1)=1\times 3\times 5\times 7\times\cds\times(2n-3)=(2n-3)!!

14.2 Agglomerative Hierarchical Clustering
------------------------------------------

.. image:: ../_static/Algo14.1.png

14.2.1 Distance between Clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The between-cluster distances are ultimately based on the distance between two 
points, which is typically computed using the Euclidean distance or :math:`L_2`
-*norm*, defined as

.. math::

    \lv\x-\y\rv=\bigg(\sum_{i=1}^d(x_i-y_i)^2\bigg)^{1/2}

**Single Link**

Given two clusters :math:`C_i` and :math:`C_j`, the distance between them, 
denoted :math:`\delta(C_i,C_j)`, is defined as the minimum distance between a
point in :math:`C_i` and a point in :math:`C_j`

.. note::

    :math:`\delta(C_i,C_j)=\min\{\lv\x-\y\rv|\x\in C_i,\y\in C_j\}`

**Complete Link**

The distance between two clusters is defined as the maximum distance between a 
point in :math:`C_i` and a point in :math:`C_j`:

.. note::

    :math:`\delta(C_i,C_j)=\max\{\lv\x-\y\rv|\x\in C_i,\y\in C_j\}`

**Group Average**

The distance between two clusters is defined as the average pairwise distance 
between points in :math:`C_i` and :math:`C_j`:

.. note::

    :math:`\dp\delta(C_i,C_j)=\frac{\sum_{\x\in C_i}\sum_{\y\in C_j}\lv\x-\y\rv}{n_i\cd n_j}`

where :math:`n_i=|C_i|` denotes the number of points in cluster :math:`C_i`.

**Mean Distance**

The distance between two clusters is defined as the distance between the means or centroids of the two clusters:

.. note::

    :math:`\delta(C_i,C_j)=\lv\mmu_i-\mmu_j\rv`

where :math:`\mmu_i=\frac{1}{n_i}\sum_{\x\in C_i}\x`.

**Minimum Variance: Ward's Method**

The sum of a squared errors (SSE) for a given cluster :math:`C_i` is given as

.. math::

    SSE_i&=\sum_{\x\in C_i}\lv\x-\mmu_i\rv^2
    
    &=\sum_{\x\in C_i}\lv\x-\mmu_i\rv^2
    
    &=\sum_{\x\in C_i}\x^T\x-2\sum_{\x\in C_i}\x^T\mmu_i+\sum_{\x\in C_i}\mmu_i^T\mmu_i
    
    &=\bigg(\sum_{\x\in C_i}\x^T\x\bigg)-n_i\mmu_i^T\mmu_i

The SSE for a clustering :math:`\cl{C}=\{C_1,\cds,C_m\}` is given as

.. math::

    SSE=\sum_{i=1}^mSSE_i=\sum_{i=1}^m\sum_{\x\in C_i}\lv\x-\mmu_i\rv^2

After simplification, we get

.. note::

    :math:`\dp\delta(C_i,C_j)=\bigg(\frac{n_in_j}{n_i+n_j}\bigg)\lv\mmu_i-\mmu_j\rv^2`

14.2.2 Updating Distance Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Whenever two clusters :math:`C_i` and :math:`C_j` are merged into 
:math:`C_{ij}`, we need to update the distance matrix by recomputing the 
distances from the newly created cluster :math:`C_{ij}` to all other clusters 
:math:`C_r` (:math:`r \ne i` and :math:`r \ne j`).
The Lance–Williams formula provides a general equation to recompute the 
distances for all of the cluster proximity measures we considered earlier; it is 
given as

.. note::

    :math:`\delta(C_{ij},C_r)=\alpha_i\cd\delta(C_i,C_r)+\alpha_j\cd\delta(C_j,C_r)+\beta\cd\delta(C_i,C_j)+\gamma\cd|\delta(C_i,C_r)-\delta(C_j,C_r)|`

The coefficients :math:`\alpha_i,\alpha_j,\beta` and :math:`\gamma` differ from one measure to another.

14.2.3 Computational Complexity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The computational complexity of hierarchical clustering is :math:`O(n^2\log n)`.