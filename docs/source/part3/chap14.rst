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