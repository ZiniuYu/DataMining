Chapter 17 Clustering Validation
================================

Cluster validation and assessment encompasses three main tasks: *clustering*
*evaluation* seeks to asses the goodness or quality of the clustering,
*clustering stability* seeks to understand the sensitivity of the clustering
result to various algorithmic parameters, and *clustering tendency* assesses the
suitability of applying clustering in the first place.

**External**: external validation measures employ criteria that are not inherent to the dataset.

**Internal**: Internal validation measures employ critieria that are derived from the data itself.

**Relative**: Relative validation measures aim to directly compare different 
clusterings, usually those obtained via different parameter settings for the 
same algorithm.

17.1 External Measures
----------------------

External measures assume that the correct or ground-truth clustering is known a *priori*.
The true cluster labels play the role of external information that is used to evaluate a given clustering.

Let :math:`\D` be a dataset consisting of :math:`n` points :math:`\x_i` in a 
*d*-dimensional space, partitioned into :math:`k` clusters.
Let :math:`y_i\in\{1,2,\cds,k\}` denote the ground-truth cluster membership or label information for each point.
The ground-truth clustering is given as :math:`\cl{T}=\{T_1,T_2,\cds,T_k\}`, 
where the cluster :math:`T_j` consists of all the points with label :math:`j`,
i.e., :math:`T_j=\{\x_i\in\D|y_i=j\}`.
Also, let :math:`\cl{C}=\{C_1,\cds,C_r\}` dentoe a clustering of the same 
dataset into :math:`r` clusters, obtained via some clustering algorithm, and let 
:math:`\hat{y_i}\in\{1,2,\cds,r\}` denote the cluster label for :math:`\x_i`.

External evaluation measures try capture the extent to which points from the 
same partition appear in the same cluster, and the extent to which points from
different partitions are grouped in different clusters.
All of the external measures rely on the :math:`r\times k` *contingency tabel*
:math:`\N` that is induced by a clustering :math:`\cl{C}` and the ground-truth
partitioning :math:`\cl{T}`, defined as follows

.. math::

    \N(i,j)=n_{ij}=|C_i\cap T_j|

In other words, the count :math:`n_{ij}` denotes the number of points that are 
common to cluster :math:`C_i` and ground-truth partition :math:`T_j`.

17.1.1 Matching Based Measures
------------------------------

**Purity**

.. math::

    purity_i=\frac{1}{n_i}\max_{j=1}^k\{n_{ij}\}

The purity of clustering :math:`\cl{C}` is defined as the weighted sum of the clusterwise purity values:

.. note::

    :math:`\dp purity=\sum_{i=1}^r\frac{n_i}{n}purity_i=\frac{1}{n}\sum_{i=1}^r\max_{j=1}^k\{n_{ij}\}`

The larger the purity of :math:`\cl{C}`, the better the agreement with the groundtruth.
The maximum value of purity is 1, when each cluster comprises points from only one partition.
When :math:`r=k`, a purity value of 1 indicates a perfect clustering, with a 
one-to-one correspondence between the clsuters and partitions.
However, purity can be 1 even for :math:`r>k`, when each of the clusters is a subset of a ground-truth partition.
When :math:`r<k`, purity can never by 1, because at least one cluster must contain points from more than one partition.

**Maximum Matching**






17.2 Internal Measures
----------------------









17.3 Relative Measures
----------------------