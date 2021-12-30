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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

The maximum matching measure selects the mapping between clusters and 
partitions, such that the sum of the number of common points (:math:`n_{ij}`) is 
maximized, provided that onlyl one cluster can match with a given partition.

Formally, we treat the contigency table as a complete weighted bipartite graph 
:math:`G=(V,E)`, where each partition and cluster is a node, that is, 
:math:`V=\cl{C}\cup\cl{T}`, and there exists an edge :math:`(C_i,T_j)\in E`,
with weight :math:`w(C_i,T_i)=n_{ij}`, for all :math:`C_i\in\cl{C}` and 
:math:`T_j\in\cl{T}`.
A *matching* :math:`M` in :math:`G` is a subset of :math:`E`, such that the 
edges in :math:`M` are pairwise nonadjacent, that is, they do not have a common 
vertex.
The maximum matching measure is defined as the *maximum weight matching* in :math:`G`:

.. math::

    match=\arg\max_M\bigg\{\frac{w(M)}{n}\bigg\}

where the weight of a matching :math:`M` is simply the sum of all the edge 
weights in :math:`M`, given as :math:`w(M)=\sum)_{e\in M}w(e)`.
The maximum matching can be computed in time 
:math:`O(|V|^2\cd|E|)=O((r+k)^2rk)`, which is equivalent to :math:`O(k^4)` if 
:math:`r=O(k)`.

**F-Measure**

Given cluster :math:`C_i`, let :math:`j_i` denote the partition that contains 
the maximum number of points from :math:`C_i`, that is, 
:math:`j_i=\max_{j=1}^k\{n_{ij}\}`.
The *precision* of a cluster :math:`C_i` is the same as its purity:

.. note::

    :math:`\dp prec_i=\frac{1}{n_i}\max_{j=1}^k\{n_{ij}\}=\frac{n_{ij_i}}{n_i}`

It measures the fraction of points in :math:`C_i` from the majority partition :math:`T_{j_i}`.

The *recall* of cluster :math:`C_i` is defined as

.. note::

    :math:`\dp recall_i=\frac{n_{ij_i}}{|T_{j_{i}}|}=\frac{n_{ij_i}}{m_{j_i}}`

where :math:`m_{j_i}=|T_{j_i}|`.
It measures the fraction of point in partition :math:`T_{j_i}` shared in common with cluster :math:`C_i`.

The F-measure is the harmonic mean of the precision and recall values for each cluster.
The F-measure for cluster :math:`C_i` is therefore given as

.. note::

    :math:`\dp F_i=\frac{2}{\frac{1}{prec_i}+\frac{1}{recall_i}}=\frac{2\cd prec_i\cd recall_i}{prec_i+recall_i}`
    :math:`\dp=\frac{2n_{ij_i}}{n_i+m_{j_i}}`

The F-measures for the clustering :math:`\cl{C}` is the mean of clusterwise F-measure values:

.. math::

    F=\frac{1}{r}\sum_{i=1}^rF_i

F-measure thus tries to balance the precision and recall values across all the clusters.
For a perfect clustering, when :math:`r=k`, the maximum value of the F-measure is 1.

17.1.2 Entropy-based Measures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Conditional Entropy**

The entropy of a clustering :math:`\cl{C}` is defined as

.. math::

    H(\cl{C})=-\sum_{i=1}^rp_{C_i}\log p_{C_i}

where :math:`p_{C_i}=\frac{n_i}{n}` is the probability of cluster :math:`C_i`.
The entropy of the partitioning :math:`\cl{T}` is defined as

.. math::

    H(\cl{T})=-\sum_{j=1}^kp_{T_j}\log p_{T_j}

where :math:`p_{T_j}=\frac{m_j}{n}` is the probability of partition :math:`T_j`.

The cluster-specific entropy of :math:`\cl{T}`, that is, the conditional entropy 
of :math:`\cl{T}` with respect to cluster :math:`C_i` is defined as

.. math::

    H(\cl{T}|C_i)=-\sum_{j=1}^k\bigg(\frac{n_{ij}}{n_i}\bigg)\log\bigg(\frac{n_{ij}}{n_i}\bigg)

The conditional entropy of :math:`\cl{T}` given clustering :math:`\cl{C}` is then defined as the weighted sum:

.. math::

    H(\cl{T}|\cl{C})=\sum_{i=1}^r\frac{n_i}{n}H(\cl{T}|C_i)=
    \sum_{i=1}^r\sum_{j=1}^k\frac{n_{ij}}{n}\log\bigg(\frac{n_{ij}}{n_i}\bigg)

.. note::

    :math:`\dp=-\sum_{i=1}^r\sum_{j=1}^kp_{ij}\log\bigg(\frac{p_{ij}}{p_{C_i}}\bigg)`

where :math:`p_{ij}=\frac{n_{ij}}{n}` is the probability that a point in cluster 
:math:`i` also belongs to partition :math:`j`.
For a perfect clustering, the conditional entropy value is zero, whereas the 
worst possible conditional entropy value is :math:`\log k`.

.. math::

    H(\cl{T}|\cl{C})&=-\sum_{i=1}^r\sum_{j=1}^kp_{ij}(\log p_{ij}-\log p_{C_i})

    &=-\bigg(\sum_{i=1}^r\sum_{j=1}^kp_{ij}\log p_{ij}\bigg)+\sum_{i=1}^r\bigg(\log p_{C_i}\sum_{j=1}^kp_{ij}\bigg)

    &=-\sum_{i=1}^r\sum_{j=1}^kp_{ij}\log p_{ij}+\sum_{i=1}^rp_{C_i}\log p_{C_i}

    &=H(\cl{C},\cl{T})-H(\cl{C})

where :math:`H(\cl{C},\cl{T})=-\sum_{i=1}^r\sum_{j=1}^kp_{ij}\log p_{ij}` is the 
joint entropy of :math:`\cl{C}` and :math:`\cl{T}`.
The conditional entropy :math:`H(\cl{T}|\cl{C})` thus measures the remaining 
entropy of :math:`\cl{T}` given the clustering :math:`\cl{C}`.
In particular, :math:`H(\cl{T}|\cl{C})=0` if and only if :math:`\cl{T}` is 
completely determined by :math:`\cl{C}`, corresponding to the ideal clustering.
On the other hand, if :math:`\cl{C}` and :math:`\cl{T}` are independent of each
other, then :math:`H(\cl{T}|\cl{C})=H(\cl{T})`, which means that :math:`\cl{C}`
provides no information about :math:`\cl{T}`.

**Normalized Mutual Information**



17.2 Internal Measures
----------------------









17.3 Relative Measures
----------------------