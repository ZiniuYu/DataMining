Chapter 4 Graph Data
====================

4.1 Graph Concepts
------------------

**Graph**

A *graph* :math:`G=(V,E)` is a mathematical structure consisting of a finite 
nonempty set :math:`V` of *vertices* or *nodes*, and a set 
:math:`E\seq V\times V` of *edges* consisting of *unordered* pairs of
vertices.
An edge from a node to itself, :math:`(v_i,v_i)`, is called a *loop*.
An undirected graph without loops is called a *simple graph*.
An edge :math:`e=(v_i,v_j)` between :math:`v_i` and :math:`v_j` is said to be 
*incident with* nodes :math:`v_i` and :math:`v_j`l in this case we also say that
:math:`v_i` and :math:`v_j` are *adjacent* to one another, and that they are
*neighbors*.
The number of nodes in the graph :math:`G`, given as :math:`|V|=n` is called the
*order* of the graph, and the number of edges in the graph, given as 
:math:`|E|=m`, is called the *size* of :math:`G`.

A *directed graph* or *digraph* has an edge set :math:`E` consisting of *ordered* pairs of vertices.
A directed edge :math:`(v_i,v_j)` is also called an *arc*, and is said to be *from* :math:`v_i` *to* :math:`v_j`.
We also say that :math:`v_i` is the *tail* and :math:`v_j` the *head* of the arc.

A *weight graph* consists of a graph together with a weight :math:`w_{ij}` for each edge :math:`(v_i,v_j)\in E`.

**Subgraphs**

A graph :math:`H=(V_H,E_H)` is called a *subgraph* of :math:`G=(V,E)` if 
:math:`V_H\seq V` and :math:`E_H\seq E`.
We also say that :math:`G` is a *supergraph* of :math:`H`.
Given a subset of the vertices :math:`V\pr\seq V`, the *induced subgraph* 
:math:`G\pr=(V\pr,E\pr)` consists exactly of all the edges present in :math:`G` 
between vertices in :math:`V\pr`.
A (sub)graph is called *complete* (or a *clique*) if there exists an edge between all pairs of nodes.

**Degree**

The *degree* of a node :math:`v_i\in V` is the number of edges incident with it, 
and is denoted as :math:`d(v_i)` or just :math:`d_i`.
The *degree sequence* of a graph is the list of the degrees of the nodes sorted in non-increasing order.

Let :math:`N_k` denote the number of vertices with degree :math:`k`.
The *degree frequency distribution* of a graph is given as

.. math::

    (N_0,N_1,\cds,N_t)

where :math:`t` is the maximum degree for a node in :math:`G`.
Let :math:`X` be a random variable donoting the degree of a node.
The *degree distribution* of a graph gives the probability mass function :math:`f` for :math:`X`, given as

.. math::

    (f(0),f(1),\cds,f(t))

where :math:`f(k)=P(X=k)=\frac{N_k}{n}` is the probability of a node with degree :math:`k`.

For directed graphs, the *indegree* of node :math:`v_i` denoted as 
:math:`id(v_i)`, is the number of edges with :math:`v_i` as head, that is, the
number of incoming edges at :math:`v_i`.
The *outdegree* of :math:`v_i`, denoted :math:`od(v_i)`, is the number of edges
with :math:`v_i` as the tail, that is, the number of outgoing edges from 
:math:`v_i`.

**Path and Distance**

A *walk* in a graph :math:`G` between nodes :math:`x` and :math:`y` is an 
ordered sequence of vertices, starting at :math:`x` and ending at :math:`y`,

.. math::

    x=v_0,v_1,\cds,v_{t-1},v_t=y

such that there is an edge between every pair of consecutive verices, that is
:math:`(v_{i-1},v_i)\in E` for all :math:`i=1,2,\cds,t`.
The length of the walk, :math:`t`, is measured in terms of *hops*--the number of edges along the walk.
Both the vertices and edges may be repeated in a walk.
A walk starting and ending at the same vertex is called *closed*.
A *trail* is a walk with distinct edges, and a *path* is a walk with *distinct* 
vertices (with the exception of the start and end vertices).
A closed path with length :math:`t\geq 3` is called a *cycle*.

A path of minumum length between nodes :math:`x` and :math:`y` is called a 
*shortest path*, and the length of the shortest path is called the *distance*
between :math:`x` and :math:`y`, denoted as :math:`d(x,y)`.
If no path exists between the two nodes, the distance is assumed to be :math:`d(x,y)=\infty`.

**Connectedness**

Two nodes :math:`v_i` and :math:`v_j` are said to be *connected* if there exists a path between them.
A graph is *connected* if there is a path between all pairs of vertices.
A *connected component*, or just *component*, of a graph is a maximal connected subgraph.
If a graph has only one component it is connected; otherwise it is *disconnected*.

For a directed graph, we say that it is *strongly connected* if there is a 
(directed) path between all ordered pairs of vertices.
We say that it is *weakly connected* if there exists a path between node pairs only by considering edges as undirected.

**Adjacency Matrix**

A graph :math:`G=(V,E)`, with :math:`|V|=n` vertices, can be conveniently 
represented in the form of an :math:`n\times n`, symmetric binary
*adjacency matrix*, :math:`\A`, defined as

.. note::

    :math:`\A(i,j)=\left\{\begin{array}{lr}1\quad\rm{if\ }v_i\rm{\ is\ adjacent\ to\ }v_j\\0\quad\rm{otherwise}\end{array}\right.`

If the graph is directed, then the adjacency matrix :math:`\A` is not symmetric, 
as :math:`(v_i,v_j)\in E` does not imply that :math:`(v_j,v_i)\in E`.

If the graph is weighted, then we obtain an :math:`n\times n` *weighted adjacency matrix*, :math:`\A`, defined as

.. note::

    :math:`\A(i,j)=\left\{\begin{array}{lr}w_{ij}\quad\rm{if\ }v_i\rm{\ is\ adjacent\ to\ }v_j\\0\quad\rm{otherwise}\end{array}\right.`

where :math:`w_{ij}` is the weight on edge :math:`(v_i,v_j)\in E`.
A weighted adjacency matrix can always be converted into a binary one, if 
desired, by using some threshold :math:`\tau` on the edge weights

.. math::

    \A(i,j)=\left\{\begin{array}{lr}1\quad\rm{if\ }w_{ij}\geq\tau\\0\quad\rm{otherwise}\end{array}\right.

**Graph from Data Matrix**

Let :math:`\D` be a dataset consisting of :math:`n` points :math:`\x_i\in\R^d` in a :math:`d`-dimensional space.
We can define a weighted graph :math:`G=(V,E)`, where there exists a node for 
each point in :math:`\D`, and there exists a node for each point in :math:`\D`,
and there exists an edge between each pair of points, with weight

.. math::

    w_{ij}=sim(\x_i,\x_j)

where :math:`sim(\x_i,\x_j)` denotes the similarity between points :math:`\x_i` and :math:`\x_j`.
For instance, similarity can be defined as being inversely related to the 
Euclidean distance between the points via the transformation

.. math::

    w_{ij}=sim(\x_i,\x_j)=\exp\bigg\{-\frac{\lv\x_i-\x_j\rv^2}{2\sg^2}\bigg\}

where :math:`\sg` is the spread parameter.

4.2 Topological Attributes
--------------------------

The topological attributes of graphs are *local* if they apply to only a single 
node, or *global* if they refer to the entire graph.

**Degree**

The degree of a node :math:`\v_i` is defined as the number of its neighbors.

.. note::

    :math:`\dp d_i=\sum_j\A(i,j)`

One of the simplest global attribute is the *average degree*:

.. note::

    :math:`\dp \mu_d=\frac{\sum_id_i}{n}`

**Average Path Length**

The *average path length*, also called the *characteristic path length*, of a connected graph is given as

.. note::

    :math:`\dp\mu_L=\frac{\sum_i\sum_{j>i}d(v_i,v_j)}{\bp n\\2 \ep}=\frac{2}{n(n-1)}\sum_i\sum_{j>i}d(v_i,v_j)`

For a directed graph, the average is over all ordered pairs of vertices:

.. note::

    :math:`\dp\mu_L=\frac{1}{n(n-1)}\sum_i\sum_jd(v_i,v_j)`

For a disconnected graph the average is taken over only the connected pairs of vertices.

**Eccentricity**

The *eccentricity* of a node :math:`v_i` is the maximum distance from :math:`v_i` to any other node in the graph:

.. note::

    :math:`\dp e(v_i)=\max_j\{d(v_i,v_j)\}`

If the graph is disconnected the eccentricity is computed only over pairs of 
vertices with finite distance, that is, only for verticese connected by a path.

**Radius and Diameter**

The *radius* of a connected graph, denoted :math:`r(G)`, is the minimum eccentricity of any node in the graph:

.. note::

    :math:`r(G)=\min_i\{e(v_i)\}=\min_i\{\max_j\{d(v_i,v_j)\}\}`

The *diameter*, denoted :math:`d(G)`, is the maximum eccentricity of any vertex in the graph:

.. note::

    :math:`d(G)=\max_i\{e(v_i)\}=\max_{i,j}\{d(v_i,v_j)\}`

For a disconnected graph, the diameter is the maximum eccentricity over all the connected components of the graph.

The diameter of a graph :math:`G` is sensitive to outliers.
A more robust notion is *effective diameter*, defined as the minimum number of 
hops for which a large fraction, typically :math:`90\%`, of all connected pairs
of nodes can reach each other.

**Clustering Coefficient**

The *clustering coefficient* of a node :math:`v_i` is a measure of the density 
of edges in the neighborhood of :math:`v_i`.
Let :math:`G_i=(V_i,E_i)` be the subgraph induced by the neighbors of vertex :math:`v_i`.
Note that :math:`v_i\notin V_i`, as we assume that :math:`G` is simple.
Let :math:`|V_i|=n_i` be the number of neighbors of :math:`v_i` and 
:math:`|E_i|=m_i` be the number of edges among the neighbors of :math:`v_i`.
The clustering coefficient of :math:`v_i` is defined as

.. note::

    :math:`\dp C(v_i)=\frac{\rm{no.\ of\ edges\ in\ }G_i}{\rm{maximum\ number\ of\ edges\ in\ }G_i}=`
    :math:`\dp\frac{m_i}{\bp n_i\\2 \ep}=\frac{2\cd m_i}{n_i(n_i-1)}`

The *clustering coefficient* of a graph :math:`G` is simply the average 
clustering coefficient over all the nodes, given as

.. note::

    :math:`\dp C(G)=\frac{1}{n}\sum_iC(v_i)`

Because :math:`C(v_i)` is well defined only for nodes with degree 
:math:`d(v_i)\geq 2`, we can define :math:`C(v_i)=0` for nodes with degree less 
than 2.

Define the subgraph composed of the edges :math:`(v_i,v_j)` and :math:`(v_i,v_k)` 
to be a *connected triple* centered at :math:`v_i`.
A connected triple centered at :math:`v_i` that includes :math:`(v_j,v_k)` is called a *triangle*.
The clustering coefficient of node :math:`v_i` can be expressed as

.. math::

    C(v_i)=\frac{\rm{no.\ of\ triangles\ including\ }v_i}{\rm{no.\ of\ connected\ triples\ centered\ at\ }v_i}

The *transitivity* of the graph is defined as

.. math::

    T(G)=\frac{3\times\rm{no.\ of\ triangles\ in\ }G}{\rm{no.\ of\ connected\ triples\ in\ }G}

**Efficiency**

The *efficiency* for a pair of nodes :math:`v_i` and :math:`v_j` is defined as :math:`\frac{1}{d(v_i,v_j)`/
If :math:`v_i` and :math:`v_j` are not connected, then :math:`d(v_i,v_j)=\infty` 
and the efficiency is :math:`1/\infty=0`.
The *efficiency* of a graph :math:`G` is  the average efficiency over all pairs 
of nodes, whether connected or not, given as

.. math::

    \frac{2}{n(n-1)}\sum_i\sum_{j>i}\frac{1}{d(v_i,v_j)}

The maximum efficiency value is 1, which holds for a complete graph.

The *local efficiency* for a node :math:`v_i` is defined as the efficiency of 
the subgraph :math:`G_i` induced by the neighbors of :math:`v_i`.

4.3 Centrality Analysis
-----------------------

The notion of *centrality* is used to rank the vertices of a graph in terms of how "central" or important they are.
A centrality can be formally defined as a function :math:`c:V\ra\R`, that induces a total order on :math:`V`.
We say that :math:`v_i` is at least as central as :math:`v_j` if :math:`c(v_i)\geq c(v_j)`.

4.3.1 Basic Centralities
^^^^^^^^^^^^^^^^^^^^^^^^

**Degree Centrality**

The simplest notion of centrality is the degree :math:`d_i` of a vertex 
:math:`v_i`--the higher the degree, the more important or central the vertex.

**Eccentricity Centrality**

Eccentricity centrality is defined as follows:

.. note::

    :math:`\dp c(v_i)\frac{1}{e(v_i)}=\frac{1}{\max_j\{d(v_i,v_j)\}}`

A node :matj:`v_i` that has the least eccentricity, that is, for which the
eccentricity equals the graph radius, :math:`e(v_i)=r(G)`, is called a 
*center node*, whereas a node that has the highest eccentricity, that is, for
which eccentricity equals the graph diameter, :math:`e(v_i)=d(G)`, is called a
*periphery node*.

Eccentricity centrality is related to the problem of *facility location*, that
is, choosing the optimum location for a resource or facility.

**Closeness Centrality**

Closeness centrality uses the sum of all the distances to rank how central a node is

.. note::

    :math:`\dp c(v_i)=\frac{1}{\sum_jd(v_i,v_j)}`

A node :math:`v_i` with the smallest total distance, :math:`\sum_jd(v_i,v_j)` is called the *median node*.

**Betweenness Centrality**

The Betweenness centrality measures how many shortest paths between all pairs of vertices include :math:`v_i`.
This gives an indication as to the central "monitoring" role played by :math:`v_i` for various pairs of nodes.
Let :math:`\eta_{jk}` denote the number of shortest paths between vertices 
:math:`v_j` and :math:`v_k`, and let :math:`\eta_{jk}(v_j)` denote the number of
such paths that include or contain :math:`v_i`.
Then the fraction of paths through :math:`v_i` is denoted as

.. math::

    \gamma_{jk}(v_i)=\frac{\eta_{jk}(v_i)}{\eta_{ij}}

If the two vertices :math:`v_i` and :math:`v_k` are not connected, we assume :math:`\gamma_{jk}(v_i)=0`.

The betweenness centrality for a node :math:`v_i` is defined as

.. note::

    :math:`\dp c(v_i)=\sum_{j\neq i}\sum_{k\neq i,k>j}\gamma_{jk}(v_i)=`
    :math:`\dp\sum_{j\neq i}\sum_{k\neq i,k>j}\frac{\eta_{jk}(v_i)}{\eta_{jk}}`

4.3.2 Web Centralities
^^^^^^^^^^^^^^^^^^^^^^

**Prestige**

Let :math:`G=(V,E)` be a directed graph, with :math:`|V|=n`.
The adjacency matrix of :math:`G` is an :math:`n\times n` asymmetric matrix :math:`\A` givne as

.. math::

    \A(u,v)=\left\{\begin{array}{lr}1\quad\rm{if\ }(u,v)\in E\\0\quad\rm{if\ }(u,v)\notin E\end{array}\right.

Let :math:`p(u)` be a positive real number, called the *prestige* or *eigenvector centrality* score for node :math:`u`.

.. math::

    p(u)=\sum_u\A(u,v)\cd p(u)=\sum_u\A^T(v,u)\cd p(u)

Across all the nodes, we can recursively express the prestige scores as

.. note::

    :math:`\p\pr=\A^T\p`

where :math:`\p` is an :math:`n`-dimensional column vector corresponding to the prestige scores for each vertex.

.. math::

    \p_k&=\A^T\p_{k-1}

    &=\A^T(\A^T\p_{k-2})=(\A^T)^2\p_{k-2}

    &=(\A^T)^2(\A^T\p_{k-3})=(\A^T)^3\p_{k-3}

    &\vds

    &=(\A^T)^k\p_0

The dominant eigenvector of :math:`\A^T` and the corresponding eigenvalue can be computed using the *power iteration*
approach.

.. image:: ./_static/Algo4.1.png

**PageRank**

The PageRank of a Web page is defined to be the probability of a random web surfer landing at that page.

**Normalized Prestige**

We assume for the moment that each node :math:`u` has outdegree at least 1.
Let :math:`od(u)=\sum_v\A(u,v)` denote the outdegree of node :math:`u`.
Because a randodm surfer can choose among any of its outgoing links, if there is
a link from :math:`u` to :math:`v`, then the probability of visiting :math:`v`
from :math:`u` is :math:`\frac{1}{od(u)}`

Staring from an initial probability or PageRank :math:`p_0(u)` for each node, such that

.. math::

    \sum_up_0(u)=1

we can compute an updated PageRank vector for :math:`v` as follows:

.. math::

    p(v)=\sum_u\frac{\A(u,v)}{od(u)}\cd p(u)=\sum_u\N(u,v)\cd p(u)=\sum_u\N^T(v,u)\cd p(u)

where :math:`\N` is the normalized adjacency matrix of the graph, given as

.. math::

    \N(u,v)=\left\{\begin{array}{lr}\frac{1}{od(u)}\quad\rm{if\ }(u,v)\in E\\0\quad\rm{if\ }(u,v)\notin E\end{array}\right.

Across all nodes, we can express the PageRank vector as follows:

.. math::

    \p\pr=\N^T\p

**Random Jumps**

In the random surfing approach, there is a small probability of jumping from one
node to any of the other nodes in the graph, even if they do not have a link
between them.
For the random surfer matrix, the outdegree of each node is :math:`od(u)=n`, and
the probability of jumping from :math:`u` to any node :math:`v` is simply
:math:`\frac{1}{od(u)}=\frac{1}{n}`.
The PageRank can then be computed analogously as

.. math::

    p(v)=\sum_u\frac{\A_r(u,v)}{od(u)}\cd p(u)=\sum_u\N_r(u,v)\cd p(u)=\sum_u\N_r^T(v,u)\cd p(u)

where :math:`\N_r` is the normalized adjacency matrix of the fully connected Web graph, given as

.. math::

    \N_r=\frac{1}{n}\A_r=\frac{1}{n}\1_{n\times n}

Across all the nodes the random jump PageRank vector can be represented as

.. math::

    \p\pr=\N_r^T\p

**PageRank**

The full PageRank is computed by assuming that with some small probability,
:math:`\alpha`, a random Web surfer jumps from the current node :math:`u` to any
other random node :math:`v`, and with probability :math:`1-\alpha` the user 
follows an existing link from :math:`u` to :math:`v`.

.. note::

    :math:`\p\pr=(1-\alpha)\N^T\p+\alpha\N_r^T\p=((1-\alpha)\N^T+\alpha\N_r^T)\p=\bs{\rm{M}}^T\p`

When a node :math:`u` does not have any outgoing edges, that is, when 
:math:`od(u)=0`, it acts like a sink for the normalized prestige score, and it
can only jump to another random node.
Thus, we need to make sure that if :math:`od(u)=0` then for the row 
corresponding to :math:`u` in :math:`\bs{\rm{M}}`, denoted as 
:math:`\bs{\rm{M}}_u`, we set :math:`\alpha=1`, that is

.. math::

    \bs{\rm{M}}_u=\left\{\begin{array}{lr}\bs{\rm{M}}_u\quad\rm{if\ }od(u)>0\\
    \frac{1}{n}\1_n^T\quad\rm{if\ }od(u)=0\end{array}\right.

**Hub and Authority Scores**

The *authority score* of a page is analogous to PageRank or prestige, and it 
depends on how many "good" pages point to it.
The *hub score* of a  page is based on how many "good" pages it points to.

We denote by :math:`a(u)` the authority score and by :math:`h(u)` the hub score of node :math:`u`.

.. math::

    a(v)=\sum_u\A^T(v,u)\cd h(u)

    h(v)=\sum_u\A(v,u)\cd a(u)

In matrix notation, we obtain

.. note::

    :math:`\a\pr=\A^T\bs{\rm{h}}\quad\bs{\rm{h}}\pr=\A\a`

In fact, we can write the above recursively as follows:

.. math::

    \a_k=\A^T\bs{\rm{h}}_{k-1}=\A^T(\A\a_{k-1})=(\A^T\A)\a_{k-1}

    \bs{\rm{h}}_k=\A\a_{k-1}=\A(\A^T\bs{\rm{h}}_{k-1})=(\A\A^T)\bs{\rm{h}}_{k-1}

In other words, as :math:`k\ra\infty`, the authority score converges to the 
dominant eigenvector of :math:`\A^T\A`, whereas the hub score converges to the
dominant eigenvector of :math:`\A\A^T`.

4.4 Graph Models
----------------

**Small-world Property**

A graph :math:`G` exhibits small-world behavior if the average path length 
:math:`\mu_L` scales logarithmically with the number of nodes in the graph, that
is, if

.. note::

    :math:`\mu_L\varpropto\log n`

A graph is said to have *ultra-small-world* property if the average path length 
is much smaller than :math:`\log n`, that is, if :math:`\mu_L\ll\log n`.

**Scale-free Property**

In many real-world graphs, the probability that a node has degree :math:`k` satisfies the condition

.. note::

    :math:`f(k)\varpropto k^{-\gamma}`