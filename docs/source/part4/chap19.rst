Chapter 19 Decision Tree Classifier
===================================

Let :math:`\cl{R}` denote the data space that encompasses the set of input points :math:`\D`.
A decision tree uses an axis-parallel hyperplane to split the data space 
:math:`\cl{R}` into two resulting half-spaces or regines, :math:`\cl{R}_1` and
:math:`\cl{R}_2`, which also induces a partition of the input points into
:math:`\D_1` and :math:`\D_2`, respectively.
Each of these regions is recursively split via axis-parallel hyperplanes until 
the points within an induced partition are relatively pure in terms of their 
class labels, that is, most of the points belong to the same class.
The resulting hierarchy of split decisions constitutes the decision tree model,
with the leaf nodes labeled with the majority class among pooints in those 
regions. To classify a new *test* point we have to recursively evaluate which 
half-space it belongs to until we reach a leaf node in the decision tree, at
which point we predict its class as the label of the leaf.

19.1 Decision Trees
-------------------

**Axis-Parallel Hyperplanes**

A hyperplane :math:`h(\x)` is defined as the set of all points :math:`\x` that satisfy the following equation

.. note::

    :math:`h(\x):\w^T\x+b=0`

Here :math:`\w\in\R^d` is a *weight vector* that is normal to the hyperplane, 
and :math:`b` is the offset of the hyperplane from the origin.
A decision tree considers only *axis-parallel hyperplanes*, that is, the weight 
vector must be parallel to one of the original dimensions or axes :math:`X_j`.
Put differently, the weight vector :math:`\w` is restricted *a priori* to one of 
the standard basis vectors :math:`\{\e_1,\e_2,\cds,\e_d\}`, where 
:math:`\e_i\in\R^d` has a 1 for the :math:`j`\ th dimension, and 0 for all other
dimensions.
If :math:`\x=(x_1,x_2,\cds,x_d)^T` and assuming :math:`\w=\e_j`, we can rewrite as

.. math::

    h(\x):\e_j^T\x+b=x_j+b=0

where the choice of the offset :math:`b` yields different hyperplanes along dimension :math:`X_j`.

**Split Points**

A hyperplane specifies a decision or *split point* because it splits the data space :math:`\cl{R}` into two half-spaces.
All points :math:`\x` such that :math:`h(\x)\leq 0` are on the hyperplane or to 
one side of the hyperplane, whereas all points such that :math:`h(\x)>0` are on
the hyperplane or to one side of the hyperplane, whereas all points such that
:math:`h(\x)>0` are on the other side.
The split point associated with an axis-parallel hyperplane can be written as
:math:`h(\x)\leq 0`, which implies that :math:`x_i+b\leq 0`, or 
:math:`x_i\leq-b`.
Because :math:`x_i` is some value from dimension :math:`X_j` and the offset 
:math:`b` can be chosen to be any value, the generic form of a split point for a 
numeric attribute :math:`X_j` is given as

.. math::

    X_j\leq v

where :math:`v=-b` is some value in the domain of attribute :math:`X_j`.
The decision or split point :math:`X_j\leq v` thus splits the input data space
:math:`\cl{R}` into two regions :math:`\cl{R}_Y` and :math:`\cl{R}_N`, which
denote the set of *all possible points* that satisfy the decision and those that
do not.

**Data partition**

Each split of :math:`\cl{R}` into :math:`\cl{R}_Y` and :math:`\cl{R}_N` also
induces a binary partition of the corresponding input data points :math:`\D`.
That is, a split point of the form :math:`X_j\leq v` induces the data partition

.. math::

    \D_Y=\{\x^T|\x\in\D,x_j\leq v\}
    
    \D_N=\{\x^T|\x\in\D,x_j>v\}

**Purity**

Purity is the fraction of points with the majority label in :math:`\D_j`

.. note::

    :math:`\dp purity(\D_j)=\max_i\bigg\{\frac{n_{ji}}{n_j}\bigg\}`

where :math:`n_j=|\D_j|` is the total number of data points in the region 
:math:`\cl{R}_j`, and :math:`n_{ji}` is the number of points in :math:`\D_j`
with class label :math:`c_i`.

**Categorical Attributes**

For a categorical attribute :math:`X_j`, the split points or decisions are of 
the :math:`X_j\in V`, where :math:`V\subset dom(X_j)`, and :math:`dom(X_j)`
denotes the domain for :math:`X_j`.
It results in two "half-spaces", one region :math:`\cl{R}_Y` consisting of 
points :math:`\x` that satisfy the condition :math:`x_i\in V`, and the other 
region :math:`\cl{R}_N` comprising points that satisfy the condition 
:math:`x_i\notin V`.

**Decision Rules**

A tree can be read as set of decision rules, with each rule???s antecedent 
comprising the decisions on the internal nodes along a path to a leaf, and its 
consequent being the label of the leaf node. 
Further, because the regions are all disjoint and cover the entire space, the 
set of rules can be interpreted as a set of alternatives or disjunctions.

19.2 Decision Tree Algorithm
----------------------------

.. image:: ../_static/Algo19.1.png

19.2.1 Split Point Evaluation Measures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Entropy**

.. note::

    :math:`\dp H(\D)=-\sum_{i=1}^kP(c_i|\D)\log_2P(C_i|\D)`

where :math:`P(c_i|\D)` is the probability of class :math:`c_i` in :math:`\D`, and :math:`k` is the number of classes.
If a region is pure, that is, has points from the same class, then the entropy is zero.
On the other hand, if the classes are all miaxed up, and each appears with equal 
probability :math:`P(c_i|\D)=\frac{1}{k}`, then the entropy has the highest
value, :math:`H(\D)=\log_2k`.

Define the *split entropy* as the weighted entropy of the resulting partitions, given as

.. note::

    :math:`\dp H(\D_Y,\D_N)=\frac{n_Y}{n}H(\D_Y)+\frac{n_N}{n}H(\D_N)`

where :math:`n=|\D|` is the number of points in :math:`\D`, and :math:`n_Y=|\D_Y|` and :math:`n_N=|\D_N|`.

The *information gain* for a given split point is defined as follows:

.. note::

    :math:`Gain(\D,\D_Y,\D_N)=H(\D)-H(\D_Y,\D_N)`

The higher the information gain, the more the reduction in entropy, and the 
better the split point. Thus, given split points and their corresponding 
partitions, we can score each split point and choose the one that gives the 
highest information gain.

**Gini Index**

.. note::

    :math:`\dp G(\D)=1-\sum_{i=1}^kP(c_i|\D)^2`

If the partition is pure, then the probability of the majority class is 1 and 
the probability of all other classes is 0, and thus, the Gini index is 0.
On the other hand, when each class is equally represented, with probability 
:math:`P(c_i|\D)=\frac{1}{k}`, then the Gini index has value 
:math:`\frac{k-1}{k}`.

We can compute the weighted Gini index of a split point as follows:

.. math::

    G(\D_Y,\D_N)=\frac{n_Y}{n}G(\D_Y)+\frac{n_N}{n}G(\D_N)

The lower the Gini index value, the better the split point.

The Classification And Regression Trees (CART) measure is given as

.. note::

    :math:`\dp CART(\D_Y,\D_N)=2\frac{n_Y}{n}\frac{n_N}{n}\sum_{i=1}^k|P(c_i|\D_Y)-P(c_i|\D_N)|`

This measure thus prefers a split point that maximizes the difference between 
the class probability mass function for the two partitions; the higher the CART
measure, the better the split point.

19.2.2 Evaluating Split Points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Numeric Attributes**

One reasonable approach is to consider only the midpoints between two successive 
distinct values for :math:`X` in the sample :math:`\D`.
Because there can be at most :math:`n` distinct values for :math:`X`, there are 
at most :math:`n-1` midpoint values to consider.

Let :math:`\{v_1,\cds,v_m\}` denote the set of all such midpoints, such that :math:`v_1<v_2<\cds<v_m`.
For each split point :math:`X\leq v`, we have to estimate the class PMFs:

.. math::

    \hat{P}(c_i|\D_Y)=\hat{P}(c_i|X\leq v)

    \hat{P}(c_i|\D_N)=\hat{P}(c_i|X>v)

Let :math:`I()` be an indicator variable that takes on the value 1 only when its argument is true, and is 0 otherwise.

.. math::

    \hat{P}(c_i|X\leq v)=\frac{\hat{P}(X\leq v|c_i)\hat{P}(c_i)}
    {\hat{P}(X\leq v)}=\frac{\hat{P}(X\leq v|c_i)\hat{P}(c_i)}
    {\sum_{j=1}^k\hat{P}(X\leq v|c_j)\hat{P}(c_j)}

.. math::

    \hat{P}(c_i)=\frac{1}{n}\sum_{j=1}^nI(y_j=c_i)=\frac{n_i}{n}

Define :math:`N_{vi}` as the number of points :math:`x_j\leq v` with class 
:math:`c_i`, where :math:`x_j` is the value of data point :math:`\x_j` for the 
attribute :math:`X`, given as

.. math::

    N_{vi}=\sum_{j=1}^nI(x_j\leq v\ \rm{and}\ y_j=c_i)

We can then estimate :math:`P(X\leq v|c_i)` as follows:

.. math::

    \hat{P}(X\leq v|c_i)=\frac{\hat{P}(X\leq v\ \rm{and}\ c_i)}{\hat{P}(c_i)}=
    \bigg(\frac{1}{n}\sum_{j=1}^nI(x_j\leq v\ \rm{and}\ y_j=c_i)\bigg)\bigg/
    (n_i/n)=\frac{N_{vi}}{n_i}

.. note::

    :math:`\dp\hat{P}(c_i|\D_N)=\hat{P}(c_i|X>v)=`
    :math:`\dp\frac{\hat{P}(X>v|c_i)|\hat{P}(c_i)}{\sum_{j=1}^k\hat{P}(X>v|c_j)\hat{P}(c_j)}`
    :math:`=\dp\frac{n_i-N_{vi}}{\sum_{j=1}^k(n_j-N_{vj})}`

.. image:: ../_static/Algo19.2.png

The total cost of numeric split point evaluation is :math:`O(n\log n)`.

**Categorical Attributes**

If :math:`X` is a categorical attribute we evaluate split points of the form 
:math:`X\in V`, where :math:`V\subset dom(X)` and :math:`V\ne\emptyset`.
Because the split point :math:`X\in V` yields the same partition as 
:math:`X\in\bar{V}`, where :math:`\bar{V}=dom(X)\\V` is the complement of 
:math:`V`, the total number of distinct partitions is given as

.. math::

    \sum_{i=1}^{\lfloor m/2\rfloor}\bp m\\i\ep=O(2^{m-1})

where :math:`m=|dom(X)|`.
The number of possible split points to consider is therefore exponential in 
:math:`m`, which can pose problems if :math:`m` is large.
One simplification is to restrict :math:`V` to be of size one, so that there are 
only :math:`m` split points of the form :math:`X_j\in\{ v\}`, where 
:math:`v\in dom(X_j)`.

To evaluate a given split point :math:`X\in V` we have to compute the following class probability mass functions:

.. math::

    P(c_i|\D_Y)=P(c_i|X\in V)\quad\quad P(c_i|\D_N)=P(c_i|X\notin V)

Making use of the Bayes theorem, we have

.. math::

    P(c_i|X\in V)=\frac{P(X\in V|c_i)P(c_i)}{P(X\in V)}=\frac{P(X\in V|c_i)P(c_i)}{\sum_{j=1}^kP(X\in V|c_j)P(c_j)}

However, note that a given point :math:`\x` can take on only one value in the 
domain of :math:`X`, and thus the values :math:`v\in dom(X)` are mutually 
exclusive.
Therefore, we have

.. math::

    P(X\in V|c_i)=\sum_{v\in V}P(X=v|c_i)

.. math::

    P(c_i|\D_Y)=\frac{\sum_{v\in V}P(X=v|c_i)P(c_i)}{\sum_{j=1}^k\sum_{v\in V}P(X=v|c_j)P(c_j)}

Define :math:`n_{vi}` as the number of points :math:`\x_j\in\D`, with value
:math:`x_j=v` for attribute :math:`X` and having class :math:`y_j=c_i`:

.. math::

    n_{vi}=\sum_{j=1}^nI(x_j=v\ \rm{and}\ y_j=c_i)

The class conditional empirical PMF for :math:`X` is then given as

.. math::

    \hat{P}(X=v|c_i)=\frac{\hat{P}(X=v\ \rm{and}\ c_i)}{\hat{P}(c_i)}
    =\bigg(\frac{1}{n}\sum_{j=1}^nI(x_j=v\ \rm{and}\ y_i=c_i)\bigg)\bigg/(n_i/n)
    =\frac{n_{vi}}{n_i}

.. note::

    :math:`\dp\hat{P}(c_i|\D_Y)=`
    :math:`\dp\frac{\sum_{v\in V}\hat{P}(X=v|c_i)\hat{P}(c_i)}{\sum_{j=1}^k\sum_{v\in V}\hat{P}(X=v|c_j)\hat{P}(c_j)}`
    :math:`\dp=\frac{\sum_{v\in V}n_{vi}}{\sum_{j=1}^k\sum_{v\in V}n_{vj}}`

.. note::

    :math:`\dp\hat{P}(c_i|\D_N)=\hat{P}(c_i|X\notin V)=\frac{\sum_{v\notin V}n_{vi}}{\sum_{j=1}^k\sum_{v\notin V}n_{vj}}`

.. image:: ../_static/Algo19.3.png

The total cost for categorical attributes is :math:`O(n+mk2^{m-1})`.
If we make the assumption that :math:`2^{m-1}=O(n)`, that is, if we bound the
maximum size of :math:`V` to :math:`l=O(\log n)`, then the cost of categorical
splits is bounded as :math:`O(n\log n)`, ignoring :math:`k`.

19.2.3 Computational Complexity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The total cost in the worst case is :math:`O(dn^2\log n)`.