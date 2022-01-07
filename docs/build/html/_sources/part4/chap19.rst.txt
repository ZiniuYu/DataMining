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

A tree can be read as set of decision rules, with each rule’s antecedent 
comprising the decisions on the internal nodes along a path to a leaf, and its 
consequent being the label of the leaf node. 
Further, because the regions are all disjoint and cover the entire space, the 
set of rules can be interpreted as a set of alternatives or disjunctions.