Chapter 25 Neural Networks
==========================

*Artificial neural networks* or simply *neural networks* are inspired by biological neuronal networks.
Artifical neural networks are comprised of abstract neurons that try to mimic real neurons at a very high level.
They can be described via a weighte directed graph :math:`G=(V,E)`, with each 
node :math:`v_i\in V` representing a neuron, and each directed edge 
:math:`(v_i,v_j)\in E` representing a synaptic to dendritic connection from 
:math:`v_i` to :math:`v_j`.
The weight of the edge :math:`w_{ij}` denotes the synaptic strength.
Nerual networks are characterized by the type of activation function used to
generate an output, and the architecture of the network in terms of how the 
nodes are interconnected.

25.1 Artificial Neuron: Activation Functions
--------------------------------------------

A neuro :math:`z_k` has incoming edges from neurons :math:`x_1,\cds,x_d`.
Let :math:`x_i` denotes neuron :math:`i`, and also the value of that neuron.
The net input at :math:`z_k`, denoted :math:`net_k`, is given as the weighted sum

.. note::

    :math:`net_k=b_k+\sum_{i=1}^dw_{ik}\cd x_i=b_k+\w^T\x`

where :math:`\w_k=(w_{1k},w_{2k},\cds,w_{dk})^T\in\R^d` and :math:`\x=(x_1,x_2,\cds,x_d)^T\in\R^d` is an input point.
Notice that :math:`x_0` is a special *bias neuron* whose value is always fixed 
at 1, and the weight from :math:`x_0` to :math:`z_k` is :math:`b_k`, which 
specifies the bias term for the neuron.
Finally, the output value of :math:`z_k` is given as some *activation function*, 
:math:`f(\cd)`, applied to the net input at :math:`z_k`

.. math::

    z_k=f(net_k)

The value :math:`z_k` is then passed along the outgoing edges from :math:`z_k` to other neurons.

**Linear/Identity Function:**

    .. math::

        f(net_k)=net_k

**Step Function:**

    .. math::

        f(net_k)=\left\{\begin{array}{lr}1\quad\rm{if\ }net_k\leq 0\\0\quad\rm{if\ }net_k>0\end{array}\right.

**Rectified Linear Unit (ReLU):**

    .. math::

        f(net_k)=\left\{\begin{array}{lr}1\quad\rm{if\ }net_k\leq 0\\net_k\quad\rm{if\ }net_k>0\end{array}\right.

    An alternative expression for the ReLU activation is given as :math:`f(net_k)=\max\{0,net_k\}`.

**Sigmoid:**

    .. math::

        f(net_k)=\frac{1}{1+\exp\{-net_k\}}

**Hyperbolic Tangent (tanh):**

    .. math::

        f(net_k)=\frac{\exp\{net_k\}-\exp\{-net_k\}}{\exp\{net_k\}+\exp\{-net_k\}}=
        \frac{\exp\{2\cd net_k\}-1}{\exp\{2\cd net_k\}+1}

**Softmax:**

    Softmax is mainly used at the output layer in a neural network, and unlike the 
    other functions it depends not only on the net input at neuron :math:`k`, but it 
    depends on the net signal at all other neurons in the output layer.
    Thus, given the net input vector, 
    :math:`\bs{\rm{net}}=(net_1,net_2,\cds,net_p)^T`, for all the :math:`p` output
    neurons, the output of the softmax function for the :math:`k`\ th neuron is 
    given as

    .. math::

        f(net_k|\bs{\rm{net}})=\frac{\exp\{net_k\}}{\sum_{i=1}^p\exp\{net_i\}}

25.1.1 Derivatives for Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


**Linear/Identity Function:**

    .. math::

        \frac{\pd f(net_j)}{\pd net_j}=1

**Step Function:**

    The step function has a derivative of 0 everywhere except for the 
    discontinuity at 0, where the derivative is :math:`\infty`.

**Rectified Linear Unit (ReLU):**

    .. math::

        \frac{\pd f(net_k)}{\pd net_j}=\left\{\begin{array}{lr}0\quad\rm{if\ }net_j\leq 0\\1\quad\rm{if\ }net_k>0\end{array}\right.

    At 0, we can set the derivative to be any value in the range :math:`[0,1]`.

**Sigmoid:**

    .. math::

        \frac{\pd f(net_j)}{\pd net_j}=f(net_j)\cd (1-f(net_j))

**Hyperbolic Tangent (tanh):**

    .. math::

        \frac{\pd f(net_j)}{\pd net_j}=1-f(net_j)^2

**Softmax:**

    Since softmax is used at the output layer, if we denote the :math:`i`\ th 
    output neuron as :math:`o_i`, then :math:`f(net_i)=o_i`.

    .. math::

        \frac{\pd f(net_j|\bs{\rm{net}})}{\pd net_k}=\frac{\pd o_j}{\pd net_k}=
        \left\{\begin{array}{lr}o+j\cd(1-o_j)\quad\rm{if\ }k=j\\
        -o_k\cd o_j\quad\quad\quad\quad\rm{if\ }k\ne j\end{array}\right.

25.2 Neural Networks: Regression and Classification
---------------------------------------------------

25.2.1 Regression
^^^^^^^^^^^^^^^^^

Consider the multiple (linear) regression problem, where given an input 
:math:`\x_i\in\R^d`, the goal is to predict the response as follows:

.. math::

    \hat{y_i}=b+w_1x_{i1}+w_2x_{i2}+\cds+w_dx_{id}

Given a training data :math:`\D` comprising :math:`n` points :math:`\x_i` in a 
:math:`d`-dimensional space, along with their corresponding true response value
:math:`y_i`, the bias and weights for linear regression are chosen so as to 
minimize the sum of squared errors between the true and predicted response over
all data points

.. math::

    SSE=\sum_{i=1}^n(y_i-\hat{y_i})^2

A neural network with :math:`d+1` input neurons :math:`x_0,x_1,\cds,x_d`, 
including the bias neuron :math:`x_0=1`, and a single output neuron :math:`o`,
all with identity activation functions and with :math:`\hat{y}=o`, represents
the exact same model as multiple linear regression.
Wereas the multiple regression problem has a closed form solution, neural 
networks learn the bias and weights via a gradient descent approach that 
minimizes the squared error.

Neural networks can just as easily model the multivariate (linear) regression 
task, where we have a :math:`p`-dimensional response vector :math:`\y_i\in\R^p`
instead of a single value :math:`y_i`.
That is, the training data :math:`\D` comprises :math:`n` points 
:math:`\x_i\in\R^d` and their true response vectors :math:`\y_i\in\R^p`.
Multivariate regression can be modeled by a neural network with :math:`d+1` 
input neurons, and :math:`p` output neurons :math:`o_1,o_2,\cds,o_p`, with all
input and output neurons using the identity activation function.
A neural network learns the weights by comparing its predicted output 
:math:`\hat\y=\o=(o_1,o_2,\cds,o_p)^T` with the true response vector 
:math:`\y=(y_1,y_2,\cds,y_p)^T`.
That is, training happens by first computing the *error function* or *loss function* between :math:`\o` and :math:`\y`.
When the prediction matches the true output the loss should be zero.
The most common loss function for regression is the squared error function

.. math::

    \cl{E}_\x=\frac{1}{2}\lv\y-\o\rv^2=\frac{1}{2}\sum_{j=1}^p(y_j-o_j)^2

where :math:`\cl{E}_\x` denotes the error on input :math:`\x`.
Across all the points in a dataset, the total sum of squared error is

.. math::

    \cl{E}=\sum_{i=1}^n\cl{E}_{\x_i}=\frac{1}{2}\cd\sum_{i=1}^n\lv\y_i-\o_i\rv^2