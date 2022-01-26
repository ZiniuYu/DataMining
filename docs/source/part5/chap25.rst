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

25.2.2 Classification
^^^^^^^^^^^^^^^^^^^^^

Consider the binary classification problem, where :math:`y=1` dentoes that the 
point belongs to the positive class, and :math:`y=0` means that it belongs to 
the negative class.
In logistic regression, we model the probability of the positive class viar the logistic (or sigmoid) function:

.. math::

    \pi(\x)=P(y=1|\x)=\frac{1}{1+\exp\{-(b+\w^T\x)\}}

    P(y=0|\x)=1-P(y=1|\x)=1-\pi(\x)

Given input :math:`\x`, true response :math:`y`, and predicted response 
:math:`o`, recall that the *cross-entropy error* is defined as

.. math::

    \cl{E}_\x=-(y\cd\ln(o)+(1-y)\cd\ln(1-o))

With sigmoid activation, the output of the neural network is given as

.. math::

    o=f(net_o)=\rm{sigmoid}(b+\w^T\x)=\frac{1}{1+\exp\{-(b+\w^T\x)\}=\pi(\x)

which is the same as the logstic regression model.

**Multiclass Logistic Regression**

For the general classification problem with :math:`K` classes 
:math:`\{c_1,c_2,\cds,c_K\}`, the true response :math:`y` is encoded as a 
one-hot vector.
Thus, class :math:`c_i` is encoded as :math:`\e_1=(1,0,\cds,0)^T`, and so on,
with :math:`\e_1\in\{0,1\}^K` for :math:`i=1,2,\cds,K`.
Thus, we encode :math:`y` as a multivariate vector :math:`\y\in\{\e_1,\e_2,\cds,\e_K\}`.
Recall that in multiclass logistic regression the task is to estimate the per
class bias :math:`b_i` and weight vector :math:`\w_i\in\R^d`, with the last 
class :math:`c_K` used as the base class with fixed bias :math:`b_K=0` and fixed
weight vector :math:`\w_K=(0,0,\cds,0)^T\in\R^d`.
The probability vector across all :math:`K` classes is modeled via the softmax function:

.. math::

    \pi_i(\x)=\frac{\exp\{b_i+\w_i^T\x\}}{\sum_{j=1}^K\exp\{b_J+\w_J^T\x\}},\ \rm{for\ all\ }i=1,2,\cds,K

Therefore, the neural network can solve the multiclass logistic regression task, 
provided we use a softmax activation at the outputs, and use the :math:`K`-way 
cross-entropy error, defined as

.. math::

    \cl{E}_\x=-(y_1\cd\ln(o_1)+\cds+y_K\cd\ln(o_K))

where :math:`\x` is an input vector, :math:`\o=(o_1,o_2,\cds,o_K)^T` is the 
predicted response vector, and :math:`\y=(y_1,y_2,\cds,y_k)^T` is the true 
response vector.
Note that only one element of :math:`\y` is 1, and the rest are 0, due to the one-hot encoding.

With softmax activation, the output of the neural network is given as

.. math::

    o_i=P(\y=\e_i|\x)=f(net_i|\bs{\rm{net}})=\frac{\exp\{net_i\}}{\sum_{j=1}^p\exp\{net_j\}}=\pi_i(\x)

The only restriction we have to impose on the neural network is that weights on 
edges into the last output neuron should be zero to model the base class weights 
:math:`\w_K`.
However, in practice, we relax this restriction, and just learn a regular weight vector for class :math:`c_K`.

25.2.3 Error Functions
^^^^^^^^^^^^^^^^^^^^^^

**Squared Error:**

    Given an input vector :math:`\x\in\R^d`, the squared error loss function 
    measures the squared deviation between the predicted output vector 
    :math:`\o\in\R^p` and the true response :math:`\y\in\R^p`, defined as 
    follows:

    .. note::

        :math:`\dp\cl{E}_\x=\frac{1}{2}\lv\y-\o\rv^2=\frac{1}{2}\sum_{j=1}^p(y_j-o_j)^2`

    where :math:`\cl{E}_\x` denotes the error on input :math:`\x`.

    .. math::

        \frac{\pd\cl{E}_\x}{\pd o_j}=\frac{1}{2}\cd 2\cd(y_j-o_j)\cd -1=o_j-y_j

        \frac{\pd\cl{E}_\x}{\pd\o}=\o-\y

**Cross-Entropy Error:**

    For classification tasks, with :math:`K` classes 
    :math:`\{c_1,c_2,\cds,c_K\}`, we usually set the number of output neurons
    :math:`p=K`, with one output neuron per class.
    Furthermore, each of the classes is coded as a one-hot vector, with class
    :math:`c_i` encoded as the :math:`i`\ th standard basis vector
    :math:`\e_i=(e_{i1},e_{i2},\cds,e_{iK})^T\in\{0,1\}^K`, with 
    :math:`e_{ii}=1` and :math:`e_{ij}=0` for all :math:`j\ne i`.
    Thus, given input :math:`\x\in\R^d`, with the true response
    :math:`\y=(y_1,y_2,\cds,y_K)^T`, where :math:`\y\in\{\e_1,\e_2,\cds,\e_K\}`,
    the cross-entropy loss is defined as

    .. note::

        :math:`\dp\cl{E}_\x=-\sum_{i=1}^Ky_i\cd\ln(o_i)=-(y_1\cd\ln(o_1)+\cds+y_K\cd\ln(o_K))`

    .. math::

        \frac{\pd\cl{E}_\x}{\dp o_j}=-\frac{y_j}{o_j}

        \frac{\pd\cl{E}_\x}{\pd\o}=\bigg(\frac{\pd\cl{E}_\x}{\dp o_1},
        \frac{\pd\cl{E}_\x}{\dp o_2},\cds,\frac{\pd\cl{E}_\x}{\dp o_K}\bigg)^T=
        \bigg(-\frac{y_1}{o_1},-\frac{y_2}{o_2},\cds,-\frac{y_K}{o_K}\bigg)^T

**Binary Cross-Entropy Error:**

    For classification tasks with binary classes, it is typical to encode the
    positive class as 1 and the negative class as 0, as opposed to using a one-\
    hot encoding as in the general :math:`K`-class case.
    Given an input :math:`\x\in\R^d`, with true response :math:`y\in\{0,1\}`, there is only one output neuron :math:`o`.
    Therefore, the binary cross-entropy error is defined as

    .. note::

        :math:`\cl{E}_\x=-(y\cd\ln(o)+(1-y)\cd\ln(1-o))`

    .. math::

        \frac{\pd}{\cl{E}_\x}&=\frac{\pd}{\pd o}\{y\cd\ln(o)-(1-y)\cd\ln(1-o)\}

        &=-\bigg(\frac{y}{o}+\frac{1-y}{1-o}\cd-1\bigg)=\frac{-y\cd(1-o)+(1-y)\cd o}{o\cd(1-o)}

        &=\frac{o-y}{o\cd(1-o)}

25.3 Multilayer Perceptron: One Hidden Layer
--------------------------------------------

A multilayer perceptron (MLP) is a neural network that has distinct layers of neurons.
The inputs to the neural network comprise the *input layer*, and the ﬁnal 
outputs from the MLP comprise the *output layer*. 
Any intermediate layer is called a *hidden layer*, and an MLP can have one or many hidden layers. 
Networks with many hidden layers are called *deep neural networks*. 
An MLP is also a feed-forward network.
Typically, MLPs are fully connected between layers.

25.3.1 Feed-forward Phase
^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\D` denote the training dataset, comprising :math:`n` input points 
:math:`\x_i\in\R^d` and corresponding true response vectors :math:`\y_i\in\R^p`.
For each pair :math:`(\x,\y)` in the data, in the feed-forward phase, the point
:math:`\x=(x_1,x_2,\cds,x_d)^T\in\R^d` is supplied as an input to the MLP.

Given the input neuron values, we compute the output value for each hidden neuron :math:`z_k` as follows:

.. math::

    z_k=f(net_k)=f\bigg(b_K+\sum_{i=1}^dw_{ik}\cd x_i\bigg)

where :math:`w_{ik}` denotes the weight between input neuron :math:`x_i` and hidden neuron :math:`z_k`.

.. math::

    o_j=f(net_j)=f\bigg(b_j+\sum_{i=1}^mw_{ij}\cd z_i)

where :math:`w_{ij}` denotes the weight between hidden neuron :math:`z_i` and output neuron :math:`o_j`.

We define the :math:`d\times m` matrix :math:`\W_h` comprising the weights 
between input and hidden layer neurons, and vector :math:`\b_j\in\R^m` 
comprising the bias terms for hidden layer neurons, given as

.. math::

    \W_h=\bp w_{11}&w_{12}&\cds&w_{1m}\\w_{21}&w_{22}&\cds&w_{2m}\\\vds&\vds&
    \dds&\vds\\w_{d1}&w_{d2}&\cds&w_{dm}\ep\quad\b_h=\bp b_1\\b_2\\\vds\\b_m\ep

where :math:`w_{ij}` denotes the weight on the edge between input neuron 
:math:`x_i` and hidden neuron :math:`z_j`, and :math:`b_i` denotes the bias 
weight from :math:`x_0` to :math:`z_i`.

.. note::

    :math:`\bs{\rm{net}}_h=\b_h+\W_h^T\x`

    :math:`\z=f(\bs{\rm{net}}_h=f(\b_h+\w_h^T\x)`

Here, :math:`\bs{\rm{net}}_h=(net_1,\cds,net_m)^T` denotes the net input at each 
hidden neuron, and :math:`\z=(z_1,z_2,\cds,z_m)^T` denotes the vector of hidden
neuron values.

Likewise, let :math:`\W_o\in\R^{m\times p}` denote the weight matrix between the 
hidden and output layers, and let :math:`\b_o\in\R^p` be the bias vector for 
output neurons, given as

.. math::

    \W_o=\bp w_{11}&w_{12}&\cds&w_{1p}\\w_{21}&w_{22}&\cds&w_{2p}\\\vds&\vds&
    \dds&\vds\\w_{m1}&w_{m2}&\cds&w_{mp}\ep\quad\b_h=\bp b_1\\b_2\\\vds\\b_p\ep

.. note::

    :math:`\bs{\rm{net}}_o=\b_o+\W_o^T\z`

    :math:`\o=f(\bs{\rm{net}}_o=f(\b_o+\w_o^T\z)`

.. math::

    \o=f(\b_o+\w_o^T\z)=f(\b_o+\W_o^T\cd f(\b_h+\W_h^T\x))

25.3.2 Backpropagation Phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

