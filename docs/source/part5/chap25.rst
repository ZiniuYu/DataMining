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
The inputs to the neural network comprise the *input layer*, and the ???nal 
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

For a given input pair :math:`(\x,\y)` in the training data, the MLP first 
computes the output vector :math:`\o` via the feed-forward step.
Next, it computes the error in the predicted output *vis-a-vis* the true 
response :math:`\y` using the squared error function

.. math::

    \cl{E}_\x=\frac{1}{2}\lv\y-\o\rv^2=\frac{1}{2}\sum_{j=1}^p(y_j-o_j)^2

The weight update is done via a gradient descent approach to minimize the error.
Let :math:`\nabla_{w_{ij}}` be the gradient of the error function with respect 
to :math:`w_{ij}`, or simply the *weight gradient* at :math:`w_{ij}`.
Given the previous weight estimate :math:`w_{ij}`, a new weight is computed by
taking a small step :math:`\eta` in a direction that is opposite to the weight
gradient at :math:`w_{ij}`

.. math::

    w_{ij}=w_{ij}-\eta\cd\nabla_{w_{ij}}

In a similar manner, the bias term :math:`b_j` is also updated via gradient descent

.. math::

    b_j=b_j-\eta\cd\nabla_{b_j}

where :math:`\nabla_{b_j}` is the gradient of the error function with respect to
:math:`b_j`, which we call the *bias gradient* at :math:`b_j`.

**Updating Parameters Between Hidden and Output Layer**

.. math::

    \nabla_{w_{ij}}&=\frac{\pd\cl{E}_\x}{\pd w_{ij}}=\frac{\pd\cl{E}_\x}
    {\pd net_j}\cd\frac{\pd net_j}{\pd w_{ij}}=\delta_j\cd z_i

    \nabla_{b_j}&=\frac{\pd\cl{E}_\x}{\pd b_j}=\frac{\pd\cl{E}_\x}{\pd net_j}\cd
    \frac{\pd net_j}{\pd b_j}=\delta_j

where we use the symbol :math:`\delta_j` to denote the partial derivative of the
error with respect to net signal at :math:`o_j`, which we also call the 
*net gradient* at :math:`o_j`

.. math::

    \delta_j=\frac{\pd\cl{E}_\x}{\pd net_j}

Futhermore, the partial derivative of :math:`net_j` with respect to :math:`w_{ij}` and :math:`b_j` is given as

.. math::

    \frac{\pd net_j}{\pd w_{ij}}=\frac{\pd}{\pd w_{ij}}\bigg\{b_j+\sum_{k=1}^m
    w_{kj}\cd z_k\bigg\}=z_i\quad\quad\frac{\pd net_j}{\pd b_j}=\frac{\pd}
    {\pd b_j}\bigg\{b_j+\sum_{k=1}^mw_{kj}\cd z_k\bigg\}=1

.. math::

    \delta_j=\frac{\pd\cl{E}_\x}{\pd net_j}=\frac{\pd\cl{E}_\x}{\pd f(net_j)}\cd\frac{\pd f(net_j)}{\pd net_j}

Note that :math:`f(net_j)=o_j`, we have

.. math::

    \frac{\pd\cl{E}_\x}{\pd f(net_j)}=\frac{\pd\cl{E}_\x}{\pd o_j}=\frac{\pd}
    {\pd o_j}\bigg\{\frac{1}{2}\sum_{k=1}^p(y_k-o_k)^2\bigg\}=(o_j-y_j)

.. math::

    \frac{\pd f(net_j)}{\pd net_j}=o_j\cd(1-o_j)

Putting it all together, we get

.. math::

    \delta_j=(o_j-y_j)\cd o_j\cd(1-o_j)

Let :math:`\bs\delta_o=(\delta_1,\delta_2,\cds,\delta_p)^T` denote the vector of 
net gradients at each output neuron, which we call the *net gradient vector* for
the output layer.
We can write :math:`\bs\delta_o` as

.. note::

    :math:`\bs\delta_o=\o\od(\1-\o)\od(\o-\y)`

where :math:`\od` denotes the element-wise product (also called the *Hadamard product*) between the vectors.

Let :math:`\z=(z_1,z_2,\cds,z_m)^T` denote the vector comprising the values of 
all hidden layer neurons (after applying the activation function).
We can compute the gradients :math:`\delta_{w_{ij}}` for all hidden to output 
neuron connections via the outer product of :math:`\z` and :math:`\bs\delta_o`:

.. note::

    :math:`\dp\bs\nabla_{\W_o}=\bp\delta_{w_{11}}&\delta_{w_{12}}&\cds&\delta_{w_{1p}}\\\delta_{w_{21}}&\delta_{w_{22}}&\cds&\delta_{w_{2p}}\\\vds&\vds&\dds&\vds\\\delta_{w_{m1}}&\delta_{w_{m2}}&\cds&\delta_{w_{mp}}\ep=\z\cd\bs\delta_o^T`

The vector of bias gradients is given as:

.. note::

    :math:`\bs\nabla_{\b_o}=(\nabla_{b_1},\nabla_{b_2},\cds,\nabla_{b_p})^T=\bs\delta_o`

.. note::

    :math:`\W_o=\W_o-\eta\cd\bs\nabla_{\w_o}`

    :math:`\b_o=\b_o-\eta\cd\nabla_{\b_o}`

**Updating Parameters Between Input and Hidden Layer**

.. math::

    \nabla_{w_{ij}}&=\frac{\pd\cl{E}_\x}{\pd w_{ij}}=\frac{\pd\cl{E}_\x}
    {\pd net_j}\cd\frac{\pd net_j}{\pd w_{ij}}=\delta_j\cd x_i

    \nabla_{b_j}&=\frac{\pd\cl{E}_\x}{\pd b_j}=\frac{\pd\cl{E}_\x}{\pd net_j}\cd
    \frac{\pd net_j}{\pd b_j}=\delta_j   

which follows from 

.. math::

    \frac{\pd net_j}{\pd w_{ij}}=\frac{\pd}{\pd w_{ij}}\bigg\{b_j+\sum_{k=1}^m
    w_{kj}\cd x_k\bigg\}=x_i\quad\quad\frac{\pd net_j}{\pd b_j}=\frac{\pd}
    {\pd b_j}\bigg\{b_j+\sum_{k=1}^mw_{kj}\cd x_k\bigg\}=1

.. math::

    \delta_j&=\frac{\pd\cl{E}_\x}{\pd net_j}=\sum_{k=1}^p\frac{\pd\cl{E}_\x}
    {\pd net_k}\cd\frac{\pd net_k}{\pd z_j}\cd\frac{\pd z_j}{\pd net_j}=
    \frac{\pd z_j}{\pd net_j}\cd\sum_{k=1}^p\frac{\pd\cl{E}_\x}{\pd net_k}\cd
    \frac{\pd net_k}{\pd z_j}

    &=z_j\cd(1-z_j)\cd\sum_{k=1}^p\delta_k\cd w_{jk}

where :math:`\frac{\pd z_j}{\pd net_j}=z_j\cd(1-z_j)`, since we assume a sigmoid 
activation function for the hidden neurons.

Let :math:`\bs\delta_o=(\delta_1,\delta_2,\cds,\delta_p)^T` denote the vector of 
net gradients at the output nerurons, and 
:math:`\bs\delta_h=(\delta_1,\delta_2,\cds,\delta_m)^T` the net gradients at the
hidden layer neurons.
We can write :math:`\bs\delta_h` compactly as

.. note::

    :math:`\bs\delta_h=\z\od(\1-\z)\od(\W_o\cd\bs\delta_o)`

Furthermore, :math:`\W_o\cd\bs\delta_o\in\R^m` is the vector of weighted gradients at each hidden neuron, since

.. math::

    \W_o\cd\bs\delta_o=\bigg(\sum_{k=1}^p\delta_k\cd w_{1k},\sum_{k=1}^p\delta_k
    \cd w_{2k},\cds,\sum_{k=1}^p\delta_k\cd w_{mk}\bigg)^T

Let :math:`\x=(x_1,x_2,\cds,x_d)^T` denote the input vector, we can compute the
gradients :math:`\nabla_{w_{ij}}` for all input to hidden layer connections via
the outer product:

.. note::

    :math:`\dp\bs\nabla_{\W_h}=\bp\delta_{w_{11}}&\delta_{w_{12}}&\cds&\delta_{w_{1m}}\\\delta_{w_{21}}&\delta_{w_{22}}&\cds&\delta_{w_{2m}}\\\vds&\vds&\dds&\vds\\\delta_{w_{d1}}&\delta_{w_{d2}}&\cds&\delta_{w_{dm}}\ep=\x\cd\bs\delta_h^T`

The vector of bias gradients is given as:

.. note::

    :math:`\nabla_{\b_j}=(\nabla_{b1},\nabla_{b2},\cds,\nabla_{bm})^T=\bs\delta_h`

.. note::

    :math:`\W_h=\W_h-\eta\cd\bs\nabla_{\W_h}`

    :math:`\b_h=\b_h-\eta\cd\nabla_{\b_h}`

25.3.3 MLP Training
^^^^^^^^^^^^^^^^^^^

.. image:: ../_static/Algo25.1.png

The total training time per iteration is :math:`O(dm+mp)`.

25.4 Deep Multilayer Perceptrons
--------------------------------

Consider an MLP with :math:`h` hidden layers.
We assume that the input to the MLP comprises :math:`n` points 
:math:`\x_i\in\R^d` with the corresponding true response vector 
:math:`\y_i\in\R^p`.
We denote the input neurons as layer :math:`l=0`, the first hidden layer as 
:math:`l=1`, the last hidden layer as :math:`l=h`, and the output layer as layer
:math:`l=h+1`.
We use :math:`n_l` to denote the number of neurons in layer :math:`l`.
We have :math:`n_0=d`, and :math:`n_{h+1}=p`.
The vector of neuron values for layer :math:`l` (for :math:`l=0,\cds,h+1`) is denoted as

.. math::

    \z^l=(z_1^l,\cds,z_{n_l}^l)^T

Each layer except the output layer has one extra bias neuron, which is the neuron at index 0.
Thus, the bias neuron for layer :math:`l` is denoted :math:`z_0^l` and its value is fixed at :math:`z_0^l=1`.

The vector of input neuron values is also written as

.. math::

    \x=(x_1,x_2,\cds,x_d)^T=(z_1^0,z_2^0,\cds,z_d^0)^T=\z^0

and the vector of output neuron values is also denoted as

.. math::

    \o=(o_1,o_2,\cds,o_p)^T=(z_1^{h+1},z_2^{h+1},\cds,z_p^{h+1})^T=\z^{h+1}

The weight matrix between layer :math:`l` and layer :math:`l+1` neurons is 
denoted :math:`\W_l\in\R^{n_l\times n_{l+1}}`, and the vector of bias terms from
the bias neuron :math:`z_0^l` to neurons in layer :math:`l+1` is denoted 
:math:`\b_l\in\R^{n_{l+1}}`, for :math:`l=0,1,\cds,h`.

Define :math:`\delta_i^l` as the net gradient, i.e., the partial derivative of 
the error function with respect to the net value at :math:`z_i^l`

.. math::

    \delta_i^l=\frac{\pd\cl{E}_\x}{\pd net_i}

and let :math:`\bs\delta^l` denote the net gradient vector at layer :math:`l`, for :math:`l=1,2,\cds,h+1`

.. math::

    \bs\delta^l=(\delta_1^l,\cds,\delta_{n_l}^l)^t

Let :math:`f^l` denote the activation function for layer :math:`l`, for
:math:`l=0,1,\cds,h+1`, and futher let :math:`\pd\f^l` denote the vector of the
derivatives of the activation function with respect to :math:`net_i` for all
neurons :math:`z_i^l` in layer :math:`l`:

.. math::

    \pd\f^l=\bigg(\frac{\pd f^l(net_1)}{\pd net_1},\cds,\frac{\pd f^l(net_{n_l})}{\pd net_{n_l}}\bigg)^T

Finally, let :math:`\pd\cl{E}_\x` denote the vector of partial derivatives of 
the error function with respect to the values :math:`o_i` for all output 
neurons:

.. math::

    \pd\bs{\cl{E}}_\x=\bigg(\frac{\pd\cl{E}_\x}{\pd o_1},
    \frac{\pd\cl{E}_\x}{\pd o_2},\cds,\frac{\pd\cl{E}_\x}{\pd o_p}\bigg)^T

25.4.1 Feed-forward Phase
^^^^^^^^^^^^^^^^^^^^^^^^^

We assume a fixed activation function :math:`f^l` for all neurons in a given layer.
For a given input pair :math:`(\x,\y)\in\D`, the deep MLP computes the output vector via the feed-forward process:

.. math::

    \o&=f^{h+1}(\b_h+\W_h^T\cd\z^h)

    &=f^{h+1}(\b_h+\W_h^T\cd f^h(\b_{h-1}+\W_{h-1}^T\cd\z^{h-1}))
    
    &=\vds

    &=f^{h+1}(\b_h+\W_h^T\cd f^h(\b_{h-1}+\W_{h-1}^T\cd f^{h-1}(\cds f^2(\b_1+\W_1^T\cd f^1(\b_0+\W_0^T\cd\x)))))

Note that each :math:`f^l` distributes over its argument.
That is

.. math::

    f^l(\b_{l-1}+\W_{l-1}^T\cd\x)=(f^l(net_1),f^l(net_2),\cds,f^l(net_{n_l}))^T

25.4.2 Backpropagation Phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`z_i^l` be a neuron in layer :math:`l`, and :math:`z_j^{l+1}` a neuron in the next layer :math:`l+1`.
Let :math:`w_{ij}^l` be the weight between :math:`z_i^l` and :math:`z_j^{l+1}`,
and let :math:`b_j^l` denote the bias term between :math:`z_0^l` and 
:math:`z_j^{l+1}`.
The weight and bias are updated using the gradient descent approach

.. math::

    w_{ij}^l=w_{ij}^l-\eta\cd\nabla_{w_{ij}^l}\quad\quad b_j^l=b_j^l-\eta\cd\nabla_{b_j^l}

.. math::

    \nabla_{w_{ij}^l}=\frac{\pd\cl{E}_\x}{\pd w_{ij}^l}=
    \frac{\pd\cl{E}_\x}{\pd net_j}\cd\frac{\pd net_j}{\pd w_{ij}^l}=
    \delta_j^{l+1}\cd z_i^l=z_i^l\cd\delta_j^{l+1}

.. math::

    \nabla_{b_j^l}=\frac{\pd\cl{E}_\x}{\pd b_j^l}=\frac{\pd\cl{E}_\x}{\pd net_j}
    \cd\frac{\pd net_j}{\pd b_j^l}=\delta_j^{l+1}

.. math::

    \frac{\pd net_j}{\pd w_{ij}^l}=\frac{\pd}{\pd w_{ij}^l}\bigg\{b_j^l+
    \sum_{k=0}^{n_l}w_{kj}^l\cd z_k^l\bigg\}=z_i^l\quad\quad
    \frac{\pd net_j}{\pd b_j^l}=\frac{\pd}{\pd b_j^l}\bigg\{b_j^l+
    \sum_{k=0}^{n_l}w_{kj}^l\cd z_k^l\bigg\}=1

Given the vector of neuron values at layer :math:`l`, namely 
:math:`\z^l=(z_1^l,\cds,z_{n_l}^l)^T`, we can compute the entire weight gradient
matrix via an outer product operation

.. note::

    :math:`\nabla_{w_l}=\z^l\cd(\bs\delta^{l+1})^T`

and the bias gradient vector as:

.. note::

    :math:`\nabla_{\b_l}=\bs\delta^{l+1}`

with :math:`l=0,1,\cds,h`.

.. note::

    :math:`\W_l=\W_l-\eta\cd\nabla_{\w_l}`

    :math:`\b_l=\b_l-\eta\cd\nabla_{\b_l}`

We observe that to compute the weight and bias gradients for layer :math:`l` we 
need to compute the net gradients :math:`\bs\delta^{l+1}` at layer :math:`l+1`.

25.4.3 Net Gradients at Output Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If all of the output neurons are independent, the net gradient is obtained by
differentiating the error function with respect to the net signal at the output 
neuron.

.. math::

    \delta_j^{h+1}=\frac{\pd\cl{E}_\x}{\pd net_j}=\frac{\pd\cl{E}_\x}
    {\pd f^{h+1}(net_j)}\cd\frac{\pd f^{h+1}(net_j)}{\pd net_j}=
    \frac{\pd\cl{E}_\x}{\pd o_j}\cd\frac{\pd f^{h+1}(net_j)}{\pd net_j}

.. note::

    :math:`\bs\delta^{h+1}=\pd\f^{h+1}\od\pd\cl{\bs{E}}_\x`

If the output neurons are not independent, then we have to modify the 
computation of thg net gradient at each output neuron as follows:

.. math::

    \delta_j^{h+1}=\frac{\pd\cl{E}_\x}{\pd net_j}=\sum_{i=1}^p\frac{\pd
    \cl{E}_\x}{\pd f^{h+1}(net_i)}\cd\frac{\pd f^{h+1}(net_i)}{\pd net_j}

.. note::

    :math:`\bs\delta^{h+1}=\pd\bs{\rm{F}}^{h+1}\cd\pd\cl{\bs{E}}_\x`

where :math:`\pd\bs{\rm{F}}^{h+1}` is the matrix of derivatives of 
:math:`o_i=f^{h+1}(net_i)` with respect to :math:`net_j` for all 
:math:`i,j=1,2,\cds,p`, given as

.. math::

    \pd\bs{\rm{F}}^{h+1}=\bp\frac{\pd o_1}{\pd net_1}&\frac{\pd o_1}{\pd net_2}&
    \cds&\frac{\pd o_1}{\pd net_p}\\\frac{\pd o_2}{\pd net_1}&\frac{\pd o_2}
    {\pd net_2}&\cds&\frac{\pd o_2}{\pd net_p}\\\vds&\vds&\dds&\vds\\
    \frac{\pd o_p}{\pd net_1}&\frac{\pd o_p}{\pd net_2}&\cds&\frac{\pd o_p}
    {\pd net_p}\ep

**Squared Error:**

    The error gradient is given as

    .. math::

        \pd\cl{\bs{E}}_\x=\frac{\pd\cl{\bs{E}}_\x}{\pd\o}=\o-\y

    The net gradient at the output layer is given as 

    .. math::

        \bs\delta^{h+1}=\pd\f^{h+1}\od\pd\cl{\bs{E}}_\x

    Typically, for regression tasks, we use a linear activation at the output neurons.
    In that case, we have :math:`\pd\f^{h+1}=\1`.

**Cross-Entropy Error (binary output, sigmoid activation):**

    The binary cross-entropy error is given as

    .. math::

        \cl{E}_\x=-(y\cd\ln(o)+(1-y)\cd\ln(1-o))

    .. math::

        \pd\cl{\bs{E}}_\x=\frac{\pd\cl{E}_\x}{\pd o}=\frac{o-y}{o\cd(1-o)}

    For sigmoid activaton, we have

    .. math::

        \pd\f^{h+1}=\frac{\pd f(net_o)}{\pd net_o}=o\cd(1-o)

    Therefore, the net gradient at the output neuron is

    .. math::

        \delta^{h+1}=\pd\cl{\bs{E}}_\x\cd\pd\f^{h+1}=\frac{o-y}{o\cd(1-o)}\cd o(1-o)=o-y

**Cross-Entropy Error (**\ :math:`K` **outputs, softmax activation):**

    The cross-entropy error function is given as

    .. math::

        \cl{E}_\x=-\sum_{i=1}^K y_i\cd\ln(o_i)=-(y_1\cd\ln(o_1)+\cds+y_K\cd\ln(o_K))

    .. math::

        \pd\cl{\bs{E}}_\x=\bigg(\frac{\pd\cl{E}_\x}{\pd o_1},\frac{\pd\cl{E}_\x}
        {\pd o_2},\cds,\frac{\pd\cl{E}_x}{\pd o_K}\bigg)^T=\bigg(
        -\frac{y_1}{o_1},-\frac{y_2}{o_2},\cds,-\frac{y_K}{o_K}\bigg)^t

    Cross-entropy error is typically used with the softmax activation so that we 
    get a (normalized) probability value for each class.
    That is,

    .. math::

        o_j=\rm{softmax}(net_j)=\frac{\exp\{net_j\}}{\sum_{i=1}^K\exp\{net_i\}}

    so that the output neuron values sum to one, :math:`\sum_{j=1}^Ko_j=1`.

    .. math::

        \pd\bs{\rm{F}}^{h+1}=\bp\frac{\pd o_1}{\pd net_1}&\frac{\pd o_1}
        {\pd net_2}&\cds&\frac{\pd o_1}{\pd net_K}\\\frac{\pd o_2}{\pd net_1}&
        \frac{\pd o_2}{\pd net_2}&\cds&\frac{\pd o_2}{\pd net_K}\\\vds&\vds&\dds
        &\vds\\\frac{\pd o_K}{\pd net_1}&\frac{\pd o_K}{\pd net_2}&\cds&
        \frac{\pd o_K}{\pd net_K}\ep
        
        =\bp o_1\cd(1-o_1)&-o_1\cd o_2&\cds&-o_1\cd o_K\\-o_1\cd o_2&o_2
        \cd(1-o_2)&\cds&-o_2\cd o_K\\\vds&\vds&\dds&\vds\\-o_1\cd o_K&-o_2
        \cd o_K&\cds&o_K\cd(1-o_K)\ep

    Therefore, the net gradient vecgtor at the output layer is

    .. math::

        \bs\delta^{h+1}=\pd\bs{\rm{F}}^{h+1}\cd\pd\cl{\bs{E}}_\x=\o-\y

25.4.4 Net Gradients at Hidden Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

    \delta_j^l=\frac{\pd\cl{E}_\x}{\pd net_j}&=\sum_{k=1}^{n_l+1}
    \frac{\pd\cl{E}_\x}{\pd net_k}\cd\frac{\pd net_k}{\pd f^l(net_j)}\cd
    \frac{\pd f^l(net_j)}{\pd net_j}

    &=\frac{\pd f^l(net_j)}{\pd net_j}\cd\sum_{k=1}^{n_l+1}\delta_k^{l+1}\cd w_{jk}^l

.. note::

    :math:`\bs\delta^l=\pd\f^l\od(\W_l\cd\bs\delta^{l+1})`

.. math::

    \pd\f^l=\left\{\begin{array}{lr}\1\quad\quad\quad\quad\quad\;\;\,
    \rm{for\ linear}\\\z^l(\1-\z^l)\quad\quad\,\rm{for\ sigmoid}\\
    (\1-\z^l\od\z^l)\quad\rm{for\ tanh}\end{array}\right.

.. math::

    \bs\delta^h&=\pd\g^h\od(\W_h\cd\bs\delta^{h+1})

    \bs\delta^{h-1}&=\pd\f^{h-1}\od(\W_{h-1}\cd\bs\delta^h)=\pd\f^{h-1}\od
    (\W_{h-1}\cd(\pd\f^h\od(\W_h\cd\bs\delta^{h+1})))

    &\vds

    \bs\delta^1&=\pd\f^1\od(\W_1\cd(\pd\f^2\od(\W_2\cds(\pd\f^h\od(\W_h\cd\bs\delta^{h+1})))))

25.4.5 Training Deep MLPs
^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../_static/Algo25.2.png

In practice, it is commonto update the gradients by considering a fixed sized 
subset of the training points called a *minibatch* instead of using single 
points.
That is, the training data is divided into minibatches using an additional
parameter called *batch size*, and a gradient descent step is performed after
computing the bias and weight gradient from each minibatch.
This helps better estimate the gradients, and also allows vectorized matrix 
operations over the minibatch of points, which can lead to faster convergence 
and substantial speedups in the learning.

One caveat while training very deep MLPs is the problem of vanishing and exploding gradients. 
In the *vanishing gradient* problem, the norm of the net gradient can decay 
exponentially with the distance from the output layer, that is, as we 
backpropagate the gradients from the output layer to the input layer.
In this case the network will learn extremely slowly, if at all, since the 
gradient descent method will make minuscule changes to the weights and biases. 
On the other hand, in the *exploding gradient* problem, the norm of the net 
gradient can grow exponentially with the distance from the output layer.
In this case, the weights and biases will become exponentially large, resulting in a failure to learn. 
The gradient explosion problem can be mitigated to some extent by 
*gradient thresholding*, that is, by resetting the value if itexceeds an upper 
bound. 
The vanishing gradients problem is more dif???cult to address.
Typically sigmoid activations are more susceptible to this problem, and one 
solution is to use a lternative activation functions such as ReLU. 
In general, recurrent neural networks, which are deep neural networks with 
*feedback* connections, are more prone to vanishing and exploding gradients.