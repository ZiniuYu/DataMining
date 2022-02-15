Chapter 26 Deep Learning
========================

26.1 Recurrent Neural Networks
------------------------------

Multilayer perceptrons are feed-forward networks in which the information flows 
in only one direction, namely from the input layer to the output layer via the 
hidden layers.
In constrast, recurrent neural networks (RNNs) are dynamically driven, with a 
*feedback* loop between two (or more) layers, which makes such networks ideal 
for learning from sequence data.

Let :math:`\cX=\lag\x_1,\x_2,\cds,\x_\tau\rag` denote a sequence of vectors,
where :math:`\x_t\in\R^d` is a :math:`d`-dimensional vector 
:math:`(t=1,2\,\cds,\tau)`.
Thus, :math:`\cX` is an input sequence of length :math:`\tau`, with 
:math:`\x_t` denoting the input at time step :math:`t`.
Let :math:`\cY=\lag\y_1,\y_2,\cds,\y_\tau\rag` denote a sequence of vectors,
with :math:`\y_t\in\R^p` a :math:`p`-dimensional vector.
Here :math:`\cY` is the desired target or response sequence, with 
:math:`\y_t` denoting the response vector at time :math:`t`.
Finally, let :math:`\cl{O}=\lag\o_1,\o_2,\cds,\o_\tau\rag` denote the predicted
or output sequence from the RNN.
Here :math:`\o_t\in\R^p` is also a :math:`p`-dimensional vector to match the corresponding true response :math:`\y_t`.
The task of an RNN is to learn a function that predicts the target sequence
:math:`\cY` given the input sequence :math:`\cX`.
That is, the predicted output :math:`\o_t` on input :math:`\x_t` should be 
similar or close to the target response :math:`\y_t`, for each time point 
:math:`t`.

To learn dependencies between elements of the input sequence, an RNN maintains a
sequence of :math:`m`-dimensional hidden state vectors :math:`\h_t\in\R^m`, 
where :math:`\h_t` captures the essential features of the input sequences up to
time :math:`t`.
The hidden vector :math:`\h_t` at time :math:`t` depends on the input vector 
:math:`\x_t` at time :math:`t` and the previous hidden state vector 
:math:`\h_{t-1}` from time :math:`t-1`, and it is computed as follows:

.. note::

    :math:`\h_t=f^h(\W_i^T\x_t+\W_h^T\h_{t-1}+\b_h)`

Here, :math:`f^h` is the hidden state activation function, typically tanh or ReLU.
Also, we need an initial hidden state vector :math:`\h_0` that serves as the prior state to compute :math:`\h_1`.
This is uaually set to the zero vector, or seeded from a prior RNN prediction step.
The matrix :math:`\W_i\in\R^{d\times m}` specifies the weights between the input vectors and the hidden state vectors.
The matrix :math:`\W_h\in\R^{m\times m}` specifies the weight matrix between the 
hidden state vectors at time :math:`t-1` and :math:`t`, with :math:`\b_h\in\R^m`
specifying the bias terms associated with the hidden states.
Note that we need only one bias vector :math:`\b_h` associated with the hidden 
state neurons; we do not need a separate bias vector between the input and 
hidden neurons.

Given the hidden state vector at time :math:`t`, the output vector :math:`\o_t` 
at time :math:`t` is computed as follows:

.. note::

    :math:`\o_y=f^o(\W_o^T\h_t+\b_o)`

Here, :math:`\W_o\in\R^{m\times \p}` specifies the weights between the hidden 
state and output vectors, with bias vector :math:`\b_o`.
The output activation function :math:`f^o` typically uses linear or identity 
activation, or a softmax activation for one-hot encoded categorical output 
values.

It is important to note that all the weight matrices and bias vectors are *independent* of the time :math:`t`.
For example, for the hidden layer, the same weight matrix :math:`\W_h` and bias 
vector :math:`\b_h` is used and updated while training the model, over all time
steps :math:`t`.
This is an exmaple of *parameter sharing* or *weight tying* between different layers or components of a neural network.
Likewise, the input weight matrix :math:`\W_i`, the output weight matrix 
:math:`\W_o` and the bias vector :math:`\b_o` are all shared across time.
This greatly reduces the number of parameters that need to be learned by the 
RNN, but it also relies on the assumption that all relevant sequential features 
can be captured by the shared parameters.

The training data for the RNN is given as :math:`\D=\{\cX_i,\cY_y\}_{i=1}^n`, 
comprising :math:`n` input sequences :math:`\cX_i` and the corresponding target 
response sequences :math:`\cY_i`, with sequence length :math:`\tau_i`.
Given each pair :math:`(\cX,\cY)\in\D`, with 
:math:`\cX=\lag\x_1,\x_2,\cds,\x_\tau\rag` and 
:math:`\cY=\lag\y_1,\y_2,\cds,\y_\tau\rag`, the RNN has to update the model 
parameters :math:`\W_i,\W_h,\b_h,\W_o,\b_o` for the input, hidden and output
layers, to learn the corresponding output sequence 
:math:`\cl{O}=\lag\o_1,\o_2,\cds,\o_\tau\rag`.
For training the network, we compute the error or *loss* between the predicted and response vectors over all time steps.
The squared error loss is given as

.. math::

    \cE_\cX=\sum_{t=1}^\tau\cE_{\x_t}=\frac{1}{2}\cd\sum_{t=1}^\tau\lv\y_t-\o_t\rv^2

If we use a softmax activation at the output layer, then we use the cross-entropy loss, given as

.. math::

    \cE_\cX=\sum_{t=1}^\tau\cE_{\x_t}=-\sum_{t=1}^\tau\sum_{i=1}^py_{ti}\cd\ln(o_{ti})

where :math:`\y_t=(y_{t1},y_{t2},\cds,y_{tp})^T\in\R^p` and :math:`\o_t=(o_{t1},o_{t2},\cds,o_{tp})^T\in\R^p`.
On training input of length :math:`\tau` we first unfold the RNN for 
:math:`\tau` steps, following which the parameters can be learned via the
standard feed-forward and backpropagation steps, keeping in mind the connections
between the layers.

26.1.1 Feed-forward in Time
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The feed-forward process starts at time :math:`t=0`, taking as input the initial 
hidden state vector :math:`\h_0`, which us usually set to :math:`\0` or it can
be user-specified, say from a previous prediction step.

.. math::

    \o_t&=f^o(\W_o^T\h_t+\b_o)

    &=f^o(\W_o^Tf^h(\W_i^T\x_t+\W_h^T\h_{t-1}+\b_h)+\b_o)

    &=\vds

    &=f^o(\W_o^Tf^h(\W_i^T\x_t+\W_h^Tf^h(\cds f^h(\W_i^T\x_1+\W_h^T\h_0+\b_h)+\cds)+\b_h)+\b_o)

We can observe that the RNN implicitly makes a prediction for every prefix of 
the input sequence, since :math:`\o_t` depends on all the previous input vectors
:math:`\x_1,\x_2,\cds,\x_t` but not on any future inputs 
:math:`\x_{t+1},\cds,\x_\tau`.

26.1.2 backpropagation in Time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the backpropagation step it is easier to view the RNN in terms of the 
distinct layers based on the dependencies, as opposed to unfolding in time.

Let :math:`\cE_{\x_t}` denote the loss on input vector :math:`\x_t` from the 
input sequence :math:`\cX=\lag\x_1,\x_2,\cds,\x_\tau\rag`.
The unfolded feed-forward RNN for :math:`\cX` has :math:`l=\tau+1` layers.
Define :math:`\bs\delta_t^o` as the net gradient vector for the output vector 
:math:`\o_t`, i.e., the derivative of the error function :math:`\cE_{\x_t}` with 
respect to the net value at each neuron in :math:`\o_t`, given as

.. math::

    \bs\delta_t^o=\bigg(\frac{\pd\cE_{\x_t}}{\pd net_{t1}^o},
    \frac{\pd\cE_{\x_t}}{\pd net_{t2}^o},\cds,
    \frac{\pd\cE_{\x_t}}{\pd net_{tp}^o}\bigg)^T

where :math:`\o_t=(o_{t1},o_{t2},\cds,o_{tp})^T\in\R^p` is the :math:`p`-\
dimensional output vector at time :math:`t`, and :math:`net_{ti}^o` is the net
value at output neuron :math:`o_{ti}` at time :math:`t`.
Likewise, let :math:`\bs\delta_t^h` denote the net gradient vector for the 
hidden state neurons :math:`\h_t` at time :math:`t`

.. math::

    \bs\delta_t^h=\bigg(\frac{\pd\cE_{\x_t}}{\pd net_{t1}^h},
    \frac{\pd\cE_{\x_t}}{\pd net_{t2}^h},\cds,
    \frac{\pd\cE_{\x_t}}{\pd net_{tm}^h}\bigg)^T

where :math:`\h_t=(h_{t1},h_{t2},\cds,h_{tm})^T\in\R^m` is the :math:`m`-\
dimensional hidden state vector at time :math:`t`, and :math:`net_{ti}^h` is the
net value at hidden neuron :math:`h_{ti}` at time :math:`t`.
Let :math:`f^h` and :math:`f^o` denote the activation functions for the hidden
state and output neurons, and let :math:`\pd\f_t^h` and :math:`\pd\f_t^o` denote
the vector of the derivatives of the activation function with respect to the net
signal for the hidden and output neurons at time :math:`t`, given as

.. math::

    \pd\f_t^h\bigg(\frac{\pd f^h(net_{t1}^h)}{\pd net_{t1}^h}, 
    \frac{\pd f^h(net_{t2}^h)}{\pd net_{t2}^h},\cds,
    \frac{\pd f^h(net_{tm}^h)}{\pd net_{tm}^h}\bigg)^T

    \pd\f_t^o\bigg(\frac{\pd f^o(net_{t1}^o)}{\pd net_{t1}^o}, 
    \frac{\pd f^o(net_{t2}^o)}{\pd net_{t2}^o},\cds,
    \frac{\pd f^o(net_{tp}^o)}{\pd net_{tp}^o}\bigg)^T

Finally, let :math:`\pd\bs\cE_{\x_t}` denote the vector of partial derivatives 
of the error function with respect to :math:`\o_t`:

.. math::

    \pd\bs\cE_{\x_t}=\bigg(\frac{\pd\cE_{\x_t}}{\pd o_{t1}},\frac{\pd\cE_{\x_t}}
    {\pd o_{t2}},\cds,\frac{\pd\cE_{\x_t}}{\pd o_{tp}}\bigg)^T

**Computing Net Gradients**

The net gradient vector at the output :math:`\o_t` can be computed as follows:

.. note::

    :math:`\bs\delta_t^o=\pd\f_t^o\od\pd\bs\cE_{\x_t}`

For example, if :math:`\cE_{\x_t}` is the squared error function, and the output 
layer uses the identity funciton, then we have

.. math::

    \bs\delta_t^o=\1\od(\o_t-\y_t)

On the other hand, the net gradients at each of the hidden layers need to 
account for the incoming net gradients from :math:`\o_t` and from 
:math:`\h_{t+1}`.
The net gradient vector for :math:`\h_t(\rm{for\ }t=1,2,\cds,\tau-1)` is given as

.. note::

    :math:`\bs\delta_t^h=\pd\f_t^h\od((\W_o\cd\bs\delta_t^o)+(\W_h\cd\bs\delta_{t+1}^h))`

Note that for :math:`\h_\tau`, it depends only on :math:`\o_\tau`, therefore

.. math::

    \bs\delta_\tau^h=\pd\f_\tau^h\od(\W_o\cd\bs\delta_\tau^o)

For the tanh activation, which is commonly used in RNNs, the derivative of the
activation function with respect to the net values at :math:`\h_t` is given as

.. math::

    \pd\f_t^h=(\1-\h_t\od\h_t)

Finally, note that the net gradients do not have to be computed for :math:`\h_0`
or for any of the input neurons :math:`\x_t`, since these are leaf nodes in the
backpropagation graph, and thus do not backpropagate the gradients beyond those
neurons.

**Stochastic Gradient Descent**

The net gradients for the output :math:`\bs\delta_t^o` and hidden 
:math:`\bs\delta_t^h` at time :math:`t` can be used to compute the gradients for 
the weight matrices and bias vectors at each time point.
However, since an RNN uses parameter sharing across time, the gradients are 
obtained by summing up all of the contributions from each time step :math:`t`.
Define :math:`\nabla_{\w_o}^t` and :math:`\nabla_{\b_o}^t` as the gradients of
the weights and biases between the hidden neurons :math:`\h_t` and output 
neurons :math:`\o_t` for time :math:`t`.
Using the backpropagation equations, for deep multilayer perceptrons, these 
gradients are computed as follows:

.. math::

    \nabla_{\b_o}=\sum_{t=1}^\tau\nabla_{\b_o}^t=\sum_{t=1}^\tau\bs\delta_t^o
    \quad\quad\nabla_{\w_o}=\sum_{t=1}^\tau\nabla_{\w_o}^t=\sum_{t=1}^\tau\h_t
    \cd(\bs\delta_t^o)^T

Likewise, the gradients of the other shared parameters between hidden layers
:math:`\h_{t-1}` and :math:`\h_t`, and between the input layer :math:`\x_t` and
hidden layer :math:`\h_t`, are obtained as follows:

.. math::

    \nabla_{\b_h}&=\sum_{t=1}^\tau\nabla_{\b_h}^t=\sum_{t=1}^\tau\bs\delta_t^h
    \quad\quad\nabla_{\w_h}=\sum_{t=1}^\tau\nabla_{\W_h}^t=\sum_{t=1}^\tau
    \h_{t-1}\cd(\bs\delta_t^h)^T

    \nabla_{\w_i}&=\sum_{t=1}^\tau\nabla_{\w_i}^t=\sum_{t=1}^\tau\x_t\cd(\bs
    \delta_t^h)^T

.. note::

    :math:`\W_i=\W_i-\eta\cd\nabla_{\w_i}\quad\W_h=\W_h-\eta\cd\nabla_{\w_h}\quad\b_h=\b_h-\eta\cd\nabla_{\b_h}`

    :math:`\W_o=\W_o-\eta\cd\nabla_{\w_o}\quad\b_o=\b_o-\eta\cd\nabla_{\b_o}`

26.1.3 Training RNNs
^^^^^^^^^^^^^^^^^^^^

.. image:: ../_static/Algo26.1.png

Note that Line 15 shows the case where the output layer neurons are independent;
if they are not independent we can replace it by 
:math:`\pd\bs{\rm{F}}^o\cd\pd\bs\cE_{\x_t}`.

In practice, RNNs are trained using subsets or *minibatches* of input sequences instead of single sequences.
This helps to speed up the computation and convergence of gradient descent, 
since minibatches provide better estimates of the bias and weight gradients and
allow the use of vectorized operations.

26.1.4 Bidirectional RNNs
^^^^^^^^^^^^^^^^^^^^^^^^^

A bidirectional RNN (BRNN) extends the RNN model to also include information from the future.
In particular, a BRNN maintains a backward hidden state vector 
:math:`\b_t\in\R^m` that depends on the next backward hidden state 
:math:`\b_{t+1}` and the current input :math:`\x_t`.
The output at time :math:`t` is a function of both :math:`\h_t` and :math:`\b_t`.

.. note::

    :math:`\h_t=f^h(\W_{ih}^T\x_t+\W_h^T\h_{t-1}+\b_h)`

    :math:`\b_t=f^b(\W_{ib}^T\x_t+\W_b^T\b_{t+1}+\b_b)`

Also, a BRNN needs two initial state vectors :math:`\h_0` and 
:math:`\b_{\tau+1}` to compute :math:`\b_1` and :math:`\b_\tau`, respectively.
These are usually set to :math:`\0\in\R^m`.
The forward and backward hidden states are computed independently, with the
forward hidden states omputed by considering the input sequence in the forward
direction, and with the backward hidden states computed by considering the 
sequence in reverse order.
The output at time :math:`t` is computed only when both :math:`\h_t` and :math:`\b_t` are available, and is given as

.. math::

    \o_t=f^o(\W_{ho}^T\h_t+\W_{bo}^T\b_t+\b_o)

26.2 Gated RNNs: Long Short-Term Memory Networks
------------------------------------------------

One of the problems in training RNNs is their susceptibility to either the 
*vanishing gradient* or *exploding gradient* problem.
For example, consider the task of computing the net gradient vector 
:math:`\bs\delta_t^h` for the hidden layer at time :math:`t`, given as

.. math::

    \bs\delta_t^h=\pd\f_t^h\od((\W_o\cd\bs\delta_t^o)+(\W_h\cd\bs\delta_{t+1}^h))

Assume for simplicity that we use a linear activation function, i.e.,
:math:`\pd\f_t^h=\1`, and let us ignore the net gradient vector for the output
layer, focusing only on the dependence on the hidden layers.
Then for an input sequence of length :math:`\tau`, we have

.. math::

    \bs\delta_t^h=\W_h\cd\bs\delta_{t+1}^h=\W_h(\W_h\cd\bs\delta_{t+2}^h)=
    \W_h^2\cd\bs\delta_{t+2}^h=\cds=\W_h^{\tau-t}\cd\bs\delta_\tau^h

We can observe that the net gradient from time :math:`\tau` affects the net
gradient vector at time :math:`t` as a function of :math:`\W_h^{\tau-t}`, i.e.,
as powers of the hidden weight matrix :math:`\W_h`.
Let the *spectral radius* of :math:`\W_h`, defined as the absolute value of its 
largest eigenvalue, be given as :math:`|\ld_1|`.
It turns out that if :math:`|\ld_1|<1`, then :math:`\lv\W_h^k\rv\ra 0` as 
:math:`k\ra\infty`, that is, the gradients vanish as we train on long sequences.
On the other hand, if :math:`|\ld_1|>1`, the nat least one element of 
:math:`\W_h^k` becomes unbounded and thus :math:`\lv\W_h^k\rv\ra\infty` as
:math:`k\ra\infty`, that is, the gradients explode as we train on long 
sequences.
Therefore, for the error to neither vanish nor explode, the spectral radius of 
:math:`\W_h` should remian 1 or very close to it.

Long short-term memory (LSTM) networks alleviate the vanishing gradients problem
by using *gate neurons* to control access to the hidden states.
Consider the :math:`m`-dimensional hidden state vector :math:`\h_t\in\R^m` at time :math:`t`.
In a regular RNN, we update the hidden state as follows:

.. math::

    \h_t=f^h(\W_i^T\x_t+\W_h^T\h_{t-1}+\b_h)

Let :math:`\g\in\{0,1\}^m` be a binary vector.
If we take the element-wise product of :math:`\g` and :math:`\h_t`, namely, 
:math:`\g\od\h_t`, then elements of :math:`\g` act as gates that either allow 
the corresponding element of :math:`\h_t` to be retained or set to zero.
The vector :math:`\g` thus acts as logical gate that allows selected elements of 
:math:`\h_t` to be remembered or fogotten.
However, for backpropagation we need *differentiable gates*, for which we use 
sigmoid activation on the gate neurons so that their value lies in the range 
:math:`[0,1]`.
Like a logical gate, such neurons allow the inputs to be completely remembered
if the value is 1, or forgotten if the value is 0.
In addition, they allow a weighted memory, allowing partial remembrance of the 
elements of :math:`\h_t`, for values between 0 and 1.

26.2.1 Forget Gate
^^^^^^^^^^^^^^^^^^

We consider an RNN with a *forget gate*.
Let :math:`\h_t\in\R^m` be the hidden state vector, and let :math:`\bs\phi_t\in\R^m` be a forget gate vector.
Both these vectors have the same number of neurons, :math:`m`.

In a regular RNN, assuming tanh activation, the hidden state vector is updated unconditionally, as follows:

.. math::

    \h_t=\tanh(\W_i^T\x_t+\W_h^T\h_{t-1}+\b_h)

Instead of directly updating :math:`\h_t`, we will employ the forget gate 
neurons to control how much of the prvious hidden state vector to forget when 
computing its new value, and also to control how to update it in light of the
new input :math:`\x_t`.

Given input :math:`\x_t` and previous hidden state :math:`\h_{t-1}`, we first 
compute a candidate update vector :math:`\u_t`, as follows:

.. note::

    :math:`\u_t=\tanh(\W_u^T\x_t+\W_{hu}^T\h_{t-1}+\b_u)`

The candidate update vector :math:`\u_t` is essentially the unmodified hidden state vector, as in a regular RNN.

Using the forget gate, we can compute the new hidden state vector as follows:

.. note::

    :math:`\h_t=\bs\phi_t\od\h_{t-1}+(1-\bs\phi_t)\od\u_t`

We can see that the new hidden state vector retains a fraction of the previous 
hidden state values, and a (complementary) fraction of the candidate update 
values.
Observe that if :math:`\bs\phi_t=\0`, i.e., if we want to entirely forget the
previous hidden state, then :math:`\1-\bs\phi_t=\1`, which means that the hidden
state will be updated completely at each time step just like in a regular RNN.
Finally, given the hidden state :math:`\h_t`, we can compute the output vector :math:`\o_t` as follows

.. math::

    \o_t=f^o(\W_o^T\h_t+\b_o)

.. note::

    :math:`\bs\phi_t=\sg(\W_\phi^T\x_t+\W_{h\phi}^T\h_{t-1}+\b_\phi)`

where we use a sigmoid activation function, denoted :math:`\sg`, to ensure that 
all the neuron values are in the range :math:`[0,1]`, denoting the extent to 
which the corresponding previous hidden state values should be forgotten.

A forget gate vector :math:`\bs\phi_t` is a layer that depends on the previous
hidden state layer :math:`\h_{t-1}` and the current input layer :math:`\x_t`;
these connections are fully connected, and are specified by the corresponding 
weight matrices :math:`\W_{h\phi}` and :math:`\W_{\phi}`, and the bias vector 
:math:`\b_\phi`.
On the other hand, the output of the forget gate layer :math:`\bs\phi_t` needs 
to modify the previous hidden state layer :math:`\h_{t-1}`, and therefore, both
:math:`\bs\phi_t` and :math:`\h_{t-1}` feed into what is essentially a new 
*element-wise* product layer.
Finally, the output of this element-wise product layer is used as input to the 
new hidden layer :math:`\h_t` that also takes input from another element-wise 
gate that computes the output from the candidate update vector :math:`\u_t` and
the complemented forget gate, :math:`\1-\bs\phi_t`.
Thus, unlike regular layers that are fully connected and have a weight matrix 
and bias vector between the layers, the connections between :math:`\bs\phi_t`
and :math:`\h_t` via the element-wise layer are all one-to-one, and the weights
are fixed at the value 1 with bias 0.
Likewise the connections between :math:`\u_t` and :math:`\h_t` via the other
element-wise layer are also one-to-one, with weights fixed at 1 and bias at 0.

Computing Net Gradients
^^^^^^^^^^^^^^^^^^^^^^^

An RNN with a forget gate has the following parameters it needs to learn, namely
the weight matrices :math:`\W_u,\W_{hu},\W_\phi,\W_{h\phi},\W_o`, and the bias
vectors :math:`\b_u,\b_\phi,\b_o`.

Let :math:`\bs\delta_t^o, \bs\delta_t^h, \bs\delta_t^\phi, \bs\delta_t^u` denote
the net gradient vectors at the output, hidden, forget gate, and candidate 
update layers, respectively.
During backpropagation, we need to compute the net gradients at each layer.
The net gradients at the outputs are computed by considering the partial 
derivatives of the activation function :math:`\pd\f_t^o` and the error function
:math:`\pd\bs\cE_{\x_t}`:

.. math::

    \bs\delta_t^o=\pd\f_t^o\od\pd\bs\cE_{\x_t}

For the other layers, we can reverse all the arrows to determine the dependencies between the layers.
Therefore, to compute the net gradient for the update layer 
:math:`\bs\delta_t^u`, notice that in backpropagation it has only one incoming 
edge from :math:`\h_t` via the element-wise product 
:math:`(\1-\bs\phi_t)\od\u_t`.
The net gradient :math:`\delta_{ti}^u` at update layer neuron :math:`i` at time :math:`t` is given as

.. math::

    \delta_{ti}^u=\frac{\pd\cE_\x}{\pd net_{ti}^u}=\frac{\pd\cE_\x}
    {\pd net_{ti}h}\cd\frac{\pd net_{ti}^h}{\pd u_{ti}}\cd\frac{\pd u_{ti}}
    {\pd net_{ti}^u}=\delta_{ti}^h\cd(1-\phi_{ti})\cd(1-u_{ti}^2)