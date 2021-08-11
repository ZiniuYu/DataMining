Chapter 5 Kernel Methods
========================

Given a data instance :math:`\x`, we need to find a mapping :math:`\phi`, so 
that :math:`\phi(\x)` is the vector representation of :math:`\x`.
Even when the input data is a numeric data matrix, if we wish to discover 
nonlinear relationships among the attributes, then a nonlinear mapping 
:math:`\phi` may be used, so that :math:`\phi(\x)` represents a vector in the 
corresponding high-dimensional space comprising nonlinear attributes.
We use the *input space* to reefer to the data space for the input data 
:math:`\x` and *feature space* to refer to the space of mapped vectors 
:math:`\phi(\x)`.

Kernel methods avoid explicitly transforming each point :math:`\x` in the input
space into the mapped point :math:`\phi(\x)` in the feature space.
Instead, the input objects are represented via their :math:`n\times n` pairwise similarity values.
The similarity function, called a *kernel*, is chosen so that it represents a 
dot product in some high-dimensional feature space, yet it can be computed 
without directly constructing :math:`\phi(\x)`.
Let :math:`\cl{I}` denote the input space, and let :math:`\D\subset\cl{I}` be a 
dataset comprising :math:`n` objects :math:`\x_i\ (i=1,2,\cds,n)` in the input 
space.
We can represent the pairwise similarity values between points in :math:`\D` via the :math:`n\times n` *kernel matrix*

.. math::

    \K=\bp K(\x_1,\x_1)&K(\x_1,\x_2)&\cds&K(\x_1,\x_n)\\
    K(\x_2,\x_1)&K(\x_2,\x_2)&\cds&K(\x_2,\x_n)\\\vds&\vds&\dds&\vds\\
    K(\x_m,\x_1)&K(\x_n,\x_2)&\cds&K(\x_n,\x_n) \ep

where :math:`K:\cl{I}\times\cl{I}\ra\R` is a *kernel function* on any two points in input space.
For any :math:`\x_i,\x_j\in\cl{I}`, the kernel function should satisfy the condition

.. note::

    :math:`K(\x_i,\x_j)=\phi(\x_i)^T\phi(\x_j)`

Intuitively, this means that we should be able to compute the value of the dot 
product using the original input representation :math:`\x`, without having 
recourse to the mapping :math:`\phi(\x)`.

many data mining methods can be *kernelized*, that is, instead of mapping the
input points into feature space, the data can be represented via the 
:math:`n\times n` kernel matrix :math:`\K`, and all relevant analysis can be
performed over :math:`\K`.
This is usually done via the so-called *kernel trick*, that is, show that the
analysis task requires only dot products :math:`\phi(\x_i)^T\phi(\x_j)` that can
be computed efficiently in input space.
Once the kernel matrix has been computed, we no longer even need the input 
points :math:`\x_i`, as all operations involving only dot products inthe feature
space can be performed over the :math:`n\times n` kernel matrix :math:`\K`.