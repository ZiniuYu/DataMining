Part 4 Classification
=====================

The classification task is to predict the label or class for a given unlabeled point.
Formally, a classifier is a model or function :math:`M` that predicts the class 
label :math:`\hat{y}` for a given input example :math:`\x`, that is, 
:math:`\hat{y}=M(\x)`, where :math:`\hat{y}\in\{c_1,c_2,\cds,c_k\}` and each
:math:`c_i` is a class label (a categorical attribute value).
Classification is a *supervised learning* approach, since learning the model 
requires a set of points with their correct class labels, which is called a 
*training set*.
After learning the model :math:`M`, we can automatically predict the class for any new point.

.. toctree::
    :maxdepth: 2
 
    chap18
    chap19
    chap20
    chap21
    chap22