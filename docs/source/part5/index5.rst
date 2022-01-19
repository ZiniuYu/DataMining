Part 5 Regression
=================

The regression task is to predict the value of a (real-valued) dependent 
variable :math:`Y` given a set of independent variables 
:math:`X_1,X_2,\cds,X_d`.
That is, the goal is to learn a function :math:`f` such that 
:math:`\hat{y}=f(\x)`, where :math:`\hat{y}` is the predicted response value
given the input point :math:`\x`.
In constrast to classification, which predicts a categorical response, in
regression the response variable is real-valued.
Like classification, regression is also a *supervised learning* approach, where
we use a *training* dataset, comprising points :math:`\x_i` alongwith their true
response values :math:`y_i`, to learn the model parameters.
After training, the model can be used to predict the response for new *test* points.

.. toctree::
    :maxdepth: 2
 
    chap23
    chap24
    chap25
    chap26
    chap27