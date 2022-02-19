Chapter 27 Regression Evaluation
================================

Given a set of predictor attributes or independent variables 
:math:`X_1,X_2,\cds,X_d`, and given the response attribute :math:`Y`, the goal 
of regression is to learn a :math:`f`, such that

.. math::

    Y=f(X_1,X_2,\cds,X_d)+\ve=f(\X)+\ve

where :math:`\X=(X_1,X_2,\cds,X_d)^T` is the :math:`d`-dimensional multivariate 
random variable comprised of the predictor variables.
Here, the random variable :math:`\ve` denotes the inferent *error* in the 
response that is not explained by the linear model.

When estimating the regression function :math:`f`, we make assumptions about the form of :math:`f`.
Once we have estimated the bias and coefficients, we need to formulate a 
probabilistic model of regression to evaluate the learned model in terms of 
goodness of fit, confidence intervals for the parameters, and to test for the 
regression effects, namely whether :math:`\X` really helps in predicting 
:math:`Y`.
In particular, we assume that even if the value of :math:`\X` has been fixed,
there can still be uncertainty in the response :math:`Y`.
Further, we will assume that the error :math:`\ve` is independent of :math:`\X`
and follows a normal (or Guassian) distribution with mean :math:`\mu=0` and
variance :math:`\sg^2`, that is, we assume that the errors are independent and
identically distributed with zero mean and fixed variance.

The probabilistic regression model comprises two components-the 
*deterministic component* comprising the observed predictor attributes, and the
*random error component* comprising the error term, which is assumed to be 
independent of the predictor attributes.

27.1 Univariate Regression
--------------------------

We assume that the true relationship can be modeled as a linear function

.. math::

    Y=f(X)+\ve=\beta+\omega\cd X+\ve

where :math:`\omega` is the slope of the best fitting line and :math:`\beta` is 
its intercept, and :math:`\ve` is the random error variable that follows a 
normal distribution with mean :math:`\mu=0` and variance :math:`\sg^2`.

**Mean and Variance of Response Variable**

Consider a fixed value :math:`x` for the independent variable :math:`X`.
The expected value of the response variable :math:`Y` given :math:`x` is

.. math::

    E[Y|X=x]=E[\beta+\omega\cd x+\ve]=\beta+\omega\cd x+E[\ve]=\beta+\omega\cd x

The last step follows from our assumption that :math:`E[\ve]=\mu=0`.
Also, since :math:`x` is assumed to be fixed, and :math:`\beta` and 
:math:`\omega` are constants, the expected value 
:math:`E[\beta+\omega\cd x]=\beta+\omega\cd x`.
Next, consider the variance of :math:`Y` given :math:`X=x`, we have

.. math::

    \rm{var}(Y|X=x)=\rm{var}(\beta+\omega\cd x+\ve)=\rm{var}(\beta+\omega\cd x)+\rm{var}(\ve)=0+\sg^2=\sg^2

Here :math:`\rm{var}(\beta+\omega\cd x)=0`, since :math:`\beta,\omega,x` are all constants.
Thus, given :math:`X=x`, the response variable :math:`Y` follows a normal 
distribution with mean :math:`E[Y|X=x]=\beta+\omega\cd x`, and variance 
:math:`\rm{var}(Y|X=x)=\sg^2`

**Estimated Parameters**

The true parameters :math:`\beta,\omega,\sg^2` are all unknown, and have to be 
estimated from the training data :math:`\D` comprising :math:`n` points 
:math:`x_i` and corresponding response values :math:`y_i`, for 
:math:`i=1,2,\cds,n`.
Let :math:`b` and :math:`w` denote the estimated bias and weight terms; we can 
then make predictions for any given value :math:`x_i` as follows:

.. math::

    \hat{y_i}=b+w\cd x_i

The estimated bias :math:`b` and weight :math:`w` are obtained by minimizing the sum of squared errors, given as

.. math::

    SSE=\sum_{i=1}^n(y_i-\hat{y_i})^2=\sum_{i=1}^n(y_i-b-w\cd x_i)^2

with the least squares estimates given as

.. math::

    w=\frac{\sg_{XY}}{\sg_X^2}\quad\quad b=\mu_Y-w\cd\mu_X