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

27.1.1 Estimating Variance (:math:`\sg^2`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

According to our model, the variance in prediction is entirely due to the random error term :math:`\ve`.
We can estimate this variance by considering the predicted value 
:math:`\hat{y_i}` and its deviation from the true response :math:`y_i`, that is, 
by looking at the residual error

.. math::

    \epsilon_i=y_i-\hat{y_i}

One of the properties of the estimated values :math:`b` and :math:`w` is that the sum of residual errors is zero, since

.. math::

    \sum_{i=1}^n\epsilon_i&=\sum_{i=1}^n(y_i-b-w\cd x_i)

    &=\sum_{i=1}^n(y_i-\mu_Y+w\cd\mu_X-w\cd x_i)

    &=\bigg(\sum_{i=1}^ny_i\bigg)-n\cd\mu_Y+w\cd\bigg(n\mu_X-\sum_{i=1}^n x_i\bigg)

    &=n\cd\mu_Y-n\cd\mu_Y+w\cd(n\cd\mu_X-n\cd\mu_X)=0

Thus, the expected value of :math:`\epsilon_i` is zero, since :math:`E[\epsilon_i]=\frac{1}{n}\sum_{i=1}^n\epsilon_i=0`.

The estimated variance :math:`\hat\sg^2` is given as

.. math::

    \hat\sg^2=\rm{var}(\epsilon_i)=\frac{1}{n-2}\cd\sum_{i=1}^n(\epsilon_i-
    E[\epsilon_i])^2=\frac{1}{n-2}\cd\sum_{i=1}^n\epsilon_i^2=\frac{1}{n-2}\cd
    \sum_{i=1}^n(y_i-\hat{y_i})^2

Thus, the estimated variance is

.. note::

    :math:`\dp\hat\sg^2=\frac{SSE}{n-2}`

We divide by :math:`n-2` to get an unbiased estimate, since :math:`n-2` is the 
number of degrees of freedom for estimating SSE.

The squared root of the variance is called the *standard error of regression*

.. note::

    :math:`\dp\hat\sg=\sqrt{\frac{SSE}{n-2}}`

27.1.2 Goodness of Fit
^^^^^^^^^^^^^^^^^^^^^^

The *total scatter*, also called *total sum of squares*, for the dependent variable :math:`Y`, is defined as

.. math::

    TSS=\sum_{i=1}^n(y_i-\mu_Y)^2

The total scatter can be decomposed into two components by adding and subtracting :math:`\hat{y_i}` as follows

.. math::

    TSS&=\sum_{i=1}^n(y_i-\mu_Y)^2=\sum_{i=1}^n(y_i-\hat{y_i}+\hat{y_i}-\mu_Y)^2

    &=\sum_{i=1}^n(y_i-\hat{y_i})^2+\sum_{i=1}^n(\hat{y_i}-\mu_Y)^2+2\sum_{i=1}^n(y_i-\hat{y_i})\cd(\hat{y_i}-\mu_Y)

    &=\sum_{i=1}^n(y_i-\hat{y_i})^2+\sum_{i=1}^n(\hat{y_1}-\mu_Y)^2=SSE+RSS

where we use the fact that :math:`\sum_{i=1}^n(y_i-\hat{y_i})\cd(\hat{y_i}-\mu_Y)=0`, and

.. math::

    RSS=\sum_{i=1}^n(\hat{y_i}-\mu_Y)^2

is a new term called *regression sum of squares* that measures the squared 
deviation of the predictions from the true mean.
TSS can thus be decomposed into two parts: SSE, which is the amount of variation 
not explained by the model, and RSS, which is the amount of variance explained 
by the model.
Therefore, the fraction of the variation left unexplained by the model is given by the ration :math:`\frac{SSE}{TSS}`.
Conversely, the fraction of the variation that is explained by the model called 
the *coefficient of determination* or simply the :math:`R^2` *statistic*, is
given as

.. note::

    :math:`\dp R^2=\frac{TSS-SSE}{TSS}=1-\frac{SSE}{TSS}=\frac{RSS}{TSS}`

The higher the :math:`R^2` statistic the better the estimated model, with :math:`R^2\in[0,1]`.

**Geometry of Goodness of Fit**

Recall that :math:`Y` can be decomposed into two orthogonal parts

.. math::

    Y=\hat{Y}+\bs\epsilon

where :math:`\hat{Y}` is the projection of :math:`Y` onto the subspace spanned by :math:`\{\1,X\}`.
Using the fact that this subspace is the same as that spanned by the orthogonal 
vectors :math:`\{\1,\bar{X}\}`, with :math:`\bar{X}=X-\mu_X\cd\1`, we can 
further decompose :math:`\hat{Y}` as follows

.. math::

    \hat{Y}=\rm{proj}_\1(Y)\cd\1+\rm{proj}_{\bar{X}}(Y)\cd\bar{X}=\mu_Y\cd\1+
    \frac{Y^T\bar{X}}{\bar{X}^T\bar{X}}\cd\bar{X}=\mu_Y\cd\1+w\cd\bar{X}

Likewise, the vector :math:`Y` and :math:`\hat{Y}` can be centered by 
subtracting their projections along the vector :math:`\1`

.. math::

    \bar{Y}=Y-\mu_Y\cd\1\quad\quad\hat{\bar{Y}}=\hat{Y}-\mu_Y\cd\1=w\cd\bar{X}

The centered vectors :math:`\bar{Y},\hat{\bar{Y}},\bar{X}` all lie in the 
:math:`n-1` dimensional subspace orthogonal to the vector :math:`\1`.

In this subspace, the centered vectors :math:`\bar{Y}` and 
:math:`\hat{\bar{Y}}`, and the error vector :math:`\bs\epsilon` form a right 
triangle, since :math:`\hat{\bar{Y}}` is the orthogonal projection of 
:math:`\bar{Y}` onto the vector :math:`\bar{X}`.
Noting that :math:`\bs\epsilon=Y-\hat{Y}=\bar{Y}-\hat{\bar{Y}}`, by the Pythagoras theorem, we have

.. math::

    \lv\bar{Y}\rv^2=\lv\hat{\bar{Y}}\rv^2+\lv\bs\epsilon\rv^2=\lv\hat{\bar{Y}}\rv^2+\lv Y-\hat{Y}\rv^2

This equation is equivalent to the decomposition of the total scatter, TSS, into 
sum of squared erros, SSE, and residual sum of squares, RSS.

.. math::

    TSS&=\sum_{i=1}^n(y_i-\mu_Y)^2=\lv T-\mu_Y\cd\1\rv^2=\lv\bar{Y}\rv^2

    RSS&=\sum_{i=1}^n(\hat{y_i}-\mu_Y)^2=\lv\hat{Y}-\mu_Y\cd\1\rv^2=\lv\hat{\bar{Y}}\rv^2

    SSE&=\lv\bs\epsilon\rv^2=\lv Y-\hat{Y}\rv^2

.. math::

    \lv\bar{Y}\rv^2&=\lv\hat{\bar{Y}}\rv^2+\lv Y-\hat{Y}\rv^2

    \lv Y-\mu_Y\cd\1\rv^2&=\lv\hat{Y}-\mu_Y\cd\1\rv^2+\lv Y-\hat{Y}\rv^2

    TSS&=RSS+SSE

Notice further that since :math:`\bar{Y},\hat{\bar{Y}},\bs\epsilon` form a right 
triangle, the cosine of the angle between :math:`\bar{Y}` and 
:math:`\hat{\bar{Y}}` is given as the ratio of the base to the hypotenuse.
On the other hand, the cosine of the angle is also the correlation between 
:math:`Y` and :math:`\hat{Y}` denoted :math:`\rho_{Y\hat{Y}}`.
Thus, we have:

.. math::

    \rho_{Y\hat{Y}}=\cos\th=\frac{\lv\hat{\bar{Y}}\rv}{\lv\bar{Y}\rv}

We can observe that

.. math::

    \lv\hat{\bar{Y}}\rv=\rho_{Y\hat{Y}}\cd\lv\bar{Y}\rv

Note that, whereas :math:`|\rho_{Y\hat{Y}}|\leq 1`, due to the projection 
operation, the angle between :math:`Y` and :math:`\hat{Y}` is always less than
or equal to :math:`90^\circ`, which means that :math:`\rho_{Y\hat{Y}}\in[0,1]`
for univariate regression.
Thus, the predicted response vector :math:`\hat{\bar{Y}}` is smaller than the 
true response vector :math:`\bar{Y}` by an amount equal to the correlation 
between them.
Furthermore, the coefficient of determination is the same as the squared 
correlation between :math:`Y` and :math:`\hat{Y}`

.. math::

    R^2=\frac{RSS}{TSS}=\frac{\lv\hat{\bar{Y}}\rv^2}{\lv\bar{Y}\rv^2}=\rho^2_{Y\hat{Y}}

27.1.3 Inference about Regression Coefficient and Bias Term
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The estimated values of the bias and regression coefficient, :math:`b` and 
:math:`w`, are only point estimates for the true parameters :math:`\beta` and 
:math:`\omega`.
To obtain confidence intervals for these parameters, we treat each :math:`y_i` 
as a random variable for the response given the corresponding fixed value 
:math:`x_i`.
These random variables are all independent and identically distributed as 
:math:`Y`, with expected value :math:`\beta+\omega\cd x_i` and variance 
:math:`\sg^2`.
On the other hand, the :math:`x_i` values are fixed *a priori* and therefore 
:math:`\mu_X` and :math:`\sg_X^2` are also fixed values.

We can now treat :math:`b` and :math:`w` as random variables, with

.. math::

    b&=\mu_Y-w\cd\mu_X

    w&=\frac{\sum_{i=1}^n(x_i-\mu_X)(y_i-\mu_Y)}{\sum_{i=1}^n(x_i-\mu_X)^2}=
    \frac{1}{s_X}\sum_{i=1}^n(x_i-\mu_X)\cd y_i=\sum_{i=1}^nc_i\cd y_i

where :math:`c_i` is a constant, given as

.. math::

    c_i=\frac{x_i-\mu_X}{s_X}

and :math:`s_X=\sum_{i=1}^n(x_i-\mu_X)^2` is the total scatter for :math:`X`, 
defined as the sum of squared deviations of :math:`x_i` from its mean 
:math:`\mu_X`.
We also use the fact that

.. math::

    \sum_{i=1}^n(x_i-\mu_X)\cd\mu_Y=\mu_Y\cd\sum_{i=1}^n(x_i-\mu_X)=0

Note that

.. math::

    \sum_{i=1}^nc_i=\frac{1}{s_X}\sum_{i=1}^n(x_i-\mu_X)=0

**Mean and Variance of Regression Coefficient**