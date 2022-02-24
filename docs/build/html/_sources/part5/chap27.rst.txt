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

The expected value of :math:`w` is given as

.. math::

    E[w]&=E\bigg[\sum_{i=1}^nc_iy_i\bigg]=\sum_{i=1}^nc_i\cd E[y_i]=\sum_{i=1}^nc_i(\beta+\omega\cd x_i)

    &=\beta\sum_{i=1}^nc_i+\omega\cd\sum_{i=1}^nc_i\cd x_i=\frac{\omega}{s_X}\cd
    \sum_{i=1}^n(x_i-\mu_X)\cd x_i=\frac{\omega}{s_X}\cd s_X=\omega

which follows from the observation that :math:`\sum_{i=1}^nc_i=0`, and further

.. math::

    s_X=\sum_{i=1}^n(x_i-\mu_X)^2=\bigg(\sum_{i=1}^nx_i^2\bigg)-n\cd\mu_X^2=\sum_{i=1}^n(x_i-\mu_X)\cd x_i

Thus, :math:`w` is an unbiased estimator for the true parameter :math:`\omega`.
Using the fact that the variables :math:`y_i` are independent and identically 
distributed as :math:`Y`, we can compute the variance of :math:`w` as follows

.. math::

    \rm{var}(w)=\rm{var}\bigg(\sum_{i=1}^nc_i\cd y_i\bigg)=\sum_{i=1}^nc_i^2\cd
    \rm{var}(y_i)=\sg^2\cd\sum_{i=1}^nc_i^2=\frac{\sg^2}{s_X}

since :math:`c_i` is a constant, :math:`\rm{var}(y_i)=\sg^2`, and further

.. math::

    \sum_{i=1}^nc_i^2=\frac{1}{s^2_X}\cd\sum_{i=1}^n(x_i-\mu_X)^2=\frac{s_X}{s^2_X}=\frac{1}{s_X}

The standard deviation of :math:`w`, also called the standard error of :math:`w`, is given as

.. note::

    :math:`\dp\rm{se}(w)=\sqrt{\rm{var}(w)}=\frac{\sg}{\sqrt{s_X}}`

**Mean and Variance of Bias Term**

The expected value of :math:`b` is given as

.. math::

    E[b]&=E[\mu_Y-w\cd\mu_X]=E\bigg[\frac{1}{n}\sum_{i=1}^ny_i-w\cd\mu_x\bigg]

    &=\bigg(\frac{1}{n}\cd\sum_{i=1}^nE[y_i]\bigg)-\mu_X\cd E[w]=\bigg(
    \frac{1}{n}\sum_{i=1}^n(\beta+\omega\cd x_i)\bigg)-\omega\cd\mu_X

    &=\beta+\omega\cd\mu_X-\omega\cd\mu_X=\beta

Thus, :math:`b` is an unbiased estimator for the true parameter :math:`beta`.

Using the observation that all :math:`y_i` are independent, the variance of the bias term can be computed as follows

.. math::

    \rm{var}(b)&=\rm{var}(\mu_Y-w\cd\mu_X)

    &=\rm{var}\bigg(\frac{1}{n}\sum_{i=1}^ny_i\bigg)+\rm{var}(\mu_X\cd w)

    &=\frac{1}{n^2}\cd n\sg^2+\mu_X^2\cd\rm{var}(w)=\frac{1}{n}\cd\sg^2+\mu_X^2\cd\frac{\sg^2}{s_X}

    &=\bigg(\frac{1}{n}+\frac{\mu_X^2}{s_X}\bigg)\cd\sg^2

The standard deviation of :math:`b`, also called the standard error of :math:`b`, is given as

.. note::

    :math:`\dp\rm{se}(b)=\sqrt{\rm{var}(b)}=\sg\cd\sqrt{\frac{1}{n}+\frac{\mu_X^2}{s_X}}`

**Covariance of Regression Coefficient and Bias**

.. math::

    \rm{cov}(w,b)&=E[w\cd b]-E[w]\cd E[b]=E[(\mu_Y-w\cd\mu_X)\cd w]-\omega\cd\beta

    &=\mu_Y\cd E[w]-\mu_X\cd E[w^2]-\omega\cd\beta=\mu_Y\cd\omega-\mu_X\cd(\rm{Var}(w)+E[w]^2)-\omega\cd\beta

    &=\mu_Y\cd\omega-\mu_X\cd\bigg(\frac{\sg^2}{s_X}-\omega^2\bigg)-\omega\cd
    \beta=\omega\cd(\mu_Y-\omega\cd\mu_X)-\frac{\mu_X\cd\sg^2}{s_X}-\omega\cd
    \beta

    &=-\frac{\mu_X\cd\sg^2}{s_X}

**Confidence Intervals**

Since the :math:`y_i` variables are all normally distributed, their linear 
combination also follows a normal distribution.
Thus :math:`w` follows a normal distribution with mean :math:`\omega` and variance :math:`\sg^2/s_X`.
Like wise, :math:`b` follows a normal distribution with mean :math:`\beta` and 
variance :math:`(1/n+\mu_X^2/s_X)\cd\sg^2`.

Since the true variance :math:`\sg^2` is unknown, we use the estimated variance
:math:`\hat\sg^2`, to define the standardized variables :math:`Z_w` amd 
:math:`Z_b` as follows

.. note::

    :math:`\dp Z_w=\frac{w-E[w]}{\rm{se}(w)}=\frac{w-\omega}{\frac{\hat\sg}{\sqrt{s_X}}}\quad\quad`
    :math:`\dp Z_b=\frac{b-E[b]}{\rm{se}(b)}=\frac{b-\beta}{\hat\sg\sqrt{(1/n+\mu_X^2/s_X)}}`

These variables follow the Student's :math:`t` distribution with :math:`n-2` degrees of freedom.
Let :math:`T_{n-2}` denote the cumulative :math:`t` distribution with 
:math:`n-2` degrees of freedom, and let :math:`t_{\alpha/2}` denote the critical
value of :math:`T_{n-2}` that encompasses :math:`\alpha/2` of the probability
mass in the right tail.

.. math::

    P(Z\geq t_{\alpha/2})=\frac{\alpha}{2}\rm{\ or\ equivalently\ }T_{n-2}(t_{\alpha/2})=1-\frac{\alpha}{2}

    P(Z\geq-t_{\alpha/2})=1-\frac{\alpha}{2}\rm{\ or\ equivalently\ }T_{n-2}(-t_{\alpha/2})=\frac{\alpha}{2}

Given confidence level :math:`1-\alpha`, i.e., significance level 
:math:`\alpha\in(0,1)`, the :math:`100(1-\alpha)\%` confidence interval for the 
true values, :math:`\omega` and :math:`\beta`, are therefore as follows

.. math::

    P(w-t_{\alpha/2}\cd\rm{se}(w)\leq\omega\leq w+t_{\alpha/2}\cd\rm{se}(w))&=1-\alpha

    P(b-t_{\alpha/2}\cd\rm{se}(b)\leq\beta\leq b+t_{\alpha/2}\cd\rm{se}(b))&=1-\alpha

27.1.4 Hypothesis Testing for Regression Effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the regression model, :math:`Y` depends on :math:`X` through the parameter 
:math:`\omega`, therefore, we can check for the regression effect by assuming
the null hypothesis :math:`H_0` that :math:`\omega=0`, with the alternative
hypothesis :math:`H_a` being :math:`\omega\ne 0`:

.. math::

    H_0:\omega=0\quad\quad H_a:\omega\ne 0

When :math:`\omega=0`, the response :math:`Y` depends only on the bias :math:`\beta` and the random error :math:`\ve`.

Under the null hypothesis we have :math:`E[w]=\omega=0`.
Thus,

.. note::

    :math:`\dp Z_w=\frac{w-E[w]}{\rm{se}(w)}=\frac{w}{\hat\sg/\sqrt{s_X}}`

Given significance level :math:`\alpha`, we reject the null hypothesis if the p-value is below :math:`\alpha`.
In this case, we accept the alternative hypothesis that the estimated value of 
the slope parameter is significantly different from zero.

We can also define the :math:`f`-statistic, which is the ratio of the regression 
sum of squares, RSS, to the estimated variance, given as

.. note::

    :math:`\dp f=\frac{RSS}{\hat\sg^2}=\frac{\sum_{i=1}^n(\hat{y_i}-\mu_Y)^2]{\sum_{i=1}^n(y_i-\hat{y_i}^2/n-2)}`

Under the null hypothesis, one can show that

.. math::

    E[RSS]=\sg^2

Further, it is also true that

.. math::

    E[\hat\sg^2]=\sg^2

Thus, under the null hypothesis the :math:`f`-statistic has a value close to 1, 
which indicates that there is no relationship between the predictor and response 
variables.
On the other hand, if the alternative hypothesis is true, then 
:math:`E[RSS]\geq\sg^2`, resulting in a larger :math:`f` value.
In fact, the :math:`f`-statistic follows a :math:`F`-distribution with 1, 
:math:`(n-2)` degrees of freedom; therefore, we can reject the null hypothesis
that :math:`w=0` if the p-value of :math:`f` is less than the significance level
:math:`\alpha`.

Interestingly the :math:`f`-test is equivalent to the :math:`t`-test since :math:`Z_w^2=f`.

.. math::

    f&=\frac{1}{\hat\sg^2}\cd\sum_{i=1}^n(\hat{y_i}-\mu_Y)^2=\frac{1}{\hat\sg^2}\cd\sum_{i=1}^n(b+w\cd x_i-\mu_Y)^2

    &=\frac{1}{\hat\sg^2}\cd\sum_{i=1}^n(\mu_Y-w\cd\mu_X+w\cd x_i-\mu_Y)^2=
    \frac{1}{\hat\sg^2}\cd\sum_{i=1}^n(w\cd(x_i-\mu_X))^2

    &=\frac{1}{\hat\sg^2}\cd w^2\cd\sum_{i=1}^n(x_i-\mu_X)^2=\frac{w^2\cd s_X}{\hat\sg^2}

    &=\frac{w^2}{\hat\sg^2/s_X}=Z_w^2

**Test for Bias Term**

Note that we can also test if the bias value is statistically significant or not 
by setting up the null hypothesis, :math:`H_0:\beta=0`, versus the alternative 
hypothesis :math:`H_a:\beta\neq 0`.
We then evaluate the :math:`Z_b` statistic under the null hypothesis:

.. note::

    :math:`\dp Z_b=\frac{b-E[b]}{\rm{se}(b)}=\frac{b}{\hat\sg\cd\sqrt{(1/n)+\mu_X^2/s_X)}}`

since, under the null hypothesis :math:`E[b]=\beta=0`.
Using a two-tailed :math:`t`-test with :math:`n-2` degrees of freedom, we can compute the p-value of :math:`Z_b`.
We reject the null hypothesis if this value is smaller than the significance level :math:`\alpha`.

27.1.5 Standardized Residuals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our assumption about the true errors :math:`\ve_i` is that they are normally 
distributed with mean :math:`\mu=0` and fixed variance :math:`\sg^2`.

The mean of :math:`\epsilon_i` is given as

.. math::

    E[\epsilon_i]&=E[y_i-\hat{y_i}]=E[y_i]-E[\hat{y_i}]

    &=\beta+\omega\cd x_i-E[b+w\cd x_i]=\beta+\omega\cd x_i-(\beta+\omega\cd x_i)=0

To compute the variance of :math:`\epsilon_i`, we will express it as a linear 
combination of the :math:`y_i` variables, by noting that

.. math::

    w&=\frac{1}{s_X}\bigg(\sum_{j=1}^nx_iy_i-n\cd\mu_X\cd\mu_Y\bigg)=
    \frac{1}{s_X}\bigg(\sum_{j=1}^nx_jy_j-\sum_{j=1}^n\mu_X\cd y_j\bigg)=
    \sum_{j=1}^n\frac{(x_j-\mu_X)}{s_X}\cd y_j

    b&=\mu_Y-w\cd\mu_X=\bigg(\sum_{j=1}^n\frac{1}{n}\cd y_i\bigg)-w\cd\mu_X

.. math::

    \epsilon_i=y_i-\hat{y_i}&=y_i-b-w\cd x_i=y_i-\sum_{j=1}^n\frac{1}{n}y_i+w\cd\mu_X-w\cd x_i

    &=y_i-\sum_{j=1}^n\frac{1}{n}y_j-(x_i-\mu_X)\cd w

    &=y_i-\sum_{j=1}^n\frac{1}{n}y_j-\sum_{j=1}^n\frac{(x_i-\mu_X)\cd(x_j-\mu_X)}{s_X}\cd y_j

    &=\bigg(1-\frac{1}{n}-\frac{(x_i-\mu_X)^2}{s_X}\bigg)\cd y-i-\sum_{j\neq i}
    \bigg(\frac{1}{n}+\frac{(x_i-\mu_X)\cd(x_j-\mu_X)}{s_X}\bigg)\cd y_j

Define :math:`a_j` as follows:

.. math::

    a_j=\bigg(\frac{1}{n}+\frac{(x_i-\mu_x)\cd(x_j-\mu_X)}{s_X}\bigg)

.. math::

    \rm{var}(\epsilon_i)&=\rm{var}\bigg((1-a_i)\cd y_i-\sum_{j\neq i}a_j\cd y_j\bigg)

    &=(1-a_i)^2\cd\rm{var}(y_i)+\sum_{j\neq i}a_j^2\cd\rm{var}(y_j)

    &=\sg^2\cd(1-2a_i+a_i^2+\sum_{j\neq i}a_j^2)

    &=\sg^2\cd(1-2a_i+\sum_{j=1}^na_j^2)

Consider the term :math:`\sum_{j=1}^na_j^2`, we have

.. math::

    \sum_{j=1}^na_j^2&=\sum_{j=1}^n\bigg(\frac{1}{n}+\frac{(x_i-\mu_X)\cd(x_j-\mu_X)}{s_X}\bigg)^2

    &=\sum_{j=1}^n\bigg(\frac{1}{n^2}-\frac{2\cd(x_i-\mu_X)\cd(x_j-\mu_X)}{n\cd 
    s_X}+\frac{(x_i-\mu_X)^2\cd(x_j-\mu_x)^2}{s_X^2}\bigg)

    &=\frac{1}{n}-\frac{2\cd(x_i-\mu_x)}{n\cd s_X}\sum_{j=1}^n(x_j-\mu_X)+
    \frac{(x_i-\mu_X)^2}{s_X^2}\sum_{j=1}^n(x_j-\mu_X)^2

    &=\frac{1}{n}+\frac{x_i-\mu_X)^2}{s_X}

.. math::

    \rm{var}(\epsilon_i)&=\sg^2\cd\bigg(1-\frac{2}{n}-\frac{2\cd(x_i-\mu_x)^2}
    {s_X}+\frac{1}{n}+\frac{(x_i-\mu_X)^2}{s_X}\bigg)

    &=\sg^2\cd\bigg(1-\frac{1}{n}-\frac{(x_i-\mu_x)^2}{s_x}\bigg)

We can now define the *standardized residual* :math:`\epsilon_i^*` by dividing 
:math:`\epsilon_i` by its standard deviation after replacing :math:`\sg^2` by
its estimated value :math:`\hat\sg^2`.

.. note::

    :math:`\dp\epsilon_i^*=\frac{\epsilon_i}{\sqrt{\rm{var}(\epsilon_i)}}`
    :math:`\dp=\frac{\epsilon_i}{\hat\sg\cd\sqrt{1-\frac{1}{n}-\frac{(x_i-\mu_x)^2}{s_X}}}`

These standardized residuals should follow a standard normal distribution. 
We can thus plot the standardized residuals against the quantiles of a standard 
normal distribution, and check if the normality assumption holds. 
Signiﬁcant deviations would indicate that our model assumptions may not be correct.

27.2 Multiple Regression
------------------------

In multiple regression there are multiple independent attributes 
:math:`X_1,X_2,\cds,X_d` and a single dependent or response attribute :math:`Y`,
and we assume that the true relationship can be modeled as a linear function

.. math::

    Y=\beta+\omega_1\cd X_1+\omega_2\cd X_2+\cds+\omega_d X_d+\ve

where :math:`\beta` is the intercept or bias term and :math:`\omega_i` is the 
regression coefficient for attribute :math:`X_i`.
We assume that :math:`\ve` is a random variable that is normally distributed 
with mean :math:`\mu=0` and variance :math:`\sg^2`.

**Mean and Variance of Response Variable**

Let :math:`\X=(X_1,X_2,\cds,X_d)^T\in\R^d` denote the multivariate random 
variable comprising the independent attributes.
Let :math:`\x=(x_1,x_2,\cds,x_d)^T` be some *fixed* value of :math:`\X`, and let
:math:`\bs\omega=(\omega_1,\omega_2,\cds,\omega_d)^T`.
The expected response value is then given as

.. math::

    E[Y|\X=\x]&=E[\beta+\omega_1\cd x_1+\cds+\omega_d\cd x_d+\ve]=E\bigg[\beta+\sum_{i=1}^d\omega_i\cd x_i]+E[\ve]

    &=\beta+\omega_1\cd x_1+\cds+\omega_d\cd x_d=\beta+\bs\omega^T\bs\x

which follows from the assumption that :math:`E[\ve]=0`.
The variance of the response variable is given as

.. math::

    \rm{var}(Y|\X=\x)=\rm{var}\bigg(\beta+\sum_{i=1}^d\omega_i\cd x_i+\ve\bigg)=
    \rm{var}\bigg(\beta+\sum_{i=1}^d\omega_i\cd x_i\bigg)+\rm{var}(\ve)=0+\sg^2=
    \sg^2

which follows from the assumption that all :math:`x_i` are fixed *a priori*.
Thus, we conclude that :math:`Y` also follows a normal distribution with mean 
:math:`E[Y|\x]=\beta+\sum_{i=1}^d\omega_i\cd x_i=\beta+\bs\omega^T\x` and 
variance :math:`\rm{var}(Y|\x)=\sg^2`.

**Estimated Parameters**

We augment the data matrix by adding a new column :math:`X_0` with all values fixed at 1, that is, :math:`X_0=\1`.
Thus, the augmented data :math:`\td\D\in\R^{n\times(d+1)}` comprises the 
:math:`(d+1)` attributes :math:`X_0,X_1,X_2,\cds,X_d`, and each augmented point 
is given as :math:`\td\x_i=(1,x_{i1},x_{i2},\cds,x_{id})^T`.

Let :math:`b=w_0` denote the estimated bias term, and let :math:`w_i` denote the estimated regression weights.
The augmented vector of estimated weights, including the bias term, is

.. math::

    \td\w=(w_0,w_1,\cds,w_d)^T

We then make predictions for any given point :math:`\x_i` as follows:

.. math::

    \hat{y_i}=b\cd 1+w_1\cd x_{i1}+\cds+w_d\cd x_{id}=\td\w^T\td{\x_i}

Recall that these estimates are obtained by minimizing the sum of squared errors (SSE), given as

.. math::

    SSE=\sum_{i=1}^n(y_i-\hat{y_i})^2=\sum_{i=1}^n\bigg(y_i-b-\sum_{j=1}^dw_\cd x_{ij}\bigg)^2

with the least squares estimate given as

.. math::

    \td\w=(\td\D^T\td\D)\im\td\D^TY

The estimated variance :math:`\hat\sg^2` is then given as

.. note::

    :math:`\dp\hat\sg^2=\frac{SSE}{n-(d+1)}=\frac{1}{n-d-1}\cd\sum_{i=1}^n(y_i-\hat{y_i})^2`

We divide by :math:`n-(d+1)` to get an unbiased estimate, since :math:`n-(d+1)` 
is the number of degrees of freedom for estimating SSE.

**Estimated Variance is Unbiased**

Recall that

.. math::

    \hat{Y}=\td\D\td\w=\td\D(\td\D^T\td\D)\im\td\D^TY=\H Y

where :math:`\H` is the :math:`n\times n` hat matrix (assuming that :math:`(\td\D^T\td\D)\im` exists).
Note that :math:`\H` is an *orthogonal projection matrix*, since it is symmetric 
(:math:`\H^T=\H`) and idempotent (:math:`\H^2=\H`).

.. math::

    \H^T&=(\td\D(\td\D^T\td\D)\im\td\D^T)^T=(\td\D^T)^T((\td\D^T\td\D)^T)\im\td\D^T=\H

    \H^2&=\td\D(\td\D^T\td\D)\im\td\D^T\td\D(\td\D^T\td\D)\im\td\D^T=\td\D(\td\D^T\td\D)\im\td\D^T=\H

Furthermore, the trace of the hat matrix is given as

.. math::

    \rm{tr}(\H)=\rm{tr}(\td\D(\td\D^T\td\D)\im\td\D^T)=\rm{tr}(\td\D^T\td\D(\td\D^T\td\D)\im)=\rm{tr}(\I_{(d+1)})=d+1

Finally, note that the matrix :math:`\I-\H` is also symmetric and idempotent, since

.. math::

    (\I-\H)^T&=\I^T-\H^T=\I-\H

    (\I-\H)^2&=(\I-\H)(\I-\H)=\I-\H-\H+\H^2=\I-\H

Now consider the squared error; we have

.. math::

    SSE&=\lv Y-\hat{Y}\rv^2=\lv Y-\H Y\rv^2=\lv(\I-\H)Y\rv^2

    &=Y^T(\I-\H)(\I-\H)Y=Y^T(\I-\H)Y

However, note that the response vector :math:`Y` is given as

.. math::

    Y=\td\D\td{\bs\omega}+\bs\ve

where :math:`\td{\bs\omega}=(\omega_0,\omega_1,\cds,\omega_d)^T` is the true
(augmented) vector of parameters of the model, and 
:math:`\bs\ve=(\ve_1,\ve_2,\cds,\ve_n)^T` is the true error vector, which is
assumed to be normally distributed with mean :math:`E[\bs\ve]=\0` and with fixed
variance :math:`\ve_i\sg^2` for each point, so that 
:math:`\rm{cov}(\bs\ve)=\sg^2\I`.

.. math::

    SSE&=Y^T(\I-\H)Y=(\td\D\td{\bs\omega}+\bs\ve)^T(\I-\H)(\td\D\td{\bs\omega}+\bs\ve)
    
    &=(\td\D\td{\bs\omega}+\bs\ve)^T((\I-\H)\td\D\td{\bs\omega}+(\I-\H)\bs\ve)

    &=((\I-\H)\bs\ve)^T(\td\D\td{\bs\ve}+\bs\ve)=\bs\ve^T(\I-\H)(\td\D\td{\bs\omega}+\bs\ve)

    &=\bs\ve^T(\I-\H)\td\D\td{\bs\omega}+\bs\ve^T(\I-\H)\bs\ve=\bs\ve^T(\I-\H)\bs\ve

where we use the observation that

.. math::

    (\I-\H)\td\D\td{\bs\omega}=\td\D\td{\bs\omega}-\H\td\D\td{\bs\omega}=
    \td\D\td{\bs\omega}-(\td\D(\td\D^T\td\D)\im\td\D^T)\td\D\td{\bs\omega}=
    \td\D\td{\bs\omega}-\td\D\td{\bs\omega}=\0

.. math::

    E[SSE]&=E[\bs\ve^T(\I-\H)\bs\ve]

    &=E\bigg[\sum_{i=1}^n\ve_i^2-\sum_{i=1}^n\sum_{j=1}^nh_{ij}\ve_i\ve_j\bigg]=
    \sum_{i=1}^nE[\ve_i^2]-\sum_{i=1}^n\sum_{j=1}^nh_{ij}E[\ve_i\ve_j]

    &=\sum_{i=1}^n(1-h_{ii})E[\ve_i^2]

    &=\bigg(n-\sum_{i=1}^nh_{ii}\bigg)\sg^2=(n-\rm{tr}(\H))\sg^2=(n-d-1)\cd\sg^2

It follows that

.. math::

    \hat\sg^2=E\bigg[\frac{SSE}{(n-d-1)}\bigg]=\frac{1}{(n-d-1)}E[SSE]=\frac{1}{(n-d-1)}\cd(n-d-1)\cd\sg^2=\sg^2

27.2.1 Goodness of Fit
^^^^^^^^^^^^^^^^^^^^^^

The decomposition of the total sum of squares, TSS, into the sum of squared 
errors, SSE, and the residual sum of squares, RSS, holds true for multiple 
regression as well:

.. math::

    TSS&=SSE+RSS

    \sum_{i=1}^n(y_i-\mu_Y)^2&=\sum_{i=1}^n(y_i\hat{y_i})^2+\sum_{i=1}^n(\hat{y_i}-\mu_Y)^2

The *coefficient of multiple determinations*, :math:`R^2`, gives the goodness of 
fit, measured as the fraction of the variation explained by the linear model:

.. note::

    :math:`\dp R^2=1-\frac{SSE}{TSS}=\frac{TSS-SSE}{TSS}=\frac{RSS}{TSS}`

One of the potential problems with the :math:`R^2` measure is that it is 
susceptible to increase as the number of attributes increase, even though the 
additional attributes may be uninformative.
To counter this, we can consider the *adjusted coefficient of determination*,
which takes into account the degrees of freedom in both TSS and SSE

.. note::

    :math:`\dp R_a^2=1-\frac{SSE/(n-d-1)}{TSS/(n-1)}=1-\frac{(n-1)\cd SSE}{(n-d-1)\cd TSS}`

We can observe that the adjusted :math:`R_a^2` measure is always less than 
:math:`R^2`, since the ratio :math:`\frac{n-1}{n-d-1}>1`.
If there is too much of a difference between :math:`R^2` and :math:`R_a^2`, it 
might indicate that there are potentially many, possibly irrelevant, attributes 
being used to fit the model.

**Geometry of Goodness of Fit**

.. math::

    \bar{X_i}=X_i-\mu_{X_i}\cd\1\quad\quad\bar{Y}=Y-\mu_Y\cd\1\quad\quad\hat{\bar{Y}}=\hat{Y}-\mu_Y\cd\1

The centered vectors :math:`\bar{Y}` and :math:`\hat{\bar{Y}}`, and the error 
vector :math:`\bs\epsilon` form a right triangle, and thus, by the Pythagoras 
theorem, we have

.. math::

    \lv\bar{Y}\rv^2&=\lv\hat{\bar{Y}}\rv^2=\lv\bs\epsilon\rv^2=\lv\hat{\bar{Y}}\rv^2+\lv Y-\hat{Y}\rv^2

    YSS&=RSS+SSE

The correlation between :math:`Y` and :math:`\hat{Y}` is the cosine of the angle 
between :math:`\bar{Y}` and :math:`\hat{\bar{Y}}`, which is also given as the 
ratio of the base to the hypotenuse

.. math::

    \rho_{Y\hat{Y}}=\cos\th=\frac{\lv\hat{\bar{Y}}\rv}{\lv\bar{Y}\rv}

The coefficient of multiple determination is given as

.. math::

    R^2=\frac{RSS}{TSS}=\frac{\lv\hat{\bar{Y}}\rv^2}{\lv\bar{Y}\rv^2}=\rho_{Y\hat{Y}}^2

27.2.2 Inference about Regression Coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`Y` be the response vector over all observations.
Let :math:`\td\w=(w_0,w_1,w_2,\cds,w_d)^T` be the estimated vector of regression coefficients, computed as

.. math::

    \td\w=(\td\D^T\td\D)\im\td\D^TY

The expected value of :math:`\td\w` is given as follows:

.. math::

    E[\td\w]&=E[(\td\D^T\td\D)\im\td\D^TY]=(\td\D^T\td\D)\im\td\D^T\cd E[Y]

    &=(\td\D^T\td\D)\im\td\D^T\cd E[\td\D\td{\bs\omega}+\bs\ve]=(\td\D^T\td\D)
    \im(\td\D^T\td\D)\td{\bs\omega}=\td{\bs\omega}

Thus, :math:`\td\w` is an unbiased estimator for the true regressions coefficients vector :math:`\td{\bs\omega}`.

.. math::

    \rm{cov}(\td\w)&=\rm{cov}((\td\D^T\td\D)\im\td\D^TY)

    &=\rm{cov}(\A Y)=\A\rm{cov}(Y)\A^T

    &=\A\cd(\sg^2\cd\I)\cd\A^T

    &=(\td\D^T\td\D)\im\td\D^T(\sg^2\cd\I)\td\D(\td\D^T\td\D)\im

    &=\sg^2\cd(\td\D^T\td\D)\im(\td\D^T\td\D)(\td\D^T\td\D)\im

    &=\sg^2(\td\D^T\td\D)\im

Here, we made use of the fact that :math:`\A=(\td\D^T\td\D)\im\td\D^T` is a 
matrix of fixed values, and therefore :math:`\rm{cov}(\A Y)=\A\rm{cov}(Y)\A^T`.
Also, we have :math:`\rm{cov}(Y)=\sg^2\cd\I`, which follows from the fact that
the observed response :math:`y_i`'s are all independent and have the same 
variance :math:`\sg^2`.

Note that :math:`\td\D^T\td\D\in\R^{(d+1)\times(d+1)}` is the uncentered scatter matrix for the augmented data.
Let :math:`\C` denote the inverse of :math:`\td\D^T\td\D`.

.. math::

    (\td\D^T\td\D)\im=\C

Therefore, the covariance matrix for :math:`\td\w` can be written as

.. math::

    \rm{cov}(\td\w)=\sg^2\C

In particular, the diagonal entries :math:`\sg^2\cd c_{ii}` give the variance 
for each of the regression coefficient estimates, and their squared roots 
specify the standard erros.

.. math::

    \rm{var}(w_i)=\sg^2\cd c_{ii}\quad\quad\rm{se}(w_i)=\sqrt{\rm{var}(w_i)}=\sg\cd\sqrt{c_{ii}}

We can now define the standardized variable :math:`Z_{w_i}` that can be used to 
derive the confidence intervals for :math:`w_i` as follows

.. note::

    :math:`\dp Z_{w_i}=\frac{w_i-E[w_i]}{\rm{se}(w_i)}=\frac{w_i-\omega_i}{\hat\sg\sqrt{c_{ii}}}`

Each of the variables :math:`Z_{w_i}` follows a :math:`t`-distribution with 
:math:`n-d-1` degrees of freedom, from which we can obtain the 
:math:`100(1-\alpha)\%` confidence interval of the true value :math:`\omega_i`
as follows:

.. math::

    P(w_i-t_{\alpha/2}\cd\rm{se}(w_i)\leq\omega_i\leq w_i+t_{\alpha/2}\cd\rm{se}(w_i))=1-\alpha

Here, :math:`t_{\alpha/2}` is the critical value of the :math:`t` distribution, 
with :math:`n-d-1` degrees of freedom, that encompasses :math:`\alpha/2` 
fraction of the probability mass in the right tail, given as

.. math::

    P(Z\geq t_{\alpha/2})=\frac{\alpha}{2}\rm{\ or\ equivalently\ }T_{n-d-1}(t_{\alpha/2})=1-\frac{\alpha}{2}

27.2.3 Hypothesis Testing
^^^^^^^^^^^^^^^^^^^^^^^^^

We set up the null hypothesis that all the true weights are zero, except for the bias term (:math:`\beta=\omega_0`).
We contrast the nul hypothesis with the alternative hypothesis that at least one of the weights is not zero

.. math::

    H_0&:\omega_1=0,\omega_2=0,\cds,\omega_d=0

    H_a&:\exists i,\rm{\ such\ that\ }\omega_i\neq 0

The null hypothesis can also be written as :math:`H_0:\bs\omega=\0`.

We use the :math:`F`-test that compares the ratio of the adjusted RSS value to 
the estimated variance :math:`\hat\sg^2`, defined via the :math:`f`-statistic

.. note::

    :math:`\dp f=\frac{RSS/d}{\hat\sg^2}=\frac{RSS/d}{SEE/(n-d-1)}`

Under the null hypothesis, we have

.. math::

    E[RSS/d]=\sg^2

To see this, consider

.. math::

    \hat{Y}&=b\cd\1+w_1\cd X_1+\cds+w_d\cd X_d

    \hat{Y}&=(\mu_Y-w_1\mu_{X_1}-\cds-w_d\mu_{X_d})\cd\1+w_1\cd X_1+\cds+w_d\cd X_d

    \hat{Y}-\mu_Y\cd\1&=w_1(X_1-\mu_{X_1}\cd\1)+\cds+w_d(X_d-\mu_{X_d}\cd\1)

    \hat{\bar{Y}}&=w_1\bar{X_1}+w_2\bar{X_2}+\cds+w_d\bar{X_d}=\sum_{i=1}^dw_i\bar{X_i}

Consider the RSS value; we have

.. math::

    RSS&=\lv\hat{Y}-\mu_Y\cd\1\rv^2=\lv\hat{\bar{Y}}\rv^2=\hat{\bar{Y}}^T\hat{\bar{Y}}

    &=\bigg(\sum_{i=1}^dw_i\bar{X_i}\bigg)^T\bigg(\sum_{j=1}^dw_j\bar{X_j}\bigg)
    =\sum_{i=1}^d\sum_{j=1}^dw_iw_j\bar{X_i}^T\bar{X_j}=\w^T(\bar\D^T\bar\D)\w

The expected value of RSS is thus given as

.. math::

    E[RSS]&=E[\w^T(\bar\D^T\bar\D)\w]

    &=\rm{tr}(E[\w^T\bar\D^T\bar\D)\w])

    &=E[\rm{tr}(\w^T(\bar\D^T\bar\D)\w)]

    &=E[\rm{tr}((\bar\D^T\bar\D)\w\w^T)]

    &=\rm{tr}((\bar\D^T\bar\D)\cd E[\w\w^T])

    &=\rm{tr}((\bar\D^T\bar\D)\cd(\rm{cov}(\w)+E[\w]\cd E[\w]^T))

    &=\rm{tr}((\bar\D^T\bar\D)\cd\rm{cov}(\w))

    &=\rm{tr}((\bar\D^T\bar\D)\cd\sg^2(\bar\D^T\bar\D)\im)

    &=\sg^2\rm{tr}(\I_d)=d\cd\sg^2

.. math::

    E\bigg[\frac{RSS}{d}\bigg]&=\frac{1}{d}E[RSS]=\frac{1}{d}\cd d\cd\sg^2=\sg^2

    E[\hat\sg^2]&=\sg^2

Thus, under the null hypothesis the :math:`f`-statistic has a value close to 1,
which indicates that there is no relationship between the predictor and response 
variables.
On the other hand, if the alternative hypothesis is true, then 
:math:`E[RSS/d]\geq\sg^2`, resulting in a larger :math:`f` value.

The ratio :math:`f` follows a :math:`F`-distribution with :math:`d`, 
:math:`(n-d-1)` degrees of freedom for the numerator and denominator, 
respectively.
Therefore, we can reject the null hypothesis if the p-value is less than the chosen significance level.

Notice that, since :math:`R^2=1-\frac{SSE}{TSS}=\frac{RSS}{TSS}`, we have

.. math::

    SSE=(1-R^2)\cd TSS\quad\quad RSS=R^2\cd TSS

Therefore, we can rewrite the :math:`f` ratio as follows

.. note::

    :math:`\dp f=\frac{RSS/d}{SSE/(n-d-1)}=\frac{n-d-1}{d}\cd\frac{R^2}{1-R^2}`

In other words, the :math:`F`-test compares the adjusted fraction of explained variation to the unexplained variation.
If :math:`R^2` is high, it means the model can fit the data well, and that is 
more evidence to reject the null hypothesis.

**Hypothesis Testing for Individual Parameters**

For attribute :math:`X_i`, we set up the null hypothesis :math:`H_0:\omega_i=0` 
and contrast it with the alternative hypothesis :math:`H_a:\omega_i\neq 0`.
The standardized variable :math:`Z_{w_i}` under the null hypothesis is given as

.. note::

    :math:`\dp Z_{w_i}=\frac{w_i-E[w_i]}{\rm{se}(w_i)}=\frac{w_i}{\rm{se}(w_i)}=\frac{w_i}{\hat\sg\sqrt{c_{ii}}}`

Next, using a two-tailed :math:`t`-test with :math:`n-d-1` degrees of freedom, we compute p-value (:math:`Z_{w_i}`).
If this probability is smaller than the significance level :math:`\alpha`, we can reject the null hypothesis.
Otherwise, we accept the null hypothesis, which would imply that :math:`X_i` 
does not add significant value in predicting the response in light of other
attributes already used to fit the model.
The :math:`t`-test can also be used to test whether the bias term is significantly different from 0 or not.

27.2.4 Geometric Approach to Statistical Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\bar{X_i}=X_i-\mu_{X_i}\cd\1` denote the centered attribute vector, 
and let :math:`\bar\X=(\bar{X_1},\bar{X_2},\cds,\bar{X_d})^T` denote the
multivariate centered vector of predictor variables.
The :math:`n`-dimensional space over the points is divided into three mutually 
orthogonal subspaces, namely the 1-dimensioal *mean space* 
:math:`\cl{S}_\mu=span(\1)`, the :math:`d` dimensional *centered variable space*
:math:`\cl{S}_{\bar{X}}=span(\bar\X)`, and the :math:`n-d-1` dimensional
*error space* :math:`\cl{S}_\epsilon`, which contains the error vector 
:math:`\bs\epsilon`.
The response vector :math:`Y` can thus be decomposed into three components

.. math::

    Y=\mu_Y\cd\1+\hat{\bar{Y}}+\bs\epsilon

Recall that the *degrees of freedom* of a random vector is defined as the dimensionality of its enclosing subspace.
Since the original dimensionality of the point space is :math:`n`, we have a total of :math:`n` degrees of freedom.
The mean space has dimensionality :math:`dim(\cl{S}_\mu)=1`, the centered 
variable space has :math:`dim(\cl{S}_{\bar{X}})=d`, and the error space has
:math:`dim(\cl{S}_\epsilon)=n-d-1`, so that we have

.. math::

    dim(\cl{S}_\mu)+\dim(\cl{S}_{\bar{X}})+dim(\cl{S}_\epsilon)=1+d+(n-d-1)=n

**Population Regression Model**

For a fixed value :math:`\x=(x_{i1},x_{i2},\cds,x_{id})^T`, the true response :math:`y_i` is given as

.. math::

    y_i=\beta+\omega_1\cd x_{i1}+\cds+\omega_d\cd x_{id}+\ve_i

where the systematic part of the model 
:math:`\beta+\sum_{j-1}^d\omega_j\cd x_{ij}` is fixed, and the error term 
:math:`\ve_i` varies randomly, with the assumption that :math:`\ve_i` follows a 
normal distribution with mean :math:`\mu=0` and variance :math:`\sg^2`.
We also assume that the :math:`\ve_i` values are all independent of each other.

.. math::

    y_i&=\mu_Y+\omega_1\cd(x_{i1}-\mu_{X_1})+\cds+\omega_d\cd(x_{id}-\mu_{X_d})+\ve_i

    &=\mu_Y+\omega_1\cd\bar{x_{i1}}+\cds+\omega_d\cd\bar{x_{id}}+\ve_i

Across all the points, we can rewrite the above equation in vector form

.. math::

    Y=\mu_Y\cd\1+\omega_1\bar{X_1}+\cds+\omega_d\cd\bar{X_d}+\bs\ve

We can also center the vector :math:`Y`, so that we obtain a regression model 
over the centered response and predictor variables

.. math::

    \bar{Y}=Y-\mu_Y\cd\1=\omega_1\cd\bar{X_1}+\omega_2\cd\bar{X_2}+\cds+\omega_d
    \cd\bar{X_d}+\bs\ve=E[\bar{Y}|\bar{\X}]+\bs\ve

In this equation, :math:`\sum_{i=1}^d\omega_i\cd\bar{X_i}` is a fixed vector 
that denotes the expected value :math:`E[\bar{Y}|\bar\X]` and :math:`\bs\ve` is
an :math:`n`-dimensional random vector that is distributed according to a 
:math:`n`-dimensional multivariate normal vector with mean :math:`\mmu=\0`, and
a fixed variance :math:`\sg^2` along all dimensions, so that its covariance 
matrix is :math:`\bs\Sg=\sg^2\cd\I`.
The distribution of :math:`\bs\ve` is therefore given as

.. math::

    f(\bs\ve)=\frac{1}{(\sqrt{2\pi})^n\cd\sqrt{|\bs\Sg|}}\cd\exp\bigg\{-\frac{
    \bs\ve^T\bs\Sg\im\bs\ve}{2}\bigg\}=\frac{1}{(\sqrt{2\pi})^n\cd\sg^n}\cd\exp
    \bigg\{-\frac{\lv\bs\ve\rv^2}{2\cd\sg^2}\bigg\}

which follows from the fact that 
:math:`|\bs\Sg|=\det(\bs\Sg)=\det(\sg^2\I)=(\sg^2)^n` and 
:math:`\bs\Sg\im=\frac{1}{\sg^2}\I`.

The density of :math:`\bs\ve` is thus a function of its squared length :math:`\lv\bs\ve\rv^2`, independent of its angle.
In other words, the vector :math:`\bs\ve` is distributed uniformly over all 
angles and is equally likely to point in any direction.

**Hypothesis Testing**

Consider the population regression model

.. math::

    Y=\mu_Y\cd\1+\omega_1\cd\bar{X_1}+\cds+\omega_d\cd\bar{X_d}+\bs\ve=\mu_Y\cd\1+E[\bar{Y}|\bs\X]+\bs\ve

The null hypothesis is

.. math::

    H_0:\omega_1=0,\omega_2=0,\cds,\omega_d=0

In this case, we have

.. math::

    Y=\mu_Y\cd\1+\bs\ve\Rightarrow Y-\mu_Y\cd\1=\bs\ve\Rightarrow\bar{Y}=\bs\ve

Since :math:`\bs\ve` is normally distributed with mean :math:`\0` and covariance 
matrix :math:`\sg^2\cd\I`, under the null hypothesis, the variation in 
:math:`\bar{Y}` for a given value of :math:`\x` will therefore be centered 
around the origin :math:`\0`.

On the other hand, under the alternative hypothesis :math:`H_a` that at least 
one of the :math:`\omega_i` is non-zero, we have

.. math::

    \bar{Y}=E[\bar{Y}|\bar\X]+\bs\ve

Thus, the variation in :math:`\bar{Y}` is shifted away from the origin 
:math:`\0` in the direction :math:`E[\bar{Y}|\bar\X]`.

We estimate its true value by projecting the centered observation vector 
:math:`\bar{Y}` onto the subspace :math:`\cl{S}_{\bar{X}}` and 
:math:`\cl{S}_\epsilon`, as follows

.. math::

    \bar{Y}=w_1\cd\bar{X_1}+\w_2\cd\bar{X_2}+\cds+w_d\cd\bar{X_d}+\bs\epsilon=\hat{\bar{Y}}+\bs\epsilon

Under the null hypothesis, the true centered response vector is 
:math:`\bar{Y}=\bs\ve`, and therefore, :math:`\hat{\bar{Y}}` and 
:math:`\bs\epsilon` are simply the projections of the random error vector
:math:`\bs\ve` onto the subspaces :math:`\cl{S}_{\bar{X}}` and 
:math:`\cl{S}_\epsilon`.
In this case, we also expect the length of :math:`\bs\epsilon` and :math:`\hat{\bar{Y}}` to be roughly comparable.
On the other hand, under the alternative hypothesis, we have 
:math:`\bar{Y}=E[\bar{Y}|\bar\X]+\bs\ve`, and so :math:`\hat{\bar{Y}}` will be
relatively much longer compared to :math:`\bs\epsilon`.

Define the mean squared length of per dimension for the two vectors 
:math:`\hat{\bar{Y}}` and :math:`\bs\epsilon`, as follows

.. math::

    M(\hat{\bar{Y}})&=\frac{\lv\hat{\bar{Y}}\rv^2}{dim(\cl{S}_{\bar{X}})}=\frac{\lv\hat{\bar{Y}}\rv^2}{d}

    M(\bs\epsilon)&=\frac{\lv\bs\epsilon\rv^2}{dim(\cl{S}_\epsilon)}=\frac{\lv\bs\epsilon\rv^2}{n-d-1}

The geometric ratio test is identical to the F-test since

.. math::

    \frac{M(\hat{\bar{Y}})}{M(\bs\epsilon)}=\frac{\lv\hat{\bar{Y}}\rv^2/d}
    {\lv\bs\epsilon\rv^2/(n-d-1)}=\frac{RSS/d}{SSE/(n-d-1)}=f

The geometric approach makes it clear that if :math:`f\simeq 1` then the null 
hypothesis holds, and we conclude that :math:`Y` does not depend on the 
predictor variables :math:`X_1,X_2,\cds,X_d`.
On the other hand, if :math:`f` is large, with a p-value less than the 
significance level, then we can reject the null hypothesis and accept the
alternative hypothesis that :math:`Y` depends on at least one predictor variable
:math:`X_i`.