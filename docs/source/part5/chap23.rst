Chapter 23 Linear Regression
============================

Given a set of attributes or variables :math:`X_1,X_2,\cds,X_d`, called the 
*predictor*, *explanatory*, or *independent* variables, and given a real-valued 
attribute of interest :math:`Y`, called the *response* or *dependent* variable,
the aim of *regression* is to predict the response variable based on the 
independent variables.
That is, the goal is to learn a *regression function* :math:`f`, such that

.. math::

    Y=f(X_1,X_2,\cds,X_d)+\ve=f(\X)+\ve

where :math:`\X=(X_1,X_2,\cds,X_d)^T` is the multivariate random variable 
comprising the predictor attributes, and :math:`\ve` is a random *error term* 
that is assumed to be independent of :math:`\X`.
In other words, :math:`Y` is comrpised of two components, one dependent on the
observed predictor attributes, and the other, coming from the error term,
independent of the predictor attributes.
The error term encapsulates inherent uncertainty in :math:`Y`, as well as,
possibly the effect of unobserved, hidden or *latent* variables.

23.1 Linear Regression Model
----------------------------

In *linear regression* the function :math:`f` is assumed to be linear in its parameters, that is

.. note::

    :math:`\dp f(\X)=\beta+\omega_1X_1+\omega_2X_2+\cds+\omega_dX_d=\beta+\sum_{i=1}^d\omega_iX_i=\beta+\bs{\omega}^T\X`

Here, the parameter :math:`\beta` is the true (unknown) *bias* term, the 
parameter :math:`\omega_i` is the true (unknown) *regression coefficient* or
*weight* for attribute :math:`X_i`, and 
:math:`\bs{\omega}=(\omega_1,\omega_2,\cds,\omega_d)^T` is the true :math:`d`-
dimensional weight vector.
Observe that :math:`f` specifies a hyperplane in :math:`\R^{d+1}`, where 
:math:`\bs{\omega}` is the weight vector that is normal or orthogonal to the
hyperplane, and :math:`\beta` is the *intercept* or offset term.
We can see that :math:`f` is completely specified by the :math:`d+1` parameters
comprising :math:`\beta` and :math:`\omega_i`, for :math:`i=1,\cds,\d`.

The true bias and regression coefficients are unknown.
Therefore, we have to estimate them from the training dataset :math:`\D` 
comprising :math:`n` points :math:`\x_i\in\R^d` in a :math:`d`-dimensional 
space, and the corresponding response values :math:`y_i\in\R`, for 
:math:`i=1,2,\cds,n`.
Let :math:`b` denote the estimated value for the true bias :math:`\beta`, and 
let :math:`w_i` denote the estimated value for the true regression coefficient 
:math:`w_i`, for :math:`i=1,2,\cds,d`.
Let :math:`\w=(w_1,w_2,\cds,w_d)^T` denote the vector of estimated weights.
Given the estimated bias and weight values, we can predict the response for any 
given input or test point :math:`\x=(x_1,x_2,\cds,x_d)^T`, as follows:

.. math::

    \hat{y}=b+w_1x_1+\cds+w_dx_d=b+\w^T\x

The difference between the observed and predicted response, called the *residual error*, is given as

.. math::

    \epsilon=y-\hat{y}=y-b-\w^T\X

The residual error :math:`\epsilon` is an estimator of the random error term :math:`\ve`.

A common approach to predicting the bias and regression coefficients is to use the method of *least squares*.
That is, given the training data :math:`\D` with points :math:`\x_i` and 
response values :math:`y_i` (for :math:`i=1,\cds,n`), we seek values :math:`b`
and :math:`\w`, so as to minimize the sum of squared residual errors (SSE)

.. note::

    :math:`\dp SSE=\sum_{i=1}^n\epsilon_i^2=\sum_{i=1}^n(y_i-\hat{y_i})^2=\sum_{i=1}^n(y_i-b-\w^T\x_i)^2`

23.2 Bivariate Regression
-------------------------

Let us first consider the case where the input data :math:`\D` comprises a 
single predictor attribute, :math:`W=(x_1,x_2,\cds,x_n)^T`, along with the
response variable, :math:`Y=(y_1,y_2,\cds,y_n)^T`.
Since :math:`f` is linear, we have

.. note::

    :math:`\hat{y_i}=f(x_i)=b+w\cd x_i`

Thus, we seek the straight line :math:`f(x)` with slope :math:`w` and intercept :math:`b` that *best fits* the data.
The residual error, which is the difference between the predicted value (also
called *fitted value*) and the observed value of the response variable, is given
as

.. math::

    \epsilon_i=y_i-\hat{y_i}

Note that :math:`|\epsilon_i|` denotes the vertical distance between the fitted and observed response.
The best fitting line minimizes the sum of squared errors

.. note::

    :math:`\dp\min_{b,w}SSE=\sum_{i=1}^n\epsilon_i^2=\sum_{i=1}^n(y_i-\hat{y_i})^2=\sum_{i=1}^n(y_i-b-w\cd x_i)^2`

.. math::

    \frac{\pd}{\pd b}SSE&=-2\sum_{i=1}^n(y_i-b-w\cd x_i)=0

    &\Rightarrow\sum_{i=1}^n b=\sum_{i=1}^ny_i-w\sum_{i=1}^nx_i

    &\Rightarrow b=\frac{1}{n}\sum_{i=1}^ny_i-w\cd\frac{1}{n}\sum_{i=1}^nx_i

.. note::

    :math:`b=\mu_Y-w\cd\mu_X`

where :math:`\mu_Y` is the sample mean for the response and :math:`\mu_X` is the 
sample mean for the predictor attribute.

.. math::

    \frac{\pd}{\pd w}SSE&=-2\sum_{i=1}^nx_i(y_i-b-w\cd x_i)=0

    &\Rightarrow\sum_{i=1}^nx_i\cd y_i-b\sum_{i=1}^nx_i-w\sum_{i=1}^nx_i^2=0

    &\Rightarrow\sum_{i=1}^nx_i\cd y_i-\mu_Y\sum_{i=1}^nx_i+w\cd\mu_X\sum_{i=1}^nx_i-w\sum_{i=1}^nx_i^2=0

    &\Rightarrow w\bigg(\sum_{i=1}^nx_i^2-n\cd\mu_X^2\bigg)=\bigg(\sum_{i=1}^nx_i\cd y_i\bigg)-n\cd\mu_X\mu_Y

    &\Rightarrow w=\frac{(\sum_{i=1}^nx_i\cd y_i)-n\cd\mu_X\cd\mu_Y}{(\sum_{i=1}^nx_i^2)-n\cd\mu_X^2}

.. note::

    :math:`\dp w=\frac{\sum_{i=1}^n(x_i-\mu_X)(y_i-\mu_Y)}{\sum_{i=1}^n(x_i-\mu_X)^2}=`
    :math:`\dp\frac{\sg_{XY}}{\sg_X^2}=\frac{\rm{cov}(X,Y)}{\rm{var}(X)}`

where :math:`\sg_X^2` is the variance of :math:`X` and :math:`\sg_{XY}` is the 
covariance between :math:`X` and :math:`Y`.
Noting that the correlation between :math:`X` and :math:`Y` is given as 
:math:`\rho_{XY}=\frac{\sg_{XY}}{\sg_X\cd\sg_Y}`, we can also write :math:`w` as

.. math::

    w=\rho_{XY}=\frac{\sg_Y}{\sg_X}

.. math::

    \hat{y_i}=b+w\cd x_i=\mu_Y-w\cd\mu_X+w\cd x_i=\mu_Y+w(x_i-\mu_X)

Thus, the point :math:`(\mu_X,\mu_Y)` lins on the regression line.

23.2.1 Geometry of Bivariate Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

