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

Let :math:`X=(x_1,x_2,\cds,x_n)^T` be the :math:`n`-dimensional vector denoting 
the training data sample, :math:`Y=(y_1,y_2,\cds,y_n)^T` the sample vector for
the response variable, and 
:math:`\hat{Y}=(\hat{y_1},\hat{y_2},\cds,\hat{y_n})^T` the vector of predicted
values, then we can express the :math:`n` equations, :math:`y_i=b+w\cd x_i` for
:math:`i=1,2,\cds,n`, as a single vector equation:

.. math::

    \hat{Y}=b\cd\1+w\cd X

This equation indicates that the predicted vector :math:`\hat{Y}` is a linear
combination of :math:`\1` and :math:`X`, i.e., it must lie in the column space
spanned by :math:`\1` and :math:`X`, given as :math:`\rm{span}(\{\1,X\})`.
On the other hand, the response vector :math:`Y` will not usually lie in the same column space.
In fact, the residual error vector 
:math:`\bs{\epsilon}=(\epsilon_1,\epsilon_2,\cds,\epsilon_n)^T` captures the 
deviation between the response and predicted vectors

.. math::

    \bs{\epsilon}=Y-\hat{Y}

The geometry of the problem makes it clear that the optimal :math:`\hat{Y}` that
minimizes the error is the orthogonal projection of :math:`Y` onto the subspace
spanned by :math:`\1` and :math:`X`.
The residual error vector :math:`\bs{\epsilon}` is thus *orthogonal* to the 
subspace spanned by :math:`\1` and :math:`X`, and its squared length (or 
magnitude) equals the SSE value, Since

.. math::

    \lv\bs{\epsilon}\rv^2=\lv Y-\hat{Y}\rv^2=\sum_{i=1}^n(y_i-\hat{y_i})^2=\sum_{i=1}^n\epsilon_i=SSE

Even though :math:`\1` and :math:`X` are linearly independent and form a basis 
for the column space, they need not be orthogonal.
We can create an orthogonal basis by decomposing :math:`X` into a component 
along :math:`\1` and a component orthogonal to :math:`\1`.

.. math::

    \rm{proj}_\1(X)\cd\1=\bigg(\frac{X^T\1}{\1^T\1}\bigg)\cd\1=\bigg(\frac{\sum_{i=1}^nx_i}{n}\bigg)\cd\1=\mu_X\cd\1

.. math::

    X=\mu_X\cd\1+(X-\mu_X\cd\1)=\mu_X\cd\1+\bar{X}

where :math:`\bar{X}=X-\mu_X\cd\1` is the centered attribute vector, obtained by 
subtracting the mean :math:`\mu_X` from all points.

The two vectors :math:`\1` and :math:`\bar{X}` form an *orthogonal basis* for the subspace.
We can thus obtain the predicted vector :math:`\bar{Y}` by projecting :math:`Y` 
onto :math:`\1` and :math:`\bar{X}`, and summing up these two components.
That is,

.. math::

    \hat{Y}=\rm{proj}_\1(Y)\cd\1+\rm{proj}_{\bar{X}}(Y)\cd\bar{X}=
    \bigg(\frac{Y^T\1}{\1^T\1}\bigg)\1+
    \bigg(\frac{Y^T\bar{X}}{\bar{X}^T\bar{X}}\bigg)\bar{X}=
    \mu_Y\cd\1+\bigg(\frac{Y^T\bar{X}}{\bar{X}^T\bar{X}}\bigg)\bar{X}

.. math::

    \hat{Y}=b\cd\1+w\cd X=b\cd\1+w(\mu_X\cd\1+\bar{X})=(b+w\cd\mu_X)\cd\1+w\cd\bar{X}

Since both are expressions for :math:`\hat{Y}`, we can equate them to obtain

.. math::

    \mu_Y=b+w\cd\mu_X\quad\rm{or}\quad b=\mu_Y-w\cd\mu_X\quad\quad w=\frac{y^T\bar{X}}{\bar{X}^T\bar{X}}

.. math::

    w=\frac{y^T\bar{X}}{\bar{X}^T\bar{X}}=\frac{Y^T\bar{X}}{\lv\bar{X}\rv^2}=
    \frac{Y^T(X-\mu_X\cd\1)}{\lv X-\mu_X\cd\1\rv^2}=\frac{(\sum_{i=1}^nx_i\cd 
    y_i)-n\cd\mu_X\cd\mu_Y}{(\sum_{i=1}^nx_i^2)-n\cd\mu_X^2}

22.3 Multiple Regression
------------------------

We now consider the more general case called *multiple regression* where we have
multiple predictor attributes :math:`X_1,X_2,\cds,X_d` and a single response 
attribute :math:`Y`.
The training data sample :math:`\D\in\R^{n\times d}` comprises :math:`n` points
:math:`\x_i=(x_{i1},x_{i2},\cds,x_{id})^T` in a :math:`d`-dimensional space,
along with the corresponding observed response value :math:`y_i`.
The vector :math:`Y=(y_1,y_2,\cds,y_n)^T` denotes the observed response vector.
The predicted response value for input :math:`\x_i` is given as

.. math::

    \hat{y_i}=b+w_1x_{i1}+w_2x_{i2}+\cds+w_dx_{id}=b+\w^T\x_i

where :math:`\w=(w_1,w_2,\cds,w_d)^T` is the weight vector comprising the 
regression coefficients or weights :math:`w_j` along each attribute :math:`X_j`.

Instead of dealing with the bias :math:`b` separately from the weights 
:math:`w_i` for each attribute, we can introduce a new "constant" valued
attribute :math:`X_0` whose value is always fixed at 1, so that each input point
:math:`\x_i=(x_{i1},x_{i2},\cds,x_{id})^T\in\R^d` is mapped to an augmented
point :math:`\td{\x_i}=(x_{i0},x_{i1},x_{i2},\cds,x_{id})^T\in\R^{d+1}`, where
:math:`x_{i0}=1`.
Likewise, the weight vector :math:`\w=(w_1,w_2,\cds,w_d)^T` is mapped to an 
augmented weight vector :math:`\td{\w}=(w_0,w_1,w_2,\cds,w_d)^T`, where 
:math:`w_0=b`.
The predicted response value for an augmented :math:`(d+1)` dimensional point :math:`\td{\x_i}` can be written as

.. note::

    :math:`\hat{y_i}=w_0x_{i0}+w_1x_{i1}+w_2x_{i2}+\cds+w_dx_{id}=\td{\w}^T\td{\x_i}`

We can compactly write all thes :math:`n` equations as a single matrix equation, given as

.. math::

    \hat{Y}=\td{\D}\td{\w}

where :math:`\td{\D}\in\R^{n\times(d+1})` is the *augmented data matrix*, which
includes the constant attribute :math:`X_0` in addition to the predictor 
attributes :math:`X_1,X_2,\cds,X_d`, and 
:math:`\hat{Y}=(\hat{y_1},\hat{y_2},\cds,\hat{y_n})^T` is the vector of 
predicted responses.

The multiple regression task can now be stated as finding the *best fitting* 
*hyperplane* defined by the weight vector :math:`\td{\w}` that minimizes the sum
of squared errors

.. math::

    \min_{\td\w}SSE&=\sum_{i=1}^n\epsilon_i^2=\lv\bs\epsilon\rv^2=\lv Y-\hat{Y}\rv^2

    &=(Y-\hat{Y})^T(Y-\hat{Y})=Y^TY-2Y^T\hat{Y}+\hat{Y}^T\hat{Y}

    &=Y^TY-2Y^T(\td\D\td\w)+(\td\D\td\w)^T(\td\D\td\w)

    &=Y^TY-2\td\w^T(\td\D^TY)+\td\w^T(\td\D^T\td\D)\td\w

.. math::

    \frac{\pd}{\pd\td\w}SSE&=-2\td\D^TY+2(\td\D^T\td\D)\td\w=\0

    &\Rightarrow(\td\D^T\td\D)\td\w=\td\D^TY

.. note::

    :math:`\td\w=(\td\D^T\td\D)\im\td\D^TY`

.. math::

    \hat{Y}=\td\D\td\w=\td\D(\td\D^T\td\D)\im\td\D Y=\bs{\rm{H}}Y

23.3.1 Geometry of Multiple Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\td\D` be the augmented data matrix comprising the :math:`d` 
independent attributes :math:`X_i`, along with the new constant attribute 
:math:`X_0=\1\in\R^n`, given as

.. math::

    \td\D=\bp |&|&|&&|\\X_0&X_1&X_2&\cds&X_d\\|&|&|&&| \ep

Let :math:`\td\w=(w_0,w_1,\cds,w_d)^T\in\R^(d+1)` be the augmented weight vector
that incorporates the bias term :math:`b=w_0`.

.. math::

    \hat{Y}=b\cd\1+w_1\cd X_1+w_2\cd X_2+\cds+w_d\cd X_d=\sum_{i=0}^dw_i\cd X_i=\td\D\td\w

This euqation makes it clear that the predicted vector must lie in the column 
space of the augmented data matrix :math:`\td\D`, denoted :math:`col(\td\D)`,
i.e., it must be a linear combination of the attribute vectors :math:`X_i`, 
:math:`i=0,\cds,d`.

To minimize the error in prediction, :math:`\hat{Y}` must be the orthogonal 
projection of :math:`Y` onto the subspace :math:`col(\td\D)`.
The residual error vector :math:`\bs\epsilon=Y-\hat{Y}` is thus orthogonal to 
the subspace :math:`col(\td\D)`, which means that it is orthogonal to each 
attribute vector :math:`X_i`.

.. math::

    &\ \ \ \ \ \ \ X_i^T\bs\epsilon=0

    &\Rightarrow X_i^T(Y-\hat{Y})=0

    &\Rightarrow X_i^T\hat{Y}=X_i^TY

    &\Rightarrow X_i^T(\td\D\td\w)=X_i^TY

    &\Rightarrow w_0\cd X_i^TX_0+w_1\cd X_i^TX_1+\cds+w_d\cd X_i^TX_d=X_i^TY

We thus have :math:`(d+1)` equations, called the *normal equations*, in 
:math:`(d+1)` unknowns, namely the regression coefficients or weights 
:math:`w_i` (including the bias term :math:`w_0`).

.. math::

    \bp X_0^TX_0&X_0^TX_1&\cds&X_0^TX_d\\X_1^TX_0&X_1^TX_1&\cds&X_1^TX_d\\
    \vds&\vds&\dds&\vds\\X_d^TX_0&X_d^TX_1&\cds&X_d^TX_d\ep\td\w=\td\D^TY

.. math::

    (\td\D^T\td\D)\td\w&=\td\D^TY

    \td\w&=(\td\D^T\td\D)\im(\td\D^TY)

More insight can be obtained by noting that the attribute vectors comprising the 
column space of :math:`\td\D` are not necessarily orthogonal, even if we assume
they are linearly independent.
To obtain the projected vector :math:`\hat{Y}`, we first need to construct an orthogonal basis for :math:`col(\td\D)`.

Let :math:`U_0,U_1,\cds,U_d` denote the set of orthogonal basis vectors for :math:`col(\td\D)`.
We construct these vectors in a step-wise manner via *Gram-Schmidt orthogonalization*, as follows

.. math::

    U_0&=X_0

    U_1&=X_1-p_{10}\cd U_0

    U_2&=X_2-p_{20}\cd U_0-p_{21}\cd U_1

    \vds&=\vds

    U_d&=X_d-p_{d0}\cd U_0-p_{d1}\cd U_1-\cd-p_{d,d-1}\cd U_{d-1}

where

.. math::

    p_{ji}=\rm{proj}_{U_i}(X_j)=\frac{X_j^TU_i}{\lv U_i\rv^2}

denotes the scalar projection of attribute :math:`X_j` onto the basis vector :math:`U_i`.

Rearranging the equations above, we get

.. math::

    X_0&=U_0

    X_1&=p_{10}\cd U_0+U_1

    X_2&=P_{20}\cd U_0+p_{21}\cd U_1+U_2

    \vds&=\vds

    X_d&=p_{d0}\cd U_0+p_{d1}\cd U_1+\cds+p_{d,d-1}\cd U_{d-1}+U_d

The Gram-Schmidt method thus results in the so-called *QR-factorization* of the 
data matrix, namely :math:`\td\D=\bs{\rm{Q}}\bs{\rm{R}}`, where by construction 
:math:`\bs{\rm{Q}}` is an :math:`n\times(d+1)` matrix with orthogonal columns

.. math::

    \bs{\rm{Q}}=\bp |&|&|&&|\\U_0&U_1&U_2&\cds&U_d\\|&|&|&&| \ep

and :math:`\bs{\rm{R}}` is the :math:`(d+1)\times(d+1)` upper-triangular matrix

.. math::

    \bs{\rm{R}}=\bp 1&p_{10}&p_{20}&\cds&p_{d0}\\0&1&p_{21}&\cds&p_{d1}\\0&0&1&
    \cds&p_{d2}\\\vds&\vds&\vds&\dds&\vds\\0&0&0&1&p_{d,d-1}\\0&0&0&0&1\ep

.. math::

    \hat{Y}=\rm{proj}_{U_0}(Y)\cd U_0+\rm{proj}_{U_1}(Y)\cd U_1+\cds+\rm{proj}_{U_d}(Y)\cd U_d

**Bias Term**

Defien :math:`\bar{X_i}` to be the centered attribute vector

.. math::

    \bar{X_i}=X_i-\mu_{X_i}\cd\1

.. math::

    \hat{Y}&=b\cd\1+w_1\cd X_1+w_2\cd X_2+\cds+2_d\cd X_d

    &=b\cd\1+w_1\cd(\bar{X_1}+\mu_{X_1}\cd\1)+\cds+w_d\cd(\bar{X_d}+\mu_{X_d}\cd\1)

    &=(b+w_1\cd\mu_{X_1}+\cds+w_d\cd\mu_{X_d})\cd\1+w_1\cd\bar{X_1}+\cds+w_d\cd\bar{X_d}

On the other hand, since :math:`\1` is orthogonal to all :math:`\bar{X_i}`, we 
can obtain another expression for :math:`\bar{Y}` in terms of the projection of 
:math:`Y` onto the subspace spanned by the vectors 
:math:`\{\1,\bar{X_1},\cds,\bar{X_d}\}`.
Let the new orthogonal basis for these centered attribute vectors be 
:math:`\{\bar{U_0},\bar{U_1},\cds,\bar{U_d}\}`, where :math:`\bar{U_0}=\1`.
Thus, :math:`\hat{Y}` can also be written as

.. math::

    \hat{Y}=\rm{proj}_{\bar{U_0}}(Y)\cd\bar{U_0}+\sum_{i=1}^d
    \rm{proj}_{\bar{U_i}}(Y)\cd\bar{U_i}=\rm{proj}_\1+\sum_{i=1}^d\rm{proj}_
    {\bar{U_i}}(Y)\cd\bar{U_i}

.. math::

    \rm{proj}_\1(Y)=\mu_Y&=(b+w_1\cd\mu_{X_1}+\cds+w_d\cd\mu_{X_d})

    \Rightarrow b&=\mu_Y-w_1\cd\mu_{X_1}-\cds-w_d\cd\mu_{X_d}=\mu_Y-\sum_{i=1}^dw_i\cd\mu_{X_i}

.. math::

    \rm{proj}_\1(Y)=\frac{Y^T\1}{\1^T\1}=\frac{1}{n}\sum_{i=1}^ny_i=\mu_Y

23.3.2 Multiple Regression Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../_static/Algo23.1.png

.. math::

    \Q^T\Q=\bp\lv U_0\rv^2&0&\cds&0\\0&\lv U_1\rv^2&\cds&0\\0&0&\dds&0\\0&0&\cds&\lv U_d\rv^2\ep=\Delta

.. math::

    (\td\D^T\td\D)\td\w&=\td\D^TY

    (\Q\bs{\rm{R}})^T(\Q\bs{\rm{R}})\td\w&=(\Q\bs{\rm{R}})^TY

    \bs{\rm{R}}^T(\Q^T\Q)\bs{\rm{R}}\td\w&=\bs{\rm{R}}^TQ^TY

    \bs{\rm{R}}^T\Delta\bs{\rm{R}}\td\w&=\bs{\rm{R}}^TQ^TY

    \Delta\bs{\rm{R}}\td\w&=\Q^TY

    \bs{\rm{R}}\td\w&=\Delta\im\Q^TY

.. math::

    \hat{Y}=\td\D\td\w=\Q\bs{\rm{R}}\bs{\rm{R}}\im\Delta\im\Q^TY=\Q(\Delta\im\Q^TY)

.. math::

    \Delta\im\Q^TY=\bp\rm{proj}_{U_0}(Y)\\\rm{proj}_{U_1}(Y)\\\vds\\\rm{proj}_{U_d}(Y)\ep

.. math::

    \hat{Y}=\Q\bp\rm{proj}_{U_0}(Y)\\\rm{proj}_{U_1}(Y)\\\vds\\
    \rm{proj}_{U_d}(Y)\ep=\rm{proj}_{U_0}(Y)\cd U_0+\rm{proj}_{U_1}(Y)\cd U_1+
    \cds+\rm{proj}_{U_d}(Y)\cd U_d

23.3.3 Multiple Regression: Stochastic Gradient Descent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the SSE obejective

.. math::

    \min_{\td\w}SSE=\frac{1}{2}(Y^TY-2\td\w^T(\td\D^TY)+\td\w^T(\td\D^T\td\D)\td\w)

The gradient of the SSE objective is given as

.. math::

    \nabla_{\td\w}=\frac{\pd}{\pd\td\w}SSE=-\td\D^TY+(\td\D^T\td\D)\td\w

Using gradient descent, starting from an initial weight vector estimate 
:math:`\td\w^0`, we can iteratively update :math:`\td\w` as follows

.. math::

    \td\w^{t+1}=\td\w^t-\eta\cd\nabla_{\td\w}=\td\w^t+\eta\cd\td\D^T(Y-\td\D\cd\td\w^t)

In stochastic gradient descent (SGD), we update the weight vector by considering only one (random) point at each time.
Restricting to a single point :math:`\td\x_k` in the training data 
:math:`\td\D`, the gradient at the point :math:`\td\x_k` is given as

.. math::

    \nabla_{\td\w}(\td\x_k)=-\td\x_ky_k+\td\x_k\td\x_k^T\td\w=-(y_k-\td\x_k^T\td\w)\td\x_k

Therefore, the stochastic gradient update rule is given as

.. math::

    \td\w^{t+1}&=\td\w^t-\eta\cd\nabla_{\td\w}(\td\x_k)

    &=\td\w^t+\eta\cd(y_k-\td\x_k^T\td\w^t)\cd\td\x_k

.. image:: ../_static/Algo23.2.png
