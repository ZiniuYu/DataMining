Chapter 24 Logistic Regression
==============================

Given a set of predictor attributes or independent variables 
:math:`X_1,X_2,\cds,X_d`, and given a *categorical* response or dependent 
variable :math:`Y`, the aim of *logistic regression* is to predict the 
probability of the response variable values based on the independent variables.
Logistic regression is in fact a classification technique, that given a point
:math:`\x_i\in\R^d` predicts :math:`P(c_i|\x_j)` for each class :math:`c_i` in
the domain of :math:`Y` (the set of possible classes or values for the response
variable).

24.1 Binary Logistic Regression
-------------------------------

In logistic regression, we are given a set of :math:`d` predictor or independent 
variables :math:`X_1,X_2,\cds,X_d`, and a *binary* or *Bernoulli* response 
variable :math:`Y` that takes on only two values, namely, 0 and 1.
Thus, we are given a training dataset :math:`\D` comprising :math:`n` points 
:math:`\x_i\in\R^d` and the corresponding observed values :math:`y_i\in\{0,1\}`.
We augment the data matrix :math:`\D` by adding a new attribute :math:`X_0` that 
is always fixed at the value 1 for each point, so that 
:math:`\td{\x_i}=(1,x_1,x_2,\cds,x_d)^T\in\R^{d+1}` denotes the augmented point, 
and the multivariate random vector :math:`\td\X`, comprising all the independent 
attributes is given as :math:`\td\X=(X_0,X_1,\cds,X_d)^T`.
The augmented training dataset is given as :math:`\td\D` comprising the 
:math:`n` augmented points :math:`\td{\x_i}` along with the class labels 
:math:`y_i` for :math:`i=1,2,\cds,n`.

Since there are only two outcomes for the response variable :math:`Y`, its 
probability mass function for :math:`\td\X=\td\x` is given as:

.. math::

    P(Y=1|\td\X=\td\x)=\pi(\td\x)\quad\quad P(Y=0|\td\X=\td\x)=1-\pi(\td\x)

where :math:`\pi(\td\x)` is the unknown true parameter value, denoting the 
probability of :math:`Y=1` given :math:`\td\X=\td\x`.

.. math::

    E[Y|\td\X=\td\x]&=1\cd P(Y=1|\td\X=\td\x)+0\cd P(Y=0|\td\X=\td\x)

    &=P(Y=1|\td\X=\td\x)=\pi(\td\x)

Therefore, in logistic regression, instead of directly predicting the response 
value, the goal is to learn the probability, :math:`P(Y=1|\td\X=\td\x)`, which
is also the expected value of :math:`Y` given :math:`\td\X=\td\x`.

Since :math:`P(Y=1|\td\X=\td\x)` is a probability, it is **not appropriate** to directly use the linear regression model

.. math::

    f(\td\x)=\omega_0\cd x_0+\omega_1\cd x_1+\omega_2\cd x_2+\cds+\omega_d\cd x_d=\td{\bs\omega}^T\td\x

where :math:`\td{\bs\omega}=(\omega_0,\omega_1,\cds,\omega_d)^T\in\R^{d+1}` is 
the true augmented weight vector, with :math:`\omega_0=\beta` the true unknown
bias term, and :math:`\omega_i` the true unknown regression coefficient or 
weight for attribute :math:`X_i`.
The reason we cannot simply use :math:`P(Y=1|\td\X=\td\x)=f(\td\x)` is due to 
the fact that :math:`f(\td\x)` can be arbitrarily large or arbitrarily small,
whereas for logistic regression, we require that the output represents a
probability value, and thus we need a model that results in an output that lies
in the interval :math:`[0,1]`.
The name "logistic regression" comes from the *logstic* function (also called 
the *sigmoid* function) that meets this requirement.

.. note::

    :math:`\dp\th(z)=\frac{1}{1+\exp\{-z\}}=\frac{\exp\{z\}}{1+\exp\{z\}}`

The logstic function "squashes" the output to be between 0 and 1 for any scalar input :math:`z`.

.. math::

    1-\th(z)=1-\frac{\exp\{z\}}{1+\exp\{z\}}=\frac{1+\exp\{z\}-\exp\{z\}}{1+\exp\{z\}}=\frac{1}{1+\exp\{z\}}=\th(-z)

Using the logistic function, we define the logistic regression model as follows:

.. math::

    P(Y=1|\td\X=\td\x)=\pi(\td\x)=\th(f(\td\x))=\th(\td{\bs\omega}^T\td\x)-
    \frac{\exp\{\td{\bs\omega}^T\td\x\}}{1+\exp\{\td{\bs\omega}^T\td\x\}}

On the other hand, the probability for :math:`Y=0` is given as

.. math::

    P(Y=0|\td\X=\td\x)=1-P(Y=1|\td\X=\td\x)=\th(-\td{\bs\omega}^T\td\x)=\frac{1}{1+\exp\{\td{\bs\omega}^T\td\x\}}

Combining these two cases the full logistic regression model is given as

.. note::

    :math:`P(Y|\td\X=\td\x)=\th(\td{\bs\omega}^T\td\x)^Y\cd\th(-\td{\bs\omega}^T\td\x)^{1-Y}`

**Log-Odds Ratio**

Define the *odds ratio* for the occurence of :math:`Y=1` as follows:

.. math::

    \rm{odds}(Y=1|\td\X=\td\x)&=\frac{P(Y=1|\td\X=\td\x)}{P(Y=0|\td\X=\td\x)}=
    \frac{\th(\td{\bs\omega}^T\td\x)}{\th(-\td{\bs\omega}^T\td\x)}

    &=\frac{\exp\{\td{\bs\omega}^T\td\x\}}{1+\exp\{\td{\bs\omega}^T\td\x\}}\cd(1+\exp\{\td{\bs\omega}^T\td\x\})

    &=\exp\{\td{\bs\omega}^T\td\x\}

The logarithm of the odds ratio, called the *log-odds ratio*, is therefore given as:

.. math::

    \ln(\rm{odds}(Y=1|\td\X=\td\x))&=\ln\bigg(\frac{P(Y=1|\td\X=\td\x)}
    {1-P(Y=1|\td\X=\td\x)}\bigg)=\ln(\exp\{\td{\bs\omega}^T\td\x\})=
    \td{\bs\omega}^T\td\x

    &=\omega_0\cd x_0+\omega_1\cd x_1+\cds+\omega_d\cd x_d

The log-odds ratio function is also called the *logit* function, defined as

.. math::

    \rm{logit}(z)=\ln\bigg(\frac{z}{1-z}\bigg)

It is the inverse of the logistic function.
We can see that

.. math::

    \ln(\rm{odds}(Y=1|\td\X=\td\x))=\rm{logit}(P(Y=1|\td\X=\td\x))

The logistic regression model is therefore based on the assumption that the log-
odds ratio for :math:`Y=1` given :math:`\td\X=\td\x` is a linear function (or a
weighted sum) of the independent attributes.
Let us consider the effect of attribute :math:`X_i` by fixing the values for all other attributes; we get

.. math::

    &\quad\ \ \ln(\rm{odds}(Y=1|\td\X=\td\x))=\omega_i\cd x_i+C

    &\Rightarrow\rm{odds}(Y=1|\td\X=\td\x)=\exp\{\omega_i\cd x_i+C\}=
    \exp\{\omega_i\cd x_i\}\cd\exp\{C\}\propto\exp\{\omega_i\cd x_i\}

where :math:`C` is a constant comprising the fixed attributes.
The regression coefficient :math:`\omega_i` can therefore be interpreted as the
change in the log-odds ratio for :math:`Y=1` for a unit change in :math:`X_i`,
or equivalently the odds ratio for :math:`Y=1` increases exponentially per unit
change in :math:`X_i`.