Chapter 2 Numeric Attributes
============================

2.1 Univariate Analysis
-----------------------

Univariate analysis focuses on a single attribute at a time; thus the data
matrix :math:`\D` can be thought of as an :math:`n\times 1` matrix, or simply a
column vector, given as

.. math::

    \D=\bp X\\x_1\\x_2\\\vds\\x_n \ep

where :math:`X` is the numeric attribute of interest, with :math:`x+i\in\R`.
:math:`X` is assumed to be a random variable, with each point 
:math:`x_i(1\leq i\leq b)` itself treated as an identity random variable.

**Empirical Cumulative Distribution Function**

The *empirical cumulative distribution function (CDF)* of :math:`X` is given as

.. math::

    \hat{F}(x)=\frac{1}{n}\sum_{i=1}^nI(x_i\leq x)

where

.. math::

    I(x_i\leq x)=\left\{\begin{array}{lr}1\quad\rm{if\ }x_i\leq x\\0\quad\rm{if\ }x_i>x\end{array}\right.

is a binary *indicator variable* that indicates whether the given condition is satisfied or not.
Note that we use the notation :math:`\hat{F}` to denote the fact that the 
empirical CDF is an estimate for the unknown population CDF :math:`F`.

**Inverse Cumulative Distribution Function**

Define the *inverse cumulative distribution function* or *quantile function* for a random variable :math:`X` as follows:

.. math::

    F\im(q)=\min\{x|F(x)\geq q\}\quad\rm{for\ }q\in[0,1]

**Empirical Probability Mass Function**

The *empirical probability mass function (PMF)* of :math:`X` is given as

.. math::

    \hat{f}(x)=P(X=x)=\frac{1}{n}\sum_{i=1}^nI(x_i=x)

where

.. math::

    I(x_i=x)=\left\{\begin{array}{lr}1\quad\rm{if\ }x_i=x\\0\quad\rm{if\ }x_i\neq x\end{array}\right.

2.1.1 Measures of Central Tendency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Mean**







2.2 Bivariate Analysis
----------------------









2.3 Multivariate Analysis
-------------------------









2.4 Data Normalization
----------------------









2.5 Normal Distribution
-----------------------