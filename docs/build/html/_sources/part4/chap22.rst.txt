Chapter 22 Classification Assessment
====================================

We may think of the classifier as a model or function :math:`M` that predicts
the class label :math:`\hat{y}` for a given input example :math:`\x`:

.. math::

    \hat{y}=M(\x)

where :math:`\x=(x_1,x_2,\cds,x_d)^T\in\R^d` is a point in :math:`d`-dimensional 
space and :math:`\hat{y}=\{c_1,c_2,\cds,c_k\}` is its predicted class.

To build the classification model :math:`M` we need a *training set* of points along with their known classes.
Different classifiers are obtained depending on the assumptions used to build the model :math:`M`.
Once the model :math:`M` has been trained, we assess its performance over a 
separate *testing set* of points for which we know the true classes.
Finally, the model can be deployed to predict the class for future points whose class we typically do not know.

22.1 Classification Performance Measures
----------------------------------------

Let :math:`\D` be the testing set comprising :math:`n` points in a :math:`d` 
dimensional space, let :math:`\{c_1,c_2,\cds,c_k\}` denote the set of :math:`k`
class labels, and let :math:`M` be a classifier.
For :math:`\x_i\in\D`, let :math:`y_i` denote its true class, and let 
:math:`\hat{y_i}=M(\x_i)` denote its predicted class.

**Error Rate**

.. note::

    :math:`\dp Error\ Rate=\frac{1}{n}\sum_{i=1}^nI(y_i\ne\hat{y_i})`

**Accuracy**

.. note::

    :math:`\dp Accuracy=\frac{1}{n}\sum_{i=1}^nI(y_i=\hat{y_i})=1-Error\ Rate`

22.1.1 Contingency Table-based Measures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\cl{D}=\{\D_1,\D_2,\cds,\D_k\}` denote a partitioning of the testing 
points based on their true class labels, where

.. math::

    \D_j=\{\x_i^T|y_i=c_j\}\quad\rm{and}\quad n_i=|\D_i|

Let :math:`\cl{R}=\{\bs{R}_1,\bs{R}_2,\cds,\bs{R}_k\}` denote a partitioning of
the testing points based on the predicted labels, that is,

.. math::

    \bs{R}_j=\{\x_i^T|\hat{y_i}=c_j\}\quad\rm{and}\quad m_j=|\bs{R}_j|

The partitionings :math:`\cl{R}` and :math:`\cl{D}` induce a :math:`k\times k` 
contingency table :math:`\N`, also called a *confusion matrix*, defined as 
follows:

.. math::

    \N(i,j)=n_{ij}=|\bs{R}_i\cap\D_j|=|\{\x_a\in\D|\hat{y_a}=c_i\ \rm{and}\ y_a=c_j\}|

where :math:`1\leq i,j\leq k`.

**Accuracy/Precision**

.. note::

    :math:`\dp acc_i=prec_i=\frac{n_{ii}}{m_i}`

where :math:`m_i` is the number of examples predicted as :math:`c_i` by classifier :math:`M`.
The higher the accuracy on class :math:`c_i` the better the classifier.

.. note::

    :math:`\dp Accuracy=Precision=\sum_{i=1}^k\bigg(\frac{m_i}{n}\bigg)acc_i=\frac{1}{n}\sum_{i=1}^kn_{ii}`

**Coverage/Recall**

.. note::

    :math:`\dp coverage_i=recall_i=\frac{n_{ii}}{n_i}`

where :math:`n_i` is the number of points in class :math:`c_i`.
The higher the coverage the better the classifier.

**F-measure**

.. note::

    :math:`\dp F_i=\frac{2}{\frac{1}{prec_i}+\frac{1}{recall_i}}=\frac{2\cd prec_i\cd recall_i}{prec_i+recall_i}`
    :math:`\dp=\frac{2n_{ii}}{n_i+m_i}`

The higher the :math:`F_i` value the better the classifier.

.. note::

    :math:`\dp F=\frac{1}{k}\sum_{i=1}^rF_i`

For a perfect classifier, the maximum value of the F-measure is 1.

22.1.2 Binary Classification: Positive and Negative Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When there are only :math:`k=2` classes, we call class :math:`c_i` the positive 
class and :math:`c_2` the negative class.

* *True Positives (TP)*

.. math::

    TP=n_{11}|\{\x_i|\hat{y_1}=y_1=c_1\}|

* *False Positives (FP)*

.. math::

    FP=n_{12}=|\{\x_i|\hat{y_i}=c_1\ \rm{and}\ y_i=c_2\}|

* *False Negatives (FN)*

.. math::

    FN=n_{21}=|\{\x_i|\hat{y_i}=c_2\ \rm{and}\ y_i=c_1\}|

* *True Negatives (TN)*

.. math::

    TN=n_{22}=|\{\x_i|\hat{y_i}=y_i=c_2\}|

**Error Rate**

.. note::

    :math:`\dp Error\ Rate=\frac{FP+FN}{n}`

**Accuracy**

.. note::

    :math:`\dp Accuracy=\frac{TP+TN}{n}`

**Class-specific Precision**

.. note::

    :math:`\dp prec_P=\frac{TP}{TP+FP}=\frac{TP}{m_1}`

    :math:`\dp prec_N=\frac{TN}{TN+FN}=\frac{TN}{m_2}`

where :math:`m_i=|\bs{R}_i|` is the number of points predicted by :math:`M` as having class :math:`c_i`.

**Sensitivity: True Positive Rate**

.. note::

    :math:`\dp TPR=recall_P=\frac{TP}{TP+FN}=\frac{TP}{n_1}`

where :math:`n_1` is the size of the positive class.

**Specificity: True Negative Rate**

.. note::

    :math:`\dp TNR=specificity=recall_N=\frac{TN}{FP+TN}=\frac{TN}{n_2}`

where :math:`n_2` is the size of the negative class.

**False Negative Rate**

.. note::

    :math:`FNR=\frac{FN}{TP+FN}=\frac{FN}{n_1}=1-sensitivity`

**False Positive Rate**

.. note::

    :math:`FPR=\frac{FP}{FP+TN}=\frac{FP}{n_2}=1-specificity`

22.1.3 ROC Analysis
^^^^^^^^^^^^^^^^^^^

Receiver Operating Characteristic (ROC) analysis is a popular strategy for 
assessing the performance of classifiers when there are two classes.

Typically, a binary calssifier chooses some positive score threshold 
:math:`\rho`, and classifies all points with score above :math:`\rho` as 
positive, with the remaining points classified as negative.
ROC analysis plots the performance of the classifier over all possible values of 
the threshold parameter :math:`\rho`.
In particular, for each value of :math:`\rho`, it plots the false positive rate
on the :math:`x`-axis versus the true positive rate on the :math:`y`-axis.
The resulting plot is called the *ROC curve* or *ROC plot* for the classifier.

Let :math:`S(\x_i)` denote the real-valued score for the positive class output
by a classifier :math:`M` for the point :math:`\x_i`.
Let the maximum and minimum score thresholds observed on testing dataset :math:`\D` be as follows:

.. math::

    \rho^\min=\min_i\{S(\x_i)\}\quad\quad\rho^\max=\max_i\{S(\x_i)\}

Initially, we classify all points as negative.
Both *TP* and *FP* are thus initially zero, resulting in *TPR* and *FPR* rates 
of zero, which correspond to the point (0,0) at the lower left corner in 
the ROC plot.
Next for each distinct value of :math:`\rho` in the range 
:math:`[\rho^\min,\rho^\max]`, we tabulate the set of positive points:

.. math::

    \bs{R}_1(\rho)=\{\x_i\in\D:S(\x_i)>\rho\}

and we compute the corresponding true and false positive rates, to obtain a new point in the ROC plot.
Finally, in the last step, we classify all points as positive.
Both *FN* and *TN* are thus zero, resulting in *TPR* and *FPR* values of 1.
This results in the point (1,1) at the top right-hand corner in the ROC plot.
An ideal classifier corresponds to the top left point (0,1), which correspoinds
to the case :math:`FPR=0` and :math:`TPR=1`, that is, the classifier has no 
false positives, and identifies all true positives.

As such, a ROC curve indicates the extent to which the classifier ranks positive 
instances higher than the negative instances.
An ideal classifier should score all positive points higher than any negative point.
Thus, a classifier with a curve closer to the ideal case, that is, closer to the 
upper left corner, is a better classifier.

**Area Under ROC Curve**

Because the total area of the plot is 1, the AUC lies in the interval :math:`[0,1]` - the higher the better.
The AUC value is essentially the probability that the classifier will rank a 
random positive test case higher than a random negative test instance.

**ROC/AUC Algorithm**

.. image:: ../_static/Algo22.1.png

**Random Classifier**

A random classifier corresponds to a diagonal line in the ROC plot.
It follows that if the ROC curve for any classifier is below the diagonal, it 
indicates performance worse than random guessing.
For such cases, inverting the class assignment will produce a better classifier.

**Class Imbalance**

It is worth remarking that ROC curves are insensitive to class skew. 
This is because the *TPR*, interpreted a s the probability of predicting a 
positive point as positive, and the *FPR*, interpreted as the probability of 
predicting a negative point as positive, do not depend on the ratio of the 
positive to negative class size.

22.2 Classifier Evaluation
--------------------------

The input dataset :math:`\D` is randomly split into a disjoint training set and testing set.
The training set is used to learn the model :math:`M`, and the testing set is
used to evaluate the measure :math:`\theta`.

22.2.1 :math:`K`-fold Cross-Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cross-validation divides the dataset :math:`\D` into :math:`K` equal-sized 
parts, called *folds*, namely :math:`\D_1,\D_2,\cds,\D_k`.
Each fold :math:`\D_i` is, in turn, treated as the testing set, with the 
remaining folds comprising the training set 
:math:`\D\backslash\D_i=\bigcup_{j\ne i}\D_j`.
After training the model :math:`M_i` on :math:`\D\backslash\D_i`, we assess its
performance on the testing set :math:`\D_i` to obtain the :math:`i`\ th estimate
:math:`\th_i`.
The expected value of the performance measure can then be estimated as

.. math::

    \hat{\mu_\th}=E[\th]=\frac{1}{K}\sum_{i=1}^K\th_i

and its variance as

.. math::

    \hat{\sg_\th}^2=\frac{1}{K}\sum_{i=1}^K(\th_i-\hat{\mu_\th})^2

.. image:: ../_static/Algo22.2.png

Usually :math:`K` is chosen to be 5 or 10.
The special case, when :math:`K=n`, is called *leave-one-out* cross-validation, 
where the tseting set comprises a single point and the remaining data is used 
for training purposes.

22.2.2 Bootstrap Resampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The bootstrap method draws :math:`K` random samples of size :math:`n` *with replacement* from :math:`\D`.
Each sample :math:`\D_i` is thus the same size as :math:`\D`, and has several repeated points.
The probability that a point is selected is given as :math:`p=\frac{1}{n}`, and 
thus the probability that it is not selected is

.. math::

    q=1-p=\bigg(1-\frac{1}{n}\bigg)

Because :math:`\D_i` has :math:`n` points, the probability that :math:`\x_j` is 
not selected even after :math:`n` tries is given as

.. math::

    P(\x_j\notin\D_i)=q^n=\bigg(1-\frac{1}{n}\bigg)^n\simeq e\im=0.368

On the other hand, the probability that :math:`\x_j\in\D_i` is given as

.. math::

    P(\x_j\in\D_i)=1-P(\x_j\notin\D_i)=1-0.368=0.632

This means that each bootstrp sample contains approximately 63.2% of the points from :math:`\D`.

.. image:: ../_static/Algo22.3.png

22.2.3 Confidence Intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sum of a large number of independent and identically distributed (IID) 
random variables has approximately a normal distribution, regardless of the
distribution of the individual random variables.
Let :math:`\th_1,\th_2,\cds,\th_K` be a sequence of IID random variables,
representing, for example, the error rate or some other performance measure over
the :math:`K`-folds in cross-validation or :math:`K` bootstrap samples.
Assume that each :math:`\th_i` has a finite mean :math:`E[\th_i]=\mu` and finite
variance :math:`\rm{var}(\th_i)=\sg^2`.

.. math::

    \hat{\mu}=\frac{1}{K}(\th_1+\th_2+\cds+\th_K)

.. math::

    E[\hat{mu}]=E\bigg[\frac{1}{K}(\th_1+\th_2+\cds+\th_K)\bigg]=\frac{1}{K}\sum_{i=1}^KE[\th_i]=\frac{1}{K}(K\mu)=\mu

.. math::

    \rm{var}(\hat{\mu})=var\bigg(\frac{1}{K}(\th_1+\th_2+\cds+\th_K)\bigg)=
    \frac{1}{K^2}\sum_{i=1}^K\rm{var}(\th_i)=\frac{1}{K^2}(K\sg^2)=
    \frac{\sg^2}{K}

.. math::

    std(\hat{\mu})=\sqrt{\rm{var}(\hat{\mu})}=\frac{\sg}{\sqrt{K}}

.. math::

    Z_K=\frac{\hat{\mu}-E[\hat{\mu}]}{std(\hat{\mu})}=\frac{\hat{\mu}-\mu}
    {\frac{\sg}{\sqrt{K}}}=\sqrt{K}\bigg(\frac{\hat{\mu}-\mu}{\sg}\bigg)

:math:`Z_K` specifiese the deviation of the estimated mean from the true mean in terms of its standard deviation.
The central limit theorem states that, as the sample size increases, the random 
variable :math:`Z_K` *converges in distribution* to the standard normal 
distribution.
That is, as :math:`K\rightarrow\infty`, for any :math:`x\in\R`, we have

.. math::

    \lim_{k\rightarrow\infty}P(Z_K\leq x)=\Phi(x)

where :math:`\Phi(x)` is the cumulative distribution function for the standard normal density function :math:`f(x|0,1)`.
Given significance level :math:`\alpha\in(0,1)`, let :math:`z_{\alpha/w}` denote 
the critical :math:`z`-score value for the standard normal distribution that 
encompasses :math:`\alpha/2` of the probability mass in the right tail, defined 
as

.. math::

    P(Z_K\geq z_{\alpha/w})=\frac{\alpha}{2},\rm{or\ equivalently\ }
    \Phi(z_{\alpha/2})=P(Z_K\leq z_{\alpha/2})=1-\frac{\alpha}{2}

Also, because the normal distribution is symmetric about the mean, we have

.. math::

    P(Z_K\geq -z_{\alpha/2})=1-\frac{\alpha}{2},\rm{or\ equivalently\ }\Phi(-z_{\alpha/2})=\frac{\alpha}{2}

Thus, given confidence level :math:`1-\alpha`, we can find the lower and upper 
critical :math:`z`-score values, so as to encompass :math:`1-\alpha` fraction of
the probability mass, which is given as

.. math::

    P(-z_{\alpha/2}\leq Z_K\leq z_{\alpha/2})=\Phi(z_{\alpha/2})-
    \Phi(-z_{\alpha/2})=1-\frac{\alpha}{2}-\frac{\alpha}{2}=1-\alpha

Note that

.. math::

    -z_{\alpha/2}\leq Z_K\leq z_{\alpha/2}&\Rightarrow -z_{\alpha/2}\leq\sqrt{K}
    \bigg(\frac{\hat{\mu}-\mu}{\sg}\bigg)\leq z_{\alpha/2}

    &\Rightarrow -z_{\alpha/2}\frac{\sg}{\sqrt{K}}\leq\hat{\mu}-\mu\leq z_{\alpha/2}\frac{\sg}{\sqrt{K}}

    &\Rightarrow \bigg(\hat{\mu}-z_{\alpha/2}\frac{\sg}{\sqrt{K}}\bigg)\leq\mu
    \leq\bigg(\hat{\mu}+z_{\alpha/2}\frac{\sg}{\sqrt{K}}\bigg)

.. note::

    :math:`\dp P\bigg(\hat{\mu}-z_{\alpha/2}\frac{\sg}{\sqrt{K}}\leq\mu`
    :math:`\dp\leq\hat{\mu}+z_{\alpha/2}\frac{\sg}{\sqrt{K}}\bigg)=1-\alpha`

Thus, for any given level of confidence :math:`1-\alpha`, we can compute the
corresponding :math:`100(1-\alpha)\%` confidence interval 
:math:`(\hat{\mu}-z_{\alpha/2}\frac{\sg}{\sqrt{K}},`
:math:`\hat{\mu}+z_{\alpha/2}\frac{\sg}{\sqrt{K}})`.

**Unknown Variance**

We can replace :math:`\sg^2` by the sample variance

.. math::

    \hat{\sg}^2=\frac{1}{K}\sum_{i=1}^K(\th_i\hat{\mu})^2

because :math:`\hat{\sg}^2` is a *consistent* estimator for :math:`\sg^2`, that 
is, as :math:`K\rightarrow\infty`, :math:`\hat{\sg}^2` converges with 
probability 1, also called *converges almost surely*, to :math:`\sg^2`.
The central limit theorem then states that the random variable :math:`Z_K^*`
defined below converges in distribution to the standard normal distribution:

.. math::

    Z_K^*=\sqrt{K}\bigg(\frac{\hat{\mu}-\mu}{\hat{\sg}}\bigg)

.. note::

    :math:`\dp\lim_{K\rightarrow\infty}P\bigg(\hat{\mu}-z_{\alpha/2}\frac{\hat{\sg}}{\sqrt{K}})`
    :math:`\dp\leq\mu\leq\hat{\mu}-z_{\alpha/2}\frac{\hat{\sg}}{\sqrt{K}}\bigg)=1-\alpha`

In other words, :math:`(\hat{\mu}-z_{\alpha/2}\frac{\hat{\sg}}{\sqrt{K}},)`
:math:`\hat{\mu}-z_{\alpha/2}\frac{\hat{\sg}}{\sqrt{K}})` is the 
:math:`100(1-\alpha)\%` confidence interval for :math:`\mu`.

**Small Sample Size**

Consider the random variables :math:`V_i`, for :math:`i=1,\cds,K`, defined as

.. math::

    V_i=\frac{\th_i-\hat{\mu}}{\sg}

Further, consider the sum of their squares:

.. math::

    S=\sum_{i=1}^KV_i^2=\sum_{i=1}^K\bigg(\frac{\th_i-\hat{\mu}}{\sg}\bigg)^2=
    \frac{1}{\sg^2}\sum_{i=1}^K(\th_i-\hat{\mu})^2=\frac{K\hat{\sg}^2}{\sg^2}

If we assume that the :math:`V_i`'s are IID with the standard normal 
distribution, then the sum :math:`S` follows a chi-squared distribution with
:math:`K-1` degrees of freedom, denoted :math:`\chi^2(K-1)`, since :math:`S` is
the sum of the squares of :math:`K` random variables :math:`V_i`.
There are only :math:`K-1` degrees of freedom because each :math:`V_i` depends
on :math:`\hat{\mu}` and the sum of the :math:`\th_i`'s is thus fixed.

.. math::

    Z_K^*&=\sqrt{K}\bigg(\frac{\hat{\mu}-\mu}{\hat{\sg}}\bigg)=\bigg(\frac{\hat{\mu}-\mu}{\hat{\sg}/\sqrt{K}}\bigg)

    &=\bigg(\frac{\hat{\mu}-\mu}{\hat{\sg}/\sqrt{K}}\bigg/
    \frac{\hat{\sg}/\sqrt{K}}{\sg/\sqrt{K}}\bigg)=\bigg(
    \frac{\frac{\hat{\mu}-\mu}{\hat{\sg}/\sqrt{K}}}{\hat{\sg}/\sg}\bigg)=
    \frac{Z_K}{\sqrt{S/K}}

Assuming that :math:`Z_K` follows a standard normal distribution, and noting 
that :math:`S` follows a chi-squared distribution with :math:`K-1` degrees of
freedom, then the distribution of :math:`Z_K^*` is precisely the Student's
:math:`t` distribution with :math:`K-1` degrees of freedom.
Thus, in the small sample case, instead of using the standard normal density to 
derive the confidence interval, we use the :math:`t` distribution.
In particular, given confidence level :math:`1-\alpha` we choose the critical
value :math:`t_{\alpha/2}` such that the cumulative :math:`t` distribution 
function with :math:`K-1` degrees of freedom encompasses :math:`\alpha/2` of the
probability mass in the right tail.
That is,

.. math::

    P(Z_K^*\geq t_{\alpha/2})=1-T_{K-1}(t_{\alpha/2})=\alpha/2

.. math::

    P\bigg(\hat{\mu}-t_{\alpha/2}\frac{\hat{\sg}}{\sqrt{K}}\leq\mu\leq
    \hat{\mu}-t_{\alpha/2}\frac{\hat{\sg}}{\sqrt{K}}\bigg)=1-\alpha

The :math:`100(1-\alpha)%` confidence interval for the true mean :math:`\mu` is thus

.. note::

    :math:`\dp\bigg(\hat{\mu}-t_{\alpha/2}\frac{\hat{\sg}}{\sqrt{K}}\leq`
    :math:`\dp\mu\leq\hat{\mu}-t_{\alpha/2}\frac{\hat{\sg}}{\sqrt{K}}\bigg)`

As :math:`K` increases, the :math:`t` distribution very rapidly converges in 
distribution to the standard normal distribution, consistent with the large 
sample case.
Thus, for large samples, we may use the usual :math:`z_{\alpha/2}` threshold.

22.2.4 Comparing Classifiers: Paired :math:`t`-Test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We look at a method that allows us to test for a significant difference in the 
classification performance of two alternative classifiers, :math:`M^A` and 
:math:`M^B`.
We want to assess which of them has a superior classification performance on a given dataset :math:`\D`.
We perform a *paired test*, with both classifiers trained and tested on the same data.
Let :math:`\th_1^A,\th_2^A,\cds,\th_K^A` and 
:math:`\th_1^B,\th_2^B,\cds,\th_K^B` denote the performance values for 
:math:`M^A` and :math:`M^B`, respectively.
To determine if the two classifiers have different or similar performance, 
define the random variable :math:`\delta_i` as the difference in their
performance on the :math:`i`\ th dataset:

.. math::

    \delta_i=\th_i^A-\th_i^B

.. math::

    \hat{\mu_delta}=\frac{1}{K}\sum_{i=1}^K\delta_i\quad\quad\hat{\sg_\delta}^2=
    \frac{1}{K}\sum_{i=1}^K(\delta_i-\hat{\mu_\delta})^2

The null hypothesis :math:`H_0` is that their performance is the same, that is, 
the true expected difference is zero, whereas the alternative hypothesis 
:math:`H_a` is that they are not the same, that is, the true expected difference
:math:`\mu_\delta` is not zero:

.. math::

    H_0: \mu_\delta=0\quad\quad H_a:\mu_\delta\neq 0

.. math::

    Z_\delta^*=\sqrt{K}\bigg(\frac{\hat{\mu_\delta}-\mu_\delta}{\hat{\sg_\delta}}\bigg)

.. note::

    :math:`\dp Z_\delta^*=\frac{\sqrt{K}\hat{\mu_\delta}}{\hat{\sg_\delta}}\sum t_{K-1}`

where the notation :math:`Z_\delta^*\sim t_{K-1}` means that :math:`Z_\delta^*` 
follows the :math:`t` distribution with :math:`K-1` degress of freedom.

Given a desired confidence level :math:`1-\alpha`, we conclude that

.. math::

    P(-t_{\alpha/2}\leq Z_\delta^*\leq t_{\alpha/2})=1-\alpha

.. image:: ../_static/Algo22.4.png