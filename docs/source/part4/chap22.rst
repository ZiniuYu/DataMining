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

