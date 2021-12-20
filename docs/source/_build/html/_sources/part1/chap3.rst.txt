Chapter 3 Categorical Attributes
================================

3.1 Univariate Analysis
-----------------------

Consider a single categorical attribute, :math:`X`, with domain 
:math:`dom(X)=\{a_1,a_2,\cds,a_m\}` comprising :math:`m` symbolic values. 
The data :math:`\D` is an :math:`n\times 1` symbolic data matrix given as

.. math::

    \D=\bp X\\\hline x_1\\x_2\\\vds\\x_n \ep

where each point :math:`x_i\in dom(X)`.

3.1.1 Bernoulli Variable
^^^^^^^^^^^^^^^^^^^^^^^^

When the categorical attribute :math:`X` has domain :math:`\{a_1,a_2\}`, with :math:`m=2`.
We can model :math:`X` as a Bernoulli random variables, which takes on two 
distinct values, 1 and 0, according to the mapping

.. math::

    X(v)=\left\{\begin{array}{lr}1\quad\rm{if\ }v=a_1\\0\quad\rm{if\ }v=a_2\end{array}\right.

The probability mass funciton (PMF) of :math:`X` is given as

.. math::

    P(X=x)=f(x)=\left\{\begin{array}{lr}p_1\quad\rm{if\ }x=1\\p_0\quad\rm{if\ }x=0\end{array}\right.

where :math:`p_1` and :math:`p_0` are the parameters of the distribution, which must satisfy the condition

.. math::

    p_1+p_0=1

Denote :math:`p_1=p`, from which it follows that :math:`p_0=1-p`.
The PMF of Bernoulli random variable :math:`X` can then be written compactly as

.. note::

    :math:`P(X=x)=f(x)=p^x(1-p)^{1-x}`

**Mean and Variance**

The expected value of :math:`X` is given as

.. note::

    :math:`\mu=E[X]=1\cd p+0\cd(1-p)=0`

and the variance of :math:`X` is given as

.. math::

    \sg^2=\rm{var}(X)=E[X^2]-(E[X])^2=(1^2\cd p+0^2\cd (1-p))-p^2=p-p^2

which implies

.. note::

    :math:`\sg^2=p(1-p)`

**Sample Mean and Variance**

The sample mean is given as

.. math::

    \hat\mu=\frac{1}{n}\sum_{i=1}^nx_i=\frac{n_1}{n}=\hat{p}

Let :math:`n_0=n-n_1` denote the number of points with :math:`x_i=0` in the random sample.
The sample variance is given as

.. math::

    \hat\sg^2&=\frac{1}{n}\sum_{i=1}^n(x_i-\hat\mu)^2

    &=\frac{n_1}{n}(1-\hat p)^2+\frac{n-n_1}{n}(0-\hat p)^2=\hat p(1-\hat p)^2+(1-\hat p)\hat p^2

    &= \hat p(1-\hat p)(1-\hat p+\hat p)=\hat p(1-\hat p)

**Bionomial Distribution: Number of Occurrences**

Given the Bernoulli variable :math:`X`, let :math:`\{x_1,x_2,\cds,x_n\}` denote
a random sample of size :math:`n` drawn from :math:`X`.
Let :math:`N` be the random variable denoting the number of occurrences of the
symbol :math:`a_1` (value :math:`X=1`) in the sample.
:math:`N` has a binomial distribution, given as

.. note::

    :math:`f(N=n_1|n,p)=\bp n\\n_1 \ep p^{n_1}(1-p)^{n-n_1}`

:math:`N` is the sum of the :math:`n` independent Bernoulli random variables 
:math:`x_i` IID with :math:`X`, that is, :math:`N=\sum_{i=1}^nx_i`.
The mean or expected number of occurrences of symbol :math:`a_1` is given as

.. note::

    :math:`\dp\mu_N=E[N]=E\bigg[\sum_{i=1}^nx_i\bigg]=\sum_{i=1}^nE[x_i]=\sum_{i=1}^np=np`

Because :math:`x_i` are all independent, the variance of :math:`N` is given as

.. note::

    :math:`\dp\sg_N^2=\rm{var}(N)=\sum_{i=1}^n\rm{var}(x_i)=\sum_{i=1}^np(1-p)=np(1-p)`

3.1.2 Multivariate Bernoulli Variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the general case when :math:`dom(X)=\{a_1,a_2,\cds,a_m\}`, we model 
:math:`X` as an :math:`m`-dimensional Bernoulli random variable 
:math:`\X=(A_1,A_2,\cds,A_m)^T`, where each :math:`A_i` is a Bernoulli variable
with parameter :math:`p_i` denoting the probability of observing symbol 
:math:`a_i`.
However, :math:`X` can assume only one of the symbolic values at any one time.
Thus,

.. math::

    \X(b)=\e_i\rm{\ if\ }v=a_i

where :math:`\e_i` is the :math:`i`-th standard basis vector in :math:`m` dimensions.
The range of :math:`\X` consists of :math:`m` distinct vector values :math:`\{\e_1,\e_2,\cds,\e_m\}`.

The PMF of :math:`\X` is given as

.. math::

    p(\X=\e_i)=f(\e_i)=p_i

where :math:`p_i` is the probability of observing value :math:`a_i`.
These parameters must satisfy the condition

.. math::

    \sum_{i=1}^m p_i=1

The PMF can be written compactly as follows:

.. note::

    :math:`\dp P(\X=\e_i)=f(\e_i)=\prod_{j=1}^m p_i^{e_{ij}}`

Because :math:`e_{ii}=1` and :math:`e_{ij}=0` for :math:`j\neq i`, we can see that, as expected, we have

.. math::

    f(\e_i)=\prod_{j=1}^mp_j^{e_{ij}}=p_1^{e_{i0}}\times\cds p_i^{e_{ii}}\cds
    \times p_m^{e_{im}}=p_1^0\times\cds p_i^1\cds\times p_m^0=p_i

**Mean**

The mean or expected value of :math:`\X` can be obtained as

.. math::

    \mmu=E[\X]=\sum_{i=1}^m\e_if(\e_i)=\sum_{i=1}^m\e_ip_i=\bp 1\\0\\\vds\\0\ep
    p_1+\cds+\bp 0\\0\\\vds\\1\ep p_m=\bp p_1\\p_2\\\vds\\p_m\ep=\p

**Sample Mean**

.. math::

    \hat\mmu=\frac{1}{n}\sum_{i=1}^n\x_i=\sum_{i=1}^m\frac{n_i}{n}\e_i=
    \bp n_1/n\\n_2/n\\\vds\\n_m/n\ep=\bp\hat{p}_1\\\hat{p}_2\\\vds\\\hat{p}_m\ep
    =\hat\p

where :math:`n_i` is the number of occurrences of the vector value :math:`\e_i` 
in the sample, which is equivalent to the number of occurrences of the symnbol 
:math:`a_i`.
Furthermore, we have :math:`\sum_{i=1}^mn_i=n`.

**Covariance Matrix**

.. note::

    :math:`\sg_i^2=\rm{var}(A_i)=p_i(1-p_i)`

.. note::

    :math:`\sg_{ij}=E[A_iA_j]-E[A_i]\cd E[A_j]=0-p_ip_j=-p_ip_j`

which follows from the fact taht :math:`E[A_iA_j]=0`, as :math:`A_i` and :math:`A_j` cannot both be 1 at the same time.

The :math:`m\times m` covariance matrix for :math:`\X` is given as

.. math::

    \Sg=\bp\sg_1^2&\sg_{12}&\cds&\sg_{1m}\\\sg_{21}&\sg_{2}^2&\cds&\sg_{2m}\\
    \vds&\vds&\dds&\vds\\\sg_{1m}&\sg_{2m}&\cds&\sg_m^2\ep=
    \bp p_1(1-p_1)&-p_1p_2&\cds&-p_1p_m\\-p_1p_2&p_2(1-p_2)&\cds&-p_2p_m\\
    \vds&\vds&\dds&\vds\\-p_1p_m&-p_2p_m&\cds&p_m(1-p_m) \ep

Define :math:`\` as the :math:`m\times m` diagonal matrix:

.. math::

    \P=\rm{diag}(\p)=\rm{diag}(p_1,p_2,\cds,p_m)=
    \bp p_1&0&\cds&0\\0&p_2&\cds&0\\\vds&\vds&\dds&\vds\\0&0&\cds&p_m \ep

We can compactly write the covariance matrix of :math:`\X` as

.. math::

    \Sg=\P-\p\cd\p^T

**Sample Covariance Matrix**

.. math::

    \hat\Sg=\hat\P-\hat\p\cd\hat\p^T

where :math:`\hat\P=\rm{diag}(\hat\p)`, and 
:math:`\hat\p=\hat\mmu=(\hat{p}_1,\hat{p}_2,\cds,\hat{p}_m)^T` denotes the 
empirical probability mass function for :math:`\X`.

**Multinomial Distribution: Number of Occurrences**

Let :math:`\{\x_1,\x_2,\cds,\x_n\}` drawn from :math:`\X`.
Let :math:`N_i` be the random variable corresponding to the number of 
occurrences of symbol :math:`a_i` in the sample, and let 
:math:`\N=(N_1,N_2,\cds,N_m)^T` denote the vector random variable 
corresponding to the joint distribution of the number of occurrences over all
the symbols.
Then :math:`\N` has a multinomial distribution, given as

.. note::

    :math:`\dp f(\N=(n_1,n_2,\cds,n_m)|\p)=\bp n\\n_1n_2\cds n_m \ep\prod_{i=1}^mp_i^{n_i}`

The term

.. math::

    \bp n\\n_1n_2\cds n_m \ep=\frac{n!}{n_1!n_2!\cds n_m!}

denotes the number of ways of choosing :math:`n_i` occurrences of each symbol 
:math:`a_i` from a sample of size :math:`n`, with :math:`\sum_{i=1}^mn_i=n`.

The mean of :math:`\N` is given as

.. math::

    \mmu_\N=E[\N]=nE[\X]=n\cd\mmu=n\cd\p=\bp np_1\\\vds\\np_m \ep

and its covariance matrix is given as

.. math::

    \Sg_\N=n\cd(\P=\p\p^T)=
    \bp np_1(1-p_1)&-np_1p_2&\cds&-np_1p_m\\-np_1p_2&np_2(1-p_2)&\cds&-np_2p_m\\
    \vds&\vds&\dds&\vds\\-np_1p_m&-np_2p_m&\cds&np_m(1-p_m) \ep

The sample mean and covariance matrix for :math:`\N` are given as

.. math::

    \hat\mmu_\N=n\hat\p\quad\hat\Sg_\N=n(\hat\P-\hat\p\hat\p^T)

3.2 Bivariate Analysis
----------------------

Assume that the data comprises two categorical attributes, :math:`X_1` and :math:`X_2`, with

.. math::

    dom(X_1)=\{a_{11},a_{12},\cds,a_{1m_1}\}

    dom(X_2)=\{a_{21},a_{22},\cds,a_{2m_2}\}

The dataset is now an :math:`n\times 2` symbolic data matrix:

.. math::

    \D=\bp X_1&X_2\\\hline x_{11}&x_{12}\\x_{21}&x_{22}\\\vds&\vds\\x_{n1}&x_{n2} \ep

We model :math:`X_1` and :math:`X_2` as multivariate Bernoulli variables 
:math:`\X_1` and :math:`\X_2` with dimensions :math:`m_1` and :math:`m_2`.
The probability mass funcitons for :math:`\X_1` and :math:`\X_2` are given as

.. math::

    P(\X_1=\e_{1i})=f_1(\e_{1i})=p_i^1=\prod_{k=1}^{m1}(p_i^1)^{e_{ik}^1}

    P(\X_2=\e_{2j})=f_2(\e_{2j})=p_j^2=\prod_{k=1}^{m2}(p_j^2)^{e_{jk}^2}

where

.. math::

    \sum_{i=1}^{m1}p_i^1=1\quad\rm{and}\quad\sum_{j=1}^{m2}p_j^2=1

The joint distribution of :math:`\X_1` and :math:`\X_2` is modeled as the 
:math:`d\pr=m_1+m_2` dimensional vector variable :math:`\X=\bp \X_1,\X_2 \ep`

.. math::

    \X((v_1,v_2)^T)=\bp \X_1(v_1)\\\X_2(v_2) \ep=\bp \e_{1i}\\\e_{2j} \ep

provided that :math:`v_1=a_{1i}` and :math:`v_2=a_{2j}`.
The joint PMF of :math:`\X` is given as

.. math::

    P(\X=(\e_{1i},\e_{2j})^T)=f(\e_{1i},\e_{2j})=p_{ij}=\prod_{r=1}^{m1}\prod_{s=1}^{m2}p_{ij}^{e_{ir}^1\cd e_{is}^2}

where :math:`p_{ij}` the probability of observing the symbol pair :math:`(a_{1i},a_{2j})`.
The probability paramemters must satisfy :math:`\sum_{i=1}^{m1}\sum_{j=1}^{m2}p_{ij}=1`.
The joint PMF for :math:`\X` can be expressed as the :math:`m_1\times m_2` matrix

.. math::

    \P_{12}=\bp p_{11}&p_{12}&\cds&p_{1m_2}\\p_{21}&p_{22}&\cds&
    p_{2m_2}\\\vds&\vds&\dds&\vds\\p_{m_11}&p_{m_12}&\cds&p_{m_1m_2} \ep

**Mean**

.. math::

    \mmu=E[\X]=E\bigg[\bp\X_1\\\X_2\ep\bigg]=\bp E[\X_1]\\E[\X_2] \ep=\bp\mmu_1\\\mmu_2\ep=\bp\p_1\\\p_2\ep

**Sample Mean**

.. math::

    \hat\mmu=\frac{1}{n}\sum_{i=1}^n\x_i=
    \frac{1}{n}\bp\sum_{i=1}^{m_1}n_i^1\e_{1i}\\\sum_{j=1}^{m_2}n_j^2\e_{2j}\ep
    =\frac{1}{n}\bp n_1^1\\\vds\\n_{m_1}^1\\n_1^2\\\vds\\n_{m_2}^2 \ep=
    \bp\hat{p}_1^1\\\vds\\\hat{p}_{m_1}^1\\\hat{p}_1^2\\\vds\\\hat{p}_{m_2}^2\ep
    =\bp\hat\p_1\\\hat\p_2\ep=\bp\hat\mmu_1\\\hat\mmu_2\ep

**Covariance Matrix**

The covariance matrix for :math:`\X` is the :math:`d\pr\times d\pr=(m_1+m_2)\times(m_1+m_2)` matrix given as

.. math::

    \Sg=\bp \Sg_{11}&\Sg_{12}\\\Sg_{12}^T&\Sg_{22} \ep

    \Sg_{11}=\P_1-\p_1\p_1^T

    \Sg_{22}=\P_2-\p_2\p_2^T

.. math::

    \Sg_{12}&=E[(\X_1-\mmu_1)(\X_2-\mmu_2)^T]

    &=E[\X_1\X_2^T]-E[\X_1]E[\X_2]^T

    &=\P_{12}-\mmu_1\mmu_2^T

    &=\P_{12}-\p_1\p_2^T

.. math::

    =\bp p_{11}-p_1^1p_1^2&p_{12}-p_1^1p_2^2&\cds&p_{1m_2}-p_1^1p_{m_2}^2\\
    p_{21}-p_1^2p_1^2&p_{22}-p_2^1p_2^2&\cds&p_{2m_2}-p_2^1p_{m_2}^2\\
    \vds&\vds&\dds&\vds\\p_{m_11}-p_{m_1}^1p_1^2&p_{m_12}-p_{m_1}^1p_2^2&\cds
    &p_{m_1m_2}-p_{m_1}^1p_{m_2}^2 \ep

Each row and each column of :math:`\Sg_{12}` sums to zero.
Consider row :math:`i` and column :math:`j`:

.. math::

    \sum_{k=1}^{m_2}(p_{ik}-p_i^1p_k^2)=\bigg(\sum_{k=1}^{m_2}p_{ik}\bigg)-p_i^1=p_i^1-p_i^1=0

    \sum_{k=1}^{m_1}(p_{kj}-p_k^1p_j^2)=\bigg(\sum_{k=1}^{m_1}p_{kj}\bigg)-p_k^2=p_j^2-p_j^2=0

**Sample Covariance Matrix**

The sample covariance matrix is given as

.. math::

    \hat\Sg=\bp \hat\Sg_{11}&\hat\Sg_{12}\\\hat\Sg_{12}^T&\hat\Sg_{22} \ep

where

.. math::

    \hat\Sg_{11}&=\hat\P_1-\hat\p_1\hat\p_1^T

    \hat\Sg_{22}&=\hat\P_2-\hat\p_2\hat\p_2^T

    \hat\Sg_{12}&=\hat\P_{12}-\hat\p_1\hat\p_2^T

:math:`\hat\P_{12}` specifies the empirical joint PMF for :math:`\X_1` and :math:`\X_2`, given as

.. math::

    \hat\P_{12}(i,j)=\hat{f}(\e_{1i},\e_{2j})=
    \frac{1}{n}\sum_{k=1}^nI_{ij}(\x_k)=\frac{n_{ij}}{n}=\hat{p}_{ij}

where :math:`I_{ij}` is the indicator variable

.. math::

    I_{ij}(\x_k)=\left\{\begin{array}{lr}1\quad\rm{if\ }x_{k1}=\e_{1i}
    \rm{\ and\ }\x_{k2}=\e_{2j}\\0\quad\rm{otherwise}\end{array}\right.

3.2.1 Attribute Dependence: Contingency Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Testing for the independence of the two categorical random variables :math:`X_1`
and :math:`X_2` can be down via *contingency table analysis*.

**Contingency Table**

A contingency table for :math:`\X_1` and :math:`\X_2` is the 
:math:`m_1\times m_2` matrix of observed counts :math:`n_{ij}` for all pairs of 
values :math:`(\e_{1i},\e_{2j})` in the given sample of size :math:`n`, defined 
as

.. math::

    \N_{12}=n\cd\hat\P_{12}=\bp n_{11}&n_{12}&\cds&n_{1m_2}\\
    n_{21}&n_{22}&\cds&n_{2m_2}\\\vds&\vds&\dds&\vds\\
    n_{m_11}&n_{m_12}&\cds&n_{m_1m_2} \ep

where :math:`\hat\P_{12}` is the empirical joint PMF for :math:`\X_1` and :math:`\X_2`.
The contingency table is then augmented with row and column marginal counts, as follows:

.. math::

    \N_1=n\cd\hat\p_1=\bp n_1^1\\\vds\\n_{m_1}^1\ep\quad\N_2=n\cd\hat\p_2=\bp n_1^2\\\vds\\n_{m_2}^2 \ep

Note that the marginal row and column entries and the sample size satisfy the following constraints:

.. math::

    n_i^1=\sum_{j=1}^{m_2}n_{ij}\quad n_j^2=\sum_{i=1}^{m_1}n_{ij}\quad n=
    \sum_{j=1}^{m_1}n_i^1=\sum_{j=1}^{m_2}n_j^2=
    \sum_{i=1}^{m_1}\sum_{j=1}^{m_2}n_{ij}

It is worth noting that both :math:`\N_1` and :math:`\N_2` have a multinomial 
distribution with parameters :math:`\p_1=(p_1^1,\cds,p_{m_1}^1)` and
:math:`\p_2=(p_1^2,\cds,p_{m_2}^2)`, respectively.
Further, :math:`\N_{12}` also has a multinomial distribution with parameters
:math:`\P_{12}=\{p_{ij}\}`, for :math:`1\leq i\leq m_i` and 
:math:`1\leq j\leq m_2`.

:math:`\bs{\chi^2}` **Statistic and Hypothesis Testing**

Under the null hypothesis :math:`\X_1` and :math:`\X_2` are assumed to be 
independent, which means that their joint probability mass function is given as

.. math::

    \hat{p}_{ij}=\hat{p}_i^1\cd\hat{p}_j^2

Under this independence assumption, the expected frequency for each pair of values is given as

.. math::

    e_{ij}=n\cd\hat{p}_{ij}=n\cd\hat{p}_i^1\cd\hat{p}_j^2=n\cd\frac{n_i^1}{n}\cd\frac{n_j^2}{n}=\frac{n_i^1n_j^2}{n}

The :math:`\chi^2` statistic quantifies the difference between observed and 
expected counts for each pair of values; it is defined as follows:

.. note::

    :math:`\dp\chi^2=\sum_{i=1}^{m_1}\sum_{j=1}^{m_2}\frac{(n_{ij}-e{ij})^2}{e_{ij}}`

For the :math:`\chi^2` statistic it is known that its sampling distribution 
follows the *chi-squared* density function with :math:`q` degrees of freedom:

.. math::

    f(x|q)=\frac{1}{2^{q/2}\Gamma(q/2)}x^{\frac{q}{2}-1}e^{-\frac{x}{2}}

where the gamma function :math:`\Gamma` is defined as

.. math::

    \Gamma(k>0)=\int_0^\infty x^{k-1}e^{-x}dx

The total degrees of freedom is

.. math::

    q&=|dom(X_1)|\times|dom(X_2)|-(|dom(X_1)|+|dom(X_2)|)+1

    &=m_1m_2-m_1-m_2+1

    &=(m_1-1)(m_2-1)

**p-value**

The *p-value* of a statistic is defined as the probability of obtaining a value 
at least as extreme as the observed value under the null hypothesis.
For the :math:`\chi^2` statistic computed above, its p-value is defined as follows

.. note::

    p-value\ :math:`(\chi^2)=P(x\geq\chi^2)=1-F_1(\chi^2)`

where :math:`F_q` is the cumulative :math:`\chi^2` probability distribution with :math:`q` degrees of freedom.

The null hypothesis is rejected if the p-value is below some *significance level*, :math:`\alpha`.
The value :math:`1-\alpha` is also called the *confidence level*.

For a given significance level :math:`\alpha` (or equivalently, confidence level 
:math:`1-\alpha`), define the corresponding *critical value*, :math:`v_\alpha`,
of the test statistic as follows:

.. math::

    P(x\geq v_\alpha)=1-F_q(v_\alpha)=\alpha,\rm{\ or\ equivalently\ }F_q(v_\alpha)=1-\alpha

For the given significance value :math:`\alpha`, we can find the critical value from the quantile funtion :math:`F_q\im`:

.. math::

    v_\alpha=F_q\im(1-\alpha)

An alternative test for rejection of the null hypothesis is to check if 
:math:`\chi^2\geq v_\alpha`, as in that case 
:math:`P(x\geq\chi^2)\leq P(x\geq v_\alpha)`, and therefore, the p-value of the
observed :math:`\chi^2` value is bounded above by :math:`\alpha`, that is, 
p-value\ :math:`(\chi^2)\leq` p-value\ :math:`(v_\alpha)=\alpha`.

3.3 Multivariate Analysis
-------------------------

For an :math:`n\times d` symbolic matrix

.. math::

    \D=\bp X_1&X_2&\cds&X_d\\\hline x_{11}&x_{12}&\cds&x_{1d}\\
    x_{21}&x_{22}&\cds&x_{2d}\\\vds&\vds&\dds&\vds\\x_{n1}&x_{n2}&\cds&x_{nd}\ep

The joint distribution is modeled as a :math:`d\pr=\sum_{j=1}^dm_j` dimensional vector random variable

.. math::

    \X=\bp \X_1\\\vds\\\X_d \ep

Each categorical data point :math:`\v=(v_1,v_2,\cds,v_d)^T` is represented as a :math:`d\pr`\ -dimensional binary vector

.. math::

    \X(\v)=\bp\X_1(v_1)\\\vds\\\X_d(v_d)\ep=\bp\e_{1k_1}\\\vds\\\e_{dk_d}\ep

provided :math:`v_i=a_{ik_i}`, the :math:`k_i`\ th symbol of :math:`X_i`.

**Mean**

The mean and sample mean for :math:`\X` are given as

.. math::

    \mmu=E[\X]=\bp\mmu_1\\\vds\\\mmu_d\ep=\bp\p_1\\\vds\\\p_d\ep\quad
    \hat\mmu=\bp\hat\mmu_1\\\vds\\\hat\mmu_d\ep=\bp\hat\p_1\\\vds\\\hat\p_d\ep

The covariance matrix for :math:`\X`, and its estimate from the sample, are 
given as the :math:`d\pr\times d\pr` matrices:

.. math::

    \Sg=\bp\Sg_{11}&\Sg_{12}&\cds&\Sg_{1d}\\\Sg_{12}^T&\Sg_{12}&\cds&\Sg_{2d}\\
    \vds&\vds&\dds&\vds\\\Sg_{1d}^T&\Sg_{2d}^T&\cds&\Sg_{dd}\ep\quad
    \hat\Sg=\bp\hat\Sg_{11}&\hat\Sg_{12}&\cds&\hat\Sg_{1d}\\
    \hat\Sg_{12}^T&\hat\Sg_{12}&\cds&\hat\Sg_{2d}\\\vds&\vds&\dds&\vds\\
    \hat\Sg_{1d}^T&\hat\Sg_{2d}^T&\cds&\hat\Sg_{dd}\ep

where

.. math::

    \Sg_{ij}=\P_{ij}-\p_i\p_j^T\quad\hat\Sg_{ij}=\hat\P_{ij}-\hat\p_i\hat\p_j^T

3.3.1 Multiway Contingency Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The empirical joint probability mass function for :math:`\X` is

.. math::

    \hat{f}(\e_{1i_1},\e_{2i_2},\cds,\e_{di_d})=
    \frac{1}{n}\sum_{k=1}^nI_{i_1i_2\cds i_d}(\x_k)=
    \frac{n_{i_1i_2\cds i_d}}{n}=\hat{p}_{i_1i_2\cds i_d}

where :math:`I_{i_1i_2\cds i_d}` is the indicator variable

.. math::

    I_{i_1i_2\cds i_d}(\x_k)=\left\{\begin{array}{lr}1\quad\rm{if\ }
    x_{k1}=\e_{1i_1},x_{k2}=\e_{2i_2},\cds,x_{kd}=\e_{di_d}\\
    0\quad\rm{otherwise}\end{array}\right.

Using the notation :math:`\i=(i_1,i_2,\cds,i_d)` to denote the index tuple, we
can write the joint empirical PMF as the :math:`d`\ -dimensional matrix
:math:`\hat\P` of size 
:math:`m_1\times m_2\times\cds\times m_d=\prod_{i=1}^dm_i`, given as

.. math::

    \hat\P(\i)=\{\hat{p}_\i\}\rm{\ for\ all\ index\ tuples\ }\i,\rm{\ with\ }1\leq i_1\leq m_1,\cds,1\leq i_d\leq m_d

where :math:`\hat{p}_\i=\hat{p}_{i_1i_2\cds i_d}`.
The :math:`d`\ -dimensional contingency table is then given as

.. math::

    \N=n\times\hat\P=\{n_\i\}\rm{\ for\ all\ index\ tuples\ }\i,\rm{\ with\ }1\leq i_1\leq m_1,\cds,1\leq i_d\leq m_d

where :math:`n_\i=n_{i_1i_2\cds i_d}`.
The contingency table is augmented with the marginal count vectors :math:`\N_i` 
for all :math:`d` attributes :math:`\X_i`:

.. math::

    \N_i=n\hat\p_i=\bp n_1^i\\\vds\\n_{m_i}^i \ep

:math:`\bs{\chi^2}`\ **-Test**

The null hypothesis :math:`H_0` is that the attributes are :math:`d`\ -way independent.
The alternative hypothesis :math:`H_1` is that they are dependent in some way.

The expected number of occurrences of the symbol tuple :math:`(a_{1i_1},a_{2i_2},\cds,a_{di_d})` is given as

.. math::

    e_\i=n\cd\hat\p_\i=n\cd\prod_{j=1}^d\hat{p}_{i_j}^j=\frac{n_{i_1}^1n_{i_2}^2\cds n_{i_d}^d}{n^{d-1}}

The chi-squared statistic measures the difference between the observed counts 
:math:`n_\i` and the expected count :math:`e_\i`:

.. note::

    :math:`\dp\chi^2=\sum_\i\frac{(n_\i-e_\i)^2}{e_\i}=\sum_{i_1=1}^{m_1}\sum_{i_2=1}^{m_2}\cds\sum_{i_d=1}^{m_d}`
    :math:`\dp\frac{(n_{i_1,i_2,\cds,i_d}-e_{i_1,i_2,\cds,i_d})^2}{e_{i_1,i_2,\cds,i_d}}`

The total number of degrees of freedom is given as

.. math::

    q&=\prod_{i=1}^d|dom(X_i)|-\sum_{i=1}^d|dom(X_i)|+(d-1)

    &=\bigg(\prod_{i=1}^dm_i\bigg)-\bigg(\sum_{i=1}^dm_i\bigg)+d-1

3.4 Distance and Angle
----------------------

With the modeling of categorical attributes as multivariate Bernoulli variables, 
it is possible to compute the distance or the angle between any two points 
:math:`\x_i` and :math:`\x_j`:

.. math::

    \x_i=\bp\e_{1i_1}\\\vds\\\e_{di_d}\ep\quad\x_j=\bp\e_{1j_1}\\\vds\\\e_{dj_d}\ep

The number of matching values :math:`s`

.. math::

    s=\x_i^T\x_j=\sum_{k=1}^d(\e_{ki_k)^T\e_{kj_k)

The norm of each point

.. math::

    \lv\x_i\rv^2=\x_i^T\x_i=d

**Euclidean Distance**

.. math::

    \lv\x_i-\x_j\rv=\sqrt{\x_i^T\x_i-2\x_i\x_j+\x_j^T\x_j}=\sqrt{2(d-s)}

**Hamming Distance**

.. math::

    \delta_H(\x_i,\x_j)=d-s=\frac{1}{2}\lv\x_i-\x_j\rv^2

**Cosine Similarity**

.. math::

    \cos\th=\frac{\x_i^T\x_j}{\lv\x_i\rv\cd\lv\x_j\rv}=\frac{s}{d}

**Jaccard Coefficient**

The *Jaccard Coefficient* is defined as the ratio of the number of matching
values to the number of distinct values that appear in both :math:`\x_i` and
:math:`\x_j`, across the :math:`d` attributes:

.. math::

    J(\x_i,\x_j)=\frac{s}{2(d-s)+s}=\frac{s}{2d-s}

3.5 Discretization
------------------

*Discretization*, also called *binning*, converts numeric attributes into categorical ones.
Formally, given a numeric attribute :math:`X`, and a random sample 
:math:`\{x_i\}_{i=1}^n` of size :math:`n` drawn from :math:`X`, the 
discretization task is to divide the value range of :math:`X` into :math:`k`
consecutive intervals, also called *bins*, by finding :math:`k-1` boundary
values :math:`v_1,v_2,\cds,v_{k-1}` that yield the :math:`k` intervals:

.. math::

    [x_\min,v_1],(v_1,v_2],\cds,(v_{k-1},x_\max]

where the extremes of the range of :math:`X` are given as

.. math::

    x_\min=\min_i\{x_i\}\quad x_\max=\max_i\{x_i\}

**Equal-Width Intervals**

.. math::

    w=\frac{x_\max-x_\min}{k}

The :math:`i`\ th interval boundary is given as

.. math::

    v_i=x_\min+iw,\rm{\ for\ }i=1,\cds,k-1

**Equal-Frequency Intervals**

.. math::

    v_i=\hat{F}\im(i/k)\rm{\ for\ }i=1,\cds,k-1