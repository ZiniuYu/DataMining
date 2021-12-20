Chapter 6 High-dimensional Data
===============================

6.1 High-Dimensional Objects
----------------------------

Consider the :math:`n\times d` data matrix

.. math::
    
    \left(\begin{array}{c|cccc}&X_1&X_2&\cds&X_d\\ \hline 
    \x_1&x_{11}&x_{12}&\cds&x_{1d}\\\x_2&x_{21}&x_{22}&\cds&x_{2d}\\ 
    \vds&\vds&\vds&\dds&\vds\\\x_n&x_{n1}&x_{n2}&\cds&x_{nd}\end{array}\right)

where each point :math:`\x\in\R^d` and each attribute :math:`X_j\in\R^n`.

**Hypercube**

let the minimum and amximum values for each attribute :math:`X_j` be given as

.. math::

    \min(X_j)=\min_i\{x_{ij}\}\quad\max(X_j)=\max_i\{x_{ij}\}

The data hyperspace can be considered as a :math:`d`-dimensional *hyper-rectangle*, defined as

.. math::

    R_d&=\prod_{j=1}^d[\min(X_j),\max(X_j)]

    &=\{\x=(x_1,x_2,\cds,x_d)^T|x_j\in[\min(X_j),\max(X_j)],\rm{\ for\ }j=1,\cds,d\}

Assume the data is centered to have mean :math:`\mmu=\0`.
Let :math:`m` denote the largest absolute value in :math:`\D`, given as

.. math::

    m=\max_{j=1}^d\max_{i=1}^n\{|x_{ij}|\}

The data hyperspace can be represented as a *hypercube*, centered at :math:`\0`, 
with all sides of length :math:`l=2m`, given as

.. math::

    H_d(l)=\{\x=(x_1,x_2,\cds,x_d)^T|\forall i,x_i\in[-l/2,l/2]\}

The *unit hypercube* has all sides of length :math:`l=1`, and is denoted as :math:`H_d(1)`.

**Hypersphere**

Assume that the data has been centered, so that :math:`\mmu=\0`.
Let :math:`r` denote the largest magnitude among all points:

.. math::

    r=\max_i\{\lv\x_i\rv\}

The data hyperspace can also be represented as a :math:`d`-dimensional 
*hyperball* centered at :math:`\0` with radius :math:`r`, defined as

.. note::

    :math:`B_d(r)=\{\x|\lv\x\rv\leq r\}`

.. math::

    \rm{or\ }B_d(r)=\{\x=(x_1,x_2,\cds,x_d)^T|\sum_{j=1}^dx_j^2\leq r^2\}

The surface of the hyperball is called a *hypersphere*, and it consists of all 
the points exactly at distance :math:`r` from the center of the hyperball, 
defined, as

.. note::

    :math:`S_d(r)=\{\x|\lv\x\rv=r\}`

.. math::

    \rm{or\ }S_d(r)=\{\x=(x_1,x_2,\cds,x_d)^T|\sum_{j=1}^d(x_j)^2=r^2\}

Because the hyperball consists of all the surface and interior points, it is also called a *closed hypersphere*.

**Hyperplanes**

A hyperplane in :math:`d` dimensions is given as the set of all points 
:math:`\x\in\R^d` that satisfy the equation :math:`h(\x)=0`, where :math:`h(\x)`
is the *hyperplane function*, defined as follows:

.. math::

    h(\x)=\w^T\x+b=w_1x_1+w_2x_2+\cds+w_dx_d+b

Here, :math:`\w` is a :math:`d` dimensional *weight vector* and :math:`b` is a scalar, called the *bias*.
For points that comprise the hyperplane, we have

.. note::

    :math:`h(\x)=\w^T\x+b=0`

The hyperplane is thus defined as the set of all points such that :math:`\w^T\x=-b`.

Note that a hyperplane in :math:`d`-dimensions has dimension :math:`d-1`.
A hyperplane splits the original :math:`d`-dimensional space into two *half-space*.
Points on one side satisfy the equation :math:`h(\x)>0`, points on the other 
side satisfy the equation :math:`h(\x)<0`, and points on the hyperplane satisfy
the condition :math:`h(\x)=0`.

6.2 High-Dimensional Volumes
----------------------------

**Hypercube**

.. note::

    :math:`\rm{vol}(H_d(l))=l^d`

**Hypersphere**

.. note::

    :math:`\dp\rm{vol}(S_d(r))=K_d\cd r^d=\bigg(\frac{\pi^{\frac{d}{2}}}{\Gamma(\frac{d}{2}+1)}\bigg)r^d`

where

.. math::

    K_d=\frac{\pi^{\frac{d}{2}}}{\Gamma(\frac{d}{2}+1)}

is a scalar that depends on the dimenionality :math:`d`, and :math:`\Gamma` is the gamma function defined as

.. math::

    \Gamma(\alpha)=\int_0^\infty x^{\alpha-1}e^{-x}dx

The gamma function has the following property for any :math:`\alpha>1`

.. math::

    \Gamma(\alpha)=(\alpha-1)\Gamma(\alpha-1)

For any integer :math:`n\geq 1`, we immediately have

.. math::

    \Gamma(n)=(n-1)!

When :math:`d` is even, then :math:`\frac{d}{2}+1` is an integer, and we have

.. math::

    \Gamma\bigg(\frac{d}{2}+1\bigg)=\bigg(\frac{d}{2}\bigg)!

and when :math:`d` is odd, we have

.. math::

    \Gamma\bigg(\frac{d}{2}+1\bigg)=\bigg(\frac{d}{2}\bigg)\bigg(\frac{d-2}{2}
    \cds\bigg(\frac{d-(d-1)}{2}\bigg)\Gamma\bigg(\frac{1}{2}\bigg)=
    \bigg(\frac{d!!}{2^{(d+1)/2}}\bigg)\sqrt\pi

where :math:`d!!` denotes the double factorial (or multifactorial), given as

.. math::

    d!!=\left\{\begin{array}{lr}1\quad\rm{if\ }d=0\rm{\ or\ }d=1\\d\cd(d-2)!!\quad\rm{if\ }d\geq 2\end{array}\right.

Putting it all together we have

.. note::

    :math:`\dp\Gamma\bigg(\frac{d}{2}+1\bigg)=`
    :math:`\left\{\begin{array}{lr}(\frac{d}{2})!\quad\rm{if\ }d\rm{\ is\ even}\\\sqrt\pi(\frac{d!!}{2^{(d+1)/2}})\quad\rm{if\ }d\rm{\ is\ odd}\end{array}\right.`

**Surface Area**

The *surface area* of the hypersphere can be obtained by differentiating its volume with respect to :math:`r`, given as

.. math::

    \rm{area}(S_d(r))=\frac{d}{dr}\rm{vol}(S_d(r))=
    \bigg(\frac{\pi^{\frac{d}{2}}}{\Gamma(\frac{d}{2}+1)}\bigg)dr^{d-1}=
    \bigg(\frac{2\pi^{\frac{d}{2}}}{\Gamma(\frac{d}{2})}\bigg)r^{d-1}

**Asymptotic Volume**

For the unit hypersphere with :math:`r=1`

.. note::

    :math:`\dp\lim_{d\ra\infty}\rm{vol}(S_d(1))=\lim_{d\ra\infty}\frac{\pi^{\frac{d}{2}}}{\Gamma(\frac{d}{2}+1)}\ra 0`

6.3 Hypersphere Inscribed Within Hypercube
------------------------------------------

In two dimensions, we have

.. math::

    \frac{\rm{vol}(S_2(r))}{\rm{vol}(H_2(2r))}=\frac{\pi r^2}{4r^2}=\frac{\pi}{4}=78.5\%

In three dimensions, we have

.. math::

    \frac{\rm{vol}(S_3(r))}{\rm{vol}(H_3(2r))}=\frac{\frac{4}{3}\pi r^3}{8r^3}=\frac{\pi}{6}=52.4\%

As the dimensionality :math:`d` increases asymptotically, we get

.. note::

    :math:`\dp\lim_{d\ra\infty}\frac{\rm{vol}(S_d(r))}{\rm{vol}(H_d(2r))}=`
    :math:`\dp\lim_{d\ra\infty}\frac{\pi^{\frac{d}{2}}}{2^d\Gamma(\frac{d}{2}+1)}\ra 0`

This means that as the dimensionality increases, most of the volume of the 
hypercube is in the "corners", whereas the center is essentially empty.

6.4 Volume of Thin Hypersphere Shell
------------------------------------

Let :math:`S_d(r,\epsilon)` denote the thin hypershell of width :math:`\epsilon`.
Its volume is given as

.. math::

    \rm{vol}(S_d(r,\epsilon))=\rm{vol}(S_d(r))-\rm{vol}(S_d(r-\epsilon))=K_dr^d-K_d(r-\epsilon)^d.

Let us consider the ratio of the volume of the thin shell to the volume of the outer sphere:

.. math::

    \frac{\rm{vol}(S_d(r,\epsilon))}{\rm{vol}(S_d(r))}=
    \frac{K_dr^d-K_d(r-\epsilon)^d}{K_dr^d}=1-\bigg(1-\frac{\epsilon}{r}\bigg)^d

**Asymptotic Volume**

As :math:`d` increases, in the limit we obtain

.. note::

    :math:`\dp\lim_{d\ra\infty}\frac{\rm{vol}(S_d(r,\epsilon))}{\rm{vol}(S_d(r))}=`
    :math:`\dp\lim_{d\ra\infty}1-\bigg(1-\frac{\epsilon}{r}\bigg)^d\ra 1`

That is, almoast all of the volume of the hypersphere is contained in the thin shell as :math:`d\ra\infty`.
This means that in high-dimensional spaces, unlike in lower dimensions, most of 
the volume is concentrated around the surface (within :math:`\epsilon`) of the
hypersphere, and the center is essentially void.

6.5 Diagonals in Hyperspace
---------------------------

Consider the angle :math:`\th_d` betwen the ones vector :math:`\1` and the first 
standard basis vector :math:`\e_1`, in :math:`d` dimensions:

.. math::

    \cos\th_d=\frac{\e_1^T\1}{\lv\e_1\rv\lv\1\rv}=
    \frac{\e_1^T\1}{\sqrt{\e_1^T\e_1}\sqrt{\1^T\1}}=\frac{1}{\sqrt{1}\sqrt{d}}=
    \frac{1}{\sqrt{d}}

**Asymptotic Angle**

As :math:`d` increases, the angle between the :math:`d`-dimensional ones vector 
:math:`\1` and the first axis vector :math:`\e_1` is given as

.. math::

    \lim_{d\ra\infty}\cos\th_d=\lim_{d\ra\infty}\frac{1}{\sqrt{d}}\ra 0

which implies that

.. note::

    :math:`\dp\lim_{d\ra\infty}\th_d\ra\frac{\pi}{2}=90^\circ`

This implies that in high dimensions all of the diagonal vectors are 
perpendicular (or orthogonal) to all the coordinates axes.
Because there are :math:`2^d` corners in a :math:`d`-dimensional hyperspace, 
there are :math:`2^d` diagonal vectors from the origin to each of the corners.
Because the diagonal vectors in opposite directions define a new axis, we obtain 
:math:`2^{d-1}` new axes, each of which is essentially orthogonal to all of the 
:math:`d` principal coordinate axes.

6.6 Density of The Multivariate Normal
--------------------------------------

Consider the probability of a point being with a fraction :math:`\alpha>0`, of the peack density at the mean.

For a multivariate normal distribution, with :math:`\mmu=\0_d`, and :math:`\Sg=\I_d`, we have

.. math::

    f(\x)=\frac{1}{(\sqrt{2\pi})^d}\exp\bigg\{-\frac{\x^T\x}{2}\bigg\}

At the mean :math:`\mmu=\0_d`, the peak density is :math:`f(\0_d)=\frac{1}{(\sqrt{2\pi})^d}`.
Thus, the set of points :math:`\x` with density at least :math:`\alpha` fraction 
of the density at the mean, with :math:`0<\alpha<1`, is given as

.. math::

    \frac{f(\x)}{f(\0)}\geq\alpha

which implies that

.. math::

    \exp\bigg\{-\frac{\x^T\x}{2}\bigg\}&\geq\alpha

    \rm{or\ }\x^T\x&\leq-2\ln(\alpha)

    \rm{and\ thus\ }\sum_{i=1}^d(x_i)^2&\leq-2\ln(\alpha)

The probability that a point :math:`\x` is within :math:`\alpha` times the 
density at the mean can be computed from the :math:`\chi_d^2` density function

.. note::

    :math:`\dp P\bigg(\frac{f(\x)}{f(\0)}\geq\alpha\bigg)=P(\x^T\x\leq-2\ln(\alpha))=`
    :math:`\dp\int_0^{-2\ln(\alpha)}f_{\chi_d^2}(\x^T\x)=F_{\chi_d^2}(-2\ln(\alpha))`

where :math:`f_{\chi_d^2}` is the chi-squared probability density function with :math:`q` degrees of freedom:

.. math::

    f_{\chi_d^2}(x)=\frac{1}{2^{q/2}\Gamma(q/2)}x^{\frac{q}{2}-1}e^{-\frac{x}{2}}

and :math:`F_{\chi_d^2}` is its cumulative distribution function.

As dimensionality increases, this probability decreases sharply, and eventually tends to zero, that is,

.. note::

    :math:`\dp\lim_{d\ra\infty}P(\x^T\x\leq-2\ln(\alpha))\ra 0`

Thus, in higher dimensions the probability density around the mean decreases 
very rapidly as one moves away from the mean. 
In essence the entire probability mass migrates to the tail regions.

**Distance of Points from the Mean**

Let :math:`r^2` denote the square of the distance of a point :math:`\x` to the center :math:`\mmu=\0`, given as

.. math::

    r^2=\lv\x-\0\rv^2=\x^T\x=\sum_{i=1}^dx_i^2

:math:`\x^T\x` follows a :math:`\chi^2` distribution with :math:`d` degress of 
freedom, which has mean :math:`d` and variance :math:`2d`.
It follows that the mean and variance of the random variable :math:`r^2` is

.. math::

    \mu_{r^2}=d\quad\sg_{r^2}^2=2d

We conclude that for large :math:`d`, the radius :math:`r` follows a normal 
distribution with mean :math:`\sqrt{d}` and standard deviation 
:math:`1/\sqrt{2}`.
The density at the mean distance :math:`r=\sqrt{d}` is exponentially smaller than that at the peak density because

.. math::

    \frac{f(\x)}{f(\0)}=\exp\{-\x^T\x/2\}=\exp\{-d/2\}

Whereas the density of the standard multivariate normal is maximized at the 
center :math:`\0`, most of the probability mass is concentrated in a small band 
around the mean distance of :math:`\sqrt{d}` from the center.