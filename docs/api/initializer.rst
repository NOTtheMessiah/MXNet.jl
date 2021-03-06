
Initializers
============
Interface
---------




.. class:: AbstractInitializer

   The abstract base class for all initializers.

To define a new initializer, it is
enough to derive a new type, and implement one or more of the following methods:

.. function:: _init_weight(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
.. function:: _init_bias(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
.. function:: _init_gamma(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
.. function:: _init_beta(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)

Or, if full behavior customization is needed, override the following function

.. function:: call(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)




Built-in initializers
---------------------




.. class:: UniformInitializer

   Initialize weights according to a uniform distribution within the provided scale.




.. function UniformInitializer(scale=0.07)

   Construct a :class:`UniformInitializer` with the specified scale.




.. class:: NormalInitializer

   Initialize weights according to a univariate Gaussian distribution.




.. function:: NormalIninitializer(; mu=0, sigma=0.01)

   Construct a :class:`NormalInitializer` with mean ``mu`` and variance ``sigma``.




.. class:: XavierInitializer

   The initializer documented in the paper [Bengio and Glorot 2010]: *Understanding
   the difficulty of training deep feedforward neuralnetworks*.

   There are several different version of the XavierInitializer used in the wild.
   The general idea is that the variance of the initialization distribution is controlled
   by the dimensionality of the input and output. As a distribution one can either choose
   a normal distribution with μ = 0 and σ² or a uniform distribution from -σ to σ.

   Several different ways of calculating the variance are given in the literature or are
   used by various libraries.

   - [Bengio and Glorot 2010]: ``mx.XavierInitializer(distribution = mx.xv_uniform, regularization = mx.xv_avg, magnitude = 1)``
   - [K. He, X. Zhang, S. Ren, and J. Sun 2015]: ``mx.XavierInitializer(distribution = mx.xv_gaussian, regularization = mx.xv_in, magnitude = 2)``
   - caffe_avg: ``mx.XavierInitializer(distribution = mx.xv_uniform, regularization = mx.xv_avg, magnitude = 3)``



