
Overfitting demo
================

Create a dataset based on a true sinusoidal relationship
--------------------------------------------------------

Let's look at a synthetic dataset consisting of 30 points drawn from the
sinusoid :math:`y = \sin(4x)`:

.. code:: python

    # python standard library
    import math
    import random
    
    # third party
    import graphlab
    from matplotlib import pyplot as plt
    import numba
    import numpy
    
    %matplotlib inline

Create random values for x in interval [0,1)

.. code:: python

    random.seed(98103)
    n = 30
    x = graphlab.SArray([random.random() for i in range(n)]).sort()

``x`` is an array of random numbers (from 0 to 1 (non-inclusive)) in
non-decreasing order.

Compute y.

.. code:: python

    y = x.apply(lambda x: math.sin(4*x))

:math:`y = \sin(4x)`

Add random Gaussian noise to y

.. code:: python

    random.seed(1)
    e = graphlab.SArray([random.gauss(0,1.0/3.0) for i in range(n)])
    y = y + e

Put data into an SFrame to manipulate later
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    data = graphlab.SFrame({'X1':x,'Y':y})
    data




.. parsed-literal::

    Columns:
    	X1	float
    	Y	float
    
    Rows: 30
    
    Data:
    +-----------------+----------------+
    |        X1       |       Y        |
    +-----------------+----------------+
    | 0.0395789449501 | 0.587050191026 |
    | 0.0415680996791 | 0.648655851372 |
    | 0.0724319480801 | 0.307803309485 |
    |  0.150289044622 | 0.310748447417 |
    |  0.161334144502 | 0.237409625496 |
    |  0.191956312795 | 0.705017157224 |
    |  0.232833917145 | 0.461716676992 |
    |  0.259900980166 | 0.383260507851 |
    |  0.380145814869 | 1.06517691429  |
    |  0.432444723508 | 1.03184706949  |
    +-----------------+----------------+
    [30 rows x 2 columns]
    Note: Only the head of the SFrame is printed.
    You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.



Create a function to plot the data, since we'll do it many times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    def plot_data(data):
        """
        :param:
         - `data`: frame with 'X1' and 'Y' columns
        :postcondition: scatter plot of `data` created
        """
        plt.plot(data['X1'], data['Y'], '.')
        plt.xlabel('x')
        plt.ylabel('y')
    
    plot_data(data)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa8adfb10d0>


Define some useful polynomial regression functions
--------------------------------------------------

Define a function to create our features for a polynomial regression
model of any degree:

1. make a copy of the data
2. for each power from 1 to degree - 1: 2.a. Add a column power+1 that
   is a multiple of data raised to power times original data-values (add
   columns of increasing polynomial degree to the data)
3. Return the copy of the data with the higher polynomial degrees added

.. code:: python

    def polynomial_features(data, degree):
        """
        :param:
         - `data`: array of data
         - `degree`: highest degree to raise each data-point
        :return: copy of data with columns raised to power from 1 to degree
        """
        data_copy=data.copy()
        # :precondition: data_copy has a column 'X1' representing data raised to power 1
        assert 'X1' in data.column_names()
        for i in range(1, degree):
            # :invariant: i is the power of the highest column in the data so far
            # :sentinel: i < degree
            data_copy['X' + str(i + 1)] = data_copy['X' + str(i)] * data_copy['X1']
            # :progress: data_copy has columns raised to power from 1 to i + 1
            # :stop: i + 1 = degree
        # :postcondition: data_copy has columns with original data raised from 1 to degree
        assert i + 1 == degree
        return data_copy

Define a function to fit a polynomial linear regression model of degree
"deg" to the data in "data":

.. code:: python

    def polynomial_regression(data, degree):
        """
        :param:
         - `data`: array of data to create linear model
         - `degree`: highest power to raise the x-data (calls polynomial_features)
        :return: SFrame model fit to data
        """
        model = graphlab.linear_regression.create(polynomial_features(data, degree), 
                                                  target='Y', l2_penalty=0., l1_penalty=0.,
                                                  validation_set=None, verbose=False)
        return model

Define function to plot data and predictions made, since we are going to
use it many times.

.. code:: python

    def plot_poly_predictions(data, model):
        """
        :param:
         - `data`: frame with polynomial columns
         - `model: linear model fit to data
        :postcondition: plot with 
        """
        plot_data(data)
    
        # Get the degree of the polynomial
        degree = len(model.coefficients['value']) - 1
        
        # Create 200 points in the x axis and compute the predicted value for each point
        x_pred = graphlab.SFrame({'X1':[i/200.0 for i in range(200)]})
        y_pred = model.predict(polynomial_features(x_pred, degree))
        
        # plot predictions
        plt.plot(x_pred['X1'], y_pred, 'g-', label='degreeree ' + str(degree) + ' fit')
        plt.legend(loc='upper left')
        plt.axis([0,1,-1.5,2])

Create a function that prints the polynomial coefficients in a pretty
way :)

.. code:: python

    def print_coefficients(model):
        """
        :param:
         - `model`: linear regression model
        :postcondition: print equation of the model to the screen
        """
        # Get the degree of the polynomial
        deg = len(model.coefficients['value'])-1
    
        # Get learned parameters as a list
        w = list(model.coefficients['value'])
    
        # Numpy has a nifty function to print out polynomials in a pretty way
        # (We'll use it, but it needs the parameters in the reverse order)
        print('Learned polynomial for degree ' + str(deg) + ':\n')
        w.reverse()
        print(numpy.poly1d(w))

Fit a degree-2 polynomial
-------------------------

Fit our degree-2 polynomial to the data generated above:

.. code:: python

    model = polynomial_regression(data, degree=2)

Inspect learned parameters

.. code:: python

    print_coefficients(model)


.. parsed-literal::

    Learned polynomial for degree 2:
    
            2
    -5.129 x + 4.147 x + 0.07471


Form and plot our predictions along a grid of x values:

.. code:: python

    plot_poly_predictions(data, model)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa89e3b4850>


Fit a degree-4 polynomial
-------------------------

.. code:: python

    model = polynomial_regression(data, degree=4)
    print_coefficients(model)
    plot_poly_predictions(data,model)


.. parsed-literal::

    Learned polynomial for degree 4:
    
           4         3         2
    23.87 x - 53.82 x + 35.23 x - 6.828 x + 0.7755



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa8da7e4790>


Fit a degree-16 polynomial
--------------------------

.. code:: python

    model = polynomial_regression(data, degree=16)
    print_coefficients(model)


.. parsed-literal::

    Learned polynomial for degree 16:
    
                16             15             14            13
    -4.537e+05 x  + 1.129e+06 x  + 4.821e+05 x  - 3.81e+06 x 
                  12             11             10             9
     + 3.536e+06 x  + 5.753e+04 x  - 1.796e+06 x  + 2.178e+06 x
                  8             7            6             5             4
     - 3.662e+06 x + 4.442e+06 x - 3.13e+06 x + 1.317e+06 x - 3.356e+05 x
                 3        2
     + 5.06e+04 x - 4183 x + 160.8 x - 1.621


oah!!!! Those coefficients are *crazy*! On the order of 10^6.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    plot_poly_predictions(data, model)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa89e124e10>


Above: Fit looks pretty wild, too. Here's a clear example of how overfitting is associated with very large magnitude estimated coefficients.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ridge Regression
----------------

Ridge regression aims to avoid overfitting by adding a cost to the RSS
term of standard least squares that depends on the 2-norm of the
coefficients :math:`\|w\|`. The result is penalizing fits with large
coefficients. The strength of this penalty, and thus the fit vs. model
complexity balance, is controled by a parameter lambda (here called
"L2\_penalty").

Define our function to solve the ridge objective for a polynomial
regression model of any degree:

.. code:: python

    def polynomial_ridge_regression(data, degree, l2_penalty):
        """
        :param:
         - `data`: frame with 'X1' and 'Y' columns
         - `degree`: degree for highest polynomial column to add
         - `l2_penalty`: penalty to add for ridge regression
        :return: ridge-regression model fit to the data
        """
        model = graphlab.linear_regression.create(polynomial_features(data, degree), 
                                                  target='Y', l2_penalty=l2_penalty,
                                                  validation_set=None, verbose=False)
        return model

Perform a ridge fit of a degree-16 polynomial using a *very* small penalty strength
-----------------------------------------------------------------------------------

.. code:: python

    model = polynomial_ridge_regression(data, degree=16, l2_penalty=1e-25)
    print_coefficients(model)


.. parsed-literal::

    Learned polynomial for degree 16:
    
                16             15             14            13
    -4.537e+05 x  + 1.129e+06 x  + 4.821e+05 x  - 3.81e+06 x 
                  12             11             10             9
     + 3.536e+06 x  + 5.753e+04 x  - 1.796e+06 x  + 2.178e+06 x
                  8             7            6             5             4
     - 3.662e+06 x + 4.442e+06 x - 3.13e+06 x + 1.317e+06 x - 3.356e+05 x
                 3        2
     + 5.06e+04 x - 4183 x + 160.8 x - 1.621


.. code:: python

    plot_poly_predictions(data, model)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa89e124710>


Perform a ridge fit of a degree-16 polynomial using a very large penalty strength
---------------------------------------------------------------------------------

.. code:: python

    model = polynomial_ridge_regression(data, degree=16, l2_penalty=100)
    print_coefficients(model)


.. parsed-literal::

    Learned polynomial for degree 16:
    
            16          15          14          13          12         11
    -0.301 x  - 0.2802 x  - 0.2604 x  - 0.2413 x  - 0.2229 x  - 0.205 x 
               10          9          8          7          6           5
     - 0.1874 x  - 0.1699 x - 0.1524 x - 0.1344 x - 0.1156 x - 0.09534 x
                4           3           2
     - 0.07304 x - 0.04842 x - 0.02284 x - 0.002257 x + 0.6416


.. code:: python

    plot_poly_predictions(data,model)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa89e0d95d0>


Let's look at fits for a sequence of increasing lambda values
-------------------------------------------------------------

.. code:: python

    for l2_penalty in [1e-25, 1e-10, 1e-6, 1e-3, 1e2]:
        model = polynomial_ridge_regression(data, degree=16, l2_penalty=l2_penalty)
        print 'lambda = %.2e' % l2_penalty
        print_coefficients(model)
        print '\n'
        plt.figure()
        plot_poly_predictions(data,model)
        plt.title('Ridge, lambda = %.2e' % l2_penalty)


.. parsed-literal::

    lambda = 1.00e-25
    Learned polynomial for degree 16:
    
                16             15             14            13
    -4.537e+05 x  + 1.129e+06 x  + 4.821e+05 x  - 3.81e+06 x 
                  12             11             10             9
     + 3.536e+06 x  + 5.753e+04 x  - 1.796e+06 x  + 2.178e+06 x
                  8             7            6             5             4
     - 3.662e+06 x + 4.442e+06 x - 3.13e+06 x + 1.317e+06 x - 3.356e+05 x
                 3        2
     + 5.06e+04 x - 4183 x + 160.8 x - 1.621
    
    
    lambda = 1.00e-10
    Learned polynomial for degree 16:
    
               16             15             14             13
    4.975e+04 x  - 7.821e+04 x  - 2.265e+04 x  + 3.949e+04 x 
                  12        11             10             9             8
     + 4.366e+04 x  + 3074 x  - 3.332e+04 x  - 2.786e+04 x + 1.032e+04 x
                  7        6             5             4        3         2
     + 2.962e+04 x - 1440 x - 2.597e+04 x + 1.839e+04 x - 5596 x + 866.1 x - 65.19 x + 2.159
    
    
    lambda = 1.00e-06
    Learned polynomial for degree 16:
    
           16         15         14        13         12         11
    329.1 x  - 356.4 x  - 264.2 x  + 33.8 x  + 224.7 x  + 210.8 x 
              10         9       8         7         6         5         4
     + 49.62 x  - 122.4 x - 178 x - 79.13 x + 84.89 x + 144.9 x + 5.123 x
              3         2
     - 156.9 x + 88.21 x - 14.82 x + 1.059
    
    
    lambda = 1.00e-03
    Learned polynomial for degree 16:
    
           16         15         14         13         12          11
    6.364 x  - 1.596 x  - 4.807 x  - 4.778 x  - 2.776 x  + 0.1238 x 
              10         9         8         7          6         5
     + 2.977 x  + 4.926 x + 5.203 x + 3.248 x - 0.9291 x - 6.011 x
              4         3         2
     - 8.395 x - 2.655 x + 9.861 x - 2.225 x + 0.5636
    
    
    lambda = 1.00e+02
    Learned polynomial for degree 16:
    
            16          15          14          13          12         11
    -0.301 x  - 0.2802 x  - 0.2604 x  - 0.2413 x  - 0.2229 x  - 0.205 x 
               10          9          8          7          6           5
     - 0.1874 x  - 0.1699 x - 0.1524 x - 0.1344 x - 0.1156 x - 0.09534 x
                4           3           2
     - 0.07304 x - 0.04842 x - 0.02284 x - 0.002257 x + 0.6416
    
    



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa89e1c1550>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa89e170c90>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa89df34910>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa89e1246d0>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa89dd52d10>


Perform a ridge fit of a degree-16 polynomial using a "good" penalty strength
-----------------------------------------------------------------------------

We will learn about cross validation later in this course as a way to
select a good value of the tuning parameter (penalty strength) lambda.
Here, we consider "leave one out" (LOO) cross validation, which one can
show approximates average mean square error (MSE). As a result, choosing
lambda to minimize the LOO error is equivalent to choosing lambda to
minimize an approximation to average MSE.

.. code:: python

    # LOO cross validation -- return the average MSE
    import sys
    
    def leave_one_out(data, degree, l2_penalty_values):
        """
        :param: 
         - `data`: frame with 'X1' and 'Y' columns
         - `degree`: highest degree of polynomial column to add
         - `l2_penalty_values`: penalty for ridge_regression
    
        :return: (mean-squared error, l2 penalty) for case with the lowest MSE
        """
        # Create polynomial features
        data = polynomial_features(data, degree)
        
        # Create as many folds for cross validatation as number of data points
        num_folds = len(data)
        folds = graphlab.cross_validation.KFold(data, num_folds)
        
        # for each value of l2_penalty, fit a model for each fold and compute average MSE
        l2_penalty_mse = []
        min_mse = sys.maxint
        best_l2_penalty = None
        for l2_penalty in l2_penalty_values:
            next_mse = 0.0
            for train_set, validation_set in folds:
                # train model
                model = graphlab.linear_regression.create(train_set,
                                                          target='Y', 
                                                          l2_penalty=l2_penalty,
                                                          validation_set=None,
                                                          verbose=False)
                
                # predict on validation set 
                y_test_predicted = model.predict(validation_set)
                # compute squared error
                next_mse += ((y_test_predicted-validation_set['Y'])**2).sum()
            
            # save squared error in list of MSE for each l2_penalty
            next_mse = next_mse/num_folds
            l2_penalty_mse.append(next_mse)
            if next_mse < min_mse:
                min_mse = next_mse
                best_l2_penalty = l2_penalty
                
        return l2_penalty_mse, best_l2_penalty

Run LOO cross validation for "num" values of lambda, on a log scale

.. code:: python

    l2_penalty_values = numpy.logspace(-4, 10, num=10)

.. code:: python

    l2_penalty_mse, best_l2_penalty = leave_one_out(data, 16, l2_penalty_values)

Plot results of estimating LOO for each value of lambda

.. code:: python

    plt.plot(l2_penalty_values, l2_penalty_mse, '-')
    plt.xlabel('$\L2_{penalty}$')
    plt.ylabel('LOO cross validation error')
    plt.xscale('log')
    plt.yscale('log')



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa89e00d590>


Find the value of lambda, :math:`\lambda_{\mathrm{CV}}`, that minimizes
the LOO cross validation error, and plot resulting fit.

.. code:: python

    best_l2_penalty




.. parsed-literal::

    0.12915496650148839



.. code:: python

    model = polynomial_ridge_regression(data, degree=16, l2_penalty=best_l2_penalty)
    print_coefficients(model)


.. parsed-literal::

    Learned polynomial for degree 16:
    
           16         15          14          13          12           11
    1.345 x  + 1.141 x  + 0.9069 x  + 0.6447 x  + 0.3569 x  + 0.04947 x 
               10          9          8         7         6         5
     - 0.2683 x  - 0.5821 x - 0.8701 x - 1.099 x - 1.216 x - 1.145 x
               4           3          2
     - 0.7837 x - 0.07406 x + 0.7614 x + 0.7703 x + 0.3918


.. code:: python

    plot_poly_predictions(data, model)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa89e0a7690>


Lasso Regression
----------------

Lasso regression jointly shrinks coefficients to avoid overfitting, and
implicitly performs feature selection by setting some coefficients
exactly to 0 for sufficiently large penalty strength lambda (here called
"L1\_penalty"). In particular, lasso takes the RSS term of standard
least squares and adds a 1-norm cost of the coefficients :math:`\|w\|`.

Define our function to solve the lasso objective for a polynomial
regression model of any degree:

.. code:: python

    def polynomial_lasso_regression(data, degree, l1_penalty):
        """
        :param:
         - `data`: frame with 'X1' and 'Y' columns
         - `degree`: highest polynomial degree to add to columns
         - `l1_penalty`: penalty for the regression
        :return: lasso regression model fitted to polynomial data
        """
        model = graphlab.linear_regression.create(polynomial_features(data, degree), 
                                                  target='Y',
                                                  l2_penalty=0.,
                                                  l1_penalty=l1_penalty,
                                                  validation_set=None, 
                                                  solver='fista',
                                                  verbose=False,
                                                  max_iterations=3000,
                                                  convergence_threshold=1e-10)
        return model

Explore the lasso solution as a function of a few different penalty strengths
-----------------------------------------------------------------------------

We refer to lambda in the lasso case below as "l1\_penalty"

.. code:: python

    for l1_penalty in [0.0001, 0.01, 0.1, 10]:
        model = polynomial_lasso_regression(data, degree=16, l1_penalty=l1_penalty)
        print 'l1_penalty = %e' % l1_penalty
        print 'number of nonzeros = %d' % (model.coefficients['value']).nnz()
        print_coefficients(model)
        print '\n'
        plt.figure()
        plot_poly_predictions(data,model)
        plt.title('LASSO, lambda = %.2e, # nonzeros = %d' % (l1_penalty, (model.coefficients['value']).nnz()))


.. parsed-literal::

    l1_penalty = 1.000000e-04
    number of nonzeros = 17
    Learned polynomial for degree 16:
    
           16        15         14         13         12         11
    29.02 x  + 1.35 x  - 12.72 x  - 16.93 x  - 13.82 x  - 6.698 x 
              10         9         8         7         6         5
     + 1.407 x  + 8.939 x + 12.88 x + 11.44 x + 3.759 x - 8.062 x
              4         3         2
     - 16.28 x - 7.682 x + 17.86 x - 4.384 x + 0.685
    
    
    l1_penalty = 1.000000e-02
    number of nonzeros = 14
    Learned polynomial for degree 16:
    
            16             15           11          10         9          8
    -1.181 x  - 0.0003031 x  + 0.08677 x  + 0.7382 x  + 3.828 x + 0.4755 x
               7            6          5         4             3         2
     + 0.1277 x + 0.005139 x - 0.6156 x - 10.11 x - 0.0001921 x + 6.685 x - 1.28 x + 0.5057
    
    
    l1_penalty = 1.000000e-01
    number of nonzeros = 5
    Learned polynomial for degree 16:
    
          16         6         5
    2.21 x  - 1.002 x - 2.962 x + 1.216 x + 0.3473
    
    
    l1_penalty = 1.000000e+01
    number of nonzeros = 2
    Learned polynomial for degree 16:
    
            9
    -1.526 x + 0.5755
    
    



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa8973d3590>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa8973d3650>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa89700bfd0>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fa8971d2890>


Above: We see that as lambda increases, we get sparser and sparser
solutions. However, even for our non-sparse case for lambda=0.0001, the
fit of our high-order polynomial is not too wild. This is because, like
in ridge, coefficients included in the lasso solution are shrunk
relative to those of the least squares (unregularized) solution. This
leads to better behavior even without sparsity. Of course, as lambda
goes to 0, the amount of this shrinkage decreases and the lasso solution
approaches the (wild) least squares solution.
