
Regression Week 4: Ridge Regression (gradient descent)
======================================================

In this notebook, you will implement ridge regression via gradient
descent. You will: \* Convert an SFrame into a Numpy array \* Write a
Numpy function to compute the derivative of the regression weights with
respect to a single feature \* Write gradient descent function to
compute the regression weights given an initial weight vector, step
size, tolerance, and L2 penalty

Fire up graphlab create
-----------------------

Make sure you have the latest version of GraphLab Create (>= 1.7)

.. code:: python

    # python standard library
    import os
    
    # third party
    import graphlab
    import numpy
    import matplotlib.pyplot as plot
    import seaborn
    
    # this code
    import machine_learning
    from machine_learning.coursera.regression.common_utilities.numpy_helpers import get_numpy_data
    from machine_learning.coursera.regression.common_utilities.numpy_helpers import predict_output

.. code:: python

    %matplotlib inline

Load in house sales data
------------------------

Dataset is from house sales in King County, the region where the city of
Seattle, WA is located.

.. code:: python

    path = os.path.join(machine_learning.__path__[0], machine_learning.large_data_path, 'kc_house_data.gl/')
    sales = graphlab.SFrame(path)


.. parsed-literal::

    [INFO] GraphLab Server Version: 1.7.1
    [INFO] Start server at: ipc:///tmp/graphlab_server-17306 - Server binary: /home/charon/.virtualenvs/machinelearning/lib/python2.7/site-packages/graphlab/unity_server - Server log: /tmp/graphlab_server_1451092352.log
    [INFO] [1;32m1451092352 : INFO:     (initialize_globals_from_environment:282): Setting configuration variable GRAPHLAB_FILEIO_ALTERNATIVE_SSL_CERT_FILE to /home/charon/.virtualenvs/machinelearning/local/lib/python2.7/site-packages/certifi/cacert.pem
    [0m[1;32m1451092352 : INFO:     (initialize_globals_from_environment:282): Setting configuration variable GRAPHLAB_FILEIO_ALTERNATIVE_SSL_CERT_DIR to 
    [0mThis non-commercial license of GraphLab Create is assigned to necromuralist@gmail.com and will expire on October 20, 2016. For commercial licensing options, visit https://dato.com/buy/.
    


.. code:: python

    print(sales.column_names())


.. parsed-literal::

    ['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']


If we want to do any "feature engineering" like creating new features or
adjusting existing ones we should do this directly using the SFrames as
seen in the first notebook of Week 2. For this notebook, however, we
will work with the existing features.

Import useful functions from previous notebook
----------------------------------------------

As in Week 2, we convert the SFrame into a 2D Numpy array. Copy and
paste ``get_num_data()`` from the second notebook of Week 2.

Also, copy and paste the ``predict_output()`` function to compute the
predictions for an entire matrix of features given the matrix and the
weights:

Computing the Derivative
------------------------

We are now going to move to computing the derivative of the regression
cost function. Recall that the cost function is the sum over the data
points of the squared difference between an observed output and a
predicted output, plus the L2 penalty term.

::

    Cost(w)
    = SUM[ (prediction - output)^2 ]
    + l2_penalty*(w[0]^2 + w[1]^2 + ... + w[k]^2).

Since the derivative of a sum is the sum of the derivatives, we can take
the derivative of the first part (the RSS) as we did in the notebook for
the unregularized case in Week 2 and add the derivative of the
regularization part. As we saw, the derivative of the RSS with respect
to ``w[i]`` can be written as:

::

    2*SUM[ error*[feature_i] ].

The derivative of the regularization term with respect to ``w[i]`` is:

::

    2*l2_penalty*w[i].

Summing both, we get

::

    2*SUM[ error*[feature_i] ] + 2*l2_penalty*w[i].

That is, the derivative for the weight for feature i is the sum (over
data points) of 2 times the product of the error and the feature itself,
plus ``2*l2_penalty*w[i]``.

**We will not regularize the constant.** Thus, in the case of the
constant, the derivative is just twice the sum of the errors (without
the ``2*l2_penalty*w[0]`` term).

Recall that twice the sum of the product of two vectors is just twice
the dot product of the two vectors. Therefore the derivative for the
weight for feature\_i is just two times the dot product between the
values of feature\_i and the current errors, plus ``2*l2_penalty*w[i]``.

With this in mind complete the following derivative function which
computes the derivative of the weight given the value of the feature
(over all data points) and the errors (over all data points). To decide
when to we are dealing with the constant (so we don't regularize it) we
added the extra parameter to the call ``feature_is_constant`` which you
should set to ``True`` when computing the derivative of the constant and
``False`` otherwise.

.. code:: python

    def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
        # If feature_is_constant is True, derivative is twice the dot product of errors and feature
        
        # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
        derivative = 2 * errors.dot(feature)
        if not feature_is_constant:
            derivative += 2 * l2_penalty * weight
        return derivative

To test your feature derivartive run the following:

.. code:: python

    (example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
    my_weights = numpy.array([1., 10.])
    test_predictions = predict_output(example_features, my_weights) 
    errors = test_predictions - example_output # prediction errors
    
    # next two lines should print the same values
    actual = feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False)
    expected = numpy.sum(errors*example_features[:,1])*2+20.
    assert actual == expected
    
    # next two lines should print the same values
    actual = feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True)
    expected =  numpy.sum(errors)*2.
    assert actual == expected

Gradient Descent
----------------

Now we will write a function that performs a gradient descent. The basic
premise is simple. Given a starting point we update the current weights
by moving in the negative gradient direction. Recall that the gradient
is the direction of *increase* and therefore the negative gradient is
the direction of *decrease* and we're trying to *minimize* a cost
function.

The amount by which we move in the negative gradient *direction* is
called the 'step size'. We stop when we are 'sufficiently close' to the
optimum. Unlike in Week 2, this time we will set a **maximum number of
iterations** and take gradient steps until we reach this maximum number.
If no maximum number is supplied, the maximum should be set 100 by
default. (Use default parameter values in Python.)

With this in mind, complete the following gradient descent function
below using your derivative function above. For each step in the
gradient descent, we update the weight for each feature before computing
our stopping criteria.

.. code:: python

    def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
        """
        :param:
         - `feature_matrix`: numpy array of features
         - `output`: numpy array of target outputs
         - `step_size`: size for gradient descent steps
         - `l2_penalty`: ridge regression penalty
         - `max_iterations`: most tries before giving up
        """
        weights = numpy.array(initial_weights) # make sure it's a numpy array
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            # compute the predictions based on feature_matrix and weights using your predict_output() function
            predictions = predict_output(feature_matrix, weights)
            # compute the errors as predictions - output
            errors = predictions - output
            for i in xrange(len(weights)): # loop over each weight
                # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
                # compute the derivative for weight[i].
                #(Remember: when i=0, you are computing the derivative of the constant!)
                derivative = feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty, i==0)
                # subtract the step size times the derivative from the current weight
                weights[i] -= step_size * derivative
        return weights

Visualizing effect of L2 penalty
--------------------------------

The L2 penalty gets its name because it causes weights to have smaller
L2 norms than otherwise. Let's see how large weights get penalized. Let
us consider a simple model with 1 feature:

.. code:: python

    simple_features = ['sqft_living']
    my_output = 'price'

Let us split the dataset into training set and test set. Make sure to
use ``seed=0``:

.. code:: python

    train_data, test_data = sales.random_split(.8, seed=0)

In this part, we will only use ``'sqft_living'`` to predict ``'price'``.
Use the ``get_numpy_data`` function to get a Numpy versions of your data
with only this feature, for both the ``train_data`` and the
``test_data``.

.. code:: python

    (simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
    (simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

Let's set the parameters for our optimization:

.. code:: python

    initial_weights = numpy.array([0., 0.])
    step_size = 1e-12
    max_iterations = 1000

First, let's consider no regularization. Set the ``l2_penalty`` to
``0.0`` and run your ridge regression algorithm to learn the weights of
your model. Call your weights:

``simple_weights_0_penalty``

we'll use them later.

.. code:: python

    simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, 0.0, max_iterations)

Next, let's consider high regularization. Set the ``l2_penalty`` to
``1e11`` and run your ridge regression algorithm to learn the weights of
your model. Call your weights:

``simple_weights_high_penalty``

we'll use them later.

.. code:: python

    simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, 1e11, max_iterations)

This code will plot the two learned models. (The blue line is for the
model with no regularization and the red line is for the one with high
regularization.)

.. code:: python

    seaborn.set_palette('husl')
    seaborn.set_style('whitegrid')
    figure = plot.figure()
    axe = figure.gca()
    x = simple_feature_matrix[:,1]
    lines = axe.plot(x, output, '.', label="data")
    lines = axe.plot(x, predict_output(simple_feature_matrix, initial_weights),'-', label='Zero Weights')
    lines = axe.plot(x, predict_output(simple_feature_matrix, simple_weights_0_penalty),'-', label='No Regularization')
    lines = axe.plot(x, predict_output(simple_feature_matrix, simple_weights_high_penalty),'-', label='High Regularization')
    axe.set_xlabel("Living Area (Sq Ft)")
    axe.set_ylabel("Price ($)")
    axe.legend()
    title = axe.set_title("Ridge Regression Penalty Effects")



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f4593dc2b90>


Compute the RSS on the TEST data for the following three sets of
weights: 1. The initial weights (all zeros) 2. The weights learned with
no regularization 3. The weights learned with high regularization

Which weights perform best?

.. code:: python

    predictions_initial = predict_output(simple_test_feature_matrix, initial_weights)
    predictions_no_regularization = predict_output(simple_test_feature_matrix, simple_weights_0_penalty)
    predictions_high_regularization = predict_output(simple_test_feature_matrix, simple_weights_high_penalty)

.. code:: python

    residuals_initial = predictions_initial - test_output
    residuals_no_regularization = predictions_no_regularization - test_output
    residuals_high_regularization = predictions_high_regularization - test_output

***QUIZ QUESTIONS*** 1. What is the value of the coefficient for
``sqft_living`` that you learned with no regularization, rounded to 1
decimal place? What about the one with high regularization? 2. Comparing
the lines you fit with the with no regularization versus high
regularization, which one is steeper? 3. What are the RSS on the test
data for each of the set of weights above (initial, no regularization,
high regularization)?

.. code:: python

    print("No Regularization `sqft_living`: {0:.1f}".format(simple_weights_0_penalty[1]))
    print("High Regularization `sqft_living`: {0:.1f}".format(simple_weights_high_penalty[1]))


.. parsed-literal::

    No Regularization `sqft_living`: 263.0
    High Regularization `sqft_living`: 124.6


No Regularization is steeper than High Regularization.

.. code:: python

    print("RSS Zero Weights: {0}".format((residuals_initial**2).sum()))
    print("RSS No regularization: {0}".format((residuals_no_regularization**2).sum()))
    print("RSS High Regularization: {0}".format((residuals_high_regularization**2).sum()))


.. parsed-literal::

    RSS Initial: 1.78427328252e+15
    RSS No regularization: 2.75723634598e+14
    RSS High Regularization: 6.94642100914e+14


Running a multiple regression with L2 penalty
---------------------------------------------

Let us now consider a model with 2 features:
``['sqft_living', 'sqft_living15']``.

First, create Numpy versions of your training and test data with these
two features.

.. code:: python

    model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
    my_output = 'price'
    (feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
    (test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

We need to re-initialize the weights, since we have one extra parameter.
Let us also set the step size and maximum number of iterations.

.. code:: python

    initial_weights = numpy.array([0.0,0.0,0.0])
    step_size = 1e-12
    max_iterations = 1000

First, let's consider no regularization. Set the ``l2_penalty`` to
``0.0`` and run your ridge regression algorithm to learn the weights of
your model. Call your weights:

``multiple_weights_0_penalty``

.. code:: python

    l2_penalty = 0.0
    multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)

Next, let's consider high regularization. Set the ``l2_penalty`` to
``1e11`` and run your ridge regression algorithm to learn the weights of
your model. Call your weights:

``multiple_weights_high_penalty``

.. code:: python

    multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 1e11, max_iterations)

Compute the RSS on the TEST data for the following three sets of
weights: 1. The initial weights (all zeros) 2. The weights learned with
no regularization 3. The weights learned with high regularization

Which weights perform best?

.. code:: python

    predictions_initial = predict_output(test_feature_matrix, initial_weights)
    predictions_no_regularization = predict_output(test_feature_matrix, multiple_weights_0_penalty)
    predictions_high_regularization = predict_output(test_feature_matrix, multiple_weights_high_penalty)

.. code:: python

    residuals_initial = predictions_initial - test_output
    residuals_no_regularization = predictions_no_regularization - test_output
    residuals_high_regularization = predictions_high_regularization - test_output

Predict the house price for the 1st house in the test set using the no
regularization and high regularization models. (Remember that python
starts indexing from 0.) How far is the prediction from the actual
price? Which weights perform best for the 1st house?

.. code:: python

    house_price = test_output[0]
    print(house_price)


.. parsed-literal::

    310000.0


.. code:: python

    predict_no_regularization = predictions_no_regularization[0]
    print(predict_no_regularization)


.. parsed-literal::

    387465.476465


.. code:: python

    predict_high_regularization = predictions_high_regularization[0]
    print(predict_high_regularization)


.. parsed-literal::

    270453.530305


***QUIZ QUESTIONS*** 1. What is the value of the coefficient for
``sqft_living`` that you learned with no regularization, rounded to 1
decimal place? What about the one with high regularization? 2. What are
the RSS on the test data for each of the set of weights above (initial,
no regularization, high regularization)? 3. We make prediction for the
first house in the test set using two sets of weights (no regularization
vs high regularization). Which weights make better prediction for that
particular house?

.. code:: python

    multiple_weights_0_penalty




.. parsed-literal::

    array([  -0.35743482,  243.0541689 ,   22.41481594])



.. code:: python

    print("No Regularization `sqft_living`: {0:.1f}".format(multiple_weights_0_penalty[1]))
    print("High Regularization `sqft_living`: {0:.1f}".format(multiple_weights_high_penalty[1]))


.. parsed-literal::

    No Regularization `sqft_living`: 243.1
    High Regularization `sqft_living`: 91.5


.. code:: python

    print("RSS Initial: {0}".format((residuals_initial**2).sum()))
    print("RSS No regularization: {0}".format((residuals_no_regularization**2).sum()))
    print("RSS High Regularization: {0}".format((residuals_high_regularization**2).sum()))


.. parsed-literal::

    RSS Initial: 1.78427328252e+15
    RSS No regularization: 2.74067618287e+14
    RSS High Regularization: 5.0040480058e+14


.. code:: python

    if abs(predict_no_regularization - house_price) > abs(predict_high_regularization - house_price):
        print('High Regularization was closer to the real value')
    else:
        print('no Regularization was closer to the real value')
    print("High Regularization Error: {0}".format(abs(predict_high_regularization - house_price)))
    print("No Regularization Error: {0}".format(abs(predict_no_regularization - house_price)))


.. parsed-literal::

    high regularization was closer to the real value
    High Regularization Error: 39546.4696951
    No Regularization Error: 77465.4764647

