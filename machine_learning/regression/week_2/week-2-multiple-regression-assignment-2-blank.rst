
Regression Week 2: Multiple Regression (gradient descent)
=========================================================

In the first notebook we explored multiple regression using graphlab
create. Now we will use graphlab along with numpy to solve for the
regression weights with gradient descent.

In this notebook we will cover estimating multiple regression weights
via gradient descent. You will: \* Add a constant column of 1's to a
graphlab SFrame to account for the intercept \* Convert an SFrame into a
Numpy array \* Write a predict\_output() function using Numpy \* Write a
numpy function to compute the derivative of the regression weights with
respect to a single feature \* Write gradient descent function to
compute the regression weights given an initial weight vector, step size
and tolerance. \* Use the gradient descent function to estimate
regression weights for multiple features

.. code:: python

    # python standard library
    from math import sqrt
    
    # third-party
    import graphlab
    import numpy as np
    import numpy

Fire up graphlab create
-----------------------

Make sure you have the latest version of graphlab (>= 1.7)

Load in house sales data
------------------------

Dataset is from house sales in King County, the region where the city of
Seattle, WA is located.

.. code:: python

    sales = graphlab.SFrame('large_data/kc_house_data.gl/')

.. code:: python

    class RegressionData(object):
        """
        Breaks a frame into feature and target numpy arrays
        """
        def __init__(self, frame, features, target_feature):
            """
            :param:
             - `frame`: Sframe or data frame with the data
             - `features`: list of features used to predict
             - `target_feature`: name of column to predict
            """
            self._frame = None
            self.frame = frame
            self._features = None
            self.features = features
            self.target_feature = target_feature
            self._features_frame = None
            self._features_matrix = None
            self._target_array = None
            return
    
        @property
        def frame(self):
            """
            :return: source data frame
            """
            return self._frame
    
        @frame.setter
        def frame(self, new_frame):
            """
            adds constant column and saves the frame
            :param:
             - `new_frame`: frame to set
            """
            new_frame['constant'] = 1
            self._frame = new_frame
            return
    
        @property
        def features(self):
            """
            :return: list of prediction features
            """
            return self._features
    
        @features.setter
        def features(self, new_features):
            """
            adds a 'constant' column to and stores the list
            """
            self._features = ['constant'] + new_features
            return
    
        @property
        def features_frame(self):
            if self._features_frame is None:
                self._features_frame = self.frame[self.features]
            return self._features_frame
    
        @property
        def features_matrix(self):
            """
            :return: features frame converted to numpy matrix
            """
            if self._features_matrix is None:
                self._features_matrix = self.features_frame.to_numpy()
            return self._features_matrix
    
        @property
        def target_array(self):
            """
            :return: numpy array of target data
            """
            if self._target_array is None:
                self._target_array = self.frame[self.target_feature].to_numpy()
            return self._target_array
    # end class RegressionData

.. code:: python

    def get_numpy_data(data_sframe, features, output):
        data_sframe['constant'] = 1 # this is how you add a constant column to an SFrame
        # add the column 'constant' to the front of the features list so that we can extract it along with the others:
        features = ['constant'] + features # this is how you combine two lists
        
        # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
        features_sframe = data_SFrame[features]
        
        # the following line will convert the features_SFrame into a numpy matrix:
        feature_matrix = features_sframe.to_numpy()
        
        # assign the column of data_sframe associated with the output to the SArray output_sarray
        output_array = data_sframe[output_array]
        
        # the following will convert the SArray into a numpy array by first converting it to a list
        output_array = output_sarray.to_numpy()
        return(feature_matrix, output_array)

.. code:: python

    class GradientDescent(object):
        def __init__(self, data, initial_weights=None, step_size=None, tolerance=None):
            """
            :param:
            
              - `data`: RegressionData instance
              - `initial_weights`: array of starting coefficients
              - `step_size`: size for each step in the descent
              - `tolerance`: upper bound for allowed error
            """
            self.data = data
            self.initial_weights = initial_weights
            self.step_size = step_size
            self.tolerance = tolerance
            self._weights = None
            return
        
        @property
        def weights(self):
            """
            :return: array of coefficient weights for the model
            """
            if self._weights is None:
                self._weights = self.regression_gradient_descent()
            return self._weights
    
        def predict_output(self, weights):
            """
            calculate vector of predicted outputs
            """
            return self.data.features_matrix.dot(weights)
    
        def feature_derivative(self, errors, feature):
            """
            Both arrays must be of the same size
            :param:
             - `errors`: array of error terms
             - `feature`: array of feature data 
            :return: the derivative of the features array
            """
            return  2 * errors.dot(feature)
    
        def regression_gradient_descent(self):
            """
            :return: vector of weights
            """
            converged = False 
            weights = np.array(self.initial_weights)
    
            while not converged:
                # compute the predictions based on feature_matrix and weights using your predict_output() function
                predictions = self.predict_output(weights)
                
                # compute the errors as predictions - output
                errors = predictions - self.data.target_array
    
                # initialize the gradient sum of squares                                                    
                gradient_sum_squares = 0
    
                # while we haven't reached the tolerance yet, update each feature's weight
                for i in range(len(weights)): # loop over each weight
                    # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
                    # compute the derivative for weight[i]:
                    derivative = self.feature_derivative(errors, self.data.features_matrix[:, i])
    
                    # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
                    gradient_sum_squares += derivative**2
    
                    # subtract the step size times the derivative from the current weight
                    weights[i] -= self.step_size * derivative
    
                    # compute the square-root of the gradient sum of squares to get the gradient magnitude:
                    gradient_magnitude = sqrt(gradient_sum_squares)
                if gradient_magnitude < self.tolerance:
                    converged = True
            return(weights)

If we want to do any "feature engineering" like creating new features or
adjusting existing ones we should do this directly using the SFrames as
seen in the other Week 2 notebook. For this notebook, however, we will
work with the existing features.

Convert to Numpy Array
----------------------

Although SFrames offer a number of benefits to users (especially when
using Big Data and built-in graphlab functions) in order to understand
the details of the implementation of algorithms it's important to work
with a library that allows for direct (and optimized) matrix operations.
Numpy is a Python solution to work with matrices (or any
multi-dimensional "array").

Recall that the predicted value given the weights and the features is
just the dot product between the feature and weight vector. Similarly,
if we put all of the features row-by-row in a matrix then the predicted
value for *all* the observations can be computed by right multiplying
the "feature matrix" by the "weight vector".

First we need to take the SFrame of our data and convert it into a 2D
numpy array (also called a matrix). To do this we use graphlab's built
in .to\_dataframe() which converts the SFrame into a Pandas (another
python library) dataframe. We can then use Panda's .as\_matrix() to
convert the dataframe into a numpy matrix.

Now we will write a function that will accept an SFrame, a list of
feature names (e.g. ['sqft\_living', 'bedrooms']) and a target feature
e.g. ('price') and will return two things:

-  A numpy matrix whose columns are the desired features plus a constant
   column (this is how we create an 'intercept')
-  A numpy array containing the values of the output

With this in mind, complete the following function (where there's an
empty line you should write a line of code that does what the comment
above indicates)

**Please note you will need GraphLab Create version at least 1.7.1 in
order for .to\_numpy() to work!**

.. code:: python

    def sframe_to_numpy(data_sframe, features, target_feature):
        """
        :param:
    
         - `data_sframe`: Sframe of data to fit
         - `features`: list of column names in the data to use
         - `target_feature`: column you are trying to predict
    
        :return: (numpy matrix of feature data, numpy array of target data)
        """
        # this is how you add a constant column to an SFrame
        data_sframe['constant'] = 1
        
        # add the column 'constant' to the front of the features list so that we can extract it along with the others:
        # this is how you prepend an item to the list
        features = ['constant'] + features 
        
        # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
        features_sframe = data_sframe[features]
        
        # the following line will convert the features_SFrame into a numpy matrix:
        feature_matrix = features_sframe.to_numpy()
        
        # assign the column of data_sframe associated with the output to the SArray output_sarray
        output_sarray = data_sframe[target_feature]
        
        # the following will convert the SArray into a numpy array by first converting it to a list
        output_array = output_sarray.to_numpy()
        return(feature_matrix, output_array)

For testing let's use the 'sqft\_living' feature and a constant as our
features and price as our output:

.. code:: python

    expected_row = [1., sales['sqft_living'][0]]
    expected_price = sales['price'][0]

.. code:: python

    # the [] around 'sqft_living' makes it a list
    example_data = RegressionData(frame=sales, features=['sqft_living'],
                             target_feature='price')
    example_features, example_output = sframe_to_numpy(sales, ["sqft_living"],
                                                       'price')
    
    print example_features[0,:] # this accesses the first row of the data the ':' indicates 'all columns'
    print example_output[0] # and the corresponding output
    assert example_data.target_array[0] == example_output[0]


.. parsed-literal::

    [  1.00000000e+00   1.18000000e+03]
    221900.0


Predicting output given regression weights
------------------------------------------

Suppose we had the weights [1.0, 1.0] and the features [1.0, 1180.0] and
we wanted to compute the predicted output 1.0\*1.0 + 1.0\*1180.0 =
1181.0 this is the dot product between these two arrays. If they're
numpy arrayws we can use np.dot() to compute this:

.. code:: python

    my_weights = np.array([1., 1.]) # the example weights
    my_features = example_features[0,] # we'll use the first data point
    predicted_value = np.dot(my_features, my_weights)
    print predicted_value


.. parsed-literal::

    1181.0


.. code:: python

    example_model = GradientDescent(example_data)
    example_model.predict_output(my_weights)[0]




.. parsed-literal::

    1181.0



.. code:: python

    print(my_weights.dot(my_features))
    print(my_features.dot(my_weights))


.. parsed-literal::

    1181.0
    1181.0


np.dot() also works when dealing with a matrix and a vector. Recall that
the predictions from all the observations is just the RIGHT (as in
weights on the right) dot product between the features *matrix* and the
weights *vector*. With this in mind finish the following
``predict_output`` function to compute the predictions for an entire
matrix of features given the matrix and the weights:

.. code:: python

    def predict_output(feature_matrix, weights):
        # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
        # create the predictions vector by using np.dot()
        return feature_matrix.dot(weights)

If you want to test your code run the following cell:

.. code:: python

    test_predictions = predict_output(example_features, my_weights)
    gd_test_predictions = example_model.predict_output(my_weights)
    assert test_predictions[0] == 1181.0
    assert test_predictions[1] == 2571.0
    
    assert gd_test_predictions[0] == 1181.0
    assert gd_test_predictions[1] == 2571
    
    print test_predictions[1] # should be 2571.0
    print test_predictions[0] # should be 1181.0


.. parsed-literal::

    2571.0
    1181.0


Computing the Derivative
------------------------

We are now going to move to computing the derivative of the regression
cost function. Recall that the cost function is the sum over the data
points of the squared difference between an observed output and a
predicted output (RSS?).

Since the derivative of a sum is the sum of the derivatives we can
compute the derivative for a single data point and then sum over data
points. We can write the squared difference between the observed output
and predicted output for a single point as follows:

(w[0]\*[CONSTANT] + w[1]\*[feature\_1] + ... + w[i] \*[feature\_i] + ...
+ w[1]\*[feature\_k] - output)^2

Where we have k features and a constant. So the derivative with respect
to weight w[i] by the chain rule is:

2\*(w[0]\*[CONSTANT] + w[1]\*[feature\_1] + ... + w[i] \*[feature\_i] +
... + w[1]\*[feature\_k] - output)\* [feature\_i]

The term inside the paranethesis is just the error (difference between
prediction and output). So we can re-write this as:

2\*error\*[feature\_i]

That is, the derivative for the weight for feature i is the sum (over
data points) of 2 times the product of the error and the feature itself.
In the case of the constant then this is just twice the sum of the
errors!

Recall that twice the sum of the product of two vectors is just twice
the dot product of the two vectors. Therefore the derivative for the
weight for feature\_i is just two times the dot product between the
values of feature\_i and the current errors.

With this in mind complete the following derivative function which
computes the derivative of the weight given the value of the feature
(over all data points) and the errors (over all data points).

.. code:: python

    def feature_derivative(errors, feature):
        # Assume that errors and feature are both numpy arrays of the same length (number of data points)
        # compute twice the dot product of these vectors as 'derivative' and return the value
        derivative = 2 * errors.dot(feature)
        return(derivative)

To test your feature derivative run the following:

.. code:: python

    
    my_weights = np.array([0., 0.]) # this makes all the predictions 0
    
    gd_predictions = example_model.predict_output(my_weights)
    test_predictions = predict_output(example_features, my_weights)
    
    # just like SFrames 2 numpy arrays can be elementwise subtracted with '-': 
    errors = test_predictions - example_output # prediction errors in this case is just the -example_output
    gd_errors = gd_predictions - example_data.target_array
    
    feature = example_features[:,0] # let's compute the derivative with respect to 'constant', the ":" indicates "all rows"
    gd_feature = example_data.features_matrix[:, 0]
    
    derivative = feature_derivative(errors, feature)
    gd_derivative = example_model.feature_derivative(gd_errors, gd_feature)
    print derivative
    print(gd_derivative)
    print( -(example_output * 2).sum())
    assert derivative == gd_derivative



.. parsed-literal::

    -23345850022.0
    -23345850022.0
    -23345850022.0


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
optimum. We define this by requiring that the magnitude (length) of the
gradient vector to be smaller than a fixed 'tolerance'.

With this in mind, complete the following gradient descent function
below using your derivative function above. For each step in the
gradient descent we update the weight for each feature befofe computing
our stopping criteria

.. code:: python

    def regression_gradient_descent(feature_matrix, target_data, initial_weights, step_size, tolerance):
        """
        :param:
         - `feature_matrix`: matrix of feature-data
         - `target_data`: array of real target values
         - `initial_weights`: array of weights for the regression
         - `step_size`: how much to increment on each iteration
         - `tolerance`: upper bound for allowed error
        :return: vector of weights
        """
        converged = False 
        weights = np.array(initial_weights) # make sure it's a numpy array
    
        while not converged:
            # compute the predictions based on feature_matrix and weights using your predict_output() function
            predictions = predict_output(feature_matrix, weights)
    
            # compute the errors as predictions - output
            errors = predictions - target_data
    
            # initialize the gradient sum of squares                                                    
            gradient_sum_squares = 0 #
    
            # while we haven't reached the tolerance yet, update each feature's weight
            for i in range(len(weights)): # loop over each weight
                # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
                # compute the derivative for weight[i]:
                derivative = feature_derivative(errors, feature_matrix[:, i])
    
                # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
                gradient_sum_squares += derivative**2
    
                # subtract the step size times the derivative from the current weight
                weights[i] -= step_size * derivative
    
            # compute the square-root of the gradient sum of squares to get the gradient magnitude:
            gradient_magnitude = sqrt(gradient_sum_squares)
            if gradient_magnitude < tolerance:
                converged = True
        return(weights)

A few things to note before we run the gradient descent. Since the
gradient is a sum over all the data points and involves a product of an
error and a feature the gradient itself will be very large since the
features are large (squarefeet) and the output is large (prices). So
while you might expect "tolerance" to be small, small is only relative
to the size of the features.

For similar reasons the step size will be much smaller than you might
expect but this is because the gradient has such large values.

Running the Gradient Descent as Simple Regression
-------------------------------------------------

First let's split the data into training and test data.

.. code:: python

    train_data, test_data = sales.random_split(.8,seed=0)

Although the gradient descent is designed for multiple regression since
the constant is now a feature we can use the gradient descent function
to estimat the parameters in the simple regression on squarefeet. The
folowing cell sets up the feature\_matrix, output, initial weights and
step size for the first model:

.. code:: python

    # let's test out the gradient descent
    simple_features = ['sqft_living']
    my_output = 'price'
    initial_weights = np.array([-47000., 1.])
    step_size = 7e-12
    tolerance = 2.5e7
    
    og_1_features, og_1_output = sframe_to_numpy(train_data, simple_features, my_output)
    
    model_1_data = RegressionData(train_data, simple_features, my_output)
    model_1 = GradientDescent(data=model_1_data,
                              initial_weights=initial_weights,
                              step_size=step_size,
                              tolerance = tolerance)


Next run your gradient descent with the above parameters.

.. code:: python

    og_weights = regression_gradient_descent(og_1_features, og_1_output,
                                             initial_weights, step_size, tolerance)
    print(model_1.weights)
    print(og_weights)
    for index, weight in enumerate(model_1.weights):
        assert og_weights[index] == weight,\
            "OG: {0} GD: {1}".format(og_weights[index],
                                     weight)


.. parsed-literal::

    [-46999.88716555    281.91211912]
    [-46999.88716555    281.91211912]


.. code:: python

    week_1 = np.array([-43579.0852515, 280.622770886])
    print(model_1.weights - week_1)
    print(og_weights - week_1)


.. parsed-literal::

    [ -3.42080191e+03   1.28934823e+00]
    [ -3.42080191e+03   1.28934823e+00]


How do your weights compare to those achieved in week 1 (don't expect
them to be exactly the same)?

The intercept differs by 340 dollars while the interect differs by 1.28.

**Quiz Question: What is the value of the weight for sqft\_living -- the
second element of ‘simple\_weights’ (rounded to 1 decimal place)?**

.. code:: python

    print('weight for "sqft_living": {0:.1f}'.format(model_1.weights[1]))
    print("OG: {0:.1f}".format(og_weights[1]))


.. parsed-literal::

    weight for "sqft_living": 281.9
    OG: 281.9


Use your newly estimated weights and your predict\_output() function to
compute the predictions on all the TEST data (you will need to create a
numpy array of the test feature\_matrix and test output first:

.. code:: python

    model_1_test_data = RegressionData(test_data, simple_features, my_output)
    og_test_features, og_test_output = sframe_to_numpy(test_data, simple_features, my_output)

Now compute your predictions using test\_simple\_feature\_matrix and
your weights from above.

.. code:: python

    model_1_test = GradientDescent(model_1_test_data)
    model_1_predictions = model_1_test.predict_output(model_1.weights)
    og_model_1_predictions = predict_output(og_test_features, og_weights)
                                                                                                                         

**Quiz Question: What is the predicted price for the 1st house in the
TEST data set for model 1 (round to nearest dollar)?**

.. code:: python

    predict_model_1_house_1 = model_1_predictions[0]
    print("Predicted price for the first house: $ {0:.0f}".format(predict_model_1_house_1))
    print("OG: {0:.0f}".format(og_model_1_predictions[0]))
    assert predict_model_1_house_1 == og_model_1_predictions[0]


.. parsed-literal::

    Predicted price for the first house: $ 356134
    OG: 356134


Now that you have the predictions on test data, compute the RSS on the
test data set. Save this value for comparison later. Recall that RSS is
the sum of the squared errors (difference between prediction and
output).

.. code:: python

    def residual_sum_of_squares(target, predictions):
        residuals = target - predictions
        return (residuals**2).sum()

.. code:: python

    og_rss = residual_sum_of_squares(og_test_output, og_model_1_predictions)
    rss = residual_sum_of_squares(model_1_test_data.target_array, model_1_predictions)
    assert og_rss == rss

Running a multiple regression
-----------------------------

Now we will use more than one actual feature. Use the following code to
produce the weights for a second model with the following parameters:

.. code:: python

    model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
    my_output = 'price'
    (feature_matrix, model_2_target) = sframe_to_numpy(train_data, model_features, my_output)
    initial_weights = np.array([-100000., 1., 1.])
    step_size = 4e-12
    tolerance = 1e9

Use the above parameters to estimate the model weights. Record these
values for your quiz.

.. code:: python

    multiple_weights = regression_gradient_descent(feature_matrix,
                                                   model_2_target,
                                                   initial_weights,
                                                   step_size,
                                                   tolerance)

Use your newly estimated weights and the predict\_output function to
compute the predictions on the TEST data. Don't forget to create a numpy
array for these features from the test set first!

.. code:: python

    test_features_2, test_target_2 = sframe_to_numpy(test_data, model_features, my_output)
    model_2_predictions = predict_output(test_features_2, multiple_weights)

**Quiz Question: What is the predicted price for the 1st house in the
TEST data set for model 2 (round to nearest dollar)?**

\*\* according to the quiz it is not $ 366651 \*\*

.. code:: python

    predict_model_2_house_1 = model_2_predictions[0]
    print("Predicted price for the first house: $ {0:.0f}".format(predict_model_2_house_1))


.. parsed-literal::

    Predicted price for the first house: $ 366651


What is the actual price for the 1st house in the test data set?

.. code:: python

    actual_test_price_house_1 = test_data['price'][0]
    print("Actual price for the first house in the test data: $ {0:.0f}".format(actual_test_price_house_1))


.. parsed-literal::

    Actual price for the first house in the test data: $ 310000


**Quiz Question: Which estimate was closer to the true price for the 1st
house on the Test data set, model 1 or model 2?**

.. code:: python

    model_1_error = predict_model_1_house_1 - actual_test_price_house_1
    model_2_error = predict_model_2_house_1 - actual_test_price_house_1
    print('Model 1 Error: {0:.0f}'.format(model_1_error))
    print('Model 2 Error: {0:.0f}'.format(model_2_error))
    
                                                                                                                         


.. parsed-literal::

    Model 1 Error: 46134
    Model 2 Error: 56651


Model 1 was closer to the true price for house 1.

Now use your predictions and the output to compute the RSS for model 2
on TEST data.

.. code:: python

    model_2_rss = residual_sum_of_squares(test_target_2, model_2_predictions)
    print(model_2_rss)


.. parsed-literal::

    2.70263446465e+14


**Quiz Question: Which model (1 or 2) has lowest RSS on all of the TEST
data? **

.. code:: python

    print("model 1 RSS: {0:.2f}".format(model_1_rss))
    print("model 2 RSS: {0:.2f}".format(model_2_rss))
    print("model 1 has a higher RSS by {0:.2f}".format(model_1_rss - model_2_rss))
    assert model_2_rss < model_1_rss


.. parsed-literal::

    model 1 RSS: 275400047593155.94
    model 2 RSS: 270263446465244.06
    model 1 has a higher RSS by 5136601127911.88

