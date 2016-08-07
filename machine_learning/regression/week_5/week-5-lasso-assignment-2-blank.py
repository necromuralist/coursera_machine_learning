
# coding: utf-8

# # Regression Week 5: LASSO (coordinate descent)

# In this notebook, you will implement your very own LASSO solver via coordinate descent. You will:
# * Write a function to normalize features
# * Implement coordinate descent for LASSO
# * Explore effects of L1 penalty

# ## Imports

# Make sure you have the latest version of graphlab (>= 1.7)

# In[1]:

# python standard library
import os

# third-party
import graphlab
import numpy as numpy

# this code
import machine_learning
from machine_learning.coursera.regression.common_utilities.numpy_helpers import (get_numpy_data,
                                                                                 predict_output)


# ## Load in house sales data
# 
# Dataset is from house sales in King County, the region where the city of Seattle, WA is located.

# In[2]:

data_path = os.path.join(machine_learning.__path__[0], machine_learning.large_data_path, 'kc_house_data.gl/')
sales = graphlab.SFrame(data_path)
# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to int, before using it below
sales['floors'] = sales['floors'].astype(int) 


# If we want to do any "feature engineering" like creating new features or adjusting existing ones we should do this directly using the SFrames as seen in the first notebook of Week 2. For this notebook, however, we will work with the existing features.

# ## Import useful functions from previous notebook

# As in Week 2, we convert the SFrame into a 2D Numpy array. Copy and paste `get_num_data()` from the second notebook of Week 2.

# Signature: get_numpy_data(data_sframe, features, output)
# Docstring:
# :param:
#  - `data_sframe`: SFrame to convert
#  - `features`: list of column names
#  - `output`: the target
# :return: (matrix of data_sframe columns
#           with 'constant' column and 'features' column,
#           array from data_sframe using 'output' column)
#  
# File:      ~/projects/machine_learning/machine_learning/coursera/regression/common_utilities/numpy_helpers.py
# Type:      function

# Also, copy and paste the `predict_output()` function to compute the predictions for an entire matrix of features given the matrix and the weights:

# Signature: predict_output(feature_matrix, weights)
# Docstring:
# :param:
#  - `feature_matrix`:numpy matrix containing the features
#  - `weights`: vector to apply to the feature_matrix
# 
# :return: dot-product of feature_matrix and weights
# File:      ~/projects/machine_learning/machine_learning/coursera/regression/common_utilities/numpy_helpers.py
# Type:      function

# ## Normalize features
# In the house dataset, features vary wildly in their relative magnitude: `sqft_living` is very large overall compared to `bedrooms`, for instance. As a result, weight for `sqft_living` would be much smaller than weight for `bedrooms`. This is problematic because "small" weights are dropped first as `l1_penalty` goes up. 
# 
# To give equal considerations for all features, we need to **normalize features** as discussed in the lectures: we divide each feature by its 2-norm so that the transformed feature has norm 1.
# 
# Let's see how we can do this normalization easily with Numpy: let us first consider a small matrix.

# In[3]:

X = numpy.array([[3.,5.,8.],[4.,12.,15.]])
print(X)


# Numpy provides a shorthand for computing 2-norms of each column:

# In[4]:

norms = numpy.linalg.norm(X, axis=0) # gives [norm(X[:,0]), norm(X[:,1]), norm(X[:,2])]
print(norms)


# In[5]:

def normalize(x):
    return numpy.sqrt((x**2).sum())


# In[6]:

X.shape


# In[7]:

for i in range(X.shape[1]):
    print(normalize(X[:, i]))


# To normalize, apply element-wise division:

# In[8]:

print( X / norms) # gives [X[:,0]/norm(X[:,0]), X[:,1]/norm(X[:,1]), X[:,2]/norm(X[:,2])]


# Using the shorthand we just covered, write a short function called `normalize_features(feature_matrix)`, which normalizes columns of a given feature matrix. The function should return a pair `(normalized_features, norms)`, where the second item contains the norms of original features. As discussed in the lectures, we will use these norms to normalize the test data in the same way as we normalized the training data. 

# In[9]:

def normalize_features(feature_matrix):
    """
    :param:
     - `feature_matrix`: numpy array to be normalized (along columns)
    :return: (normalized feature_matrix, matrix of norms for feature_matrix)
    """
    norms = numpy.linalg.norm(feature_matrix, axis=0)
    return feature_matrix/norms, norms


# To test the function, run the following:

# In[10]:

features, norms = normalize_features(numpy.array([[3.,6.,9.],[4.,8.,12.]]))
print(features)

expected_features = numpy.array( [[ 0.6,  0.6,  0.6],
                                  [ 0.8,  0.8,  0.8]])
assert (features == expected_features).all()
print(norms)

expected_norms = numpy.array( [5.,  10.,  15.])
assert (expected_norms == norms).all()


# ## Implementing Coordinate Descent with normalized features

# We seek to obtain a sparse set of weights by minimizing the LASSO cost function
# ```
# SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|).
# ```
# (By convention, we do not include `w[0]` in the L1 penalty term. We never want to push the intercept to zero.)
# 
# The absolute value sign makes the cost function non-differentiable, so simple gradient descent is not viable (you would need to implement a method called subgradient descent). Instead, we will use **coordinate descent**: at each iteration, we will fix all weights but weight `i` and find the value of weight `i` that minimizes the objective. That is, we look for
# ```
# argmin_{w[i]} [ SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|) ]
# ```
# where all weights other than `w[i]` are held to be constant. We will optimize one `w[i]` at a time, circling through the weights multiple times.  
#   1. Pick a coordinate `i`
#   2. Compute `w[i]` that minimizes the cost function `SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|)`
#   3. Repeat Steps 1 and 2 for all coordinates, multiple times

# For this notebook, we use **cyclical coordinate descent with normalized features**, where we cycle through coordinates 0 to (d-1) in order, and assume the features were normalized as discussed above. The formula for optimizing each coordinate is as follows:
# ```
#        ┌ (ro[i] + lambda/2)     if ro[i] < -lambda/2
# w[i] = ├ 0                      if -lambda/2 <= ro[i] <= lambda/2
#        └ (ro[i] - lambda/2)     if ro[i] > lambda/2
# ```
# where
# ```
# ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ].
# ```
# 
# Note that we do not regularize the weight of the constant feature (intercept) `w[0]`, so, for this weight, the update is simply:
# ```
# w[0] = ro[i]
# ```

# ## Effect of L1 penalty

# Let us consider a simple model with 2 features:

# In[11]:

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)


# Don't forget to normalize features:

# In[12]:

simple_feature_matrix, norms = normalize_features(simple_feature_matrix)


# We assign some random set of initial weights and inspect the values of `ro[i]`:

# In[13]:

weights = numpy.array([1., 4., 1.])


# Use `predict_output()` to make predictions on this data.

# In[14]:

prediction = predict_output(simple_feature_matrix, weights)


# Compute the values of `ro[i]` for each feature in this simple model, using the formula given above, using the formula:
# ```
# ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ]
# ```
# 
# *Hint: You can get a Numpy vector for feature_i using:*
# ```
# simple_feature_matrix[:,i]
# ```

# In[15]:

simple_feature_matrix[:,0]


# In[16]:

output - prediction + weights[0] * simple_feature_matrix[:, 0]


# In[17]:

ro = []
for column in range(simple_feature_matrix.shape[1]):
    ro.append((simple_feature_matrix[:, column] * (output - prediction + weights[column] * simple_feature_matrix[:, column])).sum())


# ***QUIZ QUESTION***
# 
# Recall that, whenever `ro[i]` falls between `-l1_penalty/2` and `l1_penalty/2`, the corresponding weight `w[i]` is sent to zero. Now suppose we were to take one step of coordinate descent on either feature 1 or feature 2. What range of values of `l1_penalty` **would not** set `w[1]` zero, but **would** set `w[2]` to zero, if we were to take a step in that coordinate? 

# -l1_penalty < 2ro < l1_penalty

# In[18]:

def zero_weight(value, l1_penalty):
    if -l1_penalty/2 < value < l1_penalty/2:
        return 0
    return value


# In[19]:

base_penalty_1 = ro[1] * 2
penalty = base_penalty_1 + .00001
print("r0[1]: {0}".format(zero_weight(ro[1], penalty)))
print("r0[2]: {0}".format(zero_weight(ro[2], penalty)))
print("\nl1 penalty > {0:.2f}".format(base_penalty_1))


# In[20]:

base_penalty_2 = ro[2] * 2

penalty = base_penalty_2 + .00001

print("r0[1]: {0}".format(zero_weight(ro[1], penalty)))
print("r0[2]: {0}".format(zero_weight(ro[2], penalty)))
print("\nl1 penalty > {0:.2f}".format(base_penalty_2))


# In[21]:

print("{0:.2f} < l1 penalty < {1:.2f}".format(base_penalty_2,
                                              base_penalty_1))


# In[83]:

# options are the quiz multiple-choice options
options = [1.4e8, 1.64e8, 1.73e8, 1.9e8, 2.3e8]
print("option\t\tvalid")
print('-----------------------')
for option in options:
    print("{0:.2e}\t{1}".format(option,
                            base_penalty_1 > option > base_penalty_2))


# ***QUIZ QUESTION***
# 
# What range of values of `l1_penalty` would set **both** `w[1]` and `w[2]` to zero, if we were to take a step in that coordinate? 

# In[22]:

penalty = base_penalty_1 + .000001
print("r0[1]: {0}".format(zero_weight(ro[1], penalty)))
print("r0[2]: {0}".format(zero_weight(ro[2], penalty)))
print("\nl1 penalty > {0:.2f}".format(base_penalty_1))


# In[84]:

print("option\t\tvalid")
print('-----------------------')
for option in options:
    print("{0:.2e}\t{1}".format(option,
                                option > base_penalty_1))


# So we can say that `ro[i]` quantifies the significance of the i-th feature: the larger `ro[i]` is, the more likely it is for the i-th feature to be retained.

# ## Single Coordinate Descent Step

# Using the formula above, implement coordinate descent that minimizes the cost function over a single feature i. Note that the intercept (weight 0) is not regularized. The function should accept feature matrix, output, current weights, l1 penalty, and index of feature to optimize over. The function should return new weight for feature i.

# In[23]:

def lasso_coordinate_descent_step(column_index, feature_matrix, output, weights, l1_penalty):
    """
    Regularizes the weight based on:
               ┌ (ro[i] + lambda/2)     if ro[i] < -lambda/2
        w[i] = ├ 0                      if -lambda/2 <= ro[i] <= lambda/2
               └ (ro[i] - lambda/2)     if ro[i] > lambda/2

    where
        ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ].

    The intercept isn't regularized so:
        w[0] = ro[i]
    
    :param:
     - `column_index`: index of column to regularize
     - `feature_matrix`: numpy array with column to regularize
     - `output`: vector of target values
     - `weights`: regression weights
     - `l1_penalty`: regression penalty (lambda)
    :return: new (regularized) weight for column
    """
    # compute prediction
    prediction = predict_output(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = (feature_matrix[:, column_index] * (output - prediction +
                                               weights[column_index] *
                                               feature_matrix[:, column_index])).sum()
    
    half_lambda = l1_penalty/2.
    if column_index == 0: # intercept -- do not regularize
        new_weight_i = ro_i 
    elif ro_i < -half_lambda:
        new_weight_i = ro_i + half_lambda
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - half_lambda
    else:
        new_weight_i = 0.
    
    return new_weight_i


# To test the function, run the following cell:

# In[24]:

expected = 0.425558846691
# we have 12 decimal places in the expected
# so the upper bound on the difference is 1 x 10^{-13}
delta = 1e-13
import math
actual = lasso_coordinate_descent_step(1, numpy.array([[3./math.sqrt(13),1./math.sqrt(10)],[2./math.sqrt(13),3./math.sqrt(10)]]), 
                                       numpy.array([1., 1.]), numpy.array([1., 4.]), 0.1)
print(actual)
assert abs(expected- actual) < delta, "Difference: {0}".format(abs(expected-actual))


# ## Cyclical coordinate descent 

# Now that we have a function that optimizes the cost function over a single coordinate, let us implement cyclical coordinate descent where we optimize coordinates 0, 1, ..., (d-1) in order and repeat.
# 
# When do we know to stop? Each time we scan all the coordinates (features) once, we measure the change in weight for each coordinate. If no coordinate changes by more than a specified threshold, we stop.

# For each iteration:
# 1. As you loop over features in order and perform coordinate descent, measure how much each coordinate changes.
# 2. After the loop, if the maximum change across all coordinates is falls below the tolerance, stop. Otherwise, go back to step 1.
# 
# Return weights
# 
# **IMPORTANT: when computing a new weight for coordinate i, make sure to incorporate the new weights for coordinates 0, 1, ..., i-1. One good way is to update your weights variable in-place. See following pseudocode for illustration.**
# ```
# for i in range(len(weights)):
#     old_weights_i = weights[i] # remember old value of weight[i], as it will be overwritten
#     # the following line uses new values for weight[0], weight[1], ..., weight[i-1]
#     #     and old values for weight[i], ..., weight[d-1]
#     weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
#     
#     # use old_weights_i to compute change in coordinate
#     ...
# ```

# In[25]:

def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights,
                                      l1_penalty, tolerance):
    """
    optimizes coordinates in order until threshold is reached

    For each iteration:

        1. loop over features in order and perform coordinate descent
           1.1 measure how much each coordinate changes.
        2. check if the maximum change across all coordinates is below tolerance
           2.1 if so, stop. 
           2.2 Otherwise, go back to step 1.

    :param:
     - `feature_matrix`: numpy matrix of feature-data
     - `output`: vector with the target data
     - `initial_weights`: vector of weights to start the descent
     - `l1_penalty`: regression penalty (lambda)
     - `tolerance`: threshold for change to decide when to stop

    :return: vector of new weights
    """
    weights = initial_weights.copy()
    maximum_change = tolerance + 1
    while maximum_change > tolerance:
        changes = []
        for feature in range(len(weights)):
            old_weights = weights[feature]
            weights[feature] = lasso_coordinate_descent_step(column_index=feature,
                                                             feature_matrix=feature_matrix,
                                                             output=output,
                                                             weights=weights,
                                                             l1_penalty=l1_penalty)
            changes.append(abs(old_weights - weights[feature]))
        maximum_change = max(changes)
    return weights
    


# Using the following parameters, learn the weights on the sales dataset. 

# In[86]:

class LassoRegression(object):
    """
    Lasso Regression with coordinate descent
    """
    def __init__(self, features, training_data, tolerance, l1_penalty, target='price'):
        """
        :param:
         - `features`: list of features to train on
         - `training_data`: data to train
         - `tolerance`: tolerance to stop coordinate descent
         - `l1_penalty`: penalty for regression
         - `target`: feature to predict
        """
        self.features = features
        self.training_data = training_data
        self.tolerance = tolerance
        self.l1_penalty = l1_penalty
        self.target = target
        self.reset()
        return

    def __str__(self):
        """
        return string of given values (except data)
        """
        return "features: {0}\ntolerance: {1}\nL1 Penalty: {2}\nTarget: '{3}'".format(','.join(self.features),
                                                                                      self.tolerance,
                                                                                      self.l1_penalty,
                                                                                      self.target)

    @property
    def feature_matrix(self):
        """
        :return: numpy matrix
        """
        if self._feature_matrix is None:
            self._feature_matrix, self._output = get_numpy_data(self.training_data,
                                                                self.features,
                                                                self.target)
        return self._feature_matrix

    @property
    def output(self):
        """
        :return: vector of target data
        """
        if self._output is None:
            self._feature_matrix, self._output = get_numpy_data(self.training_data,
                                                                self.features,
                                                                self.target)
        return self._output

    @property
    def normalized_feature_matrix(self):
        """
        :return: normalized self.feature_matrix
        """
        if self._normalized_feature_matrix is None:
            self._normalized_feature_matrix = self.feature_matrix/self.norms
        return self._normalized_feature_matrix

    @property
    def norms(self):
        """
        :return: vector of norms used to create normalized features
        """
        if self._norms is None:
            self._norms = numpy.linalg.norm(self.feature_matrix, axis=0)
        return self._norms

    @property
    def initial_weights(self):
        """
        :return: vector of zeros for coordinate descent
        """
        if self._initial_weights is None:
            # add one for the intercept
            self._initial_weights = numpy.zeros(len(self.features) + 1)
        return self._initial_weights

    @property
    def weights(self):
        """
        :return: vector of feature-weights
        """
        if self._weights is None:
            self._weights = lasso_cyclical_coordinate_descent(feature_matrix=self.normalized_feature_matrix,
                                                              output=self.output,
                                                              initial_weights=self.initial_weights,
                                                              l1_penalty=self.l1_penalty,
                                                              tolerance=self.tolerance)
        return self._weights

    @property
    def weights_normalized(self):
        """
        :return: vector of weights/norms
        """
        if self._weights_normalized is None:
            self._weights_normalized = self.weights/self.norms
        return self._weights_normalized
    
    def filter_features(self, comparison=lambda weight: weight==0):
        """
        :param:
         - `comparison`: boolean function to filter the features (using weights)
        :return: list of filtered features,weights 
        """
        return [(feature, self.weights[index])
                for index, feature in enumerate(['intercept'] + self.features)
                                      if comparison(self.weights[index])]
    @property    
    def non_zero_weights(self):
        """
        :return: list of the features-weights that had non-zero weights
        """
        if self._non_zero_weights is None:
            self._non_zero_weights = self.filter_features(comparison=lambda weight: weight!=0)
        return self._non_zero_weights

    @property
    def zero_weights(self):
        """
        :return: list of features,weights for zero-weighted features
        """
        if self._zero_weights is None:
            self._zero_weights = self.filter_features(lambda w: w==0)
        return self._zero_weights

    def lasso_cyclical_coordinate_descent(self):
        """
        optimizes coordinates in order until threshold is reached
        
        For each iteration:
        
            1. loop over features in order and perform coordinate descent
               1.1 measure how much each coordinate changes.
            2. check if the maximum change across all coordinates is below tolerance
               2.1 if so, stop. 
               2.2 Otherwise, go back to step 1.
        
        :return: vector of new weights
        """
        weights = self.initial_weights.copy()
        maximum_change = self.tolerance + 1
        while maximum_change > self.tolerance:
            changes = []
            for feature in range(len(weights)):
                old_weights = weights[feature]
                weights[feature] = self.lasso_coordinate_descent_step(column_index=feature,
                                                                      weights=weights)
                changes.append(abs(old_weights - weights[feature]))
            maximum_change = max(changes)
        return weights

    def lasso_coordinate_descent_step(self, column_index, weights):
        """
        Regularizes the weight based on:
                   ┌ (ro[i] + lambda/2)     if ro[i] < -lambda/2
            w[i] = ├ 0                      if -lambda/2 <= ro[i] <= lambda/2
                   └ (ro[i] - lambda/2)     if ro[i] > lambda/2
        
        where
            ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ].
        
        The intercept isn't regularized so:
            w[0] = ro[i]
        
        :param:
         - `column_index`: index of column to regularize
         - `weights`: regression weights
        :return: new (regularized) weight for column
        """
        feature_matrix = self.normalized_feature_matrix
        # compute prediction        
        prediction = predict_output(feature_matrix, weights)
        # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
        ro_i = (feature_matrix[:, column_index] * (self.output - prediction +
                                                   weights[column_index] *
                                                   feature_matrix[:, column_index])).sum()
        
        half_lambda = self.l1_penalty/2.
        
        if column_index == 0: # intercept -- do not regularize
            new_weight_i = ro_i 
        elif ro_i < -half_lambda:
            new_weight_i = ro_i + half_lambda
        elif ro_i > l1_penalty/2.:
            new_weight_i = ro_i - half_lambda
        else:
            new_weight_i = 0.
        return new_weight_i

    def residual_sum_of_squares(self, data, output=None, data_is_normalized=False):
        """
        :param:
         - `data`: array of data to predict outcome
         - `output`: target to compare predictions to
         - `data_is_normalized`: if True use un-normalized weights
        :return: RSS for predctions using data
        """
        if data_is_normalized:
            weights = self.weights
        else:
            weights = self.weights_normalized
        if output is None:
            output = self.output
                
        prediction = predict_output(data, weights)
        return ((prediction - output)**2).sum()

    def print_feature_weights(self, zeros=False, non_zero_weights=False):
        """
        print out feature, weight

         - if both or neither parameter is true, show all the weights

        :param:
         - `zeros`: if true, only show zeros
         - `non_zero_weights`: if True only non-zero-weights
        """
        longest = max(max(len(feature) for feature in self.features), len('intercept'))
        output_string = '{{f:<{longest}}}\t{{w:.2f}}'.format(longest=longest)
        header = '{{f:<{l}}}'.format(l=longest).format(f='Feature')  + 'Weight'
        print(header)
        print('-' * len(header))
        if all((zeros, non_zero_weights)) or not any((zeros, non_zero_weights)):
            features = self.filter_features(comparison=lambda weight: True)
        elif zeros:
            features = self.zero_weights
        else:
            features = self.non_zero_weights
        for feature, weight in features:
            print(output_string.format(f=feature, w=weight))
        return

    def reset(self):
        self._feature_matrix = None
        self._feature_matrix = None
        self._output = None
        self._normalized_feature_matrix = None
        self._norms = None
        self._initial_weights = None
        self._weights = None
        self._weights_normalized = None
        self._non_zero_weights = None
        self._zero_weights = None
        return
# end class LassoRegression


# In[27]:

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
initial_weights = numpy.zeros(3)
l1_penalty = 1e7
tolerance = 1.0


# In[87]:

simple_lasso = LassoRegression(features=simple_features,
                               training_data=sales,
                               tolerance=tolerance,
                               l1_penalty=l1_penalty)


# First create a normalized version of the feature matrix, `normalized_simple_feature_matrix`

# In[36]:

(simple_feature_matrix, simple_output) = get_numpy_data(sales, simple_features, my_output)
(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix) # normalize features


# In[37]:

print(normalized_simple_feature_matrix.shape)
print(simple_lasso.normalized_feature_matrix.shape)


# Then, run your implementation of LASSO coordinate descent:

# In[38]:

simple_weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, simple_output,
                                                   initial_weights, l1_penalty, tolerance)


# In[39]:

simple_weights


# In[88]:

simple_lasso.reset()
assert (simple_lasso.weights == simple_weights).all(), simple_lasso.weights


# ***QUIZ QUESTIONS***
# 1. What is the RSS of the learned model on the normalized dataset?
# 2. Which features had weight zero at convergence?

# In[89]:

prediction = predict_output(normalized_simple_feature_matrix, simple_weights)
rss = ((prediction - simple_output)**2).sum()
print( "RSS: {0:.2e}".format(rss))


# In[90]:

lasso_rss = simple_lasso.residual_sum_of_squares(normalized_simple_feature_matrix, data_is_normalized=True)
print("RSS: {0:.2e}".format(lasso_rss))
assert rss == lasso_rss


# In[43]:

def filter_features(features, weights, comparison=lambda x: x==0):
    """
    :param:
     - `features`: vector of feature names (without 'intercept')
     - `weights`: vector of weights for the features (including intercept)
     - `comparison`: boolean function to filter (in) features
    :return: generator of filtered features, weights
    """
    return((feature, weights[index]) for index, feature in enumerate(['intercept'] + features)
           if comparison(weights[index]))
            
for feature in filter_features(simple_features, simple_weights):
    print("Zero-Weight Feature: {0}".format(feature))


# In[44]:

simple_lasso.print_feature_weights(zeros=True)


# In[45]:

simple_lasso.print_feature_weights(non_zero_weights=True)


# ## Evaluating LASSO fit with more features

# Let us split the sales dataset into training and test sets.

# In[46]:

from collections import namedtuple

TrainTestData = namedtuple('TrainTestData', 'training testing'.split())


# In[47]:

train_data,test_data = sales.random_split(.8,seed=0)
train_test_data = TrainTestData(training=train_data,
                                testing=test_data)


# Let us consider the following set of features.

# In[48]:

all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront', 
                'view', 
                'condition', 
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 
                'yr_renovated']


# First, create a normalized feature matrix from the TRAINING data with these features.  (Make you store the norms for the normalization, since we'll use them later)

# In[49]:

feature_matrix, all_output = get_numpy_data(train_test_data.training, all_features, my_output)
normalized_feature_matrix, all_norms = normalize_features(feature_matrix)


# First, learn the weights with `l1_penalty=1e7`, on the training data. Initialize weights to all zeros, and set the `tolerance=1`.  Call resulting weights `weights1e7`, you will need them later.

# In[50]:

initial_weights = numpy.zeros(len(all_features) + 1)
tolerance = 1
l1_penalty = 1e7


# In[51]:

lasso_1e7 = LassoRegression(features=all_features,
                            training_data=train_test_data.training,
                            tolerance=tolerance,
                            l1_penalty=1e7)


# In[52]:

all_weights = lasso_cyclical_coordinate_descent(normalized_feature_matrix, all_output,
                                                initial_weights, l1_penalty, tolerance)


# In[53]:

print(' + '.join(["{0:.2e} x".format(w) for w in all_weights]))


# ***QUIZ QUESTION***
# 
# What features had non-zero weight in this case?

# In[54]:

lasso_1e7.print_feature_weights(non_zero_weights=True)


# In[55]:

lasso_1e7.print_feature_weights(zeros=True)


# Next, learn the weights with `l1_penalty=1e8`, on the training data. Initialize weights to all zeros, and set the `tolerance=1`.  Call resulting weights `weights1e8`, you will need them later.

# In[56]:

lasso_1e8 = LassoRegression(features=all_features,
                            training_data=train_test_data.training,
                            tolerance=1,
                            l1_penalty=1e8)


# ***QUIZ QUESTION***
# What features had non-zero weight in this case?

# In[57]:

lasso_1e8.print_feature_weights(non_zero_weights=True)


# In[58]:

lasso_1e8.print_feature_weights(zeros=True)


# Finally, learn the weights with `l1_penalty=1e4`, on the training data. Initialize weights to all zeros, and set the `tolerance=5e5`.  Call resulting weights `weights1e4`, you will need them later.  (This case will take quite a bit longer to converge than the others above.)

# In[91]:

lasso_1e4 = LassoRegression(features=all_features,
                            training_data=train_test_data.training,
                            tolerance=5e5,
                            l1_penalty=1e4)
print(str(lasso_1e4))


# ***QUIZ QUESTION***
# 
# What features had non-zero weight in this case?

# In[92]:

lasso_1e4.print_feature_weights(non_zero_weights=True)


# In[93]:

lasso_1e4.print_feature_weights(zeros=True)


# ## Rescaling learned weights

# Recall that we normalized our feature matrix, before learning the weights.  To use these weights on a test set, we must normalize the test data in the same way.
# 
# Alternatively, we can rescale the learned weights to include the normalization, so we never have to worry about normalizing the test data: 
# 
# In this case, we must scale the resulting weights so that we can make predictions with *original* features:
#  1. Store the norms of the original features to a vector called `norms`:
# ```
# features, norms = normalize_features(features)
# ```
#  2. Run Lasso on the normalized features and obtain a `weights` vector
#  3. Compute the weights for the original features by performing element-wise division, i.e.
# ```
# weights_normalized = weights / norms
# ```
# Now, we can apply `weights_normalized` to the test data, without normalizing it!

# Create a normalized version of each of the weights learned above. (`weights1e4`, `weights1e7`, `weights1e8`).

# In[73]:

expected = 161.31745624837794
actual = lasso_1e7.weights_normalized[3]

# expected has 14 significant digits (allow two decimal points for the rounding)
delta = 1e-12
assert abs(expected - actual) < delta, "Difference: {0}".format(abs(actual - expected))


# To check your results, if you call `normalized_weights1e7` the normalized version of `weights1e7`, then:
# ```
# print normalized_weights1e7[3]
# ```
# should return 161.31745624837794.

# ## Evaluating each of the learned models on the test data

# Let's now evaluate the three models on the test data:

# In[74]:

(test_feature_matrix, test_output) = get_numpy_data(train_test_data.training, all_features, 'price')


# Compute the RSS of each of the three normalized weights on the (unnormalized) `test_feature_matrix`:

# In[94]:

models = {'1e7': lasso_1e7,
          '1e8': lasso_1e8,
          '1e4': lasso_1e4}
outcomes = {name: model.residual_sum_of_squares(data=test_feature_matrix, output=test_output)
            for name, model in models.iteritems()}


# In[95]:

header = "L1_Penalty RSS"
print(header)
print('-' * len(header))
for name, value in outcomes.iteritems():
    print("{0}\t{1:.2}".format(name, value))


# In[96]:

min_rss = min(outcomes.values())
for name, value in outcomes.iteritems():
    if value == min_rss:
        print("{0}\t{1}".format(name, value))


# ***QUIZ QUESTION***
# 
# Which model performed best on the test data?
# 
# 1e4
