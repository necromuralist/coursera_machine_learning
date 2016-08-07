
Regression Week 2: Multiple Regression (Interpretation)
=======================================================

The goal of this first notebook is to explore multiple regression and
feature engineering with existing graphlab functions.

In this notebook you will use data on house sales in King County to
predict prices using multiple regression. You will: \* Use SFrames to do
some feature engineering \* Use built-in graphlab functions to compute
the regression weights (coefficients/parameters) \* Given the regression
weights, predictors and outcome write a function to compute the Residual
Sum of Squares \* Look at coefficients and interpret their meanings \*
Evaluate multiple models via RSS

Imports

.. code:: python

    from collections import OrderedDict
    import graphlab
    from assertions import assert_almost_equal

Fire up graphlab create
-----------------------

Load in house sales data
------------------------

Dataset is from house sales in King County, the region where the city of
Seattle, WA is located.

.. code:: python

    sales = graphlab.SFrame('large_data/kc_house_data.gl/')

Split data into training and testing.
-------------------------------------

We use seed=0 so that everyone running this notebook gets the same
results. In practice, you may set a random seed (or let GraphLab Create
pick a random seed for you).

.. code:: python

    train_data, test_data = sales.random_split(.8,seed=0)

Learning a multiple regression model
------------------------------------

Recall we can use the following code to learn a multiple regression
model predicting 'price' based on the following features:

example\_features = ['sqft\_living', 'bedrooms', 'bathrooms']

on training data with the following code:

(Aside: We set validation\_set = None to ensure that the results are
always the same)

.. code:: python

    example_features = ['sqft_living', 'bedrooms', 'bathrooms']
    example_model = graphlab.linear_regression.create(train_data, target = 'price', features = example_features, 
                                                      validation_set = None)


.. parsed-literal::

    PROGRESS: Linear regression:
    PROGRESS: --------------------------------------------------------
    PROGRESS: Number of examples          : 17384
    PROGRESS: Number of features          : 3
    PROGRESS: Number of unpacked features : 3
    PROGRESS: Number of coefficients    : 4
    PROGRESS: Starting Newton Method
    PROGRESS: --------------------------------------------------------
    PROGRESS: +-----------+----------+--------------+--------------------+---------------+
    PROGRESS: | Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |
    PROGRESS: +-----------+----------+--------------+--------------------+---------------+
    PROGRESS: | 1         | 2        | 0.009576     | 4146407.600631     | 258679.804477 |
    PROGRESS: +-----------+----------+--------------+--------------------+---------------+
    PROGRESS: SUCCESS: Optimal solution found.
    PROGRESS:


Now that we have fitted the model we can extract the regression weights
(coefficients) as an SFrame as follows:

.. code:: python

    example_weight_summary = example_model.get("coefficients")
    print( example_weight_summary)


.. parsed-literal::

    +-------------+-------+----------------+
    |     name    | index |     value      |
    +-------------+-------+----------------+
    | (intercept) |  None | 87910.0724924  |
    | sqft_living |  None | 315.403440552  |
    |   bedrooms  |  None | -65080.2155528 |
    |  bathrooms  |  None | 6944.02019265  |
    +-------------+-------+----------------+
    [4 rows x 3 columns]
    


Making Predictions
------------------

In the gradient descent notebook we used numpy to do our regression. In
this book we will use existing graphlab create functions to analyze
multiple regressions.

Recall that once a model is built we can use the .predict() function to
find the predicted values for data we pass. For example using the
example model above:

.. code:: python

    example_predictions = example_model.predict(train_data)
    print(example_predictions[0])
    expected = 271789.505878
    actual = example_predictions[0]
    tolerance = 0.0000001
    assert_almost_equal(expected, actual, tolerance)



.. parsed-literal::

    271789.505878


Compute RSS
-----------

Now that we can make predictions given the model, let's write a function
to compute the RSS of the model. Complete the function below to
calculate RSS given the model, data, and the outcome.

.. code:: python

    loss = example_model['training_loss']

.. code:: python

    summary = example_model.summary(output='dict')
    rss = summary['sections'][3][0][1]
    print(rss)
    assert loss == rss



.. parsed-literal::

    1.16325455379e+15


.. code:: python

    %%latex
    \begin{align}
    RSS &= \sum_{i=1}^{n} (y_x - f(x_i))^2\\
    \end{align}



.. parsed-literal::

    <IPython.core.display.Latex object>


.. code:: python

    %%writefile regression_functions.py --append
    def residual_sum_of_squares(model, data, target_data, verbose=False):
        """
        Calculate the residuals sum of squares
    
        :param:
         - `model`: model fitted to training data
         - `data`: data to use to make predictions
         - `targe_data`: test data for the column you are predicting
         - `verbose`: whether to print the steps as they go
        """
        if verbose:
            print('getting predictions from data')
        predictions = model.predict(data)
    
        if verbose:
            print("computing the residuals/errors")
        residuals = target_data - predictions
    
        if verbose:
            print("calculating the sum of the squares of the residuals")
        RSS = (residuals**2).sum()
        return(RSS)    


.. parsed-literal::

    Writing regression_functions.py


Test your function by computing the RSS on TEST data for the example
model:

.. code:: python

    rss_example_train = residual_sum_of_squares(example_model, test_data, test_data['price'])
    print(rss_example_train) # should be 2.7376153833e+14
    expected = 2.7376153833 * 10**14
    assert_almost_equal(rss_example_train, expected, tolerance=200)


.. parsed-literal::

    2.7376153833e+14


The tolerance has to be large because the scientific notation only has
10 decimal places so the value in the comment is too imprecise to be
equal to the real value.

Create a New Feature
--------------------

Although we often think of multiple regression as including multiple
different features (e.g. # of bedrooms, squarefeet, and # of bathrooms)
we can also consider transformations of existing features e.g. the log
of the squarefeet or even "interaction" features such as the product of
bedrooms and bathrooms.

You will use the logarithm function to create a new feature. so first
you should import it from the math library.

.. code:: python

    from math import log

Next create the following 4 new features as column in both TEST and
TRAIN data: \* bedrooms\_squared = bedrooms\*bedrooms \*
bed\_bath\_rooms = bedrooms\*bathrooms \* log\_sqft\_living =
log(sqft\_living) \* lat\_plus\_long = lat + long

As an example here's the first one:

.. code:: python

    data = [train_data, test_data]
    
    for frame in data:
         frame['bedrooms_squared'] = frame['bedrooms'].apply(lambda x: x**2)
         frame['log_sqft_living'] = frame['sqft_living'].apply(lambda x: log(x))
         frame['bed_bath_rooms']  = frame['bedrooms'] * frame['bathrooms']
         frame['lat_plus_long'] = frame['lat'] + frame['long']
    assert 'log_sqft_living' in  data[0].column_names()

-  Squaring bedrooms will increase the separation between not many
   bedrooms (e.g. 1) and lots of bedrooms (e.g. 4) since 1^2 = 1 but 4^2
   = 16. Consequently this feature will mostly affect houses with many
   bedrooms.
-  bedrooms times bathrooms gives what's called an "interaction"
   feature. It is large when *both* of them are large.
-  Taking the log of squarefeet has the effect of bringing large values
   closer together and spreading out small values.
-  Adding latitude to longitude is totally non-sensical but we will do
   it anyway (you'll see why)

**Quiz Question: What is the mean (arithmetic average) values of your 4
new features on TEST data? (round to 2 digits)**

.. code:: python

    new_features = ['bedrooms_squared', 'log_sqft_living', 'bed_bath_rooms', 'lat_plus_long']
    print('')
    for feature in new_features:
        print("{0}: {1:.2f}".format(feature, test_data[feature].mean()))


.. parsed-literal::

    
    bedrooms_squared: 12.45
    log_sqft_living: 7.55
    bed_bath_rooms: 7.50
    lat_plus_long: -74.65


Learning Multiple Models
------------------------

Now we will learn the weights for three (nested) models for predicting
house prices. The first model will have the fewest features the second
model will add one more feature and the third will add a few more: \*
Model 1: squarefeet, # bedrooms, # bathrooms, latitude & longitude \*
Model 2: add bedrooms\*bathrooms \* Model 3: Add log squarefeet,
bedrooms squared, and the (nonsensical) latitude + longitude

.. code:: python

    model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
    model_2_features = model_1_features + ['bed_bath_rooms']
    model_3_features = model_2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']

Now that you have the features, learn the weights for the three
different models for predicting target = 'price' using
graphlab.linear\_regression.create() and look at the value of the
weights/coefficients:

.. code:: python

    # Learn the three models: (don't forget to set validation_set = None)
    models = OrderedDict()
    models["model_1"] = graphlab.linear_regression.create(train_data,
                                                          target='price',
                                                          features=model_1_features,
                                                          validation_set=None,
                                                          verbose=False)
    models['model_2'] = graphlab.linear_regression.create(train_data,
                                                          target='price',
                                                          features=model_2_features,
                                                          validation_set=None,
                                                          verbose=False)
    models["model_3"] = graphlab.linear_regression.create(train_data,
                                                          target='price',
                                                          features=model_3_features,
                                                          validation_set=None,
                                                          verbose=False)

Examine/extract each model's coefficients:
------------------------------------------

.. code:: python

    print('')
    for name, model in models.iteritems():
        print(name)
        coefficients = model.get('coefficients')
        print(model.coefficients)


.. parsed-literal::

    
    model_1
    +-------------+-------+----------------+
    |     name    | index |     value      |
    +-------------+-------+----------------+
    | (intercept) |  None | -56140675.7444 |
    | sqft_living |  None | 310.263325778  |
    |   bedrooms  |  None | -59577.1160682 |
    |  bathrooms  |  None | 13811.8405418  |
    |     lat     |  None | 629865.789485  |
    |     long    |  None | -214790.285186 |
    +-------------+-------+----------------+
    [6 rows x 3 columns]
    
    model_2
    +----------------+-------+----------------+
    |      name      | index |     value      |
    +----------------+-------+----------------+
    |  (intercept)   |  None | -54410676.1152 |
    |  sqft_living   |  None | 304.449298057  |
    |    bedrooms    |  None | -116366.043231 |
    |   bathrooms    |  None | -77972.3305135 |
    |      lat       |  None | 625433.834953  |
    |      long      |  None | -203958.60296  |
    | bed_bath_rooms |  None | 26961.6249092  |
    +----------------+-------+----------------+
    [7 rows x 3 columns]
    
    model_3
    +------------------+-------+----------------+
    |       name       | index |     value      |
    +------------------+-------+----------------+
    |   (intercept)    |  None | -52974974.0602 |
    |   sqft_living    |  None | 529.196420564  |
    |     bedrooms     |  None | 28948.5277313  |
    |    bathrooms     |  None |  65661.207231  |
    |       lat        |  None | 704762.148408  |
    |       long       |  None | -137780.01994  |
    |  bed_bath_rooms  |  None | -8478.36410518 |
    | bedrooms_squared |  None | -6072.38466067 |
    | log_sqft_living  |  None | -563467.784269 |
    |  lat_plus_long   |  None | -83217.1979248 |
    +------------------+-------+----------------+
    [10 rows x 3 columns]
    


**Quiz Question: What is the sign (positive or negative) for the
coefficient/weight for 'bathrooms' in model 1?**

.. code:: python

    bathroom_coefficient = models['model_1']['coefficients'][models['model_1']['coefficients']['name'] == 'bathrooms']
    print(bathroom_coefficient)


.. parsed-literal::

    +-----------+-------+---------------+
    |    name   | index |     value     |
    +-----------+-------+---------------+
    | bathrooms |  None | 13811.8405418 |
    +-----------+-------+---------------+
    [? rows x 3 columns]
    Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.
    You can use len(sf) to force materialization.


**Quiz Question: What is the sign (positive or negative) for the
coefficient/weight for 'bathrooms' in model 2?**

Think about what this means.

.. code:: python

    bathroom_coefficient_2 = models['model_2']['coefficients'][models['model_2']['coefficients']['name'] == 'bathrooms']
    print(bathroom_coefficient_2)


.. parsed-literal::

    +-----------+-------+----------------+
    |    name   | index |     value      |
    +-----------+-------+----------------+
    | bathrooms |  None | -77972.3305135 |
    +-----------+-------+----------------+
    [? rows x 3 columns]
    Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.
    You can use len(sf) to force materialization.


The ``bathrooms`` coefficient goes from positive, when it is the only
variable that involves bathrooms, to negative when
``bedrooms * bathrooms`` is added, suggesting that the
``bedrooms * bathrooms`` variable change the predicted price to be
higher than the ``bedrooms`` variable will allow so it changes to
bringing the price down.

Comparing multiple models
-------------------------

Now that you've learned three models and extracted the model weights we
want to evaluate which model is best.

First use your functions from earlier to compute the RSS on TRAINING
Data for each of the three models.

Compute the RSS on TRAINING data for each of the three models and record the values:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    def print_rss(models, data_set):
        for name, model in models.iteritems():
            print('')
            print(name)
            rss = residual_sum_of_squares(model, data_set, data_set['price'])
            print("RSS: {0}".format(rss))
        
    print_rss(models, train_data)



.. parsed-literal::

    
    model_1
    RSS: 9.71328233544e+14
    
    model_2
    RSS: 9.61592067856e+14
    
    model_3
    RSS: 9.05276314555e+14


**Quiz Question: Which model (1, 2 or 3) has lowest RSS on TRAINING
Data?** Is this what you expected?

Model 3 has the lowest rss on the training data, which is what you would
expect, since it has the most features.

Now compute the RSS on on TEST data for each of the three models.

.. code:: python

    # Compute the RSS on TESTING data for each of the three models and record the values:
    print_rss(models, test_data)


.. parsed-literal::

    
    model_1
    RSS: 2.26568089093e+14
    
    model_2
    RSS: 2.24368799994e+14
    
    model_3
    RSS: 2.51829318952e+14


**Quiz Question: Which model (1, 2 or 3) has lowest RSS on TESTING
Data?** Is this what you expected? Think about the features that were
added to each model from the previous.

Model 2 has the lowest RSS when using the testing data. This makes sense
because it adds the ``bedrooms * bathrooms`` variable to model 1 which
added useful information (the combined effect of number bedrooms and
bathrooms) while not adding the non-sensical ``lat + long`` value or
over-emphasizing bathrooms, the way that model 3 does.

model experiment
~~~~~~~~~~~~~~~~

I was curious if the log-sqft-living variable by itself would add
anything.

.. code:: python

    model_4_features = model_2_features + ['log_sqft_living']
    models["model_4"] = graphlab.linear_regression.create(train_data,
                                                          target='price',
                                                          features=model_4_features,
                                                          validation_set=None,
                                                          verbose=False)
    print_rss(models, test_data)


.. parsed-literal::

    
    model_1
    RSS: 2.26568089093e+14
    
    model_2
    RSS: 2.24368799994e+14
    
    model_3
    RSS: 2.51829318952e+14
    
    model_4
    RSS: 2.1416173629e+14


Adding just the 'log\_sqft\_living' variable to model 3 (to create model
4) gives the residual sum of squares.
