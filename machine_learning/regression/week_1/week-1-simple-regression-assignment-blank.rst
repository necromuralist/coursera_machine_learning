Regression Week 1: Simple Linear Regression
===========================================

In this notebook we will use data on house sales in King County to
predict house prices using simple (one input) linear regression. You
will: \* Use graphlab SArray and SFrame functions to compute important
summary statistics \* Write a function to compute the Simple Linear
Regression weights using the closed form solution \* Write a function to
make predictions of the output given the input feature \* Turn the
regression around to predict the input given the output \* Compare two
different models for predicting house prices

In this notebook you will be provided with some already complete code as
well as some code that you should complete yourself in order to answer
quiz questions. The code we provide to complete is optional and is there
to assist you with solving the problems but feel free to ignore the
helper code and write your own.

Fire up graphlab create
-----------------------

.. code:: python

    import graphlab
    import numpy
    import matplotlib.pyplot as plt
    import seaborn

.. code:: python

    %matplotlib inline

Load house sales data
---------------------

Dataset is from house sales in King County, the region where the city of
Seattle, WA is located.

.. code:: python

    sales = graphlab.SFrame('data/kc_house_data.gl/')
    print(sales.column_names)


.. parsed-literal::

    <bound method SFrame.column_names of Columns:
    	id	str
    	date	datetime
    	price	float
    	bedrooms	float
    	bathrooms	float
    	sqft_living	float
    	sqft_lot	int
    	floors	str
    	waterfront	int
    	view	int
    	condition	int
    	grade	int
    	sqft_above	int
    	sqft_basement	int
    	yr_built	int
    	yr_renovated	int
    	zipcode	str
    	lat	float
    	long	float
    	sqft_living15	float
    	sqft_lot15	float
    
    Rows: 21613
    
    Data:
    +------------+---------------------------+-----------+----------+-----------+
    |     id     |            date           |   price   | bedrooms | bathrooms |
    +------------+---------------------------+-----------+----------+-----------+
    | 7129300520 | 2014-10-13 00:00:00+00:00 |  221900.0 |   3.0    |    1.0    |
    | 6414100192 | 2014-12-09 00:00:00+00:00 |  538000.0 |   3.0    |    2.25   |
    | 5631500400 | 2015-02-25 00:00:00+00:00 |  180000.0 |   2.0    |    1.0    |
    | 2487200875 | 2014-12-09 00:00:00+00:00 |  604000.0 |   4.0    |    3.0    |
    | 1954400510 | 2015-02-18 00:00:00+00:00 |  510000.0 |   3.0    |    2.0    |
    | 7237550310 | 2014-05-12 00:00:00+00:00 | 1225000.0 |   4.0    |    4.5    |
    | 1321400060 | 2014-06-27 00:00:00+00:00 |  257500.0 |   3.0    |    2.25   |
    | 2008000270 | 2015-01-15 00:00:00+00:00 |  291850.0 |   3.0    |    1.5    |
    | 2414600126 | 2015-04-15 00:00:00+00:00 |  229500.0 |   3.0    |    1.0    |
    | 3793500160 | 2015-03-12 00:00:00+00:00 |  323000.0 |   3.0    |    2.5    |
    +------------+---------------------------+-----------+----------+-----------+
    +-------------+----------+--------+------------+------+-----------+-------+------------+
    | sqft_living | sqft_lot | floors | waterfront | view | condition | grade | sqft_above |
    +-------------+----------+--------+------------+------+-----------+-------+------------+
    |    1180.0   |   5650   |   1    |     0      |  0   |     3     |   7   |    1180    |
    |    2570.0   |   7242   |   2    |     0      |  0   |     3     |   7   |    2170    |
    |    770.0    |  10000   |   1    |     0      |  0   |     3     |   6   |    770     |
    |    1960.0   |   5000   |   1    |     0      |  0   |     5     |   7   |    1050    |
    |    1680.0   |   8080   |   1    |     0      |  0   |     3     |   8   |    1680    |
    |    5420.0   |  101930  |   1    |     0      |  0   |     3     |   11  |    3890    |
    |    1715.0   |   6819   |   2    |     0      |  0   |     3     |   7   |    1715    |
    |    1060.0   |   9711   |   1    |     0      |  0   |     3     |   7   |    1060    |
    |    1780.0   |   7470   |   1    |     0      |  0   |     3     |   7   |    1050    |
    |    1890.0   |   6560   |   2    |     0      |  0   |     3     |   7   |    1890    |
    +-------------+----------+--------+------------+------+-----------+-------+------------+
    +---------------+----------+--------------+---------+-------------+
    | sqft_basement | yr_built | yr_renovated | zipcode |     lat     |
    +---------------+----------+--------------+---------+-------------+
    |       0       |   1955   |      0       |  98178  | 47.51123398 |
    |      400      |   1951   |     1991     |  98125  | 47.72102274 |
    |       0       |   1933   |      0       |  98028  | 47.73792661 |
    |      910      |   1965   |      0       |  98136  |   47.52082  |
    |       0       |   1987   |      0       |  98074  | 47.61681228 |
    |      1530     |   2001   |      0       |  98053  | 47.65611835 |
    |       0       |   1995   |      0       |  98003  | 47.30972002 |
    |       0       |   1963   |      0       |  98198  | 47.40949984 |
    |      730      |   1960   |      0       |  98146  | 47.51229381 |
    |       0       |   2003   |      0       |  98038  | 47.36840673 |
    +---------------+----------+--------------+---------+-------------+
    +---------------+---------------+-----+
    |      long     | sqft_living15 | ... |
    +---------------+---------------+-----+
    | -122.25677536 |     1340.0    | ... |
    |  -122.3188624 |     1690.0    | ... |
    | -122.23319601 |     2720.0    | ... |
    | -122.39318505 |     1360.0    | ... |
    | -122.04490059 |     1800.0    | ... |
    | -122.00528655 |     4760.0    | ... |
    | -122.32704857 |     2238.0    | ... |
    | -122.31457273 |     1650.0    | ... |
    | -122.33659507 |     1780.0    | ... |
    |  -122.0308176 |     2390.0    | ... |
    +---------------+---------------+-----+
    [21613 rows x 21 columns]
    Note: Only the head of the SFrame is printed.
    You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.>


.. parsed-literal::

    [INFO] GraphLab Server Version: 1.7.1
    [INFO] Start server at: ipc:///tmp/graphlab_server-30525 - Server binary: /home/charon/.virtualenvs/machinelearning/lib/python2.7/site-packages/graphlab/unity_server - Server log: /tmp/graphlab_server_1449120637.log
    [INFO] [1;32m1449120637 : INFO:     (initialize_globals_from_environment:282): Setting configuration variable GRAPHLAB_FILEIO_ALTERNATIVE_SSL_CERT_FILE to /home/charon/.virtualenvs/machinelearning/local/lib/python2.7/site-packages/certifi/cacert.pem
    [0m[1;32m1449120637 : INFO:     (initialize_globals_from_environment:282): Setting configuration variable GRAPHLAB_FILEIO_ALTERNATIVE_SSL_CERT_DIR to 
    [0mThis non-commercial license of GraphLab Create is assigned to necromuralist@gmail.com and will expire on October 20, 2016. For commercial licensing options, visit https://dato.com/buy/.
    


Split data into training and testing
------------------------------------

We use seed=0 so that everyone running this notebook gets the same
results. In practice, you may set a random seed (or let GraphLab Create
pick a random seed for you).

.. code:: python

    train_data,test_data = sales.random_split(.8,seed=0)

Useful SFrame summary functions
-------------------------------

In order to make use of the closed form solution as well as take
advantage of graphlab's built in functions we will review some important
ones. In particular: \* Computing the sum of an SArray \* Computing the
arithmetic average (mean) of an SArray \* multiplying SArrays by
constants \* multiplying SArrays by other SArrays

Let's compute the mean of the House Prices in King County in 2 different ways.
------------------------------------------------------------------------------

.. code:: python

    prices = sales['price'] # extract the price column of the sales SFrame -- this is now an SArray

*recall that the arithmetic average (the mean) is the sum of the prices
divided by the total number of houses*

method 1
--------

.. code:: python

    sum_prices = prices.sum()
    num_houses = prices.size() # when prices is an SArray .size() returns its length
    avg_price_1 = sum_prices/num_houses

method 2
--------

.. code:: python

    avg_price_2 = prices.mean() # if you just want the average, the .mean() function

.. code:: python

    print( "average price via method 1: " + str(avg_price_1))
    print( "average price via method 2: " + str(avg_price_2))
    delta = 0.0000001
    assert (avg_price_1 - avg_price_2) < delta, "Method 1: {0}, Method 2: {1} Difference: {2}".format(avg_price_1,
                                                                                             avg_price_2,
                                                                                             avg_price_1 - avg_price_2)


.. parsed-literal::

    average price via method 1: 540088.141905
    average price via method 2: 540088.141905


As we see we get the same answer both ways

if we want to multiply every price by 0.5 it's a simple as:

.. code:: python

    half_prices = 0.5*prices

Let's compute the sum of squares of price. We can multiply two SArrays
of the same length elementwise also with \*

.. code:: python

    prices_squared = prices*prices
    sum_prices_squared = prices_squared.sum() # price_squared is an SArray of the squares and we want to add them up.
    print "the sum of price squared is: " + str(sum_prices_squared)


.. parsed-literal::

    the sum of price squared is: 9.21732513355e+15


Aside: The python notation x.xxe+yy means x.xx \* 10^(yy). e.g 100 =
10^2 = 1\*10^2 = 1e2

Build a generic simple linear regression function
-------------------------------------------------

Armed with these SArray functions we can use the closed form solution
found from lecture to compute the slope and intercept for a simple
linear regression on observations stored as SArrays: input\_feature,
output.

Complete the following function (or write your own) to compute the
simple linear regression slope and intercept:

Simple Linear Regression
------------------------

From https://en.wikipedia.org/wiki/Simple\_linear\_regression

slope = (mean of x\ *y - (mean of x * mean of y)/(mean of x\ **2 - (mean
of x)**\ 2) = (mean\_of\_xy - mean\_of\_x \*
mean\_of\_y)/(mean\_of\_x\_squared - square\_of\_mean\_of\_x)

intercept = mean\_of\_y - slope \* mean\_of\_x

.. code:: python

    def simple_linear_regression(input_feature, output):
        # compute the mean of the input_feature and the mean of the output
        mean_of_x = input_feature.mean()
        mean_of_y = output.mean()
        
        # compute the product of the output and the input_feature and its mean
        mean_of_xy = (input_feature * output).mean()
        
        # compute the squared value of the input_feature and its mean
        mean_of_x_squared = (input_feature**2).mean()
        
        # use the formula for the slope
        slope = (mean_of_xy - (mean_of_x * mean_of_y))/(mean_of_x_squared - mean_of_x**2)
        
        # use the formula for the intercept
        intercept = mean_of_y - slope * mean_of_x
        return (intercept, slope)

We can test that our function works by passing it something where we
know the answer. In particular we can generate a feature and then put
the output exactly on a line: output = 1 + 1\*input\_feature then we
know both our slope and intercept should be 1

.. code:: python

    house_model = graphlab.linear_regression.create(sales, target='price',
                                                    features=['sqft_living'],
                                                    validation_set=None,
                                                    verbose=False)
    
    coefficients = house_model.get('coefficients')
    print(coefficients)


.. parsed-literal::

    +-------------+-------+----------------+
    |     name    | index |     value      |
    +-------------+-------+----------------+
    | (intercept) |  None | -43579.0852515 |
    | sqft_living |  None | 280.622770886  |
    +-------------+-------+----------------+
    [2 rows x 3 columns]
    


.. code:: python

    simple_linear_regression(sales['sqft_living'], sales['price'])




.. parsed-literal::

    (-43580.740327082574, 280.62356663364426)



.. code:: python

    test_feature = graphlab.SArray(range(5))
    test_output = graphlab.SArray(1 + 1*test_feature)
    (test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)
    print "Intercept: " + str(test_intercept)
    print "Slope: " + str(test_slope)


.. parsed-literal::

    Intercept: 1.0
    Slope: 1.0


Now that we know it works let's build a regression model for predicting
price based on sqft\_living. Rembember that we train on train\_data!

.. code:: python

    sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])
    
    def sqft_model(input):
        return sqft_intercept + sqft_slope * input
    
    print "Intercept: " + str(sqft_intercept)
    print "Slope: " + str(sqft_slope)


.. parsed-literal::

    Intercept: -47116.0765749
    Slope: 281.958838568


Predicting Values
-----------------

Now that we have the model parameters: intercept & slope we can make
predictions. Using SArrays it's easy to multiply an SArray by a constant
and add a constant value. Complete the following function to return the
predicted output given the input\_feature, slope and intercept:

.. code:: python

    def get_regression_predictions(input_feature, intercept, slope):
        # calculate the predicted values:
        predicted_values = slope * input_feature + intercept
        return predicted_values

Now that we can calculate a prediction given the slop and intercept
let's make a prediction. Use (or alter) the following to find out the
estimated price for a house with 2650 squarefeet according to the
squarefeet model we estiamted above.

**Quiz Question: Using your Slope and Intercept from (4), What is the
predicted price for a house with 2650 sqft?**

.. code:: python

    my_house_sqft = 2650
    estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
    print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)


.. parsed-literal::

    The estimated price for a house with 2650 squarefeet is $700074.85


Residual Sum of Squares
-----------------------

Now that we have a model and can make predictions let's evaluate our
model using Residual Sum of Squares (RSS). Recall that RSS is the sum of
the squares of the residuals and the residuals is just a fancy word for
the difference between the predicted output and the true output.

Complete the following (or write your own) function to compute the RSS
of a simple linear regression model given the input\_feature, output,
intercept and slope.

.. code:: python

    def get_residual_sum_of_squares(input_feature, output, intercept, slope):
        # First get the predictions
        predictions = slope * input_feature + intercept
        # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
        residuals = output - predictions
        # square the residuals and add them up
        RSS = numpy.power(residuals, 2).sum()
        return RSS

Let's test our get\_residual\_sum\_of\_squares function by applying it
to the test model where the data lie exactly on a line. Since they lie
exactly on a line the residual sum of squares should be zero!

.. code:: python

    print get_residual_sum_of_squares(test_feature, test_output, test_intercept, test_slope) # should be 0.0


.. parsed-literal::

    0.0


Now use your function to calculate the RSS on training data from the
squarefeet model calculated above.

**Quiz Question: According to this function and the slope and intercept
from the squarefeet model What is the RSS for the simple linear
regression using squarefeet to predict prices on TRAINING data?**

.. code:: python

    rss_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)
    print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)


.. parsed-literal::

    The RSS of predicting Prices based on Square Feet is : 1.20191835632e+15


Predict the squarefeet given price
----------------------------------

What if we want to predict the squarefoot given the price? Since we have
an equation y = a + b\*x we can solve the function for x. So that if we
have the intercept (a) and the slope (b) and the price (y) we can solve
for the estimated squarefeet (x).

Comlplete the following function to compute the inverse regression
estimate, i.e. predict the input\_feature given the output!

x = (y - a) / b

.. code:: python

    def inverse_regression_predictions(output, intercept, slope):
        # solve output = slope + intercept*input_feature for input_feature. Use this equation to compute the inverse predictions:
        estimated_feature = (output - intercept)/slope
        return estimated_feature

Now that we have a function to compute the squarefeet given the price
from our simple regression model let's see how big we might expect a
house that coses $800,000 to be.

**Quiz Question: According to this function and the regression slope and
intercept from (3) what is the estimated square-feet for a house costing
$800,000?**

.. code:: python

    my_house_price = 800000
    estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
    print "The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet)


.. parsed-literal::

    The estimated squarefeet for a house worth $800000.00 is 3004


.. code:: python

    figure = plt.figure()
    axe = figure.gca()
    lines = axe.plot(train_data['sqft_living'], train_data['price'], '.')
    x = numpy.arange(train_data['sqft_living'].min(), train_data['sqft_living'].max() + 1)
    y = sqft_model(x)
    lines = axe.plot(x, y, '-')
    lines = axe.plot([estimated_squarefeet], [my_house_price], 'ro')



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7fdff8120710>


New Model: estimate prices from bedrooms
----------------------------------------

We have made one model for predicting house prices using squarefeet, but
there are many other features in the sales SFrame. Use your simple
linear regression function to estimate the regression parameters from
predicting Prices based on number of bedrooms. Use the training data!

.. code:: python

    # Estimate the slope and intercept for predicting 'price' based on 'bedrooms'
    bedrooms_intercept, bedrooms_slope = simple_linear_regression(train_data['bedrooms'], train_data['price'])


Test your Linear Regression Algorithm
-------------------------------------

Now we have two models for predicting the price of a house. How do we
know which one is better? Calculate the RSS on the TEST data (remember
this data wasn't involved in learning the model). Compute the RSS from
predicting prices using bedrooms and from predicting prices using
squarefeet.

**Quiz Question: Which model (square feet or bedrooms) has lowest RSS on
TEST data? Think about why this might be the case.**

.. code:: python

    # Compute RSS when using bedrooms on TEST data:
    rss_bedrooms = get_residual_sum_of_squares(test_data['bedrooms'], test_data['price'], bedrooms_intercept, bedrooms_slope)
    print(rss_bedrooms)


.. parsed-literal::

    4.93364582868e+14


.. code:: python

    # Compute RSS when using squarfeet on TEST data:
    rss_squarefeet = get_residual_sum_of_squares(test_data['sqft_living'], test_data['price'], sqft_intercept, sqft_slope)
    print(rss_squarefeet)


.. parsed-literal::

    2.75402936247e+14

