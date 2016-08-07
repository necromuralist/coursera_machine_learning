
Regression Week 3: Assessing Fit (polynomial regression)
========================================================

.. code:: python

    # python standard library
    from abc import ABCMeta, abstractproperty, abstractmethod
    from collections import namedtuple
    
    # third-party
    import graphlab
    import matplotlib.pyplot as plt
    from matplotlib import pylab
    import numpy
    import pandas
    import seaborn
    from sklearn.cross_validation import train_test_split
    import statsmodels.api as statsmodels
    
    # this code
    from machine_learning.coursera.regression.common_utilities.regression_functions import residual_sum_of_squares

.. code:: python

    %matplotlib inline
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)

In this notebook you will compare different regression models in order
to assess which model fits best. We will be using polynomial regression
as a means to examine this topic. In particular you will: \* Write a
function to take an SArray and a degree and return an SFrame where each
column is the SArray to a polynomial value up to the total degree e.g.
degree = 3 then column 1 is the SArray column 2 is the SArray squared
and column 3 is the SArray cubed \* Use matplotlib to visualize
polynomial regressions \* Use matplotlib to visualize the same
polynomial degree on different subsets of the data \* Use a validation
set to select a polynomial degree \* Assess the final fit using test
data

We will continue to use the House data from previous notebooks.

Fire up graphlab create
-----------------------

Next we're going to write a polynomial function that takes an SArray and
a maximal degree and returns an SFrame with columns containing the
SArray to all the powers up to the maximal degree.

The easiest way to apply a power to an SArray is to use the .apply() and
lambda x: functions. For example to take the example array and compute
the third power we can do as follows: (note running this cell the first
time may take longer than expected since it loads graphlab)

.. code:: python

    tmp = graphlab.SArray([1., 2., 3.])
    tmp_cubed = tmp.apply(lambda x: x**3)
    print( tmp)
    print( tmp_cubed)


.. parsed-literal::

    [1.0, 2.0, 3.0]
    [1.0, 8.0, 27.0]


.. code:: python

    temp = pandas.Series([1,2,3], dtype=float)
    temp_cubed = temp.apply(lambda x: x**3)
    print(temp)
    print(temp_cubed)


.. parsed-literal::

    0    1
    1    2
    2    3
    dtype: float64
    0     1
    1     8
    2    27
    dtype: float64


We can create an empty SFrame using graphlab.SFrame() and then add any
columns to it with ex\_sframe['column\_name'] = value. For example we
create an empty SFrame and make the column 'power\_1' to be the first
power of tmp (i.e. tmp itself).

.. code:: python

    ex_sframe = graphlab.SFrame()
    ex_sframe['power_1'] = tmp
    print ex_sframe


.. parsed-literal::

    +---------+
    | power_1 |
    +---------+
    |   1.0   |
    |   2.0   |
    |   3.0   |
    +---------+
    [3 rows x 1 columns]
    


.. code:: python

    ex_frame = pandas.DataFrame()
    ex_frame['power_1'] = temp
    print(ex_frame)


.. parsed-literal::

       power_1
    0        1
    1        2
    2        3


Polynomial\_sframe function
---------------------------

Using the hints above complete the following function to create an
SFrame consisting of the powers of an SArray up to a specific degree:

.. code:: python

    def polynomial_sframe(feature, degree):
        # assume that degree >= 1
        # initialize the SFrame:
        poly_sframe = graphlab.SFrame()
        # and set poly_sframe['power_1'] equal to the passed feature
        poly_sframe['power_1'] = feature
        # first check if degree > 1
        if degree > 1:
            # then loop over the remaining degrees:
            # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
            for power in range(2, degree+1): 
                # first we'll give the column a name:
                name = 'power_' + str(power)
                # then assign poly_sframe[name] to the appropriate power of feature
                poly_sframe[name] = feature.apply(lambda x: x**power)
        return poly_sframe

.. code:: python

    def polynomial_dframe(feature, degree):
        # assume that degree >= 1
        # initialize the DataFrame:
        poly_dframe = pandas.DataFrame()
        # and set poly_dframe['power_1'] equal to the passed feature
        poly_dframe['power_1'] = feature
        # first check if degree > 1
        if degree > 1:
            # then loop over the remaining degrees:
            # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
            for power in range(2, degree+1): 
                # first we'll give the column a name:
                name = 'power_' + str(power)
                # then assign poly_dframe[name] to the appropriate power of feature
                poly_dframe[name] = feature.apply(lambda x: x**power)
        return poly_dframe

To test your function consider the smaller tmp variable and what you
would expect the outcome of the following call:

.. code:: python

    print(polynomial_sframe(tmp, 3))


.. parsed-literal::

    +---------+---------+---------+
    | power_1 | power_2 | power_3 |
    +---------+---------+---------+
    |   1.0   |   1.0   |   1.0   |
    |   2.0   |   4.0   |   8.0   |
    |   3.0   |   9.0   |   27.0  |
    +---------+---------+---------+
    [3 rows x 3 columns]
    


.. code:: python

    print(polynomial_dframe(temp, 3))


.. parsed-literal::

       power_1  power_2  power_3
    0        1        1        1
    1        2        4        8
    2        3        9       27


Visualizing polynomial regression
---------------------------------

Let's use matplotlib to visualize what a polynomial regression looks
like on some real data.

.. code:: python

    sales = graphlab.SFrame('../../large_data/kc_house_data.gl/')
    sales_frame = pandas.read_csv('../../large_data/csvs/kc_house_data.csv')

As in Week 3, we will use the sqft\_living variable. For plotting
purposes (connecting the dots), you'll need to sort by the values of
sqft\_living. For houses with identical square footage, we break the tie
by their prices.

.. code:: python

    sales = sales.sort('sqft_living')
    sales_frame = sales_frame.sort_values(by='sqft_living')

Let's start with a degree 1 polynomial using 'sqft\_living' (i.e. a
line) to predict 'price' and plot what it looks like.

.. code:: python

    poly1_data = polynomial_sframe(sales['sqft_living'], 1)
    poly1_data['price'] = sales['price'] # add price to the data since it's the target

.. code:: python

    poly_pandas_data = polynomial_dframe(sales_frame['sqft_living'], 1)
    poly_pandas_data = statsmodels.add_constant(poly_pandas_data)

NOTE: for all the models in this notebook use validation\_set = None to
ensure that all results are consistent across users.

.. code:: python

    model1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)


.. parsed-literal::

    PROGRESS: Linear regression:
    PROGRESS: --------------------------------------------------------
    PROGRESS: Number of examples          : 21613
    PROGRESS: Number of features          : 1
    PROGRESS: Number of unpacked features : 1
    PROGRESS: Number of coefficients    : 2
    PROGRESS: Starting Newton Method
    PROGRESS: --------------------------------------------------------
    PROGRESS: +-----------+----------+--------------+--------------------+---------------+
    PROGRESS: | Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |
    PROGRESS: +-----------+----------+--------------+--------------------+---------------+
    PROGRESS: | 1         | 2        | 0.010455     | 4362074.696077     | 261440.790724 |
    PROGRESS: +-----------+----------+--------------+--------------------+---------------+
    PROGRESS: SUCCESS: Optimal solution found.
    PROGRESS:


.. code:: python

    model_1_frame = statsmodels.OLS(sales_frame['price'], poly_pandas_data)
    results = model_1_frame.fit()
    print(results.params)


.. parsed-literal::

    const     -43580.743094
    power_1      280.623568
    dtype: float64


.. code:: python

    #let's take a look at the weights before we plot
    coefficients_1 = model1.get("coefficients")
    coefficients_1.column_names()
    coefficients_1




.. parsed-literal::

    Columns:
    	name	str
    	index	str
    	value	float
    
    Rows: 2
    
    Data:
    +-------------+-------+----------------+
    |     name    | index |     value      |
    +-------------+-------+----------------+
    | (intercept) |  None | -43579.0852515 |
    |   power_1   |  None | 280.622770886  |
    +-------------+-------+----------------+
    [2 rows x 3 columns]



.. code:: python

    coefficients_1['value']




.. parsed-literal::

    dtype: float
    Rows: 2
    [-43579.08525145205, 280.6227708858481]



.. code:: python

    class BaseRegressionModel(object):
        """
        Base regression model
        """
        __metaclass__ = ABCMeta
        def __init__(self, data=sales, degree=1, predictor='sqft_living',
                     target='price'):
            """
            :param:
             - `data`: frame with the source data
             - `degree`: degree of the polynomial for the regression
             - `predictor`: name of the predictive variable
             - `target`: name of the variable to predict
             - `version`: Identifier for plot title
            """
            self.data = data
            self.degree = degree
            self.predictor = predictor
            self.target = target
            self._poly_data = None
            self._feature_name = None
            self._model = None
            self._coefficients = None
            self._frame_definition = None
            self._predictions = None
            self._plot_title = None
            self._version = None
            return
    
        @abstractproperty
        def version(self):
            """
            :return: which version this is (SFrame | DataFrame)
            """
            
        @abstractproperty
        def frame_definition(self):
            """
            :return: definition of frame (e.g. graphlab.SFrame)
            """
    
        @abstractproperty
        def coefficients(self):
            """
            :return: Frame with the coefficients for the model
            """
            return self._coefficients
    
        @property
        def feature_name(self):
            """
            :return: name of the column in the polynomial frame that we want
            """
            if self._feature_name is None:
                self._feature_name = 'power_{0}'.format(self.degree)
            return self._feature_name
    
        @property
        def poly_data(self):
            """
            :return: frame of self.data, columns raised to degrees up to self.degree
            """
            if self._poly_data is None:
                feature = self.data[self.predictor]
                self._poly_data = self.frame_definition()
                self._poly_data['power_1'] = feature
                if self.degree > 1:
                    for power in range(2, self.degree + 1): 
                        name = 'power_{0}'.format(power)
                        self._poly_data[name] = feature.apply(lambda x: x**power)
                # the model needs to know the features without the target
                try:        
                    self.features = self._poly_data.column_names()
                    # but to fit, the data also needs the target column added
                    self._poly_data[self.target] = self.data[self.target]
                except AttributeError:
                    # this means it's pandas/statsmodels
                    self.features = self._poly_data.columns
                    self._poly_data = statsmodels.add_constant(self._poly_data)
            return self._poly_data
    
        @abstractproperty
        def model(self):
            """
            :return: linear model
            """
            return self._model
    
        @property
        def predictions(self):
            """
            :return: vector of predictions based on model and poly-data
            """
            if self._predictions is None:
                self._predictions = self.model.predict(self.poly_data)
            return self._predictions
    
        @property
        def plot_title(self):
            if self._plot_title is None:
                self._plot_title = "{p} vs {t} (degree {d} - {v})".format(p=self.predictor,
                                                                          t=self.target,
                                                                          d=self.degree,
                                                                          v=self.version)
            return self._plot_title
            
        def plot_fit(self):
            """
            Plot the data and regression line
            """
            figure = plt.figure()
            axe = figure.gca()
            # always use power-1 or the scale will change so it always looks like
            # a straight line
            x = self.poly_data['power_1']
            lines = axe.plot(x, self.data[self.target],'.', label='data')
            lines = axe.plot(x, self.predictions, '-', label='regression')
            legend = axe.legend()
            title = axe.set_title(self.plot_title)
            label = axe.set_ylabel(self.target)
            label = axe.set_xlabel(self.predictor)
            return
    
        def predict(self, input):
            """
            :param:
             - `input`: vector of input values
            :return: vector of predicted output values based on model
            """
            return self.model.predict(input)
    
        def reset(self):
            """
            :postcondition: calculated properties set to None
            """
            self._model = None
            self._poly_data = None
            self._feature_name = None
            self._coefficients = None
            self._predictions = None
            return
    # end class BaseRegressionModel

.. code:: python

    class RegressionModel(BaseRegressionModel):
        def __init__(self, *args, **kwargs):
            """
            :param:
             - `data`: frame with the source data
             - `degree`: degree of the polynmial for the regression
             - `predictor`: name of the predictive variable
             - `target`: name of the variable to predict
            """
            super(RegressionModel, self).__init__(*args, **kwargs)
            return
    
        @property
        def version(self):
            """
            :return: string 'SFrame'
            """
            return 'SFrame'
        
        @property
        def coefficients(self):
            """
            coefficients['value'] - (intercept, slope)
            :return: SFrame with the coefficients for the model
            """
            if self._coefficients is None:
                self._coefficients = self.model.get('coefficients')
            return self._coefficients
    
        @property
        def frame_definition(self):
            """
            :return: SFrame constructor
            """
            if self._frame_definition is None:
                self._frame_definition = graphlab.SFrame
            return self._frame_definition
    
        @property
        def model(self):
            """
            :return: linear model
            """
            if self._model is None:
                self._model = graphlab.linear_regression.create(self.poly_data,
                                                                target=self.target,
                                                                features=self.features,
                                                                validation_set=None,
                                                                verbose=False)
            return self._model
    # end class RegressionModel

.. code:: python

    class FrameRegressionModel(BaseRegressionModel):
        def __init__(self, *args, **kwargs):
            super(FrameRegressionModel, self).__init__(*args, **kwargs)
            return
    
        @property
        def version(self):
            """
            :return: string 'DataFrame'
            """
            if self._version is None:
                self._version = 'DataFrame'
            return self._version
        
        @property
        def frame_definition(self):
            """
            :return: DataFrame constructor
            """
            return pandas.DataFrame
    
        @property
        def coefficients(self):
            """
            :return: params Series
            """
            return self.model.params
    
        @property
        def model(self):
            """
            :return: OLS statsmodel
            """
            if self._model is None:
                self._model = statsmodels.OLS(self.data[self.target], self.poly_data)
                self._model = self._model.fit()
            return self._model
    # end class FrameRegressionModel

.. code:: python

    model_1 = RegressionModel()
    def check_coefficients(coefficients_0, coefficients_1, coefficient_count=2):
        """
        :param:
         - `coefficients_0`: Sframe of model coefficients
         - `coefficients_1': Sframe of coefficients to compare
         - `coefficient_count`: number of coefficients (including intercept)
        """
        c0 = coefficients_0['value']
        c1 = coefficients_1['value']
        
        for i in range(coefficient_count):
            assert c0[i] == c1[i],\
                "Index: {0} First: {1} Second: {2}".format(i, c0[i], c1[i])
        return
    
    check_coefficients(model_1.coefficients, coefficients_1)


.. code:: python

    model_frame_1 = FrameRegressionModel(data=sales_frame)

.. code:: python

    model_1.plot_fit()



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6e919f10>


.. code:: python

    model_frame_1._plot_title = "Sq Ft Living vs Price (degree 1) Pandas Version"
    model_frame_1.plot_fit()



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6e668bd0>


Let's unpack that plt.plot() command. The first pair of SArrays we
passed are the 1st power of sqft and the actual price we then ask it to
print these as dots '.'. The next pair we pass is the 1st power of sqft
and the predicted values from the linear model. We ask these to be
plotted as a line '-'.

We can see, not surprisingly, that the predicted values all fall on a
line, specifically the one with slope 280 and intercept -43579. What if
we wanted to plot a second degree polynomial?

.. code:: python

    model_2 = RegressionModel(degree=2)
    model_frame_2 = FrameRegressionModel(data=sales_frame, degree=2)
    poly2_data = polynomial_sframe(sales['sqft_living'], 2)
    my_features = poly2_data.column_names() # get the name of the features
    poly2_data['price'] = sales['price'] # add price to the data since it's the target
    model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)


.. parsed-literal::

    PROGRESS: Linear regression:
    PROGRESS: --------------------------------------------------------
    PROGRESS: Number of examples          : 21613
    PROGRESS: Number of features          : 2
    PROGRESS: Number of unpacked features : 2
    PROGRESS: Number of coefficients    : 3
    PROGRESS: Starting Newton Method
    PROGRESS: --------------------------------------------------------
    PROGRESS: +-----------+----------+--------------+--------------------+---------------+
    PROGRESS: | Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |
    PROGRESS: +-----------+----------+--------------+--------------------+---------------+
    PROGRESS: | 1         | 2        | 0.012246     | 5913020.984255     | 250948.368758 |
    PROGRESS: +-----------+----------+--------------+--------------------+---------------+
    PROGRESS: SUCCESS: Optimal solution found.
    PROGRESS:


.. code:: python

    coefficients_2 = model2.get("coefficients")
    check_coefficients(model_2.coefficients, coefficients_2, 3)
    print(coefficients_2)


.. parsed-literal::

    +-------------+-------+-----------------+
    |     name    | index |      value      |
    +-------------+-------+-----------------+
    | (intercept) |  None |  199222.496445  |
    |   power_1   |  None |  67.9940640677  |
    |   power_2   |  None | 0.0385812312789 |
    +-------------+-------+-----------------+
    [3 rows x 3 columns]
    


.. code:: python

    model_2.plot_fit()



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6f195e90>


.. code:: python

    model_frame_2._plot_title = 'Sqft Living vs Price (degree 2, Pandas Version)'
    model_frame_2.plot_fit()



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6cb83790>


The resulting model looks like half a parabola. Try on your own to see
what the cubic looks like:

.. code:: python

    model_3 = RegressionModel(degree=3)
    model_frame_3 = FrameRegressionModel(data=sales_frame, degree=3)
    model_3.coefficients




.. parsed-literal::

    Columns:
    	name	str
    	index	str
    	value	float
    
    Rows: 4
    
    Data:
    +-------------+-------+-------------------+
    |     name    | index |       value       |
    +-------------+-------+-------------------+
    | (intercept) |  None |   336788.117952   |
    |   power_1   |  None |   -90.1476236119  |
    |   power_2   |  None |   0.087036715081  |
    |   power_3   |  None | -3.8398521196e-06 |
    +-------------+-------+-------------------+
    [4 rows x 3 columns]



.. code:: python

    model_3.plot_fit()



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6f151b10>


.. code:: python

    model_frame_3._plot_title = "Sqft Living Space vs Price (degree 3, Pandas)"
    model_frame_3.plot_fit()



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6cafde90>


Now try a 15th degree polynomial:

.. code:: python

    model_15 = RegressionModel(degree=15)
    model_frame_15 = FrameRegressionModel(data=sales_frame, degree=15,
                                          version='DataFrame')
    model_15.coefficients




.. parsed-literal::

    Columns:
    	name	str
    	index	str
    	value	float
    
    Rows: 16
    
    Data:
    +-------------+-------+--------------------+
    |     name    | index |       value        |
    +-------------+-------+--------------------+
    | (intercept) |  None |   73619.7521129    |
    |   power_1   |  None |   410.287462534    |
    |   power_2   |  None |  -0.230450714428   |
    |   power_3   |  None |  7.5884054245e-05  |
    |   power_4   |  None | -5.65701802657e-09 |
    |   power_5   |  None | -4.57028130583e-13 |
    |   power_6   |  None | 2.66360206431e-17  |
    |   power_7   |  None | 3.38584769292e-21  |
    |   power_8   |  None | 1.14723104086e-25  |
    |   power_9   |  None | -4.65293586102e-30 |
    +-------------+-------+--------------------+
    [16 rows x 3 columns]
    Note: Only the head of the SFrame is printed.
    You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.



.. code:: python

    model_15.plot_fit()



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6c5eed10>


.. code:: python

    model_frame_15.plot_fit()



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6c4cb0d0>


What do you think of the 15th degree polynomial? Do you think this is
appropriate? If we were to change the data do you think you'd get pretty
much the same curve? Let's take a look.

Changing the data and re-learning
---------------------------------

We're going to split the sales data into four subsets of roughly equal
size. Then you will estimate a 15th degree polynomial model on all four
subsets of the data. Print the coefficients (you should use
.print\_rows(num\_rows = 16) to view all of them) and plot the resulting
fit (as we did above). The quiz will ask you some questions about these
results.

To split the sales data into four subsets, we perform the following
steps: \* First split sales into 2 subsets with
``.random_split(0.5, seed=0)``. \* Next split the resulting subsets into
2 more subsets each. Use ``.random_split(0.5, seed=0)``.

We set ``seed=0`` in these steps so that different users get consistent
results. You should end up with 4 subsets (``set_1``, ``set_2``,
``set_3``, ``set_4``) of approximately equal size.

.. code:: python

    train, test = sales.random_split(0.5, seed=0)
    set_1, set_2 = train.random_split(0.5, seed=0)
    set_3, set_4 = test.random_split(0.5, seed=0)


Fit a 15th degree polynomial on set\_1, set\_2, set\_3, and set\_4 using
sqft\_living to predict prices. Print the coefficients and make a plot
of the resulting model.

.. code:: python

    def print_plot(model):
        model.coefficients.print_rows(num_rows=16)
        model.plot_fit()
        return

Set 1
~~~~~

.. code:: python

    model_1_15 = RegressionModel(data=set_1, degree=15)
    print_plot(model_1_15)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6f195e50>


.. parsed-literal::

    +-------------+-------+--------------------+
    |     name    | index |       value        |
    +-------------+-------+--------------------+
    | (intercept) |  None |   197099.450457    |
    |   power_1   |  None |   166.862882905    |
    |   power_2   |  None |  -0.0697578810024  |
    |   power_3   |  None | 3.63129717598e-05  |
    |   power_4   |  None | -3.74901359183e-09 |
    |   power_5   |  None | -8.76717441662e-14 |
    |   power_6   |  None | 1.41557620304e-17  |
    |   power_7   |  None | 1.12198026361e-21  |
    |   power_8   |  None | 2.77358582356e-26  |
    |   power_9   |  None | -1.94539657313e-30 |
    |   power_10  |  None | -2.88642493802e-34 |
    |   power_11  |  None | -2.08839000817e-38 |
    |   power_12  |  None | -9.81836993907e-43 |
    |   power_13  |  None | -1.48610009052e-47 |
    |   power_14  |  None | 3.16615964613e-51  |
    |   power_15  |  None | 5.00656478257e-55  |
    +-------------+-------+--------------------+
    [16 rows x 3 columns]
    


.. code:: python

    # the splitting doesn't maintain the sort ordering so the plot will be messed up if not sorted
    model_frame_set_1 = FrameRegressionModel(data=frame_1.sort_values(by='sqft_living'), degree=15)
    model_frame_set_1.plot_fit()



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6c9150d0>


Set 2
~~~~~

.. code:: python

    model_2_15 = RegressionModel(data=set_2, degree=15)
    print_plot(model_2_15)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6f0287d0>


.. parsed-literal::

    +-------------+-------+--------------------+
    |     name    | index |       value        |
    +-------------+-------+--------------------+
    | (intercept) |  None |   160515.194668    |
    |   power_1   |  None |   161.068906214    |
    |   power_2   |  None |  0.0072128855415   |
    |   power_3   |  None | -1.53767451326e-05 |
    |   power_4   |  None |  5.531012769e-09   |
    |   power_5   |  None | 3.44914141886e-13  |
    |   power_6   |  None | -8.4134933128e-17  |
    |   power_7   |  None | -1.1755754411e-20  |
    |   power_8   |  None | -3.24855695774e-25 |
    |   power_9   |  None | 8.06950508756e-29  |
    |   power_10  |  None | 1.36060382518e-32  |
    |   power_11  |  None | 1.06720789577e-36  |
    |   power_12  |  None | 1.92370157322e-41  |
    |   power_13  |  None | -7.10368037447e-45 |
    |   power_14  |  None | -1.01868434938e-48 |
    |   power_15  |  None | -1.74692416675e-53 |
    +-------------+-------+--------------------+
    [16 rows x 3 columns]
    


set 3
~~~~~

.. code:: python

    model_3_15 = RegressionModel(data=set_3, degree=15)
    print_plot(model_3_15)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6ed3b4d0>


.. parsed-literal::

    +-------------+-------+--------------------+
    |     name    | index |       value        |
    +-------------+-------+--------------------+
    | (intercept) |  None |   64031.5743611    |
    |   power_1   |  None |   419.963446533    |
    |   power_2   |  None |  -0.217032383683   |
    |   power_3   |  None | 5.71721871042e-05  |
    |   power_4   |  None | 6.42456678907e-10  |
    |   power_5   |  None | -8.76764336026e-13 |
    |   power_6   |  None |  -4.39079425e-17   |
    |   power_7   |  None | 4.65780734822e-21  |
    |   power_8   |  None | 8.56537636021e-25  |
    |   power_9   |  None | 5.87177076692e-29  |
    |   power_10  |  None | -1.18822533167e-34 |
    |   power_11  |  None | -5.48520101257e-37 |
    |   power_12  |  None | -7.6344609578e-41  |
    |   power_13  |  None | -5.62621010165e-45 |
    |   power_14  |  None | 2.10087982472e-50  |
    |   power_15  |  None |  8.8038750037e-53  |
    +-------------+-------+--------------------+
    [16 rows x 3 columns]
    


Set 4
~~~~~

.. code:: python

    model_4_15 = RegressionModel(data=set_4, degree=15)
    print_plot(model_4_15)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6ed3bc50>


.. parsed-literal::

    +-------------+-------+--------------------+
    |     name    | index |       value        |
    +-------------+-------+--------------------+
    | (intercept) |  None |   238215.539488    |
    |   power_1   |  None |   35.6890462037    |
    |   power_2   |  None |  0.0384180337865   |
    |   power_3   |  None | 1.00407290044e-05  |
    |   power_4   |  None | -5.35136998645e-09 |
    |   power_5   |  None | 3.35662136019e-13  |
    |   power_6   |  None | 1.81755721549e-16  |
    |   power_7   |  None | 6.62015234254e-21  |
    |   power_8   |  None | -3.13250645182e-24 |
    |   power_9   |  None |  -6.114958954e-28  |
    |   power_10  |  None | -4.37319305529e-32 |
    |   power_11  |  None | 3.54666004611e-36  |
    |   power_12  |  None | 1.47563558462e-39  |
    |   power_13  |  None | 2.10094882446e-43  |
    |   power_14  |  None | 8.12201994906e-48  |
    |   power_15  |  None | -4.45547001478e-51 |
    +-------------+-------+--------------------+
    [16 rows x 3 columns]
    


.. code:: python

    train_frame, test_frame = train_test_split(sales_frame, train_size=.5, random_state=0)
    frame_1, frame_2 = train_test_split(train_frame, train_size=.5, random_state=0)
    frame_3, frame_4 = train_test_split(test_frame, train_size=.5, random_state=0)
    frame_list = [frame_1, frame_2, frame_3, frame_4]
    frames = {'frame_{0}'.format(index):frame_list[index] for index in range(len(frame_list))}

.. code:: python

    for name in sorted(frames):
        model = FrameRegressionModel(data=frames[name].sort_values(by='sqft_living'), degree=15)
        model._version = "DataFrame, {0}".format(name)
        model.plot_fit()



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6ece0310>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6c42b2d0>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6c5f90d0>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6c04e510>


Some questions you will be asked on your quiz:

**Quiz Question: Is the sign (positive or negative) for power\_15 the
same in all four models?**

**Quiz Question: (True/False) the plotted fitted lines look the same in
all four plots**

Selecting a Polynomial Degree
-----------------------------

Whenever we have a "magic" parameter like the degree of the polynomial
there is one well-known way to select these parameters: validation set.
(We will explore another approach in week 4).

We split the sales dataset 3-way into training set, test set, and
validation set as follows:

-  Split our sales data into 2 sets: ``training_and_validation`` and
   ``testing``. Use ``random_split(0.9, seed=1)``.
-  Further split our training data into two sets: ``training`` and
   ``validation``. Use ``random_split(0.5, seed=1)``.

Again, we set ``seed=1`` to obtain consistent results for different
users.

.. code:: python

    training_and_validation, testing = sales.random_split(0.9, seed=1)
    training, validation = training_and_validation.random_split(0.5, seed=1)

Next you should write a loop that does the following: \* For degree in
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] (to get this in
python type range(1, 15+1)) \* Build an SFrame of polynomial data of
train\_data['sqft\_living'] at the current degree \* hint: my\_features
= poly\_data.column\_names() gives you a list e.g. ['power\_1',
'power\_2', 'power\_3'] which you might find useful for
graphlab.linear\_regression.create( features = my\_features) \* Add
train\_data['price'] to the polynomial SFrame \* Learn a polynomial
regression model to sqft vs price with that degree on TRAIN data \*
Compute the RSS on VALIDATION data (here you will want to use
.predict()) for that degree and you will need to make a polynmial SFrame
using validation data. \* Report which degree had the lowest RSS on
validation data (remember python indexes from 0)

(Note you can turn off the print out of linear\_regression.create() with
verbose = False)

.. code:: python

    %pdef residual_sum_of_squares


.. parsed-literal::

     [0mresidual_sum_of_squares[0m[1;33m([0m[0mmodel[0m[1;33m,[0m [0mdata[0m[1;33m,[0m [0mtarget_data[0m[1;33m,[0m [0mverbose[0m[1;33m=[0m[0mFalse[0m[1;33m)[0m[1;33m[0m[0m
     

.. code:: python

    len(training['price']) - len(validation['price'])




.. parsed-literal::

    126



.. code:: python

    print(len(validation['price']))


.. parsed-literal::

    9635


.. code:: python

    model = RegressionModel(data=training, degree=5)
    predictions = model.model.predict(validation)
    residuals = predictions - validation['price']
    (residuals**2).sum()





.. parsed-literal::

    1364735037670161.8



.. code:: python

    residual_sum_of_squares(model.model, validation, validation['price'])




.. parsed-literal::

    1364735037670161.8



.. code:: python

    FrameRss = namedtuple('FrameRss', 'rss train_model test_model predictions'.split())

.. code:: python

    def frame_rss(training, testing, degree, model=RegressionModel):
        """
        :param:
         - `training`: SFrame data for training
         - `testing`: SFrame for testing
         - `degree`: Maximum degree for the polynomial data
         - `model`: class definition RegressionModel or FrameRegressionModel
        :return: RSS between prediction from training model and testing data
        """
        train_model = model(data=training, degree=degree)
        test_model = model(data=testing, degree=degree)
        predictions = train_model.predict(test_model.poly_data)
        residuals = predictions - test_model.data['price']
        return FrameRss(rss=(residuals**2).sum(), train_model=train_model,
                        test_model=test_model, predictions=predictions)

.. code:: python

    rss_s = []
    models = []
    for degree in range(1, 16):
        rss = frame_rss(training, validation, degree)
        rss_s.append(rss.rss)
        models.append(train_model)

.. code:: python

    rss = numpy.array(rss_s)
    rss.min()
    rss.max()
    
    rss.max() - rss.min()
    min_rss_index = rss.argmin()
    min_rss_degree = min_rss_index + 1
    print("Min RSS Index: {0}".format(min_rss_index))
    assert rss[rss.argmin()] == rss.min()
    print("RSS Max: {0}".format(rss.max()))
    print("RSS Min: {0}".format(rss.min()))
    print("RSS Difference: {0}".format(rss.max() - rss.min()))
    print("Min RSS Degree: {0}".format(min_rss_degree))


.. parsed-literal::

    Min RSS Index: 5
    RSS Max: 6.91195074764e+14
    RSS Min: 6.03331784575e+14
    RSS Difference: 8.78632901887e+13
    Min RSS Degree: 6


**Quiz Question: Which degree (1, 2, …, 15) had the lowest RSS on
Validation data?**

.. code:: python

    print("Degree {0}".format(min_rss_degree))


.. parsed-literal::

    Degree 6


Now that you have chosen the degree of your polynomial using validation
data, compute the RSS of this model on TEST data. Report the RSS on your
quiz.

.. code:: python

    rss = frame_rss(training, testing, min_rss_degree)
    print("{0:.5e}".format(rss.rss))



.. parsed-literal::

    1.28190e+14


.. code:: python

    def plot_train_test(rss):
        figure = plt.figure()
        axe = figure.gca()
        lines = axe.plot(rss.train_model.poly_data['power_1'],
                         rss.train_model.data['price'], '.', label='training Data')
        lines = axe.plot(rss.test_model.poly_data['power_1'],
                         rss.test_model.data['price'], 'o', label='Test Data')
        lines = axe.plot(rss.test_model.poly_data['power_1'], rss.predictions, '-',
                         label='Test Predictions')
        axe.legend()
        axe.set_xlabel('Living Space (Sq Ft)')
        axe.set_ylabel('Price ($)')
        title = axe.set_title('Living Space vs Price')
        return
    plot_train_test(rss)




.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6c127690>


.. code:: python

    print(training['sqft_living'].max())
    print(testing['sqft_living'].max())
    print(testing['sqft_living'].max()/(training['sqft_living'].max()))


.. parsed-literal::

    13540.0
    7120.0
    0.525849335303


Despite the high degree, the prediction line looks like a flatter
parabola. It's notable, perhaps, that the training data has an extreme
outlier with a large living space but only a moderately high price,
while the largest living area for the testing set is almost half of the
training set, but the fit still looks reasonably good.

**Quiz Question: what is the RSS on TEST data for the model with the
degree selected from Validation data? (Make sure you got the correct
degree from the previous question)**

1.28190e+14

.. code:: python

    train_validate, test = train_test_split(sales_frame, train_size=0.9, random_state=1)
    train, validate = train_test_split(train_validate, train_size=0.5, random_state=1)

.. code:: python

    frame_rss_s = []
    
    for degree in range(1, 16):
        rss = frame_rss(training, validation, degree,
                        FrameRegressionModel)
        frame_rss_s.append(rss.rss)

.. code:: python

    frame_rss_s = numpy.array(frame_rss_s)
    min_rss = frame_rss_s.min()
    min_rss_index = frame_rss_s.argmin()
    min_rss_degree = min_rss_index + 1
    print("Min RSS: {0}".format(min_rss))
    print("Min RSS Index: {0}".format(min_rss_index))
    print("Min RSS Degree: {0}".format(min_rss_degree))


.. parsed-literal::

    Min RSS: 6.22931073145e+14
    Min RSS Index: 1
    Min RSS Degree: 2


.. code:: python

    rss = frame_rss(training, testing, min_rss_degree)
    print("{0:.5e}".format(rss.rss))


.. parsed-literal::

    1.28092e+14


.. code:: python

    plot_train_test(rss)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ffa6c7b1810>


I don't know why the the degrees are so different, but the RSS for the
testing data is the same for the pandas version and the SFRame version
