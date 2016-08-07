
Linear Regression With SFrames
==============================

This is a summary (by example) of how to perform a linear regression.

Imports
-------

.. code:: python

    import graphlab
    import matplotlib.pyplot as plt
    import seaborn as sns


.. parsed-literal::

    [INFO] GraphLab Server Version: 1.7.1
    [INFO] Start server at: ipc:///tmp/graphlab_server-20429 - Server binary: /home/charon/.virtualenvs/machinelearning/lib/python2.7/site-packages/graphlab/unity_server - Server log: /tmp/graphlab_server_1449108093.log
    [INFO] [1;32m1449108093 : INFO:     (initialize_globals_from_environment:282): Setting configuration variable GRAPHLAB_FILEIO_ALTERNATIVE_SSL_CERT_FILE to /home/charon/.virtualenvs/machinelearning/local/lib/python2.7/site-packages/certifi/cacert.pem
    [0m[1;32m1449108093 : INFO:     (initialize_globals_from_environment:282): Setting configuration variable GRAPHLAB_FILEIO_ALTERNATIVE_SSL_CERT_DIR to 
    [0mThis non-commercial license of GraphLab Create is assigned to necromuralist@gmail.com and will expire on October 20, 2016. For commercial licensing options, visit https://dato.com/buy/.
    


.. code:: python

    %matplotlib inline

Load the data
-------------

.. code:: python

    sales = graphlab.SFrame.read_csv('data/Philadelphia_Crime_Rate_noNA.csv')
    sales.head()




.. parsed-literal::

    Columns:
    	HousePrice	int
    	HsPrc ($10,000)	float
    	CrimeRate	float
    	MilesPhila	float
    	PopChg	float
    	Name	str
    	County	str
    
    Rows: 10
    
    Data:
    +------------+-----------------+-----------+------------+--------+------------+
    | HousePrice | HsPrc ($10,000) | CrimeRate | MilesPhila | PopChg |    Name    |
    +------------+-----------------+-----------+------------+--------+------------+
    |   140463   |     14.0463     |    29.7   |    10.0    |  -1.0  |  Abington  |
    |   113033   |     11.3033     |    24.1   |    18.0    |  4.0   |   Ambler   |
    |   124186   |     12.4186     |    19.5   |    25.0    |  8.0   |   Aston    |
    |   110490   |      11.049     |    49.4   |    25.0    |  2.7   |  Bensalem  |
    |   79124    |      7.9124     |    54.1   |    19.0    |  3.9   | Bristol B. |
    |   92634    |      9.2634     |    48.6   |    20.0    |  0.6   | Bristol T. |
    |   89246    |      8.9246     |    30.8   |    15.0    |  -2.6  | Brookhaven |
    |   195145   |     19.5145     |    10.8   |    20.0    |  -3.5  | Bryn Athyn |
    |   297342   |     29.7342     |    20.2   |    14.0    |  0.6   | Bryn Mawr  |
    |   264298   |     26.4298     |    20.4   |    26.0    |  6.0   | Buckingham |
    +------------+-----------------+-----------+------------+--------+------------+
    +----------+
    |  County  |
    +----------+
    | Montgome |
    | Montgome |
    | Delaware |
    |  Bucks   |
    |  Bucks   |
    |  Bucks   |
    | Delaware |
    | Montgome |
    | Montgome |
    |  Bucks   |
    +----------+
    [10 rows x 7 columns]



.. parsed-literal::

    PROGRESS: Finished parsing file /home/charon/repositories/code/explorations/machine_learning_experiments/machine_learning/coursera/regression/week_1/data/Philadelphia_Crime_Rate_noNA.csv
    PROGRESS: Parsing completed. Parsed 99 lines in 0.058149 secs.
    ------------------------------------------------------
    Inferred types from first line of file as 
    column_type_hints=[int,float,float,float,float,str,str]
    If parsing fails due to incorrect types, you can correct
    the inferred type list above and pass it to read_csv in
    the column_type_hints argument
    ------------------------------------------------------
    PROGRESS: Finished parsing file /home/charon/repositories/code/explorations/machine_learning_experiments/machine_learning/coursera/regression/week_1/data/Philadelphia_Crime_Rate_noNA.csv
    PROGRESS: Parsing completed. Parsed 99 lines in 0.028802 secs.


Fit the regression model
------------------------

The target here it the sale-price of a house ('HousePrice') and the
prediction variable is the crime-rate in the house's area ('CrimeRate')

.. code:: python

    print(graphlab.linear_regression.create.__doc__)


.. parsed-literal::

    
        Create a :class:`~graphlab.linear_regression.LinearRegression` to
        predict a scalar target variable as a linear function of one or more
        features. In addition to standard numeric and categorical types, features
        can also be extracted automatically from list- or dictionary-type SFrame
        columns.
    
        The linear regression module can be used for ridge regression, Lasso, and
        elastic net regression (see References for more detail on these methods). By
        default, this model has an l2 regularization weight of 0.01.
    
        Parameters
        ----------
        dataset : SFrame
            The dataset to use for training the model.
    
        target : string
            Name of the column containing the target variable.
    
        features : list[string], optional
            Names of the columns containing features. 'None' (the default) indicates
            that all columns except the target variable should be used as features.
    
            The features are columns in the input SFrame that can be of the
            following types:
    
            - *Numeric*: values of numeric type integer or float.
    
            - *Categorical*: values of type string.
    
            - *Array*: list of numeric (integer or float) values. Each list element
              is treated as a separate feature in the model.
    
            - *Dictionary*: key-value pairs with numeric (integer or float) values
              Each key of a dictionary is treated as a separate feature and the
              value in the dictionary corresponds to the value of the feature.
              Dictionaries are ideal for representing sparse data.
    
            Columns of type *list* are not supported. Convert such feature
            columns to type array if all entries in the list are of numeric
            types. If the lists contain data of mixed types, separate
            them out into different columns.
    
        l2_penalty : float, optional
            Weight on the l2-regularizer of the model. The larger this weight, the
            more the model coefficients shrink toward 0. This introduces bias into
            the model but decreases variance, potentially leading to better
            predictions. The default value is 0.01; setting this parameter to 0
            corresponds to unregularized linear regression. See the ridge
            regression reference for more detail.
    
        l1_penalty : float, optional
            Weight on l1 regularization of the model. Like the l2 penalty, the
            higher the l1 penalty, the more the estimated coefficients shrink toward
            0. The l1 penalty, however, completely zeros out sufficiently small
            coefficients, automatically indicating features that are not useful for
            the model. The default weight of 0 prevents any features from being
            discarded. See the LASSO regression reference for more detail.
    
        solver : string, optional
            Solver to use for training the model. See the references for more detail
            on each solver.
    
            - *auto (default)*: automatically chooses the best solver for the data
              and model parameters.
            - *newton*: Newton-Raphson
            - *lbfgs*: limited memory BFGS
            - *gd*: gradient descent
            - *fista*: accelerated gradient descent
    
            The model is trained using a carefully engineered collection of methods
            that are automatically picked based on the input data. The ``newton``
            method  works best for datasets with plenty of examples and few features
            (long datasets). Limited memory BFGS (``lbfgs``) is a robust solver for
            wide datasets (i.e datasets with many coefficients).  ``fista`` is the
            default solver for l1-regularized linear regression. Gradient-descent
            (GD) is another well tuned method that can work really well on
            l1-regularized problems.  The solvers are all automatically tuned and
            the default options should function well. See the solver options guide
            for setting additional parameters for each of the solvers.
    
        feature_rescaling : boolean, optional
            Feature rescaling is an important pre-processing step that ensures that
            all features are on the same scale. An l2-norm rescaling is performed
            to make sure that all features are of the same norm. Categorical
            features are also rescaled by rescaling the dummy variables that are
            used to represent them. The coefficients are returned in original scale
            of the problem. This process is particularly useful when features
            vary widely in their ranges.
    
        validation_set : SFrame, optional
    
            A dataset for monitoring the model's generalization performance.
            For each row of the progress table, the chosen metrics are computed
            for both the provided training dataset and the validation_set. The
            format of this SFrame must be the same as the training set.
            By default this argument is set to 'auto' and a validation set is
            automatically sampled and used for progress printing. If
            validation_set is set to None, then no additional metrics
            are computed. The default value is 'auto'.
    
        convergence_threshold : float, optional
    
          Convergence is tested using variation in the training objective. The
          variation in the training objective is calculated using the difference
          between the objective values between two steps. Consider reducing this
          below the default value (0.01) for a more accurately trained model.
          Beware of overfitting (i.e a model that works well only on the training
          data) if this parameter is set to a very low value.
    
        lbfgs_memory_level : int, optional
    
          The L-BFGS algorithm keeps track of gradient information from the
          previous ``lbfgs_memory_level`` iterations. The storage requirement for
          each of these gradients is the ``num_coefficients`` in the problem.
          Increasing the ``lbfgs_memory_level`` can help improve the quality of
          the model trained. Setting this to more than ``max_iterations`` has the
          same effect as setting it to ``max_iterations``.
    
        max_iterations : int, optional
    
          The maximum number of allowed passes through the data. More passes over
          the data can result in a more accurately trained model. Consider
          increasing this (the default value is 10) if the training accuracy is
          low and the *Grad-Norm* in the display is large.
    
        step_size : float, optional (fista only)
    
          The starting step size to use for the ``fista`` and ``gd`` solvers. The
          default is set to 1.0, this is an aggressive setting. If the first
          iteration takes a considerable amount of time, reducing this parameter
          may speed up model training.
    
        verbose : bool, optional
            If True, print progress updates.
    
        Returns
        -------
        out : LinearRegression
            A trained model of type
            :class:`~graphlab.linear_regression.LinearRegression`.
    
        See Also
        --------
        LinearRegression, graphlab.boosted_trees_regression.BoostedTreesRegression, graphlab.regression.create
    
        Notes
        -----
        - Categorical variables are encoded by creating dummy variables. For a
          variable with :math:`K` categories, the encoding creates :math:`K-1` dummy
          variables, while the first category encountered in the data is used as the
          baseline.
    
        - For prediction and evaluation of linear regression models with sparse
          dictionary inputs, new keys/columns that were not seen during training
          are silently ignored.
    
        - Any 'None' values in the data will result in an error being thrown.
    
        - A constant term is automatically added for the model intercept. This term
          is not regularized.
    
    
        References
        ----------
        - Hoerl, A.E. and Kennard, R.W. (1970) `Ridge regression: Biased Estimation
          for Nonorthogonal Problems
          <http://amstat.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634>`_.
          Technometrics 12(1) pp.55-67
    
        - Tibshirani, R. (1996) `Regression Shrinkage and Selection via the Lasso <h
          ttp://www.jstor.org/discover/10.2307/2346178?uid=3739256&uid=2&uid=4&sid=2
          1104169934983>`_. Journal of the Royal Statistical Society. Series B
          (Methodological) 58(1) pp.267-288.
    
        - Zhu, C., et al. (1997) `Algorithm 778: L-BFGS-B: Fortran subroutines for
          large-scale bound-constrained optimization
          <http://dl.acm.org/citation.cfm?id=279236>`_. ACM Transactions on
          Mathematical Software 23(4) pp.550-560.
    
        - Barzilai, J. and Borwein, J. `Two-Point Step Size Gradient Methods
          <http://imajna.oxfordjournals.org/content/8/1/141.short>`_. IMA Journal of
          Numerical Analysis 8(1) pp.141-148.
    
        - Beck, A. and Teboulle, M. (2009) `A Fast Iterative Shrinkage-Thresholding
          Algorithm for Linear Inverse Problems
          <http://epubs.siam.org/doi/abs/10.1137/080716542>`_. SIAM Journal on
          Imaging Sciences 2(1) pp.183-202.
    
        - Zhang, T. (2004) `Solving large scale linear prediction problems using
          stochastic gradient descent algorithms
          <http://dl.acm.org/citation.cfm?id=1015332>`_. ICML '04: Proceedings of
          the twenty-first international conference on Machine learning p.116.
    
    
        Examples
        --------
    
        Given an :class:`~graphlab.SFrame` ``sf`` with a list of columns
        [``feature_1`` ... ``feature_K``] denoting features and a target column
        ``target``, we can create a
        :class:`~graphlab.linear_regression.LinearRegression` as follows:
    
        >>> data =  graphlab.SFrame('http://s3.amazonaws.com/dato-datasets/regression/houses.csv')
    
        >>> model = graphlab.linear_regression.create(data, target='price',
        ...                                  features=['bath', 'bedroom', 'size'])
    
    
        For ridge regression, we can set the ``l2_penalty`` parameter higher (the
        default is 0.01). For Lasso regression, we set the l1_penalty higher, and
        for elastic net, we set both to be higher.
    
        .. sourcecode:: python
    
          # Ridge regression
          >>> model_ridge = graphlab.linear_regression.create(data, 'price', l2_penalty=0.1)
    
          # Lasso
          >>> model_lasso = graphlab.linear_regression.create(data, 'price', l2_penalty=0.,
                                                                       l1_penalty=1.0)
    
          # Elastic net regression
          >>> model_enet  = graphlab.linear_regression.create(data, 'price', l2_penalty=0.5,
                                                                     l1_penalty=0.5)
    
        


.. code:: python

    crime_model = graphlab.linear_regression.create(sales, target='HousePrice',
                                                    features=['CrimeRate'],
                                                    validation_set=None,
                                                    verbose=False)

Plot the line
-------------

.. code:: python

    def plot_data(data, model, title):
        figure = plt.figure()
        axe = figure.gca()
        lines = axe.plot(data['CrimeRate'],data['HousePrice'],'.', label='Data')
        lines = axe.plot(data['CrimeRate'], model.predict(data),'-', label='Fit')
        label = axe.set_xlabel("Crime Rate")
        label = axe.set_ylabel("House Price")
        title = axe.set_title(title)
        legend = axe.legend()


.. code:: python

    plot_data(sales, crime_model, 'Philadelpdhia Crime Rate vs House Price')



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f382198d2d0>


Identify the outlier
--------------------

.. code:: python

    maximum_crime = sales['CrimeRate'].argmax()
    outlier = sales[maximum_crime]
    print(outlier)


.. parsed-literal::

    {'Name': 'Phila,CC', 'PopChg': 4.8, 'County': 'Phila', 'HousePrice': 96200, 'MilesPhila': 0.0, 'HsPrc ($10,000)': 9.62, 'CrimeRate': 366.1}


Get the model coefficients
--------------------------

.. code:: python

    coefficients = crime_model.get('coefficients')
    
    print(coefficients)


.. parsed-literal::

    +-------------+-------+----------------+
    |     name    | index |     value      |
    +-------------+-------+----------------+
    | (intercept) |  None | 176626.046881  |
    |  CrimeRate  |  None | -576.804949058 |
    +-------------+-------+----------------+
    [2 rows x 3 columns]
    


.. code:: python

    intercept, slope = coefficients['value']
    print("y = {m:.2f} x + {b:.2f}".format(m=slope, b=intercept))


.. parsed-literal::

    y = -576.80 x + 176626.05


Predict House Price based on new crime rate
-------------------------------------------

.. code:: python

    print(crime_model.predict.__doc__)


.. parsed-literal::

    
            Return target value predictions for ``dataset``, using the trained
            linear regression model. This method can be used to get fitted values
            for the model by inputting the training dataset.
    
            Parameters
            ----------
            dataset : SFrame | pandas.Dataframe
                Dataset of new observations. Must include columns with the same
                names as the features used for model training, but does not require
                a target column. Additional columns are ignored.
    
            missing_value_action : str, optional
                Action to perform when missing values are encountered. This can be
                one of:
    
                - 'auto': Default to 'impute'
                - 'impute': Proceed with evaluation by filling in the missing
                  values with the mean of the training data. Missing
                  values are also imputed if an entire column of data is
                  missing during evaluation.
                - 'error': Do not proceed with prediction and terminate with
                  an error message.
    
    
            Returns
            -------
            out : SArray
                Predicted target value for each example (i.e. row) in the dataset.
    
            See Also
            ----------
            create, evaluate
    
            Examples
            ----------
            >>> data =  graphlab.SFrame('http://s3.amazonaws.com/dato-datasets/regression/houses.csv')
    
            >>> model = graphlab.linear_regression.create(data,
                                                 target='price',
                                                 features=['bath', 'bedroom', 'size'])
            >>> results = model.predict(data)
            


Although I'm predicting values, I'll use real data points so that the
values can be checked.

.. code:: python

    new_data = graphlab.SFrame({'CrimeRate': [sales[0]['CrimeRate']]})
    prediction = crime_model.predict(new_data)
    actual = sales[0]['HousePrice']
    print("Prediction: {0:.2f}".format(prediction[0]))
    print("Actual: {0:.2f}".format(actual))
    print('Difference: {0:.2f}'.format(prediction[0] - actual))


.. parsed-literal::

    Prediction: 159494.94
    Actual: 140463.00
    Difference: 19031.94


.. code:: python

    outlier_check = crime_model.predict(outlier)
    print("Prediction: {0:.2f}".format(outlier_check[0]))
    print("Actual Data: {0:.2f}".format(outlier['HousePrice']))
    print("Error predicting the outlier: {0:.2f}".format(outlier['HousePrice'] - outlier_check[0]))


.. parsed-literal::

    Prediction: -34542.24
    Actual Data: 96200.00
    Error predicting the outlier: 130742.24

