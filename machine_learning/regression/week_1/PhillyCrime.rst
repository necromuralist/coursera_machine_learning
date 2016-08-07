
Fire up graphlab create
=======================

.. code:: python

    import graphlab

Load some house value vs. crime rate data
=========================================

Dataset is from Philadelphia, PA and includes average house sales price
in a number of neighborhoods. The attributes of each neighborhood we
have include the crime rate ('CrimeRate'), miles from Center City
('MilesPhila'), town name ('Name'), and county name ('County').

.. code:: python

    sales = graphlab.SFrame.read_csv('Philadelphia_Crime_Rate_noNA.csv/')


.. parsed-literal::

    PROGRESS: Finished parsing file /home/charon/repositories/code/explorations/machine_learning_experiments/machine_learning/coursera/regression/Philadelphia_Crime_Rate_noNA.csv
    PROGRESS: Parsing completed. Parsed 99 lines in 0.051951 secs.
    ------------------------------------------------------
    Inferred types from first line of file as 
    column_type_hints=[int,float,float,float,float,str,str]
    If parsing fails due to incorrect types, you can correct
    the inferred type list above and pass it to read_csv in
    the column_type_hints argument
    ------------------------------------------------------
    PROGRESS: Finished parsing file /home/charon/repositories/code/explorations/machine_learning_experiments/machine_learning/coursera/regression/Philadelphia_Crime_Rate_noNA.csv
    PROGRESS: Parsing completed. Parsed 99 lines in 0.021983 secs.


.. parsed-literal::

    [INFO] GraphLab Server Version: 1.7.1
    [INFO] Start server at: ipc:///tmp/graphlab_server-27021 - Server binary: /home/charon/.virtualenvs/machinelearning/lib/python2.7/site-packages/graphlab/unity_server - Server log: /tmp/graphlab_server_1449012840.log
    [INFO] [1;32m1449012840 : INFO:     (initialize_globals_from_environment:282): Setting configuration variable GRAPHLAB_FILEIO_ALTERNATIVE_SSL_CERT_FILE to /home/charon/.virtualenvs/machinelearning/local/lib/python2.7/site-packages/certifi/cacert.pem
    [0m[1;32m1449012840 : INFO:     (initialize_globals_from_environment:282): Setting configuration variable GRAPHLAB_FILEIO_ALTERNATIVE_SSL_CERT_DIR to 
    [0mThis non-commercial license of GraphLab Create is assigned to necromuralist@gmail.com and will expire on October 20, 2016. For commercial licensing options, visit https://dato.com/buy/.
    


.. code:: python

    sales




.. parsed-literal::

    Columns:
    	HousePrice	int
    	HsPrc ($10,000)	float
    	CrimeRate	float
    	MilesPhila	float
    	PopChg	float
    	Name	str
    	County	str
    
    Rows: 99
    
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
    [99 rows x 7 columns]
    Note: Only the head of the SFrame is printed.
    You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.



Exploring the data
==================

The house price in a town is correlated with the crime rate of that
town. Low crime towns tend to be associated with higher house prices and
vice versa.

.. code:: python

    graphlab.canvas.set_target('ipynb')
    sales.show(view="Scatter Plot", x="CrimeRate", y="HousePrice")




Fit the regression model using crime as the feature
===================================================

.. code:: python

    crime_model = graphlab.linear_regression.create(sales, target='HousePrice', features=['CrimeRate'],validation_set=None,verbose=False)

Let's see what our fit looks like
=================================

Matplotlib is a Python plotting library that is also useful for
plotting. You can install it with:

'pip install matplotlib'

.. code:: python

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    %matplotlib inline

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
        
    plot_data(sales, crime_model, 'Philadelpdhia Crime Rate vs House Price')



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f95701bcc10>


Above: red dots are original data, blue line is the fit from the simple
regression.

Remove Center City and redo the analysis
========================================

Center City is the one observation with an extremely high crime rate,
yet house prices are not very low. This point does not follow the trend
of the rest of the data very well. A question is how much including
Center City is influencing our fit on the other datapoints. Let's remove
this datapoint and see what happens.

.. code:: python

    maximum_crime = sales['CrimeRate'].argmax()
    outlier = sales[maximum_crime]
    print(outlier)


.. parsed-literal::

    {'Name': 'Phila,CC', 'PopChg': 4.8, 'County': 'Phila', 'HousePrice': 96200, 'MilesPhila': 0.0, 'HsPrc ($10,000)': 9.62, 'CrimeRate': 366.1}



.. code:: python

    sales_noCC = sales[sales['CrimeRate'] != outlier['CrimeRate']] 

.. code:: python

    sales_noCC.show(view="Scatter Plot", x="CrimeRate", y="HousePrice")




Refit our simple regression model on this modified dataset:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    crime_model_noCC = graphlab.linear_regression.create(sales_noCC, target='HousePrice', features=['CrimeRate'],validation_set=None, verbose=False)

Look at the fit:
~~~~~~~~~~~~~~~~

.. code:: python

    plot_data(sales_noCC, crime_model_noCC, "Phil Crime vs House Price (outlier removed)")



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ff87d3cbb10>


Compare coefficients for full-data fit versus no-Center-City fit
================================================================

Visually, the fit seems different, but let's quantify this by examining
the estimated coefficients of our original fit and that of the modified
dataset with Center City removed.

.. code:: python

    coefficients = crime_model.get('coefficients')
    intercept, slope = coefficients['value']
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

    print("y = {m:.2f} x + {b:.2f}".format(m=slope, b=intercept))


.. parsed-literal::

    y = -576.80 x + 176626.05


.. code:: python

    noCC_coefficients = crime_model_noCC.get('coefficients')
    noCC_intercept, noCC_slope = noCC_coefficients['value']
    print(noCC_coefficients)


.. parsed-literal::

    +-------------+-------+----------------+
    |     name    | index |     value      |
    +-------------+-------+----------------+
    | (intercept) |  None | 225204.604303  |
    |  CrimeRate  |  None | -2287.69717443 |
    +-------------+-------+----------------+
    [2 rows x 3 columns]
    


.. code:: python

    print('Slope Difference (price drop per increase in crime) between with influential observation and without: {d:.2f}'.format(d=noCC_slope - slope))


.. parsed-literal::

    Slope Difference (price drop per increase in crime) between with influential observation and without: -1710.89


Above: We see that for the "no Center City" version, per unit increase
in crime, the predicted decrease in house prices is 2,287. In contrast,
for the original dataset, the drop is only 576 per unit increase in
crime. This is significantly different!

High leverage points:
~~~~~~~~~~~~~~~~~~~~~

Center City is said to be a "high leverage" point because it is at an
extreme x value where there are not other observations. As a result,
recalling the closed-form solution for simple regression, this point has
the *potential* to dramatically change the least squares line since the
center of mass is heavily influenced by this one point and the least
squares line will try to fit close to that outlying (in x) point. If a
high leverage point follows the trend of the other data, this might not
have much effect. On the other hand, if this point somehow differs, it
can be strongly influential in the resulting fit.

Influential observations:
~~~~~~~~~~~~~~~~~~~~~~~~~

An influential observation is one where the removal of the point
significantly changes the fit. As discussed above, high leverage points
are good candidates for being influential observations, but need not be.
Other observations that are *not* leverage points can also be
influential observations (e.g., strongly outlying in y even if x is a
typical value).

Remove high-value outlier neighborhoods and redo analysis
=========================================================

Based on the discussion above, a question is whether the outlying
high-value towns are strongly influencing the fit. Let's remove them and
see what happens.

.. code:: python

    sales_nohighend = sales_noCC[sales_noCC['HousePrice'] < 350000] 
    crime_model_nohighend = graphlab.linear_regression.create(sales_nohighend, target='HousePrice', features=['CrimeRate'],validation_set=None, verbose=False)

Do the coefficients change much?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    crime_model_noCC.get('coefficients')




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
        <tr>
            <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
            <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
            <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
        </tr>
        <tr>
            <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
            <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
            <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">225204.604303</td>
        </tr>
        <tr>
            <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">CrimeRate</td>
            <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
            <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-2287.69717443</td>
        </tr>
    </table>
    [2 rows x 3 columns]<br/>
    </div>



.. code:: python

    no_highend_coefficients = crime_model_nohighend.get('coefficients')
    print(no_highend_coefficients)


.. parsed-literal::

    +-------------+-------+----------------+
    |     name    | index |     value      |
    +-------------+-------+----------------+
    | (intercept) |  None | 199073.589615  |
    |  CrimeRate  |  None | -1837.71280989 |
    +-------------+-------+----------------+
    [2 rows x 3 columns]
    


.. code:: python

    nohigh_intercept, nohigh_slope = no_highend_coefficients['value']
    print("Difference in slope: {d:.2f}".format(d=nohigh_slope - noCC_slope))


.. parsed-literal::

    Difference in slope: 449.98


Above: We see that removing the outlying high-value neighborhoods has
*some* effect on the fit, but not nearly as much as our high-leverage
Center City datapoint.

.. code:: python

    plot_data(sales_nohighend, crime_model_nohighend, "Philadelphia House Price vs Crime (no highend)")



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7ff87cb64610>

