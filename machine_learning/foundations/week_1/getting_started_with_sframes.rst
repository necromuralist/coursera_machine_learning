
Getting Started with SFrames
============================

Fire up GraphLab Create
-----------------------

We always start with this line before using any part of GraphLab Create

.. code:: python

    import graphlab

Load a tabular data set
-----------------------

.. code:: python

    sf = graphlab.SFrame('people-example.csv')


.. parsed-literal::

    PROGRESS: Finished parsing file /home/charon/repositories/code/explorations/machine_learning_experiments/machine_learning/coursera/foundations/week_1/people-example.csv
    PROGRESS: Parsing completed. Parsed 7 lines in 0.062633 secs.
    ------------------------------------------------------
    Inferred types from first line of file as 
    column_type_hints=[str,str,str,int]
    If parsing fails due to incorrect types, you can correct
    the inferred type list above and pass it to read_csv in
    the column_type_hints argument
    ------------------------------------------------------
    PROGRESS: Finished parsing file /home/charon/repositories/code/explorations/machine_learning_experiments/machine_learning/coursera/foundations/week_1/people-example.csv
    PROGRESS: Parsing completed. Parsed 7 lines in 0.02343 secs.


.. parsed-literal::

    [INFO] GraphLab Server Version: 1.7.1
    [INFO] Start server at: ipc:///tmp/graphlab_server-17743 - Server binary: /home/charon/.virtualenvs/machinelearning/lib/python2.7/site-packages/graphlab/unity_server - Server log: /tmp/graphlab_server_1449346723.log
    [INFO] [1;32m1449346723 : INFO:     (initialize_globals_from_environment:282): Setting configuration variable GRAPHLAB_FILEIO_ALTERNATIVE_SSL_CERT_FILE to /home/charon/.virtualenvs/machinelearning/local/lib/python2.7/site-packages/certifi/cacert.pem
    [0m[1;32m1449346723 : INFO:     (initialize_globals_from_environment:282): Setting configuration variable GRAPHLAB_FILEIO_ALTERNATIVE_SSL_CERT_DIR to 
    [0mThis non-commercial license of GraphLab Create is assigned to necromuralist@gmail.com and will expire on October 20, 2016. For commercial licensing options, visit https://dato.com/buy/.
    


Frame basics
------------

.. code:: python

    sf #we can view first few lines of table




.. parsed-literal::

    Columns:
    	First Name	str
    	Last Name	str
    	Country	str
    	age	int
    
    Rows: 7
    
    Data:
    +------------+-----------+---------------+-----+
    | First Name | Last Name |    Country    | age |
    +------------+-----------+---------------+-----+
    |    Bob     |   Smith   | United States |  24 |
    |   Alice    |  Williams |     Canada    |  23 |
    |  Malcolm   |    Jone   |    England    |  22 |
    |   Felix    |   Brown   |      USA      |  23 |
    |    Alex    |   Cooper  |     Poland    |  23 |
    |    Tod     |  Campbell | United States |  22 |
    |   Derek    |    Ward   |  Switzerland  |  25 |
    +------------+-----------+---------------+-----+
    [7 rows x 4 columns]



.. code:: python

    sf.tail()  # view end of the table




.. parsed-literal::

    Columns:
    	First Name	str
    	Last Name	str
    	Country	str
    	age	int
    
    Rows: 7
    
    Data:
    +------------+-----------+---------------+-----+
    | First Name | Last Name |    Country    | age |
    +------------+-----------+---------------+-----+
    |    Bob     |   Smith   | United States |  24 |
    |   Alice    |  Williams |     Canada    |  23 |
    |  Malcolm   |    Jone   |    England    |  22 |
    |   Felix    |   Brown   |      USA      |  23 |
    |    Alex    |   Cooper  |     Poland    |  23 |
    |    Tod     |  Campbell | United States |  22 |
    |   Derek    |    Ward   |  Switzerland  |  25 |
    +------------+-----------+---------------+-----+
    [7 rows x 4 columns]



GraphLab Canvas
---------------

.. code:: python

    # .show() visualizes any data structure in GraphLab Create
    sf.show()


.. parsed-literal::

    Canvas is accessible via web browser at the URL: http://localhost:38269/index.html
    Opening Canvas in default web browser.


.. code:: python

    # If you want Canvas visualization to show up on this notebook, 
    # rather than popping up a new window, add this line:
    graphlab.canvas.set_target('ipynb')

.. code:: python

    sf['age'].show(view='Categorical')




Inspect columns of dataset
--------------------------

.. code:: python

    sf['Country']




.. parsed-literal::

    dtype: str
    Rows: 7
    ['United States', 'Canada', 'England', 'USA', 'Poland', 'United States', 'Switzerland']



.. code:: python

    sf['age']




.. parsed-literal::

    dtype: int
    Rows: 7
    [24, 23, 22, 23, 23, 22, 25]



Some simple columnar operations

.. code:: python

    sf['age'].mean()




.. parsed-literal::

    23.142857142857146



.. code:: python

    sf['age'].max()




.. parsed-literal::

    25



Create new columns in our SFrame
--------------------------------

.. code:: python

    sf




.. parsed-literal::

    Columns:
    	First Name	str
    	Last Name	str
    	Country	str
    	age	int
    
    Rows: 7
    
    Data:
    +------------+-----------+---------------+-----+
    | First Name | Last Name |    Country    | age |
    +------------+-----------+---------------+-----+
    |    Bob     |   Smith   | United States |  24 |
    |   Alice    |  Williams |     Canada    |  23 |
    |  Malcolm   |    Jone   |    England    |  22 |
    |   Felix    |   Brown   |      USA      |  23 |
    |    Alex    |   Cooper  |     Poland    |  23 |
    |    Tod     |  Campbell | United States |  22 |
    |   Derek    |    Ward   |  Switzerland  |  25 |
    +------------+-----------+---------------+-----+
    [7 rows x 4 columns]



.. code:: python

    sf['Full Name'] = sf['First Name'] + ' ' + sf['Last Name']

.. code:: python

    sf




.. parsed-literal::

    Columns:
    	First Name	str
    	Last Name	str
    	Country	str
    	age	int
    	Full Name	str
    
    Rows: 7
    
    Data:
    +------------+-----------+---------------+-----+----------------+
    | First Name | Last Name |    Country    | age |   Full Name    |
    +------------+-----------+---------------+-----+----------------+
    |    Bob     |   Smith   | United States |  24 |   Bob Smith    |
    |   Alice    |  Williams |     Canada    |  23 | Alice Williams |
    |  Malcolm   |    Jone   |    England    |  22 |  Malcolm Jone  |
    |   Felix    |   Brown   |      USA      |  23 |  Felix Brown   |
    |    Alex    |   Cooper  |     Poland    |  23 |  Alex Cooper   |
    |    Tod     |  Campbell | United States |  22 |  Tod Campbell  |
    |   Derek    |    Ward   |  Switzerland  |  25 |   Derek Ward   |
    +------------+-----------+---------------+-----+----------------+
    [7 rows x 5 columns]



.. code:: python

    sf['age'] * sf['age']




.. parsed-literal::

    dtype: int
    Rows: 7
    [576, 529, 484, 529, 529, 484, 625]



Use the apply function to do a advance transformation of our data
-----------------------------------------------------------------

.. code:: python

    sf['Country']




.. parsed-literal::

    dtype: str
    Rows: 7
    ['United States', 'Canada', 'England', 'USA', 'Poland', 'United States', 'Switzerland']



.. code:: python

    sf['Country'].show()




.. code:: python

    def transform_country(country):
        if country == 'USA':
            return 'United States'
        else:
            return country

.. code:: python

    transform_country('Brazil')




.. parsed-literal::

    'Brazil'



.. code:: python

    transform_country('Brasil')




.. parsed-literal::

    'Brasil'



.. code:: python

    transform_country('USA')




.. parsed-literal::

    'United States'



.. code:: python

    sf['Country'].apply(transform_country)




.. parsed-literal::

    dtype: str
    Rows: 7
    ['United States', 'Canada', 'England', 'United States', 'Poland', 'United States', 'Switzerland']



.. code:: python

    sf['Country'] = sf['Country'].apply(transform_country)

.. code:: python

    sf['Country'].show()




.. code:: python

    sf['Country'].apply(lambda x: x.lower())




.. parsed-literal::

    dtype: str
    Rows: 7
    ['united states', 'canada', 'england', 'united states', 'poland', 'united states', 'switzerland']



.. code:: python

    sf['Country'].show()



