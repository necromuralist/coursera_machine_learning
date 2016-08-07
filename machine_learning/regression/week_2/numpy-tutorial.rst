
02 - Numpy Tutorial
===================

Numpy is a computational library for Python that is optimized for
operations on multi-dimensional arrays. In this notebook we will use
numpy to work with 1-d arrays (often called vectors) and 2-d arrays
(often called matrices).

For a the full user guide and reference for numpy see:
http://docs.scipy.org/doc/numpy/

.. code:: python

    import numpy as np

Creating Numpy Arrays
---------------------

New arrays can be made in several ways. We can take an existing list and
convert it to a numpy array:

.. code:: python

    mylist = [1., 2., 3., 4.]
    mynparray = np.array(mylist)
    mynparray




.. parsed-literal::

    array([ 1.,  2.,  3.,  4.])



You can initialize an array (of any dimension) of all ones or all zeroes
with the ones() and zeros() functions:

.. code:: python

    one_vector = np.ones(4)
    print one_vector # using print removes the array() portion


.. parsed-literal::

    [ 1.  1.  1.  1.]


.. code:: python

    one2Darray = np.ones((2, 4)) # an 2D array with 2 "rows" and 4 "columns"
    print one2Darray


.. parsed-literal::

    [[ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]]


.. code:: python

    zero_vector = np.zeros(4)
    print zero_vector


.. parsed-literal::

    [ 0.  0.  0.  0.]


You can also initialize an empty array which will be filled with values.
This is the fastest way to initialize a fixed-size numpy array however
you must ensure that you replace all of the values.

.. code:: python

    empty_vector = np.empty(5)
    print empty_vector


.. parsed-literal::

    [  6.94467966e-310   6.94467966e-310   6.94467966e-310   6.94467966e-310
       2.37151510e-322]


Accessing Array Values
----------------------

Accessing an array is straight forward. For vectors you access the index
by referring to it inside square brackets. Recall that indices in Python
start with 0.

.. code:: python

    mynparray[2]




.. parsed-literal::

    3.0



2D arrays are accessed similarly by referring to the row and column
index separated by a comma:

.. code:: python

    my_matrix = np.array([[1, 2, 3], [4, 5, 6]])
    print my_matrix


.. parsed-literal::

    [[1 2 3]
     [4 5 6]]


.. code:: python

    print my_matrix[1, 2]


.. parsed-literal::

    6


Sequences of indices can be accessed using ':' for example

.. code:: python

    print my_matrix[0:2, 2] # recall 0:2 = [0, 1]


.. parsed-literal::

    [3 6]


.. code:: python

    print my_matrix[0, 0:3]


.. parsed-literal::

    [1 2 3]


You can also pass a list of indices.

.. code:: python

    fib_indices = np.array([1, 1, 2, 3])
    random_vector = np.random.random(10) # 10 random numbers between 0 and 1
    print random_vector


.. parsed-literal::

    [ 0.5257354   0.76694778  0.57363726  0.91460887  0.25633289  0.07798102
      0.71637898  0.31806257  0.95406657  0.43416871]


.. code:: python

    print random_vector[fib_indices]


.. parsed-literal::

    [ 0.76694778  0.76694778  0.57363726  0.91460887]


You can also use true/false values to select values

.. code:: python

    my_vector = np.array([1, 2, 3, 4])
    select_index = np.array([True, False, True, False])
    print my_vector[select_index]


.. parsed-literal::

    [1 3]


For 2D arrays you can select specific columns and specific rows. Passing
':' selects all rows/columns

.. code:: python

    select_cols = np.array([True, False, True]) # 1st and 3rd column
    select_rows = np.array([False, True]) # 2nd row

.. code:: python

    print my_matrix[select_rows, :] # just 2nd row but all columns


.. parsed-literal::

    [[4 5 6]]


.. code:: python

    print my_matrix[:, select_cols] # all rows and just the 1st and 3rd column


.. parsed-literal::

    [[1 3]
     [4 6]]


Operations on Arrays
--------------------

You can use the operations '\*', '\*\*', '\\', '+' and '-' on numpy
arrays and they operate elementwise.

.. code:: python

    my_array = np.array([1., 2., 3., 4.])
    print my_array*my_array


.. parsed-literal::

    [  1.   4.   9.  16.]


.. code:: python

    print my_array**2


.. parsed-literal::

    [  1.   4.   9.  16.]


.. code:: python

    print my_array - np.ones(4)


.. parsed-literal::

    [ 0.  1.  2.  3.]


.. code:: python

    print my_array + np.ones(4)


.. parsed-literal::

    [ 2.  3.  4.  5.]


.. code:: python

    print my_array / 3


.. parsed-literal::

    [ 0.33333333  0.66666667  1.          1.33333333]


.. code:: python

    print my_array / np.array([2., 3., 4., 5.]) # = [1.0/2.0, 2.0/3.0, 3.0/4.0, 4.0/5.0]


.. parsed-literal::

    [ 0.5         0.66666667  0.75        0.8       ]


You can compute the sum with np.sum() and the average with np.average()

.. code:: python

    print np.sum(my_array)


.. parsed-literal::

    10.0


.. code:: python

    print np.average(my_array)


.. parsed-literal::

    2.5


.. code:: python

    my_array.mean()




.. parsed-literal::

    2.5



.. code:: python

    print np.sum(my_array)/len(my_array)


.. parsed-literal::

    2.5


The dot product
---------------

An important mathematical operation in linear algebra is the dot
product.

When we compute the dot product between two vectors we are simply
multiplying them elementwise and adding them up. In numpy you can do
this with np.dot()

.. code:: python

    array1 = np.array([1., 2., 3., 4.])
    array2 = np.array([2., 3., 4., 5.])
    print np.dot(array1, array2)


.. parsed-literal::

    40.0


.. code:: python

    print np.sum(array1*array2)


.. parsed-literal::

    40.0


Recall that the Euclidean length (or magnitude) of a vector is the
squareroot of the sum of the squares of the components. This is just the
squareroot of the dot product of the vector with itself:

.. code:: python

    array1_mag = np.sqrt(np.dot(array1, array1))
    print array1_mag


.. parsed-literal::

    5.47722557505


.. code:: python

    print np.sqrt(np.sum(array1*array1))


.. parsed-literal::

    5.47722557505


We can also use the dot product when we have a 2D array (or matrix).
When you have an vector with the same number of elements as the matrix
(2D array) has columns you can right-multiply the matrix by the vector
to get another vector with the same number of elements as the matrix has
rows. For example this is how you compute the predicted values given a
matrix of features and an array of weights.

.. code:: python

    my_features = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    print my_features


.. parsed-literal::

    [[ 1.  2.]
     [ 3.  4.]
     [ 5.  6.]
     [ 7.  8.]]


.. code:: python

    my_weights = np.array([0.4, 0.5])
    print my_weights


.. parsed-literal::

    [ 0.4  0.5]


.. code:: python

    my_predictions = np.dot(my_features, my_weights) # note that the weights are on the right
    print my_predictions # which has 4 elements since my_features has 4 rows


.. parsed-literal::

    [ 1.4  3.2  5.   6.8]


Similarly if you have a vector with the same number of elements as the
matrix has *rows* you can left multiply them.

.. code:: python

    my_matrix = my_features
    my_array = np.array([0.3, 0.4, 0.5, 0.6])

.. code:: python

    print np.dot(my_array, my_matrix) # which has 2 elements because my_matrix has 2 columns


.. parsed-literal::

    [  8.2  10. ]


.. code:: python

    ## Multiplying Matrices

If we have two 2D arrays (matrices) matrix\_1 and matrix\_2 where the
number of columns of matrix\_1 is the same as the number of rows of
matrix\_2 then we can use np.dot() to perform matrix multiplication.

.. code:: python

    matrix_1 = np.array([[1., 2., 3.],[4., 5., 6.]])
    print matrix_1


.. parsed-literal::

    [[ 1.  2.  3.]
     [ 4.  5.  6.]]


.. code:: python

    matrix_2 = np.array([[1., 2.], [3., 4.], [5., 6.]])
    print matrix_2


.. parsed-literal::

    [[ 1.  2.]
     [ 3.  4.]
     [ 5.  6.]]


.. code:: python

    print np.dot(matrix_1, matrix_2)


.. parsed-literal::

    [[ 22.  28.]
     [ 49.  64.]]

