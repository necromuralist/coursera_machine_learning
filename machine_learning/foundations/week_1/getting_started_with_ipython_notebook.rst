
Getting Started With Python and GraphLab Create
===============================================

Prerequisites
-------------

-  Installed Python
-  Started Ipython Notebook

Getting started with Python
---------------------------

.. code:: python

    print( 'Hello World!')


.. parsed-literal::

    Hello World!


Create some variables in Python
-------------------------------

.. code:: python

    i = 4  #int

.. code:: python

    type(i)




.. parsed-literal::

    int



.. code:: python

    f = 4.1  #float

.. code:: python

    type(f)




.. parsed-literal::

    float



.. code:: python

    b = True  #boolean variable

.. code:: python

    s = "This is a string!"

.. code:: python

    print s


.. parsed-literal::

    This is a string!


Advanced python types
---------------------

.. code:: python

    l = [3,1,2]  #list

.. code:: python

    print(l)


.. parsed-literal::

    [3, 1, 2]


.. code:: python

    d = {'foo':1, 'bar':2.3, 's':'my first dictionary'}  #dictionary

.. code:: python

    print d


.. parsed-literal::

    {'s': 'my first dictionary', 'foo': 1, 'bar': 2.3}


.. code:: python

    print d['foo']  #element of a dictionary


.. parsed-literal::

    1


.. code:: python

    n = None  #Python's null type

.. code:: python

    type(n)




.. parsed-literal::

    NoneType



Advanced printing
-----------------

.. code:: python

    print "Our float value is %s. Our int value is %s." % (f,i)  #Python is pretty good with strings


.. parsed-literal::

    Our float value is 4.1. Our int value is 4.


Conditional statements in python
--------------------------------

.. code:: python

    if i == 1 and f > 4:
        print "The value of i is 1 and f is greater than 4."
    elif i > 4 or f > 4:
        print "i or f are both greater than 4."
    else:
        print "both i and f are less than or equal to 4"



.. parsed-literal::

    i or f are both greater than 4.


Conditional loops
-----------------

.. code:: python

    print l


.. parsed-literal::

    [3, 1, 2]


.. code:: python

    for e in l:
        print e


.. parsed-literal::

    3
    1
    2


Note that in Python, we don't use {} or other markers to indicate the
part of the loop that gets iterated. Instead, we just indent and align
each of the iterated statements with spaces or tabs. (You can use as
many as you want, as long as the lines are aligned.)

.. code:: python

    counter = 6
    while counter < 10:
        print counter
        counter += 1


.. parsed-literal::

    6
    7
    8
    9


Creating functions in Python
----------------------------

Again, we don't use {}, but just indent the lines that are part of the
function.

.. code:: python

    def add2(x):
        y = x + 2
        return y

.. code:: python

    i = 5

.. code:: python

    add2(i)




.. parsed-literal::

    7



We can also define simple functions with lambdas:

.. code:: python

    square = lambda x: x*x

.. code:: python

    square(5)




.. parsed-literal::

    25


