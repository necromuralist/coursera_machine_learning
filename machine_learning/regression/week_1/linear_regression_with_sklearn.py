
# coding: utf-8

# # Linear Regression With Sklearn

# This is a summary (by example) of how to perform a linear regression.

# ## Imports

# In[9]:

# third party

from IPython import get_ipython
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
import sklearn


# In[10]:

if __name__ != 'linear_regression_with_sklearn':
    get_ipython().magic(u'matplotlib inline')


# ## Load the data

# In[11]:

sales = pandas.read_csv('../../../large_data/csvs/Philadelphia_Crime_Rate_noNA.csv')
sales.head()


# In[12]:

sales.shape


# ## Fit the regression model

# The target here it the sale-price of a house ('HousePrice') and the prediction variable is the crime-rate in the house's area ('CrimeRate')

# In[13]:

TARGET = 'HousePrice'
FEATURES = ['CrimeRate']


# In[14]:

#crime_model = graphlab.linear_regression.create(sales, target='HousePrice',
#                                                features=['CrimeRate'],
#                                                validation_set=None,
#                                                verbose=False)
crime_model = None


# ## Plot the line

# In[15]:

def plot_data(data, model, title):
    figure = plt.figure()
    axe = figure.gca()
    lines = axe.plot(data['CrimeRate'],data['HousePrice'],'.', label='Data')
    lines = axe.plot(data['CrimeRate'], model.predict(data),'-', label='Fit')
    label = axe.set_xlabel("Crime Rate")
    label = axe.set_ylabel("House Price")
    title = axe.set_title(title)
    legend = axe.legend()


# In[16]:

#plot_data(sales, crime_model, 'Philadelpdhia Crime Rate vs House Price')


# ## Identify the outlier

# In[23]:

maximum_crime = sales['CrimeRate'].argmax()
outlier = sales.ix[maximum_crime]
print(outlier)


# ## Get the model coefficients

# In[25]:

#coefficients = crime_model.get('coefficients')

#print(coefficients)


# In[26]:

#intercept, slope = coefficients['value']
#print("y = {m:.2f} x + {b:.2f}".format(m=slope, b=intercept))


# ## Predict House Price based on new crime rate

# Although I'm predicting values, I'll use real data points so that the values can be checked.

# In[ ]:

new_data = graphlab.SFrame({'CrimeRate': [sales[0]['CrimeRate']]})
prediction = crime_model.predict(new_data)
actual = sales[0]['HousePrice']
print("Prediction: {0:.2f}".format(prediction[0]))
print("Actual: {0:.2f}".format(actual))
print('Difference: {0:.2f}'.format(prediction[0] - actual))


# In[ ]:

outlier_check = crime_model.predict(outlier)
print("Prediction: {0:.2f}".format(outlier_check[0]))
print("Actual Data: {0:.2f}".format(outlier['HousePrice']))
print("Error predicting the outlier: {0:.2f}".format(outlier['HousePrice'] - outlier_check[0]))

