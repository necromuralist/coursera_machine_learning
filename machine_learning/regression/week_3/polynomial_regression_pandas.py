
# python standard library
from collections import namedtuple

# third-party
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy
import pandas
import seaborn
from sklearn.cross_validation import train_test_split
import statsmodels.api as statsmodels

# this code
from regression_model import FrameRegressionModel

figure_prefix = "polynomial_regression_pandas_"

sales_frame = pandas.read_csv('../../large_data/csvs/kc_house_data.csv')

sales_frame = sales_frame.sort_values(by='sqft_living')

model_frame_1 = FrameRegressionModel(data=sales_frame)

path = model_frame_1.plot_fit(figure_prefix + 'degree_1.png')
print(".. image:: {0}".format(path))