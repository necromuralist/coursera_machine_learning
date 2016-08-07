
# python standard library
from abc import ABCMeta, abstractproperty, abstractmethod

# third party
import matplotlib.pyplot as plt
import pandas
import statsmodels.api as statsmodels

class BaseRegressionModel(object):
    """
    Base regression model
    """
    __metaclass__ = ABCMeta
    def __init__(self, data, degree=1, predictor='sqft_living',
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
        
    def plot_fit(self, filename, output=True):
        """
        Plot the data and regression line

        :param:
         - `filename`: name to save the file to
         - `output`: print rst directive if True
        :return: path to the saved image
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
        file_path = 'figures/{0}'.format(filename)
        figure.savefig(file_path)
        print('.. image:: {0}'.format(file_path))
        return file_path

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

class FrameRegressionModel(BaseRegressionModel):
    """
    Concrete RegressionModel for DataFrames
    """
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