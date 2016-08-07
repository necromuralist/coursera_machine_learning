def residual_sum_of_squares(model, data, target_data, verbose=False):
    """
    Calculate the residuals sum of squares

    :param:
     - `model`: model fitted to training data
     - `data`: data to use to make predictions
     - `targe_data`: test data for the column you are predicting
     - `verbose`: whether to print the steps as they go
    """
    if verbose:
        print('getting predictions from data')
    predictions = model.predict(data)

    if verbose:
        print("computing the residuals/errors")
    residuals = target_data - predictions

    if verbose:
        print("calculating the sum of the squares of the residuals")
    RSS = (residuals**2).sum()
    return(RSS)
