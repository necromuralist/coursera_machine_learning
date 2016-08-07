function unittests
    jupyter nbconvert linear_regression_with_sklearn.ipynb --to python
    jupyter nbconvert unit_tests.ipynb --to python
    nosetests unit_tests.py
end
