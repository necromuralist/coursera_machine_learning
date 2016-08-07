function functionaltests
    jupyter nbconvert linear_regression_with_sklearn.ipynb --to python
    jupyter nbconvert functional_tests.ipynb --to python
    python functional_tests.py
end
