def assert_almost_equal(first, second, tolerance=.0000001):
    """
    assert the difference is within tolerance

    :param:
     - `first`: first term (float)
     - `second`: second term
     - `tolerance`: upper bound for the size of the difference
    :raise: AssertionError if difference > tolerance
    """
    assert abs(first - second) < tolerance, \
        "Term 1: {t1}, Term 2: {t2} Difference: {d} Tolerance: {t}".format(t1=first,
                                                                           t2=second,
                                                                           d=abs(first-second),
                                                                           t=tolerance)
