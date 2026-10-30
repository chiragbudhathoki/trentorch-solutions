import numpy as np


def delivery_time_regressor(
    x: np.ndarray, y: np.ndarray, queries: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    x: n x d training feature matrix.
    y: length-n training targets.
    queries: m x d feature matrix to predict for.

    Fit w = pinv(X_aug) @ y, where X_aug prepends a column of ones to x
    (intercept term). Use an SVD-based pseudoinverse or lstsq, never an
    explicit inverse of X^T X, so collinear / rank-deficient features
    don't blow up.


    Return (intercept, coefficients, predictions), where coefficients has
    shape (d,) and predictions has shape (m,).
    """
    # TODO: augment with an intercept column, solve via pseudoinverse/lstsq,
    # then apply the fitted model to each query row.
    
    x_aug = np.c_[np.ones(x.shape[0]),x]
    #psuedoinverse X+
    x_plus = np.linalg.pinv(x_aug)
    #weight
    w = x_plus @ y
    #unpacking the weight
    bias = w[0]
    weight = w[1:]
    yhat = queries @ weight + bias
    return (bias,weight,yhat)
