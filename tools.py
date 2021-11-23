import numpy as np
import math
import skimage.measure as measure

def RSE(x, x_true, tol=0):
    #||X-Y||_F / ||Y||_F
    return np.linalg.norm(x - x_true) / (np.linalg.norm(x_true)+tol)

def PSNR(x, x_true, MAX=1, average=False):
    #10 log10(1^2/MSE) MSE = ||X-Y||_F^2 / num(Y)
    if average and len(x.shape) > 2:
        return np.mean([PSNR(x[:,:,i], x_true[:,:,i], MAX, average=False) for i in range(x.shape[-1])])
    else:
        mse = np.linalg.norm(x - x_true)**2 / np.size(x)
        return 10*math.log10(MAX**2 / mse)

def SSIM(im1 , im2):
    return measure.compare_ssim(im1, im2, dynamic_range=1, multichannel=True)
    # u_true = np.mean(y_true)
    # u_pred = np.mean(y_pred)
    # var_true = np.var(y_true)
    # var_pred = np.var(y_pred)
    # std_true = np.sqrt(var_true)
    # std_pred = np.sqrt(var_pred)
    # c1 = np.square(0.01*7)
    # c2 = np.square(0.03*7)
    # ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    # denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    # return ssim / denom

def MSE(y_true, y_pred, axis=None):
    return np.mean((y_true - y_pred) ** 2, axis=axis)

def RMSE(y_true, y_pred, axis=None):
    """Returns the regularised mean squared error between the two predictions
    (the square-root is applied to the mean_squared_error)
    """
    return np.sqrt(MSE(y_true, y_pred, axis=axis))

def reflective_correlation_coefficient(y_true, y_pred, axis=None):
    """Reflective variant of Pearson's product moment correlation coefficient
    where the predictions are not centered around their mean values.

    Parameters
    ----------
    y_true : array of shape (n_samples, )
        Ground truth (correct) target values.
    y_pred : array of shape (n_samples, )
        Estimated target values.

    Returns
    -------
    float: reflective correlation coefficient
    """
    return np.sum(y_true*y_pred, axis=axis)/np.sqrt(np.sum(y_true**2, axis=axis)*np.sum(y_pred**2, axis=axis))


def covariance(y_true, y_pred, axis=None):
    centered_true = np.mean(y_true, axis=axis)
    centered_pred = np.mean(y_pred, axis=axis)

    if axis is not None:
        # TODO: write a function to do this..
        shape = list(np.shape(y_true))
        shape[axis] = 1
        centered_true = np.reshape(centered_true, shape)
        shape = list(np.shape(y_pred))
        shape[axis] = 1
        centered_pred = np.reshape(centered_pred, shape)

    return np.mean((y_true - centered_true)*(y_pred - centered_pred), axis=axis)


def variance(y, axis=None):
    return covariance(y, y, axis=axis)


def standard_deviation(y, axis=None):
    return np.sqrt(variance(y, axis=axis))


def correlation(y_true, y_pred, axis=None):
    """Pearson's product moment correlation coefficient"""
    return covariance(y_true, y_pred, axis=axis)/np.sqrt(variance(y_true, axis)*variance(y_pred, axis))


