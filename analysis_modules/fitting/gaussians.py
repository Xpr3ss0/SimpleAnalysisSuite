import numpy as np
from scipy.optimize import curve_fit

def fit_1d_gaussian(ydata, return_fit_data=False, initial_guess=None, **kwargs):
    """
    Fit a 1D Gaussian to the input data.
    Note: If xdata is not provided, x0 and sigma are in data coordinates.

    Args:
        data (1D array or tuple of two 1D arrays): The input data to fit. If a tuple, should be (xdata, ydata).
        return_fit_data (bool): Whether to return the fitted data.
        initial_guess (tuple): Initial guess for the parameters. Ordered as (amplitude, mean, sigma, offset).
        **kwargs: Additional keyword arguments to pass to curve_fit. Common options include sigma, absolute_sigma, maxfev, etc.

    Returns:
        dict: Fitted parameters and optionally the fitted data.

        Parameters:
        - amplitude: Peak height of the Gaussian.
        - xo: x-coordinate of the center (in data coordinates, unless xdata is provided).
        - sigma: Standard deviation (width) of the Gaussian (in data coordinates, unless xdata is provided).
        - offset: Constant offset.
        - covariance: The estimated covariance of the fitted parameters.
        - fit_data: The fitted 2D Gaussian data (if return_fit_data is True).
    """

    def gaussian_1d(x, amplitude, mean, sigma, offset):
        return offset + amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    if isinstance(ydata, tuple) and len(ydata) == 2:
        xdata, ydata = ydata
        if not isinstance(xdata, np.ndarray):
            xdata = np.array(xdata)
        if not isinstance(ydata, np.ndarray):
            ydata = np.array(ydata)
    elif isinstance(ydata, np.ndarray) or isinstance(ydata, list):
        xdata = np.arange(len(ydata))
        ydata = np.array(ydata)
    else:
        raise ValueError("Data must be a 1D array or a tuple of two 1D arrays (xdata, ydata).")

    # Initial guess for the parameters
    if initial_guess is None:
        initial_guess = (
            np.max(ydata),  # amplitude
            xdata[np.argmax(ydata)],  # mean
            (xdata[-1] - xdata[0]) / 10,  # sigma
            np.min(ydata)  # offset
        )

    # Fit the data
    try:
        popt, pcov = curve_fit(
            gaussian_1d,
            xdata,
            ydata,
            p0=initial_guess
        )
    except RuntimeError as e:
        print(f"Error fitting data: {e}")
        return None

    results = {
        'amplitude': popt[0],
        'mean': popt[1],
        'sigma': popt[2],
        'offset': popt[3],
        'covariance': pcov
    }

    if return_fit_data:
        fit_data = gaussian_1d(xdata, *popt)
        results['fit_data'] = fit_data

    return results


def fit_2d_gaussian_sym(data, return_fit_data=False, initial_guess=None, **kwargs):
    """
    Fit a 2D symmetric Gaussian to the input data.

    Args:
        data (2D array): The input data to fit.
        return_fit_data (bool): Whether to return the fitted data.
        initial_guess (tuple): Initial guess for the parameters. Ordered as (amplitude, xo, yo, sigma, offset).
        **kwargs: Additional keyword arguments to pass to curve_fit. Common options include sigma, absolute_sigma, maxfev, etc.

    Returns:
        dict: Fitted parameters and optionally the fitted data.

        Parameters:
        - amplitude: Peak height of the Gaussian.
        - xo: x-coordinate of the center (in pixels).
        - yo: y-coordinate of the center (in pixels).
        - sigma: Standard deviation (width) of the Gaussian (in pixels).
        - offset: Constant offset.
        - covariance: The estimated covariance of the fitted parameters.
        - fit_data: The fitted 2D Gaussian data (if return_fit_data is True).
    """
    def gaussian_2d_sym(xy, amplitude, xo, yo, sigma, offset):
        x, y = xy
        g = offset + amplitude * np.exp(
            -(((x - xo) ** 2 + (y - yo) ** 2) / (2 * sigma ** 2))
        )
        return g.ravel()

    # Create x and y indices
    y_indices, x_indices = np.indices(data.shape)

    # Initial guess for the parameters
    if initial_guess is None:
        initial_guess = (
            np.max(data),  # amplitude
            data.shape[1] / 2,  # xo
            data.shape[0] / 2,  # yo
            (data.shape[1] + data.shape[0]) / 20,  # sigma
            np.min(data)  # offset
        )

    # Fit the data
    try:
        popt, pcov = curve_fit(
            gaussian_2d_sym,
            (x_indices, y_indices),
            data.ravel(),
            p0=initial_guess
        )
    except RuntimeError as e:
        print(f"Error fitting data: {e}")
        return None
    
    results = {
        'amplitude': popt[0],
        'xo': popt[1],
        'yo': popt[2],
        'sigma': popt[3],
        'offset': popt[4],
        'covariance': pcov
    }

    if return_fit_data:
        fit_data = gaussian_2d_sym((x_indices, y_indices), *popt).reshape(data.shape)
        results['fit_data'] = fit_data

    return results


def fit_2d_gaussian_asym(data, return_fit_data=False, initial_guess=None, **kwargs):
    """
    Fit a 2D asymmetric Gaussian to the input data.

    Args:
        data (2D array): The input data to fit.
        return_fit_data (bool): Whether to return the fitted data.
        initial_guess (tuple): Initial guess for the parameters.
        **kwargs: Additional keyword arguments to pass to curve_fit. Common options include sigma, absolute_sigma, maxfev, etc.


    Returns:
        dict: Fitted parameters and optionally the fitted data.

        Parameters:
        - amplitude: Peak height of the Gaussian.
        - xo: x-coordinate of the center (in pixels).
        - yo: y-coordinate of the center (in pixels).
        - theta: Rotation angle of the Gaussian (in radians).
        - sigma_x: Standard deviation (width) of the Gaussian in the x-direction (in pixels).
        - sigma_y: Standard deviation (width) of the Gaussian in the y-direction (in pixels).
        - offset: Constant offset.
        - covariance: The estimated covariance of the fitted parameters.
        - fit_data: The fitted 2D Gaussian data (if return_fit_data is True
    """

    def gaussian_2d_asym(xy, amplitude, xo, yo, theta, sigma_x, sigma_y, offset):
        x, y = xy
        a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
        c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
        g = offset + amplitude * np.exp(
            -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
        )
        return g.ravel()

    # Create x and y indices
    y_indices, x_indices = np.indices(data.shape)

    # Initial guess for the parameters
    if initial_guess is None:
        initial_guess = (
            np.max(data),  # amplitude
            data.shape[1] / 2,  # xo
            data.shape[0] / 2,  # yo
            0,  # theta
            (data.shape[1] + data.shape[0]) / 20,  # sigma_x
            (data.shape[1] + data.shape[0]) / 20,  # sigma_y
            np.min(data)  # offset
        )

    # Fit the data
    try:
        popt, pcov = curve_fit(
            gaussian_2d_asym,
            (x_indices, y_indices),
            data.ravel(),
            p0=initial_guess
        )
    except RuntimeError as e:
        print(f"Error fitting data: {e}")
        return None

    results = {
        'amplitude': popt[0],
        'xo': popt[1],
        'yo': popt[2],
        'theta': popt[3],
        'sigma_x': popt[4],
        'sigma_y': popt[5],
        'offset': popt[6],
        'covariance': pcov
    }

    if return_fit_data:
        fit_data = gaussian_2d_asym((x_indices, y_indices), *popt).reshape(data.shape)
        results['fit_data'] = fit_data

    return results


if __name__ == "__main__":

    # test the function with synthetic data
    import matplotlib.pyplot as plt

    # Create synthetic data for asym. tilted Gaussian
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    x, y = np.meshgrid(x, y)
    params = {
        'amplitude': 1,
        'xo': 30,
        'yo': 70,
        'theta': np.radians(30),
        'sigma_x': 15,
        'sigma_y': 5,
        'offset': 10
    }
    a = (np.cos(params['theta']) ** 2) / (2 * params['sigma_x'] ** 2) + (np.sin(params['theta']) ** 2) / (2 * params['sigma_y'] ** 2)
    b = -(np.sin(2 * params['theta'])) / (4 * params['sigma_x'] ** 2) + (np.sin(2 * params['theta'])) / (4 * params['sigma_y'] ** 2)
    c = (np.sin(params['theta']) ** 2) / (2 * params['sigma_x'] ** 2) + (np.cos(params['theta']) ** 2) / (2 * params['sigma_y'] ** 2)
    g = params['offset'] + params['amplitude'] * np.exp(
        -(a * ((x - params['xo']) ** 2) + 2 * b * (x - params['xo']) * (y - params['yo']) + c * ((y - params['yo']) ** 2))
    )

    data = g + 0.04 * np.random.normal(size=g.shape)

    # Fit the data
    results = fit_2d_gaussian_asym(data, return_fit_data=True)



    print("Fitted parameters:")
    for key, value in results.items():
        if key != 'fit_data' and key != 'covariance':
            print(f"{key}: {value:.4f}")

    # Plot original data and fit
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Noisy Data")
    plt.imshow(data, cmap='gist_ncar', origin='lower')
    plt.contour(results['fit_data'], colors='w')

    plt.subplot(1, 2, 2)
    plt.title("Original Data")
    plt.imshow(g, cmap='gist_ncar', origin='lower')
    plt.contour(results['fit_data'], colors='w')

    plt.tight_layout()
    plt.show()