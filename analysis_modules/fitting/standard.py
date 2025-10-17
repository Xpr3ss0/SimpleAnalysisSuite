import numpy as np
from scipy.optimize import curve_fit

def fit_2d_gaussian_sym(data, return_fit_data=False, initial_guess=None):
    """
    Fit a 2D symmetric Gaussian to the input data.

    Parameters:
    data (2D array): The input data to fit.

    Returns:
    popt (tuple): Optimal values for the parameters of the Gaussian. The parameters are:
        - amplitude: Peak height of the Gaussian.
        - xo: x-coordinate of the center (in pixels).
        - yo: y-coordinate of the center (in pixels).
        - sigma: Standard deviation (width) of the Gaussian (in pixels).
        - offset: Constant offset.
    pcov (2D array): The estimated covariance of popt.
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


if __name__ == "__main__":

    # test the function with synthetic data
    import matplotlib.pyplot as plt

    # Create synthetic data
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    x, y = np.meshgrid(x, y)
    data = 3 * np.exp(-(((x - 20) ** 2 + (y - 70) ** 2) / (2 * 5 ** 2))) + 10
    data += 0.01 * np.random.normal(size=data.shape)

    # Fit the data
    results = fit_2d_gaussian_sym(data, return_fit_data=True, initial_guess=(4, 25, 65, 4, 9))



    print("Fitted parameters:")
    for key, value in results.items():
        if key != 'fit_data' and key != 'covariance':
            print(f"{key}: {value}")

    # Plot original data and fit
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Data")
    plt.imshow(data, cmap='gist_ncar', origin='lower')
    plt.contour(results['fit_data'], colors='w')

    plt.tight_layout()
    plt.show()