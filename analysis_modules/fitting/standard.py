import numpy as np
from scipy.optimize import curve_fit

def fit_2d_gaussian_sym(data):
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
    initial_guess = (
        np.max(data),  # amplitude
        data.shape[1] / 2,  # xo
        data.shape[0] / 2,  # yo
        1.0,  # sigma
        np.min(data)  # offset
    )

    # Fit the data
    popt, pcov = curve_fit(
        gaussian_2d_sym,
        (x_indices, y_indices),
        data.ravel(),
        p0=initial_guess
    )

    return popt, pcov


if __name__ == "__main__":

    # test the function with synthetic data
    import matplotlib.pyplot as plt

    # Create synthetic data
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    x, y = np.meshgrid(x, y)
    data = 3 * np.exp(-(((x - 50) ** 2 + (y - 50) ** 2) / (2 * 5 ** 2))) + 10
    data += 0.2 * np.random.normal(size=data.shape)

    # Fit the data
    popt, pcov = fit_2d_gaussian_sym(data)

    # compute area under the Gaussian
    amplitude, xo, yo, sigma, offset = popt
    area = 2 * np.pi * amplitude * sigma**2

    # Print the results
    print("Optimal parameters:", popt)
    print("Estimated area under the Gaussian:", area)

    # Plot the results
    plt.imshow(data, extent=(0, 100, 0, 100), origin='lower')
    plt.colorbar()
    plt.title("Synthetic Data")

    # plot contours of the fitted Gaussian
    x_indices, y_indices = np.indices(data.shape)
    fitted_data = popt[4] + popt[0] * np.exp(
        -(((x_indices - popt[1]) ** 2 + (y_indices - popt[2]) ** 2) / (2 * popt[3] ** 2))
    )
    plt.contour(x_indices, y_indices, fitted_data, colors='w')

    plt.savefig("synthetic_data.png")
    print("Synthetic data plot saved as 'synthetic_data.png'")