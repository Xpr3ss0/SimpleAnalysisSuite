import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# various tools for selecting ROIs in images

# interactive ROI selection using matplotlib
def select_roi_interactive(image, cmap='gray'):
    """
    Select a rectangular ROI interactively from the given image.
    The user selects the area using the default zoom tools, then presses Enter to confirm.

    Args:
        image (2D array): The input image from which to select the ROI.
    Returns:
        dict: A dictionary with keys 'x_start', 'x_end', 'y_start', 'y_end' defining the ROI.
    """

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title("Use zoom tool to select ROI, then press Enter")

    roi = {}

    def on_key(event):
        if event.key == 'enter':
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            roi['x_start'] = int(xlim[0])
            roi['x_end'] = int(xlim[1])
            roi['y_start'] = int(ylim[1])
            roi['y_end'] = int(ylim[0])
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    return roi

# automatic based on intensity contours
def select_roi_threshold(image, threshold=None, mode='dynamic', min_size_rel=0.1):
    """
    Automatically select a rectangular ROI based on intensity thresholding. 
    Uses contours to find connected regions above the threshold. 
    Returns the bounding box of the largest connected region.

    Has two modes:
    1) Fixed threshold: If 'mode' is 'fixed' and 'threshold' is provided, uses that threshold directly.
       Raises an error if no ROI is found.
    2) Iterative thresholding: If 'mode' is 'dynamic', threshold is taken as a starting point and decreased
       until a non-zero ROI larger than 'min_size_rel' of the image size is found.

    Args:
        image (2D array): The input image from which to select the ROI.
        threshold (float): Intensity threshold for selecting pixels. If None and mode='fixed', uses mean + std of the image. Otherwise starts from a high threshold and decreases.
        mode (string): If 'dynamic', threshold will be adjusted until a non-zero ROI is found. If 'fixed', uses the provided threshold directly. Defaults to 'dynamic'.
        min_size_rel (float): Minimum relative size (i.e. pixels) of the ROI compared to the image size. Ignored if mode='fixed'. Defaults to 10%.

    Returns:
        dict: A dictionary with keys 'x_start', 'x_end', 'y_start', 'y_end' defining the ROI.
    """

    if mode not in ['fixed', 'dynamic']:
        raise ValueError("Mode must be either 'fixed' or 'dynamic'.")

    if threshold is None:
        if mode == 'fixed':
            # use threshold that has a good chance of finding something
            threshold = np.mean(image) + np.std(image)
        elif mode == 'dynamic':
            # start from a high threshold and decrease to hit min_size_rel
            threshold = np.max(image) - (np.max(image) - np.min(image)) * 0.01

    # Initial pixel selection
    coords = np.argwhere(np.zeros_like(image))

    threshold_step = (np.max(image) - np.min(image)) * 0.001
    new_threshold = threshold

    while coords.size < image.size * min_size_rel:

        # create binary mask
        mask = image >= new_threshold
        # find contours
        contours, hierarchy = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # find largest contour
        max_contour = max(contours, key=cv.contourArea) if contours else None
        # get coordinates of pixels in largest contour
        if max_contour is not None:
            contour_mask = np.zeros_like(image, dtype=np.uint8)
            cv.drawContours(contour_mask, [max_contour], -1, color=1, thickness=-1) # fill the contour
            coords = np.argwhere(contour_mask)
        
        # increase the threshold iteratively (will be ignored if already sufficient)
        new_threshold -= threshold_step

        if max_contour is None and mode == 'fixed':
            raise ValueError("No ROI found with the given threshold. Consider lowering the threshold or enabling iteration.")
        elif mode == 'fixed':
            break

    final_threshold = new_threshold + threshold_step  # last valid threshold

    # Get bounding box of the selected pixels
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    if final_threshold != threshold:
        print(f"Adjusted threshold from {threshold:.2f} to {final_threshold:.2f} to find ROI.") # add back last step since loop exits after increment

    roi = {
        'x_start': x_min,
        'x_end': x_max,
        'y_start': y_min,
        'y_end': y_max,
        'threshold': new_threshold
    }

    return roi

if __name__ == "__main__":
    # test the interactive ROI selection
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)
    Z = 5*np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
    noise = np.random.rand(200, 200)
    test_image = Z + 0.1 * noise
    roi = select_roi_threshold(test_image)
    print("Selected ROI:", roi)
    plt.imshow(test_image, cmap='gray')
    plt.gca().add_patch(plt.Rectangle(
        (roi['x_start'], roi['y_start']),
        roi['x_end'] - roi['x_start'],
        roi['y_end'] - roi['y_start'],
        edgecolor='red', facecolor='none', lw=2
    ))
    plt.title("Selected ROI")
    plt.show()
    plt.figure()