import numpy as np
import matplotlib.pyplot as plt

# various tools for selecting ROIs in images

# interactive ROI selection using matplotlib
def select_roi_interactive(image):
    """
    Select a rectangular ROI interactively from the given image.
    The user selects the area using the default zoom tools, then presses Enter to confirm.
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
            roi['y_start'] = int(ylim[0])
            roi['y_end'] = int(ylim[1])
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    return roi


if __name__ == "__main__":
    # test the interactive ROI selection
    test_image = np.random.rand(200, 200)
    roi = select_roi_interactive(test_image)
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