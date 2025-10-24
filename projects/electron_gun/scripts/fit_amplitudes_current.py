from analysis_modules.fitting.standard import fit_2d_gaussian_sym
from analysis_modules.image_processing.roi_selection import select_roi_interactive
import os
from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import configparser

data_dir = "projects/electron_gun/data/17.10.2025/controlled_comparison/"

materials = ['CerYAG', 'P43', 'tube_phosphor', 'P_homemade']

config_names = ['pset1', 'pset2', 'pset3', 'pset4']

overwrite_crops = False

check_fits = False
confirm_fit = False

def on_button_press(event):
    global confirm_fit
    if event.key == 'enter':
        confirm_fit = True
        plt.close()

if __name__ == "__main__":

    # read exposure from configs
    exposure_times = {"pset1": 4.215000, "pset2": 1.590333, "pset3": 9.506667, "pset4": 3.368333}  # default values


    # crop all .bmp images in the data directory using interactive ROI selection
    for filename in os.listdir(data_dir):
        if filename.endswith('.bmp'):

            # cropping images
            if "cropped_" + filename in os.listdir(os.path.join(data_dir, "cropped/")) and not overwrite_crops:
                print(f"Skipping {filename}, cropped version already exists.")
                continue

            filepath = os.path.join(data_dir, filename)
            im = Image.open(filepath)
            img_array = np.array(im)
            sat_filter = img_array >= 255
            img_array[sat_filter] = 0 
            print(f"Selecting ROI for image: {filename}")
            roi = select_roi_interactive(img_array, cmap='gist_ncar')
            if not roi:
                print("No ROI selected, skipping this image.")
                continue

            # crop and save image
            cropped_img = img_array[roi['y_start']:roi['y_end'], roi['x_start']:roi['x_end']]
            print(cropped_img.shape)
            cropped_im = Image.fromarray(cropped_img)
            cropped_filepath = os.path.join(data_dir, f"cropped/cropped_{filename}")
            cropped_im.save(cropped_filepath)
            print(f"Cropped image saved to: {cropped_filepath}")

            print(f"Finished cropping. {filename}\n")

    # fit all cropped images and print amplitude results

    results = defaultdict(list)

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))
    for i, config_name in enumerate(config_names):
        for material in materials:
            amplitudes = []
            amp_errors = []
            currents = []
            for filename in tqdm(os.listdir(os.path.join(data_dir, "cropped/"))):
                if filename.endswith('.bmp') and material in filename and config_name in filename:
                    filepath = os.path.join(data_dir, "cropped/", filename)
                    im = Image.open(filepath)
                    img_array = np.array(im)
                    current = float(filename.split('_')[-1].replace('uA.bmp', ''))

                    fit_result = fit_2d_gaussian_sym(img_array, return_fit_data=True,
                                                     initial_guess=(
                                                        150,  # amplitude
                                                        img_array.shape[1] / 2,  # xo
                                                        img_array.shape[0] / 2,  # yo
                                                        (img_array.shape[1] + img_array.shape[0]) / 10,  # sigma
                                                        np.min(img_array)  # offset
                                                        )
                                                     )
                    if fit_result is None:
                        print(f"Fit failed for {filename}, skipping.")
                        continue

                    if fit_result['covariance'][0, 0] / (fit_result['amplitude']**2) > 1:
                        print(f"Unreliable fit for {filename}, skipping.")
                        continue
                    
                    

                    fit_data = fit_result.get('fit_data')
                    if check_fits:
                        fig_fit, ax_fit = plt.subplots()
                        fig_fit.canvas.mpl_connect('key_press_event', on_button_press)
                        confirm_fit = False
                        ax_fit.set_title(f"Fit for {filename}. Press Enter to confirm.")
                        ax_fit.imshow(img_array, cmap='gist_ncar')
                        if fit_data is not None:
                            # plot FWHM contour
                            ax_fit.contour(fit_data, levels=[fit_result['offset'] + 0.5 * fit_result['amplitude']], colors='red')
                        fig_fit.show()
                        plt.waitforbuttonpress()

                        if not confirm_fit:
                            print(f"Fit not confirmed for {filename}, skipping.")
                            plt.close(fig_fit)
                            continue

                    amplitude = fit_result['amplitude']
                    amp_error = np.sqrt(fit_result['covariance'][0, 0])
                    # print(f"Fit result for {filename}: Amplitude = {amplitude:.2f} ± {amp_error:.2f}")
                    amplitudes.append(amplitude)
                    amp_errors.append(amp_error)
                    currents.append(current)

                    results['material'].append(material)
                    results['amplitude'].append(amplitude)
                    results['amp_error'].append(amp_error)
                    results['current_uA'].append(current)
                    results['exposure'].append(exposure_times[config_name])
                    results['config'].append(config_name)
            
            # plot amplitude vs current
            axs[i // 2, i % 2].errorbar(currents, amplitudes, yerr=amp_errors, xerr=0.05, fmt='o', label=material, capsize=5)
            axs[i // 2, i % 2].set_title(f"Exposure: {exposure_times[config_name]:.4f}ms")
            axs[i // 2, i % 2].set_xlabel(r"Current ($\mathrm{\mu A}$)")
            axs[i // 2, i % 2].set_ylabel("Amplitude (a.u.)")
            axs[i // 2, i % 2].legend()

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(os.path.join(data_dir, "fit_results.csv"), index=False)
    plt.tight_layout()
    plt.show()
