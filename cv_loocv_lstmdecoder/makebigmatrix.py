import pickle
from pathlib import Path
from scipy.ndimage import gaussian_filter1d  # Import Gaussian filter for smoothing

import pandas as pd
import tensorflow as tf
import neo
import numpy as np
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from numba import njit, prange
# import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
from keras import backend as K
from viziphant.rasterplot import rasterplot

from sklearn.utils import resample
import astropy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from datetime import datetime
from astropy.stats import bootstrap
import sklearn
from instruments.helpers.util import simple_xy_axes, set_font_axes
from instruments.helpers.neural_analysis_helpers import get_word_aligned_raster_squinty, split_cluster_base_on_segment_zola

from helpers.neural_analysis_helpers_zolainter import get_word_aligned_raster, get_word_aligned_raster_ore
from instruments.helpers.euclidean_classification_minimal_function import classify_sweeps
# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle

def generate_matrix_image(dir, trained = True):
    if trained == True:
        animal_list = ['F1702_Zola', 'F1815_Cruella', 'F1604_Squinty', 'F1606_Windolene']
    else:
        animal_list = ['F2003_Orecchiette', 'F1812_Nala', 'F1901_Crumble', 'F1902_Eclair']
    big_matrix_list = []
    for animal in animal_list:
        print(animal)
        pkl_path = Path(f'E:/rastersms4spikesortinginter/{animal}/npyfiles_dict_highperforming')
        #load the pkl file
        with open(pkl_path / 'spiketraindict_means.pkl', 'rb') as f:
            all_mean_units_for_animal = pickle.load(f)
        #now sort units with instruments in the ID
        #get the units with instruments in the ID
        units_with_targ_in_ID = []
        units_with_distractors_in_ID = []
        for individual_dict in all_mean_units_for_animal:
            unit_ID_dict_dist ={}
            units_with_instruments_in_ID = {}
            #predefine the keys
            for unit in individual_dict:
                unit_ID_number = unit.split('_')[0]
                unit_ID_dict_dist[unit_ID_number] = []
                units_with_instruments_in_ID[unit_ID_number] = []


            for unit in individual_dict:
                if 'instrument' in unit:
                    unit_ID_number = unit.split('_')[0]

                    units_with_instruments_in_ID[unit_ID_number].append(individual_dict[unit])
                else:
                    unit_ID_number = unit.split('_')[0]
                    unit_ID_dict_dist[unit_ID_number].append(individual_dict[unit])
            #now take the average of the units with the same ID
            for unit_id in unit_ID_dict_dist:
                unit_ID_mean = np.mean(unit_ID_dict_dist[unit_id], axis=0)
                unit_ID_target = np.mean(units_with_instruments_in_ID[unit_id], axis=0)
                try:
                    length = len(unit_ID_mean)
                    length_target = len(unit_ID_target)
                    units_with_distractors_in_ID.append(unit_ID_mean)
                    units_with_targ_in_ID.append(unit_ID_target)

                except Exception as e:
                    print(e)
                    continue


        #now make a big matrix of all the units, subtract the mean of the units with instruments in the ID
        #and then plot the matrix
        #make a big matrix of all the units, first make the list into a numpy array
        big_matrix_inst = np.array(units_with_targ_in_ID)
        #remove any nans from big_matrix_dist
        big_matrix_dist = np.array(units_with_distractors_in_ID)

        big_matrix_animal =big_matrix_inst - big_matrix_dist
        #now append the big matrix to the list of big matrices
        big_matrix_list.append(big_matrix_animal)
    #now plot the big matrix
    #first concatenate the big matrices
    #remove all empty arrays
    big_matrix_list = [x for x in big_matrix_list if x.size != 0]
    big_matrix = np.concatenate(big_matrix_list, axis=0)
    #sort by the mean of the first 10 timepoints
    big_matrix = big_matrix[np.argsort(np.mean(big_matrix[:, :], axis=1))]


    #now plot the big matrix
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the figsize to increase width
    im = ax.imshow(big_matrix, cmap='viridis', aspect='auto')  # Set aspect='auto' to adjust aspect ratio
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    # if trained == False:
    #     ax.set_yticks([0, 5, 10, 15])
    ax.set_xticklabels([0,0.1,0.2,0.3,0.4,0.5], fontsize = 15)
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel('Unit number', fontsize = 20)
    plt.yticks(fontsize=15)
    im.set_clim(-50, 100)

    if trained == True:
        ax.set_title('Mean target - mean distractor firing rates, trained', fontsize=20)
    else:
        ax.set_title('Mean target - mean distractor firing rates, naive', fontsize=20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=15)  # Adjust the font size (change 12 to your desired size)

    fig.savefig(f'G:/neural_chapter/figures/big_matrix_highperforming_units_trained_{trained}_14052024_onlythesiswords.png')
    plt.show()



            # target_vs_probe_with_raster(new_blocks,clust_ids = clust_ids, talker=talker, stream = stream, phydir=repeating_substring, animal = animal, brain_area = brain_area, gen_psth=True)




def main():

    directories = ['zola_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        generate_matrix_image(dir, trained = False)


if __name__ == '__main__':
    main()
