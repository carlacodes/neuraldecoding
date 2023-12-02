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

def generate_matrix_image(dir):

    big_matrix_list = []
    for animal in ['F1702_Zola', 'F1815_Cruella', 'F1604_Squinty', 'F1606_Windolene']:
        print(animal)
        pkl_path = Path(f'E:/rastersms4spikesortinginter/{animal}/npyfiles_dict')
        #load the pkl file
        with open(pkl_path / 'spiketraindict_means.pkl', 'rb') as f:
            all_mean_units_for_animal = pickle.load(f)
        #now sort units with instruments in the ID
        #get the units with instruments in the ID
        units_with_instruments_in_ID = []
        units_with_distractors_in_ID = []
        for individual_dict in all_mean_units_for_animal:
            unit_ID_dict = {}
            for unit in individual_dict:
                if 'instrument' in unit:
                    units_with_instruments_in_ID.append(individual_dict[unit])
                else:
                    unit_ID_number = unit.split('_')[0]
                    unit_ID_dict[unit_ID_number] = individual_dict[unit]
            #now take the average of the units with the same ID
            for unit_ID in unit_ID_dict:
                unit_ID_dict[unit_ID] = np.mean(unit_ID_dict[unit_ID], axis=0)
                units_with_distractors_in_ID.append(unit_ID_dict)
        #now make a big matrix of all the units, subtract the mean of the units with instruments in the ID
        #and then plot the matrix
        #make a big matrix of all the units
        big_matrix_dist = np.concatenate(units_with_distractors_in_ID, axis=0)
        big_matrix_inst = np.concatenate(units_with_instruments_in_ID, axis=0)

        big_matrix_animal = big_matrix_inst - big_matrix_dist
        #now append the big matrix to the list of big matrices
        big_matrix_list.append(big_matrix_animal)
    #now plot the big matrix
    #first concatenate the big matrices
    big_matrix = np.concatenate(big_matrix_list, axis=0)
    #now plot the big matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(big_matrix, cmap='viridis')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Units')
    ax.set_title('Big matrix of all units')
    fig.savefig('G:/neural_chapter/figures/big_matrix_all_units.png')




            # target_vs_probe_with_raster(new_blocks,clust_ids = clust_ids, talker=talker, stream = stream, phydir=repeating_substring, animal = animal, brain_area = brain_area, gen_psth=True)




def main():

    directories = ['zola_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        generate_matrix_image(dir)


if __name__ == '__main__':
    main()
