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

from helpers.neural_analysis_helpers_zolainter import get_word_aligned_raster, get_word_aligned_raster_zola_cruella
from instruments.helpers.euclidean_classification_minimal_function import classify_sweeps
# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle



def find_repeating_substring(string):
    length = len(string)
    half_length = length // 2

    # Iterate through possible lengths of the repeating substring
    for i in range(1, half_length + 1):
        substring = string[:i]
        times = length // i

        # Construct the potential repeating substring
        potential_repeat = substring * times

        # Check if the constructed substring matches the original string
        if potential_repeat == string:
            return substring

    return None

def run_cleaning_of_rasters(blocks, datapath):
    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']
    for cluster_id in clust_ids:
        new_blocks = split_cluster_base_on_segment_zola(blocks, cluster_id, num_clusters=2)
    with open(datapath / 'new_blocks.pkl', 'wb') as f:
        pickle.dump(new_blocks, f)
    return new_blocks




def target_vs_probe_with_raster(datapaths, talker =1, animal='F1702_Zola'):
    #load the blocks
    blocks = {}
    for datapath in datapaths:
        with open(datapath / 'blocks.pkl', 'rb') as f:
            dir = str(datapath).split('\\')[-3]
            blocks[dir] = pickle.load(f)

    # clust_id_dict = {}
    # clust_id_dict['BB2BB3_zola_intertrialroving_26092023_BB2BB3_zola_intertrialroving_26092023_BB_2'] = [0,2,4, 6]
    # clust_id_dict['BB2BB3_zola_intertrialroving_26092023_BB2BB3_zola_intertrialroving_26092023_BB_3'] = [0,304]

    #make a dataframe from clust_id_dict
    # df = pd.DataFrame.from_dict(clust_id_dict, orient='index')

    #read the dataframe
    ids_to_plot = pd.read_excel('G:\plotting_csvs\F1702_zola_overlaid_IDS.csv')


    tarDir = Path(f'E:/rastersms4spikesortinginter/{animal}/figs_overlaid/{dir}/')

    saveDir = tarDir
    saveDir.mkdir(exist_ok=True, parents=True)

    binsize = 0.01
    window = [0, 0.6]
    probewords_list = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]

    animal_id_num = animal.split('_')[0]
    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']

    spiketraindict = {}

    for j, cluster_id in enumerate(ids_to_plot['ID']):
        #make a figure of 2 columns and 10 rows
        count = 0

        blocks_cluster = blocks[ids_to_plot['Folder'][j]]

        probeword = ids_to_plot['Probe_index'][j]
        pitchshift_option = ids_to_plot['Pitchshift'][j]
        if pitchshift_option == 'TRUE':
            pitchshift_option = True
        else:
            pitchshift_option = False



        raster_target, raster_target_compare = get_word_aligned_raster_zola_cruella(blocks_cluster, cluster_id, word=probeword,
                                                                                  pitchshift=pitchshift_option,
                                                                                  correctresp=True,
                                                                                  df_filter=['No Level Cue'], talker = 'female')
        raster_target = raster_target.reshape(raster_target.shape[0], )
        if len(raster_target) == 0:
            print('raster target empty:', cluster_id)
            continue

        bins = np.arange(window[0], window[1], binsize)

        unique_trials_targ = np.unique(raster_target['trial_num'])
        raster_targ_reshaped = np.empty([len(unique_trials_targ), len(bins) - 1])
        count = 0
        for trial in (unique_trials_targ):
            raster_targ_reshaped[count, :] = \
            np.histogram(raster_target['spike_time'][raster_target['trial_num'] == trial], bins=bins,
                         range=(window[0], window[1]))[0]
            count += 1

        spiketrains = []
        for trial_id in unique_trials_targ:
            selected_trials = raster_target[raster_target['trial_num'] == trial_id]
            spiketrain = neo.SpikeTrain(selected_trials['spike_time'], units='s', t_start=min(selected_trials['spike_time']), t_stop=max(selected_trials['spike_time']))
            spiketrains.append(spiketrain)


        spiketraindict[cluster_id] = spiketrains









    return



def generate_rasters(dir):

    datapath_big = Path(f'D:/ms4output_16102023/F1702_Zola/')
    animal = str(datapath_big).split('\\')[-1]

    datapaths = [Path('D:\ms4output_16102023\F1702_Zola\BB2BB3_zola_intertrialroving_26092023\BB2BB3_zola_intertrialroving_26092023_BB2BB3_zola_intertrialroving_26092023_BB_2\mountainsort4\phy'),
                      Path('D:\ms4output_16102023\F1702_Zola\BB2BB3_zola_intertrialroving_26092023\BB2BB3_zola_intertrialroving_26092023_BB2BB3_zola_intertrialroving_26092023_BB_3\mountainsort4\phy')]

    for talker in [1]:
        target_vs_probe_with_raster(datapaths,talker=talker, animal = animal)
        # target_vs_probe_with_raster(new_blocks,clust_ids = clust_ids, talker=talker, stream = stream, phydir=repeating_substring, animal = animal, brain_area = brain_area, gen_psth=True)




def main():

    directories = ['zola_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        generate_rasters(dir)


if __name__ == '__main__':
    main()
