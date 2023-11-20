import pickle
from pathlib import Path

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
from instruments.helpers.neural_analysis_helpers import get_soundonset_alignedraster, split_cluster_base_on_segment_zola

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
def target_vs_probe_with_raster(blocks, talker=1,  stream = 'BB_3', phydir = 'phy', animal = 'F1702_Zola', brain_area = [], gen_psth = False):

    tarDir = Path(f'E:/rastersms4spikesortinginter/{animal}/figs_nothreshold_ANDPSTH_2011/{phydir}/{stream}/')
    #load the high generalizable clusters, csv file

    saveDir = tarDir
    saveDir.mkdir(exist_ok=True, parents=True)

    binsize = 0.01
    window = [0, 0.6]
    probewords_list = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']

    for j, cluster_id in enumerate(clust_ids):
        #make a figure of 2 columns and 10 rows
        fig, ax = plt.subplots(len(probewords_list), 2, figsize=(10, 30))
        count = 0
        for idx, probewords in enumerate(probewords_list):
            for pitchshift_option in [True, False]:

                raster_target, raster_target_compare = get_word_aligned_raster_zola_cruella(blocks, cluster_id, word=probewords[0],
                                                                                          pitchshift=pitchshift_option,
                                                                                          correctresp=False,
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

                try:
                    if probewords[0] == 4 and pitchshift_option == False:
                        probeword_text = 'when a'
                        color_option = 'green'
                    elif probewords[0] == 4 and pitchshift_option == True:
                        probeword_text = 'when a'
                        color_option = 'lightgreen'

                    elif probewords[0] == 1 and pitchshift_option == False:
                        probeword_text = 'instruments'
                        color_option = 'blue'
                    elif probewords[0] == 1 and pitchshift_option == True:
                        probeword_text = 'instruments'
                        color_option = 'skyblue'


                    elif probewords[0] == 2 and pitchshift_option == False:
                        probeword_text = 'craft'
                        color_option = 'deeppink'
                    elif probewords[0] == 2 and pitchshift_option == True:
                        probeword_text = 'craft'
                        color_option = 'pink'

                    elif probewords[0] == 3 and pitchshift_option == False:
                        probeword_text = 'in contrast'
                        color_option = 'mediumpurple'
                    elif probewords[0] == 3 and pitchshift_option == True:
                        probeword_text = 'in contrast'
                        color_option = 'purple'

                    elif probewords[0] == 5 and pitchshift_option == False:
                        probeword_text = 'accurate'
                        color_option = 'black'

                    elif probewords[0] == 5 and pitchshift_option == True:
                        probeword_text = 'accurate'
                        color_option = 'grey'
                    elif probewords[0] == 6 and pitchshift_option == False:
                        probeword_text = 'pink noise'
                        color_option = 'navy'
                    elif probewords[0] == 6 and pitchshift_option == True:
                        probeword_text = 'pink noise'
                        color_option = 'lightblue'

                    elif probewords[0] == 7 and pitchshift_option == False:
                        probeword_text = 'of science'
                        color_option = 'coral'
                    elif probewords[0] == 7 and pitchshift_option == True:
                        probeword_text = 'of science'
                        color_option = 'orange'


                    elif probewords[0] == 8 and pitchshift_option == False:
                        probeword_text = 'rev. instruments'
                        color_option = 'plum'
                    elif probewords[0] == 8 and pitchshift_option == True:
                        probeword_text = 'rev. instruments'
                        color_option = 'darkorchid'
                    elif probewords[0] == 9 and pitchshift_option == False:
                        probeword_text = 'boats'
                        color_option = 'slategrey'
                    elif probewords[0] == 9 and pitchshift_option == True:
                        probeword_text = 'boats'
                        color_option = 'royalblue'

                    elif probewords[0] == 10 and pitchshift_option == False:
                        probeword_text = 'today'
                        color_option = 'gold'
                    elif probewords[0] == 10 and pitchshift_option == True:
                        probeword_text = 'today'
                        color_option = 'yellow'
                    else:
                        probeword_text = 'error'
                        color_option = 'red'
                    #if pitchshift plot on the second column
                    custom_xlim = (-0.1, 0.6)

                    if pitchshift_option:
                        if gen_psth:
                            psth = np.histogram(spiketrains, bins=100)
                            ax[idx, 1].plot(psth[1][:-1], psth[0], color=color_option)
                            ax[idx, 1].set_ylabel('spikes/s')
                        else:
                            rasterplot(spiketrains, c=color_option, histogram_bins=0, axes=ax[idx, 1], s=0.3)
                        ax[idx, 1].set_xlim(custom_xlim)
                        ax[idx, 1].set_title(f'Unit: {cluster_id}_{stream}, animal: {animal}')
                        ax[idx, 1].text(-0.2, 0.5, probeword_text, horizontalalignment='center',
                                        verticalalignment='center', rotation=90, transform=ax[idx, 1].transAxes)
                    else:
                        if gen_psth:
                            psth = np.histogram(spiketrains, bins=100)
                            ax[idx, 0].plot(psth[1][:-1], psth[0], color=color_option)
                            ax[idx, 1].set_ylabel('spikes/s')

                        else:
                            rasterplot(spiketrains, c=color_option, histogram_bins=0, axes=ax[idx, 0], s=0.3)
                        ax[idx, 0].set_xlim(custom_xlim)
                        ax[idx, 0].set_ylabel('trial')
                        ax[idx, 0].set_title(f'Unit: {cluster_id}_{stream}, animal: {animal}')

                        ax[idx, 0].text(-0.2, 0.5, probeword_text, horizontalalignment='center',
                                        verticalalignment='center', rotation=90, transform=ax[idx, 0].transAxes)
                except Exception as e:
                    print(f"Error: {e}")
                    continue


        # ax[0, 1].set_title('Pitch-shifted F0')
        # ax[0, 0].set_title('Control F0')
        plt.subplots_adjust(wspace=0.2, hspace=1.0)

        plt.suptitle(f'Rasters for {animal}, unit id: {cluster_id}, stream: {stream},', fontsize=15)
        plt.savefig(
            str(saveDir) + f'/targdist_grid_clusterid_{cluster_id}_{stream}_' + str(
                cluster_id) + '.png', bbox_inches='tight')
        plt.savefig(
            str(saveDir) + f'/targdist_grid_clusterid_{cluster_id}_{stream}_' + str(
                cluster_id) + '.svg', bbox_inches='tight')
                # plt.show()



    return



def generate_rasters(dir):


    datapath_big = Path(f'G:/F2003_Orecchiette/')
    animal = str(datapath_big).split('\\')[-1]
    datapaths = [x for x in datapath_big.glob('**/*kilosort//phy//') if x.is_dir()]
    datapaths = datapaths[-1]
    for datapath in [datapaths]:
        stream = str(datapath).split('\\')[-3]
        stream = stream[-4:]
        print(stream)
        folder = str(datapath).split('\\')[-4]
        with open(datapath / 'blocks.pkl', 'rb') as f:
            new_blocks = pickle.load(f)

        high_units = pd.read_csv(f'G:/neural_chapter/figures/unit_ids_trained_topgenindex_{animal}.csv')
        # remove trailing steam
        rec_name = folder[:-5]
        #find the unique string
        repeating_substring = find_repeating_substring(rec_name)


        #remove the repeating substring

        # find the units that have the phydir

        max_length = len(rec_name) // 2

        if folder.__contains__('s2') and not folder.__contains__('mod'):
            stream = 't_s2'
        elif folder.__contains__('s3'):
            stream = 't_s3'
        elif folder.__contains__('gmod'):
            stream = 'gmod'

        #print out the unique streams
        unique_streams = high_units['stream'].unique()
        high_units = high_units[(high_units['stream'] == stream)]
        clust_ids = high_units['ID'].to_list()
        brain_area = high_units['BrainArea'].to_list()


        for talker in [1]:
            target_vs_probe_with_raster(new_blocks,talker=talker, stream = stream, phydir=repeating_substring, animal = animal, brain_area = brain_area)
            target_vs_probe_with_raster(new_blocks,talker=talker, stream = stream, phydir=repeating_substring, animal = animal, brain_area = brain_area, gen_psth=True)





def main():

    directories = ['zola_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        generate_rasters(dir)


if __name__ == '__main__':
    main()
