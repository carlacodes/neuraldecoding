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
def target_vs_probe_with_raster(blocks, talker=1,  clust_ids = [], stream = 'BB_3', phydir = 'phy', animal = 'F1702_Zola', brain_area = [], gen_psth = False, csv_info = []):
    tarDir = Path(f'E:/rastersms4spikesortinginter/{animal}/figs_nothreshold_ANDPSTH_12062024/{phydir}/{stream}/')
    #load the high generalizable clusters, csv file

    saveDir = tarDir
    saveDir.mkdir(exist_ok=True, parents=True)

    binsize = 0.01
    window = [0, 0.6]
    probewords_list = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
    probewords_list = [(1,1), (5,6), (20,22), (2,2), (42, 49), (32,38)]
    probewords_list = [ (1,1), (2, 2), (20, 22), (5, 6), (42, 49), (32, 38), (56, 56), (57,57)]

    animal_id_num = animal.split('_')[0]
    #cast clust_ids as int
    clust_ids = [int(x) for x in clust_ids]
    #only get the unique clust_ids
    clust_ids = list(set(clust_ids))
    for j, cluster_id in enumerate(clust_ids):
        #make a figure of 2 columns and 10 rows
        cluster_info = csv_info[csv_info['ID_small'] == cluster_id]
        #convert cluster_info probeword to int
        cluster_info['ProbeWord'] = cluster_info['ProbeWord'].astype(int)
        fig, ax = plt.subplots(len(probewords_list), 2, figsize=(10, 30))
        count = 0
        mean_score_for_cluster = cluster_info['MeanScore'].values[0]
        #round to the nearest 2 decimal places
        mean_score_for_cluster = round(mean_score_for_cluster, 2)
        mean_perm_score_for_cluster = cluster_info['score_permutation'].values[:]
        #round to the nearest 2 decimal places
        #convert to numpy array
        #take the mean
        mean_perm_score_for_cluster = np.mean(mean_perm_score_for_cluster)
        mean_perm_score_for_cluster = round(mean_perm_score_for_cluster, 2)

        # mean_perm_score_for_cluster = np.array(mean_perm_score_for_cluster)

        for idx, probewords in enumerate(probewords_list):
            for pitchshift_option in [0,1]:
                #get score for that probeword
                probeword = probewords[0]
                individual_info = cluster_info[(cluster_info['ProbeWord'] == probeword) & (cluster_info['PitchShift'] == pitchshift_option)]
                #if individual_info is empty, skip
                if individual_info.empty:
                    print('empty individual info:', cluster_id, probeword, pitchshift_option)
                    individual_score = None
                    individual_perm_score = None
                else:
                    individual_score = individual_info['Score'].values[0]
                    individual_perm_score = individual_info['score_permutation'].values[0]
                    individual_score = round(individual_score, 2)
                    individual_perm_score = round(individual_perm_score, 2)


                raster_target, raster_target_compare = get_word_aligned_raster_zola_cruella(blocks, cluster_id, word=probewords[0],
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

                try:
                    if probewords[0] == 2 and pitchshift_option == False:
                        probeword_text = 'when a'
                        color_option = 'green'
                    elif probewords[0] == 2 and pitchshift_option == True:
                        probeword_text = 'when a'
                        color_option = 'lightgreen'

                    elif probewords[0] == 1 and pitchshift_option == False:
                        probeword_text = 'instruments'
                        color_option = 'blue'
                    elif probewords[0] == 1 and pitchshift_option == True:
                        probeword_text = 'instruments'
                        color_option = 'skyblue'


                    elif probewords[0] == 5 and pitchshift_option == False:
                        probeword_text = 'craft'
                        color_option = 'deeppink'
                    elif probewords[0] == 5 and pitchshift_option == True:
                        probeword_text = 'craft'
                        color_option = 'pink'

                    elif probewords[0] == 20 and pitchshift_option == False:
                        probeword_text = 'in contrast'
                        color_option = 'mediumpurple'
                    elif probewords[0] == 20 and pitchshift_option == True:
                        probeword_text = 'in contrast'
                        color_option = 'purple'

                    elif probewords[0] == 42 and pitchshift_option == False:
                        probeword_text = 'accurate'
                        color_option = 'black'

                    elif probewords[0] == 42 and pitchshift_option == True:
                        probeword_text = 'accurate'
                        color_option = 'grey'
                    elif probewords[0] == 56 and pitchshift_option == False:
                        probeword_text = 'pink noise'
                        color_option = 'navy'
                    elif probewords[0] == 56 and pitchshift_option == True:
                        probeword_text = 'pink noise'
                        color_option = 'lightblue'

                    elif probewords[0] == 32 and pitchshift_option == False:
                        probeword_text = 'of science'
                        color_option = 'coral'
                    elif probewords[0] == 32 and pitchshift_option == True:
                        probeword_text = 'of science'
                        color_option = 'orange'


                    elif probewords[0] == 57 and pitchshift_option == False:
                        probeword_text = 'rev. instruments'
                        color_option = 'plum'
                    elif probewords[0] == 57 and pitchshift_option == True:
                        probeword_text = 'rev. instruments'
                        color_option = 'darkorchid'
                    elif probewords[0] == 33 and pitchshift_option == False:
                        probeword_text = 'boats'
                        color_option = 'slategrey'
                    elif probewords[0] == 33 and pitchshift_option == True:
                        probeword_text = 'boats'
                        color_option = 'royalblue'

                    elif probewords[0] == 11 and pitchshift_option == False:
                        probeword_text = 'today'
                        color_option = 'gold'
                    elif probewords[0] == 11 and pitchshift_option == True:
                        probeword_text = 'today'
                        color_option = 'yellow'
                    else:
                        probeword_text = 'error'
                        color_option = 'red'
                    #if pitchshift plot on the second column
                    custom_xlim = (-0.1, 0.6)

                    if pitchshift_option:
                        if gen_psth:
                            bin_width = 0.01  # Width of the time bins in seconds
                            time_start = -0.1  # Start time for the PSTH (in seconds)
                            time_end = 0.6  # End time for the PSTH (in seconds)
                            stimulus_onset = 0.0  # Time of the stimulus onset (relative to the PSTH window)

                            # Calculate PSTH within the specified time range
                            num_bins = int((time_end - time_start) / bin_width) + 1
                            bins = np.linspace(time_start, time_end, num_bins + 1)
                            spike_times = [st.times.magnitude for st in spiketrains]

                            # Flatten spike times and filter within the specified time range
                            spike_times_flat = np.concatenate(spike_times)
                            spike_times_filtered = spike_times_flat[
                                (spike_times_flat >= time_start) & (spike_times_flat <= time_end)]

                            # Compute the histogram within the specified time range
                            hist, _ = np.histogram(spike_times_filtered, bins=bins)

                            # Calculate time axis for plotting within the specified time range
                            time_axis = np.linspace(time_start, time_end, num_bins) + bin_width / 2

                            # Apply smoothing using Gaussian filter
                            sigma = 2  # Smoothing parameter (adjust as needed)
                            smoothed_hist = gaussian_filter1d(hist / (bin_width * len(spiketrains)), sigma=sigma)

                            # Plot smoothed PSTH within the specified time range
                            ax[idx, 1].plot(time_axis, smoothed_hist, color=color_option, linewidth=2)

                            ax[idx, 1].set_ylabel('spikes/s')
                        else:
                            rasterplot(spiketrains, c=color_option, histogram_bins=0, axes=ax[idx, 1], s=0.3)
                            ax[idx, 1].set_ylabel('trial number')
                            ax[idx, 1].set_xlim(custom_xlim)

                        ax[idx, 1].set_title(f'Unit: {cluster_id}_{phydir}, \n {animal_id_num}, score: {individual_score}, perm score: {individual_perm_score}')
                        ax[idx, 1].text(-0.2, 0.5, probeword_text, horizontalalignment='center',
                                        verticalalignment='center', rotation=90, transform=ax[idx, 1].transAxes)
                    else:
                        if gen_psth:
                            # get the array of spiketimes
                            bin_width = 0.01  # Width of the time bins in seconds
                            time_start = -0.1  # Start time for the PSTH (in seconds)
                            time_end = 0.6  # End time for the PSTH (in seconds)
                            stimulus_onset = 0.0  # Time of the stimulus onset (relative to the PSTH window)

                            # Calculate PSTH within the specified time range
                            num_bins = int((time_end - time_start) / bin_width) + 1
                            bins = np.linspace(time_start, time_end, num_bins + 1)
                            spike_times = [st.times.magnitude for st in spiketrains]

                            # Flatten spike times and filter within the specified time range
                            spike_times_flat = np.concatenate(spike_times)
                            spike_times_filtered = spike_times_flat[
                                (spike_times_flat >= time_start) & (spike_times_flat <= time_end)]

                            # Compute the histogram within the specified time range
                            hist, _ = np.histogram(spike_times_filtered, bins=bins)

                            # Calculate time axis for plotting within the specified time range
                            time_axis = np.linspace(time_start, time_end, num_bins) + bin_width / 2

                            # Apply smoothing using Gaussian filter
                            sigma = 2  # Smoothing parameter (adjust as needed)
                            smoothed_hist = gaussian_filter1d(hist / (bin_width * len(spiketrains)), sigma=sigma)

                            # Plot smoothed PSTH within the specified time range
                            ax[idx, 0].plot(time_axis, smoothed_hist, color=color_option, linewidth=2)

                            ax[idx, 0].set_ylabel('spikes/s')

                        else:
                            rasterplot(spiketrains, c=color_option, histogram_bins=0, axes=ax[idx, 0], s=0.3)
                            ax[idx, 0].set_ylabel('trial number')

                        ax[idx, 0].set_xlim(custom_xlim)

                        ax[idx, 0].set_title(f'Unit: {cluster_id}_{phydir}, \n {animal_id_num}, score: {individual_score}, perm score: {individual_perm_score}')

                        ax[idx, 0].text(-0.2, 0.5, probeword_text, horizontalalignment='center',
                                        verticalalignment='center', rotation=90, transform=ax[idx, 0].transAxes)
                except Exception:
                    continue


        # ax[0, 1].set_title('Pitch-shifted F0')
        # ax[0, 0].set_title('Control F0')
        plt.subplots_adjust(wspace=0.3, hspace=1.0)

        if gen_psth:
            plt.suptitle(f'PSTHs for {animal}, unit id: {cluster_id}, stream: {stream}, mean score: {mean_score_for_cluster}, mean permutation score: {mean_perm_score_for_cluster}', fontsize=15)

            plt.savefig(
                str(saveDir) + f'/PSTH_targdist_grid_clusterid_{cluster_id}_{stream}_' + str(
                    cluster_id) + '.png', bbox_inches='tight')
            plt.savefig(
                str(saveDir) + f'/PSTH_targdist_grid_clusterid_{cluster_id}_{stream}_' + str(
                    cluster_id) + '.svg', bbox_inches='tight')
        else:
            plt.suptitle(f'Rasters for {animal}, unit id: {cluster_id}, stream: {stream}, mean score: {mean_score_for_cluster}, mean permutation score: {mean_perm_score_for_cluster}', fontsize=15)

            plt.savefig(
                str(saveDir) + f'/targdist_grid_clusterid_{cluster_id}_{stream}_' + str(
                    cluster_id) + '.png', bbox_inches='tight')
            plt.savefig(
                str(saveDir) + f'/targdist_grid_clusterid_{cluster_id}_{stream}_' + str(
                    cluster_id) + '.svg', bbox_inches='tight')
                # plt.show()
        plt.close('all')



    return



def generate_rasters(dir):
    datapath_big = Path(f'D:/ms4output_16102023/F1702_Zola/')
    animal = str(datapath_big).split('\\')[-1]
    datapaths = [x for x in datapath_big.glob('**/mountainsort4/phy//') if x.is_dir()]
    high_units = pd.read_csv(f'G:/neural_chapter/csvs/units_trained_highscore.csv')

    for datapath in datapaths:
        stream = str(datapath).split('\\')[-3]
        stream = stream[-4:]
        print(stream)
        folder = str(datapath).split('\\')[-3]
        with open(datapath / 'new_blocks.pkl', 'rb') as f:
            new_blocks = pickle.load(f)


        #filter for units that have an ID with the animal in it
        high_units_animal = high_units[high_units['animal'].str.contains(animal)]
        # remove trailing steam
        rec_name = folder[:-5]
        #find the unique string
        repeating_substring = find_repeating_substring(rec_name)


        #remove the repeating substring

        # find the units that have the phydir

        max_length = len(rec_name) // 2

        for length in range(1, max_length + 1):
            for i in range(len(rec_name) - length):
                substring = rec_name[i:i + length]
                if rec_name.count(substring) > 1:
                    repeating_substring = substring
                    break

        print(repeating_substring)
        rec_name = repeating_substring
        high_units_animal = high_units_animal[(high_units_animal['recname'] == rec_name) & (high_units_animal['stream'] == stream)]
        clust_ids = high_units_animal['ID_small'].to_list()
        brain_area = high_units_animal['BrainArea'].to_list()

        if clust_ids == []:
            print('no units found')
            continue
        for talker in [1]:
            target_vs_probe_with_raster(new_blocks,clust_ids = clust_ids, talker=talker, stream = stream, phydir=repeating_substring, animal = animal, brain_area = brain_area, csv_info =high_units_animal)
            target_vs_probe_with_raster(new_blocks,clust_ids = clust_ids, talker=talker, stream = stream, phydir=repeating_substring, animal = animal, brain_area = brain_area, gen_psth=True, csv_info=high_units_animal)




def main():

    directories = ['zola_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        generate_rasters(dir)


if __name__ == '__main__':
    main()
