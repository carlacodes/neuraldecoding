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
import sys
sys.path.append('../')
from analysisscriptsmodcg.cv_loocv_lstmdecoder.helpers.neural_analysis_helpers_zolainter import get_word_aligned_raster, get_word_aligned_raster_zola_cruella, get_word_aligned_raster_ore
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
    ids_to_plot = pd.read_csv(f'G:\plotting_csvs\{animal}_overlaid_IDS.csv')


    for datapath in datapaths:
        try:
            with open(datapath / 'blocks.pkl', 'rb') as f:
                dir = str(datapath).split('\\')[-4]
                blocks[dir] = pickle.load(f)
        except:
            with open(datapath / 'blocks.pkl', 'rb') as f:
                dir = str(datapath).split('\\')[-4]
                blocks[dir] = pickle.load(f)

    # clust_id_dict = {}
    # clust_id_dict['BB2BB3_zola_intertrialroving_26092023_BB2BB3_zola_intertrialroving_26092023_BB_2'] = [0,2,4, 6]
    # clust_id_dict['BB2BB3_zola_intertrialroving_26092023_BB2BB3_zola_intertrialroving_26092023_BB_3'] = [0,304]

    #make a dataframe from clust_id_dict
    # df = pd.DataFrame.from_dict(clust_id_dict, orient='index')

    #read the dataframe


    tarDir = Path(f'E:/rastersms4spikesortinginter/{animal}/figs_overlaid03042024/{dir}/')

    saveDir = tarDir
    saveDir.mkdir(exist_ok=True, parents=True)

    binsize = 0.01
    window = [0, 0.6]
    probewords_list = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]

    animal_id_num = animal.split('_')[0]


    spiketraindict = {}
    unique_trials_dict = {}

    for j, cluster_id in enumerate(ids_to_plot['ID']):
        #make a figure of 2 columns and 10 rows
        count = 0

        blocks_cluster = blocks[ids_to_plot['Folder'][j]]

        probeword = ids_to_plot['Probe_index'][j]
        pitchshift_option = ids_to_plot['Pitchshift'][j]




        # raster_target, raster_target_compare = get_word_aligned_raster_zola_cruella(blocks_cluster, cluster_id, word=probeword,
        #                                                                           pitchshift=pitchshift_option,
        #                                                                           correctresp=False,
        #                                                                           df_filter=['No Level Cue'], talker = 'female')

        raster_target, raster_targ_compare = get_word_aligned_raster_ore(blocks_cluster, cluster_id, word=probeword,
                                                                         pitchshift=pitchshift_option,
                                                                         correctresp=False,
                                                                         df_filter=[], talker='female')
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

        dict_key = f'{cluster_id}_{ids_to_plot["Folder"][j]}_{probeword}'
        spiketraindict[dict_key] = spiketrains
        unique_trials_dict[dict_key] = np.unique(raster_target['trial_num'])
    fig, ax = plt.subplots()
    for i, cluster_id in enumerate(ids_to_plot['ID']):
        #now plot the rasters

        bin_width = 0.01  # Width of the time bins in seconds
        time_start = -0.1  # Start time for the PSTH (in seconds)
        time_end = 0.6  # End time for the PSTH (in seconds)
        stimulus_onset = 0.0  # Time of the stimulus onset (relative to the PSTH window)
        probeword = ids_to_plot['Probe_index'][i]

        spiketrains = spiketraindict[f'{cluster_id}_{ids_to_plot["Folder"][i]}_{probeword}']
        unique_trials = unique_trials_dict[f'{cluster_id}_{ids_to_plot["Folder"][i]}_{probeword}']

        probeword = ids_to_plot['Probe_index'][i]
        pitchshift_option = ids_to_plot['Pitchshift'][i]
        pitchshift_text = 'inter-roved F0' if pitchshift_option else 'control F0'
        if probeword == 4 and pitchshift_option == False:
            probeword_text = 'when a'
            color_option = 'green'
        elif probeword == 4 and pitchshift_option == True:
            probeword_text = 'when a'
            color_option = 'lightgreen'

        elif probeword == 1 and pitchshift_option == False:
            probeword_text = 'instruments'
            color_option = 'black'
        elif probeword == 1 and pitchshift_option == True:
            probeword_text = 'instruments'
            color_option = 'black'


        elif probeword == 2 and pitchshift_option == False:
            probeword_text = 'craft'
            color_option = 'deeppink'
        elif probeword == 2 and pitchshift_option == True:
            probeword_text = 'craft'
            color_option = 'pink'

        elif probeword == 3 and pitchshift_option == False:
            probeword_text = 'in contrast'
            color_option = 'mediumpurple'
        elif probeword == 3 and pitchshift_option == True:
            probeword_text = 'in contrast'
            color_option = 'purple'

        elif probeword == 5 and pitchshift_option == False:
            probeword_text = 'accurate'
            color_option = 'olivedrab'

        elif probeword == 5 and pitchshift_option == True:
            probeword_text = 'accurate'
            color_option = 'limegreen'
        elif probeword == 6 and pitchshift_option == False:
            probeword_text = 'pink noise'
            color_option = 'navy'
        elif probeword == 6 and pitchshift_option == True:
            probeword_text = 'pink noise'
            color_option = 'lightblue'
        elif probeword == 7 and pitchshift_option == False:
            probeword_text = 'of science'
            color_option = 'coral'
        elif probeword == 7 and pitchshift_option == True:
            probeword_text = 'of science'
            color_option = 'orange'
        elif probeword == 8 and pitchshift_option == False:
            probeword_text = 'rev. instruments'
            color_option = 'plum'
        elif probeword == 8 and pitchshift_option == True:
            probeword_text = 'rev. instruments'
            color_option = 'darkorchid'
        elif probeword == 9 and pitchshift_option == False:
            probeword_text = 'boats'
            color_option = 'cornflowerblue'
        elif probeword == 9 and pitchshift_option == True:
            probeword_text = 'boats'
            color_option = 'royalblue'

        elif probeword == 10 and pitchshift_option == False:
            probeword_text = 'today'
            color_option = 'gold'
        elif probeword == 10 and pitchshift_option == True:
            probeword_text = 'today'
            color_option = 'yellow'
        else:
            probeword_text = 'error'
            color_option = 'red'
        # Calculate PSTH within the specified time range
        num_bins = int((time_end - time_start) / bin_width) + 1
        bins = np.linspace(time_start, time_end, num_bins + 1)
        spike_times = [st.times.magnitude for st in spiketrains]

        # Flatten spike times and filter within the specified time range
        spike_times_flat = np.concatenate(spike_times)
        spike_times_filtered = spike_times_flat[
            (spike_times_flat >= time_start) & (spike_times_flat <= time_end)]
        #histogram is dividded by the number of trials to get the mean count per bin and not the total count per bin as
        # otherwise the spike counts will be higher per time bin in stimuli that have more repetitions.
        # Then you can convert to spikes/s and smooth.
        # Compute the histogram within the specified time range
        hist, _ = np.histogram(spike_times_filtered, bins=bins)
        #get the number of trials
        num_trials = len(unique_trials)
        #divide the histogram by the number of trials
        hist_divided_bytrial_num = hist/num_trials
        #convert to spikes/s
        hist_rate = hist_divided_bytrial_num/bin_width


        # Calculate time axis for plotting within the specified time range
        time_axis = np.linspace(time_start, time_end, num_bins) + bin_width / 2

        # Apply smoothing using Gaussian filter
        sigma = 0.5  # Smoothing parameter (adjust as needed)
        smoothed_hist = gaussian_filter1d(hist_rate, sigma=sigma)

        # Plot smoothed PSTH within the specified time range
        if probeword_text == 'instruments':
            ax.plot(time_axis, hist_rate, color=color_option, linewidth=3, label=probeword_text)
        else:
            ax.plot(time_axis, hist_rate, color=color_option, linewidth=2, label = probeword_text)

        # rasterplot(spiketrains, c=color_option, histogram_bins=0, axes=ax2, s=0.3)
        # ax2.set_ylabel('trial number')
        # custom_xlim = (-0.1, 0.6)
        # ax2.set_xlim(custom_xlim)
        # ax2.set_title(f'{cluster_id}_{ids_to_plot["Folder"][i]},\n {animal_id_num}, probeword: {probeword_text}, {pitchshift_text}')

    ax.legend()
    ax.set_ylabel('spikes/s')
    ax.set_xlabel('time (s)')
    plt.savefig(saveDir / f'overlaid_psths_{animal_id_num}_{talker}.png')


    for i, cluster_id in enumerate(ids_to_plot['ID']):
        # now plot the rasters
        count = 0
        fig2, ax2 = plt.subplots()
        probeword = ids_to_plot['Probe_index'][i]

        spiketrains = spiketraindict[f'{cluster_id}_{ids_to_plot["Folder"][i]}_{probeword}']
        probeword = ids_to_plot['Probe_index'][i]
        pitchshift_option = ids_to_plot['Pitchshift'][i]
        pitchshift_text = 'inter-roved F0' if pitchshift_option else 'control F0'
        unique_trials = unique_trials_dict[f'{cluster_id}_{ids_to_plot["Folder"][i]}_{probeword}']

        if probeword == 4 and pitchshift_option == False:
            probeword_text = 'when a'
            color_option = 'green'
        elif probeword == 4 and pitchshift_option == True:
            probeword_text = 'when a'
            color_option = 'lightgreen'

        elif probeword == 1 and pitchshift_option == False:
            probeword_text = 'instruments'
            color_option = 'black'
        elif probeword == 1 and pitchshift_option == True:
            probeword_text = 'instruments'
            color_option = 'black'


        elif probeword == 2 and pitchshift_option == False:
            probeword_text = 'craft'
            color_option = 'deeppink'
        elif probeword == 2 and pitchshift_option == True:
            probeword_text = 'craft'
            color_option = 'pink'

        elif probeword == 3 and pitchshift_option == False:
            probeword_text = 'in contrast'
            color_option = 'mediumpurple'
        elif probeword == 3 and pitchshift_option == True:
            probeword_text = 'in contrast'
            color_option = 'purple'

        elif probeword == 5 and pitchshift_option == False:
            probeword_text = 'accurate'
            color_option = 'olivedrab'

        elif probeword == 5 and pitchshift_option == True:
            probeword_text = 'accurate'
            color_option = 'limegreen'
        elif probeword == 6 and pitchshift_option == False:
            probeword_text = 'pink noise'
            color_option = 'navy'
        elif probeword == 6 and pitchshift_option == True:
            probeword_text = 'pink noise'
            color_option = 'lightblue'
        elif probeword == 7 and pitchshift_option == False:
            probeword_text = 'of science'
            color_option = 'coral'
        elif probeword == 7 and pitchshift_option == True:
            probeword_text = 'of science'
            color_option = 'orange'
        elif probeword == 8 and pitchshift_option == False:
            probeword_text = 'rev. instruments'
            color_option = 'plum'
        elif probeword == 8 and pitchshift_option == True:
            probeword_text = 'rev. instruments'
            color_option = 'darkorchid'
        elif probeword == 9 and pitchshift_option == False:
            probeword_text = 'boats'
            color_option = 'cornflowerblue'
        elif probeword == 9 and pitchshift_option == True:
            probeword_text = 'boats'
            color_option = 'royalblue'

        elif probeword == 10 and pitchshift_option == False:
            probeword_text = 'today'
            color_option = 'gold'
        elif probeword == 10 and pitchshift_option == True:
            probeword_text = 'today'
            color_option = 'yellow'
        else:
            probeword_text = 'error'
            color_option = 'red'

        rec_name = str(ids_to_plot['Folder'][i])
        max_length = len(rec_name) // 2

        # for length in range(1, max_length + 1):
        #     for k in range(len(rec_name) - length):
        #         substring = rec_name[k:k + length]
        #         if rec_name.count(substring) > 1:
        #             repeating_substring = substring
        #             break
        #
        # print(repeating_substring)
        # rec_name = repeating_substring
        # rec_name = rec_name[:-1]
        # stream = str(ids_to_plot['Folder'][i])[-4:]


        rasterplot(spiketrains, c=color_option, histogram_bins=0, axes=ax2, s=0.3)
        ax2.set_ylabel('trial number')
        custom_xlim = (-0.1, 0.6)
        ax2.set_xlim(custom_xlim)
        brain_area = pd.read_csv(f'G:/neural_chapter/csvs/unit_ids_all_naive_{animal}.csv')
        if rec_name.__contains__('_s3'):
            stream = 't_s3'
        elif rec_name.__contains__('_s2'):
            stream = 't_s2'
        brain_area = brain_area[(brain_area['stream'] == stream)]

        brain_area_text = brain_area[brain_area['ID'] == cluster_id]['BrainArea'].to_list()[0]

        ax2.set_title(
            f'{cluster_id}_{rec_name},\n {animal_id_num}, probe word: {probeword_text}, {pitchshift_text}, {brain_area_text}')
        plt.savefig(saveDir / f'{cluster_id}_{ids_to_plot["Folder"][i]}_{animal_id_num}_{probeword_text}_PS{pitchshift_text}_{talker}.png')


    return



def generate_rasters(dir):

    animal = 'F2003_Orecchiette'


    datapaths = [Path('G:\F2003_Orecchiette/results_ore_pykilosort_s2/recording_0\pykilosort\phy'),
                 Path('G:\F2003_Orecchiette/results_ore_pykilosort_s3/recording_0\pykilosort\phy'),
                 Path('G:/F2003_Orecchiette/s2cgmod/recording_0\kilosort/phy')]
    for talker in [1]:
        target_vs_probe_with_raster(datapaths,talker=talker, animal = animal)
        # target_vs_probe_with_raster(new_blocks,clust_ids = clust_ids, talker=talker, stream = stream, phydir=repeating_substring, animal = animal, brain_area = brain_area, gen_psth=True)




def main():

    directories = ['zola_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        generate_rasters(dir)


if __name__ == '__main__':
    main()
