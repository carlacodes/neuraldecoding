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





def run_cleaning_of_rasters(blocks, datapath):
    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']
    for cluster_id in clust_ids:
        new_blocks = split_cluster_base_on_segment_zola(blocks, cluster_id, num_clusters=2)
    with open(datapath / 'new_blocks.pkl', 'wb') as f:
        pickle.dump(new_blocks, f)
    return new_blocks
def target_vs_probe_with_raster(blocks, talker=1, clust_ids = [], stream = 'BB_3', phydir = 'phy', animal = 'F1702_Zola'):

    tarDir = Path(f'E:/rastersms4spikesortinginter/{animal}/figs_dist_and_targ_1611/{phydir}/{stream}/')
    #load the high generalizable clusters, csv file





    saveDir = tarDir
    saveDir.mkdir(exist_ok=True, parents=True)

    binsize = 0.01
    window = [0, 0.6]
    probewords_list = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]

    for st in blocks[0].segments[0].spiketrains:
        print(f"Cluster ID: {st.annotations['cluster_id']}, Group: {st.annotations['group']}")


    for cluster_id in clust_ids:
        #make a figure of 2 columns and 10 rows
        fig, ax = plt.subplots(10, 2, figsize=(10, 20))
        count = 0
        for probeword in probewords_list:
            for pitchshift_option in [True, False]:
                print('now starting cluster')
                print(cluster_id)
                # try:
                raster_target, raster_target_compare = get_word_aligned_raster_zola_cruella(blocks, cluster_id, word=probeword,
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

                print(spiketrains)
                try:
                    if probewords[0] == 4:
                        probeword_text = 'when a'
                        color_option = 'black'
                    elif probewords[0] == 1:
                        probeword_text = 'instruments'
                        color_option = 'blue'
                    elif probewords[0] == 2:
                        probeword_text = 'craft'
                        color_option = 'black'
                    elif probewords[0] == 3:
                        probeword_text = 'in contrast'
                        color_option = 'black'
                    elif probewords[0] == 5:
                        probeword_text = 'accurate'
                        color_option = 'black'
                    elif probewords[0] == 6:
                        probeword_text = 'pink noise'
                        color_option = 'black'
                    elif probewords[0] == 7:
                        probeword_text = 'of science'
                        color_option = 'black'
                    elif probewords[0] == 8:
                        probeword_text = 'rev. instruments'
                        color_option = 'black'
                    elif probewords[0] == 9:
                        probeword_text = 'boats'
                        color_option = 'black'

                    elif probewords[0] == 10:
                        probeword_text = 'today'
                        color_option = 'green'
                    else:
                        probeword_text = 'error'
                        color_option = 'red'
                    #if pitchshift plot on the second column
                    custom_xlim = (-0.1, 0.6)

                    if pitchshift_option == True:
                        rasterplot(spiketrains, c=color_option, histogram_bins=100, axes=ax[count, 1], s=0.3 ) #s=0.5
                        ax[count, 1].set_ylabel('trial')
                        plt.setp(ax[count, 1], xlim=custom_xlim)
                        #add a title to the first row
                        ax[0, 1].set_title('Pitch-shifted F0')
                        # ax[0].set_xlabel('Time relative to word presentation (s)')
                    elif pitchshift_option == False:
                        rasterplot(spiketrains, c=color_option, histogram_bins=100, axes=ax[count, 0], s=0.3 )
                        ax[count, 0].set_ylabel('trial')
                        plt.setp(ax[count, 0], xlim=custom_xlim)
                        #add label next to the y axis for the probeword
                        ax[count, 0].text(-0.2, 0.5, probeword_text, horizontalalignment='center', verticalalignment='center', rotation=90, transform=ax[count, 0].transAxes)
                        #add a title to the first row
                        ax[0, 0].set_title('Control F0')
                        # ax[0].set_xlabel('Time relative to word presentation (s)')


                    ax[0].set_ylabel('trial')
                    # ax[0].set_xlabel('Time relative to word presentation (s)')
                    custom_xlim = (-0.1, 0.6)

                    plt.setp(ax, xlim=custom_xlim)
                    if pitchshift == False:
                        pitchtext = 'Control F0'
                    else:
                        pitchtext = 'Pitch-shifted F0'
                    count += 1

                except:
                    print('no spikes')
                    continue

    # add labels beside the axes

    plt.savefig(
        str(saveDir) + f'/targdist_grid_clusterid_{cluster_id}_{stream}_pitchshift_{pitchshift}_' + str(
            cluster_id) + '.png')
            # plt.show()



    return



def generate_rasters(dir):

    datapath_big = Path(f'D:/ms4output_16102023/F1815_Cruella/')
    animal = str(datapath_big).split('\\')[-1]
    datapaths = [x for x in datapath_big.glob('**/mountainsort4/phy//') if x.is_dir()]
    datapaths = datapaths[5:]
    for datapath in datapaths:
        stream = str(datapath).split('\\')[-3]
        stream = stream[-4:]
        print(stream)
        folder = str(datapath).split('\\')[-3]
        with open(datapath / 'new_blocks.pkl', 'rb') as f:
            new_blocks = pickle.load(f)

        high_units = pd.read_csv(f'G:/neural_chapter/figures/unit_ids_trained_highthreshold_index_{animal}.csv')
        # remove trailing steam
        rec_name = folder[:-4]

        # find the units that have the phydir
        high_units = high_units[(high_units['rec_name'] == rec_name) and (high_units['stream'] == stream)]
        clust_ids = high_units['ID'].to_list()

        for talker in [1]:
            target_vs_probe_with_raster(new_blocks,clust_ids = clust_ids, talker=talker, stream = stream, phydir=folder, animal = animal)




def main():

    directories = ['zola_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        generate_rasters(dir)


if __name__ == '__main__':
    main()
