import pickle
from pathlib import Path
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
from instruments.helpers.neural_analysis_helpers import get_word_aligned_raster_squinty
from instruments.helpers.euclidean_classification_minimal_function import classify_sweeps
# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle

# If you would prefer to load the '.h5' example file rather than the '.pickle' example file. You need the deepdish package
# import deepdish as dd

# Import function to get the covariate matrix that includes spike history from previous bins
from Neural_Decoding.preprocessing_funcs import get_spikes_with_history
import Neural_Decoding
# Import metrics
from Neural_Decoding.metrics import get_R2
from Neural_Decoding.metrics import get_rho

# Import decoder functions
from Neural_Decoding.decoders import LSTMDecoder, LSTMClassification


def target_vs_probe_with_raster(blocks, talker=1, probewords=[20, 22], pitchshift=True):
    # datapath = Path('/Users/juleslebert/home/phd/fens_data/warp_data/Trifle_June_2022/Trifle_week_16_05_22
    # /mountainsort4/phy') fname = 'blocks.pkl' with open(datapath / 'blocks.pkl', 'rb') as f: blocks = pickle.load(f)
    now = datetime.now()

    tarDir = Path(f'E:\decoding_over_time_l74\F1604_Squinty\myriad3/bb2//figs2/')
    saveDir = tarDir
    saveDir.mkdir(exist_ok=True, parents=True)


    if talker == 1:
        probeword = probewords[0]
    else:
        probeword = probewords[1]
    binsize = 0.01
    window = [0, 0.6]

    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']
    # clust_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14, 15]

    cluster_id_droplist = np.empty([])
    for cluster_id in clust_ids:
        print('now starting cluster')
        print(cluster_id)

        target_filter = ['Target trials', 'No Level Cue']  # , 'Non Correction Trials']

        try:
            # CHANGE BELOW TO SQUINTY RASTER FUNCTION:
            raster_target = get_word_aligned_raster_squinty(blocks, cluster_id, word=1, pitchshift=pitchshift,
                                                    correctresp=False,
                                                    df_filter=target_filter)
            raster_target = raster_target.reshape(raster_target.shape[0], )


            # # raster_target = raster_target[raster_target['talker'] == int(talker)]
            # if len(raster_target) == 0:
            #     print('no relevant spikes for this talker')
            #     continue
        except:
            print('No relevant target firing')
            cluster_id_droplist = np.append(cluster_id_droplist, cluster_id)
            continue

        probe_filter = ['No Level Cue']  # , 'Non Correction Trials']
        try:
            raster_probe = get_word_aligned_raster_squinty(blocks, cluster_id, word=probeword, pitchshift=pitchshift,
                                                   correctresp=False,
                                                   df_filter=probe_filter)
            raster_probe = raster_probe.reshape(raster_probe.shape[0], )

            raster_probe['trial_num'] = raster_probe['trial_num'] + np.max(raster_target['trial_num'])
            # if len(raster_probe) == 0:
            #     print('no relevant spikes for this talker')
            #     continue
        except:
            print('No relevant probe firing')
            cluster_id_droplist = np.append(cluster_id_droplist, cluster_id)

            continue
        # sample with replacement from target trials and probe trials to boostrap scores and so distributions are equal

        bins = np.arange(window[0], window[1], binsize)


        unique_trials_targ = np.unique(raster_target['trial_num'])
        unique_trials_probe = np.unique(raster_probe['trial_num'])
        raster_targ_reshaped = np.empty([len(unique_trials_targ), len(bins) - 1])
        raster_probe_reshaped = np.empty([len(unique_trials_probe), len(bins) - 1])
        count = 0
        for trial in (unique_trials_targ):
            raster_targ_reshaped[count, :] = \
            np.histogram(raster_target['spike_time'][raster_target['trial_num'] == trial], bins=bins,
                         range=(window[0], window[1]))[0]
            count += 1
        count = 0
        for trial in (unique_trials_probe):
            raster_probe_reshaped[count, :] = \
            np.histogram(raster_probe['spike_time'][raster_probe['trial_num'] == trial], bins=bins,
                         range=(window[0], window[1]))[0]
            count += 1

        # stim0 = np.full(len(raster_target), 0)  # 0 = target word
        # stim1 = np.full(len(raster_probe), 1)  # 1 = probe word
        # stim = np.concatenate((stim0, stim1))
        #
        # stim0 = np.full(len(raster_targ_reshaped), 0)  # 0 = target word
        # stim1 = np.full(len(raster_probe_reshaped), 1)  # 1 = probe word
        #
        #




        spiketrains = []
        for trial_id in unique_trials_targ:
            selected_trials = raster_target[raster_target['trial_num'] == trial_id]
            spiketrain = neo.SpikeTrain(selected_trials['spike_time'], units='s', t_start=min(selected_trials['spike_time']), t_stop=max(selected_trials['spike_time']))
            spiketrains.append(spiketrain)

        print(spiketrains)

        fig,ax = plt.subplots(2, figsize=(10, 5))
        #ax.scatter(raster_target['spike_time'], np.ones_like(raster_target['spike_time']))
        rasterplot(spiketrains, c='black', histogram_bins=100, axes=ax, s=3 )

        ax[0].set_ylabel('trial')
        ax[0].set_xlabel('Time relative to word presentation (s)')
        custom_xlim = (-0.1, 0.6)

        plt.setp(ax, xlim=custom_xlim)

        plt.suptitle('Target firings for squinty,  clus id '+ str(cluster_id)+' pitchshift = '+str(pitchshift)+', talker '+str(talker), fontsize = 12)
        plt.savefig(
            str(saveDir) + '/targ_clusterid' + str(cluster_id) + ' probeword ' + str(probeword) + ' pitch ' + str(
                pitchshift) + 'talker' + str(talker) + '.png')
        #plt.show()


        try:
            spiketrains = []
            for trial_id in unique_trials_probe:
                selected_trials = raster_probe[raster_probe['trial_num'] == trial_id]
                spiketrain = neo.SpikeTrain(selected_trials['spike_time'], units='s', t_start=min(selected_trials['spike_time']), t_stop=max(selected_trials['spike_time']))
                spiketrains.append(spiketrain)


            #ax.scatter(raster_target['spike_time'], np.ones_like(raster_target['spike_time']))
            fig2,ax = plt.subplots(2, figsize=(10, 5))

            rasterplot(spiketrains, c='blue', histogram_bins=100, s=3, axes=ax)

            ax[0].set_ylabel('trial')
            ax[0].set_xlabel('Time relative to word presentation (s)')
            custom_xlim = (-0.1, 0.6)

            plt.setp(ax, xlim=custom_xlim)
            plt.suptitle('Distractor firings for squinty,  clus id '+ str(cluster_id)+' , pitchshift = '+str(pitchshift)+ ' probeword '+str(probeword)+' talker '+str(talker), fontsize = 12)



            plt.savefig(
                str(saveDir) + '/dist_clusterid' + str(cluster_id) + ' probeword ' + str(probeword) + ' pitch ' + str(
                    pitchshift) + 'talker' + str(talker) + '.png')
        except:
            print('no distractor firing')
            continue
    return



def run_classification(dir):
    datapath = Path(f'E:\ms4output2\F1604_Squinty\BB2BB3_squinty_MYRIAD3_23092023_58noiseleveledit3medthreshold\BB2BB3_squinty_MYRIAD3_23092023_58noiseleveledit3medthreshold_BB2BB3_squinty_MYRIAD3_23092023_58noiseleveledit3medthreshold_BB_2\mountainsort4\phy/')
    with open(datapath / 'blocks.pkl', 'rb') as f:
        blocks = pickle.load(f)
    scores = {}
    probewords_list = [(2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11, 11), (12, 12), (13, 13), (14, 14)]


    for probeword in probewords_list:
        print('now starting')
        print(probeword)
        for talker in [1]:

            # target_vs_probe_with_raster(blocks, talker=talker,probewords=probeword,pitchshift=False)
            target_vs_probe_with_raster(blocks, talker=talker,probewords=probeword,pitchshift=False)




def main():

    directories = ['zola_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        run_classification(dir)


if __name__ == '__main__':
    main()
