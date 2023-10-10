import copy
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
from instruments.helpers.neural_analysis_helpers import get_soundonset_alignedraster, split_cluster_base_on_segment,get_word_aligned_raster_zola_cruella,  split_cluster_base_on_segment_zola
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
    #check if each unit is the same
    for i in range(0, len(blocks[0].segments)):
        clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[i].spiketrains]
        print(f"Debug - Cluster IDs: {clust_ids} for iteration {i}")
    with open(datapath / 'new_blocks.pkl', 'wb') as f:
        pickle.dump(new_blocks, f)
    return new_blocks
def target_vs_probe_with_raster(blocks, talker=1, probewords=[20, 22], pitchshift=True, stream = 'BB_3'):
    # datapath = Path('/Users/juleslebert/home/phd/fens_data/warp_data/Trifle_June_2022/Trifle_week_16_05_22
    # /mountainsort4/phy') fname = 'blocks.pkl' with open(datapath / 'blocks.pkl', 'rb') as f: blocks = pickle.load(f)
    now = datetime.now()

    tarDir = Path(f'E:/rastersms4spikesortinginter/F1901_Crumble/figsonset2/distandtarg/bb3_soundonset/')
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

    for cluster_id in [16, 216]:
        print('now starting cluster')
        print(cluster_id)

        target_filter = ['Target trials', 'No Level Cue']  # , 'Non Correction Trials']

        # # try:
        # raster_target = get_word_aligned_raster_zola_cruella(blocks, cluster_id, word=1, pitchshift=pitchshift,
        #                                         correctresp=False,
        #                                         df_filter=target_filter)
        raster_target, raster_target_compare = get_soundonset_alignedraster(blocks, cluster_id)
        raster_target = raster_target.reshape(raster_target.shape[0], )
        raster_target_compare = raster_target_compare.reshape(raster_target_compare.shape[0], )
        if (raster_target == raster_target_compare).all:
            print('they are the same for cluster;' + str(cluster_id))
        # except:
        #     print('No relevant target firing')
        #     cluster_id_droplist = np.append(cluster_id_droplist, cluster_id)
        #     continue

        probe_filter = ['No Level Cue']  # , 'Non Correction Trials']
        bins = np.arange(window[0], window[1], binsize)

        unique_trials_targ = np.unique(raster_target['trial_num'])
        raster_targ_reshaped = np.empty([len(unique_trials_targ), len(bins) - 1])
        count = 0
        for trial in (unique_trials_targ):
            raster_targ_reshaped[count, :] = \
                np.histogram(raster_target['spike_time'][raster_target['trial_num'] == trial], bins=bins,
                             range=(window[0], window[1]))[0]
            count += 1
        count = 0


        spiketrains = []
        for trial_id in unique_trials_targ:
            selected_trials = raster_target[raster_target['trial_num'] == trial_id]
            spiketrain = neo.SpikeTrain(selected_trials['spike_time'], units='s',
                                        t_start=min(selected_trials['spike_time']),
                                        t_stop=max(selected_trials['spike_time']))
            spiketrains.append(spiketrain)


        if cluster_id == 16:
            cluster_id_test1 = copy.deepcopy(raster_target)
        if cluster_id == 16.2:
            cluster_id_test2 =copy.deepcopy(raster_target)





    #get the contents of the cluster_id_test1 and cluster_id_test2
    #check if the two arrays are the same
    tolerance = 1e-6  # Adjust this value as needed based on your data and precision requirements

    # Check if the absolute difference between arrays is within the tolerance level
    if np.array_equal(cluster_id_test1['spike_time'], cluster_id_test2['spike_time']):
        print('The spike times are the same within the given shape')
    else:
        print('The spike times are different within the given shape')


    return



def generate_rasters(dir):
    datapath = Path(f'E:\ms4output2\F1901_Crumble\BB2BB3_crumble_29092023_2\BB2BB3_crumble_29092023_BB2BB3_crumble_29092023_BB_3\mountainsort4\phy/')


    stream = str(datapath).split('\\')[-3]
    stream = stream[-4:]
    print(stream)

    probewords_list = [(2,2),]

    with open(datapath / 'blocks.pkl', 'rb') as f:
        blocks = pickle.load(f)
    new_blocks = run_cleaning_of_rasters(blocks, datapath)

    for probeword in probewords_list:
        print('now starting')
        print(probeword)
        for talker in [1]:

            with open(datapath / 'new_blocks.pkl', 'rb') as f:
                new_blocks = pickle.load(f)
            # target_vs_probe_with_raster(blocks, talker=talker,probewords=probeword,pitchshift=False)
            target_vs_probe_with_raster(new_blocks, talker=talker,probewords=probeword,pitchshift=False, stream = stream)




def main():

    directories = ['zola_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        generate_rasters(dir)


if __name__ == '__main__':
    main()
