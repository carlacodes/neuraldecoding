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
from instruments.helpers.neural_analysis_helpers import get_soundonset_alignedraster, split_cluster_base_on_segment_zola, get_soundonset_alignedraster_tabular, get_word_aligned_raster
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
def get_spike_times_tabular(blocks, talker=1, probewords=[20, 22], pitchshift=True, stream ='BB_3'):

    tarDir = Path(f'E:/rastersms4spikesortinginter/F1702_Zola/figsonset2/{stream}/')
    saveDir = tarDir
    saveDir.mkdir(exist_ok=True, parents=True)

    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']

    for st in blocks[0].segments[0].spiketrains:
        print(f"Cluster ID: {st.annotations['cluster_id']}, Group: {st.annotations['group']}")

    cluster_spiketime_dict = {}
    for cluster_id in clust_ids:
        print('now starting cluster')
        print(cluster_id)

        filter = ['No Level Cue']  # , 'Non Correction Trials']

        # try:
        spike_times_list, df_behavior_cluster = get_soundonset_alignedraster_tabular(blocks, cluster_id, df_filter=filter)
        cluster_spiketime_dict[cluster_id]['spike_times'] = spike_times_list
        cluster_spiketime_dict[cluster_id]['behavior'] = df_behavior_cluster

    return cluster_spiketime_dict



def generate_rasters(dir):
    datapath = Path(f'E:\ms4output2\F1702_Zola\BB2BB3_zola_intertrialroving_26092023\BB2BB3_zola_intertrialroving_26092023_BB2BB3_zola_intertrialroving_26092023_BB_2\mountainsort4\phy/')
    stream = str(datapath).split('\\')[-3]
    stream = stream[-4:]
    print(stream)
    probewords_list = [(4,4),]
    with open(datapath / 'new_blocks.pkl', 'rb') as f:
        new_blocks = pickle.load(f)
    for probeword in probewords_list:
        print('now starting')
        print(probeword)
        for talker in [1, 2]:
            get_spike_times_tabular(new_blocks, talker=talker, probewords=probeword, pitchshift=False, stream = stream)




def main():
    directories = ['zola_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        generate_rasters(dir)


if __name__ == '__main__':
    main()
