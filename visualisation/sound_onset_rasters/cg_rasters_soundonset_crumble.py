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
from instruments.helpers.neural_analysis_helpers import get_soundonset_alignedraster, get_word_aligned_raster_zola_cruella2
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


def target_vs_probe_with_raster(blocks, talker=1, probewords=[20, 22], pitchshift=True, stream = 'BB_3'):

    tarDir = Path(f'E:/rastersms4spikesortinginter/F1901_Crumble/figsonset2/bb3onsettest_target/')
    saveDir = tarDir
    saveDir.mkdir(exist_ok=True, parents=True)

    binsize = 0.01
    window = [0, 0.6]

    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']

    for s, seg in enumerate(blocks[0].segments):
        unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == 0][0]
        unit_2 = [st for st in seg.spiketrains if st.annotations['cluster_id'] == 0.2][0]
        # compare the two
        # get the rasters
        times = unit.times
        times_2 = unit_2.times
        if np.array_equal(times, times_2):
            print('The spike trains have the same spike times')
        else:
            print('The spike trains have different spike times')

    # cluster_id_droplist = np.empty([])
    # # clust_ids = clust_ids[18:]
    # for cluster_id in clust_ids:
    #     print('now starting cluster')
    #     print(cluster_id)

        target_filter = ['No Level Cue']  # , 'Non Correction Trials']

        # try:




    return



def generate_rasters(dir):
    datapath = Path(f'E:\ms4output2\F1901_Crumble\BB2BB3_crumble_29092023_2\BB2BB3_crumble_29092023_BB2BB3_crumble_29092023_BB_3\mountainsort4\phy/')
    stream = str(datapath).split('\\')[-3]
    stream = stream[-4:]
    print(stream)
    with open(datapath / 'new_blocks.pkl', 'rb') as f:
        blocks = pickle.load(f)
    scores = {}
    probewords_list = [(2,2),]


    for probeword in probewords_list:
        print('now starting')
        print(probeword)
        for talker in [1]:

            # target_vs_probe_with_raster(blocks, talker=talker,probewords=probeword,pitchshift=False)
            target_vs_probe_with_raster(blocks, talker=talker,probewords=probeword,pitchshift=False, stream = stream)




def main():

    directories = ['eclair_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        generate_rasters(dir)


if __name__ == '__main__':
    main()
