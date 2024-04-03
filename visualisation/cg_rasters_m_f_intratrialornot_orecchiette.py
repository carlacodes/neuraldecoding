import pickle
from pathlib import Path
from tqdm import tqdm
from viziphant.rasterplot import rasterplot
from datetime import datetime
from instruments.helpers.neural_analysis_helpers import get_word_aligned_raster, get_soundonset_alignedraster
from instruments.helpers.euclidean_classification_minimal_function import classify_sweeps
# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
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
    if talker == 1:
        probeword = probewords[0]
    else:
        probeword = probewords[1]
    binsize = 0.01
    window = [0, 0.6]

    epochs = ['Early', 'Late']
    epoch_threshold = 1.5
    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']

    scores = {'cluster_id': [],
              'score': [],
              'cm': [],
              'bootScore': [],
              'lstm_score': [],
              'lstm_avg': [],
              'lstm_balanced': [],
              'lstm_balanced_avg': [], }
    cluster_id_droplist = np.empty([])
    for cluster_id in tqdm(clust_ids):

        target_filter = ['Target trials', 'No Level Cue']  # , 'Non Correction Trials']

        try:
            raster_target = get_word_aligned_raster(blocks, cluster_id, word=1, pitchshift=pitchshift,
                                                    correctresp=False,
                                                    df_filter=target_filter)
            raster_target = raster_target[raster_target['talker'] == int(talker)]
            if len(raster_target) == 0:
                print('no relevant spikes for this talker')
                continue
        except:
            print('No relevant target firing')
            cluster_id_droplist = np.append(cluster_id_droplist, cluster_id)
            continue

        probe_filter = ['No Level Cue']  # , 'Non Correction Trials']
        try:
            raster_probe = get_word_aligned_raster(blocks, cluster_id, word=probeword, pitchshift=pitchshift,
                                                   correctresp=False,
                                                   df_filter=probe_filter)
            raster_probe = raster_probe[raster_probe['talker'] == talker]
            raster_probe['trial_num'] = raster_probe['trial_num'] + np.max(raster_target['trial_num'])
            if len(raster_probe) == 0:
                print('no relevant spikes for this talker')
                continue
        except:
            print('No relevant probe firing')
            cluster_id_droplist = np.append(cluster_id_droplist, cluster_id)

            continue
        # sample with replacement from target trials and probe trials to boostrap scores and so distributions are equal
        lengthofraster = np.sum(len(raster_target['spike_time']) + len(raster_probe['spike_time']))
        raster_targ_reshaped = np.empty([])
        raster_probe_reshaped = np.empty([])
        bins = np.arange(window[0], window[1], binsize)

        lengthoftargraster = len(raster_target['spike_time'])
        lengthofproberaster = len(raster_probe['spike_time'])

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

        stim0 = np.full(len(raster_target), 0)  # 0 = target word
        stim1 = np.full(len(raster_probe), 1)  # 1 = probe word
        stim = np.concatenate((stim0, stim1))

        stim0 = np.full(len(raster_targ_reshaped), 0)  # 0 = target word
        stim1 = np.full(len(raster_probe_reshaped), 1)  # 1 = probe word
        # if len(stim0) < 10 or len(stim1) < 10:
        #     print('less than 10 trials')
        #     continue
        stim_lstm = np.concatenate((stim0, stim1))

        raster = np.concatenate((raster_target, raster_probe))
        raster_lstm = np.concatenate((raster_targ_reshaped, raster_probe_reshaped))

        # score, d, bootScore, bootClass, cm = classify_sweeps(raster, stim, binsize=binsize, window=window, genFig=False)

        newraster = raster.tolist()
        raster_reshaped = np.reshape(raster_lstm, (np.size(raster_lstm, 0), np.size(raster_lstm, 1), 1)).astype(
            'float32')
        stim_reshaped = np.reshape(stim_lstm, (len(stim_lstm), 1)).astype('float32')

        import neo

        # Create a 2D numpy array of spike times
        spike_times = raster_target['spike_time']

        # Create a list of SpikeTrain objects

        spiketrains = []
        for trial_id in unique_trials_targ:
            selected_trials = raster_target[raster_target['trial_num'] == trial_id]
            spiketrain = neo.SpikeTrain(selected_trials['spike_time'], units='s', t_start=min(selected_trials['spike_time']), t_stop=max(selected_trials['spike_time']))
            spiketrains.append(spiketrain)

        print(spiketrains)

        num_trials = np.shape(raster_targ_reshaped)[0]
        fig,ax = plt.subplots(2, figsize=(20, 10))
        #ax.scatter(raster_target['spike_time'], np.ones_like(raster_target['spike_time']))
        rasterplot(spiketrains, c='black', histogram_bins=100, s=3, axes=ax)

        ax[0].set_ylabel('trial')
        ax[0].set_xlabel('Time relative to word presentation (s)')
        custom_xlim = (-0.1, 0.6)

        plt.setp(ax, xlim=custom_xlim)

        plt.suptitle('Target firings for ore,  clus id ' + str(cluster_id)+'pitchshift = '+str(pitchshift)+'talker'+str(talker), fontsize = 20)
        plt.savefig('D:/Data/rasterplotsfromdecoding/ore/mandf/oretarg2_clusterid'+str(cluster_id)+' probeword '+str(probeword)+' pitch '+str(pitchshift)+'talker'+str(talker)+'.png')
        #plt.show()

        spiketrains = []
        for trial_id in unique_trials_probe:
            selected_trials = raster_probe[raster_probe['trial_num'] == trial_id]
            spiketrain = neo.SpikeTrain(selected_trials['spike_time'], units='s', t_start=min(selected_trials['spike_time']), t_stop=max(selected_trials['spike_time']))
            spiketrains.append(spiketrain)


        #ax.scatter(raster_target['spike_time'], np.ones_like(raster_target['spike_time']))
        fig2,ax = plt.subplots(2, figsize=(20, 10))

        rasterplot(spiketrains, c='blue', histogram_bins=100, s=3, axes=ax)

        ax[0].set_ylabel('trial')
        ax[0].set_xlabel('Time relative to word presentation (s)')
        custom_xlim = (-0.1, 0.6)

        plt.setp(ax, xlim=custom_xlim)
        plt.suptitle('Distractor firings for ore,  clus id '+ str(cluster_id)+' , pitchshift = '+str(pitchshift)+ 'probeword '+str(probeword)+'talker'+str(talker), fontsize = 20)



        plt.savefig('D:/Data/rasterplotsfromdecoding/Ore/Oredist2_clusterid'+  str(cluster_id)+'probe'+str(probeword)+' , pitch '+str(pitchshift)+'talker'+str(talker)+'.png')
        #plt.show()
    return

def generate_rasters_soundonset(blocks, talker=1, pitchshift=True):

    binsize = 0.01
    window = [0, 0.6]


    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']

    cluster_id_droplist = np.empty([])
    for cluster_id in tqdm(clust_ids):

        target_filter = ['No Level Cue']  # , 'Non Correction Trials']

        try:
            print('generating raster for sound onset')
            raster_target = get_soundonset_alignedraster(blocks, cluster_id)
            raster_target = raster_target[raster_target['talker'] == int(talker)]

        except:
            print('No relevant firing')
            cluster_id_droplist = np.append(cluster_id_droplist, cluster_id)
            continue


        bins = np.arange(window[0], window[1], binsize)

        lengthoftargraster = len(raster_target['spike_time'])

        unique_trials_targ = np.unique(raster_target['trial_num'])

        raster_targ_reshaped = np.empty([len(unique_trials_targ), len(bins) - 1])
        count = 0
        for trial in (unique_trials_targ):
            raster_targ_reshaped[count, :] = \
            np.histogram(raster_target['spike_time'][raster_target['trial_num'] == trial], bins=bins,
                         range=(window[0], window[1]))[0]
            count += 1


        stim0 = np.full(len(raster_target), 0)  # 0 = target word

        stim0 = np.full(len(raster_targ_reshaped), 0)  # 0 = target word
        import neo

        spiketrains = []
        for trial_id in unique_trials_targ:
            selected_trials = raster_target[raster_target['trial_num'] == trial_id]
            spiketrain = neo.SpikeTrain(selected_trials['spike_time'], units='s', t_start=min(selected_trials['spike_time']), t_stop=max(selected_trials['spike_time']))
            spiketrains.append(spiketrain)

        print(spiketrains)

        fig,ax = plt.subplots(2, figsize=(20, 10))
        rasterplot(spiketrains, c='black', histogram_bins=100, s=3, axes=ax)

        ax[0].set_ylabel('trial')
        ax[0].set_xlabel('Time relative to sound onset presentation (s)')
        custom_xlim = (-0.1, 0.6)

        plt.setp(ax, xlim=custom_xlim)

        plt.suptitle('Sound onset for Ore,  clus id ' + str(cluster_id)+'pitchshift = '+str(pitchshift)+'talker'+str(talker), fontsize = 20)
        plt.savefig('D:/Data/rasterplotsfromdecoding/Ore/mandf/Ore_clusid'+str(cluster_id)+' soundonset'+ str(pitchshift)+'talker'+str(talker)+'.png')



    return

def run_classification(dir):
    datapath = Path(f'E:/resultskilosort/F2003_Orecchiette/phy_folder')
    fname = 'blocks.pkl'
    with open(datapath / 'blocks.pkl', 'rb') as f:
        blocks = pickle.load(f)
    scores = {}
    probewords_list = [(2, 2), (20, 22), (5, 6), (42, 49), (32, 38)]
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H_%M_%S")

    tarDir = Path(f'/Users/cgriffiths/resultsms4/lstmclass_CVDATA_05042023')
    saveDir = tarDir / dt_string
    saveDir.mkdir(exist_ok=True, parents=True)
    for probeword in probewords_list:
        print('now starting')
        print(probeword)
        for talker in [1, 2]:
            binsize = 0.01
            if talker == 1:
                window = [0, 0.6]
            else:
                window = [0, 0.5]
            # window=[0,0.87]
            print(f'talker {talker}')

            scores[f'talker{talker}'] = {}

            scores[f'talker{talker}']['target_vs_probe'] = {}

            target_vs_probe_with_raster(blocks, talker=talker, probewords=probeword, pitchshift=False)
            target_vs_probe_with_raster(blocks, talker=talker, probewords=probeword, pitchshift=True)


def run_soundonset_rasters(dir):
    datapath = Path(f'E:/resultskilosort/F2003_Orecchiette/phy_folder/')
    fname = 'blocks.pkl'
    with open(datapath / 'blocks.pkl', 'rb') as f:
        blocks = pickle.load(f)
    scores = {}
    probewords_list = [(2, 2), (20, 22), (5, 6), (42, 49), (32, 38)]
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H_%M_%S")

    # tarDir = Path(f'/Users/cgriffiths/resultsms4/lstmclass_CVDATA_05042023')
    # saveDir = tarDir / dt_string
    # saveDir.mkdir(exist_ok=True, parents=True)

    for talker in [1, 2]:
        binsize = 0.01
        if talker == 1:
            window = [0, 0.6]
        else:
            window = [0, 0.5]
        # window=[0,0.87]
        print(f'talker {talker}')

        scores[f'talker{talker}'] = {}

        scores[f'talker{talker}']['target_vs_probe'] = {}

        generate_rasters_soundonset(blocks, talker=talker, pitchshift=False)

def main():

    directories = ['orecchiette_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        run_soundonset_rasters(dir)
        # run_classification(dir)


if __name__ == '__main__':
    main()
