from pathlib import Path
import h5py
from tqdm import tqdm
from datetime import datetime
from instruments.helpers.neural_analysis_helpers import get_word_aligned_raster, get_word_aligned_raster_with_pitchshift
from instruments.helpers.euclidean_classification_minimal_function import classify_sweeps
import numpy as np
import pickle



def generatewordspiketrains(blocks, talker=1, probewords=[20, 22], pitchshift=True, window = [0, 0.6]):

    if talker == 1:
        probeword = probewords[0]
    else:
        probeword = probewords[1]
    binsize = 0.01

    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']

    scores = {'cluster_id': [],
              'score': [],
              'cm': [],
              'bootScore': [],
              'lstm_score': [],
              'lstm_avg': [],
              'lstm_balanced': [],
              'lstm_balanced_avg': [],
              'lstm_score_list': [],
              'history': [],}
    cluster_id_droplist = np.empty([])
    raster_reshaped_array = []
    for cluster_id in tqdm(clust_ids):


        probe_filter = ['No Level Cue']  # , 'Non Correction Trials']
        try:
            raster_probe = get_word_aligned_raster(blocks, cluster_id, word=probeword, pitchshift=pitchshift,
                                                   correctresp=True,
                                                   df_filter=probe_filter)
            raster_probe = raster_probe[raster_probe['talker'] == talker]
            if len(raster_probe) == 0:
                print('no relevant spikes for this talker')
                continue
        except:
            print('No relevant probe firing')
            cluster_id_droplist = np.append(cluster_id_droplist, cluster_id)
            continue
        # sample with replacement from target trials and probe trials to boostrap scores and so distributions are equal
        raster_targ_reshaped = np.empty([])
        bins = np.arange(window[0], window[1], binsize)

        unique_trials_probe=np.unique(raster_probe['trial_num'])
        #subsampl

        raster_probe_reshaped = np.empty([len(unique_trials_probe), len(bins) - 1])

        count = 0
        for trial in (unique_trials_probe):
            raster_probe_reshaped[count, :] = np.histogram(raster_probe['spike_time'][raster_probe['trial_num'] == trial], bins=bins, range=(window[0], window[1]))[0]
            count+=1



        if len(raster_probe_reshaped) != 350:
            #resample with replacement to get 350 trials
            raster_probe_reshaped = raster_probe_reshaped[np.random.choice(len(raster_probe_reshaped), 350, replace=True), :]

        raster_lstm = raster_probe_reshaped

        raster_reshaped = np.reshape(raster_lstm, (np.size(raster_lstm, 0), np.size(raster_lstm, 1), 1)).astype(
            'float32')
        raster_reshaped_array.append(raster_reshaped)

        #save as h5 files
    print('saving h5 files')


    raster_reshaped_array_final= np.stack(raster_reshaped_array)
    #remove the last dimension of size 1
    raster_reshaped_array_final = raster_reshaped_array_final[:, :,:, 0]
    #reshape so the first dimension is the number of trials
    raster_reshaped_array_final2 = np.reshape(raster_reshaped_array_final, (np.size(raster_reshaped_array_final, 1), np.size(raster_reshaped_array_final, 2), np.size(raster_reshaped_array_final, 0)))



    with h5py.File(f'D:/tf_h5files/F1702_Zola/raster_reshaped_{str(talker)}_{str(probeword)}_pitchshift_{pitchshift}.h5', 'w') as hf:
        hf.create_dataset("spike_data", data=raster_reshaped_array_final2)
            # hf.create_dataset("stim_data", data=stim_reshaped)


    return




def run_export(dir):


    datapath = Path(f'D:\F1702_Zola\spkenvresults04102022allrowsbut4th')

    fname = 'blocks.pkl'
    with open(datapath / 'blocks.pkl', 'rb') as f:
        blocks = pickle.load(f)

    scores = {}
    probewords_list= [(1,1), (5,6), (2,2),(20,22), (42,49), (32, 38)]
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H_%M_%S")

    tarDir = Path(f'/Users/cgriffiths/resultsms4/lstmclass_CVDATA_14012023zola')
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


            generatewordspiketrains(blocks, talker=talker, probewords=probeword,pitchshift=False, window = window )
            generatewordspiketrains(blocks, talker=talker, probewords=probeword, pitchshift=True, window = window)


def main():
    directories = ['zola_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        run_export(dir)


if __name__ == '__main__':
    main()
