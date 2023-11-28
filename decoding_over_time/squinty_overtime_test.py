import pickle
from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, LeavePOut
from tqdm import tqdm
from keras import backend as K
import pandas as pd
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
from helpers.neural_analysis_helpers_zolainter import get_word_aligned_raster_squinty
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


def target_vs_probe(blocks, talker=1, probewords=[20, 22], pitchshift=True, window=[0, 0.5], clust_ids=[], brain_area=[]):
    if talker == 1:
        probeword = probewords[0]
        talker_text = 'female'
    else:
        probeword = probewords[1]
        talker_text = 'male'
    binsize = 0.01
    # window = [0, 0.6]

    epochs = ['Early', 'Late']
    epoch_threshold = 1.5

    scores = {'cluster_id': [],
              'score': [],
              'cm': [],
              'bootScore': [],
              'lstm_score': [],
              'lstm_balanced_avg': [],
              'lstm_accuracylist': [],
              'lstm_balancedaccuracylist': [],
              'history:': [],
              'lstm_avg': [],
              'history': [],
              'time_bin': [],
              'perm_ac': [],
              'perm_bal_ac': [],
              'brainarea': [],}


    for i, cluster_id in enumerate(clust_ids):
        print('cluster_id:')
        print(cluster_id)
        print('iteration:')
        print(i)
        # try:
        # def get_word_aligned_raster_squinty(blocks, clust_id, word=None, pitchshift=True, correctresp=True,
        #                                     df_filter=[]):
        raster_target  = get_word_aligned_raster_squinty(blocks, cluster_id, word=1,
                                                                                  pitchshift=pitchshift,
                                                                                  correctresp=True,
                                                                                  df_filter=['No Level Cue']  )
        raster_target = raster_target.reshape(raster_target.shape[0], )
        if len(raster_target) == 0:
            print('no relevant spikes for this target word:' + str(probeword) + ' and cluster: ' + str(cluster_id))
            continue

        probe_filter = ['No Level Cue']  # , 'Non Correction Trials']
        # try:
        raster_probe = get_word_aligned_raster_squinty(blocks, cluster_id, word=probeword,
                                                                                  pitchshift=pitchshift,
                                                                                  correctresp=True,
                                                                                  df_filter=['No Level Cue']  )
        # raster_probe = raster_probe[raster_probe['talker'] == talker]
        raster_probe = raster_probe.reshape(raster_probe.shape[0], )

        raster_probe['trial_num'] = raster_probe['trial_num'] + np.max(raster_target['trial_num'])
        if len(raster_probe) == 0:
            print('no relevant spikes for this probe word:' + str(probeword) + ' and cluster: ' + str(cluster_id))
            continue
        # except:
        #     print('No relevant probe firing')
        #     cluster_id_droplist = np.append(cluster_id_droplist, cluster_id)

        #     continue
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
        if (len(raster_targ_reshaped)) < 5 or (len(raster_probe_reshaped)) < 5:
            print('less than 5 trials for the target or distractor, CV would be overinflated, skipping')
            continue
        if len(raster_targ_reshaped) < 15:
            # upsample to 15 trials
            raster_targ_reshaped = raster_targ_reshaped[np.random.choice(len(raster_targ_reshaped), 15, replace=True),
                                   :]
        if len(raster_probe_reshaped) < 15:
            # upsample to 15 trials
            raster_probe_reshaped = raster_probe_reshaped[
                                    np.random.choice(len(raster_probe_reshaped), 15, replace=True), :]

        if len(raster_targ_reshaped) >= len(raster_probe_reshaped) * 2:
            print('raster of distractor at least a 1/2 of target raster')
            # upsample the probe raster
            raster_probe_reshaped = raster_probe_reshaped[
                                    np.random.choice(len(raster_probe_reshaped), len(raster_targ_reshaped),
                                                     replace=True), :]
        elif len(raster_probe_reshaped) >= len(raster_targ_reshaped) * 2:
            print('raster of target at least a 1/2 of probe raster')
            # upsample the target raster
            raster_targ_reshaped = raster_targ_reshaped[
                                   np.random.choice(len(raster_targ_reshaped), len(raster_probe_reshaped),
                                                    replace=True), :]

        print('now length of raster_probe is:')
        print(len(raster_probe_reshaped))
        stim0 = np.full(len(raster_target), 0)  # 0 = target word
        stim1 = np.full(len(raster_probe), 1)  # 1 = probe word
        stim = np.concatenate((stim0, stim1))

        stim0 = np.full(len(raster_targ_reshaped), 0)  # 0 = target word
        stim1 = np.full(len(raster_probe_reshaped), 1)  # 1 = probe word
        if (len(stim0)) < 3 or (len(stim1)) < 3:
            print('less than 3 trials for the target or distractor, skipping')
            continue

        stim_lstm = np.concatenate((stim0, stim1))

        raster = np.concatenate((raster_target, raster_probe))
        raster_lstm = np.concatenate((raster_targ_reshaped, raster_probe_reshaped))

        # score, d, bootScore, bootClass, cm = classify_sweeps(raster, stim, binsize=binsize, window=window, genFig=False)
        # fit LSTM model to the same data

        newraster = raster.tolist()
        raster_reshaped = np.reshape(raster_lstm, (np.size(raster_lstm, 0), np.size(raster_lstm, 1), 1)).astype(
            'float32')
        stim_reshaped = np.reshape(stim_lstm, (len(stim_lstm), 1)).astype('float32')
        X = raster_reshaped
        y = stim_reshaped

        # tf.keras.backend.clear_session()
        K.clear_session()

        # totalaclist = []
        # totalbalaclist = []
        # totalpermacc = []
        # totalpermbalacc = []
        # create a loop that iterates from 1 to n time point and trains the model on n-1 time points
        # break X and y into time bins from 1 to k
        X_bin = X[:, :].copy()
        y_bin = y[:, :].copy()
        # shuffle X_bin 100 times as a way of doing a permutation test
        X_bin_shuffled = X_bin.copy()
        row_mapping = []

        # Shuffle the rows over 100 times
        for j in range(100):
            row_indices = np.arange(X_bin.shape[0])
            np.random.shuffle(row_indices)
            X_bin_shuffled = X_bin_shuffled[row_indices]

        outsideloopacclist = []
        perm_outsideloopacclist = []
        outsideloopbalacclist = []
        perm_outsideloopbalacclist = []
        for k in range(1, X.shape[1], 4):
            print(k)


            # break X and y into time bins from 1 to k
            print('at bin number', k)
            X_bin = X[:, 0:k].copy()
            y_bin = y[:, 0:k].copy()
            accuracy_list = []
            bal_ac_list = []
            perm_accuracy_list = []
            perm_bal_ac_list = []
            kfold = StratifiedKFold(n_splits=5, shuffle=True)

            for train, test in kfold.split(X_bin, y_bin):
                model_lstm = LSTMClassification(units=400, dropout=0.25, num_epochs=10)
                model_lstm_permutationtest = LSTMClassification(units=400, dropout=0.25, num_epochs=10)

                model_lstm.fit(X_bin[train], y_bin[train])
                model_lstm_permutationtest.fit(X_bin_shuffled[train], y_bin[train])

                y_pred = model_lstm.model(X_bin[test], training=False)
                y_pred = np.argmax(y_pred, axis=1)
                # y_pred_high = Xbin_high
                # y_pred_low = Xbin_low

                y_pred_permutationtest = model_lstm_permutationtest.model(X_bin_shuffled[test], training=False)
                y_pred_permutationtest = np.argmax(y_pred_permutationtest, axis=1)

                accuracy = sklearn.metrics.accuracy_score(y_bin[test].flatten(), y_pred.flatten())
                perm_accuracy = sklearn.metrics.accuracy_score(y_bin[test].flatten(), y_pred_permutationtest.flatten())
                balancedacscore = sklearn.metrics.balanced_accuracy_score(y_bin[test].flatten(), y_pred.flatten())
                perm_balancedacscore = sklearn.metrics.balanced_accuracy_score(y_bin[test].flatten(),
                                                                               y_pred_permutationtest.flatten())

                bal_ac_list.append(balancedacscore)
                perm_bal_ac_list.append(perm_balancedacscore)
                accuracy_list.append(accuracy)
                perm_accuracy_list.append(perm_accuracy)

            outsideloopacclist.append(np.mean(accuracy_list))
            perm_outsideloopacclist.append(np.mean(perm_accuracy_list))
            outsideloopbalacclist.append(np.mean(bal_ac_list))
            perm_outsideloopbalacclist.append(np.mean(perm_bal_ac_list))

        #  totalaclist.append(np.mean(outsideloopacclist))
        # totalbalaclist.append(np.mean(outsideloopbalacclist))
        # totalpermacc.append(np.mean(perm_outsideloopacclist))
        # totalpermbalacc.append(np.mean(perm_outsideloopbalacclist))

        # Update the scores dictionary
        scores['cluster_id'].append(cluster_id)  # Assuming cluster_id is defined somewhere
        # scores['score'].append(score)  # Assuming score is defined somewhere
        scores['lstm_score'].append(np.mean(outsideloopacclist[-1]))
        scores['lstm_balanced_avg'].append(np.mean(outsideloopbalacclist[-1]))
        # scores['bootScore'].append(bootScore)  # Assuming bootScore is defined somewhere
        scores['lstm_accuracylist'].append(outsideloopacclist)
        scores['lstm_balancedaccuracylist'].append(outsideloopbalacclist)
        scores['perm_bal_ac'].append(np.mean(perm_outsideloopbalacclist))
        scores['perm_ac'].append(np.mean(perm_outsideloopacclist))
        scores['brainarea'].append(brain_area[i])
        scores['cm'].append(len(unique_trials_targ) + len(
            unique_trials_probe))  # Assuming unique_trials_targ and unique_trials_probe are defined somewhere

    return scores






def run_classification(datapath, ferretid, ferretid_fancy='F1902_Eclair', clust_ids=[], brain_area=[]):
    try:
        with open(datapath / 'new_blocks.pkl', 'rb') as f:
            blocks = pickle.load(f)
    except:
        print('NO NEW BLOCKS FOUND')
        try:
            with open(datapath / 'blocks.pkl', 'rb') as f:
                blocks = pickle.load(f)
        except:
            return

    scores = {}
    probewords_list =[ (2,2), (3,3), (4,4),(5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11,11), (12,12), (13,13), (14,14)]

    recname = str(datapath).split('\\')[-4]
    stream_used = str(datapath).split('\\')[-3]
    stream_used = stream_used[-4:]



    tarDir = Path(
        f'G:/results_decodingovertime_24112023/{ferretid_fancy}/{recname}/{stream_used}/')
    saveDir = tarDir
    saveDir.mkdir(exist_ok=True, parents=True)
    for probeword in probewords_list:
        print('now starting to decode the probeword:')
        print(probeword)
        for talker in [1]:
            if talker == 1:
                window = [0, 0.6]
            else:
                window = [0, 0.5]
            print(f'talker {talker}')

            scores[f'talker{talker}'] = {}

            scores[f'talker{talker}']['target_vs_probe'] = {}

            scores[f'talker{talker}']['target_vs_probe']['pitchshift'] = target_vs_probe(blocks, talker=talker,
                                                                                           probewords=probeword,
                                                                                           pitchshift=True,
                                                                                           window=window, clust_ids=clust_ids, brain_area=brain_area)


            np.save(saveDir / f'scores_2022_{ferretid}_{probeword[0]}_{ferretid}_probe_pitchshift_bs.npy',
                    scores)



def main():
    datapath_big = Path(f'D:/ms4output_16102023/F1604_Squinty/')
    ferret_id_fancy = datapath_big.parts[-1]
    ferret_id = ferret_id_fancy.split('_')[1]
    ferret_id = ferret_id.lower()
    datapaths = [x for x in datapath_big.glob('**/mountainsort4/phy//') if x.is_dir()]


    for datapath in datapaths:
        stream = str(datapath).split('\\')[-3]
        stream = stream[-4:]
        print(stream)
        folder = str(datapath).split('\\')[-3]
        with open(datapath / 'new_blocks.pkl', 'rb') as f:
            new_blocks = pickle.load(f)

        high_units = pd.read_csv(f'G:/neural_chapter/figures/unit_ids_trained_topgenindex_{ferret_id_fancy}.csv')
        # remove trailing steam
        rec_name = folder[:-5]
        # find the unique string

        # remove the repeating substring

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
        high_units = high_units[(high_units['rec_name'] == rec_name) & (high_units['stream'] == stream)]
        clust_ids = high_units['ID'].to_list()
        brain_area = high_units['BrainArea'].to_list()

        if clust_ids == []:
            print('no units found')
            continue
        print('now starting to look at the datapath'+ str(datapath))
        run_classification(datapath, ferret_id, ferretid_fancy=ferret_id_fancy, clust_ids=clust_ids, brain_area=brain_area)


if __name__ == '__main__':
    main()
