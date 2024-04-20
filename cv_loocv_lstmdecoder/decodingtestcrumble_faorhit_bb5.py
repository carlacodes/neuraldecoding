import pickle
from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, LeavePOut
from tqdm import tqdm
from keras import backend as K
import sklearn
from instruments.helpers.util import simple_xy_axes, set_font_axes
from analysisscriptsmodcg.cv_loocv_lstmdecoder.helpers.neural_analysis_helpers import *

from instruments.helpers.euclidean_classification_minimal_function import classify_sweeps
# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle
from Neural_Decoding.decoders import LSTMDecoder, LSTMClassification


def hit_vs_FA(blocks, window=[0, 0.5]):

    binsize = 0.01

    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']
    clust_ids = clust_ids[1:]

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
              'perm_bal_ac': []}

    for cluster_id in tqdm(clust_ids):
        print('cluster_id:')
        print(cluster_id)

        # try:
        raster_hit= get_before_word_raster_zola_cruella(blocks, cluster_id, word=1,
                                                                                  
                                                                                  corresp_hit=True,
                                                                                  df_filter=['No Level Cue'])
        raster_hit = raster_hit.reshape(raster_hit.shape[0], )
        if len(raster_hit) == 0:
            print('no relevant spikes for cluster:' + str(cluster_id))
            continue



        raster_FA = get_before_word_raster_zola_cruella(blocks, cluster_id, word=1,

                                                                                  corresp_hit=False,
                                                                                  df_filter=['No Level Cue'])
        # raster_FA = raster_FA[raster_FA['talker'] == talker]
        raster_FA= raster_FA.reshape(raster_FA.shape[0], )

        raster_FA['trial_num'] = raster_FA['trial_num'] + np.max(raster_hit['trial_num'])
        if len(raster_FA) == 0:
            print('no relevant spikes :'  ' for cluster: ' + str(cluster_id))
            continue
        

        bins = np.arange(window[0], window[1], binsize)

        unique_trials_targ = np.unique(raster_hit['trial_num'])
        unique_trials_probe = np.unique(raster_FA['trial_num'])
        raster_targ_reshaped = np.empty([len(unique_trials_targ), len(bins) - 1])
        raster_FA_reshaped = np.empty([len(unique_trials_probe), len(bins) - 1])
        count = 0
        for trial in (unique_trials_targ):
            raster_targ_reshaped[count, :] = \
                np.histogram(raster_hit['spike_time'][raster_hit['trial_num'] == trial], bins=bins,
                             range=(window[0], window[1]))[0]
            count += 1
        count = 0
        for trial in (unique_trials_probe):
            raster_FA_reshaped[count, :] = \
                np.histogram(raster_FA['spike_time'][raster_FA['trial_num'] == trial], bins=bins,
                             range=(window[0], window[1]))[0]
            count += 1

        if (len(raster_targ_reshaped)) < 5 or (len(raster_FA_reshaped)) < 5:
            print('less than 5 trials for the target or distractor, CV would be overinflated, skipping')
            continue
        if len(raster_targ_reshaped) < 15:
            # upsample to 15 trials
            raster_targ_reshaped = raster_targ_reshaped[np.random.choice(len(raster_targ_reshaped), 15, replace=True),
                                   :]
        if len(raster_FA_reshaped) < 15:
            # upsample to 15 trials
            raster_FA_reshaped = raster_FA_reshaped[
                                    np.random.choice(len(raster_FA_reshaped), 15, replace=True), :]

        if len(raster_targ_reshaped) >= len(raster_FA_reshaped) * 2:
            print('raster of distractor at least a 1/2 of target raster')
            # upsample the probe raster
            raster_FA_reshaped = raster_FA_reshaped[
                                    np.random.choice(len(raster_FA_reshaped), len(raster_targ_reshaped),
                                                     replace=True), :]
        elif len(raster_FA_reshaped) >= len(raster_targ_reshaped) * 2:
            print('raster of target at least a 1/2 of probe raster')
            # upsample the target raster
            raster_targ_reshaped = raster_targ_reshaped[
                                   np.random.choice(len(raster_targ_reshaped), len(raster_FA_reshaped),
                                                    replace=True), :]

        print('now length of raster_FA is:')
        print(len(raster_FA_reshaped))
        stim0 = np.full(len(raster_hit), 0)  # 0 = target word
        stim1 = np.full(len(raster_FA), 1)  # 1 = probe word
        stim = np.concatenate((stim0, stim1))

        stim0 = np.full(len(raster_targ_reshaped), 0)  # 0 = target word
        stim1 = np.full(len(raster_FA_reshaped), 1)  # 1 = probe word
        if (len(stim0)) < 3 or (len(stim1)) < 3:
            print('less than 3 trials for the target or distractor, skipping')
            continue

        stim_lstm = np.concatenate((stim0, stim1))

        raster = np.concatenate((raster_hit, raster_FA))
        raster_lstm = np.concatenate((raster_targ_reshaped, raster_FA_reshaped))

        score, d, bootScore, bootClass, cm = classify_sweeps(raster, stim, binsize=binsize, window=window, genFig=False)
        # fit LSTM model to the same data

        newraster = raster.tolist()
        raster_reshaped = np.reshape(raster_lstm, (np.size(raster_lstm, 0), np.size(raster_lstm, 1), 1)).astype(
            'float32')
        stim_reshaped = np.reshape(stim_lstm, (len(stim_lstm), 1)).astype('float32')
        X = raster_reshaped
        y = stim_reshaped

        # tf.keras.backend.clear_session()
        K.clear_session()

        totalaclist = []
        totalbalaclist = []
        # create a loop that iterates from 1 to n time point and trains the model on n-1 time points
        # break X and y into time bins from 1 to k
        X_bin = X[:, :].copy()
        y_bin = y[:, :].copy()
        # shuffle X_bin 100 times as a way of doing a permutation test
        X_bin_shuffled = X_bin.copy()
        row_mapping = []

        # Shuffle the rows over 100 times
        for i in range(100):
            row_indices = np.arange(X_bin.shape[0])
            np.random.shuffle(row_indices)
            X_bin_shuffled = X_bin_shuffled[row_indices]

        outsideloopacclist = []
        perm_outsideloopacclist = []
        outsideloopbalacclist = []
        perm_outsideloopbalacclist = []
        for i in range(0, 1):
            accuracy_list = []
            bal_ac_list = []
            perm_accuracy_list = []
            perm_bal_ac_list = []
            kfold = StratifiedKFold(n_splits=5, shuffle=True)
            print('iteration', i)

            for train, test in kfold.split(X_bin, y_bin):
                model_lstm = LSTMClassification(units=400, dropout=0.25, num_epochs=10)
                model_lstm_permutationtest = LSTMClassification(units=400, dropout=0.25, num_epochs=10)

                model_lstm.fit(X_bin[train], y_bin[train])
                model_lstm_permutationtest.fit(X_bin_shuffled[train], y_bin[train])

                y_pred = model_lstm.model(X_bin[test], training=False)
                y_pred = np.argmax(y_pred, axis=1)

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

        totalaclist.append(np.mean(outsideloopacclist))
        totalbalaclist.append(np.mean(outsideloopbalacclist))

        # Update the scores dictionary
        scores['cluster_id'].append(cluster_id)  # Assuming cluster_id is defined somewhere
        scores['score'].append(score)  # Assuming score is defined somewhere
        scores['lstm_score'].append(np.mean(totalaclist))
        scores['lstm_balanced_avg'].append(np.mean(totalbalaclist))
        scores['bootScore'].append(bootScore)  # Assuming bootScore is defined somewhere
        scores['lstm_accuracylist'].append(totalaclist)
        scores['lstm_avg'].append(np.mean(totalaclist))
        scores['lstm_balancedaccuracylist'].append(totalbalaclist)
        scores['perm_bal_ac'].append(np.mean(perm_outsideloopbalacclist))
        scores['perm_ac'].append(np.mean(perm_outsideloopacclist))
        scores['cm'].append(len(unique_trials_targ) + len(
            unique_trials_probe))  # Assuming unique_trials_targ and unique_trials_probe are defined somewhere

    return scores


def run_classification(dir, datapath, ferretid):
    fname = 'new_blocks.pkl'
    with open(datapath / 'new_blocks.pkl', 'rb') as f:
        blocks = pickle.load(f)

    scores = {}


    recname = str(datapath).split('\\')[-4]
    print('recname')
    print(recname)

    tarDir = Path(
        f'F:/test_crumble/results_16092023/F1901_Crumble/{recname}/bb3/')
    saveDir = tarDir
    saveDir.mkdir(exist_ok=True, parents=True)

    # for talker in [1, 2]:
    #     if talker == 1:
    #         window = [-0.5, 0]
    #     else:
    #         window = [-0.5, 0]
    window = [-0.5, 0]

    scores['hit_vs_FA'] = {}

    scores['hit_vs_FA']= hit_vs_FA(blocks, window=window)


    np.save(saveDir / f'scores_{dir}_hit_vs_FA_{ferretid}_probe_bs.npy',
            scores)


def main():
    directories = [
        'crumble_2022']  # , 'Trifle_July_2022']/home/zceccgr/Scratch/zceccgr/ms4output/F1702_Zola/spkenvresults04102022allrowsbut4th
    datapath = Path(
        f'D:\ms4output_16102023\F1901_Crumble\BB2BB3_crumble_29092023_2\BB2BB3_crumble_29092023_BB2BB3_crumble_29092023_BB_3\mountainsort4\phy/')
    ferretid = 'crumble'

    for dir in directories:
        run_classification(dir, datapath, ferretid)


if __name__ == '__main__':
    main()
