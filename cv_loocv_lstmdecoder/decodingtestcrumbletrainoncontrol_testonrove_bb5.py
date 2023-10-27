import pickle
from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, LeavePOut
from tqdm import tqdm
from keras import backend as K

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
from instruments.helpers.neural_analysis_helpers import get_word_aligned_raster_inter_by_pitch
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


def generate_talker_raster_for_lstm(blocks, cluster_id, window, binsize, talker_choice=1, probeword=2):
    raster_target, raster_targ_compare = get_word_aligned_raster_inter_by_pitch(blocks, cluster_id, word=1,
                                                                                talker_choice=talker_choice,
                                                                                correctresp=False,
                                                                                df_filter=['No Level Cue'])
    raster_target = raster_target.reshape(raster_target.shape[0], )
    if len(raster_target) == 0:
        print('no relevant spikes for this target word:' + str(probeword) + ' and cluster: ' + str(cluster_id))
        return


    raster_probe, raster_probe_compare = get_word_aligned_raster_inter_by_pitch(blocks, cluster_id, word=probeword,
                                                                                talker_choice=talker_choice,
                                                                                correctresp=False,
                                                                                df_filter=['No Level Cue'])
    raster_probe = raster_probe.reshape(raster_probe.shape[0], )

    raster_probe['trial_num'] = raster_probe['trial_num'] + np.max(raster_target['trial_num'])
    if len(raster_probe) == 0:
        print('no relevant spikes for this probe word:' + str(probeword) + ' and cluster: ' + str(cluster_id))
        return

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

    if (len(raster_targ_reshaped)) < 15 or (len(raster_probe_reshaped)) < 15:
        print('less than 15 trials for the target or distractor, CV would be overinflated, skipping')
        return

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

    stim_lstm = np.concatenate((stim0, stim1))

    raster = np.concatenate((raster_target, raster_probe))
    raster_lstm = np.concatenate((raster_targ_reshaped, raster_probe_reshaped))
    raster_reshaped = np.reshape(raster_lstm, (np.size(raster_lstm, 0), np.size(raster_lstm, 1), 1)).astype(
        'float32')
    stim_reshaped = np.reshape(stim_lstm, (len(stim_lstm), 1)).astype('float32')
    X = raster_reshaped
    y = stim_reshaped
    return X, y, unique_trials_probe, unique_trials_targ
def target_vs_probe(blocks, talker=1, probewords=[20, 22], pitchshift=True, window=[0, 0.5]):
    if talker == 1:
        probeword = probewords[0]
    else:
        probeword = probewords[1]
    binsize = 0.01

    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']

    scores = {'cluster_id': [],
              'cm': [],
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
              'high_pitch_ac': [],
              'high_pitch_bal_ac': [],
              'low_pitch_ac': [],
              'low_pitch_bal_ac': [],
              }




    cluster_id_droplist = np.empty([])
    for cluster_id in tqdm(clust_ids):
        print('cluster_id:')
        print(cluster_id)
        unique_trials_probe = []
        unique_trials_targ = []
        try:
            if talker == 1:
                X,y, unique_trials_probe, unique_trials_targ = generate_talker_raster_for_lstm(blocks, cluster_id, window, binsize, probeword = probeword, talker_choice=1)
                X_high, y_high, unique_trials_probe_high, unique_trials_targ_high = generate_talker_raster_for_lstm(blocks, cluster_id, window, binsize, probeword = probeword, talker_choice=3)
                X_low, y_low, unique_trials_probe_low, unique_trials_targ_low = generate_talker_raster_for_lstm(blocks, cluster_id, window, binsize, probeword = probeword, talker_choice=5)
            elif talker == 2:
                X, y, unique_trials_probe, unique_trials_targ = generate_talker_raster_for_lstm(blocks, cluster_id,
                                                                                                window, binsize,
                                                                                                probeword=probeword,
                                                                                                talker_choice=2)
                X_high, y_high, unique_trials_probe_high, unique_trials_targ_high = generate_talker_raster_for_lstm(
                    blocks, cluster_id, window, binsize, probeword=probeword, talker_choice=8)
                X_low, y_low, unique_trials_probe_low, unique_trials_targ_low = generate_talker_raster_for_lstm(blocks,
                                                                                                                cluster_id,
                                                                                                                window,
                                                                                                                binsize,
                                                                                                                probeword=probeword,
                                                                                                              talker_choice=13)
        except Exception as e:
            # print('skipping this cluster, Exception:{e}'.format(e=e))
            continue



        K.clear_session()

        totalaclist = []
        totalbalaclist = []

        X_bin = X[:, :].copy()
        y_bin = y[:, :].copy()
        #shuffle X_bin 100 times as a way of doing a permutation test
        X_bin_shuffled = X_bin.copy()

        # Shuffle the rows over 100 times
        for i in range(100):
            row_indices = np.arange(X_bin.shape[0])
            np.random.shuffle(row_indices)
            X_bin_shuffled = X_bin_shuffled[row_indices]

#

        outsideloopacclist = []
        perm_outsideloopacclist = []
        outsideloopbalacclist = []
        perm_outsideloopbalacclist = []
        high_pitch_outsideloopacclist = []
        low_pitch_outsideloopacclist = []
        high_pitch_outsideloopbalacclist = []
        low_pitch_outsideloopbalacclist = []

        for i in range(0, 1):
            accuracy_list = []
            bal_ac_list = []
            perm_accuracy_list = []
            high_pitch_ac_list = []
            low_pitch_ac_list = []
            perm_bal_ac_list = []
            high_pitch_bal_ac_list = []
            low_pitch_bal_ac_list = []
            kfold = StratifiedKFold(n_splits=3, shuffle=True)
            print('iteration', i)

            for train, test in kfold.split(X_bin, y_bin):
                model_lstm = LSTMClassification(units=400, dropout=0.25, num_epochs=10)
                model_lstm_permutationtest = LSTMClassification(units=400, dropout=0.25, num_epochs=10)

                model_lstm.fit(X_bin[train], y_bin[train])
                model_lstm_permutationtest.fit(X_bin_shuffled[train], y_bin[train])

                y_pred = model_lstm.model(X_bin[test], training=False)
                y_pred = np.argmax(y_pred, axis=1)

                y_pred_high = model_lstm.model(X_high, training=False)
                y_pred_high = np.argmax(y_pred_high, axis=1)

                y_pred_low = model_lstm.model(X_low, training=False)
                y_pred_low = np.argmax(y_pred_low, axis=1)


                accuracy_high = sklearn.metrics.accuracy_score(y_high.flatten(), y_pred_high.flatten())
                balancedacscore_high = sklearn.metrics.balanced_accuracy_score(y_high.flatten(), y_pred_high.flatten())

                accuracy_low = sklearn.metrics.accuracy_score(y_low.flatten(), y_pred_low.flatten())
                balancedacscore_low = sklearn.metrics.balanced_accuracy_score(y_low.flatten(), y_pred_low.flatten())

                y_pred_permutationtest = model_lstm_permutationtest.model(X_bin_shuffled[test], training=False)
                y_pred_permutationtest = np.argmax(y_pred_permutationtest, axis=1)


                accuracy = sklearn.metrics.accuracy_score(y_bin[test].flatten(), y_pred.flatten())
                perm_accuracy = sklearn.metrics.accuracy_score(y_bin[test].flatten(), y_pred_permutationtest.flatten())
                balancedacscore = sklearn.metrics.balanced_accuracy_score(y_bin[test].flatten(), y_pred.flatten())
                perm_balancedacscore = sklearn.metrics.balanced_accuracy_score(y_bin[test].flatten(), y_pred_permutationtest.flatten())

                bal_ac_list.append(balancedacscore)
                perm_bal_ac_list.append(perm_balancedacscore)
                high_pitch_bal_ac_list.append(balancedacscore_high)
                low_pitch_bal_ac_list.append(balancedacscore_low)

                accuracy_list.append(accuracy)
                perm_accuracy_list.append(perm_accuracy)
                high_pitch_ac_list.append(accuracy_high)
                low_pitch_ac_list.append(accuracy_low)

            outsideloopacclist.append(np.mean(accuracy_list))
            perm_outsideloopacclist.append(np.mean(perm_accuracy_list))
            high_pitch_outsideloopacclist.append(np.mean(high_pitch_ac_list))
            low_pitch_outsideloopacclist.append(np.mean(low_pitch_ac_list))


            outsideloopbalacclist.append(np.mean(bal_ac_list))
            perm_outsideloopbalacclist.append(np.mean(perm_bal_ac_list))
            high_pitch_outsideloopbalacclist.append(np.mean(high_pitch_bal_ac_list))
            low_pitch_outsideloopbalacclist.append(np.mean(low_pitch_bal_ac_list))


        totalaclist.append(np.mean(outsideloopacclist))
        totalbalaclist.append(np.mean(outsideloopbalacclist))
        # Update the scores dictionary
        scores['cluster_id'].append(cluster_id)  # Assuming cluster_id is defined somewhere
        scores['lstm_score'].append(np.mean(totalaclist))
        scores['lstm_balanced_avg'].append(np.mean(totalbalaclist))
        scores['lstm_accuracylist'].append(totalaclist)
        scores['lstm_avg'].append(np.mean(totalaclist))
        scores['lstm_balancedaccuracylist'].append(totalbalaclist)
        scores['perm_bal_ac'].append(np.mean(perm_outsideloopbalacclist))
        scores['perm_ac'].append(np.mean(perm_outsideloopacclist))
        scores['high_pitch_bal_ac'].append(np.mean(high_pitch_outsideloopbalacclist))
        scores['high_pitch_ac'].append(np.mean(high_pitch_outsideloopacclist))
        scores['low_pitch_bal_ac'].append(np.mean(low_pitch_outsideloopbalacclist))
        scores['low_pitch_ac'].append(np.mean(low_pitch_outsideloopacclist))
        scores['cm'].append(len(unique_trials_targ) + len(unique_trials_probe))  # Assuming unique_trials_targ and unique_trials_probe are defined somewhere

    return scores

def run_classification(dir, datapath, ferretid):
    with open(datapath / 'new_blocks.pkl', 'rb') as f:
        blocks = pickle.load(f)

    scores = {}
    probewords_list = [(5, 6), (42, 49), (32, 38), (2, 2), (20, 22), ]
    recname = str(datapath).split('\\')[-4]
    stream_used = str(datapath).split('\\')[-3]
    stream_used = stream_used[-4:]



    tarDir = Path(
        f'/zceccgr/lstmdecodingproject/leavepoutcrossvalidationlstmdecoder/results_testonrove_27102023/F1901_Crumble/{recname}/{stream_used}/')
    saveDir = tarDir
    saveDir.mkdir(exist_ok=True, parents=True)
    for probeword in probewords_list:
        print('now starting')
        print(probeword)
        for talker in [1,2]:
            if talker == 1:
                window = [0, 0.6]
            else:
                window = [0, 0.5]
            print(f'talker {talker}')

            scores[f'talker{talker}'] = {}

            scores[f'talker{talker}']['target_vs_probe'] = {}

            scores[f'talker{talker}']['target_vs_probe']['nopitchshiftvspitchshift'] = target_vs_probe(blocks, talker=talker,
                                                                                           probewords=probeword,
                                                                                           pitchshift=False,
                                                                                           window=window)
            # scores[f'talker{talker}']['target_vs_probe']['pitchshift'] = target_vs_probe(blocks, talker=talker,
            #                                                                              probewords=probeword,
            #                                                                              pitchshift=True, window=window)

            np.save(saveDir / f'scores_{dir}_{probeword[0]}_{ferretid}_probe_bs.npy',
                    scores)

        # fname = 'scores_' + dir + f'_probe_earlylate_left_right_win_bs_{binsize}'


def main():
    directories = [
        'crumble_2022']  # , 'Trifle_July_2022']/home/zceccgr/Scratch/zceccgr/ms4output/F1702_Zola/spkenvresults04102022allrowsbut4th

    datapath = Path(
        f'E:\ms4output2\F1901_Crumble\BB2BB3_crumble_29092023_2\BB2BB3_crumble_29092023_BB2BB3_crumble_29092023_BB_3\mountainsort4\phy/')
    ferretid = 'crumble'

    for dir in directories:
        run_classification(dir, datapath, ferretid)


if __name__ == '__main__':
    main()
