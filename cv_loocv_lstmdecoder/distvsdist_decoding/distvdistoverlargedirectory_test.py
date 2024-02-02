from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, LeavePOut
from tqdm import tqdm
from keras import backend as K
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import sklearn
from instruments.helpers.util import simple_xy_axes, set_font_axes
from helpers.neural_analysis_helpers import get_word_aligned_raster_zola_cruella
from instruments.helpers.euclidean_classification_minimal_function import classify_sweeps
# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
# Import decoder functions
from Neural_Decoding.decoders import LSTMDecoder, LSTMClassification


def generate_talker_raster_for_lstm(blocks, cluster_id, window, binsize, talker_choice=1, probeword=2, control_pitch = False):
    raster_target, raster_targ_compare = get_word_aligned_raster_inter_by_pitch(blocks, cluster_id, word=1,
                                                                                talker_choice=talker_choice,
                                                                                correctresp=True,
                                                                                df_filter=['No Level Cue'])
    raster_target = raster_target.reshape(raster_target.shape[0], )
    if len(raster_target) == 0:
        print('no relevant spikes for this target word:' + str(probeword) + ' and cluster: ' + str(cluster_id))
        raster_probe_reshaped = []
        raster_targ_reshaped = []
        unique_trials_targ = []
        unique_trials_probe = []


        return raster_probe_reshaped, raster_targ_reshaped, unique_trials_probe, unique_trials_targ


    raster_probe, raster_probe_compare = get_word_aligned_raster_inter_by_pitch(blocks, cluster_id, word=probeword,
                                                                                talker_choice=talker_choice,
                                                                                correctresp=True,
                                                                                df_filter=['No Level Cue'])
    raster_probe = raster_probe.reshape(raster_probe.shape[0], )
    if len(raster_probe) == 0:
        print('no relevant spikes for this probe word:' + str(probeword) + ' and cluster: ' + str(cluster_id))
        raster_probe_reshaped = []
        raster_targ_reshaped = []
        unique_trials_targ = []
        unique_trials_probe = []
        return raster_probe_reshaped, raster_targ_reshaped, unique_trials_probe, unique_trials_targ

    raster_probe['trial_num'] = raster_probe['trial_num'] + np.max(raster_target['trial_num'])


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
    if control_pitch == True:
        if (len(raster_targ_reshaped)) < 5 or (len(raster_probe_reshaped)) < 5:
            print('less than 15 trials for the target or distractor, CV would be overinflated, skipping')
            raster_probe_reshaped = []
            raster_targ_reshaped = []
            unique_trials_targ = []
            unique_trials_probe = []


            return raster_probe_reshaped, raster_targ_reshaped, unique_trials_probe, unique_trials_targ
    if len(raster_targ_reshaped) < 15:
        #upsample to 15 trials
        raster_targ_reshaped = raster_targ_reshaped[np.random.choice(len(raster_targ_reshaped), 15, replace=True), :]
    if len(raster_probe_reshaped) < 15:
        #upsample to 15 trials
        raster_probe_reshaped = raster_probe_reshaped[np.random.choice(len(raster_probe_reshaped), 15, replace=True), :]

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
def target_vs_probe(blocks, talker=1, probeword_1 = [20, 22], probeword_2 = [20,22], pitchshift=True, window=[0, 0.5]):
    if talker == 1:
        probeword_1 = probeword_1[0]
        probeword_2 = probeword_2[0]
        talker_text = 'female'
    else:
        probeword_1 = probeword_1[1]
        probeword_2 = probeword_2[1]
        talker_text = 'male'
    binsize = 0.01
    # window = [0, 0.6]

    epochs = ['Early', 'Late']
    epoch_threshold = 1.5
    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']

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

    # scores['cluster_id'].append(cluster_id)
    # scores['score'].append(score)
    # scores['lstm_score'].append(np.mean(totalaclist))
    # scores['lstm_balanced_avg'].append(np.mean(bal_ac_list))
    # scores['bootScore'].append(bootScore)
    # scores['lstm_accuracylist'].append(accuracy_list)
    # scores['lstm_balancedaccuracylist'].append(bal_ac_list)
    # scores['cm'].append(len(unique_trials_probe1) + len(unique_trials_probe2))

    cluster_id_droplist = np.empty([])
    for cluster_id in tqdm(clust_ids):
        print('cluster_id:')
        print(cluster_id)

        target_filter = ['Target trials', 'No Level Cue']  # , 'Non Correction Trials']

        # try:
        raster_probeword_first, raster_probeword_first_compare = get_word_aligned_raster_zola_cruella(blocks, cluster_id, word=probeword_1,
                                                                                  pitchshift=pitchshift,
                                                                                  correctresp=True,
                                                                                  df_filter=['No Level Cue'],
                                                                                  talker=talker_text)
        raster_probeword_first = raster_probeword_first.reshape(raster_probeword_first.shape[0], )
        if len(raster_probeword_first) == 0:
            print('no relevant spikes for this target word:' + str(probeword_1) + ' and cluster: ' + str(cluster_id))
            continue


        raster_probe_second, raster_probe_second_compare = get_word_aligned_raster_zola_cruella(blocks, cluster_id, word=probeword_2,
                                                                                  pitchshift=pitchshift,
                                                                                  correctresp=True,
                                                                                  df_filter=['No Level Cue'],
                                                                                  talker=talker_text)
        raster_probe_second = raster_probe_second.reshape(raster_probe_second.shape[0], )

        raster_probe_second['trial_num'] = raster_probe_second['trial_num'] + np.max(raster_probeword_first['trial_num'])
        if len(raster_probe_second) == 0:
            print('no relevant spikes for this probe word:' + str(probeword_2) + ' and cluster: ' + str(cluster_id))
            continue
        # except:
        #     print('No relevant probe firing')
        #     cluster_id_droplist = np.append(cluster_id_droplist, cluster_id)

        #     continue
        # sample with replacement from target trials and probe trials to boostrap scores and so distributions are equal
        lengthofraster = np.sum(len(raster_probeword_first['spike_time']) + len(raster_probe_second['spike_time']))
        raster_probe1_reshaped = np.empty([])
        raster_probe2_reshaped = np.empty([])
        bins = np.arange(window[0], window[1], binsize)

        lengthoftargraster = len(raster_probeword_first['spike_time'])
        lengthofproberaster = len(raster_probe_second['spike_time'])

        unique_trials_probe1 = np.unique(raster_probeword_first['trial_num'])
        unique_trials_probe2 = np.unique(raster_probe_second['trial_num'])
        raster_probe1_reshaped = np.empty([len(unique_trials_probe1), len(bins) - 1])
        raster_probe2_reshaped = np.empty([len(unique_trials_probe2), len(bins) - 1])
        count = 0
        for trial in (unique_trials_probe1):
            raster_probe1_reshaped[count, :] = \
                np.histogram(raster_probeword_first['spike_time'][raster_probeword_first['trial_num'] == trial], bins=bins,
                             range=(window[0], window[1]))[0]
            count += 1
        count = 0
        for trial in (unique_trials_probe2):
            raster_probe2_reshaped[count, :] = \
                np.histogram(raster_probe_second['spike_time'][raster_probe_second['trial_num'] == trial], bins=bins,
                             range=(window[0], window[1]))[0]
            count += 1
        if (len(raster_probe1_reshaped)) < 5 or (len(raster_probe2_reshaped)) < 5:
            print('less than 5 trials for the target or distractor, CV would be overinflated, skipping')
            continue
        if len(raster_probe1_reshaped) < 15:
            # upsample to 15 trials
            raster_probe1_reshaped = raster_probe1_reshaped[np.random.choice(len(raster_probe1_reshaped), 15, replace=True),
                                   :]
        if len(raster_probe2_reshaped) < 15:
            # upsample to 15 trials
            raster_probe2_reshaped = raster_probe2_reshaped[
                                    np.random.choice(len(raster_probe2_reshaped), 15, replace=True), :]

        if len(raster_probe1_reshaped) >= len(raster_probe2_reshaped) * 2:
            print('raster of distractor at least a 1/2 of target raster')
            # upsample the probe raster
            raster_probe2_reshaped = raster_probe2_reshaped[
                                    np.random.choice(len(raster_probe2_reshaped), len(raster_probe1_reshaped),
                                                     replace=True), :]
        elif len(raster_probe2_reshaped) >= len(raster_probe1_reshaped) * 2:
            print('raster of target at least a 1/2 of probe raster')
            # upsample the target raster
            raster_probe1_reshaped = raster_probe1_reshaped[
                                   np.random.choice(len(raster_probe1_reshaped), len(raster_probe2_reshaped),
                                                    replace=True), :]

        print('now length of raster_probe_second is:')
        print(len(raster_probe2_reshaped))
        stim0 = np.full(len(raster_probeword_first), 0)  # 0 = target word
        stim1 = np.full(len(raster_probe_second), 1)  # 1 = probe word
        stim = np.concatenate((stim0, stim1))

        stim0 = np.full(len(raster_probe1_reshaped), 0)  # 0 = target word
        stim1 = np.full(len(raster_probe2_reshaped), 1)  # 1 = probe word
        if (len(stim0)) < 3 or (len(stim1)) < 3:
            print('less than 3 trials for the target or distractor, skipping')
            continue

        stim_lstm = np.concatenate((stim0, stim1))

        raster = np.concatenate((raster_probeword_first, raster_probe_second))
        raster_lstm = np.concatenate((raster_probe1_reshaped, raster_probe2_reshaped))
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
        scores['cm'].append(len(unique_trials_probe1) + len(
            unique_trials_probe2))  # Assuming unique_trials_probe1 and unique_trials_probe2 are defined somewhere

    return scores

def run_classification(datapath, ferretid, ferretid_fancy='F1902_Eclair'):
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
    probewords_list =[ (2,2), (3,3), (4,4),(5,5), (6,6), (7,7), (8,8), (9,9), (10,10)]

    recname = str(datapath).split('/')[-4]
    stream_used = str(datapath).split('/')[-3]
    stream_used = stream_used[-4:]



    tarDir = Path(
        f'/home/zceccgr/Scratch/zceccgr/lstmdecodingproject/leavepoutcrossvalidationlstmdecoder/results_testonrove_inter_28102023/{ferretid_fancy}/{recname}/{stream_used}/')
    saveDir = tarDir
    saveDir.mkdir(exist_ok=True, parents=True)
    for probeword in probewords_list:
        print('now starting to decode the probeword:')
        print(probeword)
        for talker in [1,2]:
            if talker == 1:
                window = [0, 0.6]
            else:
                window = [0, 0.5]
            print(f'talker {talker}')

            scores[f'talker{talker}'] = {}

            scores[f'talker{talker}']['target_vs_probe'] = {}
            track_list = []
            for i in probewords_list:
                for i2 in probewords_list:
                    if f'{i[0]}_{i[1]}_{i2[0]}_{i2[1]}' in track_list or f'{i2[0]}_{i2[1]}_{i[0]}_{i[1]}' in track_list:
                        continue
                    track_list.append(f'{i[0]}_{i[1]}_{i2[0]}_{i2[1]}')

                    scores[f'talker{talker}']['target_vs_probe'][f'{i[0]}_{i[1]}_{i2[0]}_{i2[1]}'] = target_vs_probe(blocks, talker=talker, probeword_1 = i,
                                                                                           probeword_2=i2,
                                                                                           pitchshift=False,
                                                                                           window=window)
                    np.save(saveDir / f'scores_{dir}_{i[0]}_vs_{i2[1]}_{ferretid}_probe_nopitchshift_bs.npy',
                            scores)



def main():

    datapath_big = Path(f'/home/zceccgr/Scratch/zceccgr/ms4output2/F1815_Cruella/')
    ferret_id_fancy = datapath_big.parts[-1]
    ferret_id = ferret_id_fancy.split('_')[1]
    ferret_id = ferret_id.lower()
    datapaths = [x for x in datapath_big.glob('**/mountainsort4/phy//') if x.is_dir()]
    print(datapaths)
    datapaths = datapaths[7:20]
    print(datapaths)

    for datapath in datapaths:
        print('now starting to look at the datapath'+ str(datapath))
        run_classification(datapath, ferret_id, ferretid_fancy=ferret_id_fancy)


if __name__ == '__main__':
    main()
