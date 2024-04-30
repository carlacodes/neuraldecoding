from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, LeavePOut
from tqdm import tqdm
from keras import backend as K
import sklearn
from analysisscriptsmodcg.cv_loocv_lstmdecoder.helpers.neural_analysis_helpers import get_word_aligned_raster_zola_cruella
import numpy as np
import pickle
# Import decoder functions
from Neural_Decoding.decoders import LSTMDecoder, LSTMClassification


def target_vs_probe(blocks, talker=1, probewords=[20, 22], probeword_compare=[22, 20], pitchshift=True,
                    window=[0, 0.5]):
    if talker == 1:
        probeword = probewords[0]
        probeword_compare = probeword_compare[0]
    else:
        probeword = probewords[1]
        probeword_compare = probeword_compare[1]
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
              'compare_pitch_ac': [],
              'compare_pitch_bal_ac': [],

              }

    for cluster_id in tqdm(clust_ids):
        print('cluster_id:')
        print(cluster_id)
        unique_trials_probe = []
        unique_trials_targ = []
        # try:

        raster_target, raster_targ_check = get_word_aligned_raster_zola_cruella(blocks, cluster_id, word=1,
                                                                                pitchshift=pitchshift,
                                                                                correctresp=True,
                                                                                df_filter=['No Level Cue'],
                                                                                talker=talker)
        raster_target = raster_target.reshape(raster_target.shape[0], )
        if len(raster_target) == 0:
            print('no relevant spikes for this target word:' + str(probeword) + ' and cluster: ' + str(cluster_id))
            continue

        probe_filter = ['No Level Cue']  # , 'Non Correction Trials']
        # try:
        raster_probe, raster_probe_check = get_word_aligned_raster_zola_cruella(blocks, cluster_id, word=probeword,
                                                                                pitchshift=pitchshift,
                                                                                correctresp=True,
                                                                                df_filter=['No Level Cue'],
                                                                                talker=talker)
        # raster_probe = raster_probe[raster_probe['talker'] == talker]
        raster_probe = raster_probe.reshape(raster_probe.shape[0], )

        raster_probe_test, raster_probe_check_test = get_word_aligned_raster_zola_cruella(blocks, cluster_id,
                                                                                          word=probeword_compare,
                                                                                          pitchshift=pitchshift,
                                                                                          correctresp=True,
                                                                                          df_filter=['No Level Cue'],
                                                                                          talker=talker)
        # raster_probe = raster_probe[raster_probe['talker'] == talker]
        raster_probe_test = raster_probe_test.reshape(raster_probe_test.shape[0], )
        raster_probe['trial_num'] = raster_probe['trial_num'] + np.max(raster_target['trial_num'])
        raster_probe_test['trial_num'] = raster_probe_test['trial_num'] + np.max(raster_target['trial_num'])

        if len(raster_probe) == 0 or len(raster_probe_test) == 0:
            print('no relevant spikes for this probe word:' + str(probeword) + ' and cluster: ' + str(cluster_id))
            continue

        # except:
        #     print('No relevant probe firing')
        #     cluster_id_droplist = np.append(cluster_id_droplist, cluster_id)

        #     continue
        # sample with replacement from target trials and probe trials to boostrap scores and so distributions are equal

        bins = np.arange(window[0], window[1], binsize)

        unique_trials_targ = np.unique(raster_target['trial_num'])
        unique_trials_probe = np.unique(raster_probe['trial_num'])
        unique_trials_probe_test = np.unique(raster_probe_test['trial_num'])
        raster_targ_reshaped = np.empty([len(unique_trials_targ), len(bins) - 1])
        raster_probe_reshaped = np.empty([len(unique_trials_probe), len(bins) - 1])
        raster_probe_reshaped_test = np.empty([len(unique_trials_probe_test), len(bins) - 1])


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
        count = 0
        for trial in (unique_trials_probe_test):
            raster_probe_reshaped_test[count, :] = \
                np.histogram(raster_probe_test['spike_time'][raster_probe_test['trial_num'] == trial], bins=bins,
                             range=(window[0], window[1]))[0]
            count += 1

        if (len(raster_targ_reshaped)) < 5 or (len(raster_probe_reshaped)) < 5 or (len(raster_probe_reshaped_test)) < 5:
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
        if len(raster_probe_reshaped_test) < 15:
            # upsample to 15 trials
            raster_probe_reshaped_test = raster_probe_reshaped_test[
                                         np.random.choice(len(raster_probe_reshaped_test), 15, replace=True), :]

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

        if len(raster_targ_reshaped) >= len(raster_probe_reshaped_test) * 2:
            print('raster of distractor at least a 1/2 of target raster')
            # upsample the probe raster
            raster_probe_reshaped_test = raster_probe_reshaped_test[
                                         np.random.choice(len(raster_probe_reshaped_test), len(raster_targ_reshaped),
                                                          replace=True), :]
        elif len(raster_probe_reshaped_test) >= len(raster_targ_reshaped) * 2:
            print('raster of target at least a 1/2 of probe raster')
            # upsample the target raster
            raster_targ_reshaped = raster_targ_reshaped[
                                   np.random.choice(len(raster_targ_reshaped), len(raster_probe_reshaped_test),
                                                    replace=True), :]

        print('now length of raster_probe is:')
        print(len(raster_probe_reshaped))

        stim0 = np.full(len(raster_targ_reshaped), 0)  # 0 = target word
        stim1 = np.full(len(raster_probe_reshaped), 1)  # 1 = probe word
        stim1_compare = np.full(len(raster_probe_reshaped_test), 1)  # 1 = probe word

        stim_lstm = np.concatenate((stim0, stim1))
        stim_lstm_compare = np.concatenate((stim0, stim1_compare))

        raster_lstm = np.concatenate((raster_targ_reshaped, raster_probe_reshaped))
        raster_lstm_compare = np.concatenate((raster_targ_reshaped, raster_probe_reshaped_test))

        raster_reshaped = np.reshape(raster_lstm, (np.size(raster_lstm, 0), np.size(raster_lstm, 1), 1)).astype(
            'float32')
        raster_reshaped_compare = np.reshape(raster_lstm_compare, (
        np.size(raster_lstm_compare, 0), np.size(raster_lstm_compare, 1), 1)).astype('float32')
        stim_reshaped = np.reshape(stim_lstm, (len(stim_lstm), 1)).astype('float32')
        stim_reshaped_compare = np.reshape(stim_lstm_compare, (len(stim_lstm_compare), 1)).astype('float32')
        X = raster_reshaped
        y = stim_reshaped
        X_compare = raster_reshaped_compare
        y_compare = stim_reshaped_compare

        if len(X) == 0 or len(X_compare) == 0:
            print('no relevant trials for this probe word:' + str(probeword) + ' and cluster: ' + str(cluster_id))
            continue
        # except Exception as e:
        #     print('skipping this cluster, Exception:{e}'.format(e=e))
        #     continue
        K.clear_session()

        totalaclist = []
        totalbalaclist = []

        X_bin = X[:, :].copy()
        y_bin = y[:, :].copy()
        # shuffle X_bin 100 times as a way of doing a permutation test
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
        compare_pitch_outsideloopacclist = []
        compare_pitch_outsideloopbalacclist = []
        print('before lstm loop')
        for i in range(1):
            print('running first iteration of i: ' + str(i))
            accuracy_list = []
            bal_ac_list = []
            perm_accuracy_list = []
            compare_probeword_list = []
            perm_bal_ac_list = []
            compare_pitch_bal_ac_list = []
            compare_probeword_balac_list = []
            kfold = StratifiedKFold(n_splits=5, shuffle=True)
            print('iteration', i)

            for train, test in kfold.split(X_bin, y_bin):
                model_lstm = LSTMClassification(units=400, dropout=0.25, num_epochs=10)
                model_lstm_permutationtest = LSTMClassification(units=400, dropout=0.25, num_epochs=10)

                model_lstm.fit(X_bin[train], y_bin[train])
                model_lstm_permutationtest.fit(X_bin_shuffled[train], y_bin[train])

                y_pred = model_lstm.model(X_bin[test], training=False)
                y_pred = np.argmax(y_pred, axis=1)

                y_pred_compare = model_lstm.model(X_compare, training=False)
                y_pred_compare = np.argmax(y_pred_compare, axis=1)

                accuracy_compare = sklearn.metrics.accuracy_score(y_compare.flatten(), y_pred_compare.flatten())
                balancedacscore_compare = sklearn.metrics.balanced_accuracy_score(y_compare.flatten(),
                                                                                  y_pred_compare.flatten())

                y_pred_permutationtest = model_lstm_permutationtest.model(X_bin_shuffled[test], training=False)
                y_pred_permutationtest = np.argmax(y_pred_permutationtest, axis=1)

                accuracy = sklearn.metrics.accuracy_score(y_bin[test].flatten(), y_pred.flatten())
                perm_accuracy = sklearn.metrics.accuracy_score(y_bin[test].flatten(), y_pred_permutationtest.flatten())

                balancedacscore = sklearn.metrics.balanced_accuracy_score(y_bin[test].flatten(), y_pred.flatten())
                perm_balancedacscore = sklearn.metrics.balanced_accuracy_score(y_bin[test].flatten(),
                                                                               y_pred_permutationtest.flatten())

                bal_ac_list.append(balancedacscore)
                perm_bal_ac_list.append(perm_balancedacscore)
                compare_probeword_balac_list.append(balancedacscore_compare)

                accuracy_list.append(accuracy)
                perm_accuracy_list.append(perm_accuracy)
                compare_probeword_list.append(accuracy_compare)

            outsideloopacclist.append(np.mean(accuracy_list))
            perm_outsideloopacclist.append(np.mean(perm_accuracy_list))
            compare_pitch_outsideloopacclist.append(np.mean(compare_probeword_list))

            outsideloopbalacclist.append(np.mean(bal_ac_list))
            perm_outsideloopbalacclist.append(np.mean(perm_bal_ac_list))
            compare_pitch_outsideloopbalacclist.append(np.mean(compare_pitch_bal_ac_list))

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
        scores['compare_pitch_bal_ac'].append(np.mean(compare_pitch_outsideloopbalacclist))
        scores['compare_pitch_ac'].append(np.mean(compare_pitch_outsideloopacclist))
        scores['cm'].append(len(unique_trials_targ) + len(
            unique_trials_probe))  # Assuming unique_trials_targ and unique_trials_probe are defined somewhere

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
    probewords_list = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
    probewords_list_compare = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
    recname = str(datapath).split('\\')[-4]
    stream_used = str(datapath).split('\\')[-3]
    stream_used = stream_used[-4:]

    tarDir = Path(
        f'/home/zceccgr/Scratch/zceccgr/lstmdecodingproject/leavepoutcrossvalidationlstmdecoder/results_compareonprobeword_22042024/{ferretid_fancy}/{recname}/{stream_used}/')
    saveDir = tarDir
    saveDir.mkdir(exist_ok=True, parents=True)
    if tarDir.exists():
        # check how many probewords completed
        files = tarDir.glob('*score*.npy')
        files = [x for x in files]
        # check the numbers in the files
        numbers_compare = [int(str(x).split('_')[-6]) for x in files]
        print(numbers_compare)
        numbers_front = [int(str(x).split('_')[-9]) for x in files]
        print(f'numbers indentified: {numbers_front}')
        # create a set of tuples from numbers_front and numbers_compare
        completed_pairs = set(zip(numbers_front, numbers_compare))
        # check which probewords are missing
        probewords_list_compare_copy = [x for x in probewords_list_compare if x[0] not in numbers_compare]
        # remove the completed pairs from the probewords_list
        probewords_list = [x for x in probewords_list if tuple(x[:2]) not in completed_pairs]
        if probewords_list_compare_copy == []:
            print(f'directory already exists AND ALL comparative PROBEWORDS COMPLETED')
        if probewords_list == []:
            print(f'directory already exists AND ALL PROBEWORDS COMPLETED')
            return

    for probeword in probewords_list:
        print('now starting to decode the probeword:')
        print(probeword)
        for probeword_compare in probewords_list_compare:
            print('going to compare this model trained on the target vs probe with:')

            for talker in [1]:
                if talker == 1:
                    window = [0, 0.6]
                else:
                    window = [0, 0.5]
                print(f'talker {talker}')

                scores[f'talker{talker}'] = {}

                scores[f'talker{talker}']['target_vs_probe_control'] = {}

                scores[f'talker{talker}']['target_vs_probe_control'][
                    f'{probeword[0]}_testedon_{probeword_compare[0]}'] = target_vs_probe(blocks, talker=talker,
                                                                                         probewords=probeword,
                                                                                         probeword_compare=probeword_compare,
                                                                                         pitchshift=False,
                                                                                         window=window)

                np.save(
                    saveDir / f'scores_2022_{ferretid}_{probeword[0]}_compared_with_{probeword_compare[0]}_talker_{talker}_{ferretid}_probe_bs.npy',
                    scores)


def main():
    datapath_big = Path(f'D:\ms4output_16102023\F1815_Cruella/')
    ferret_id_fancy = datapath_big.parts[-1]
    ferret_id = ferret_id_fancy.split('_')[1]
    ferret_id = ferret_id.lower()
    datapaths = [x for x in datapath_big.glob('**/mountainsort4/phy//') if x.is_dir()]

    for datapath in datapaths:
        print('now starting to look at the datapath' + str(datapath))
        run_classification(datapath, ferret_id, ferretid_fancy=ferret_id_fancy)


if __name__ == '__main__':
    main()
