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
from datetime import datetime
from astropy.stats import bootstrap
import sklearn
from instruments.helpers.util import simple_xy_axes, set_font_axes
from instruments.helpers.neural_analysis_helpers import get_word_aligned_raster
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


def target_vs_probe(blocks, talker=1, probewords=[20, 22], pitchshift=True):
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
              'lstm_balanced_avg': [],
              'lstm_accuracylist': [],
              'lstm_balancedaccuracylist': [],
              'history:': [],
              'lstm_avg': [],
              'history': [],
              'time_bin': [], }

    # scores['cluster_id'].append(cluster_id)
    # scores['score'].append(score)
    # scores['lstm_score'].append(np.mean(totalaclist))
    # scores['lstm_balanced_avg'].append(np.mean(bal_ac_list))
    # scores['bootScore'].append(bootScore)
    # scores['lstm_accuracylist'].append(accuracy_list)
    # scores['lstm_balancedaccuracylist'].append(bal_ac_list)
    # scores['cm'].append(len(unique_trials_targ) + len(unique_trials_probe))

    cluster_id_droplist = np.empty([])
    for cluster_id in tqdm(clust_ids):

        target_filter = ['Target trials', 'No Level Cue']  # , 'Non Correction Trials']

        try:
            raster_target = get_word_aligned_raster(blocks, cluster_id, word=1, pitchshift=pitchshift,
                                                    correctresp=True,
                                                    df_filter=[])
            raster_target = raster_target[raster_target['talker'] == int(talker)]
            if len(raster_target) == 0:
                print('no relevant spikes for this talker')
                continue
        except Exception as error:
            print('No relevant target firing')
            print(error)
            cluster_id_droplist = np.append(cluster_id_droplist, cluster_id)
            continue

        probe_filter = ['No Level Cue']  # , 'Non Correction Trials']
        try:
            raster_probe = get_word_aligned_raster(blocks, cluster_id, word=probeword, pitchshift=pitchshift,
                                                   correctresp=True,
                                                   df_filter=[])
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

        if len(raster_targ_reshaped) >= len(raster_probe_reshaped) * 2:
            print('raster of distractor at least a 1/2 of target raster')
            #upsample the probe raster
            raster_probe_reshaped = raster_probe_reshaped[np.random.choice(len(raster_probe_reshaped), len(raster_targ_reshaped), replace=False), :]
        elif len(raster_probe_reshaped) >= len(raster_targ_reshaped) * 2:
            print('raster of target at least a 1/2 of probe raster')
            # upsample the target raster
            raster_targ_reshaped = raster_targ_reshaped[np.random.choice(len(raster_targ_reshaped), len(raster_probe_reshaped), replace=False), :]


        stim0 = np.full(len(raster_target), 0)  # 0 = target word
        stim1 = np.full(len(raster_probe), 1)  # 1 = probe word
        stim = np.concatenate((stim0, stim1))

        stim0 = np.full(len(raster_targ_reshaped), 0)  # 0 = target word
        stim1 = np.full(len(raster_probe_reshaped), 1)  # 1 = probe word
        # if (len(stim0) + len(stim1)) < 5:
        #     print('less than 5 trials')
        #     continue

        stim_lstm = np.concatenate((stim0, stim1))

        raster = np.concatenate((raster_target, raster_probe))
        raster_lstm = np.concatenate((raster_targ_reshaped, raster_probe_reshaped))

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
        for k in range(1, X.shape[1], 1):
            # break X and y into time bins from 1 to k
            print('at bin number', k)
            X_bin = X[:, 0:k].copy()
            y_bin = y[:, 0:k].copy()
            outsideloopacclist = []
            outsideloopbalacclist = []
            for i in range(0, 1):
                accuracy_list = []
                bal_ac_list = []
                kfold = StratifiedKFold(n_splits=3, shuffle=True)
                print('iteration', i)

                for train, test in kfold.split(X_bin, y_bin):
                    model_lstm = LSTMClassification(units=400, dropout=0.25, num_epochs=10)
                    model_lstm.fit(X_bin[train], y_bin[train])

                    # y_pred = model_lstm.predict(X_bin[test])
                    y_pred = model_lstm.model(X_bin[test], training=False)
                    y_pred = np.argmax(y_pred, axis=1)

                    accuracy = sklearn.metrics.accuracy_score(y_bin[test].flatten(), y_pred.flatten())
                    balancedacscore = sklearn.metrics.balanced_accuracy_score(y_bin[test].flatten(), y_pred.flatten())
                    bal_ac_list.append(balancedacscore)
                    accuracy_list.append(accuracy)

                outsideloopacclist.append(np.mean(accuracy_list))
                outsideloopbalacclist.append(np.mean(bal_ac_list))

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
        scores['time_bin'] = k
        scores['cm'].append(len(unique_trials_targ) + len(
            unique_trials_probe))  # Assuming unique_trials_targ and unique_trials_probe are defined somewhere

    return scores


def probe_early_vs_late(blocks, talker=1, noise=True, df_filter=['No Level Cue'],
                        window=[0, 0.8], binsize=0.02):
    epochs = ['Early', 'Late']
    epoch_treshold = 1.5
    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']

    scores = {'cluster_id': [],
              'score': [],
              'cm': [], }
    for cluster_id in tqdm(clust_ids):
        # df_filter = ['No Level Cue'] #, 'Non Correction Trials']
        raster = get_word_aligned_raster(blocks, cluster_id, noise=noise, df_filter=df_filter)
        raster = raster[raster['talker'] == talker]

        stim = np.zeros(len(raster), dtype=np.int64)
        stim[raster['relStart'] > epoch_treshold] = 1

        score, d, bootScore, bootClass, cm = classify_sweeps(raster, stim, binsize=binsize, iterations=100,
                                                             window=window, genFig=False)
        X_train, X_test, y_train, y_test = train_test_split(raster, stim, test_size=0.33, )
        model_lstm = LSTMDecoder(units=400, dropout=0, num_epochs=5)

        # Fit model
        model_lstm.fit(X_train, y_train)

        # Get predictions
        y_valid_predicted_lstm = model_lstm.predict(X_test)

        # Get metric of fit
        R2s_lstm = get_R2(y_test, y_valid_predicted_lstm)
        print('R2s:', R2s_lstm)

        scores['cluster_id'].append(cluster_id)
        scores['score'].append(score)
        scores['cm'].append(cm)

    return scores


def save_pdf_classification(scores, saveDir, title):
    conditions = ['silence']
    for talker in [1, 2]:
        # talker = 1
        # title = f'eucl_classification_{month}_talker{talker}_win_bs_earlylateprobe_leftright_26082022'

        comparisons = [comp for comp in scores[f'talker{talker}']]
        comp = comparisons[0]
        i = 0
        clus = scores[f'talker{talker}'][comp]['silence']['cluster_id'][i]

        with PdfPages(saveDir / f'{title}_talker{talker}.pdf') as pdf:
            for i, clus in enumerate(tqdm(scores[f'talker{talker}'][comp]['silence']['cluster_id'])):
                fig, ax = plt.subplots(figsize=(10, 5))
                y = {}
                yerrmax = {}
                yerrmin = {}
                x = np.arange(len(comparisons))
                width = 0.35
                for condition in conditions:
                    y[condition] = [scores[f'talker{talker}'][comp][condition]['score'][i][0] for comp in comparisons]
                    yerrmax[condition] = [scores[f'talker{talker}'][comp][condition]['score'][i][1] for comp in
                                          comparisons]
                    yerrmin[condition] = [scores[f'talker{talker}'][comp][condition]['score'][i][2] for comp in
                                          comparisons]
                rects1 = ax.bar(x - width / 2 - 0.01, y[conditions[0]], width, label=conditions[0],
                                color='cornflowerblue')

                ax.set_ylabel('Scores')
                ax.set_xticks(x, comparisons)
                ax.legend()

                ax.scatter(x - width / 2 - 0.01, yerrmax[conditions[0]], c='black', marker='_', s=50)
                ax.scatter(x - width / 2 - 0.01, yerrmin[conditions[0]], c='black', marker='_', s=50)

                n_trials = {}
                trial_string = ''
                for comp in comparisons:
                    n_trials[comp] = {}
                    for cond in conditions:
                        n_trials[comp][cond] = np.sum(scores[f'talker{talker}'][comp][cond]['cm'][i])
                        trial_string += f'{comp} {cond}: {n_trials[comp][cond]}\n'

                ax.bar_label(rects1, padding=3)
                # ax.bar_label(rects2, padding=3)
                ax.set_ylim([0, 1])
                simple_xy_axes(ax)
                set_font_axes(ax, add_size=10)
                fig.suptitle(f'cluster {clus}, \nn_trials: {trial_string}')
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def save_pdf_classification_lstm(scores, saveDir, title, probeword):
    conditions = ['pitchshift', 'nopitchshift']
    for talker in [1, 2]:

        comparisons = [comp for comp in scores[f'talker{talker}']]
        comp = comparisons[0]
        i = 0
        # clus = scores[f'talker{talker}'][comp]['pitchshift']['cluster_id'][i]
        if len(scores['talker1'][comp]['pitchshift']) > len(scores['talker1'][comp]['nopitchshift']):
            k = 'pitchshift'
        else:
            k = 'nopitchshift'

        with PdfPages(saveDir / f'{title}_talker{talker}_probeword{probeword[0]}.pdf') as pdf:
            for i, clus in enumerate(
                    tqdm(scores[f'talker{talker}'][comp][k]['cluster_id'])):  # ['pitchshift']['cluster_id'])):
                fig, ax = plt.subplots(figsize=(10, 5))
                y = {}
                yerrmax = {}
                yerrmin = {}
                x = np.arange(len(comparisons))
                x2 = np.arange(len(conditions))

                width = 0.35
                for condition in conditions:
                    try:
                        y[condition] = [scores[f'talker{talker}'][comp][condition]['lstm_avg'][i] for comp in
                                        comparisons]
                    except:
                        print('dimension mismatch')
                        continue

                try:
                    rects1 = ax.bar(x - width / 2 - 0.01, y[conditions[0]], width, label=conditions[0],
                                    color='cornflowerblue')
                    rects2 = ax.bar(x + width / 2 + 0.01, y[conditions[1]], width, label=conditions[1],
                                    color='lightcoral')
                except:
                    print('both conditions not satisfied')
                    continue
                ax.set_ylabel('Scores')
                ax.set_xticks(x, comparisons)
                if talker == 1:
                    talkestring = 'Female'
                else:
                    talkestring = 'Male'
                # plt.title('LSTM classification scores for extracted units,'+ talkestring+' talker')
                ax.legend()
                #
                # ax.scatter(x - width / 2 - 0.01, yerrmax[conditions[0]], c='black', marker='_', s=50)
                # ax.scatter(x - width / 2 - 0.01, yerrmin[conditions[0]], c='black', marker='_', s=50)
                # ax.scatter(x + width / 2 + 0.01, yerrmax[conditions[1]], c='black', marker='_', s=50)
                # ax.scatter(x + width / 2 + 0.01, yerrmin[conditions[1]], c='black', marker='_', s=50)
                # ax.scatter(range(len(scores)), yerrmax, c='black', marker='_', s=10)
                # ax.scatter(range(len(scores)), yerrmin, c='black', marker='_', s=10)

                n_trials = {}
                trial_string = ''
                for comp in comparisons:
                    n_trials[comp] = {}
                    for cond in conditions:
                        n_trials[comp][cond] = np.sum(scores[f'talker{talker}'][comp][cond]['cm'][i])
                        trial_string += f'{comp} {cond}: {n_trials[comp][cond]}\n'

                ax.bar_label(rects1, padding=3, fmt='%.2f')
                ax.bar_label(rects2, padding=3, fmt='%.2f')
                ax.set_ylim([0, 1])
                simple_xy_axes(ax)
                set_font_axes(ax, add_size=10)
                fig.suptitle(f'cluster {clus}, \nn_trials: {trial_string}')
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def save_pdf_classification_lstm_bothtalker(scores, saveDir, title):
    conditions = ['pitchshift', 'nopitchshift']
    for talker in [1, 2]:
        # talker = 1
        # title = f'eucl_classification_{month}_talker{talker}_win_bs_earlylateprobe_leftright_26082022'

        comparisons = [comp for comp in scores[f'talker{talker}']]
        comp = comparisons[0]
        i = 0
        clus = scores[f'talker{talker}'][comp]['pitchshift']['cluster_id'][i]
        if len(scores['talker1'][comp]['pitchshift']) > len(scores['talker1'][comp]['nopitchshift']):
            k = 'pitchshift'
        else:
            k = 'nopitchshift'

        with PdfPages(saveDir / f'{title}_talker{talker}.pdf') as pdf:
            for i, clus in enumerate(
                    tqdm(scores[f'talker{talker}'][comp][k]['cluster_id'])):  # ['pitchshift']['cluster_id'])):
                fig, ax = plt.subplots(figsize=(10, 5))
                y = {}
                yerrmax = {}
                yerrmin = {}
                x = np.arange(len(comparisons))
                x2 = np.arange(len(conditions))

                width = 0.35
                for condition in conditions:
                    try:
                        y[condition] = [scores[f'talker{talker}'][comp][condition]['lstm_avg'][i] for comp in
                                        comparisons]
                    except:
                        print('dimension mismatch')
                        continue

                rects1 = ax.bar(x - width / 2 - 0.01, y[conditions[0]], width, label=conditions[0],
                                color='cornflowerblue')
                rects2 = ax.bar(x + width / 2 + 0.01, y[conditions[1]], width, label=conditions[1], color='lightcoral')

                ax.set_ylabel('Scores')
                ax.set_xticks(x, comparisons)
                plt.title('LSTM classification scores for extracted units')
                ax.legend()

                n_trials = {}
                trial_string = ''
                for comp in comparisons:
                    n_trials[comp] = {}
                    for cond in conditions:
                        n_trials[comp][cond] = np.sum(scores[f'talker{talker}'][comp][cond]['cm'][i])
                        trial_string += f'{comp} {cond}: {n_trials[comp][cond]}\n'

                ax.bar_label(rects1, padding=3, fmt='%2f')
                ax.bar_label(rects2, padding=3, fmt='%2f')
                ax.set_ylim([0, 1])
                simple_xy_axes(ax)
                set_font_axes(ax, add_size=10)
                fig.suptitle(f'cluster {clus}, \nn_trials: {trial_string}')
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def run_classification(dir, datapath, ferretid):
    fname = 'blocks.pkl'
    with open(datapath / 'blocks.pkl', 'rb') as f:
        blocks = pickle.load(f)

    scores = {}
    probewords_list = [(2, 2), (20, 22), (5, 6), (42, 49), (32, 38)]
    # probewords_list = [(20, 22), (5, 6), (42, 49), (32, 38)]

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H_%M_%S")

    dt_string = now.strftime("%d%m%Y_%H_%M_%S")

    tarDir = Path(
        f'D:/Data/decoding_over_time_results/F1702_Zola/')
    saveDir = tarDir
    saveDir.mkdir(exist_ok=True, parents=True)
    for probeword in probewords_list:
        print('now starting')
        print(probeword)
        for talker in [2]:
            binsize = 0.01
            if talker == 1:
                window = [0, 0.6]
            else:
                window = [0, 0.5]
            print(f'talker {talker}')

            scores[f'talker{talker}'] = {}

            scores[f'talker{talker}']['target_vs_probe'] = {}

            scores[f'talker{talker}']['target_vs_probe']['nopitchshift'] = target_vs_probe(blocks, talker=talker,
                                                                                           probewords=probeword,
                                                                                           pitchshift=False)
            scores[f'talker{talker}']['target_vs_probe']['pitchshift'] = target_vs_probe(blocks, talker=talker,
                                                                                         probewords=probeword,
                                                                                         pitchshift=True)

            np.save(saveDir / f'scores_{dir}_{probeword[0]}_{ferretid}_probe_allpitches_onlymaletalker_bs.npy',
                    scores)

        fname = 'scores_' + dir + f'_probe_earlylate_left_right_win_bs_{binsize}'


def main():
    directories = [
        'orecchiette_2022']  # , 'Trifle_July_2022']/home/zceccgr/Scratch/zceccgr/ms4output/F1702_Zola/spkenvresults04102022allrowsbut4th

    datapath = Path(f'D:\F1702_Zola\spkenvresults04102022allrowsbut4th')
    ferretid = 'zola'

    for dir in directories:
        run_classification(dir, datapath, ferretid)


if __name__ == '__main__':
    main()
