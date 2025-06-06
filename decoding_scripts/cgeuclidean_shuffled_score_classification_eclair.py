import pickle
from pathlib import Path
import numpy as np
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from numba import njit, prange
# import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
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
from instruments.helpers.neural_analysis_helpers import get_word_aligned_raster
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


def target_vs_probe(blocks, talker=1, probewords=[20, 22], pitchshift=True):
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
              'lstm_shuffled_avg': []}
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


        unique_trials_targ=np.unique(raster_target['trial_num'])
        unique_trials_probe=np.unique(raster_probe['trial_num'])
        raster_targ_reshaped = np.empty([len(unique_trials_targ), len(bins) - 1])
        raster_probe_reshaped = np.empty([len(unique_trials_probe), len(bins) - 1])
        count=0
        for trial in (unique_trials_targ):
            raster_targ_reshaped[count, :] = np.histogram(raster_target['spike_time'][raster_target['trial_num'] == trial], bins=bins, range=(window[0], window[1]))[0]
            count+=1
        count = 0
        for trial in (unique_trials_probe):
            raster_probe_reshaped[count, :] = np.histogram(raster_probe['spike_time'][raster_probe['trial_num'] == trial], bins=bins, range=(window[0], window[1]))[0]
            count+=1

        stim0 = np.full(len(raster_target), 0)  # 0 = target word
        stim1 = np.full(len(raster_probe), 1)  # 1 = probe word
        stim = np.concatenate((stim0, stim1))

        stim0 = np.full(len(raster_targ_reshaped), 0)  # 0 = target word
        stim1 = np.full(len(raster_probe_reshaped), 1)  # 1 = probe word
        stim_lstm = np.concatenate((stim0, stim1))

        #
        # if len(raster_target) > len(raster_probe):
        #     raster_target = np.random.choice(raster_target, len(raster_probe), replace=True)
        # elif len(raster_target) < len(raster_probe):
        #     raster_probe = np.random.choice(raster_probe, len(raster_target), replace=True)

        raster = np.concatenate((raster_target, raster_probe))
        raster_lstm = np.concatenate((raster_targ_reshaped, raster_probe_reshaped))

        score, d, bootScore, bootClass, cm = classify_sweeps(raster, stim, binsize=binsize, window=window, genFig=False)
        # fit LSTM model to the same data
        #
        newraster = raster.tolist()
        raster_reshaped = np.reshape(raster_lstm, (np.size(raster_lstm, 0), np.size(raster_lstm, 1), 1)).astype(
            'float32')
        stim_reshaped = np.reshape(stim_lstm, (len(stim_lstm), 1)).astype('float32')
        accuracy_list = []
        for i in range(0,3):
            #do 3 iterations of the LSTM model,
            # get the average accuracy of this randomised shuffled score
            tf.keras.backend.clear_session()

            X_train, X_test, y_train, y_test = train_test_split(raster_reshaped, stim_reshaped, test_size=0.33,
                                                                )
            model_lstm = LSTMClassification(units=400, dropout=0.25, num_epochs=10)
            X_train = sklearn.utils.shuffle(X_train)


            # Fit model
            model_lstm.fit(X_train, y_train)

            # Get predictions
            y_valid_predicted_lstm = model_lstm.predict(X_test)



            y_valid_predicted_lstm_bool = abs(np.round(y_valid_predicted_lstm))
            y_test_bool = abs(np.round(y_test)).astype(bool)

            # Get metric of fit
            # R2s_lstm = get_R2(y_test, y_valid_predicted_lstm)
            accuracy = sklearn.metrics.accuracy_score(y_test.flatten(), y_valid_predicted_lstm.flatten())
            accuracy_list.append(accuracy)

        accuracytoppercentile= np.percentile(accuracy_list, 97.5)

        # print('R2s:', R2s_lstm)

        scores['cluster_id'].append(cluster_id)
        scores['score'].append(score)
        scores['lstm_score'].append(accuracytoppercentile)
        scores['lstm_shuffled_avg'].append(np.mean(accuracy_list))
        scores['bootScore'].append(bootScore)
        scores['cm'].append(len(unique_trials_targ)+len(unique_trials_probe))

    # for i, cluster in enumerate(scores['cluster_id']):
    #     print(f'cluster {cluster}')
    #     print(f'score: {scores["score"][i]}')
    #     print(f'cm: \n{scores["cm"][i]}')

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
        X_train = sklearn.utils.shuffle(X_train)

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

    # for i, cluster in enumerate(scores['cluster_id']):
    #     print(f'cluster {cluster}')
    #     print(f'score: {scores["score"][i]}')
    #     print(f'cm: \n{scores["cm"][i]}')

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
                # rects2 = ax.bar(x + width / 2 + 0.01, y[conditions[1]], width, label=conditions[1], color='lightcoral')

                ax.set_ylabel('Scores')
                ax.set_xticks(x, comparisons)
                ax.legend()

                ax.scatter(x - width / 2 - 0.01, yerrmax[conditions[0]], c='black', marker='_', s=50)
                ax.scatter(x - width / 2 - 0.01, yerrmin[conditions[0]], c='black', marker='_', s=50)
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
        # talker = 1
        # title = f'eucl_classification_{month}_talker{talker}_win_bs_earlylateprobe_leftright_26082022'

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
                        y[condition] = [scores[f'talker{talker}'][comp][condition]['lstm_shuffled_avg'][i] for comp in
                                        comparisons]
                    except:
                        print('dimension mismatch')
                        continue
                    #                     # yerrmax[condition] = [scores[f'talker{talker}'][comp][condition]['score'][i][1] for comp in
                    #                       comparisons]
                    # yerrmin[condition] = [scores[f'ta      lker{talker}'][comp][condition]['score'][i][2] for comp in
                    #                       comparisons]
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
                        y[condition] = [scores[f'talker{talker}'][comp][condition]['lstm_score'][i] for comp in
                                        comparisons]
                    except:
                        print('dimension mismatch')
                        continue
                    #                     # yerrmax[condition] = [scores[f'talker{talker}'][comp][condition]['score'][i][1] for comp in
                    #                       comparisons]
                    # yerrmin[condition] = [scores[f'ta      lker{talker}'][comp][condition]['score'][i][2] for comp in
                    #                       comparisons]
                rects1 = ax.bar(x - width / 2 - 0.01, y[conditions[0]], width, label=conditions[0],
                                color='cornflowerblue')
                rects2 = ax.bar(x + width / 2 + 0.01, y[conditions[1]], width, label=conditions[1], color='lightcoral')

                ax.set_ylabel('Scores')
                ax.set_xticks(x, comparisons)
                plt.title('LSTM classification scores for extracted units')
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

                ax.bar_label(rects1, padding=3, fmt='%2f')
                ax.bar_label(rects2, padding=3, fmt='%2f')
                ax.set_ylim([0, 1])
                simple_xy_axes(ax)
                set_font_axes(ax, add_size=10)
                fig.suptitle(f'cluster {clus}, \nn_trials: {trial_string}')
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def run_classification(dir):

    datapath = Path(f'D:\ms4output\F1902_Eclair\wpsoutput08112022bb2bb3\phy')
    fname = 'blocks.pkl'
    with open(datapath / 'blocks.pkl', 'rb') as f:
        blocks = pickle.load(f)

    scores = {}
    probewords_list= [(2,2),(20,22), (5,6), (42,49), (32, 38)]
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H_%M_%S")

    tarDir = Path(f'/Users/cgriffiths/resultsms4/lstmclass_SHUFFLEDDATA_02122022')
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
            # scores[f'talker{talker}']['left'] = {}

            # scores[f'talker{talker}']['left']['noise'] = probe_early_vs_late(blocks, talker=talker, noise = True, df_filter=['No Level Cue', 'Sound Left'], window=window, binsize=binsize)
            # scores[f'talker{talker}']['left']['silence'] = probe_early_vs_late(blocks, talker=talker, noise = False, df_filter=['No Level Cue', 'Sound Left'], window=window, binsize=binsize)

            # scores[f'talker{talker}']['right'] = {}
            # scores[f'talker{talker}']['right']['noise'] = probe_early_vs_late(blocks, talker=talker, noise = True, df_filter=['No Level Cue', 'Sound Right'], window=window, binsize=binsize)
            # scores[f'talker{talker}']['right']['silence'] = probe_early_vs_late(blocks, talker=talker, noise = False, df_filter=['No Level Cue', 'Sound Right'], window=window, binsize=binsize)

            scores[f'talker{talker}']['target_vs_probe'] = {}

            scores[f'talker{talker}']['target_vs_probe']['nopitchshift'] = target_vs_probe(blocks, talker=talker,
                                                                                           probewords=probeword,
                                                                                           pitchshift=False)
            scores[f'talker{talker}']['target_vs_probe']['pitchshift'] = target_vs_probe(blocks, talker=talker,
                                                                                         probewords=probeword,
                                                                                         pitchshift=True)


            np.save(saveDir / f'scores_{dir}_{probeword[0]}_eclair_probe_pitchshift_vs_not_by_talker_bs.npy', scores)

        fname = 'scores_' + dir + f'_probe_earlylate_left_right_win_bs_{binsize}'
        save_pdf_classification_lstm(scores, saveDir, fname, probeword)

    # title = f'eucl_classification_{dir}_26082022'
    # with PdfPages(saveDir / f'{title}.pdf') as pdf:
    #     for i, clus in enumerate(tqdm(scores['probe_early_vs_late']['cluster_id'])):
    #         fig, axes = plt.subplots(3, 1, figsize=(4,10), dpi=300)
    #         for j, comp in enumerate(scores):
    #             ax = axes[j]
    #             sns.heatmap(scores[comp]['cm'][i], annot=True, fmt="d", ax=ax, cmap='Blues')
    #             ax.set_xlabel('\nPredicted Values')
    #             ax.set_ylabel('Actual Values ');
    #             ax.set_title(comp)

    #         y = [scores[comp]['score'][i][0] for comp in scores]
    #         yerrmax = [scores[comp]['score'][i][1] for comp in scores]
    #         yerrmin = [scores[comp]['score'][i][2] for comp in scores]
    #         axes[-1].bar(range(len(scores)), y, align='center', color='cornflowerblue')
    #         axes[-1].set_xticks(range(len(scores)))
    #         axes[-1].set_xticklabels(list(scores.keys()))

    #         axes[-1].scatter(range(len(scores)), yerrmax, c='black', marker='_', s=10)
    #         axes[-1].scatter(range(len(scores)), yerrmin, c='black', marker='_', s=10)

    #         axes[-1].set_ylim([0,1])

    #         # plt.figure()
    #         # plt.imshow(scores[clus]['cm'], interpolation='nearest', cmap=plt.cm.Blues)
    #         fig.suptitle(f'cluster {clus}')
    #         fig.tight_layout()
    #         # plt.colorbar()
    #         pdf.savefig(fig)
    #         plt.close(fig)


def main():
    # from google_drive_downloader import GoogleDriveDownloader as gdd
    #
    # gdd.download_file_from_google_drive(file_id='1W3TwEtC0Z6Qmbfuz8_AWRiQHfuDb9FIS',
    #                                     dest_path='./Binned_data.zip',
    #                                     unzip=True)
    binned_spikes = np.load('../archived_data/binned_spikes.npy')
    choices = np.load('../archived_data/choices.npy') + 1
    print(binned_spikes.shape, choices.shape)
    print(choices[:10])
    directories = ['eclair_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        run_classification(dir)


if __name__ == '__main__':
    main()
