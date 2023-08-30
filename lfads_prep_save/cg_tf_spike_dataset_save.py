from pathlib import Path
import h5py
from tqdm import tqdm
from datetime import datetime
from instruments.helpers.neural_analysis_helpers import get_word_aligned_raster, get_word_aligned_raster_with_pitchshift
from instruments.helpers.euclidean_classification_minimal_function import classify_sweeps
import numpy as np
import pickle



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
              'lstm_avg': [],
              'lstm_balanced': [],
              'lstm_balanced_avg': [],
              'lstm_score_list': [],
              'history': [],}
    cluster_id_droplist = np.empty([])
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
        raster_probe_reshaped = np.empty([len(unique_trials_probe), len(bins) - 1])

        count = 0
        for trial in (unique_trials_probe):
            raster_probe_reshaped[count, :] = np.histogram(raster_probe['spike_time'][raster_probe['trial_num'] == trial], bins=bins, range=(window[0], window[1]))[0]
            count+=1


        stim1 = np.full(len(raster_probe_reshaped), 1)  # 1 = probe word
        # if len(stim0)+len(stim1)<7:
        #     print('less than 10 trials')
        #     continue

        raster_lstm = raster_probe_reshaped

        raster_reshaped = np.reshape(raster_lstm, (np.size(raster_lstm, 0), np.size(raster_lstm, 1), 1)).astype(
            'float32')

        #save as h5 files
        print('saving h5 files')
        with h5py.File(f'D:/tf_h5files/F1702_Zola/raster_reshaped_{str(talker)}_{str(probeword)}_pitchshift_{pitchshift}.h5', 'w') as hf:
            hf.create_dataset("spike_data", data=raster_reshaped)
            # hf.create_dataset("stim_data", data=stim_reshaped)


    return scores




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
            # window=[0,0.87]
            print(f'talker {talker}')

            scores[f'talker{talker}'] = {}

            scores[f'talker{talker}']['target_vs_probe'] = {}

            scores[f'talker{talker}']['target_vs_probe']['nopitchshift'] = target_vs_probe(blocks, talker=talker,
                                                                                           probewords=probeword,
                                                                                           pitchshift=False)
            scores[f'talker{talker}']['target_vs_probe']['pitchshift'] = target_vs_probe(blocks, talker=talker,
                                                                                         probewords=probeword,
                                                                                         pitchshift=True)


        #     np.save(saveDir / f'scores_{dir}_{probeword[0]}_zola_probe_pitchshift_vs_not_by_talker_bs.npy', scores)
        #
        # fname = 'scores_' + dir + f'_probe_earlylate_left_right_win_bs_{binsize}'
        # save_pdf_classification_lstm(scores, saveDir, fname, probeword)

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
    directories = ['zola_2022']  # , 'Trifle_July_2022']
    for dir in directories:
        run_export(dir)


if __name__ == '__main__':
    main()
