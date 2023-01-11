import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


scoremat = np.load('D:/Users/cgriffiths/resultsms4/lstmclass_18112022/18112022_10_58_57/scores_Eclair_2022_2_eclair_probe_pitchshift_vs_not_by_talker_bs.npy', allow_pickle=True)[()]


oldscoremat = np.load('D:/Users/juleslebert/home/phd/figures/euclidean_class_082022/eclair/17112022_16_24_15/scores_Eclair_2022_probe_earlylate_left_right_win_bs.npy', allow_pickle=True)[()]
probewordlist = [(20, 22), (2, 2), (5, 6), (42, 49), (32, 38)]

saveDir ='D:/Users/cgriffiths/resultsms4/lstmclass_18112022/19112022_11_54_53/'
def scatterplot_and_visualise(probewordlist, probewordindex=None):
    pitchshiftlist = np.empty([])
    nonpitchshiftlist = np.empty([])
    for probeword in probewordlist:
        probewordindex = probeword[0]
        stringprobewordindex = str(probewordindex)
        #scores_Eclair_2022_2_eclair_probe_pitchshift_vs_not_by_talker_bs
        scores = np.load(saveDir +'/' +r'scores_Eclair_2022_'+stringprobewordindex+'_eclair_probe_pitchshift_vs_not_by_talker_bs.npy', allow_pickle=True)[()]
        for talker in [1, 2]:
            comparisons = [comp for comp in scores[f'talker{talker}']]

            for comp in comparisons:
                for cond in ['pitchshift', 'nopitchshift']:
                    for i, clus in enumerate(scores[f'talker{talker}'][comp][cond]['cluster_id']):
                        # pitchshiftlist = scores[f'talker{talker}'][comp]['pitchshift']['lstm_score'][i]
                        # x2 = scores[f'talker{talker}'][comp]['nopitchshift']['lstm_score'][i]
                        if cond == 'pitchshift':
                            pitchshiftlist = np.append(pitchshiftlist, scores[f'talker{talker}'][comp][cond]['lstm_score'][i])
                        else:
                            nonpitchshiftlist = np.append(nonpitchshiftlist, scores[f'talker{talker}'][comp][cond]['lstm_score'][i])
                        # pitchshiftlist = np.append(pitchshiftlist, scores[f'talker{talker}'][comp]['pitchshift']['lstm_score'][i])
                        # nonpitchshiftlist = np.append(nonpitchshiftlist, scores[f'talker{talker}'][comp]['nopitchshift']['lstm_score'][i])

                        # plt.title(f'cluster {clus}')
                        # plt.show()
    return pitchshiftlist, nonpitchshiftlist

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
                        y[condition] = [scores[f'talker{talker}'][comp][condition]['lstm_score'][i] for comp in
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

def main():
    probewordlist = [(20, 22), (2, 2), (5, 6), (42, 49), (32, 38)]

    pitchshiftlist, nonpitchshiftlist = scatterplot_and_visualise(probewordlist)



if __name__ == '__main__':
    main()
