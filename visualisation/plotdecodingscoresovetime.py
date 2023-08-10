import numpy as np
import matplotlib.pyplot as plt



def run_scores_and_plot(file_path, pitchshift, output_folder, ferretname,  stringprobewordindex=str(2), talker='female'):
    if talker == 'female':
        talker_string = 'onlyfemaletalker'
    scores = np.load(
        file_path + '/' + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_'+pitchshift +'_'+talker_string+'_bs.npy',
        allow_pickle=True)[()]

    if stringprobewordindex == '2':
        word_equivalent = 'when a'

    #for each cluster plot their scores over time
    index = -1
    #make a subplot grid

    fig, axs = plt.subplots(2, int(len(scores['talker1']['target_vs_probe']['nopitchshift']['cluster_id'])/2), figsize=(50,15))
    for cluster in scores['talker1']['target_vs_probe']['nopitchshift']['cluster_id']:
        #get the scores
        index = index + 1
        cluster_scores =scores['talker1']['target_vs_probe']['nopitchshift']['lstm_balancedaccuracylist'][index]
        #get the timepoints
        timepoints = np.arange(0, len(cluster_scores)/100, 0.01)
        if index < int(len(scores['talker1']['target_vs_probe']['nopitchshift']['cluster_id'])/2):
            ax = axs[1,index]
        else:
            ax = axs[0,index-int(len(scores['talker1']['target_vs_probe']['nopitchshift']['cluster_id'])/2)]

        ax.plot(timepoints, cluster_scores)
        ax.set(xlabel='time since target word (s)', ylabel='balanced accuracy',
            title='cluster ' + str(cluster))
        ax.set_ylim([0, 1])
        ax.grid()
    plt.suptitle('LSTM balanced accuracy for ' + ferretname + ' ' + pitchshift + ' ' + talker+ ' target vs. ' + word_equivalent, fontsize = 30)
    fig.savefig(output_folder + '/' + 'multipanel' +stringprobewordindex + '.png', bbox_inches='tight')


    return scores








if __name__ == '__main__':
    print('hello')
    file_path = 'E:\\lstm_kfold_01082023_oreovertimes2'
    output_folder = 'E:\\lstm_kfold_01082023_oreovertimes2'
    ferretname = 'orecchiette'
    pitchshift = 'nopitchshift'
    stringprobewordlist = [5,2,42,32,20]
    # probewordlist = [ (5, 6),(2, 2), (42, 49), (32, 38), (20, 22)]
    for stringprobewordindex in stringprobewordlist:
        run_scores_and_plot(file_path, pitchshift, output_folder, ferretname, stringprobewordindex=str(stringprobewordindex), talker = 'female' )

