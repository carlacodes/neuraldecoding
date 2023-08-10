import numpy as np
import matplotlib.pyplot as plt



def run_scores_and_plot(file_path, pitchshift, output_folder, ferretname,  stringprobewordindex=str(2), talker='female'):
    if talker == 'female':
        talker_string = 'onlyfemaletalker'
    scores = np.load(
        file_path + '/' + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_'+pitchshift +'_'+talker_string+'_bs.npy',
        allow_pickle=True)[()]

    #for each cluster plot their scores over time
    index = -1
    for cluster in scores['talker1']['target_vs_probe']['nopitchshift']['cluster_id']:
        #get the scores
        index = index + 1
        cluster_scores =scores['talker1']['target_vs_probe']['nopitchshift']['lstm_balancedaccuracylist'][index]
        #get the timepoints
        timepoints = np.arange(0, len(cluster_scores), 1)

        fig, ax = plt.subplots()
        ax.plot(timepoints, cluster_scores)
        ax.set(xlabel='timepoints', ylabel='balanced accuracy',
                title='balanced accuracy over time')
        ax.grid()
        fig.savefig(output_folder + '/' + 'cluster' + str(cluster) + '.png')
        plt.show()


    return scores








if __name__ == '__main__':
    print('hello')
    file_path = 'E:\\lstm_kfold_01082023_oreovertimes2'
    output_folder = 'E:\\lstm_kfold_01082023_oreovertimes2'
    ferretname = 'orecchiette'
    pitchshift = 'nopitchshift'
    run_scores_and_plot(file_path, pitchshift, output_folder, ferretname, stringprobewordindex=str(2), talker = 'female' )

