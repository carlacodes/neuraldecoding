import numpy as np
import matplotlib.pyplot as plt



def run_scores_and_plot(file_path, pitchshift, output_folder, ferretname,  stringprobewordindex=str(2), talker='female', totalcount = 0):
    if talker == 'female':
        talker_string = 'onlyfemaletalker'
        talkerinput = 'talker1'
    else:
        talker_string = 'onlymaletalker'
        talkerinput = 'talker2'
    scores = np.load(
        file_path + '/' + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_'+pitchshift +'_'+talker_string+'_bs.npy',
        allow_pickle=True)[()]
    if pitchshift == 'pitchshift':
        pitchshifttext = 'intra-roved F0'
    elif pitchshift == 'nopitchshift':
        pitchshifttext = 'control F0'

    if stringprobewordindex == '2':
        word_equivalent = 'when a'
    elif stringprobewordindex == '5':
        word_equivalent = 'craft'
    elif stringprobewordindex == '42':
        word_equivalent = 'accurate'
    elif stringprobewordindex == '32':
        word_equivalent = 'of science'
    elif stringprobewordindex == '22':
        word_equivalent = 'in contrast'


    #for each cluster plot their scores over time
    index = -1
    #make a subplot grid
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    fig, axs = plt.subplots(2, int(len(scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id'])/2), figsize=(50,15))
    for cluster in scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']:
        #get the scores
        index = index + 1
        cluster_scores =scores[talkerinput]['target_vs_probe'][pitchshift]['lstm_balancedaccuracylist'][index]
        #get the timepoints
        timepoints = np.arange(0, len(cluster_scores)/100, 0.01)
        if index < int(len(scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id'])/2):
            ax = axs[1,index]
        else:
            ax = axs[0,index-int(len(scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id'])/2)]

        ax.plot(timepoints, cluster_scores, c = color_list[totalcount])
        ax.set(xlabel='time since target word (s)', ylabel='balanced accuracy',
            title='unit ' + str(cluster))
        ax.set_ylim([0, 1])
        ax.grid()
    plt.suptitle('LSTM balanced accuracy for ' + ferretname + ' ' + pitchshifttext + ' ' + talker+ ' target vs. ' + word_equivalent, fontsize = 30)
    fig.savefig(output_folder + '/' + 'multipanel' +pitchshift+ stringprobewordindex +'talker_'+talker+ '.png', bbox_inches='tight')


    return scores








if __name__ == '__main__':
    print('hello')
    file_path = 'E:\\lstm_kfold_01082023_oreovertimes2'
    output_folder = 'E:\\lstm_kfold_01082023_oreovertimes2'
    ferretname = 'orecchiette'
    pitchshift = 'pitchshift'
    stringprobewordlist = [5,2]
    # probewordlist = [ (5, 6),(2, 2), (42, 49), (32, 38), (20, 22)]
    totalcount = 0
    talkerlist = ['female', 'male']
    for talker in talkerlist:
        for probeword in stringprobewordlist:

            print(probeword)
            run_scores_and_plot(file_path, pitchshift, output_folder, ferretname, stringprobewordindex=str(probeword), talker = talker, totalcount = totalcount )
            totalcount = totalcount + 1

