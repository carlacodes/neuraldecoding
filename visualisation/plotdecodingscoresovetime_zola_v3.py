import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import scipy
from itertools import combinations, permutations


def plot_average_over_time(file_path, pitchshift, outputfolder, ferretname, high_units, talkerinput = 'talker1', animal_id = 'F1702', smooth_option = True, plot_on_one_figure = False):
    probewordslist = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    score_dict = {}
    correlations = {}
    avg_scores = {}
    animal_id = animal_id.split('_')[0]
    rec_name = file_path.parts[-2]
    stream = file_path.parts[-1]
    if pitchshift == 'nopitchshift':
        scores = np.load(
                    str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(2) + '_' + ferretname + '_probe_nopitchshift_bs.npy',
                    allow_pickle=True)[()]
    else:
        scores = np.load(
                    str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(2) + '_' + ferretname + '_probe_pitchshift_bs.npy',
                    allow_pickle=True)[()]


    #create a dictionary of scores for each cluster
    for cluster in scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']:
        score_dict[cluster] = {}
        correlations[cluster] = {}
        avg_scores[cluster] = {}

    for cluster in scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']:
        for probeword in probewordslist:
            try:
                if pitchshift == 'nopitchshift':
                    scores = np.load(
                        str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(
                            probeword) + '_' + ferretname + '_probe_nopitchshift_bs.npy',
                        allow_pickle=True)[()]
                else:
                    scores = np.load(
                        str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(
                            probeword) + '_' + ferretname + '_probe_pitchshift_bs.npy',
                        allow_pickle=True)[()]
                #find the index of the cluster
                index = scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id'].index(cluster)

                score_dict[cluster][probeword] = scores[talkerinput]['target_vs_probe'][pitchshift]['lstm_balancedaccuracylist'][index]

            except:
                print('error loading scores: ' + str(
                    file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(probeword) + '_' + ferretname + '_probe_bs.npy')
                continue
    #compute the average over time with the standard deviation
    for cluster in score_dict.keys():
        score_dict_cluster = score_dict[cluster]
        score_dict_cluster_list = []
        for probeword in probewordslist:
            try:
                score_dict_cluster_list.append(score_dict_cluster[probeword])
            except:
                continue
        #convert to numpy array
        score_dict_cluster_list = np.array(score_dict_cluster_list)

        #plot this array over time

        #take the mean over the rows

        avg_scores[cluster]['avg_score'] = np.mean(score_dict_cluster_list, axis = 0)
        avg_scores[cluster]['std'] = np.std(score_dict_cluster_list, axis=0)

        avg_scores[cluster]['std'] = np.std(score_dict_cluster_list)
    #plot the average over time

    num_clusters = len(high_units['ID'].to_list())
    meg_clusters = high_units[high_units['BrainArea'] == 'MEG']['ID'].to_list()
    peg_clusters = high_units[high_units['BrainArea'] == 'PEG']['ID'].to_list()

    num_cols = max(len(meg_clusters), len(peg_clusters))
    if len(meg_clusters) == 0 or len(peg_clusters) == 0:
        num_rows = 1
    else:
        num_rows = 2
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(40, 15))
    ax = ax.flatten()
    # fig, ax = plt.subplots(2, int(len(high_units['ID'].to_list())/2), figsize=(20, 20))
    if pitchshift == 'nopitchshift':
        color_text = 'purple'
    else:
        color_text = 'orchid'
    if plot_on_one_figure == True:
        for i, cluster in enumerate(meg_clusters + peg_clusters):
            brain_id = high_units[high_units['ID'] == cluster]['BrainArea'].to_list()[0]
            if cluster in meg_clusters:
                row = 0
                col = meg_clusters.index(cluster)
            else:
                if num_rows == 1:
                    row = 0
                else:
                    row = 1
                col = peg_clusters.index(cluster)

            axs = ax[col + row * num_cols]
            #get the brain ID
            try:
                avg_score = avg_scores[cluster]['avg_score']
            except:
                #remove the axis
                axs.axis('off')
                continue
            timepoints = np.arange(0, (len(avg_score) / 100)*4, 0.04)
            std_dev = avg_scores[cluster]['std']
            if smooth_option == True:
                avg_score = scipy.signal.savgol_filter(avg_score, 5, 3, mode='interp')
                # avg_score = scipy.ndimage.gaussian_filter1d(avg_score, sigma = 1.5)


            axs.plot(timepoints, avg_score, c='black')
            axs.fill_between(timepoints, avg_score - std_dev, avg_score + std_dev, alpha=0.3, color=color_text)


            if i == 0:
                axs.set_xlabel('time (s)', fontsize=20)
                axs.set_ylabel('balanced accuracy', fontsize=20)
                axs.set_title(f'unit:{cluster}', fontsize = 20)
                axs.set_yticks(ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels = [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
                axs.set_xticks(ticks =[0, 0.2, 0.4, 0.6], labels = [0, 0.2, 0.4, 0.6], fontsize=15)


            else:
                axs.set_xlabel('time (s)', fontsize=20)
                axs.set_title(f'unit:{cluster}', fontsize = 20)
                axs.set_yticks(ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels = [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
                axs.set_xticks(ticks =[0, 0.2, 0.4, 0.6], labels = [0, 0.2, 0.4, 0.6], fontsize=15)



            axs.set_ylim([0, 1])
            axs.grid()
        # ax[0,0].set_title('MEG')

        if num_rows == 2:
            ax[0].text(-0.4, 0.5, 'MEG', horizontalalignment='center',
                       verticalalignment='center', rotation=90, transform=ax[0].transAxes, fontsize = 20)
            ax[num_cols].set_ylabel('balanced accuracy', fontsize=20)
            ax[num_cols].text(-0.4, 0.5, 'PEG', horizontalalignment='center',
                            verticalalignment='center', rotation=90, transform=ax[num_cols].transAxes, fontsize = 20)
        else:
            if len(meg_clusters) == 0:
                ax[0].text(-0.4, 0.5, 'PEG', horizontalalignment='center',
                           verticalalignment='center', rotation=90, transform=ax[0].transAxes, fontsize = 20)
            elif len(peg_clusters) == 0:
                ax[0].text(-0.4, 0.5, 'MEG', horizontalalignment='center',
                           verticalalignment='center', rotation=90, transform=ax[0].transAxes, fontsize = 20)

        # ax[1,0].set_title('PEG')

        if pitchshift == 'nopitchshiftvspitchshift' or pitchshift == 'nopitchshift':
            pitchshift_option = False
            pitchshift_text = 'control F0'
        elif pitchshift == 'pitchshift':
            pitchshift_option = True
            pitchshift_text = 'inter-roved F0'

        plt.suptitle(f'balanced accuracy over time for {animal_id},  {pitchshift_text}, {rec_name}_{stream}',  fontsize=30)
        if smooth_option == True:
            plt.savefig(outputfolder + '/' + ferretname+'_'+rec_name+'_'+stream + '_' + pitchshift_text + '_averageovertime_smooth.png', bbox_inches='tight')
        else:
            plt.savefig(outputfolder + '/' + ferretname+'_'+rec_name+'_'+stream + '_' + pitchshift_text + '_averageovertime.png', bbox_inches='tight')
        plt.show()
    else:
        for i, cluster in enumerate(meg_clusters + peg_clusters):
            fig, axs = plt.subplots()
            brain_area = high_units[high_units['ID'] == cluster]['BrainArea'].to_list()[0]
            try:
                avg_score = avg_scores[cluster]['avg_score']
            except:
                # remove the axis
                axs.axis('off')
                continue
            timepoints = np.arange(0, (len(avg_score) / 100) * 4, 0.04)
            std_dev = avg_scores[cluster]['std']
            if smooth_option == True:
                avg_score = scipy.signal.savgol_filter(avg_score, 5, 3, mode='interp')

            axs.plot(timepoints, avg_score, c='black')
            axs.fill_between(timepoints, avg_score - std_dev, avg_score + std_dev, alpha=0.3, color=color_text)

            if pitchshift == 'nopitchshiftvspitchshift' or pitchshift == 'nopitchshift':
                pitchshift_option = False
                pitchshift_text = 'control F0'
            elif pitchshift == 'pitchshift':
                pitchshift_option = True
                pitchshift_text = 'inter-roved F0'
            axs.set_xlabel('time (s)', fontsize=20)
            axs.set_ylabel('balanced accuracy', fontsize=20)
            axs.set_title(f'unit: {cluster}_{ferretname}_{rec_name}_{stream},\n area: {brain_area}, {pitchshift_text}', fontsize=20)
            axs.set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
            axs.set_xticks(ticks=[0, 0.2, 0.4, 0.6], labels=[0, 0.2, 0.4, 0.6], fontsize=15)
            plt.savefig(outputfolder + '/' +str(cluster)+'_'+ ferretname + '_' + rec_name + '_' + stream + '_' + pitchshift_text + '_averageovertime_' + str(cluster) + '.png', bbox_inches='tight')



def calculate_correlation_coefficient(filepath, pitchshift, outputfolder, ferretname, talkerinput = 'talker1', smooth_option = True):
    probewordslist = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    score_dict = {}
    correlations = {}
    avg_correlations = {}
    if pitchshift == 'nopitchshift':
        scores = np.load(
                    str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(2) + '_' + ferretname + '_probe_nopitchshift_bs.npy',
                    allow_pickle=True)[()]
    else:
        scores = np.load(
            str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(2) + '_' + ferretname + '_probe_pitchshift_bs.npy',
            allow_pickle=True)[()]


    #create a dictionary of scores for each cluster

    for cluster in scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']:
        score_dict[cluster] = {}
        correlations[cluster] = {}
        avg_correlations[cluster] = {}


    for cluster in scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']:
        for probeword in probewordslist:
            try:
                if pitchshift == 'nopitchshift':
                    scores = np.load(
                        str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(
                            probeword) + '_' + ferretname + '_probe_nopitchshift_bs.npy',
                        allow_pickle=True)[()]
                else:
                    scores = np.load(
                        str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(
                            probeword) + '_' + ferretname + '_probe_pitchshift_bs.npy',
                        allow_pickle=True)[()]
                #find the index of the cluster
                index = scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id'].index(cluster)
                if smooth_option == True:

                    score_dict[cluster][probeword] = scipy.signal.savgol_filter(scores[talkerinput]['target_vs_probe'][pitchshift]['lstm_balancedaccuracylist'][index], 5,3)
                    # score_dict[cluster][probeword] = scipy.ndimage.gaussian_filter1d(scores[talkerinput]['target_vs_probe'][pitchshift]['lstm_balancedaccuracylist'][index], sigma = 1.5)
                else:
                    score_dict[cluster][probeword] = scores[talkerinput]['target_vs_probe'][pitchshift]['lstm_balancedaccuracylist'][index]

            except:
                print('error loading scores: ' + str(
                    file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(probeword) + '_' + ferretname + '_probe_bs.npy')
                continue
    #compute the cross correlation coefficient

    for cluster in scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']:
        score_dict_cluster = score_dict[cluster]
        for key1 in score_dict_cluster.keys():
            for key2 in score_dict_cluster.keys():
                if key1 != key2 and (key2, key1) not in correlations[cluster].keys():
                    # correlations[cluster][(key1, key2)] = scipy.stats.spearmanr(score_dict_cluster[key1], score_dict_cluster[key2])[0]
                    correlations[cluster][(key1, key2)] = np.corrcoef(score_dict_cluster[key1], score_dict_cluster[key2])[0, 1]



    # for cluster in scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']:
    #     for probeword in probewordslist:
    #         total_corr = 0.0
    #         count = 0
    #         for key_pair, correlation in correlations[cluster].items():
    #             if probeword in key_pair:
    #                 total_corr += correlation
    #                 count += 1
    #         if count > 0:
    #             avg_correlations[cluster][probeword] = total_corr / count
    #         else:
    #             avg_correlations[cluster][probeword] = 0.0  # If no correlation found for the probeword
    #do the entire average
    for cluster in scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']:
        total_corr = 0.0
        count = 0
        for key_pair, correlation in correlations[cluster].items():
            total_corr += correlation
            count += 1
        if count > 0:
            avg_correlations[cluster]['all'] = total_corr / count
        else:
            avg_correlations[cluster]['all'] = 0.0

    return avg_correlations


def calculate_total_distance(permutation):
    total_distance = 0
    for i in range(len(permutation) - 1):
        total_distance += abs(permutation[i][1] - permutation[i + 1][1])
    return total_distance

def find_peak_of_score_timeseries(filepath, pitchshift, outputfolder, ferretname, talkerinput = 'talker1', smooth_option = False):
    probewordslist = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    score_dict = {}
    correlations = {}
    peak_dict = {}
    avg_correlations = {}

    if pitchshift == 'nopitchshift':
        scores = np.load(
                    str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(2) + '_' + ferretname + '_probe_nopitchshift_bs.npy',
                    allow_pickle=True)[()]
    else:
        scores = np.load(
            str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(2) + '_' + ferretname + '_probe_pitchshift_bs.npy',
            allow_pickle=True)[()]
    #create a dictionary of scores for each cluster
    for cluster in scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']:
        score_dict[cluster] = {}
        peak_dict[cluster] = {}


    for cluster in scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']:
        for probeword in probewordslist:
            try:
                if pitchshift == 'nopitchshift':
                    scores = np.load(
                        str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(
                            probeword) + '_' + ferretname + '_probe_nopitchshift_bs.npy',
                        allow_pickle=True)[()]
                else:
                    scores = np.load(
                        str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(
                            probeword) + '_' + ferretname + '_probe_pitchshift_bs.npy',
                        allow_pickle=True)[()]
                #find the index of the cluster
                index = scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id'].index(cluster)
                if smooth_option == True:
                    score_dict[cluster][probeword] = scipy.signal.savgol_filter(scores[talkerinput]['target_vs_probe'][pitchshift]['lstm_balancedaccuracylist'][index], 5, 3)
                    # score_dict[cluster][probeword]= scipy.ndimage.gaussian_filter1d(scores[talkerinput]['target_vs_probe'][pitchshift]['lstm_balancedaccuracylist'][index], sigma = 1.5)
                else:
                    score_dict[cluster][probeword] = scores[talkerinput]['target_vs_probe'][pitchshift]['lstm_balancedaccuracylist'][index]

            except Exception as exception:
                #print exception
                print(exception)

                print('error loading scores: ' + str(
                    file_path) + '/' + r'scores_2022_' + ferretname + '_' + str(probeword) + '_' + ferretname + '_probe_bs.npy')
                continue

    for cluster in scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']:
        score_dict_cluster = score_dict[cluster]
        for key1 in score_dict_cluster.keys():
            #now compute the peak of the score timeseries
            peak = np.max(score_dict_cluster[key1])
            #find the index of the peak
            peak_index = np.where(score_dict_cluster[key1] == peak)
            #convert to a list
            peak_index = peak_index[0].tolist()
            #convert that to seconds
            peak_index = peak_index[0] * 0.04
            #add to the dictionary
            peak_dict[cluster][key1] = peak_index
        #calculate the euclidean distance between the peaks

    for cluster in scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']:
        peak_dict_unit = peak_dict[cluster]

        point_values = [(point, value) for point, value in peak_dict_unit.items()]

        # Generate all permutations of the points
        point_permutations = permutations(point_values)

        # Function to calculate total distance for a given permutation


        # Calculate total distances for all permutations and find the minimum
        min_total_distance = float('inf')
        min_distance_permutation = None

        for perm in point_permutations:
            total_distance = calculate_total_distance(perm)
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                min_distance_permutation = perm
        #add the total distance to the dictionary
        #calculate the standard deviation in the series
        peak_dict_unit_std = np.std(list(peak_dict_unit.values()))
        peak_dict[cluster]['min_distance'] = min_total_distance
        peak_dict[cluster]['std_dev'] = peak_dict_unit_std

    return peak_dict





def run_scores_and_plot(file_path, pitchshift, output_folder, ferretname,  stringprobewordindex=str(2), talker='female', totalcount = 0, smooth_option = True):
    if talker == 'female':
        talker_string = 'onlyfemaletalker'
        talkerinput = 'talker1'
    else:
        talker_string = 'onlymaletalker'
        talkerinput = 'talker2'
        # scores_2022_cruella_2_cruella_probe_bs
    try:
        if pitchshift == 'nopitchshift':
            scores = np.load(
                str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + stringprobewordindex + '_' + ferretname + '_probe_nopitchshift_bs.npy',
                allow_pickle=True)[()]
        else:
            scores = np.load(
                str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + stringprobewordindex + '_' + ferretname + '_probe_nopitchshift_bs.npy',
                allow_pickle=True)[()]
    except:
        print('error loading scores: ' + str(file_path) + '/' + r'scores_2022_' + ferretname + '_' + stringprobewordindex + '_' + ferretname + '_probe_bs.npy')
        return
    rec_name = file_path.parts[-2]
    stream = file_path.parts[-1]
    probeword = int(stringprobewordindex)
    if pitchshift == 'nopitchshiftvspitchshift' or pitchshift == 'nopitchshift':
        pitchshift_option = False
    elif pitchshift == 'pitchshift':
        pitchshift_option = True

    if probeword == 4 and pitchshift_option == False:
        probeword_text = 'when a'
        color_option = 'green'
    elif probeword == 4 and pitchshift_option == True:
        probeword_text = 'when a'
        color_option = 'lightgreen'

    elif probeword == 1 and pitchshift_option == False:
        probeword_text = 'instruments'
        color_option = 'black'
    elif probeword == 1 and pitchshift_option == True:
        probeword_text = 'instruments'
        color_option = 'black'

    elif probeword == 2 and pitchshift_option == False:
        probeword_text = 'craft'
        color_option = 'deeppink'
    elif probeword == 2 and pitchshift_option == True:
        probeword_text = 'craft'
        color_option = 'pink'

    elif probeword == 3 and pitchshift_option == False:
        probeword_text = 'in contrast'
        color_option = 'mediumpurple'
    elif probeword == 3 and pitchshift_option == True:
        probeword_text = 'in contrast'
        color_option = 'purple'

    elif probeword == 5 and pitchshift_option == False:
        probeword_text = 'accurate'
        color_option = 'olivedrab'

    elif probeword == 5 and pitchshift_option == True:
        probeword_text = 'accurate'
        color_option = 'limegreen'
    elif probeword == 6 and pitchshift_option == False:
        probeword_text = 'pink noise'
        color_option = 'navy'
    elif probeword == 6 and pitchshift_option == True:
        probeword_text = 'pink noise'
        color_option = 'lightblue'
    elif probeword == 7 and pitchshift_option == False:
        probeword_text = 'of science'
        color_option = 'coral'
    elif probeword == 7 and pitchshift_option == True:
        probeword_text = 'of science'
        color_option = 'orange'
    elif probeword == 8 and pitchshift_option == False:
        probeword_text = 'rev. instruments'
        color_option = 'plum'
    elif probeword == 8 and pitchshift_option == True:
        probeword_text = 'rev. instruments'
        color_option = 'darkorchid'
    elif probeword == 9 and pitchshift_option == False:
        probeword_text = 'boats'
        color_option = 'cornflowerblue'
    elif probeword == 9 and pitchshift_option == True:
        probeword_text = 'boats'
        color_option = 'royalblue'
    elif probeword == 10 and pitchshift_option == False:
        probeword_text = 'today'
        color_option = 'gold'
    elif probeword == 10 and pitchshift_option == True:
        probeword_text = 'today'
        color_option = 'yellow'

    #for each cluster plot their scores over time
    num_clusters = len(scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id'])
    num_cols = int(num_clusters / 2) + 1  # Calculate the number of columns
    if len (scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']) == 1:
        fig, axs = plt.subplots( figsize=(50, 15))
    else:
        fig, axs = plt.subplots(2, num_cols, figsize=(50, 15))

    index = -1  # Initialize index here

    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    # fig, axs = plt.subplots(2, int(len(scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id'])/2)+1, figsize=(50,15))
    for cluster in scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']:
        #get the scores
        index += 1

        cluster_scores = scores[talkerinput]['target_vs_probe'][pitchshift]['lstm_balancedaccuracylist'][index]
        #generate the timepoints based on the bin width of 4ms
        timepoints = np.arange(0, (len(cluster_scores) / 100)*4, 0.04)
        if len (scores[talkerinput]['target_vs_probe'][pitchshift]['cluster_id']) == 1:
            ax = axs
        else:
            row = index // num_cols
            col = index % num_cols
            # Assign ax based on row and column
            ax = axs[row, col]
        if smooth_option == True:
            cluster_scores = scipy.signal.savgol_filter(cluster_scores, 5, 3, mode='interp')
            # cluster_scores = scipy.ndimage.gaussian_filter1d(cluster_scores, sigma = 1.5)
        ax.plot(timepoints, cluster_scores, c = color_option)
        # ax.set(xlabel='time since target word (s)', ylabel='balanced accuracy',
        #     title=f'unit: {cluster}_{rec_name}_{stream}')
        ax.set_ylim([0, 1])
        ax.set_title('unit: ' + str(cluster) + ' ' + rec_name + ' ' + stream, fontsize = 20)

        ax.set_xlabel('time (s)', fontsize = 20)
        ax.set_ylabel('balanced accuracy', fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)

        ax.grid()
    if pitchshift_option == True:
        pitchshift_text = 'inter-roved F0'
    else:
        pitchshift_text = 'control F0'
    plt.suptitle('balanced accuracy for ' + ferretname + ' ' + pitchshift_text + ' ' + talker+ ' target vs. ' + probeword_text, fontsize = 30)
    output_folder2 = output_folder + '/' + rec_name + '/' + stream + '/'
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)
    fig.savefig(output_folder2 + '/' +ferretname+ 'multipanel' +pitchshift+ stringprobewordindex +'_'+probeword_text+'talker_'+talker+ '.png', bbox_inches='tight')


    return scores








if __name__ == '__main__':
    print('hello')

    big_folder = Path('G:/results_decodingovertime_28112023/F1702_Zola/')
    animal = big_folder.parts[-1]
    # file_path = 'D:\decodingresults_overtime\F1815_Cruella\lstm_kfold_balac_01092023_cruella/'
    output_folder = f'G:/decodingovertime_figures/{animal}/'
    #make the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    ferretname = animal.split('_')[1]
    ferretname = ferretname.lower()
    #typo in my myriad code, this should really be relabelled as nopitchshift
    pitchshift = 'nopitchshift'
    stringprobewordlist = [2,3,4,5,6,7,8,9,10]
    # probewordlist = [ (5, 6),(2, 2), (42, 49), (32, 38), (20, 22)]
    totalcount = 0
    talkerlist = ['female']
    #find all the subfolders, all the folders that contain the data

    subfolders = [f for f in big_folder.glob('**/BB_*/') if f.is_dir()]



    for file_path in subfolders:
        #get the subfolders
        print(file_path)
        #get the talke
        for talker in talkerlist:
            for probeword in stringprobewordlist:

                print(probeword)
                run_scores_and_plot(file_path, pitchshift, output_folder, ferretname, stringprobewordindex=str(probeword), talker = talker, totalcount = totalcount )
                totalcount = totalcount + 1
    big_correlation_dict = {}
    big_peak_dict = {}
    for pitchshift in ['pitchshift', 'nopitchshift']:
        big_correlation_dict = {}
        big_peak_dict = {}
        for file_path in subfolders:
            big_correlation_dict[file_path.parts[-2]] = {}
            big_peak_dict[file_path.parts[-2]] = {}
        for file_path in subfolders:
            for talker in talkerlist:

                # for probeword in stringprobewordlist:
                #     print(probeword)
                avg_correlations = calculate_correlation_coefficient(file_path, pitchshift, output_folder, ferretname, talkerinput = 'talker1')
                totalcount = totalcount + 1
                big_correlation_dict[file_path.parts[-2]][file_path.parts[-1]] = avg_correlations
                #find the peak of the score timeseries
                peak_dict = find_peak_of_score_timeseries(file_path, pitchshift, output_folder, ferretname, talkerinput = 'talker1')
                big_peak_dict[file_path.parts[-2]][file_path.parts[-1]] = peak_dict

        np.save(output_folder + '/' + ferretname + '_'+ pitchshift+ '_peak_dict.npy', big_peak_dict)
        np.save(output_folder + '/' + ferretname + '_'+ pitchshift + '_correlation_dict.npy', big_correlation_dict)
        print('done')

        for file_path in subfolders:
            for talker in talkerlist:
                stream = str(file_path).split('\\')[-1]
                stream = stream[-4:]
                print(stream)
                folder = str(file_path).split('\\')[-2]

                high_units = pd.read_csv(f'G:/neural_chapter/figures/unit_ids_trained_topgenindex_{animal}.csv')
                # remove trailing steam
                rec_name = folder[:-5]
                # find the unique string

                # remove the repeating substring

                # find the units that have the phydir

                # max_length = len(rec_name) // 2
                #
                # for length in range(1, max_length + 1):
                #     for i in range(len(rec_name) - length):
                #         substring = rec_name[i:i + length]
                #         if rec_name.count(substring) > 1:
                #             repeating_substring = substring
                #             break
                #
                # print(repeating_substring)
                rec_name = folder
                high_units = high_units[(high_units['rec_name'] == rec_name) & (high_units['stream'] == stream)]
                clust_ids = high_units['ID'].to_list()
                brain_area = high_units['BrainArea'].to_list()

                plot_average_over_time(file_path, pitchshift, output_folder, ferretname, high_units, talkerinput = 'talker1', animal_id = animal, smooth_option=False)



