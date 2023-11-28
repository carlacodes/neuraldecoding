import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import scipy
import seaborn as sns
from itertools import combinations, permutations





if __name__ == '__main__':
    print('hello')

    big_folder = Path('G:/results_decodingovertime_28112023/F1815_Cruella/')
    animal = big_folder.parts[-1]
    # file_path = 'D:\decodingresults_overtime\F1815_Cruella\lstm_kfold_balac_01092023_cruella/'
    #make the output folder if it doesn't exist

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
    # for file_path in subfolders:
    #     for talker in talkerlist:
    #         # for probeword in stringprobewordlist:
    #         #     print(probeword)
    #         stream = str(file_path).split('\\')[-1]
    #         stream = stream[-4:]
    #         print(stream)
    #         folder = str(file_path).split('\\')[-2]
    #
    #         high_units = pd.read_csv(f'G:/neural_chapter/figures/unit_ids_trained_topgenindex_{animal}.csv')
    #         # remove trailing steam
    #         rec_name = folder[:-5]
    #
    #         rec_name = folder
    #         high_units = high_units[(high_units['rec_name'] == rec_name) & (high_units['stream'] == stream)]
    #         clust_ids = high_units['ID'].to_list()
    #         brain_area = high_units['BrainArea'].to_list()
    #
    #         avg_correlations = calculate_correlation_coefficient(file_path, pitchshift, output_folder, ferretname, talkerinput = 'talker1', clust_ids = clust_ids)
    #         totalcount = totalcount + 1
    #         big_correlation_dict[file_path.parts[-2]][file_path.parts[-1]] = avg_correlations
    #         # find the peak of the score timeseries
    #         peak_dict = find_peak_of_score_timeseries(file_path, pitchshift, output_folder, ferretname, talkerinput = 'talker1', clust_ids = clust_ids)
    #         big_peak_dict[file_path.parts[-2]][file_path.parts[-1]] = peak_dict

    # np.save(output_folder + '/' + ferretname + '_ '+ pitchshift + '_peak_dict.npy', big_peak_dict)
    # np.save(output_folder + '/' + ferretname + '_ '+ pitchshift + '_correlation_dict.npy', big_correlation_dict)
    animal_list_trained = ['F1702_Zola', 'F1815_Cruella']
    all_peak_dict = []
    all_correlation_dict = []
    for animal in animal_list_trained:
        output_folder = f'G:/decodingovertime_figures/{animal}/'
        animal_text = animal.split('_')[1]
        animal_text = animal_text.lower()
        #load the data
        big_peak_dict = np.load(output_folder + '/' + animal_text + '_'+ pitchshift + '_peak_dict.npy', allow_pickle = True)
        big_peak_dict = big_peak_dict[()]
        big_correlation_dict = np.load(output_folder + '/' + animal_text + '_'+ pitchshift + '_correlation_dict.npy', allow_pickle = True)
        big_correlation_dict = big_correlation_dict[()]
        #append the data to a list

        for key in big_peak_dict.keys():
            for key2 in big_peak_dict[key].keys():
                for key3 in big_peak_dict[key][key2].keys():
                    try:
                        all_peak_dict.append(big_peak_dict[key][key2][key3]['std_dev'])
                    except:
                        continue
    all_peak_dict_naive = []
    animal_list_naive = ['F2003_Orecchiette']
    if pitchshift == 'pitchshift':
        pitchshifttext = 'inter-roved F0'
    elif pitchshift == 'nopitchshift':
        pitchshifttext = 'control F0'
    for animal in animal_list_naive:
        output_folder = f'G:/decodingovertime_figures/{animal}/'
        animal_text = animal.split('_')[1]
        animal_text = animal_text.lower()
        #load the data
        big_peak_dict = np.load(output_folder + '/' + animal_text + '_'+ pitchshift + '_peak_dict.npy', allow_pickle = True)
        big_peak_dict = big_peak_dict[()]
        big_correlation_dict = np.load(output_folder + '/' + animal_text + '_'+ pitchshift + '_correlation_dict.npy', allow_pickle = True)
        big_correlation_dict = big_correlation_dict[()]
        #append the data to a list

        for key in big_peak_dict.keys():
            for key2 in big_peak_dict[key].keys():
                for key3 in big_peak_dict[key][key2].keys():
                    try:
                        all_peak_dict_naive.append(big_peak_dict[key][key2][key3]['std_dev'])
                    except:
                        continue
    fig, ax = plt.subplots()
    sns.histplot(all_peak_dict, kde = True,  alpha = 0.5, label = 'trained', color = 'purple')

    sns.histplot(all_peak_dict_naive, kde = True, alpha = 0.5, label = 'naive', color = 'darkcyan')
    # ax.hist(all_peak_dict, bins = 20, alpha = 0.5, label = 'trained', color = 'purple')
    # ax.hist(all_peak_dict_naive, bins = 20, alpha = 0.5, label = 'naive', color = 'darkcyan')
    ax.set_xlabel('standard deviation of peak time of decoding scores', fontsize = 15)
    ax.set_ylabel('count', fontsize = 15)
    ax.legend()
    plt.title(f'Distribution of standard deviation of peak times, {pitchshifttext}', fontsize = 15)
    fig.savefig('G:/decodingovertime_figures/peak_time_std_dev.png')

    plt.show()

    #calculate a mann whitney u test
    print(scipy.stats.mannwhitneyu(all_peak_dict, all_peak_dict_naive))



    print('done')