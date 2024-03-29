import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import seaborn as sns
import scipy.stats as stats
import shap
import lightgbm as lgb
from pathlib import Path
import scipy
from helpers.GeneratePlotsConcise import *
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import json

def find_repeating_substring(text):
    text_length = len(text)
    max_length = text_length // 2  # Maximum possible length of repeating substring

    for length in range(1, max_length + 1):
        for i in range(text_length - 2 * length + 1):
            substring = text[i:i + length]
            rest_of_text = text[i + length:]

            if substring in rest_of_text:
                return substring

    return None



def scatterplot_and_visualise(probewordlist,
                              saveDir='D:/Users/cgriffiths/resultsms4/lstm_output_frommyriad_15012023/lstm_kfold_14012023_crumble',
                              ferretname='Crumble',
                              singleunitlist=[0,1,2],
                              multiunitlist=[0,1,2,3], noiselist=[], stream = 'BB_2', fullid = 'F1901_Crumble', report =[], permutation_scores=False):
    if permutation_scores == False:
        score_key = 'lstm_balanced_avg'
    else:
        score_key = 'perm_bal_ac'
    if probewordlist == [(2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
                             (14, 14)]:
        probewordlist_text = [(15, 15), (42,49), (4, 4), (16, 16), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
                             (14, 14)]
    else:
        probewordlist_text = probewordlist

    singleunitlist = [x - 1 for x in singleunitlist]
    multiunitlist = [x - 1 for x in multiunitlist]
    noiselist = [x - 1 for x in noiselist]
    original_cluster_list = np.empty([0])

    su_pitchshiftlist_female = np.empty([0])
    su_pitchshiftlist_male = np.empty([0])

    su_nonpitchshiftlist_female = np.empty([0])
    su_nonpitchshiftlist_male = np.empty([0])

    mu_pitchshiftlist_female = np.empty([0])
    mu_pitchshiftlist_male = np.empty([0])

    mu_nonpitchshiftlist_female = np.empty([0])
    mu_nonpitchshiftlist_male = np.empty([0])
    cluster_list_male_mu_nops = np.empty([0])
    cluster_list_male_mu = np.empty([0])

    mu_nonpitchshiftlist_male_probeword = np.empty([0])
    mu_nonpitchshiftlist_female_probeword = np.empty([0])

    mu_pitchshiftlist_male_probeword = np.empty([0])
    mu_pitchshiftlist_female_probeword = np.empty([0])


    su_nonpitchshiftlist_male_probeword = np.empty([0])
    su_nonpitchshiftlist_female_probeword = np.empty([0])

    su_pitchshiftlist_male_probeword = np.empty([0])
    su_pitchshiftlist_female_probeword = np.empty([0])



    mu_nonpitchshiftlist_male_unitid = np.empty([0])
    mu_nonpitchshiftlist_female_unitid = np.empty([0])

    mu_pitchshiftlist_male_unitid = np.empty([0])
    mu_pitchshiftlist_female_unitid = np.empty([0])


    su_nonpitchshiftlist_male_unitid = np.empty([0])
    su_nonpitchshiftlist_female_unitid = np.empty([0])

    su_pitchshiftlist_male_unitid = np.empty([0])
    su_pitchshiftlist_female_unitid = np.empty([0])

    mu_nonpitchshiftlist_male_channel_id = np.empty([0])
    mu_nonpitchshiftlist_female_channel_id = np.empty([0])

    mu_pitchshiftlist_male_channel_id= np.empty([0])
    mu_pitchshiftlist_female_channel_id = np.empty([0])

    su_nonpitchshiftlist_male_channel_id = np.empty([0])
    su_nonpitchshiftlist_female_channel_id = np.empty([0])

    su_pitchshiftlist_male_channel_id = np.empty([0])
    su_pitchshiftlist_female_channel_id= np.empty([0])

    for probeword in probewordlist:
        singleunitlist_copy = singleunitlist.copy()
        multiunitlist_copy = multiunitlist.copy()

        #load the original clusters to split from the json file
        json_file_path = f'F:\split_cluster_jsons/{fullid}/cluster_split_list.json'

        if ferretname == 'Orecchiette':
            original_to_split_cluster_ids = np.array([])
        else:
            with open(json_file_path, "r") as json_file:
                loaded_data = json.load(json_file)
            #get the recname
            recname = saveDir.split('/')[-3]
            #get the recname from the json file
            stream_id = stream[-4:]
            if recname == '01_03_2022_cruellabb4bb5':
                recname = '01_03_2022_cruella'
            elif recname == '25_01_2023_cruellabb4bb5':
                recname = '25_01_2023_cruella'
            recname_json =  loaded_data.get(recname)

            #get the cluster ids from the json file
            original_to_split_cluster_ids = recname_json.get(stream_id)
            original_to_split_cluster_ids = original_to_split_cluster_ids.get('cluster_to_split_list')
            if original_to_split_cluster_ids == 'clust_ids':
                #get all the unique clusters ids
                probewordindex = probeword[0]
                stringprobewordindex = str(probewordindex)
                scores = np.load(
                    saveDir + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_bs.npy',
                    allow_pickle=True)[()]
                original_to_split_cluster_ids = np.unique(scores['talker1']['target_vs_probe']['pitchshift']['cluster_id']+scores['talker1']['target_vs_probe']['nopitchshift']['cluster_id'])
                #make sure they are all less than 100
                original_to_split_cluster_ids = [x for x in original_to_split_cluster_ids if x < 100]
            elif original_to_split_cluster_ids == None:
                original_to_split_cluster_ids = np.array([])




        probewordindex = probeword[0]
        print(probewordindex)

        stringprobewordindex = str(probewordindex)
        try:
            scores = np.load(
                saveDir  + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_bs.npy',
                allow_pickle=True)[()]
        except:
            print('file not found: ' + saveDir  + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_bs.npy')
            continue

        for talker in [1]:
            comparisons = [comp for comp in scores[f'talker{talker}']]

            for comp in comparisons:
                for cond in ['pitchshift', 'nopitchshift']:
                    for i, clus in enumerate(scores[f'talker{talker}'][comp][cond]['cluster_id']):
                        #check if clus is greater than 100
                        if 200> clus >= 100:
                            clus_instance = int(round(clus - 100))
                            if clus_instance in singleunitlist_copy:

                                singleunitlist_copy.append(clus)
                                original_cluster_list = np.append(original_cluster_list, clus_instance)
                            elif clus_instance in multiunitlist_copy:

                                multiunitlist_copy.append(clus)
                                original_cluster_list = np.append(original_cluster_list, clus_instance)
                        elif 300 > clus >= 200:
                            clus_instance = int(round(clus - 200))
                            if clus_instance in singleunitlist_copy:

                                singleunitlist_copy.append(clus)
                                original_cluster_list = np.append(original_cluster_list, clus_instance)
                            elif clus_instance in multiunitlist_copy:

                                multiunitlist_copy.append(clus)
                                original_cluster_list = np.append(original_cluster_list, clus_instance)
                        elif 400 > clus >= 300:
                            clus_instance = int(round(clus - 300))
                            if clus_instance in singleunitlist_copy:

                                singleunitlist_copy.append(clus)
                                original_cluster_list = np.append(original_cluster_list, clus_instance)
                            elif clus_instance in multiunitlist_copy:

                                multiunitlist_copy.append(clus)
                                original_cluster_list = np.append(original_cluster_list, clus_instance)
                        else:
                            clus_instance = clus


        #get the unique values from the original cluster list
        original_cluster_list = np.unique(original_cluster_list)
        #remove the original cluster list from the single and multi unit lists
        singleunitlist_copy = [x for x in singleunitlist_copy if x not in original_cluster_list]
        multiunitlist_copy = [x for x in multiunitlist_copy if x not in original_cluster_list]

        #check original_to_split_cluster_ids is not in single or multi unit list
        singleunitlist_copy = [x for x in singleunitlist_copy if x not in original_to_split_cluster_ids]
        multiunitlist_copy = [x for x in multiunitlist_copy if x not in original_to_split_cluster_ids]


        recname = saveDir.split('/')[-3]

        if fullid == 'F2003_Orecchiette':
            try:
                report['tdt'] = pd.read_csv(f'D:\ms4output_16102023\F2003_Orecchiette/' + {stream} + '/' + 'recording_0/pykilosort/report/' + 'unit_list.csv')
                #take only the second column
                report['tdt'] = report['tdt'].iloc[:, 1]

            except:
                # make a column of 0s for the tdt column
                report['tdt'] =  np.zeros((len(report), 1))

        for talker in [1]:
            comparisons = [comp for comp in scores[f'talker{talker}']]

            for comp in comparisons:
                for cond in ['pitchshift', 'nopitchshift']:
                    for i, clus in enumerate(scores[f'talker{talker}'][comp][cond]['cluster_id']):
                        stream_small = stream[-4:]
                        clust_text = str(clus)+'_'+fullid+'_'+recname+'_'+stream_small

                        print(i, clus)

                        if probeword == (2,2) and fullid == 'F1702_Zola':
                            probeword = (4,4)
                        elif probeword == (5,6) and fullid == 'F1702_Zola':
                            probeword = (2,2)
                        elif probeword == (20,20) and fullid == 'F1702_Zola':
                            probeword = (3,3)
                        elif probeword == (42,49) and fullid == 'F1702_Zola':
                            probeword = (5,5)
                        elif probeword == (32,38) and fullid == 'F1702_Zola':
                            probeword = (7,7)

                        if fullid == 'F1604_Squinty' or fullid == 'F1606_Windolene':
                            if probeword == (3,3):
                                probeword = (5,5)
                            elif probeword == (6,6) :
                                probeword = (4,4)
                            elif probeword == (2,2):
                                probeword = (13,13)
                            elif probeword == (4,4):
                                probeword = (15,15)
                            elif probeword == (5,5):
                                probeword = (16,16)
                            elif probeword == (7,7):
                                probeword = (18,18)
                            elif probeword == (8,8):
                                probeword = (19,19)
                            elif probeword == (9,9):
                                probeword = (20,20)
                            elif probeword == (10,10):
                                probeword = (21,21)
                            elif probeword == (11,11):

                                probeword = (22,22)
                            elif probeword == (12,12):

                                probeword = (23,23)
                            elif probeword == (14,14):
                                probeword = (25,25)




                        if 200 > clus >= 100 and fullid != 'F2003_Orecchiette':
                            clus_id_report = clus - 100
                        elif 300> clus >= 200 and fullid != 'F2003_Orecchiette':
                            clus_id_report = clus - 200
                        elif 400 > clus >= 300 and fullid != 'F2003_Orecchiette':
                            clus_id_report = clus - 300
                        else:
                            clus_id_report = clus


                        if clus in singleunitlist_copy:
                            print('in single unit list')

                            if cond == 'pitchshift':
                                if talker == 1:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        su_pitchshiftlist_female = np.append(su_pitchshiftlist_female,
                                                                             scores[f'talker{talker}'][comp][cond][
                                                                                 score_key][i])
                                        su_pitchshiftlist_female_probeword = np.append(su_pitchshiftlist_female_probeword, probeword[talker-1])

                                        su_pitchshiftlist_female_unitid = np.append(su_pitchshiftlist_female_unitid, clust_text)


                                        su_pitchshiftlist_female_channel_id = np.append(su_pitchshiftlist_female_channel_id, report['tdt'][clus_id_report])

                                elif talker == 2:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        su_pitchshiftlist_male = np.append(su_pitchshiftlist_male,
                                                                           scores[f'talker{talker}'][comp][cond][
                                                                               score_key][i])
                                        su_pitchshiftlist_male_probeword = np.append(su_pitchshiftlist_male_probeword, probeword[talker -1 ])
                                        su_pitchshiftlist_male_unitid = np.append(su_pitchshiftlist_male_unitid, clust_text)
                                        su_pitchshiftlist_male_channel_id = np.append(su_pitchshiftlist_male_channel_id, report['tdt'][clus_id_report])



                                # print(pitchshiftlist.size)
                            elif cond == 'nopitchshift':
                                if talker == 1:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        su_nonpitchshiftlist_female = np.append(su_nonpitchshiftlist_female,
                                                                                scores[f'talker{talker}'][comp][cond][
                                                                                    score_key][i])
                                        su_nonpitchshiftlist_female_probeword = np.append(su_nonpitchshiftlist_female_probeword, probeword[talker -1 ])
                                        su_nonpitchshiftlist_female_unitid = np.append(su_nonpitchshiftlist_female_unitid, clust_text)
                                        su_nonpitchshiftlist_female_channel_id = np.append(su_nonpitchshiftlist_female_channel_id, report['tdt'][clus_id_report])

                                elif talker == 2:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        su_nonpitchshiftlist_male = np.append(su_nonpitchshiftlist_male,
                                                                              scores[f'talker{talker}'][comp][cond][
                                                                                  score_key][i])
                                        su_nonpitchshiftlist_male_probeword = np.append(su_nonpitchshiftlist_male_probeword, probeword[talker -1 ])
                                        su_nonpitchshiftlist_male_unitid = np.append(su_nonpitchshiftlist_male_unitid, clust_text)

                                        su_nonpitchshiftlist_male_channel_id = np.append(su_nonpitchshiftlist_male_channel_id, report['tdt'][clus_id_report])


                        elif clus in multiunitlist_copy:
                            if cond == 'pitchshift':
                                if talker == 1:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        mu_pitchshiftlist_female = np.append(mu_pitchshiftlist_female,
                                                                             scores[f'talker{talker}'][comp][cond][
                                                                                 score_key][
                                                                                 i])
                                        mu_pitchshiftlist_female_probeword = np.append(mu_pitchshiftlist_female_probeword, probeword[talker -1 ])
                                        mu_pitchshiftlist_female_unitid = np.append(mu_pitchshiftlist_female_unitid, clust_text)
                                        mu_pitchshiftlist_female_channel_id = np.append(mu_pitchshiftlist_female_channel_id, report['tdt'][clus_id_report])

                                elif talker == 2:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        mu_pitchshiftlist_male = np.append(mu_pitchshiftlist_male,
                                                                           scores[f'talker{talker}'][comp][cond][
                                                                              score_key][
                                                                               i])
                                        cluster_list_male_mu = np.append(cluster_list_male_mu, clust_text)
                                        mu_pitchshiftlist_male_probeword = np.append(mu_pitchshiftlist_male_probeword, probeword[talker -1 ])
                                        mu_pitchshiftlist_male_unitid = np.append(mu_pitchshiftlist_male_unitid, clust_text)
                                        mu_pitchshiftlist_male_channel_id = np.append(mu_pitchshiftlist_male_channel_id, report['tdt'][clus_id_report])



                            if cond == 'nopitchshift':
                                if talker == 1:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        mu_nonpitchshiftlist_female = np.append(mu_nonpitchshiftlist_female,
                                                                                scores[f'talker{talker}'][comp][cond][
                                                                                    score_key][i])
                                        mu_nonpitchshiftlist_female_probeword = np.append(mu_nonpitchshiftlist_female_probeword, probeword[talker -1 ])
                                        mu_nonpitchshiftlist_female_unitid = np.append(mu_nonpitchshiftlist_female_unitid, clust_text)
                                        mu_nonpitchshiftlist_female_channel_id = np.append(mu_nonpitchshiftlist_female_channel_id, report['tdt'][clus_id_report])

                                elif talker == 2:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        mu_nonpitchshiftlist_male = np.append(mu_nonpitchshiftlist_male,
                                                                              scores[f'talker{talker}'][comp][cond][
                                                                                  score_key][i])
                                        cluster_list_male_mu_nops= np.append(cluster_list_male_mu_nops, clust_text)
                                        mu_nonpitchshiftlist_male_probeword = np.append(mu_nonpitchshiftlist_male_probeword, probeword[talker -1 ])
                                        mu_nonpitchshiftlist_male_unitid = np.append(mu_nonpitchshiftlist_male_unitid, clust_text)
                                        mu_nonpitchshiftlist_male_channel_id = np.append(mu_nonpitchshiftlist_male_channel_id, report['tdt'][clus_id_report])



                        elif clus in noiselist:
                            pass


    dictofsortedscores = {'su_list': {'pitchshift': {'female_talker': {},
                                                     'male_talker': {}},
                                      'nonpitchshift': {'female_talker': {},
                                                        'male_talker': {}}},
                          'mu_list': {'pitchshift': {'female_talker': {},
                                                     'male_talker': {}},
                                      'nonpitchshift': {'female_talker': {},
                                                        'male_talker': {}}},

                          'su_list_probeword': {'pitchshift': {'female_talker': {},
                                                     'male_talker': {}},
                                      'nonpitchshift': {'female_talker': {},
                                                        'male_talker': {}}},
                          'mu_list_probeword': {'pitchshift': {'female_talker': {},
                                                     'male_talker': {}},
                                      'nonpitchshift': {'female_talker': {},
                                                        'male_talker': {}}},
                          'su_list_unitid': {'pitchshift': {'female_talker': {},
                                                               'male_talker': {}},
                                                'nonpitchshift': {'female_talker': {},
                                                                  'male_talker': {}}},
                          'mu_list_unitid': {'pitchshift': {'female_talker': {},
                                                               'male_talker': {}},
                                                'nonpitchshift': {'female_talker': {},
                                                                  'male_talker': {}}},
                          'su_list_chanid': {'pitchshift': {'female_talker': {},
                                                            'male_talker': {}},
                                             'nonpitchshift': {'female_talker': {},
                                                               'male_talker': {}}},
                          'mu_list_chanid': {'pitchshift': {'female_talker': {},
                                                            'male_talker': {}},
                                             'nonpitchshift': {'female_talker': {},
                                                               'male_talker': {}}}

                         }
    if len(su_pitchshiftlist_female_probeword) != len(su_pitchshiftlist_female):
        print('not equal')

    dictofsortedscores['su_list']['pitchshift']['female_talker'] = su_pitchshiftlist_female
    dictofsortedscores['su_list']['pitchshift']['male_talker'] = su_pitchshiftlist_male
    dictofsortedscores['su_list']['nonpitchshift']['female_talker'] = su_nonpitchshiftlist_female
    dictofsortedscores['su_list']['nonpitchshift']['male_talker'] = su_nonpitchshiftlist_male

    dictofsortedscores['mu_list']['pitchshift']['female_talker'] = mu_pitchshiftlist_female
    dictofsortedscores['mu_list']['pitchshift']['male_talker'] = mu_pitchshiftlist_male
    dictofsortedscores['mu_list']['nonpitchshift']['female_talker'] = mu_nonpitchshiftlist_female
    dictofsortedscores['mu_list']['nonpitchshift']['male_talker'] = mu_nonpitchshiftlist_male

    dictofsortedscores['su_list_probeword']['pitchshift']['female_talker'] = su_pitchshiftlist_female_probeword
    dictofsortedscores['su_list_probeword']['pitchshift']['male_talker']  = su_pitchshiftlist_male_probeword
    dictofsortedscores['su_list_probeword']['nonpitchshift']['female_talker'] = su_nonpitchshiftlist_female_probeword
    dictofsortedscores['su_list_probeword']['nonpitchshift']['male_talker'] = su_nonpitchshiftlist_male_probeword

    dictofsortedscores['mu_list_probeword']['pitchshift']['female_talker'] = mu_pitchshiftlist_female_probeword
    dictofsortedscores['mu_list_probeword']['pitchshift']['male_talker'] = mu_pitchshiftlist_male_probeword
    dictofsortedscores['mu_list_probeword']['nonpitchshift']['female_talker'] = mu_nonpitchshiftlist_female_probeword
    dictofsortedscores['mu_list_probeword']['nonpitchshift']['male_talker'] = mu_nonpitchshiftlist_male_probeword

    dictofsortedscores['su_list_unitid']['pitchshift']['female_talker'] = su_pitchshiftlist_female_unitid
    dictofsortedscores['su_list_unitid']['pitchshift']['male_talker']  = su_pitchshiftlist_male_unitid
    dictofsortedscores['su_list_unitid']['nonpitchshift']['female_talker'] = su_nonpitchshiftlist_female_unitid
    dictofsortedscores['su_list_unitid']['nonpitchshift']['male_talker'] = su_nonpitchshiftlist_male_unitid

    dictofsortedscores['mu_list_unitid']['pitchshift']['female_talker'] = mu_pitchshiftlist_female_unitid
    dictofsortedscores['mu_list_unitid']['pitchshift']['male_talker'] = mu_pitchshiftlist_male_unitid
    dictofsortedscores['mu_list_unitid']['nonpitchshift']['female_talker'] = mu_nonpitchshiftlist_female_unitid
    dictofsortedscores['mu_list_unitid']['nonpitchshift']['male_talker'] = mu_nonpitchshiftlist_male_unitid

    dictofsortedscores['su_list_chanid']['pitchshift']['female_talker'] = su_pitchshiftlist_female_channel_id
    dictofsortedscores['su_list_chanid']['pitchshift']['male_talker']  = su_pitchshiftlist_male_channel_id
    dictofsortedscores['su_list_chanid']['nonpitchshift']['female_talker'] = su_nonpitchshiftlist_female_channel_id
    dictofsortedscores['su_list_chanid']['nonpitchshift']['male_talker'] = su_nonpitchshiftlist_male_channel_id

    dictofsortedscores['mu_list_chanid']['pitchshift']['female_talker'] = mu_pitchshiftlist_female_channel_id
    dictofsortedscores['mu_list_chanid']['pitchshift']['male_talker'] = mu_pitchshiftlist_male_channel_id
    dictofsortedscores['mu_list_chanid']['nonpitchshift']['female_talker'] = mu_nonpitchshiftlist_female_channel_id
    dictofsortedscores['mu_list_chanid']['nonpitchshift']['male_talker'] = mu_nonpitchshiftlist_male_channel_id



    if len( dictofsortedscores['su_list_probeword']['pitchshift']['female_talker']) != len(  dictofsortedscores['su_list']['pitchshift']['female_talker']):
        print('error in dict')


    return dictofsortedscores



def cool_dict_merge(dicts_list): #ripped this from stackoverflow
    d = {**dicts_list[0]}
    for entry in dicts_list[1:]:
        for k, v in entry.items():
            d[k] = ([d[k], v] if k in d and type(d[k]) != list
                    else [*d[k], v] if k in d
            else v)
    return d


def data_merge(a, b):
    """merges b into a and return merged result

    NOTE: tuples and arbitrary objects are not handled as it is totally ambiguous what should happen"""
    key = None
    # ## debug output
    # sys.stderr.write("DEBUG: %s to %s\n" %(b,a))
    if a is None or isinstance(a, str) or isinstance(a, int) or isinstance(a, float):
        # border case for first run or if a is a primitive
        a = b
    elif isinstance(a, list):
        # lists can be only appended
        if isinstance(b, list):
            # merge lists
            a.extend(b)
        else:
            # append to list
            a.append(b)
    elif isinstance(a, dict):
        # dicts must be merged
        if isinstance(b, dict):
            for key in b:
                if key in a:
                    a[key] = data_merge(a[key], b[key])
                else:
                    a[key] = b[key]

    return a

def runboostedregressiontreeforlstmscore(df_use):
    col = 'score'
    dfx = df_use.loc[:, df_use.columns != col]
    # remove ferret as possible feature
    col = 'ferret'
    dfx = dfx.loc[:, dfx.columns != col]

    X_train, X_test, y_train, y_test = train_test_split(dfx, df_use['score'], test_size=0.2,
                                                        random_state=42)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test)

    param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    # bst = xgb.train(param, dtrain, num_round, evallist)
    xg_reg = lgb.LGBMRegressor(colsample_bytree=0.3, learning_rate=0.1,
                               max_depth=10, alpha=10, n_estimators=10, verbose=1)

    xg_reg.fit(X_train, y_train, eval_metric='MSE', verbose=1)
    ypred = xg_reg.predict(X_test)
    lgb.plot_importance(xg_reg)
    plt.title('feature importances for the lstm decoding score  model')

    plt.show()

    kfold = KFold(n_splits=10)
    results = cross_val_score(xg_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)
    results_TEST = cross_val_score(xg_reg, X_test, y_test, scoring='neg_mean_squared_error', cv=kfold)

    mse = mean_squared_error(ypred, y_test)
    print("neg MSE on test set: %.2f" % (np.mean(results_TEST)*100))
    print("negative MSE on train set: %.2f%%" % (np.mean(results) * 100.0))
    print(results)
    shap_values = shap.TreeExplainer(xg_reg).shap_values(dfx)
    fig, ax = plt.subplots(figsize=(15, 15))
    # title kwargs still does nothing so need this workaround for summary plots

    fig, ax = plt.subplots(1, figsize=(10, 10), dpi = 300)

    shap.summary_plot(shap_values,dfx,  max_display=20)
    plt.show()


    labels = [item.get_text() for item in ax.get_yticklabels()]
    print(labels)


def load_classified_report(path):
    ''' Load the classified report
    :param path: path to the report
    :return: classified report
    '''
    #join the path to the report
    if 'F2003_Orecchiette' in path:
        report_path = os.path.join(path, 'quality metrics.csv')
        #combine the paths

        report = pd.read_csv(report_path)
        if 's2cgmod' in path:
            channelpositions = pd.read_pickle(
                r'D:\spkvisanddecodeproj2/analysisscriptsmodcg/visualisation\channelpositions\F2003_Orecchiette/channelpos.pkl')
            # remove rows that are not the cluster id, represented by the first column in the np array out of three columns
            #remove rows that are below the auditory cortex
                    # get the x and y coordinates
            #only get clusters, first column that are greater than 3200 in depth
            clusters_above_hpc = channelpositions[channelpositions[:, 2] > 3200]
            #add one to the cluster ids
            clusters_above_hpc[:, 0] = clusters_above_hpc[:, 0] + 1
            singleunitlist = [1, 19, 21, 219, 227]
            #multiunit list all the other clusters in clusters above hpc
            multiunitlist = [x for x in clusters_above_hpc[:, 0] if x not in singleunitlist]
            noiselist = []
        else:
            if 's2' in str(path):
                channel_pos = os.path.join(path, 'channelpos_s2.csv')
            elif 's3' in path:
                channel_pos = os.path.join(path, 'channelpos_s3.csv')
            unit_list = pd.read_csv(os.path.join(path, 'unit_list.csv'), delimiter='\t')
            #remove the AP from each
            # combine the paths
            channel_pos = pd.read_csv(channel_pos)
            report = pd.read_csv(report_path)  # get the list of multi units and single units
            # the column is called unit_type
            multiunitlist = []
            singleunitlist = []
            noiselist = []

            # get the list of multi units and single units
            for i in range(0, len(report)):
                channel_id = unit_list['max_on_channel_id'][i]
                #remove the AP from the channel id str

                channel_id = channel_id.replace('AP', '')
                channel_id = int(channel_id)
                #get that tow from the channel pos

                row = channel_pos.iloc[channel_id-1]
                if row[1] >= 3200 and report['d_prime'][i] < 4:
                    singleunitlist.append(i+1)
                elif row[1] >= 3200 and report['d_prime'][i] >= 4:
                    multiunitlist.append(i+1)
                else:
                    noiselist.append(i+1)



    else:
        report_path = os.path.join(path, 'quality_metrics_classified.csv')
        #combine the paths

        report = pd.read_csv(report_path)    #get the list of multi units and single units
        #the column is called unit_type
        multiunitlist = []
        singleunitlist = []
        noiselist = []

        #get the list of multi units and single units
        for i in range(0, len(report['unit_type'])):
            if report['unit_type'][i] == 'mua':
                multiunitlist.append(i+1)
            elif report['unit_type'][i] == 'su':
                singleunitlist.append(i+1)
            elif report['unit_type'][i] == 'trash':
                noiselist.append(i+1)


    return report, singleunitlist, multiunitlist, noiselist

def main():
    probewordlist_zola = [(2, 2), (5, 6), (42, 49), (32, 38), (20, 22)]
    probewordlist = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
    probewordlist_l74 = [(10, 10), (2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (8, 8), (9, 9), (11, 11), (12, 12),
                         (14, 14)]
    animal_list = ['F1604_Squinty', 'F1901_Crumble', 'F1606_Windolene', 'F1702_Zola', 'F1815_Cruella',
                   'F1902_Eclair', 'F1812_Nala', 'F2003_Orecchiette', ]

    report = {}
    singleunitlist = {}
    multiunitlist = {}
    noiselist = {}
    path_list = {}

    for animal in animal_list:
        if animal == 'F2003_Orecchiette':
            path = Path('G:\F2003_Orecchiette/')
        else:
            path = Path('D:\ms4output_16102023/' + animal + '/')
        path_list[animal] = [path for path in path.glob('**/quality metrics.csv')]
        # get the parent directory of each path
        path_list[animal] = [path.parent for path in path_list[animal]]
    # report, singleunitlist, and multiunitlist and noiselist need to be modified to include the recname
    for animal in animal_list:
        report[animal] = {}
        singleunitlist[animal] = {}
        multiunitlist[animal] = {}
        noiselist[animal] = {}

        for path in path_list[animal]:
            stream_name = path.parent.absolute()
            stream_name = stream_name.parent.absolute()
            if animal == 'F2003_Orecchiette':
                stream_name = str(stream_name).split('\\')[-2]
            else:
                stream_name = str(stream_name).split('\\')[-1]

            report[animal][stream_name], singleunitlist[animal][stream_name], multiunitlist[animal][stream_name], \
            noiselist[animal][stream_name] = load_classified_report(f'{path}')

    # now create a dictionary of dictionaries, where the first key is the animal name, and the second key is the stream name
    # the value is are the decoding scores for each cluster

    dictoutput = {}
    dictoutput_trained = []
    dictoutput_trained_permutation = []
    dictoutput_naive = []
    dictoutput_naive_permutation = []
    dictoutput_all = []

    dictoutput_all_permutation = []

    for animal in animal_list:
        dictoutput[animal] = {}
        for stream in report[animal]:
            dictoutput[animal][stream] = {}
            animal_text = animal.split('_')[1]

            if animal == 'F2003_Orecchiette':
                rec_name_unique = stream

            else:
                # if stream contains BB_2
                if 'BB_2' in stream:
                    streamtext = 'bb2'
                elif 'BB_3' in stream:
                    streamtext = 'bb3'
                elif 'BB_4' in stream:
                    streamtext = 'bb4'
                elif 'BB_5' in stream:
                    streamtext = 'bb5'
                # remove F number character from animal name

                max_length = len(stream) // 2

                for length in range(1, max_length + 1):
                    for i in range(len(stream) - length):
                        substring = stream[i:i + length]
                        if stream.count(substring) > 1:
                            repeating_substring = substring
                            break

                print(repeating_substring)
                rec_name_unique = repeating_substring[0:-1]
            # rec_name_unique = stream.split('_')[0:3]

            if animal == 'F1604_Squinty':
                # try:
                dictoutput_instance = scatterplot_and_visualise(probewordlist_l74,
                                                                saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream=stream,
                                                                fullid=animal, report=report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist_l74,
                                                                            saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                            ferretname=animal_text,
                                                                            singleunitlist=singleunitlist[animal][
                                                                                stream],
                                                                            multiunitlist=multiunitlist[animal][
                                                                                stream],
                                                                            noiselist=noiselist[animal][stream],
                                                                            stream=stream, fullid=animal,
                                                                            report=report[animal][stream],
                                                                            permutation_scores=True)
                dictoutput_all_permutation.append(dictoutput_instance_permutation)

            elif animal == 'F1606_Windolene':
                dictoutput_instance = scatterplot_and_visualise(probewordlist_l74,
                                                                saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream=stream,
                                                                fullid=animal, report=report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist_l74,
                                                                            saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                            ferretname=animal_text,
                                                                            singleunitlist=singleunitlist[animal][
                                                                                stream],
                                                                            multiunitlist=multiunitlist[animal][
                                                                                stream],
                                                                            noiselist=noiselist[animal][stream],
                                                                            stream=stream, fullid=animal,
                                                                            report=report[animal][stream],
                                                                            permutation_scores=True)
                dictoutput_all_permutation.append(dictoutput_instance_permutation)

            elif animal == 'F1702_Zola':
                dictoutput_instance = scatterplot_and_visualise(probewordlist_zola,
                                                                saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream=stream,
                                                                fullid=animal, report=report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist_zola,
                                                                            saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                            ferretname=animal_text,
                                                                            singleunitlist=singleunitlist[animal][
                                                                                stream],
                                                                            multiunitlist=multiunitlist[animal][
                                                                                stream],
                                                                            noiselist=noiselist[animal][stream],
                                                                            stream=stream, fullid=animal,
                                                                            report=report[animal][stream]
                                                                            , permutation_scores=True)
                dictoutput_all_permutation.append(dictoutput_instance_permutation)

            elif animal == 'F1815_Cruella' or animal == 'F1902_Eclair':
                dictoutput_instance = scatterplot_and_visualise(probewordlist,
                                                                saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream=stream,
                                                                fullid=animal, report=report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist,
                                                                            saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                            ferretname=animal_text,
                                                                            singleunitlist=singleunitlist[animal][
                                                                                stream],
                                                                            multiunitlist=multiunitlist[animal][
                                                                                stream],
                                                                            noiselist=noiselist[animal][stream],
                                                                            stream=stream, fullid=animal,
                                                                            report=report[animal][stream]
                                                                            , permutation_scores=True)
                dictoutput_all_permutation.append(dictoutput_instance_permutation)

            elif animal == 'F2003_Orecchiette':
                # try:
                dictoutput_instance = scatterplot_and_visualise(probewordlist,
                                                                saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream=stream,
                                                                fullid=animal,
                                                                report=report[animal][stream]
                                                                )
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist,
                                                                            saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/',
                                                                            ferretname=animal_text,
                                                                            singleunitlist=singleunitlist[animal][
                                                                                stream],
                                                                            multiunitlist=multiunitlist[animal][
                                                                                stream],
                                                                            noiselist=noiselist[animal][stream],
                                                                            stream=stream,
                                                                            fullid=animal,
                                                                            report=report[animal][stream],
                                                                            permutation_scores=True
                                                                            )
                dictoutput_all_permutation.append(dictoutput_instance_permutation)

            else:
                # try:
                dictoutput_instance = scatterplot_and_visualise(probewordlist,
                                                                saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream=stream,
                                                                fullid=animal, report=report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist,
                                                                            saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                            ferretname=animal_text,
                                                                            singleunitlist=singleunitlist[animal][
                                                                                stream],
                                                                            multiunitlist=multiunitlist[animal][
                                                                                stream],
                                                                            noiselist=noiselist[animal][stream],
                                                                            stream=stream, fullid=animal,
                                                                            report=report[animal][stream]
                                                                            , permutation_scores=True)

                dictoutput_all_permutation.append(dictoutput_instance_permutation)

            female_talker_len = len(dictoutput_instance['su_list']['pitchshift']['female_talker'])
            probeword_len = len(dictoutput_instance['su_list_probeword']['pitchshift']['female_talker'])

            assert female_talker_len == probeword_len, f"Length mismatch: female_talker_len={female_talker_len}, probeword_len={probeword_len}"

            try:
                if animal == 'F1604_Squinty' or animal == 'F1606_Windolene' or animal == 'F1702_Zola' or animal == 'F1815_Cruella':
                    print('trained animal' + animal)
                    dictoutput_trained.append(dictoutput_instance)
                    dictoutput_trained_permutation.append(dictoutput_instance_permutation)
                else:
                    print('naive animal:' + animal)
                    dictoutput_naive.append(dictoutput_instance)
                    dictoutput_naive_permutation.append(dictoutput_instance_permutation)
            except:
                print('no scores for this stream')
                pass

    labels = ['F1901_Crumble', 'F1604_Squinty', 'F1606_Windolene', 'F1702_Zola', 'F1815_Cruella', 'F1902_Eclair',
              'F1812_Nala']

    colors = ['purple', 'magenta', 'darkturquoise', 'olivedrab', 'steelblue', 'darkcyan', 'darkorange']

    generate_plots(dictoutput_all, dictoutput_trained, dictoutput_naive,  labels, colors)

    return

    # return


def generate_plots(dictlist, dictlist_trained, dictlist_naive, labels, colors):
    # labels = ['cruella', 'zola', 'nala', 'crumble', 'eclair', 'ore']
    # colors = ['purple', 'magenta', 'darkturquoise', 'olivedrab', 'steelblue', 'darkcyan']


    fig, ax = plt.subplots(1, figsize=(5, 8))
    emptydict = {}
    count = 0
    # for dictoutput in dictlist:
    #     for sutype in ['su_list', 'mu_list']:
    #         for pitchshiftornot in dictoutput[sutype].keys():
    #             for talker in dictoutput[sutype][pitchshiftornot].keys():
    #                 for item in dictoutput[sutype][pitchshiftornot][talker]:
    #                     if count == 0 or count ==1:
    #                         emptydict['trained'] = emptydict.get('trained', []) + [1]
    #                     else:
    #                         emptydict['trained'] = emptydict.get('trained', []) + [0]
    #                     emptydict['ferret']= emptydict.get('ferret', []) + [count]
    #                     emptydict['score'] = emptydict.get('score', []) + [item]
    #                     if talker == 'female_talker':
    #                         emptydict['male_talker'] = emptydict.get('male_talker', []) + [0]
    #                     else:
    #                         emptydict['male_talker'] = emptydict.get('male_talker', []) + [1]
    #                     if pitchshiftornot == 'pitchshift':
    #                         emptydict['pitchshift'] = emptydict.get('pitchshift', []) + [1]
    #                     else:
    #                         emptydict['pitchshift'] = emptydict.get('pitchshift', []) + [0]
    #                     if sutype == 'su_list':
    #                         emptydict['su'] = emptydict.get('su', []) + [1]
    #                     else:
    #                         emptydict['su'] = emptydict.get('su', []) + [0]
    #     count += 1
    # for keys in emptydict.keys():
    #     emptydict[keys] = np.asarray(emptydict[keys])
    #
    #
    # for dictoutput in dictlist:
    #     for key in ['su_list', 'mu_list']:
    #         for key3 in dictoutput[key]['pitchshift'].keys():
    #             if len(dictoutput[key]['nonpitchshift'][key3]) < len(
    #                     dictoutput[key]['pitchshift'][key3]):
    #                 dictoutput[key]['pitchshift'][key3] = \
    #                     dictoutput[key]['pitchshift'][key3][
    #                     :len(dictoutput[key]['nonpitchshift'][key3])]
    #             elif len(dictoutput[key]['nonpitchshift'][key3]) > len(
    #                     dictoutput[key]['pitchshift'][key3]):
    #                 dictoutput[key]['nonpitchshift'][key3] = \
    #                     dictoutput[key]['nonpitchshift'][key3][
    #                     :len(dictoutput[key]['pitchshift'][key3])]

    bigconcatenatetrained_ps = np.empty(0)
    bigconcatenatetrained_nonps = np.empty(0)
    for dictouput in dictlist_trained:
        for key in ['su_list', 'mu_list']:
            for key3 in dictouput[key]['pitchshift'].keys():
                bigconcatenatetrained_ps = np.concatenate(
                    (bigconcatenatetrained_ps, dictouput[key]['pitchshift'][key3]))
                bigconcatenatetrained_nonps = np.concatenate(
                    (bigconcatenatetrained_nonps, dictouput[key]['nonpitchshift'][key3]))


    bigconcatenatenaive_ps = np.empty(0)
    bigconcatenatenaive_nonps = np.empty(0)

    for dictouput in dictlist_naive:
        for key in ['su_list', 'mu_list']:
            # print(key, 'key')
            for key3 in dictouput[key]['pitchshift'].keys():
                # print(key3, 'key3')
                bigconcatenatenaive_ps = np.concatenate((bigconcatenatenaive_ps, dictouput[key]['pitchshift'][key3]))
                bigconcatenatenaive_nonps = np.concatenate(
                    (bigconcatenatenaive_nonps, dictouput[key]['nonpitchshift'][key3]))

    #remake dictionary based on probe word
    emptydict = {}
    count = 0
    probewordlist = [(2, 2), (5, 6), (42, 49), (32, 38), (20, 22)]
    probewordlist_text = [(2, 2), (5, 6), (42, 49), (32, 38), (20, 22), (15, 15), (42, 49), (4, 4), (16, 16), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
                          (14, 14)]

    scoredict = {}
    scoredict_naive={}
    scoredict['(2,2)'] = {}
    scoredict['(2,2)']['female_talker'] = []
    scoredict['(5,6)'] = {}
    scoredict['(5,6)']['female_talker'] = []

    scoredict['(42,49)'] = {}
    scoredict['(42,49)']['female_talker'] = []

    scoredict['(32,38)'] = {}
    scoredict['(32,38)']['female_talker']  = []

    scoredict['(20,22)'] = {}
    scoredict['(20,22)']['female_talker'] =[]
    for probeword_range in probewordlist_text:
        probeword_key = f'({probeword_range[0]},{probeword_range[1]})'
        scoredict[probeword_key] = {}
        scoredict[probeword_key]['female_talker'] = {}
        scoredict[probeword_key]['female_talker']['nonpitchshift'] = {}
        scoredict[probeword_key]['female_talker']['pitchshift'] = {}

        scoredict[probeword_key]['female_talker']['nonpitchshift']['su_list'] = []
        scoredict[probeword_key]['female_talker']['pitchshift']['su_list'] = []

        scoredict[probeword_key]['female_talker']['nonpitchshift']['mu_list'] = []
        scoredict[probeword_key]['female_talker']['pitchshift']['mu_list'] = []


        scoredict_naive[probeword_key] = {}
        scoredict_naive[probeword_key]['female_talker'] = {}
        scoredict_naive[probeword_key]['female_talker']['nonpitchshift'] = {}
        scoredict_naive[probeword_key]['female_talker']['pitchshift'] = {}

        scoredict_naive[probeword_key]['female_talker']['nonpitchshift']['su_list'] = []
        scoredict_naive[probeword_key]['female_talker']['pitchshift']['su_list'] = []

        scoredict_naive[probeword_key]['female_talker']['nonpitchshift']['mu_list'] = []
        scoredict_naive[probeword_key]['female_talker']['pitchshift']['mu_list'] = []




    # for talker in [1,2]:
    #     for probeword in [1,2,3,4,5]:
    #         for dict in dictlist_trained:
    #             for key in dict['su_list_probeword']:
    #                 probewords = dict['su_list_probeword'][key]
    #                 count = 0
    #                 for probeword in probewords:
    #                     scoredict[int(probeword)] = dict['su_list'][key][count]
    #                     count += 1

    # for talker in [1]:
    #     if talker == 1:
    #         talker_key = 'female_talker'
    #     for i, dict in enumerate(dictlist_trained):
    #         for key in dict['su_list_probeword']:
    #             probewords = dict['su_list_probeword'][key][talker_key]
    #             count = 0
    #             for probeword in probewords:
    #                 if int(probeword) == 2:
    #                     probewordtext = '(2,2)'
    #                 elif int(probeword) == 5:
    #                     probewordtext = '(5,6)'
    #                 elif int(probeword) == 42:
    #                     probewordtext = '(42,49)'
    #                 elif int(probeword) == 32:
    #                     probewordtext = '(32,38)'
    #                 elif int(probeword) == 20:
    #                     probewordtext = '(20,22)'
    #
    #                 scoredict[probewordtext][talker_key].append(dict['su_list'][key][talker_key][count])
    #                 count = count + 1
    probeword_to_text = {
        2: '(2,2)',
        5: '(5,6)',
        42: '(42,49)',
        32: '(32,38)',
        20: '(20,22)',
        15: '(15,15)',
        4: '(4,4)',
        16: '(16,16)',
        7: '(7,7)',
        8: '(8,8)',
        9: '(9,9)',
        10: '(10,10)',
        11: '(11,11)',
        12: '(12,12)',
        14: '(14,14)'
    }
    for talker in [1]:
        if talker == 1:
            talker_key = 'female_talker'
        for i, dict in enumerate(dictlist_trained):
            for key in dict['su_list_probeword']:
                probewords = dict['su_list_probeword'][key][talker_key]
                count = 0
                for probeword in probewords:
                    probeword_range = int(probeword)
                    probewordtext = probeword_to_text.get(probeword_range)
                    if probewordtext:
                        scoredict[probewordtext][talker_key][key]['su_list'].append(dict['su_list'][key][talker_key][count])
                    count = count + 1
        for key in dict['mu_list_probeword']:
            probewords = dict['mu_list_probeword'][key][talker_key]
            count = 0
            for probeword in probewords:
                probeword_range = int(probeword)
                probewordtext = probeword_to_text.get(probeword_range)
                if probewordtext:
                    scoredict[probewordtext][talker_key][key]['mu_list'].append(dict['mu_list'][key][talker_key][count])
                count = count + 1

    #plot each mean across probeword as a bar plot
    fig, ax = plt.subplots(1, figsize=(10, 10), dpi=300)
    plot_count = 0
    for probeword in scoredict.keys():
        su_list_nops = scoredict[probeword]['female_talker']['nonpitchshift']['su_list']
        mu_list_nops = scoredict[probeword]['female_talker']['nonpitchshift']['mu_list']
        #get the mean of the su_list and mu_list
        total_control = su_list_nops + mu_list_nops
        mean = np.mean(total_control)
        std = np.std(total_control)

        #do the same for the pitchshift
        su_list_ps = scoredict[probeword]['female_talker']['pitchshift']['su_list']
        mu_list_ps = scoredict[probeword]['female_talker']['pitchshift']['mu_list']
        total_rove = su_list_ps + mu_list_ps
        mean_rove = np.mean(total_rove)
        std_rove = np.std(total_rove)
        #plot the bar plot

        if plot_count ==0:
            ax.bar(plot_count, mean, yerr=std, color='purple', alpha=0.5, label='control')
            plot_count += 1
            ax.bar(plot_count, mean_rove, yerr=std_rove, color='pink', alpha=0.5, label='rove')
        else:

            ax.bar(plot_count, mean, yerr = std, color = 'purple', alpha = 0.5, label =None)
            plot_count += 1
            ax.bar(plot_count, mean_rove, yerr = std_rove, color = 'pink', alpha = 0.5, label = None)

        plot_count += 1
    plt.legend(fontsize = 8)
    plt.xlabel('probe word')
    ax.set_xticks(np.arange(0, 32, 2))
    ax.set_xticklabels([(2, 2), (5, 6), (42, 49), (32, 38), (20, 22), (15, 15), (42, 49), (4, 4), (16, 16), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
                          (14, 14)], rotation = 45)

    plt.show()
## do the same for the naive animals
    for talker in [1]:
        if talker == 1:
            talker_key = 'female_talker'
        for i, dict in enumerate(dictlist_naive):
            for key in dict['su_list_probeword']:
                probewords = dict['su_list_probeword'][key][talker_key]
                count = 0
                for probeword in probewords:
                    probeword_range = int(probeword)
                    probewordtext = probeword_to_text.get(probeword_range)
                    if probewordtext:
                        scoredict_naive[probewordtext][talker_key][key]['su_list'].append(dict['su_list'][key][talker_key][count])
                    count = count + 1
        for key in dict['mu_list_probeword']:
            probewords = dict['mu_list_probeword'][key][talker_key]
            count = 0
            for probeword in probewords:
                probeword_range = int(probeword)
                probewordtext = probeword_to_text.get(probeword_range)
                if probewordtext:
                    scoredict_naive[probewordtext][talker_key][key]['mu_list'].append(dict['mu_list'][key][talker_key][count])
                count = count + 1

    #plot each mean across probeword as a bar plot
    fig, ax = plt.subplots(1, figsize=(10, 10), dpi=300)
    plot_count = 0
    for probeword in scoredict.keys():
        su_list_nops = scoredict_naive[probeword]['female_talker']['nonpitchshift']['su_list']
        mu_list_nops = scoredict_naive[probeword]['female_talker']['nonpitchshift']['mu_list']
        #get the mean of the su_list and mu_list
        total_control = su_list_nops + mu_list_nops
        mean = np.mean(total_control)
        std = np.std(total_control)

        #do the same for the pitchshift
        su_list_ps = scoredict_naive[probeword]['female_talker']['pitchshift']['su_list']
        mu_list_ps = scoredict_naive[probeword]['female_talker']['pitchshift']['mu_list']
        total_rove = su_list_ps + mu_list_ps
        mean_rove = np.mean(total_rove)
        std_rove = np.std(total_rove)
        #plot the bar plot
        if plot_count ==0:
            ax.bar(plot_count, mean, yerr=std, color='cyan', alpha=0.5, label='control')
            plot_count += 1
            ax.bar(plot_count, mean_rove, yerr=std_rove, color='blue', alpha=0.5, label='rove')
        else:

            ax.bar(plot_count, mean, yerr = std, color = 'cyan', alpha = 0.5, label =None)
            plot_count += 1
            ax.bar(plot_count, mean_rove, yerr = std_rove, color = 'blue', alpha = 0.5, label = None)

        plot_count += 1
    plt.legend( fontsize = 8)
    plt.xlabel('probe word')
    # ax.set_xticks(np.arange(0, 16, 1))
    ax.set_xticklabels([(2, 2), (5, 6), (42, 49), (32, 38), (20, 22), (15, 15), (42, 49), (4, 4), (16, 16), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
                          (14, 14)])
    plt.show()

    #swarm plot equivalent
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    data = []
    x_positions = []
    hue = []

    # Iterate over the keys in your dictionary
    for probeword in scoredict.keys():
        su_list_nops = scoredict_naive[probeword]['female_talker']['nonpitchshift']['su_list']
        mu_list_nops = scoredict_naive[probeword]['female_talker']['nonpitchshift']['mu_list']
        total_control = su_list_nops + mu_list_nops

        su_list_ps = scoredict_naive[probeword]['female_talker']['pitchshift']['su_list']
        mu_list_ps = scoredict_naive[probeword]['female_talker']['pitchshift']['mu_list']
        total_rove = su_list_ps + mu_list_ps

        # Create a DataFrame for seaborn
        control_df = pd.DataFrame({'Data': total_control, 'Probe Word': probeword, 'Category': 'Control'})
        rove_df = pd.DataFrame({'Data': total_rove, 'Probe Word': probeword, 'Category': 'Rove'})

        # Append the data and category
        data.extend(total_control)
        data.extend(total_rove)
        x_positions.extend([probeword] * (len(total_control) + len(total_rove)))
        hue.extend(['Control'] * len(total_control))
        hue.extend(['Rove'] * len(total_rove))

    # Create the violin plot
    sns.violinplot(x=x_positions, y=data, hue=hue, palette={"Control": "cyan", "Rove": "blue"}, split=True, ax=ax)

    # Customize the plot
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.xlabel('Probe Word')
    plt.ylabel('Data')  # Update with your actual data label
    plt.legend(title='Category')

    plt.show()
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    data = []
    x_positions = []
    hue = []

    # Iterate over the keys in your dictionary
    for probeword in scoredict.keys():
        su_list_nops = scoredict_naive[probeword]['female_talker']['nonpitchshift']['su_list']
        mu_list_nops = scoredict_naive[probeword]['female_talker']['nonpitchshift']['mu_list']
        total_control = su_list_nops + mu_list_nops

        su_list_ps = scoredict_naive[probeword]['female_talker']['pitchshift']['su_list']
        mu_list_ps = scoredict_naive[probeword]['female_talker']['pitchshift']['mu_list']
        total_rove = su_list_ps + mu_list_ps

        # Create a DataFrame for seaborn
        control_df = pd.DataFrame({'Data': total_control, 'Probe Word': probeword, 'Category': 'Control'})
        rove_df = pd.DataFrame({'Data': total_rove, 'Probe Word': probeword, 'Category': 'Rove'})

        # Append the data and category
        data.extend(total_control)
        data.extend(total_rove)
        x_positions.extend([probeword] * (len(total_control) + len(total_rove)))
        hue.extend(['Control'] * len(total_control))
        hue.extend(['Rove'] * len(total_rove))

        # Create the violin plot
    sns.violinplot(x=x_positions, y=data, hue=hue, palette={"Control": "cyan", "Rove": "blue"}, split=True, ax=ax)

    # Scatter plot for raw data
    scatter_data = pd.DataFrame({'x_positions': x_positions, 'Data': data, 'Category': hue})

    sns.scatterplot(x="x_positions", y="Data", hue="Category", data=scatter_data, s=20, ax=ax,
                    palette={"Control": "white", "Rove": "yellow"})

    # Customize the plot
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.xlabel('Probe Word')
    plt.ylabel('Data')  # Update with your actual data label
    plt.legend(title='Category')

    plt.show()


    #now do the same for the trained animals
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    data = []
    x_positions = []
    hue = []

    # Iterate over the keys in your dictionary
    for probeword in scoredict.keys():
        su_list_nops = scoredict[probeword]['female_talker']['nonpitchshift']['su_list']
        mu_list_nops = scoredict[probeword]['female_talker']['nonpitchshift']['mu_list']
        total_control = su_list_nops + mu_list_nops

        su_list_ps = scoredict[probeword]['female_talker']['pitchshift']['su_list']
        mu_list_ps = scoredict[probeword]['female_talker']['pitchshift']['mu_list']
        total_rove = su_list_ps + mu_list_ps

        # Create a DataFrame for seaborn
        control_df = pd.DataFrame({'Data': total_control, 'Probe Word': probeword, 'Category': 'Control'})
        rove_df = pd.DataFrame({'Data': total_rove, 'Probe Word': probeword, 'Category': 'Rove'})

        # Append the data and category
        data.extend(total_control)
        data.extend(total_rove)
        x_positions.extend([probeword] * (len(total_control) + len(total_rove)))


        hue.extend(['Control'] * len(total_control))
        hue.extend(['Rove'] * len(total_rove))

        # Create the violin plot
    sns.violinplot(x=x_positions, y=data, hue=hue, palette={"Control": "purple", "Rove": "pink"}, split=True, ax=ax)

    # Scatter plot for raw data
    scatter_data = pd.DataFrame({'x_positions': x_positions, 'Data': data, 'Category': hue})

    sns.scatterplot(x="x_positions", y="Data", hue="Category", data=scatter_data, s=15, ax=ax,
                    palette={"Control": "white", "Rove": "black"})

    # Customize the plot
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.xlabel('Probe Word')
    plt.ylabel('Data')  # Update with your actual data label
    plt.legend(title='Category')

    plt.show()




    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    plot_count = 0
    x_offset = 0  # Initialize x-coordinate offset
    xtick_labels = []

    for probeword in scoredict.keys():
        su_list_nops = scoredict[probeword]['female_talker']['nonpitchshift']['su_list']
        mu_list_nops = scoredict[probeword]['female_talker']['nonpitchshift']['mu_list']
        total_control = su_list_nops + mu_list_nops

        su_list_ps = scoredict[probeword]['female_talker']['pitchshift']['su_list']
        mu_list_ps = scoredict[probeword]['female_talker']['pitchshift']['mu_list']
        total_rove = su_list_ps + mu_list_ps

        sns.swarmplot(x=np.array([plot_count] * len(total_control)) + x_offset, y=total_control, color='purple',
                      alpha=0.5, label='control')
        x_offset += 0.2  # Adjust the offset to separate points

        sns.swarmplot(x=np.array([plot_count] * len(total_rove)) + x_offset, y=total_rove, color='pink', alpha=0.5,
                      label='rove')
        x_offset += 0.4  # Adjust the offset for the next category

        plot_count += 1
        # xtick_labels.append(str((2, 2)))  # Adjust this line to add appropriate labels

    plt.legend(fontsize=8)
    plt.xlabel('probe word')
    ax.set_xticks(np.arange(0, plot_count))
    # ax.set_xticklabels([(2, 2), (5, 6), (42, 49), (32, 38), (20, 22), (15, 15), (42, 49), (4, 4), (16, 16), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
    #                       (14, 14)])
    plt.show()

    # Define labels and colors for scatter plots
    #plot scatter data in a loop
    # for i, (data_dict, label, color) in enumerate(zip(dictlist, labels, colors)):
    #     ax.scatter(data_dict['mu_list']['nonpitchshift']['female_talker'],data_dict['mu_list']['pitchshift']['female_talker'], marker='P',
    #                facecolors =color, edgecolors = color, alpha=0.5)
    #     ax.scatter(data_dict['su_list']['nonpitchshift']['female_talker'],data_dict['su_list']['pitchshift']['female_talker'], marker='P', color=color, alpha=0.5)


    # if bigconcatenatenaive_nonps.size > bigconcatenatenaive_ps.size:
    #     len(bigconcatenatenaive_ps)
    #     bigconcatenatenaive_nonps = bigconcatenatenaive_nonps[:bigconcatenatenaive_ps.size]
    # elif bigconcatenatenaive_nonps.size < bigconcatenatenaive_ps.size:
    #     bigconcatenatenaive_ps = bigconcatenatenaive_ps[:bigconcatenatenaive_nonps.size]
    #
    # if bigconcatenatetrained_nonps.size > bigconcatenatetrained_ps.size:
    #     bigconcatenatetrained_nonps = bigconcatenatetrained_nonps[:bigconcatenatetrained_ps.size]
    # elif bigconcatenatetrained_nonps.size < bigconcatenatetrained_ps.size:
    #     bigconcatenatetrained_ps = bigconcatenatetrained_ps[:bigconcatenatetrained_nonps.size]

    # fig, ax = plt.subplots(1, figsize=(9, 9), dpi=300)
    # ax.scatter(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, marker='P', color='purple', alpha=0.8, label='trained', s=0.1)
    #
    # plt.title('trained animals, number of points: ' + str(len(bigconcatenatetrained_ps)))
    # plt.show()
    # unique_scores = np.unique(bigconcatenatetrained_ps)
    # len(unique_scores)
    #
    #
    # fig, ax = plt.subplots(1, figsize=(9,9), dpi=300)
    #
    # ax.scatter(bigconcatenatenaive_nonps, bigconcatenatenaive_ps, marker='P', color='darkcyan', alpha=0.5, label='naive')
    # ax.scatter(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, marker='P', color='purple', alpha=0.5, label='trained')
    # x = np.linspace(0.4, 1, 101)
    # ax.plot(x, x, color='black', linestyle = '--')  # identity line
    #
    # slope, intercept, r_value, pv, se = stats.linregress(bigconcatenatetrained_nonps, bigconcatenatetrained_ps)
    #
    # sns.regplot(x=bigconcatenatetrained_nonps, y=bigconcatenatetrained_ps, scatter=False, color='purple',
    #             label=' $y=%3.7s*x+%3.7s$' % (slope, intercept), ax=ax, line_kws={'label': ' $y=%3.7s*x+%3.7s$' % (slope, intercept)})
    # slope, intercept, r_value, pv, se = stats.linregress(bigconcatenatenaive_nonps, bigconcatenatenaive_ps)
    #
    # sns.regplot(x=bigconcatenatenaive_nonps, y=bigconcatenatenaive_ps, scatter=False, color='darkcyan', label=' $y=%3.7s*x+%3.7s$' % (slope, intercept),
    #             ax=ax, line_kws={'label': '$y=%3.7s*x+%3.7s$' % (slope, intercept)})
    #
    # ax.set_ylabel('LSTM decoding score, F0 roved', fontsize=18)
    # ax.set_xlabel('LSTM decoding score, F0 control', fontsize=18)
    #
    # ax.set_title('LSTM decoder scores for' + ' F0 control vs. roved,\n ' + ' trained and naive animals', fontsize=20)
    #
    #
    # plt.legend( fontsize=12, ncol=2)
    # fig.tight_layout()
    # plt.savefig('G:/neural_chapter/figures/scattermuaandsuregplot_mod_21062023.png', dpi=1000)
    # plt.savefig('G:/neural_chapter/figures/scattermuaandsuregplot_mod_21062023.pdf', dpi=1000)
    #
    #
    # plt.show()
    #
    # #histogram distribution of the trained and naive animals
    # fig, ax = plt.subplots(1, figsize=(8, 8))
    # #relativescoretrained = abs(bigconcatenatetrained_nonps - bigconcatenatetrained_ps)/ bigconcatenatetrained_ps
    #
    # relativescoretrained = [bigconcatenatetrained_nonps - bigconcatenatetrained_ps for bigconcatenatetrained_nonps, bigconcatenatetrained_ps in zip(bigconcatenatetrained_nonps, bigconcatenatetrained_ps)]
    # relativescorenaive = [bigconcatenatenaive_nonps - bigconcatenatenaive_ps for bigconcatenatenaive_nonps, bigconcatenatenaive_ps in zip(bigconcatenatenaive_ps, bigconcatenatenaive_nonps)]
    # relativescoretrainedfrac = [relativescoretrained / (bigconcatenatetrained_nonps + bigconcatenatenaive_nonps) for relativescoretrained, bigconcatenatetrained_nonps, bigconcatenatenaive_nonps in zip(relativescoretrained, bigconcatenatetrained_nonps, bigconcatenatenaive_nonps)]
    # relativescorenaivefrac = [relativescorenaive / (bigconcatenatenaive_nonps + bigconcatenatetrained_nonps) for relativescorenaive, bigconcatenatenaive_nonps, bigconcatenatetrained_nonps in zip(relativescorenaive, bigconcatenatenaive_nonps, bigconcatenatetrained_nonps)]
    #
    # sns.distplot(relativescoretrained, bins = 20, label='trained',ax=ax, color='purple')
    # sns.distplot(relativescorenaive, bins = 20, label='naive', ax=ax, color='darkcyan')
    # plt.axvline(x=0, color='black')
    #man whiteney test score

    #
    # manwhitscore = mannwhitneyu(relativescoretrained, relativescorenaive, alternative = 'greater')
    # sample1 = np.random.choice(relativescoretrained, size=10000, replace=True)
    #
    # # Generate a random sample of size 100 from data2 with replacement
    # sample2 = np.random.choice(relativescorenaive, size=10000, replace=True)
    #
    # # Perform a t-test on the samples
    # t_stat, p_value = stats.ttest_ind(sample1, sample2, alternative='less')
    #
    # # Print the t-statistic and p-value
    # print(t_stat, p_value)
    # plt.title('Control - roved F0 \n LSTM decoder scores between trained and naive animals', fontsize = 18)
    # plt.xlabel('Control - roved F0 \n LSTM decoder scores', fontsize = 20)
    # plt.ylabel('Density', fontsize = 20)
    # #ax.legend()
    # plt.savefig('G:/neural_chapter/figures/diffF0distribution_20062023.png', dpi=1000)
    # plt.show()
    #
    # #plot sns histogram of the relative score and with the displot function overlaid
    # fig, ax = plt.subplots(1, figsize=(8, 8))
    # sns.histplot(relativescoretrainedfrac, bins=20, label='trained', color='purple', kde = True)
    # sns.histplot(relativescorenaivefrac, bins = 20, label='naive', color='darkcyan', kde = True)
    # plt.title('Control - roved F0 \n LSTM decoder scores between trained and naive animals', fontsize = 18)
    # plt.legend(fontsize = 18)
    # plt.savefig('G:/neural_chapter/figures/diffF0distribution_relfrac_histplotwithkde_20062023.png', dpi=1000)
    # plt.show()

    #now plot a barplot of each of the scores for each word in the probewordlist
    #first get the scores for each word in the probewordlist

    #get the scores for each word in the probewordlist;

    # #need to reorganise the dictlist, create a NEW dictlist in a function
    # for key in dictlist_trained:
    #     for key2 in key.keys():
    #         for key3 in key[key2].keys():
    #             for key4 in key[key2][key3].keys():
    #                 for key5 in key[key2][key3][key4].keys():
    #                     print(key5)
    #                     print(key[key2][key3][key4][key5])





    #
    # fig, ax = plt.subplots(1, figsize=(8, 8))
    # ax = sns.distplot(relativescoretrainedfrac, bins = 20, label='trained',ax=ax, color='purple')
    # x = ax.lines[-1].get_xdata()  # Get the x data of the distribution
    # y = ax.lines[-1].get_ydata()  # Get the y data of the distribution
    # maxidtrained_idx = np.argmax(y)
    # x_coord_trained = x[maxidtrained_idx]
    # ax2 = sns.distplot(relativescorenaivefrac, bins = 20, label='naive', ax=ax, color='darkcyan')
    # x2 = ax2.lines[-1].get_xdata()  # Get the x data of the distribution
    # y2 = ax2.lines[-1].get_ydata()  # Get the y data of the distribution
    # maxidnaive_idx = np.argmax(y2)  # The id of the peak (maximum of y data)
    # x_coord_naive = x2[maxidnaive_idx]
    # plt.axvline(x=0, color='black')
    # kstestnaive = scipy.stats.kstest(relativescorenaivefrac,  stats.norm.cdf)
    # leveneteststat = scipy.stats.levene(relativescorenaivefrac, relativescoretrainedfrac)
    # manwhitscorefrac = mannwhitneyu(relativescorenaivefrac, relativescoretrainedfrac, alternative = 'less')
    # #caclulate medians of distribution
    # sample1_trained = np.random.choice(relativescoretrainedfrac, size=10000, replace=True)
    # # Generate a random sample of size 100 from data2 with replacement
    # sample2_naive = np.random.choice(relativescorenaive, size=10000, replace=True)
    # # Perform a t-test on the samples
    # t_statfrac, p_valuefrac = stats.ttest_ind(sample2_naive, sample1_trained, alternative='less')
    #
    # # Print the t-statistic and p-value
    # print(t_statfrac, p_valuefrac)
    # plt.title('Control - roved F0 \n LSTM decoder scores between trained and naive animals', fontsize = 18)
    # plt.xlabel('Control - roved F0 \n LSTM decoder scores divided by control F0', fontsize = 20)
    # plt.ylabel('Density', fontsize = 20)
    # #ax.legend(fontsize = 18)
    #
    #
    # plt.savefig('G:/neural_chapter/figures/diffF0distribution_frac_20062023wlegendintertrialroving.png', dpi=1000)
    # plt.show()



    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=800)
    ax.set_xlim([0,1])

    sns.distplot(bigconcatenatetrained_nonps,  label='trained',ax=ax, color='purple')
    sns.distplot(bigconcatenatenaive_nonps, label='naive', ax=ax, color='darkcyan')
    plt.axvline(x=0, color='black')
    plt.xlim([0.35,1])
    #man whiteney test score
    plt.title('Control F0 scores between  \n trained and naive animals', fontsize = 30)
    plt.xlabel('Control F0 LSTM decoder scores', fontsize = 30)

    plt.ylabel('Density', fontsize = 30)
    ax.set_xticks([0.4,0.5, 0.6, 0.7, 0.8,0.9, 1], labels = [0.4,0.5, 0.6, 0.7, 0.8,0.9, 1], fontsize = 20)
    ax.set_yticks([2,4,6,8,10,12], labels = [2,4,6,8,10,12], fontsize = 20)
    manwhitscorecontrolf0 = mannwhitneyu(bigconcatenatetrained_nonps, bigconcatenatenaive_nonps, alternative = 'greater')

    n1 = len(bigconcatenatetrained_nonps)
    n2 = len(bigconcatenatenaive_nonps)
    r_controlf0 = 1 - (2 * manwhitscorecontrolf0.statistic) / (n1 * n2)
    #ax.legend()
    plt.savefig('G:/neural_chapter/figures/controlF0distribution20062023intertrialroving.png', dpi=1000, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=800)
    ax.set_xlim([0,1])
    sns.distplot(bigconcatenatetrained_ps,  label='trained',ax=ax, color='purple')
    sns.distplot(bigconcatenatenaive_ps, label='naive', ax=ax, color='darkcyan')
    ax.set_xticks([0.4,0.5, 0.6, 0.7, 0.8,0.9, 1], labels = [0.4,0.5, 0.6, 0.7, 0.8,0.9, 1], fontsize = 20)
    ax.set_yticks([2,4,6,8,10,12], labels = [2,4,6,8,10,12], fontsize = 20)

    #man whiteney test score
    #manwhitscore = mannwhitneyu(relativescoretrained, relativescorenaive, alternative = 'greater')
    plt.title('Roved F0 scores between  \n trained and naive animals', fontsize = 30)
    plt.xlabel('Roved F0 LSTM decoder scores', fontsize = 20)
    plt.ylabel('Density', fontsize=20)
    manwhitscorerovedf0 = mannwhitneyu(bigconcatenatetrained_ps, bigconcatenatenaive_ps, alternative = 'greater')
    plt.xlim([0.35,1])

    n1 = len(bigconcatenatetrained_ps)
    n2 = len(bigconcatenatenaive_ps)
    r_rovef0 = 1 - (2 * manwhitscorerovedf0.statistic) / (n1 * n2)

    # ax.leg
    ax.legend(fontsize=18)
    plt.savefig('G:/neural_chapter/figures/rovedF0distribution_20062023intertrialroving.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi = 800)
    ax.set_xlim([0,1])
    sns.distplot(bigconcatenatetrained_ps,  label='trained roved',ax=ax, color='purple')
    sns.distplot(bigconcatenatetrained_nonps,  label='trained control',ax=ax, color='magenta')
    ax.legend(fontsize=18)
    plt.title('Roved and Control F0 Distributions for the Trained Animals', fontsize = 18)
    plt.xlabel(' LSTM decoder scores', fontsize = 20)
    plt.xlim([0.35,1])

    plt.savefig('G:/neural_chapter/figures/rovedF0vscontrolF0traineddistribution_20062023intertrialroving.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi = 800)
    ax.set_xlim([0,1])
    sns.distplot(bigconcatenatenaive_ps,  label='naive roved',ax=ax, color='darkcyan')
    sns.distplot(bigconcatenatenaive_nonps,  label='naive control',ax=ax, color='cyan')
    plt.xlim([0.35,1])

    ax.legend(fontsize=18)
    plt.xlabel(' LSTM decoder scores', fontsize = 20)
    plt.title('Roved and Control F0 Distributions for the Naive Animals', fontsize = 18)

    plt.savefig('G:/neural_chapter/figures/rovedF0vscontrolF0naivedistribution_20062023intertrialroving.png', dpi=1000)
    plt.show()
    kstestcontrolf0vsrovedtrained = scipy.stats.kstest(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, alternative = 'two-sided')

    #do levene's test
    leveneteststat = scipy.stats.levene(bigconcatenatetrained_nonps, bigconcatenatetrained_ps)
    kstestcontrolf0vsrovednaive = scipy.stats.kstest(bigconcatenatenaive_nonps, bigconcatenatenaive_ps, alternative='two-sided')

    # Calculating Cramér's V for effect size
    def cramers_v(n, ks_statistic):
        return np.sqrt(ks_statistic / n)

    n = len(bigconcatenatenaive_nonps) * len(bigconcatenatenaive_ps) / (len(bigconcatenatenaive_nonps) + len(bigconcatenatenaive_ps))
    effect_size_naive = cramers_v(n, kstestcontrolf0vsrovednaive.statistic)

    n_trained = len(bigconcatenatetrained_nonps) * len(bigconcatenatetrained_ps) / (len(bigconcatenatetrained_nonps) + len(bigconcatenatetrained_ps))
    effect_size_trained = cramers_v(n_trained, kstestcontrolf0vsrovedtrained.statistic)


    #run mann whitney u test
    manwhitscore_stat, manwhitescore_pvalue = mannwhitneyu(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, alternative = 'two-sided')
    manwhitscore_statnaive, manwhitescore_pvaluenaive = mannwhitneyu(bigconcatenatenaive_nonps, bigconcatenatenaive_ps, alternative = 'two-sided')

    # Calculate rank-biserial correlation coefficient

    n1 = len(bigconcatenatetrained_nonps)
    n2 = len(bigconcatenatetrained_ps)
    r = 1 - (2 * manwhitscore_stat) / (n1 * n2)

    n1 = len(bigconcatenatenaive_nonps)
    n2 = len(bigconcatenatenaive_ps)
    r_naive = 1 - (2 * manwhitscore_statnaive) / (n1 * n2)


    #put these stats into a table and export to csv
    #create a dataframe
    dataframe_stats = pd.DataFrame({'effect sizes r value': [r_controlf0, r_rovef0, r, r_naive], 'trained animals p value': [manwhitscorecontrolf0.pvalue, manwhitscorerovedf0.pvalue, manwhitescore_pvalue, manwhitescore_pvaluenaive]},  index = ['control naive vs. trained(alt = trained > naive) ', 'roved naive vs trained (alt = trained > naive)', 'control vs. roved trained (two sided)', 'control naive vs. roved naive (two sided)'])

    #export to csv
    dataframe_stats.to_csv('G:/neural_chapter/figures/stats_13112023_comparingdistributions_generalintertrialroving.csv')

    #do levene's test
    leveneteststat_naive = scipy.stats.levene(bigconcatenatenaive_nonps, bigconcatenatenaive_ps)
    naivearray=np.concatenate((np.zeros((len(bigconcatenatetrained_nonps)+len(bigconcatenatetrained_ps),1)), np.ones((len(bigconcatenatenaive_nonps)+len(bigconcatenatenaive_ps),1))))
    trainedarray=np.concatenate((np.ones((len(bigconcatenatetrained_nonps)+len(bigconcatenatetrained_ps),1)), np.zeros((len(bigconcatenatenaive_nonps)+len(bigconcatenatenaive_ps),1))))
    controlF0array=np.concatenate((np.ones((len(bigconcatenatetrained_nonps),1)), np.zeros((len(bigconcatenatetrained_ps),1)), np.ones((len(bigconcatenatenaive_nonps),1)), np.zeros((len(bigconcatenatenaive_ps),1))))
    rovedF0array = np.concatenate((np.zeros((len(bigconcatenatetrained_nonps),1)), np.ones((len(bigconcatenatetrained_ps),1)), np.zeros((len(bigconcatenatenaive_nonps),1)), np.ones((len(bigconcatenatenaive_ps),1))))
    scores = np.concatenate((bigconcatenatetrained_nonps, bigconcatenatetrained_ps, bigconcatenatenaive_nonps, bigconcatenatenaive_ps))

    dataset = pd.DataFrame({'trained': trainedarray[:,0], 'naive': naivearray[:,0], 'controlF0': controlF0array[:,0], 'rovedF0': rovedF0array[:,0], 'scores': scores})




if __name__ == '__main__':
    main()