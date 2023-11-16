import os

import matplotlib.pyplot as plt
import scipy.stats as stats
import shap
import statsmodels as sm
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
from scipy.stats import mannwhitneyu
from helpers.vis_stats_helpers import run_anova_on_dataframe, create_gen_frac_variable, runlgbmmodel_score

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
        noiselist_copy = noiselist.copy()

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
                report['tdt'] = pd.read_csv(f'G:/F2003_Orecchiette/{stream}/recording_0/pykilosort/report/' + 'unit_list.csv')
                #take only the second column
                report['tdt'] = report['tdt'].iloc[:, 1]

            except:                # make a column of 0s for the tdt column
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
                                probeword = (7,7)




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
                                    su_pitchshiftlist_female = np.append(su_pitchshiftlist_female,
                                                                         scores[f'talker{talker}'][comp][cond][
                                                                             score_key][i])
                                    su_pitchshiftlist_female_probeword = np.append(su_pitchshiftlist_female_probeword, probeword[talker-1])

                                    su_pitchshiftlist_female_unitid = np.append(su_pitchshiftlist_female_unitid, clust_text)


                                    su_pitchshiftlist_female_channel_id = np.append(su_pitchshiftlist_female_channel_id, report['tdt'][clus_id_report])

                                elif talker == 2:

                                    su_pitchshiftlist_male = np.append(su_pitchshiftlist_male,
                                                                       scores[f'talker{talker}'][comp][cond][
                                                                           score_key][i])
                                    su_pitchshiftlist_male_probeword = np.append(su_pitchshiftlist_male_probeword, probeword[talker -1 ])
                                    su_pitchshiftlist_male_unitid = np.append(su_pitchshiftlist_male_unitid, clust_text)
                                    su_pitchshiftlist_male_channel_id = np.append(su_pitchshiftlist_male_channel_id, report['tdt'][clus_id_report])



                                # print(pitchshiftlist.size)
                            elif cond == 'nopitchshift':
                                if talker == 1:

                                    su_nonpitchshiftlist_female = np.append(su_nonpitchshiftlist_female,
                                                                            scores[f'talker{talker}'][comp][cond][
                                                                                score_key][i])
                                    su_nonpitchshiftlist_female_probeword = np.append(su_nonpitchshiftlist_female_probeword, probeword[talker -1 ])
                                    su_nonpitchshiftlist_female_unitid = np.append(su_nonpitchshiftlist_female_unitid, clust_text)
                                    su_nonpitchshiftlist_female_channel_id = np.append(su_nonpitchshiftlist_female_channel_id, report['tdt'][clus_id_report])

                                elif talker == 2:

                                    su_nonpitchshiftlist_male = np.append(su_nonpitchshiftlist_male,
                                                                          scores[f'talker{talker}'][comp][cond][
                                                                              score_key][i])
                                    su_nonpitchshiftlist_male_probeword = np.append(su_nonpitchshiftlist_male_probeword, probeword[talker -1 ])
                                    su_nonpitchshiftlist_male_unitid = np.append(su_nonpitchshiftlist_male_unitid, clust_text)

                                    su_nonpitchshiftlist_male_channel_id = np.append(su_nonpitchshiftlist_male_channel_id, report['tdt'][clus_id_report])


                        elif clus in multiunitlist_copy:
                            if cond == 'pitchshift':
                                if talker == 1:

                                    mu_pitchshiftlist_female = np.append(mu_pitchshiftlist_female,
                                                                         scores[f'talker{talker}'][comp][cond][
                                                                             score_key][
                                                                             i])
                                    mu_pitchshiftlist_female_probeword = np.append(mu_pitchshiftlist_female_probeword, probeword[talker -1 ])
                                    mu_pitchshiftlist_female_unitid = np.append(mu_pitchshiftlist_female_unitid, clust_text)
                                    mu_pitchshiftlist_female_channel_id = np.append(mu_pitchshiftlist_female_channel_id, report['tdt'][clus_id_report])

                                elif talker == 2:

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

                                    mu_nonpitchshiftlist_female = np.append(mu_nonpitchshiftlist_female,
                                                                            scores[f'talker{talker}'][comp][cond][
                                                                                score_key][i])
                                    mu_nonpitchshiftlist_female_probeword = np.append(mu_nonpitchshiftlist_female_probeword, probeword[talker -1 ])
                                    mu_nonpitchshiftlist_female_unitid = np.append(mu_nonpitchshiftlist_female_unitid, clust_text)
                                    mu_nonpitchshiftlist_female_channel_id = np.append(mu_nonpitchshiftlist_female_channel_id, report['tdt'][clus_id_report])

                                elif talker == 2:

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
    xg_reg = lgb.LGBMRegressor(colsample_bytree=0.3, learning_rate=0.1,
                               max_depth=10, alpha=10, n_estimators=10, verbose=1)

    xg_reg.fit(X_train, y_train, eval_metric='MSE', verbose=1)
    ypred = xg_reg.predict(X_test)
    lgb.plot_importance(xg_reg)
    plt.title('feature importances for the lstm decoding score model')
    plt.savefig(f'G:/neural_chapter/figures/lightgbm_model_feature_importances.png', dpi = 300)
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
    plt.savefig(f'G:/neural_chapter/figures/lightgbm_summary_plot.png', dpi = 300)
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
    probewordlist =[ (2,2), (3,3), (4,4),(5,5), (6,6), (7,7), (8,8), (9,9), (10,10)]
    probewordlist_l74 = [(10, 10), (2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (8, 8), (9, 9), (11, 11), (12, 12),
                             (14, 14)]
    animal_list = [ 'F1604_Squinty', 'F1901_Crumble', 'F1606_Windolene', 'F1702_Zola','F1815_Cruella', 'F1902_Eclair', 'F1812_Nala',  'F2003_Orecchiette',]

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
        #get the parent directory of each path
        path_list[animal] = [path.parent for path in path_list[animal]]
    #report, singleunitlist, and multiunitlist and noiselist need to be modified to include the recname
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


            report[animal][stream_name], singleunitlist[animal][stream_name], multiunitlist[animal][stream_name], noiselist[animal][stream_name] = load_classified_report(f'{path}')

    # now create a dictionary of dictionaries, where the first key is the animal name, and the second key is the stream name
    #the value is are the decoding scores for each cluster


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

            if animal =='F2003_Orecchiette':
                rec_name_unique = stream

            else:
            #if stream contains BB_2
                if 'BB_2' in stream:
                    streamtext = 'bb2'
                elif 'BB_3' in stream:
                    streamtext = 'bb3'
                elif 'BB_4' in stream:
                    streamtext = 'bb4'
                elif 'BB_5' in stream:
                    streamtext = 'bb5'
                #remove F number character from animal name

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
                                                                noiselist=noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist_l74,
                                                                saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], permutation_scores=True)
                dictoutput_all_permutation.append(dictoutput_instance_permutation)

            elif animal == 'F1606_Windolene':
                dictoutput_instance = scatterplot_and_visualise(probewordlist_l74,
                                                                saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist_l74,
                                                                saveDir=f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream], permutation_scores=True)
                dictoutput_all_permutation.append(dictoutput_instance_permutation)

            elif animal =='F1702_Zola':
                dictoutput_instance = scatterplot_and_visualise(probewordlist_zola, saveDir= f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist_zola, saveDir= f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream]
                                                                            , permutation_scores=True)
                dictoutput_all_permutation.append(dictoutput_instance_permutation)

            elif animal == 'F1815_Cruella' or animal == 'F1902_Eclair':
                dictoutput_instance = scatterplot_and_visualise(probewordlist, saveDir= f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist, saveDir= f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream]
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
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream=stream,
                                                                fullid=animal,
                                                                report=report[animal][stream], permutation_scores=True
                                                                )
                dictoutput_all_permutation.append(dictoutput_instance_permutation)

            else:
                # try:
                dictoutput_instance = scatterplot_and_visualise(probewordlist, saveDir= f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist, saveDir= f'F:/results_13112023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream]
                                                                            , permutation_scores=True)

                dictoutput_all_permutation.append(dictoutput_instance_permutation)


            female_talker_len = len(dictoutput_instance['su_list']['pitchshift']['female_talker'])
            probeword_len = len(dictoutput_instance['su_list_probeword']['pitchshift']['female_talker'])

            assert female_talker_len == probeword_len, f"Length mismatch: female_talker_len={female_talker_len}, probeword_len={probeword_len}"

            try:
                if animal == 'F1604_Squinty' or animal == 'F1606_Windolene' or animal == 'F1702_Zola' or animal == 'F1815_Cruella':
                    print('trained animal'+ animal)
                    dictoutput_trained.append(dictoutput_instance)
                    dictoutput_trained_permutation.append(dictoutput_instance_permutation)
                else:
                    print('naive animal:'+ animal)
                    dictoutput_naive.append(dictoutput_instance)
                    dictoutput_naive_permutation.append(dictoutput_instance_permutation)
            except:
                print('no scores for this stream')
                pass


    labels = [ 'F1901_Crumble', 'F1604_Squinty', 'F1606_Windolene', 'F1702_Zola','F1815_Cruella', 'F1902_Eclair', 'F1812_Nala']

    colors = ['purple', 'magenta', 'darkturquoise', 'olivedrab', 'steelblue', 'darkcyan', 'darkorange']

    generate_plots(dictoutput_all, dictoutput_trained, dictoutput_naive, dictoutput_all_permutation, dictoutput_trained_permutation, dictoutput_naive_permutation, labels, colors)
    return


def generate_plots(dictlist, dictlist_trained, dictlist_naive, dictlist_permutation, dictlist_trained_permutation, dictlist_naive_permutation, labels, colors):

    probewordlist_text = [(2, 2), (3, 3), (4, 4), (5, 5), (6,6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13,13),
                          (14, 14), (15,15), (16,16), (17,17), (18,18), (19,19), (20,20), (21,21), (22,22), (23,23), (24,24), (25,25)]

    scoredict = {}
    scoredict_naive={}

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
    probeword_to_text = {
        2: '(2,2)',
        3: '(3,3)',
        4: '(4,4)',
        5: '(5,5)',
        6: '(6,6)',
        7: '(7,7)',
        8: '(8,8)',
        9: '(9,9)',
        10: '(10,10)',
        13: '(13,13)',
        15: '(15,15)',
        16: '(16,16)',
        11: '(11,11)',
        12: '(12,12)',
        14: '(14,14)',
        15: '(15,15)',
        16: '(16,16)',
        17: '(17,17)',
        18: '(18,18)',
        19: '(19,19)',
        20: '(20,20)',
        21: '(21,21)',
        22: '(22,22)',
        23: '(23,23)',
        24: '(24,24)',
        25: '(25,25)'
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



    unit_ids = []

    for dict in dictlist_trained:
        # Get all the unique unit_ids
        for key in dict['su_list_unitid']:
            for key2 in dict['su_list_unitid'][key]:
                for key3 in dict['su_list_unitid'][key][key2]:
                    unit_ids.append(key3)
        for key in dict['mu_list_unitid']:
            for key2 in dict['mu_list_unitid'][key]:
                for key3 in dict['mu_list_unitid'][key][key2]:
                    unit_ids.append(key3)

    unit_ids = np.unique(unit_ids)

    # Initialize the scoredict with unit IDs and probeword texts
    scoredict_byunit = {
        unit_id: {probeword_text: {'su_list': [], 'mu_list': [], 'channel_id': []} for probeword_text in probeword_to_text.values()} for
        unit_id in unit_ids}

    scoredict_byunit_pitchsplit = {
        unit_id: {
            probeword_text: {
                'pitchshift': {'su_list': [], 'mu_list': [], 'channel_id': []},
                'nonpitchshift': {'su_list': [], 'mu_list': [], 'channel_id': []}
            } for probeword_text in probeword_to_text.values()
        } for unit_id in unit_ids
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
                    unit_id = dict['su_list_unitid'][key][talker_key][count]
                    channel_id = dict['su_list_chanid'][key][talker_key][count]
                    #adding 16 to match the json file
                    if 'BB_3' in unit_id:
                        print(unit_id)
                        channel_id = channel_id+16
                    elif 'BB_5' in unit_id:
                        channel_id = channel_id+16


                    # Add 'channel_id'
                    if probewordtext:
                        scoredict_byunit[unit_id][probewordtext]['su_list'].append(
                            dict['su_list'][key][talker_key][count])
                        scoredict_byunit[unit_id][probewordtext]['channel_id'].append(channel_id)

                        scoredict_byunit_pitchsplit[unit_id][probewordtext][key]['su_list'].append(
                            dict['su_list'][key][talker_key][count])
                        scoredict_byunit_pitchsplit[unit_id][probewordtext][key]['channel_id'].append(channel_id)

                        # Update 'channel_id'
                    count = count + 1

            for key in dict['mu_list_probeword']:
                probewords = dict['mu_list_probeword'][key][talker_key]
                count = 0
                for probeword in probewords:
                    probeword_range = int(probeword)
                    probewordtext = probeword_to_text.get(probeword_range)
                    if probewordtext:
                        unit_id = dict['mu_list_unitid'][key][talker_key][count]
                        channel_id = dict['mu_list_chanid'][key][talker_key][count]
                        if 'BB_3' in unit_id:
                            channel_id = channel_id + 16
                        elif 'BB_5' in unit_id:
                            channel_id = channel_id + 16
                        # Add 'channel_id'
                        scoredict_byunit[unit_id][probewordtext]['mu_list'].append(
                            dict['mu_list'][key][talker_key][count])

                        scoredict_byunit_pitchsplit[unit_id][probewordtext][key]['mu_list'].append(
                            dict['mu_list'][key][talker_key][count])
                        scoredict_byunit[unit_id][probewordtext]['channel_id'].append(channel_id)  # Update 'channel_id'
                        scoredict_byunit_pitchsplit[unit_id][probewordtext][key]['channel_id'].append(channel_id)  # Update 'channel_id'
                    count = count + 1
    unit_ids_naive = []
    for dict in dictlist_naive:
        # Get all the unique unit_ids
        for key in dict['su_list_unitid']:
            for key2 in dict['su_list_unitid'][key]:
                for key3 in dict['su_list_unitid'][key][key2]:
                    unit_ids_naive.append(key3)
        for key in dict['mu_list_unitid']:
            for key2 in dict['mu_list_unitid'][key]:
                for key3 in dict['mu_list_unitid'][key][key2]:
                    unit_ids_naive.append(key3)

    unit_ids_naive = np.unique(unit_ids_naive)
    # Initialize the scoredict with unit IDs and probeword texts
    scoredict_byunit_naive = {
        unit_id: {probeword_text: {'su_list': [], 'mu_list': [], 'channel_id': []} for probeword_text in
                  probeword_to_text.values()} for
        unit_id in unit_ids_naive}

    scoredict_byunit_naive_pitchsplit = {
        unit_id: {
            probeword_text: {
                'pitchshift': {'su_list': [], 'mu_list': [], 'channel_id': []},
                'nonpitchshift': {'su_list': [], 'mu_list': [], 'channel_id': []}
            } for probeword_text in probeword_to_text.values()
        } for unit_id in unit_ids_naive
    }

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
                    unit_id = dict['su_list_unitid'][key][talker_key][count]
                    channel_id = dict['su_list_chanid'][key][talker_key][count]
                    # adding 16 to match the json file
                    if 'BB_3' in unit_id:
                        print(unit_id)
                        channel_id = channel_id + 16
                    elif 'BB_5' in unit_id:
                        channel_id = channel_id + 16

                    # Add 'channel_id'
                    if probewordtext:
                        scoredict_byunit_naive[unit_id][probewordtext]['su_list'].append(
                            dict['su_list'][key][talker_key][count])

                        scoredict_byunit_naive_pitchsplit[unit_id][probewordtext][key]['su_list'].append(
                            dict['su_list'][key][talker_key][count])
                        scoredict_byunit_naive[unit_id][probewordtext]['channel_id'].append(channel_id)
                        scoredict_byunit_naive_pitchsplit[unit_id][probewordtext][key]['channel_id'].append(channel_id)
                        # Update 'channel_id'
                    count = count + 1

            for key in dict['mu_list_probeword']:
                probewords = dict['mu_list_probeword'][key][talker_key]
                count = 0
                for probeword in probewords:
                    probeword_range = int(probeword)
                    probewordtext = probeword_to_text.get(probeword_range)
                    if probewordtext:
                        unit_id = dict['mu_list_unitid'][key][talker_key][count]
                        channel_id = dict['mu_list_chanid'][key][talker_key][count]
                        if 'BB_3' in unit_id:
                            channel_id = channel_id + 16
                        elif 'BB_5' in unit_id:
                            channel_id = channel_id + 16
                        # Add 'channel_id'
                        scoredict_byunit_naive[unit_id][probewordtext]['mu_list'].append(
                            dict['mu_list'][key][talker_key][count])

                        scoredict_byunit_naive_pitchsplit[unit_id][probewordtext][key]['mu_list'].append(
                            dict['mu_list'][key][talker_key][count])
                        scoredict_byunit_naive[unit_id][probewordtext]['channel_id'].append(channel_id)
                        scoredict_byunit_naive_pitchsplit[unit_id][probewordtext][key]['channel_id'].append(channel_id)
                        # Update 'channel_id'
                    count = count + 1
    unit_ids_naive_permutation= []
    for dict in dictlist_naive_permutation:
        # Get all the unique unit_ids
        for key in dict['su_list_unitid']:
            for key2 in dict['su_list_unitid'][key]:
                for key3 in dict['su_list_unitid'][key][key2]:
                    unit_ids_naive_permutation.append(key3)
        for key in dict['mu_list_unitid']:
            for key2 in dict['mu_list_unitid'][key]:
                for key3 in dict['mu_list_unitid'][key][key2]:
                    unit_ids_naive_permutation.append(key3)

    unit_ids_naive_permutation = np.unique(unit_ids_naive_permutation)
    # Initialize the scoredict with unit IDs and probeword texts
    scoredict_byunit_naive_perm = {
        unit_id: {probeword_text: {'su_list': [], 'mu_list': [], 'channel_id': []} for probeword_text in
                  probeword_to_text.values()} for
        unit_id in unit_ids_naive_permutation}


    scoredict_byunit_naive_perm_pitchsplit ={
        unit_id: {
            probeword_text: {
                'pitchshift': {'su_list': [], 'mu_list': [], 'channel_id': []},
                'nonpitchshift': {'su_list': [], 'mu_list': [], 'channel_id': []}
            } for probeword_text in probeword_to_text.values()
        } for unit_id in unit_ids_naive_permutation
    }
    for talker in [1]:
        if talker == 1:
            talker_key = 'female_talker'
        for i, dict in enumerate(dictlist_naive_permutation):

            for key in dict['su_list_probeword']:
                probewords = dict['su_list_probeword'][key][talker_key]
                count = 0
                for probeword in probewords:
                    probeword_range = int(probeword)
                    probewordtext = probeword_to_text.get(probeword_range)
                    unit_id = dict['su_list_unitid'][key][talker_key][count]
                    channel_id = dict['su_list_chanid'][key][talker_key][count]
                    # adding 16 to match the json file
                    if 'BB_3' in unit_id:
                        print(unit_id)
                        channel_id = channel_id + 16
                    elif 'BB_5' in unit_id:
                        channel_id = channel_id + 16

                    # Add 'channel_id'
                    if probewordtext:
                        scoredict_byunit_naive_perm[unit_id][probewordtext]['su_list'].append(
                            dict['su_list'][key][talker_key][count])
                        scoredict_byunit_naive_perm[unit_id][probewordtext]['channel_id'].append(channel_id)  # Update 'channel_id'

                        scoredict_byunit_naive_perm_pitchsplit[unit_id][probewordtext][key]['su_list'].append(
                            dict['su_list'][key][talker_key][count])
                        scoredict_byunit_naive_perm_pitchsplit[unit_id][probewordtext][key]['channel_id'].append(channel_id)  # Update 'channel_id'
                    count = count + 1

            for key in dict['mu_list_probeword']:
                probewords = dict['mu_list_probeword'][key][talker_key]
                count = 0
                for probeword in probewords:
                    probeword_range = int(probeword)
                    probewordtext = probeword_to_text.get(probeword_range)
                    if probewordtext:
                        unit_id = dict['mu_list_unitid'][key][talker_key][count]
                        channel_id = dict['mu_list_chanid'][key][talker_key][count]
                        if 'BB_3' in unit_id:
                            channel_id = channel_id + 16
                        elif 'BB_5' in unit_id:
                            channel_id = channel_id + 16
                        # Add 'channel_id'
                        scoredict_byunit_naive_perm[unit_id][probewordtext]['mu_list'].append(
                            dict['mu_list'][key][talker_key][count])
                        scoredict_byunit_naive_perm_pitchsplit[unit_id][probewordtext][key]['mu_list'].append(
                            dict['mu_list'][key][talker_key][count])
                        scoredict_byunit_naive_perm[unit_id][probewordtext]['channel_id'].append(channel_id)
                        scoredict_byunit_naive_pitchsplit[unit_id][probewordtext][key]['channel_id'].append(channel_id)        # Update 'channel_id'
                    count = count + 1

    unit_ids_trained_permutation = []
    for dict in dictlist_trained_permutation:
        # Get all the unique unit_ids
        for key in dict['su_list_unitid']:
            for key2 in dict['su_list_unitid'][key]:
                for key3 in dict['su_list_unitid'][key][key2]:
                    unit_ids_trained_permutation.append(key3)
        for key in dict['mu_list_unitid']:
            for key2 in dict['mu_list_unitid'][key]:
                for key3 in dict['mu_list_unitid'][key][key2]:
                    unit_ids_trained_permutation.append(key3)

    unit_ids_trained_permutation = np.unique(unit_ids_trained_permutation)
    # Initialize the scoredict with unit IDs and probeword texts
    scoredict_byunit_trained_perm = {
        unit_id: {probeword_text: {'su_list': [], 'mu_list': [], 'channel_id': []} for probeword_text in
                  probeword_to_text.values()} for
        unit_id in unit_ids_trained_permutation}

    scoredict_byunit_trained_perm_pitchsplit = {
        unit_id: {
            probeword_text: {
                'pitchshift': {'su_list': [], 'mu_list': [], 'channel_id': []},
                'nonpitchshift': {'su_list': [], 'mu_list': [], 'channel_id': []}
            } for probeword_text in probeword_to_text.values()
        } for unit_id in unit_ids_trained_permutation
    }
    for talker in [1]:
        if talker == 1:
            talker_key = 'female_talker'
        for i, dict in enumerate(dictlist_trained_permutation):

            for key in dict['su_list_probeword']:
                probewords = dict['su_list_probeword'][key][talker_key]
                count = 0
                for probeword in probewords:
                    probeword_range = int(probeword)
                    probewordtext = probeword_to_text.get(probeword_range)
                    unit_id = dict['su_list_unitid'][key][talker_key][count]
                    channel_id = dict['su_list_chanid'][key][talker_key][count]
                    if 'BB_3' in unit_id:
                        print(unit_id)
                        channel_id = channel_id + 16
                    elif 'BB_5' in unit_id:
                        channel_id = channel_id + 16

                    # Add 'channel_id'
                    if probewordtext:
                        scoredict_byunit_trained_perm[unit_id][probewordtext]['su_list'].append(
                            dict['su_list'][key][talker_key][count])
                        scoredict_byunit_trained_perm[unit_id][probewordtext]['channel_id'].append(
                            channel_id)  # Update 'channel_id'

                        scoredict_byunit_trained_perm_pitchsplit[unit_id][probewordtext][key]['su_list'].append(
                            dict['su_list'][key][talker_key][count])
                        scoredict_byunit_trained_perm_pitchsplit[unit_id][probewordtext][key]['channel_id'].append(
                            channel_id)
                    count = count + 1

            for key in dict['mu_list_probeword']:
                probewords = dict['mu_list_probeword'][key][talker_key]
                count = 0
                for probeword in probewords:
                    probeword_range = int(probeword)
                    probewordtext = probeword_to_text.get(probeword_range)
                    if probewordtext:
                        unit_id = dict['mu_list_unitid'][key][talker_key][count]
                        channel_id = dict['mu_list_chanid'][key][talker_key][count]
                        if 'BB_3' in unit_id:
                            channel_id = channel_id + 16
                        elif 'BB_5' in unit_id:
                            channel_id = channel_id + 16
                        # Add 'channel_id'
                        scoredict_byunit_trained_perm[unit_id][probewordtext]['mu_list'].append(
                            dict['mu_list'][key][talker_key][count])

                        scoredict_byunit_trained_perm_pitchsplit[unit_id][probewordtext][key]['mu_list'].append(
                            dict['mu_list'][key][talker_key][count])
                        scoredict_byunit_trained_perm[unit_id][probewordtext]['channel_id'].append(
                            channel_id)

                        scoredict_byunit_trained_perm_pitchsplit[unit_id][probewordtext][key]['channel_id'].append(
                            channel_id)


                        # Update 'channel_id'
                    count = count + 1
    #load the json file which has the electrode positions
    with open('D:\spkvisanddecodeproj2/analysisscriptsmodcg/json_files\electrode_positions.json') as f:
        electrode_position_data = json.load(f)
    scoredict_by_unit_meg = {}
    scoredict_by_unit_meg_pitchsplit = {}

    scoredict_by_unit_peg = {}
    scoredict_by_unit_peg_pitchsplit = {}
    scoredict_by_unit_aeg = {}
    scoredict_by_unit_aeg_pitchsplit = {}


    #now sort each of the score_dicts by channel_id
    for unit_id in scoredict_byunit.keys():
        example_unit = scoredict_byunit[unit_id]
        #load the corresponding channel_id
        if 'F1604_Squinty' in unit_id:
            animal = 'F1604_Squinty'
            side = 'left'
        elif 'F1606_Windolene' in unit_id:
            animal = 'F1606_Windolene'
        elif 'F1702_Zola' in unit_id:
            animal = 'F1702_Zola'
        elif 'F1815_Cruella' in unit_id:
            animal = 'F1815_Cruella'
        elif 'F1901_Crumble' in unit_id:
            animal = 'F1901_Crumble'
        elif 'F1902_Eclair' in unit_id:
            animal = 'F1902_Eclair'
        elif 'F1812_Nala' in unit_id:
            animal = 'F1812_Nala'
        elif 'F2003_Orecchiette' in unit_id:
            animal = 'F2003_Orecchiette'


        if 'BB_3' in unit_id and animal!='F1604_Squinty':
            side = 'right'
        elif 'BB_2' in unit_id and animal!='F1604_Squinty':
            side = 'right'
        elif 'BB_4' in unit_id:
            side = 'left'
        elif 'BB_5' in unit_id:
            side = 'left'

        for probeword in example_unit.keys():
            try:
                channel_id = example_unit[probeword]['channel_id'][0]
            except:
                continue
            if channel_id is not None:
                break
        #load the corresponding electrode position
        electrode_position_dict_for_animal = electrode_position_data[animal][side]
        #find where the TDT number is in the electrode position dict
        for electrode_position in electrode_position_dict_for_animal:
            if electrode_position['TDT_NUMBER'] == channel_id:
                electrode_position_dict = electrode_position
                break
        if animal == 'F2003_Orecchiette':
            if 'mod' in unit_id:
                scoredict_by_unit_peg[unit_id] = scoredict_byunit[unit_id]
                scoredict_by_unit_peg_pitchsplit[unit_id] = scoredict_byunit_pitchsplit[unit_id]
            elif 's2' in unit_id:
                scoredict_by_unit_peg[unit_id] = scoredict_byunit[unit_id]
                scoredict_by_unit_peg_pitchsplit[unit_id] = scoredict_byunit_pitchsplit[unit_id]

            elif 's3' in unit_id:
                scoredict_by_unit_meg[unit_id] = scoredict_byunit[unit_id]
                scoredict_by_unit_meg_pitchsplit[unit_id] = scoredict_byunit_pitchsplit[unit_id]

        else:
            if electrode_position_dict['area'] == 'MEG':
                #add it to a new dictionary
                scoredict_by_unit_meg[unit_id] = scoredict_byunit[unit_id]
                scoredict_by_unit_meg_pitchsplit[unit_id] = scoredict_byunit_pitchsplit[unit_id]
            elif electrode_position_dict['area'] == 'PEG':
                scoredict_by_unit_peg[unit_id] = scoredict_byunit[unit_id]
                scoredict_by_unit_peg_pitchsplit[unit_id] = scoredict_byunit_pitchsplit[unit_id]
            elif electrode_position_dict['area'] == 'AEG':
                scoredict_by_unit_aeg[unit_id] = scoredict_byunit[unit_id]
                scoredict_by_unit_aeg_pitchsplit[unit_id] = scoredict_byunit_pitchsplit[unit_id]
    ##do the same for the permutation data
    scoredict_by_unit_perm_meg = {}
    scoredict_by_unit_perm_peg = {}
    scoredict_by_unit_perm_aeg = {}

    scoredict_by_unit_perm_aeg_pitchsplit = {}
    scoredict_by_unit_perm_peg_pitchsplit = {}
    scoredict_by_unit_perm_meg_pitchsplit = {}

    # now sort each of the score_dicts by channel_id
    for unit_id in scoredict_byunit_trained_perm.keys():
        example_unit = scoredict_byunit_trained_perm[unit_id]
        # load the corresponding channel_id
        if 'F1604_Squinty' in unit_id:
            animal = 'F1604_Squinty'
            side = 'left'
        elif 'F1606_Windolene' in unit_id:
            animal = 'F1606_Windolene'
        elif 'F1702_Zola' in unit_id:
            animal = 'F1702_Zola'
        elif 'F1815_Cruella' in unit_id:
            animal = 'F1815_Cruella'
        elif 'F1901_Crumble' in unit_id:
            animal = 'F1901_Crumble'
        elif 'F1902_Eclair' in unit_id:
            animal = 'F1902_Eclair'
        elif 'F1812_Nala' in unit_id:
            animal = 'F1812_Nala'
        elif 'F2003_Orecchiette' in unit_id:
            animal = 'F2003_Orecchiette'

        if 'BB_3' in unit_id and animal != 'F1604_Squinty':
            side = 'right'
        elif 'BB_2' in unit_id and animal != 'F1604_Squinty':
            side = 'right'
        elif 'BB_4' in unit_id:
            side = 'left'
        elif 'BB_5' in unit_id:
            side = 'left'

        for probeword in example_unit.keys():
            try:
                channel_id = example_unit[probeword]['channel_id'][0]
            except:
                continue
            if channel_id is not None:
                break

        # load the corresponding electrode position
        electrode_position_dict_for_animal = electrode_position_data[animal][side]
        # find where the TDT number is in the electrode position dict
        for electrode_position in electrode_position_dict_for_animal:
            if electrode_position['TDT_NUMBER'] == channel_id:
                electrode_position_dict = electrode_position
                break
        if animal == 'F2003_Orecchiette':
            if 'mod' in unit_id:
                scoredict_by_unit_perm_peg[unit_id] = scoredict_byunit_trained_perm[unit_id]
                scoredict_by_unit_perm_peg_pitchsplit[unit_id] = scoredict_byunit_trained_perm_pitchsplit[unit_id]
            elif 's2' in unit_id:
                scoredict_by_unit_perm_peg[unit_id] = scoredict_byunit_trained_perm[unit_id]
                scoredict_by_unit_perm_peg_pitchsplit[unit_id] = scoredict_byunit_trained_perm_pitchsplit[unit_id]

            elif 's3' in unit_id:
                scoredict_by_unit_perm_meg[unit_id] = scoredict_byunit_trained_perm[unit_id]
                scoredict_by_unit_perm_meg_pitchsplit[unit_id] = scoredict_byunit_trained_perm_pitchsplit[unit_id]

        else:
            if electrode_position_dict['area'] == 'MEG':
                scoredict_by_unit_perm_meg[unit_id] = scoredict_byunit_trained_perm[unit_id]
                scoredict_by_unit_perm_meg_pitchsplit[unit_id] = scoredict_byunit_trained_perm_pitchsplit[unit_id]
            elif electrode_position_dict['area'] == 'PEG':
                scoredict_by_unit_perm_peg[unit_id] = scoredict_byunit_trained_perm[unit_id]
                scoredict_by_unit_perm_peg_pitchsplit[unit_id] = scoredict_byunit_trained_perm_pitchsplit[unit_id]
            elif electrode_position_dict['area'] == 'AEG':
                scoredict_by_unit_perm_aeg[unit_id] = scoredict_byunit_trained_perm[unit_id]
                scoredict_by_unit_perm_aeg_pitchsplit[unit_id] = scoredict_byunit_trained_perm_pitchsplit[unit_id]


    #now plot the scores by aeg, peg, meg for the trained animals
    #initialise a dataframe for each of the areas
    df_full = pd.DataFrame(columns=['ID', 'ProbeWord', 'Score', 'Below-chance', 'BrainArea', 'SingleUnit'])

    df_full_pitchsplit = pd.DataFrame(columns=['ID', 'ProbeWord', 'Score', 'Below-chance', 'BrainArea', 'PitchShift', 'SingleUnit'])

    for unit_id in scoredict_by_unit_meg.keys():
        example_unit = scoredict_by_unit_meg[unit_id]
        for probeword in example_unit.keys():
            su_list = example_unit[probeword]['su_list']
            mu_list = example_unit[probeword]['mu_list']
            if len(su_list)>0:
                #calculate the score
                score = np.mean(su_list)
                #compare with the permutation data
                su_list_perm = scoredict_by_unit_perm_meg[unit_id][probeword]['su_list']
                if score > np.mean(su_list_perm):
                    below_chance = 0
                else:
                    below_chance = 1
                #calculate the below chance
                #add to the dataframe
                df_full = df_full.append({'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance, 'BrainArea': 'MEG',
                                          'SingleUnit': int(1)}, ignore_index=True)
            elif len(mu_list)>0:
                #calculate the score
                score = np.mean(mu_list)
                #compare with the permutation data
                mu_list_perm = scoredict_by_unit_perm_meg[unit_id][probeword]['mu_list']
                if score > np.mean(mu_list_perm):
                    below_chance = 0
                else:
                    below_chance = 1
                #calculate the below chance
                #add to the dataframe
                df_full = df_full.append({'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance, 'BrainArea': 'MEG', 'SingleUnit': 0}, ignore_index=True)

    for unit_id in scoredict_by_unit_peg.keys():
        example_unit = scoredict_by_unit_peg[unit_id]
        for probeword in example_unit.keys():
            su_list = example_unit[probeword]['su_list']
            mu_list = example_unit[probeword]['mu_list']
            if len(su_list)>0:
                #calculate the score
                score = np.mean(su_list)
                #compare with the permutation data
                su_list_perm = scoredict_by_unit_perm_peg[unit_id][probeword]['su_list']
                if score > np.mean(su_list_perm):
                    below_chance = 0
                else:
                    below_chance = 1
                #calculate the below chance
                #add to the dataframe
                df_full = df_full.append({'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance, 'BrainArea': 'PEG',
                                          'SingleUnit': int(1)}, ignore_index=True)
            elif len(mu_list)>0:
                #calculate the score
                score = np.mean(mu_list)
                #compare with the permutation data
                mu_list_perm = scoredict_by_unit_perm_peg[unit_id][probeword]['mu_list']
                if score > np.mean(mu_list_perm):
                    below_chance = 0
                else:
                    below_chance = 1
                #calculate the below chance
                #add to the dataframe
                df_full = df_full.append({'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance, 'BrainArea': 'PEG', 'SingleUnit': 0}, ignore_index=True)

    #repeat for aeg
    for unit_id in scoredict_by_unit_aeg.keys():
        example_unit = scoredict_by_unit_aeg[unit_id]
        for probeword in example_unit.keys():
            su_list = example_unit[probeword]['su_list']
            mu_list = example_unit[probeword]['mu_list']
            if len(su_list) > 0:
                # calculate the score
                score = np.mean(su_list)
                # compare with the permutation data
                su_list_perm = scoredict_by_unit_perm_aeg[unit_id][probeword]['su_list']
                if score > np.mean(su_list_perm):
                    below_chance = 0
                else:
                    below_chance = 1
                # calculate the below chance
                # add to the dataframe
                df_full = df_full.append(
                    {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                     'BrainArea': 'AEG', 'SingleUnit': int(1)}, ignore_index=True)
            elif len(mu_list) > 0:
                # calculate the score
                score = np.mean(mu_list)
                # compare with the permutation data
                mu_list_perm = scoredict_by_unit_perm_aeg[unit_id][probeword]['mu_list']
                if score > np.mean(mu_list_perm):
                    below_chance = 0
                else:
                    below_chance = 1
                # calculate the below chance
                # add to the dataframe
                df_full = df_full.append(
                    {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                     'BrainArea': 'AEG', 'SingleUnit': 0}, ignore_index=True)

    for unit_id in scoredict_by_unit_meg_pitchsplit.keys():
        example_unit = scoredict_by_unit_meg_pitchsplit[unit_id]
        for probeword in example_unit.keys():
            for pitchshiftkey in example_unit[probeword].keys():
                su_list = example_unit[probeword][pitchshiftkey]['su_list']
                mu_list = example_unit[probeword][pitchshiftkey]['mu_list']
                if pitchshiftkey =='pitchshift':
                    pitchshiftnum = 1
                elif pitchshiftkey == 'nonpitchshift':
                    pitchshiftnum = 0


                if len(su_list) > 0:
                    # calculate the score
                    score = np.mean(su_list)
                    # compare with the permutation data
                    su_list_perm = scoredict_by_unit_perm_meg_pitchsplit[unit_id][probeword][pitchshiftkey]['su_list']
                    if score > np.mean(su_list_perm):
                        below_chance = 0
                    else:
                        below_chance = 1
                    # calculate the below chance
                    # add to the dataframe
                    df_full_pitchsplit = df_full_pitchsplit.append(
                        {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                         'BrainArea': 'MEG', 'PitchShift': pitchshiftnum, 'SingleUnit': int(1)}, ignore_index=True)
                elif len(mu_list) > 0:
                    # calculate the score
                    score = np.mean(mu_list)
                    # compare with the permutation data
                    mu_list_perm = scoredict_by_unit_perm_meg_pitchsplit[unit_id][probeword][pitchshiftkey]['mu_list']
                    if score > np.mean(mu_list_perm):
                        below_chance = 0
                    else:
                        below_chance = 1
                    # calculate the below chance
                    # add to the dataframe
                    df_full_pitchsplit = df_full_pitchsplit.append(
                        {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                         'BrainArea': 'MEG', 'PitchShift': pitchshiftnum, 'SingleUnit': 0}, ignore_index=True)
    for unit_id in scoredict_by_unit_peg_pitchsplit.keys():
        example_unit = scoredict_by_unit_peg_pitchsplit[unit_id]
        for probeword in example_unit.keys():
            for pitchshiftkey in example_unit[probeword].keys():
                su_list = example_unit[probeword][pitchshiftkey]['su_list']
                mu_list = example_unit[probeword][pitchshiftkey]['mu_list']
                if pitchshiftkey =='pitchshift':
                    pitchshiftnum = 1
                elif pitchshiftkey == 'nonpitchshift':
                    pitchshiftnum = 0


                if len(su_list) > 0:
                    # calculate the score
                    score = np.mean(su_list)
                    # compare with the permutation data
                    su_list_perm = scoredict_by_unit_perm_peg_pitchsplit[unit_id][probeword][pitchshiftkey]['su_list']
                    if score > np.mean(su_list_perm):
                        below_chance = 0
                    else:
                        below_chance = 1
                    # calculate the below chance
                    # add to the dataframe
                    df_full_pitchsplit = df_full_pitchsplit.append(
                        {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                         'BrainArea': 'PEG', 'PitchShift': pitchshiftnum, 'SingleUnit': int(1)}, ignore_index=True)
                elif len(mu_list) > 0:
                    # calculate the score
                    score = np.mean(mu_list)
                    # compare with the permutation data
                    mu_list_perm = scoredict_by_unit_perm_peg_pitchsplit[unit_id][probeword][pitchshiftkey]['mu_list']
                    if score > np.mean(mu_list_perm):
                        below_chance = 0
                    else:
                        below_chance = 1
                    # calculate the below chance
                    # add to the dataframe
                    df_full_pitchsplit = df_full_pitchsplit.append(
                        {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                         'BrainArea': 'PEG', 'PitchShift': pitchshiftnum, 'SingleUnit': 0}, ignore_index=True)
    for unit_id in scoredict_by_unit_aeg_pitchsplit.keys():
        example_unit = scoredict_by_unit_aeg_pitchsplit[unit_id]
        for probeword in example_unit.keys():
            for pitchshiftkey in example_unit[probeword].keys():
                su_list = example_unit[probeword][pitchshiftkey]['su_list']
                mu_list = example_unit[probeword][pitchshiftkey]['mu_list']
                if pitchshiftkey =='pitchshift':
                    pitchshiftnum = 1
                elif pitchshiftkey == 'nonpitchshift':
                    pitchshiftnum = 0


                if len(su_list) > 0:
                    # calculate the score
                    score = np.mean(su_list)
                    # compare with the permutation data
                    su_list_perm = scoredict_by_unit_perm_aeg_pitchsplit[unit_id][probeword][pitchshiftkey]['su_list']
                    if score > np.mean(su_list_perm):
                        below_chance = 0
                    else:
                        below_chance = 1
                    # calculate the below chance
                    # add to the dataframe
                    df_full_pitchsplit = df_full_pitchsplit.append(
                        {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                         'BrainArea': 'PEG', 'PitchShift': pitchshiftnum, 'SingleUnit': int(1)}, ignore_index=True)
                elif len(mu_list) > 0:
                    # calculate the score
                    score = np.mean(mu_list)
                    # compare with the permutation data
                    mu_list_perm = scoredict_by_unit_perm_aeg_pitchsplit[unit_id][probeword][pitchshiftkey]['mu_list']
                    if score > np.mean(mu_list_perm):
                        below_chance = 0
                    else:
                        below_chance = 1
                    # calculate the below chance
                    # add to the dataframe
                    df_full_pitchsplit = df_full_pitchsplit.append(
                        {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                         'BrainArea': 'PEG', 'PitchShift': pitchshiftnum, 'SingleUnit': 0}, ignore_index=True)

    #plot as a swarm plot with the below chance as a different colour

    for unit_id in df_full['ID']:
        df_full_unit = df_full[df_full['ID'].str.contains(unit_id)]
        #check if all the probe words are below chance
        if np.sum(df_full_unit['Below-chance']) == len(df_full_unit['Below-chance']):
            df_full = df_full[df_full['ID'] != unit_id]


    fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
    sns.stripplot(x='BrainArea', y='Score', hue='Below-chance', data=df_full, ax=ax, alpha=0.5)
    sns.violinplot(x='BrainArea', y='Score', data=df_full, ax=ax, inner=None, color='lightgray')
    plt.title('Trained animals')
    plt.savefig(f'G:/neural_chapter/figures/violinplot_ofdecodingscores_bybrainarea_trainedanimals.png', dpi = 300)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a custom color palette for ProbeWords
    probe_word_palette = sns.color_palette("Set1", n_colors=len(df_full['ProbeWord'].unique()))

    # Define a function to apply different colors for ProbeWords
    def color_by_probeword(probeword):
        return probe_word_palette[df_full['ProbeWord'].unique().tolist().index(probeword)]

    # Filter the DataFrame for above and below chance scores
    df_above_chance = df_full[df_full['Below-chance'] == 0]
    df_below_chance = df_full[df_full['Below-chance'] == 1]

    # Plot the data points color-coded by ProbeWord for above chance scores
    sns.stripplot(x='BrainArea', y='Score', data=df_above_chance, ax=ax, size=3, dodge=False, palette=probe_word_palette,
                  hue='ProbeWord', alpha=0.7, jitter = 0.2)

    # Overlay the data points for below chance scores in grey
    sns.stripplot(x='BrainArea', y='Score', data=df_below_chance, ax=ax, size=3, dodge=False, color='lightgray', alpha=0.5, jitter = 0.2)

    # Overlay the violin plot for visualization
    sns.violinplot(x='BrainArea', y='Score', data=df_full, ax=ax, inner=None, color='white')

    # Customize the legend
    legend = ax.legend(title='ProbeWord')

    # Set custom colors for ProbeWords in the legend
    for idx, probeword in enumerate(df_full['ProbeWord'].unique()):
        legend.get_texts()[idx].set_text(probeword)
        legend.legendHandles[idx].set_color(color_by_probeword(probeword))

    # Add a title
    plt.title('Trained animals')
    plt.savefig(f'G:/neural_chapter/figures/stripplot_overlaidvioline_ofdecodingscores_bybrainarea_trainedanimals.png', dpi = 300)
    plt.show()

    #now plot by animal for the trained animals
    for animal in ['F1604_Squinty', 'F1606_Windolene', 'F1702_Zola', 'F1815_Cruella', 'F1901_Crumble', 'F1902_Eclair', 'F1812_Nala', 'F2003_Orecchiette']:
        fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
        df_full_animal = df_full[df_full['ID'].str.contains(animal)]
        if len(df_full_animal) == 0:
            continue
        df_above_chance = df_full_animal[df_full_animal['Below-chance'] == 0]
        df_below_chance = df_full_animal[df_full_animal['Below-chance'] == 1]

        # Plot the data points color-coded by ProbeWord for above chance scores
        sns.stripplot(x='BrainArea', y='Score', data=df_above_chance, ax=ax, size=3, dodge=False,
                      palette=probe_word_palette,
                      hue='ProbeWord', alpha=0.7, jitter=0.2)

        # Overlay the data points for below chance scores in grey
        sns.stripplot(x='BrainArea', y='Score', data=df_below_chance, ax=ax, size=3, dodge=False, color='lightgray',
                      alpha=0.5, jitter=0.2)

        # Overlay the violin plot for visualization
        sns.violinplot(x='BrainArea', y='Score', data=df_full, ax=ax, inner=None, color='white')

        # Customize the legend
        legend = ax.legend(title='ProbeWord')

        # Set custom colors for ProbeWords in the legend
        # Add a title
        plt.title('Trained animal, {}'.format(animal))
        plt.savefig(f'G:/neural_chapter/figures/violinplot_ofdecodingscores_bybrainarea_{animal}.png', dpi=300)

        # Show the plot
        plt.show()

    ##now do the same for the naive animals
    scoredict_by_unit_naive_meg = {}
    scoredict_by_unit_naive_meg_pitchsplit = {}
    scoredict_by_unit_naive_peg = {}
    scoredict_by_unit_naive_peg_pitchsplit = {}
    scoredict_by_unit_naive_aeg = {}
    scoredict_by_unit_naive_aeg_pitchsplit = {}



    # now sort each of the score_dicts by channel_id
    for unit_id in scoredict_byunit_naive.keys():
        example_unit = scoredict_byunit_naive[unit_id]
        # load the corresponding channel_id

        if 'F1604_Squinty' in unit_id:
            animal = 'F1604_Squinty'
            side = 'left'
        elif 'F1606_Windolene' in unit_id:
            animal = 'F1606_Windolene'
        elif 'F1702_Zola' in unit_id:
            animal = 'F1702_Zola'
        elif 'F1815_Cruella' in unit_id:
            animal = 'F1815_Cruella'
        elif 'F1901_Crumble' in unit_id:
            animal = 'F1901_Crumble'
        elif 'F1902_Eclair' in unit_id:
            animal = 'F1902_Eclair'
        elif 'F1812_Nala' in unit_id:
            animal = 'F1812_Nala'
        elif 'F2003_Orecchiette' in unit_id:
            animal = 'F2003_Orecchiette'

        if 'BB_3' in unit_id and animal != 'F1604_Squinty':
            side = 'right'
        elif 'BB_2' in unit_id and animal != 'F1604_Squinty':
            side = 'right'
        elif 'BB_4' in unit_id:
            side = 'left'
        elif 'BB_5' in unit_id:
            side = 'left'

        for probeword in example_unit.keys():
            try:
                channel_id = example_unit[probeword]['channel_id'][0]
            except:
                continue
            if channel_id is not None:
                break
        # load the corresponding electrode position
        try:
            electrode_position_dict_for_animal = electrode_position_data[animal][side]
            # find where the TDT number is in the electrode position dict
            for electrode_position in electrode_position_dict_for_animal:
                if electrode_position['TDT_NUMBER'] == channel_id:
                    electrode_position_dict = electrode_position
                    break
        except:
            pass
        if animal == 'F2003_Orecchiette':
            if 'mod' in unit_id:
                scoredict_by_unit_naive_peg[unit_id] = scoredict_byunit_naive[unit_id]
                scoredict_by_unit_naive_peg_pitchsplit[unit_id] = scoredict_byunit_naive_pitchsplit[unit_id]
            elif 's2' in unit_id:
                scoredict_by_unit_naive_peg[unit_id] = scoredict_byunit_naive[unit_id]
                scoredict_by_unit_naive_peg_pitchsplit[unit_id] = scoredict_byunit_naive_pitchsplit[unit_id]
            elif 's3' in unit_id:
                scoredict_by_unit_naive_meg[unit_id] = scoredict_byunit_naive[unit_id]
                scoredict_by_unit_naive_meg_pitchsplit[unit_id] = scoredict_byunit_naive_pitchsplit[unit_id]

        else:
            if electrode_position_dict['area'] == 'MEG':
                # add it to a new dictionary
                scoredict_by_unit_naive_meg[unit_id] = scoredict_byunit_naive[unit_id]
                scoredict_by_unit_naive_meg_pitchsplit[unit_id] = scoredict_byunit_naive_pitchsplit[unit_id]
            elif electrode_position_dict['area'] == 'PEG':
                scoredict_by_unit_naive_peg[unit_id] = scoredict_byunit_naive[unit_id]
                scoredict_by_unit_naive_peg_pitchsplit[unit_id] = scoredict_byunit_naive_pitchsplit[unit_id]
            elif electrode_position_dict['area'] == 'AEG':
                scoredict_by_unit_naive_aeg[unit_id] = scoredict_byunit_naive[unit_id]
                scoredict_by_unit_naive_aeg_pitchsplit[unit_id] = scoredict_byunit_naive_pitchsplit[unit_id]
    ##do the same for the permutation data





    scoredict_by_unit_perm_naive_meg = {}
    scoredict_by_unit_perm_naive_meg_pitchsplit = {}
    scoredict_by_unit_perm_naive_peg = {}
    scoredict_by_unit_perm_naive_peg_pitchsplit = {}
    scoredict_by_unit_perm_naive_aeg = {}
    scoredict_by_unit_perm_naive_aeg_pitchsplit = {}
    # now sort each of the score_dicts by channel_id
    for unit_id in scoredict_byunit_naive_perm.keys():
        example_unit = scoredict_byunit_naive_perm[unit_id]
        # load the corresponding channel_id
        if 'F1604_Squinty' in unit_id:
            animal = 'F1604_Squinty'
            side = 'left'
        elif 'F1606_Windolene' in unit_id:
            animal = 'F1606_Windolene'
        elif 'F1702_Zola' in unit_id:
            animal = 'F1702_Zola'
        elif 'F1815_Cruella' in unit_id:
            animal = 'F1815_Cruella'
        elif 'F1901_Crumble' in unit_id:
            animal = 'F1901_Crumble'
        elif 'F1902_Eclair' in unit_id:
            animal = 'F1902_Eclair'
        elif 'F1812_Nala' in unit_id:
            animal = 'F1812_Nala'
        elif 'F2003_Orecchiette' in unit_id:
            animal = 'F2003_Orecchiette'

        if 'BB_3' in unit_id and animal != 'F1604_Squinty':
            side = 'right'
        elif 'BB_2' in unit_id and animal != 'F1604_Squinty':
            side = 'right'
        elif 'BB_4' in unit_id:
            side = 'left'
        elif 'BB_5' in unit_id:
            side = 'left'

        for probeword in example_unit.keys():
            try:
                channel_id = example_unit[probeword]['channel_id'][0]
            except:
                continue
            if channel_id is not None:
                break

        # load the corresponding electrode position
        try:
            electrode_position_dict_for_animal = electrode_position_data[animal][side]
            # find where the TDT number is in the electrode position dict
            for electrode_position in electrode_position_dict_for_animal:
                if electrode_position['TDT_NUMBER'] == channel_id:
                    electrode_position_dict = electrode_position
                    break
        except: pass
        if animal == 'F2003_Orecchiette':
            if 'mod' in unit_id:
                scoredict_by_unit_perm_naive_peg[unit_id] = scoredict_byunit_naive_perm[unit_id]
                scoredict_by_unit_perm_naive_peg_pitchsplit[unit_id] = scoredict_byunit_naive_perm_pitchsplit[unit_id]
            elif 's2' in unit_id:
                scoredict_by_unit_perm_naive_peg[unit_id] = scoredict_byunit_naive_perm[unit_id]
                scoredict_by_unit_perm_naive_peg_pitchsplit[unit_id] = scoredict_byunit_naive_perm_pitchsplit[unit_id]
            elif 's3' in unit_id:
                scoredict_by_unit_perm_naive_meg[unit_id] = scoredict_byunit_naive_perm[unit_id]
                scoredict_by_unit_perm_naive_meg_pitchsplit[unit_id] = scoredict_byunit_naive_perm_pitchsplit[unit_id]

        else:
            if electrode_position_dict['area'] == 'MEG':
                # add it to a new dictionary
                scoredict_by_unit_perm_naive_meg[unit_id] = scoredict_byunit_naive_perm[unit_id]
                scoredict_by_unit_perm_naive_meg_pitchsplit[unit_id] = scoredict_byunit_naive_perm_pitchsplit[unit_id]
            elif electrode_position_dict['area'] == 'PEG':
                scoredict_by_unit_perm_naive_peg[unit_id] = scoredict_byunit_naive_perm[unit_id]
                scoredict_by_unit_perm_naive_peg_pitchsplit[unit_id] = scoredict_byunit_naive_perm_pitchsplit[unit_id]
            elif electrode_position_dict['area'] == 'AEG':
                scoredict_by_unit_perm_naive_aeg[unit_id] = scoredict_byunit_naive_perm[unit_id]
                scoredict_by_unit_perm_naive_aeg_pitchsplit[unit_id] = scoredict_byunit_naive_perm_pitchsplit[unit_id]


    df_full_naive = pd.DataFrame(columns=['ID', 'ProbeWord', 'Score', 'Below-chance', 'BrainArea', 'SingleUnit'])
    df_full_naive_pitchsplit = pd.DataFrame(columns=['ID', 'ProbeWord', 'Score', 'Below-chance', 'BrainArea', 'PitchShift', 'SingleUnit'])

    for unit_id in scoredict_by_unit_naive_meg.keys():
        example_unit = scoredict_by_unit_naive_meg[unit_id]
        for probeword in example_unit.keys():
            su_list = example_unit[probeword]['su_list']
            mu_list = example_unit[probeword]['mu_list']
            if len(su_list) > 0:
                # calculate the score
                score = np.mean(su_list)
                # compare with the permutation data
                su_list_perm = scoredict_by_unit_perm_naive_meg[unit_id][probeword]['su_list']
                if score > np.mean(su_list_perm):
                    below_chance = 0
                else:
                    below_chance = 1
                # calculate the below chance
                # add to the dataframe
                df_full_naive = df_full_naive.append(
                    {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                     'BrainArea': 'MEG', 'SingleUnit': int(1)}, ignore_index=True)
            elif len(mu_list) > 0:
                # calculate the score
                score = np.mean(mu_list)
                # compare with the permutation data
                mu_list_perm = scoredict_by_unit_perm_naive_meg[unit_id][probeword]['mu_list']
                if score > np.mean(mu_list_perm):
                    below_chance = 0
                else:
                    below_chance = 1
                # calculate the below chance
                # add to the dataframe
                df_full_naive = df_full_naive.append(
                    {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                     'BrainArea': 'MEG', 'SingleUnit': 0}, ignore_index=True)

    for unit_id in scoredict_by_unit_naive_peg.keys():
        example_unit = scoredict_by_unit_naive_peg[unit_id]
        for probeword in example_unit.keys():
            su_list = example_unit[probeword]['su_list']
            mu_list = example_unit[probeword]['mu_list']
            if len(su_list) > 0:
                # calculate the score
                score = np.mean(su_list)
                # compare with the permutation data
                su_list_perm = scoredict_by_unit_perm_naive_peg[unit_id][probeword]['su_list']
                if score > np.mean(su_list_perm):
                    below_chance = 0
                else:
                    below_chance = 1
                # calculate the below chance
                # add to the dataframe
                df_full_naive = df_full_naive.append(
                    {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                     'BrainArea': 'PEG', 'SingleUnit': int(1)}, ignore_index=True)
            elif len(mu_list) > 0:
                # calculate the score
                score = np.mean(mu_list)
                # compare with the permutation data
                mu_list_perm = scoredict_by_unit_perm_naive_peg[unit_id][probeword]['mu_list']
                if score > np.mean(mu_list_perm):
                    below_chance = 0
                else:
                    below_chance = 1
                # calculate the below chance
                # add to the dataframe
                df_full_naive = df_full_naive.append(
                    {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                     'BrainArea': 'PEG', 'SingleUnit': 0}, ignore_index=True)

    # repeat for aeg
    for unit_id in scoredict_by_unit_naive_aeg.keys():
        example_unit = scoredict_by_unit_naive_aeg[unit_id]
        for probeword in example_unit.keys():
            su_list = example_unit[probeword]['su_list']
            mu_list = example_unit[probeword]['mu_list']
            if len(su_list) > 0:
                # calculate the score
                score = np.mean(su_list)
                # compare with the permutation data
                su_list_perm = scoredict_by_unit_perm_naive_aeg[unit_id][probeword]['su_list']
                if score > np.mean(su_list_perm):
                    below_chance = 0
                else:
                    below_chance = 1
                # calculate the below chance
                # add to the dataframe
                df_full_naive = df_full_naive.append(
                    {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                     'BrainArea': 'AEG', 'SingleUnit': int(1)}, ignore_index=True)
            elif len(mu_list) > 0:
                # calculate the score
                score = np.mean(mu_list)
                # compare with the permutation data
                mu_list_perm = scoredict_by_unit_perm_naive_aeg[unit_id][probeword]['mu_list']
                if score > np.mean(mu_list_perm):
                    below_chance = 0
                else:
                    below_chance = 1
                # calculate the below chance
                # add to the dataframe
                df_full_naive = df_full_naive.append(
                    {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                     'BrainArea': 'AEG', 'SingleUnit': 0}, ignore_index=True)


    for unit_id in scoredict_by_unit_naive_peg_pitchsplit.keys():
        example_unit = scoredict_by_unit_naive_peg_pitchsplit[unit_id]
        for probeword in example_unit.keys():
            for pitchshiftkey in example_unit[probeword].keys():
                su_list = example_unit[probeword][pitchshiftkey]['su_list']
                mu_list = example_unit[probeword][pitchshiftkey]['mu_list']
                if pitchshiftkey =='pitchshift':
                    pitchshiftnum = 1
                elif pitchshiftkey == 'nonpitchshift':
                    pitchshiftnum = 0

                if len(su_list) > 0:
                    # calculate the score
                    score = np.mean(su_list)
                    # compare with the permutation data
                    su_list_perm = scoredict_by_unit_perm_naive_peg_pitchsplit[unit_id][probeword][pitchshiftkey]['su_list']
                    if score > np.mean(su_list_perm):
                        below_chance = 0
                    else:
                        below_chance = 1
                    # calculate the below chance
                    # add to the dataframe
                    df_full_naive_pitchsplit = df_full_naive_pitchsplit.append(
                        {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                         'BrainArea': 'PEG', 'PitchShift': pitchshiftnum,'SingleUnit': int(1)}, ignore_index=True)
                elif len(mu_list) > 0:
                    # calculate the score
                    score = np.mean(mu_list)
                    # compare with the permutation data
                    mu_list_perm = scoredict_by_unit_perm_naive_peg_pitchsplit[unit_id][probeword][pitchshiftkey]['mu_list']
                    if score > np.mean(mu_list_perm):
                        below_chance = 0
                    else:
                        below_chance = 1
                    # calculate the below chance
                    # add to the dataframe
                    df_full_naive_pitchsplit = df_full_naive_pitchsplit.append(
                        {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                         'BrainArea': 'PEG', 'PitchShift': pitchshiftnum, 'SingleUnit': 0}, ignore_index=True)

    for unit_id in scoredict_by_unit_naive_meg_pitchsplit.keys():
        example_unit = scoredict_by_unit_naive_meg_pitchsplit[unit_id]
        for probeword in example_unit.keys():
            for pitchshiftkey in example_unit[probeword].keys():
                su_list = example_unit[probeword][pitchshiftkey]['su_list']
                mu_list = example_unit[probeword][pitchshiftkey]['mu_list']
                if pitchshiftkey =='pitchshift':
                    pitchshiftnum = 1
                elif pitchshiftkey == 'nonpitchshift':
                    pitchshiftnum = 0

                if len(su_list) > 0:
                    # calculate the score
                    score = np.mean(su_list)
                    # compare with the permutation data
                    su_list_perm = scoredict_by_unit_perm_naive_meg_pitchsplit[unit_id][probeword][pitchshiftkey]['su_list']
                    if score > np.mean(su_list_perm):
                        below_chance = 0
                    else:
                        below_chance = 1
                    # calculate the below chance
                    # add to the dataframe
                    df_full_naive_pitchsplit = df_full_naive_pitchsplit.append(
                        {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                         'BrainArea': 'MEG', 'PitchShift': pitchshiftnum, 'SingleUnit': int(1)}, ignore_index=True)
                elif len(mu_list) > 0:
                    # calculate the score
                    score = np.mean(mu_list)
                    # compare with the permutation data
                    mu_list_perm = scoredict_by_unit_perm_naive_meg_pitchsplit[unit_id][probeword][pitchshiftkey]['mu_list']
                    if score > np.mean(mu_list_perm):
                        below_chance = 0
                    else:
                        below_chance = 1
                    # calculate the below chance
                    # add to the dataframe
                    df_full_naive_pitchsplit = df_full_naive_pitchsplit.append(
                        {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                         'BrainArea': 'MEG', 'PitchShift': pitchshiftnum, 'SingleUnit': 0}, ignore_index=True)
    for unit_id in scoredict_by_unit_naive_aeg_pitchsplit.keys():
        example_unit = scoredict_by_unit_naive_aeg_pitchsplit[unit_id]
        for probeword in example_unit.keys():
            for pitchshiftkey in example_unit[probeword].keys():
                su_list = example_unit[probeword][pitchshiftkey]['su_list']
                mu_list = example_unit[probeword][pitchshiftkey]['mu_list']
                if pitchshiftkey =='pitchshift':
                    pitchshiftnum = 1
                elif pitchshiftkey == 'nonpitchshift':
                    pitchshiftnum = 0

                if len(su_list) > 0:
                    # calculate the score
                    score = np.mean(su_list)
                    # compare with the permutation data
                    su_list_perm = scoredict_by_unit_perm_naive_aeg_pitchsplit[unit_id][probeword][pitchshiftkey]['su_list']
                    if score > np.mean(su_list_perm):
                        below_chance = 0
                    else:
                        below_chance = 1
                    # calculate the below chance
                    # add to the dataframe
                    df_full_naive_pitchsplit = df_full_naive_pitchsplit.append(
                        {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                         'BrainArea': 'AEG', 'PitchShift': pitchshiftnum, 'SingleUnit': int(1)}, ignore_index=True)
                elif len(mu_list) > 0:
                    # calculate the score
                    score = np.mean(mu_list)
                    # compare with the permutation data
                    mu_list_perm = scoredict_by_unit_perm_naive_aeg_pitchsplit[unit_id][probeword][pitchshiftkey]['mu_list']
                    if score > np.mean(mu_list_perm):
                        below_chance = 0
                    else:
                        below_chance = 1
                    # calculate the below chance
                    # add to the dataframe
                    df_full_naive_pitchsplit = df_full_naive_pitchsplit.append(
                        {'ID': unit_id, 'ProbeWord': probeword, 'Score': score, 'Below-chance': below_chance,
                         'BrainArea': 'AEG', 'PitchShift': pitchshiftnum, 'SingleUnit': 0}, ignore_index=True)

    # plot as a swarm plot with the below chance as a different colour
    #do the same for the naive animals
    for unit_id in df_full_naive['ID']:
        df_full_unit_naive = df_full_naive[df_full_naive['ID'].str.contains(unit_id)]
        #check if all the probe words are below chance
        if np.sum(df_full_unit_naive['Below-chance']) == len(df_full_unit_naive['Below-chance']):
            df_full_naive = df_full_naive[df_full_naive['ID'] != unit_id]

    fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
    sns.stripplot(x='BrainArea', y='Score', hue='Below-chance', data=df_full_naive, ax=ax, alpha=0.5)
    sns.violinplot(x='BrainArea', y='Score', data=df_full_naive, ax=ax, inner=None, color='lightgray')
    plt.title('Naive animals')
    plt.savefig(f'G:/neural_chapter/figures/violinplot_by_area_score_naiveanimals.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a custom color palette for ProbeWords
    probe_word_palette = sns.color_palette("Set2", n_colors=len(df_full['ProbeWord'].unique()))

    # Define a function to apply different colors for ProbeWords
    def color_by_probeword(probeword):
        return probe_word_palette[df_full_naive['ProbeWord'].unique().tolist().index(probeword)]

    # Filter the DataFrame for above and below chance scores
    df_above_chance_naive = df_full_naive[df_full_naive['Below-chance'] == 0]
    df_below_chance_naive = df_full_naive[df_full_naive['Below-chance'] == 1]

    # Plot the data points color-coded by ProbeWord for above chance scores
    sns.stripplot(x='BrainArea', y='Score', data=df_above_chance_naive, ax=ax, size=3, dodge=False,
                  palette=probe_word_palette,
                  hue='ProbeWord', alpha=0.7, jitter=0.2)

    # Overlay the data points for below chance scores in grey
    sns.stripplot(x='BrainArea', y='Score', data=df_below_chance_naive, ax=ax, size=3, dodge=False, color='lightgray',
                  alpha=0.5, jitter=0.2)

    # Overlay the violin plot for visualization
    sns.violinplot(x='BrainArea', y='Score', data=df_full_naive, ax=ax, inner=None, color='white')

    # Customize the legend
    legend = ax.legend(title='ProbeWord')

    # Add a title
    plt.title('Naive animals')
    plt.savefig(f'G:/neural_chapter/figures/violinplot_ofdecodingscores_bybrainarea_naiveanimals.png', dpi = 300)
    # Show the plot
    plt.show()
    anova_table_naive, anova_model_naive = run_anova_on_dataframe(df_full_naive_pitchsplit)
    # now plot by animal for the trained animals
    for animal in [ 'F1901_Crumble', 'F1902_Eclair',
                   'F1812_Nala', 'F2003_Orecchiette']:
        fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
        df_full_animal = df_full_naive[df_full_naive['ID'].str.contains(animal)]
        if len(df_full_animal) == 0:
            continue
        df_above_chance = df_full_animal[df_full_animal['Below-chance'] == 0]
        df_below_chance = df_full_animal[df_full_animal['Below-chance'] == 1]

        # Plot the data points color-coded by ProbeWord for above chance scores
        sns.stripplot(x='BrainArea', y='Score', data=df_above_chance, ax=ax, size=3, dodge=False,
                      palette=probe_word_palette,
                      hue='ProbeWord', alpha=0.7, jitter=0.2)

        # Overlay the data points for below chance scores in grey
        sns.stripplot(x='BrainArea', y='Score', data=df_below_chance, ax=ax, size=3, dodge=False, color='lightgray',
                      alpha=0.5, jitter=0.2)

        # Overlay the violin plot for visualization
        sns.violinplot(x='BrainArea', y='Score', data=df_full, ax=ax, inner=None, color='white')

        # Customize the legend
        legend = ax.legend(title='ProbeWord')

        # Set custom colors for ProbeWords in the legend
        # Add a title
        plt.title('Naive animal, {}'.format(animal))
        plt.savefig(f'G:/neural_chapter/figures/violinplot_by_area_score_naive_{animal}.png')

        # Show the plot
        plt.show()





    for unit_id in df_full_naive_pitchsplit['ID']:
        df_full_unit_naive = df_full_naive_pitchsplit[df_full_naive_pitchsplit['ID'].str.contains(unit_id)]
        #check if all the probe words are below chance
        if np.sum(df_full_unit_naive['Below-chance']) == len(df_full_unit_naive['Below-chance']):
            df_full_naive_pitchsplit = df_full_naive_pitchsplit[df_full_naive_pitchsplit['ID'] != unit_id]

    for unit_id in df_full_pitchsplit['ID']:
        df_full_unit = df_full_pitchsplit[df_full_pitchsplit['ID'].str.contains(unit_id)]
        #check if all the probe words are below chance
        if np.sum(df_full_unit['Below-chance']) == len(df_full_unit['Below-chance']):
            df_full_pitchsplit = df_full_pitchsplit[df_full_pitchsplit['ID'] != unit_id]



    for options in ['index', 'frac']:
        df_full_pitchsplit_highsubset = create_gen_frac_variable(df_full_pitchsplit, high_score_threshold=True, index_or_frac = options)
        #remove all rows where GenFrac is nan
        df_full_pitchsplit_plot = df_full_pitchsplit_highsubset[df_full_pitchsplit_highsubset['GenFrac'].notna()]
        df_full_pitchsplit_plot = df_full_pitchsplit_plot.drop_duplicates(subset = ['ID'])

        df_full_naive_pitchsplit = create_gen_frac_variable(df_full_naive_pitchsplit, high_score_threshold=True, index_or_frac = options)
        df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit[df_full_naive_pitchsplit['GenFrac'].notna()]
        df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit_plot.drop_duplicates(subset = ['ID'])

        #plot the distplot of these scores overlaid with the histogram
        fig, ax = plt.subplots(1, dpi=300)
        sns.histplot(df_full_pitchsplit_plot['GenFrac'],ax=ax,  kde=True, bins=20, color = 'purple', label='Trained')
        sns.histplot(df_full_naive_pitchsplit_plot['GenFrac'], ax=ax, kde=True, bins=20, color = 'cyan', label='Naive')
        plt.legend()
        # plt.title(f'Distribution of generalizability scores for the trained and naive animals, upper quartile threshold, index or frac:{options}')
        if options == 'index':
            plt.xlabel('Generalizability Index of Top 25% of Units', fontsize = 20)
        elif options == 'frac':
            plt.xlabel('Generalizability Fraction of Top 25% of Units', fontsize = 20)

        plt.ylabel('Count', fontsize = 20)
        plt.savefig(f'G:/neural_chapter/figures/GenFrac_highthreshold_{options}.png')
        plt.show()

        #plot as a violin plot with brainarea on the x axis
        fig, ax = plt.subplots(1, dpi=300)
        sns.violinplot(x='BrainArea', y='GenFrac', data=df_full_pitchsplit_plot, ax=ax, inner=None, color='lightgray')
        sns.stripplot(x='BrainArea', y='GenFrac', data=df_full_pitchsplit_plot, ax=ax, size=3, color = 'purple', dodge=False)
        # plt.title(f'Generalizability scores for the trained animals, upper quartile threshold, index or frac:{options}')
        if options == 'index':
            plt.xlabel('Generalizability Index of Top 25% of Units', fontsize = 20)
        elif options == 'frac':
            plt.xlabel('Generalizability Fraction of Top 25% of Units', fontsize = 20)
        plt.ylabel('Count', fontsize = 20)
        plt.savefig(f'G:/neural_chapter/figures/GenFrac_highthreshold_violin_{options}.png')
        plt.show()

        fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
        sns.violinplot(x='BrainArea', y='GenFrac', data=df_full_naive_pitchsplit_plot, ax=ax, inner=None, color='lightgray')
        sns.stripplot(x='BrainArea', y='GenFrac', data=df_full_naive_pitchsplit_plot, ax=ax, size=3, dodge=False)
        # plt.title(f'Generalizability scores for the naive animals, 60% mean score threshold, index or frac:{options}')
        if options == 'index':
            plt.xlabel('Generalizability Index of Top 25% of Units', fontsize = 20)
        elif options == 'frac':
            plt.xlabel('Generalizability Fraction of Top 25% of Units', fontsize = 20)
        plt.savefig(f'G:/neural_chapter/figures/GenFrac_highthreshold_violin_naive_{options}.png')
        plt.show()

        #do the mann whitney u test between genfrac scores from PEG and MEG
        df_full_pitchsplit_plot_peg = df_full_pitchsplit_plot[df_full_pitchsplit_plot['BrainArea'] == 'PEG']
        df_full_pitchsplit_plot_meg = df_full_pitchsplit_plot[df_full_pitchsplit_plot['BrainArea'] == 'MEG']
        df_full_naive_pitchsplit_plot_peg = df_full_naive_pitchsplit_plot[df_full_naive_pitchsplit_plot['BrainArea'] == 'PEG']
        df_full_naive_pitchsplit_plot_meg = df_full_naive_pitchsplit_plot[df_full_naive_pitchsplit_plot['BrainArea'] == 'MEG']

        stat_peg, p_peg = mannwhitneyu(df_full_pitchsplit_plot_peg['GenFrac'], df_full_pitchsplit_plot_meg['GenFrac'], alternative = 'less')
        #plot the brain area loation of the units that are significantly different
        fig, ax = plt.subplots(1, dpi=300)
        sns.violinplot(x='BrainArea', y='GenFrac', data=df_full_pitchsplit_plot, ax=ax, inner=None, color='lightgray')
        sns.stripplot(x='BrainArea', y='GenFrac', data=df_full_pitchsplit_plot, ax=ax, color = 'purple', dodge=False)
        plt.title(f'Generalizability scores for the trained animals, upper quartile threshold, index or frac:{options}')
        if options == 'index':

            plt.xlabel('Generalizability Index of Top 25% of Units', fontsize = 20)
        elif options == 'frac':

            plt.xlabel('Generalizability Fraction of Top 25% of Units', fontsize = 20)
        plt.ylabel('Count', fontsize = 20)
        plt.savefig(f'G:/neural_chapter/figures/GenFrac_highthreshold_violin_bybrainarea_{options}.png')

        fig, ax = plt.subplots(1, dpi=300)
        sns.violinplot(x='BrainArea', y='MeanScore', data=df_full_pitchsplit_plot, ax=ax, inner=None, color='lightgray')
        sns.stripplot(x='BrainArea', y='MeanScore', data=df_full_pitchsplit_plot, color = 'purple', ax=ax, dodge=False)
        # do a mann whitney u test between the meanscores for PEG and MEG
        stat_peg, p_peg = mannwhitneyu(df_full_pitchsplit_plot_peg['MeanScore'], df_full_pitchsplit_plot_meg['MeanScore'], alternative='greater')
        if options == 'index':
            plt.xlabel(f'Mean Decoding Score of Top 25% of Units', fontsize=20)
        elif options == 'frac':
            plt.xlabel(f'Mean Decoding Score of Top 25% of Units', fontsize=20)
        plt.ylabel('Mean Score', fontsize=20)
        plt.savefig(f'G:/neural_chapter/figures/meanscore_highthreshold_violin_bybrainarea_{options}.png')

        fig, ax = plt.subplots(1, dpi=300)
        sns.violinplot(x='BrainArea', y='GenFrac', data=df_full_naive_pitchsplit_plot, ax=ax, inner=None, color='lightgray')
        sns.stripplot(x='BrainArea', y='GenFrac', data=df_full_naive_pitchsplit_plot, ax=ax, color ='cyan', dodge=False)
        plt.title(f'Generalizability scores for the naive animals, upper quartile threshold, index or frac:{options}')
        if options == 'index':
            plt.xlabel('Generalizability Index of Top 25% of Units', fontsize = 20)
        elif options == 'frac':
            plt.xlabel('Generalizability Fraction of Top 25% of Units', fontsize = 20)
        plt.ylabel('GenFrac', fontsize = 20)
        plt.savefig(f'G:/neural_chapter/figures/GenFrac_highthreshold_violin_naive_bybrainarea_{options}.png')

        stat_peg, p_peg = mannwhitneyu(df_full_naive_pitchsplit_plot_peg['GenFrac'], df_full_naive_pitchsplit_plot_meg['GenFrac'], alternative = 'less')
        #plot the brain area loation of the units that are significantly different
        fig, ax = plt.subplots(1, dpi=300)
        #make the dataframe in the order of PEG, MEG, AEG
        df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit_plot.sort_values(by=['BrainArea'])

        sns.violinplot(x='BrainArea', y='MeanScore', data=df_full_naive_pitchsplit_plot, ax=ax, inner=None, color='lightgray', order = ['MEG', 'PEG', 'AEG'])
        sns.stripplot(x='BrainArea', y='MeanScore', data=df_full_naive_pitchsplit_plot, ax=ax, color = 'cyan', dodge=False, order = ['MEG', 'PEG', 'AEG'])
        # do a mann whitney u test between the meanscores for PEG and MEG
        stat_peg_naive, p_peg_naive = mannwhitneyu(df_full_naive_pitchsplit_plot_peg['MeanScore'], df_full_naive_pitchsplit_plot_meg['MeanScore'], alternative='less')
        plt.xlabel(f'Mean Decoding Score of Top 25% of Unit', fontsize=20)
        plt.ylabel('Mean Score', fontsize=20)
        plt.savefig(f'G:/neural_chapter/figures/meanscore_highthreshold_violin_naive_bybrainarea.png')

        n1 = len(df_full_naive_pitchsplit_plot_peg)
        n2 = len(df_full_naive_pitchsplit_plot_meg)
        r_naive = 1 - (2 * stat_peg_naive) / (n1 * n2)

        n1 = len(df_full_pitchsplit_plot_peg)
        n2 = len(df_full_pitchsplit_plot_meg)
        r_trained = 1 - (2 * stat_peg) / (n1 * n2)

        #export the p values to a csv file
        df_pvalues = pd.DataFrame(columns = ['Trained/naive', 'pvalue', 'statistic', 'effectsize'])
        df_pvalues = df_pvalues.append({'Trained/naive': 'Trained', 'pvalue': p_peg, 'statistic': stat_peg, 'effectsize': r_trained}, ignore_index = True)
        df_pvalues = df_pvalues.append({'Trained/naive': 'Naive', 'pvalue': p_peg_naive, 'statistic': stat_peg_naive, 'effectsize': r_naive}, ignore_index = True)
        df_pvalues.to_csv(f'G:/neural_chapter/figures/pvalues_highthreshold_manwhittest_{options}.csv')




        #man whitney u test
        stat, p = mannwhitneyu(df_full_pitchsplit_plot['GenFrac'], df_full_naive_pitchsplit_plot['GenFrac'], alternative = 'less')
        print(f'Generalizability scores, high threshold untis, index method: {options}')
        print(stat)
        print(p)

        df_full_pitchsplit_allsubset = create_gen_frac_variable(df_full_pitchsplit, high_score_threshold=False, index_or_frac = options)
        #remove all rows where GenFrac is nan
        df_full_pitchsplit_plot = df_full_pitchsplit_allsubset[df_full_pitchsplit_allsubset['GenFrac'].notna()]
        df_full_pitchsplit_plot = df_full_pitchsplit_plot.drop_duplicates(subset = ['ID'])
        #get the subset of the data where the meanscore is above 0.75
        df_full_pitchsplit_plot_highsubset = df_full_pitchsplit_plot[df_full_pitchsplit_plot['MeanScore'] > 0.75]

        df_full_naive_pitchsplit_plot = create_gen_frac_variable(df_full_naive_pitchsplit, high_score_threshold=False, index_or_frac = options)
        df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit_plot[df_full_naive_pitchsplit_plot['GenFrac'].notna()]
        df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit_plot.drop_duplicates(subset = ['ID'])



        fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
        sns.histplot(df_full_pitchsplit_plot['GenFrac'],ax=ax,  kde=True, bins=20, label='Trained')
        sns.histplot(df_full_pitchsplit_plot_highsubset['GenFrac'],ax=ax,  kde=True, bins=20, label='Trained, 75% mean score threshold')
        sns.histplot(df_full_naive_pitchsplit_plot['GenFrac'], ax=ax, kde=True, bins=20, label='Naive')
        # sns.histplot(df_full_naive_pitchsplit_plot_highsubset['GenFrac'], ax=ax, kde=True, bins=20, label='Naive, 75% mean score threshold')
        plt.legend()
        plt.title(f'Distribution of generalisability scores for the trained and naive animals, all units, index method: {options}')



        stat_general, p_general = mannwhitneyu(df_full_pitchsplit_plot['GenFrac'],
                                               df_full_naive_pitchsplit_plot['GenFrac'], alternative='two-sided')
        print(f'Generalizability scores, all units, index method: {options}')
        print(stat_general)
        print(p_general)

        plt.savefig(f'G:/neural_chapter/figures/GenFrac_allthreshold_{options}.png')
        plt.show()

        #plot as a violin plot with brainarea on the x-axis
        fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)

        sns.violinplot(x='BrainArea', y='GenFrac', data=df_full_pitchsplit_plot, ax=ax, inner=None, color='lightgray')
        sns.stripplot(x='BrainArea', y='GenFrac', data=df_full_pitchsplit_plot, ax=ax, size=3, dodge=False)
        plt.title(f'Generalizability scores for the trained animals, all units, method: {options}')
        if options == 'index':
            plt.xlabel('Generalizability Index ', fontsize = 20)
        elif options == 'frac':
            plt.xlabel('Generalizability Fraction', fontsize = 20)

        plt.savefig(f'G:/neural_chapter/figures/GenFrac_allthreshold_violin_{options}.png')
        plt.show()

        fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
        sns.violinplot(x='BrainArea', y='GenFrac', data=df_full_naive_pitchsplit_plot, ax=ax, inner=None, color='lightgray')
        sns.stripplot(x='BrainArea', y='GenFrac', data=df_full_naive_pitchsplit_plot, ax=ax, size=3, dodge=False)

        if options == 'index':
            plt.xlabel('Generalizability Index', fontsize = 20)
        elif options == 'frac':
            plt.xlabel('Generalizability Fraction', fontsize = 20)
        plt.title(f'Generalizability scores for the naive animals, all units, method: {options}')
        plt.savefig(f'G:/neural_chapter/figures/GenFrac_allthreshold_violin_naive_{options}.png')
        #do the mann whitney u test between genfrac scores from PEG and MEG
        df_full_pitchsplit_plot_peg = df_full_pitchsplit_plot[df_full_pitchsplit_plot['BrainArea'] == 'PEG']
        df_full_pitchsplit_plot_meg = df_full_pitchsplit_plot[df_full_pitchsplit_plot['BrainArea'] == 'MEG']
        df_full_naive_pitchsplit_plot_peg = df_full_naive_pitchsplit_plot[df_full_naive_pitchsplit_plot['BrainArea'] == 'PEG']
        df_full_naive_pitchsplit_plot_meg = df_full_naive_pitchsplit_plot[df_full_naive_pitchsplit_plot['BrainArea'] == 'MEG']

        stat_peg, p_peg = mannwhitneyu(df_full_pitchsplit_plot_peg['GenFrac'], df_full_pitchsplit_plot_meg['GenFrac'], alternative = 'less')
        print(f'p value for PEG vs MEG genfrac scores for trained animals, {options} , {p_peg}')
        stat_peg_naive, p_peg_naive = mannwhitneyu(df_full_naive_pitchsplit_plot_peg['GenFrac'], df_full_naive_pitchsplit_plot_meg['GenFrac'], alternative = 'less')
        print(f'p value for PEG vs MEG genfrac scores for naive animals, {options} , {p_peg}')


    #now plot by the probe word for the trained animals
    fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)

    # Plot the data points color-coded by ProbeWord for above chance scores
    sns.stripplot(x='ProbeWord', y='Score', data=df_full, ax=ax, size=3, dodge=True,
                  palette='Set1',
                  hue='Below-chance', alpha=1, jitter=True)
    sns.violinplot(x='ProbeWord', y='Score', data=df_full, ax=ax, color='white')
    plt.title('Trained animals'' scores over distractor word')
    plt.savefig(f'G:/neural_chapter/figurestrained_animals_overdistractor.png', dpi = 300)
    plt.show()

    #plot strip plot split by pitch shift
    fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
    df_above_chance_ps = df_full_pitchsplit[df_full_pitchsplit['Below-chance'] == 0]
    df_below_chance_ps = df_full_pitchsplit[df_full_pitchsplit['Below-chance'] == 1]

    sns.stripplot(x='ProbeWord', y='Score', data=df_above_chance_ps, ax=ax, size=3, dodge=True, edgecolor = 'k', linewidth=0.1, hue='PitchShift')
    sns.stripplot(x='ProbeWord', y='Score', data=df_below_chance_ps, ax=ax, size=3, dodge=True,edgecolor = 'k', linewidth=0.1,
                  alpha=1, jitter=False, hue='PitchShift')

    sns.violinplot(x='ProbeWord', y='Score', data=df_full_pitchsplit, ax=ax, hue = 'PitchShift')
    plt.title('Trained animals'' scores over distractor word')
    plt.savefig(f'G:/neural_chapter/figurestrained_animals_overdistractor_dividedbypitchshift.png', dpi = 300)
    plt.show()

    df_kruskal = pd.DataFrame(columns=['ProbeWord', 'Kruskal_pvalue_trained', 'less than 0.05', 'epsilon_squared'])
    # Perform Kruskal-Wallis test for each ProbeWord
    for probe_word in df_above_chance_ps['ProbeWord'].unique():
        subset_data = df_above_chance_ps[df_above_chance_ps['ProbeWord'] == probe_word]

        result_kruskal = scipy.stats.kruskal(subset_data[subset_data['PitchShift'] == 1]['Score'],
                                             subset_data[subset_data['PitchShift'] == 0]['Score'])
        less_than_alpha = result_kruskal.pvalue < 0.05

        n1 = len(subset_data[subset_data['PitchShift'] == 1]['Score'])
        n2 = len(subset_data[subset_data['PitchShift'] == 0]['Score'])
        N = n1 + n2
        # Calculate degrees of freedom for the Kruskal-Wallis test
        k = 2  # Assuming two groups
        df_H = k - 1
        # Calculate total degrees of freedom
        df_T = N - 1
        # Calculate epsilon-squared (effect size)
        epsilon_squared = (result_kruskal.statistic - df_H) / (N - df_T)

        print(f"ProbeWord: {probe_word}, Kruskal-Wallis p-value: {result_kruskal.pvalue}, epsilon-squared: {epsilon_squared}")
        # append to a dataframe
        df_kruskal = df_kruskal.append({'ProbeWord': probe_word, 'Kruskal_pvalue_trained': result_kruskal.pvalue,
                                        'less than 0.05': less_than_alpha, 'epsilon_squared': epsilon_squared},
                                       ignore_index=True)
    # export the dataframe
    df_kruskal.to_csv('G:/neural_chapter/figures/kruskal_pvalues_trained.csv')

    ##do the roved - control f0 score divided by the control f0 score plot
    #first get the data into a format that can be analysed
    rel_frac_list_naive = []
    bigconcatenatenaive_ps = []
    bigconcatenatenaive_nonps = []
    bigconcatenatetrained_nonps =[]
    bigconcatenatetrained_ps = []
    rel_frac_list_trained = []
    for unit_id in df_full_naive_pitchsplit['ID']:
        df_full_unit_naive = df_full_naive_pitchsplit[df_full_naive_pitchsplit['ID'] == unit_id]
        #get all the scores where pitchshift is 1 for the each probe word
        for probeword in df_full_unit_naive['ProbeWord'].unique():
            try:
                control_df = df_full_unit_naive[(df_full_unit_naive['ProbeWord'] == probeword) & (df_full_unit_naive['PitchShift'] == 0) & (df_full_unit_naive['Below-chance'] == 0)]
                roved_df = df_full_unit_naive[(df_full_unit_naive['ProbeWord'] == probeword) & (df_full_unit_naive['PitchShift'] == 1) & (df_full_unit_naive['Below-chance'] == 0)]
                if len(control_df) == 0 and len(roved_df) == 0:
                    continue

                control_score = df_full_unit_naive[(df_full_unit_naive['ProbeWord'] == probeword) & (df_full_unit_naive['PitchShift'] == 0)]['Score'].values[0]
                pitchshift_score = df_full_unit_naive[(df_full_unit_naive['ProbeWord'] == probeword) & (df_full_unit_naive['PitchShift'] == 1)]['Score'].values[0]
            except:
                continue
            if control_score is not None and pitchshift_score is not None:
                rel_score = (pitchshift_score-control_score)/control_score
                rel_frac_list_naive.append(rel_score)
                bigconcatenatenaive_ps.append(pitchshift_score)
                bigconcatenatenaive_nonps.append(control_score)
    for unit_id in df_full_pitchsplit['ID']:
            df_full_unit= df_full_pitchsplit[df_full_pitchsplit['ID'] == unit_id]
            # get all the scores where pitchshift is 1 for the each probe word
            for probeword in df_full_unit['ProbeWord'].unique():
                try:
                    control_df = df_full_unit_naive[
                        (df_full_unit_naive['ProbeWord'] == probeword) & (df_full_unit_naive['PitchShift'] == 0) & (
                                    df_full_unit_naive['Below-chance'] == 0)]
                    roved_df = df_full_unit_naive[
                        (df_full_unit_naive['ProbeWord'] == probeword) & (df_full_unit_naive['PitchShift'] == 1) & (
                                    df_full_unit_naive['Below-chance'] == 0)]
                    if len(control_df) == 0 and len(roved_df) == 0:
                        continue
                    control_score = df_full_unit[
                        (df_full_unit['ProbeWord'] == probeword) & (df_full_unit['PitchShift'] == 0)][
                        'Score'].values[0]
                    pitchshift_score = df_full_unit[
                        (df_full_unit['ProbeWord'] == probeword) & (df_full_unit['PitchShift'] == 1)][
                        'Score'].values[0]
                except:
                    continue
                if control_score is not None and pitchshift_score is not None:
                    rel_score = (pitchshift_score - control_score) / control_score
                    rel_frac_list_trained.append(rel_score)
                    bigconcatenatetrained_nonps.append(control_score)
                    bigconcatenatetrained_ps.append(pitchshift_score)
        #check if all the probe words are below chance
    fig, ax = plt.subplots(1, figsize=(10, 10), dpi=300)
    # sns.distplot(rel_frac_list_trained, bins=20, label='trained', ax=ax, color='purple')
    # sns.distplot(rel_frac_list_naive, bins=20, label='naive', ax=ax, color='darkcyan')
    sns.histplot(rel_frac_list_trained,  label='trained', color='purple', kde=True)
    sns.histplot(rel_frac_list_naive, label='naive', color='darkcyan', kde=True)

    # get the peak of the distribution on the y axis

    #get the peak of the distribution on the y axis
    x2 = ax.lines[1].get_xdata()  # Get the x data of the distribution
    y2 = ax.lines[1].get_ydata()  # Get the y data of the distribution
    maxidnaive_idx = np.argmax(y2)  # The id of the peak (maximum of y data)
    x_coord_naive = x2[maxidnaive_idx]

    plt.axvline(x=0, color='black')

    x1 = ax.lines[0].get_xdata()  # Get the x data of the distribution
    y1 = ax.lines[0].get_ydata()  # Get the y data of the distribution
    maxidtrained_idx = np.argmax(y1)  # The id of the peak (maximum of y data)
    x_coord_trained = x1[maxidtrained_idx]


    # Perform a t-test on the samples
    t_stat, p_value = stats.ttest_ind(rel_frac_list_trained, rel_frac_list_naive, alternative='greater')

    # Print the t-statistic and p-value
    print(t_stat, p_value)
    plt.title('Control - roved F0 \n LSTM decoder scores between trained and naive animals', fontsize=18)
    plt.xlabel('Control - roved F0 \n LSTM decoder scores divided by Control F0', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    # ax.legend()
    plt.savefig('G:/neural_chapter/figures/diffF0distribution_20062023.png', dpi=1000)
    plt.show()
    fig, ax = plt.subplots(1, figsize=(9,9), dpi=300)
    ax.scatter(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, marker='P', color='purple', alpha=0.5, label='trained')
    ax.scatter(bigconcatenatenaive_nonps, bigconcatenatenaive_ps, marker='P', color='darkcyan', alpha=0.5, label='naive')
    x = np.linspace(0.4, 1, 101)
    ax.plot(x, x, color='black', linestyle = '--')  # identity line

    slope, intercept, r_value, pv, se = stats.linregress(bigconcatenatetrained_nonps, bigconcatenatetrained_ps)

    sns.regplot(x=bigconcatenatetrained_nonps, y=bigconcatenatetrained_ps, scatter=False, color='purple',
                label=' $y=%3.7s*x+%3.7s$' % (slope, intercept), ax=ax, line_kws={'label': ' $y=%3.7s*x+%3.7s$' % (slope, intercept)})
    slope, intercept, r_value, pv, se = stats.linregress(bigconcatenatenaive_nonps, bigconcatenatenaive_ps)

    sns.regplot(x=bigconcatenatenaive_nonps, y=bigconcatenatenaive_ps, scatter=False, color='darkcyan', label=' $y=%3.7s*x+%3.7s$' % (slope, intercept),
                ax=ax, line_kws={'label': '$y=%3.7s*x+%3.7s$' % (slope, intercept)})

    ax.set_ylabel('LSTM decoding score, F0 roved', fontsize=18)
    ax.set_xlabel('LSTM decoding score, F0 control', fontsize=18)

    ax.set_title('LSTM decoder scores for' + ' F0 control vs. roved,\n ' + ' trained and naive animals', fontsize=20)


    plt.legend( fontsize=12, ncol=2)
    fig.tight_layout()
    plt.savefig('G:/neural_chapter/figures/scattermuaandsuregplot_mod_21062023.png', dpi=1000)
    plt.savefig('G:/neural_chapter/figures/scattermuaandsuregplot_mod_21062023.pdf', dpi=1000)
    plt.show()

    unique_unit_ids_naive = df_full_naive_pitchsplit['ID'].unique()
    unique_unit_ids_trained = df_full_pitchsplit['ID'].unique()
    #make a kde plot
    #makea  dataframe
    df_naive = pd.DataFrame({'F0_control': bigconcatenatenaive_nonps, 'F0_roved': bigconcatenatenaive_ps})
    fig, ax = plt.subplots(1, figsize=(9,9), dpi=300)
    sns.kdeplot(df_naive, x='F0_control', y ='F0_roved', shade=True, shade_lowest=False, ax=ax, label='naive')
    plt.title('F0 control vs. roved, naive animals',fontsize = 20)
    plt.ylabel('F0 roved score', fontsize = 20)
    plt.xlabel('F0 control score',fontsize = 20)
    plt.savefig('G:/neural_chapter/figures/kdeplot_naiveanimals.png', dpi=300)
    plt.show()

    df_trained_kde = pd.DataFrame({'F0_control': bigconcatenatetrained_nonps, 'F0_roved': bigconcatenatetrained_ps})
    fig, ax = plt.subplots(1, figsize=(9,9), dpi=300)
    sns.kdeplot(df_trained_kde, x= 'F0_control', y = 'F0_roved', cmap="Reds", shade=True, shade_lowest=False, ax=ax, label='trained')
    plt.title('F0 control vs. roved, trained animals', fontsize = 20)
    plt.ylabel('F0 roved score',fontsize = 20)
    plt.xlabel('F0 control score', fontsize = 20)
    plt.savefig('G:/neural_chapter/figures/kdeplot_trainedanimals.png', dpi=300)
    plt.show()
    manwhitscorecontrolf0 = mannwhitneyu(bigconcatenatetrained_nonps, bigconcatenatenaive_nonps, alternative='greater')

    n1 = len(bigconcatenatetrained_nonps)
    n2 = len(bigconcatenatenaive_nonps)
    r_controlf0 = 1 - (2 * manwhitscorecontrolf0.statistic) / (n1 * n2)
    # ax.legend()
    plt.savefig('G:/neural_chapter/figures/controlF0distribution20062023intertrialroving.png', dpi=1000)
    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=800)
    ax.set_xlim([0, 1])
    sns.distplot(bigconcatenatetrained_ps, label='trained', ax=ax, color='purple')
    sns.distplot(bigconcatenatenaive_ps, label='naive', ax=ax, color='darkcyan')
    # man whiteney test score
    # manwhitscore = mannwhitneyu(relativescoretrained, relativescorenaive, alternative = 'greater')
    plt.title('Roved F0 LSTM decoder scores between  \n trained and naive animals', fontsize=18)
    plt.xlabel('Roved F0 LSTM decoder scores', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    manwhitscorerovedf0 = mannwhitneyu(bigconcatenatetrained_ps, bigconcatenatenaive_ps, alternative='greater')
    plt.xlim([0.35, 1])

    n1 = len(bigconcatenatetrained_ps)
    n2 = len(bigconcatenatenaive_ps)
    r_rovef0 = 1 - (2 * manwhitscorerovedf0.statistic) / (n1 * n2)

    # ax.leg
    ax.legend(fontsize=18)
    plt.savefig('G:/neural_chapter/figures/rovedF0distribution_20062023intertrialroving.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=800)
    ax.set_xlim([0, 1])
    sns.distplot(bigconcatenatetrained_ps, label='trained roved', ax=ax, color='purple')
    sns.distplot(bigconcatenatetrained_nonps, label='trained control', ax=ax, color='magenta')
    ax.legend(fontsize=18)
    plt.title('Roved and Control F0 Distributions for the Trained Animals', fontsize=18)
    plt.xlabel(' LSTM decoder scores', fontsize=20)
    plt.xlim([0.35, 1])

    plt.savefig('G:/neural_chapter/figures/rovedF0vscontrolF0traineddistribution_20062023intertrialroving.png',
                dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=800)
    ax.set_xlim([0, 1])
    sns.distplot(bigconcatenatenaive_ps, label='naive roved', ax=ax, color='darkcyan')
    sns.distplot(bigconcatenatenaive_nonps, label='naive control', ax=ax, color='cyan')
    plt.xlim([0.35, 1])

    ax.legend(fontsize=18)
    plt.xlabel(' LSTM decoder scores', fontsize=20)
    plt.title('Roved and Control F0 Distributions for the Naive Animals', fontsize=18)

    plt.savefig('G:/neural_chapter/figures/rovedF0vscontrolF0naivedistribution_20062023intertrialroving.png', dpi=1000)
    plt.show()
    kstestcontrolf0vsrovedtrained = scipy.stats.kstest(bigconcatenatetrained_nonps, bigconcatenatetrained_ps,
                                                       alternative='two-sided')

    # do levene's test
    leveneteststat = scipy.stats.levene(bigconcatenatetrained_nonps, bigconcatenatetrained_ps)
    kstestcontrolf0vsrovednaive = scipy.stats.kstest(bigconcatenatenaive_nonps, bigconcatenatenaive_ps,
                                                     alternative='two-sided')

    # Calculating Cramér's V for effect size
    def cramers_v(n, ks_statistic):
        return np.sqrt(ks_statistic / n)

    n = len(bigconcatenatenaive_nonps) * len(bigconcatenatenaive_ps) / (
                len(bigconcatenatenaive_nonps) + len(bigconcatenatenaive_ps))
    effect_size_naive = cramers_v(n, kstestcontrolf0vsrovednaive.statistic)

    n_trained = len(bigconcatenatetrained_nonps) * len(bigconcatenatetrained_ps) / (
                len(bigconcatenatetrained_nonps) + len(bigconcatenatetrained_ps))
    effect_size_trained = cramers_v(n_trained, kstestcontrolf0vsrovedtrained.statistic)

    # run mann whitney u test
    manwhitscore_stat, manwhitescore_pvalue = mannwhitneyu(bigconcatenatetrained_nonps, bigconcatenatetrained_ps,
                                                           alternative='two-sided')
    manwhitscore_statnaive, manwhitescore_pvaluenaive = mannwhitneyu(bigconcatenatenaive_nonps, bigconcatenatenaive_ps,
                                                                     alternative='two-sided')

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
    dataframe_stats.to_csv('G:/neural_chapter/figures/stats_16112023_comparingdistributions_generalintertrialroving.csv')

    #now plot by the probe word for the naive animals
    fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
    df_above_chance = df_full_naive_pitchsplit[df_full_naive_pitchsplit['Below-chance'] == 0]
    df_below_chance = df_full_naive_pitchsplit[df_full_naive_pitchsplit['Below-chance'] == 1]

    sns.stripplot(x='ProbeWord', y='Score', data=df_above_chance, ax=ax, size=3, dodge=True, palette='Set3',
                  hue='PitchShift')
    sns.stripplot(x='ProbeWord', y='Score', data=df_below_chance, ax=ax, size=3, dodge=True, color='lightgray',
                  alpha=0.5, jitter=False, hue='PitchShift')

    sns.violinplot(x='ProbeWord', y='Score', data=df_full_naive_pitchsplit, ax=ax, palette= 'Paired', hue = 'PitchShift')
    plt.title('Naive animals'' scores over distractor word')
    plt.savefig(f'G:/neural_chapter/figures/naive_animals_overdistractor_dividedbypitchshift.png', dpi = 300)
    plt.show()

    df_kruskal = pd.DataFrame(columns=['ProbeWord', 'Kruskal_pvalue', 'less than 0.05'])
    # Perform Kruskal-Wallis test for each ProbeWord
    for probe_word in df_above_chance['ProbeWord'].unique():
        subset_data = df_above_chance[df_above_chance['ProbeWord'] == probe_word]

        result_kruskal = scipy.stats.kruskal(subset_data[subset_data['PitchShift'] == 1]['Score'],
                                 subset_data[subset_data['PitchShift'] == 0]['Score'])

        less_than_alpha = result_kruskal.pvalue < 0.05

        print(f"ProbeWord: {probe_word}, Kruskal-Wallis p-value: {result_kruskal.pvalue}")
        #append to a dataframe
        df_kruskal = df_kruskal.append({'ProbeWord': probe_word, 'Kruskal_pvalue': result_kruskal.pvalue, 'less than 0.05':less_than_alpha}, ignore_index=True)
    #export the dataframe
    df_kruskal.to_csv('G:/neural_chapter/figures/kruskal_pvalues_naive.csv')
    #run an anova to see if probe word is significant
    #first get the data into a format that can be analysed


    df_full_pitchsplit_anova = df_full_pitchsplit.copy()

    unique_probe_words = df_full_pitchsplit_anova['ProbeWord'].unique()

    df_full_pitchsplit_anova = df_full_pitchsplit_anova.reset_index(drop=True)


    df_full_pitchsplit_anova['ProbeWord'] = pd.Categorical(df_full_pitchsplit_anova['ProbeWord'],
                                                           categories=unique_probe_words, ordered=True)
    df_full_pitchsplit_anova['ProbeWord'] = df_full_pitchsplit_anova['ProbeWord'].cat.codes

    df_full_pitchsplit_anova['BrainArea'] = df_full_pitchsplit_anova['BrainArea'].astype('category')

    #cast the probe word category as an int
    df_full_pitchsplit_anova['ProbeWord'] = df_full_pitchsplit_anova['ProbeWord'].astype('int')
    df_full_pitchsplit_anova['PitchShift'] = df_full_pitchsplit_anova['PitchShift'].astype('int')
    df_full_pitchsplit_anova['Below-chance'] = df_full_pitchsplit_anova['Below-chance'].astype('int')


    df_full_pitchsplit_anova["ProbeWord"] = pd.to_numeric(df_full_pitchsplit_anova["ProbeWord"])
    df_full_pitchsplit_anova["PitchShift"] = pd.to_numeric(df_full_pitchsplit_anova["PitchShift"])
    df_full_pitchsplit_anova["Below_chance"] = pd.to_numeric(df_full_pitchsplit_anova["Below-chance"])
    df_full_pitchsplit_anova["Score"] = pd.to_numeric(df_full_pitchsplit_anova["Score"])
    #change the columns to the correct type



    #remove all rows where the score is NaN
    df_full_pitchsplit_anova = df_full_pitchsplit_anova.dropna(subset = ['Score'])
    #nest ferret as a variable ,look at the relative magnittud eo fthe coefficients for both lightgbm model and anova
    print(df_full_pitchsplit_anova.dtypes)
    #now run anova
    import statsmodels.formula.api as smf
    formula = 'Score ~ C(ProbeWord) + C(PitchShift) +C(BrainArea)+C(SingleUnit)'
    model = smf.ols(formula, data=df_full_pitchsplit_anova).fit()
    anova_table = sm.stats.anova_lm(model, typ=3)
    #get the coefficient of determination
    print(model.rsquared)
    print(anova_table)
    #combine the dataframes df_full_naive_pitchsplit and
    #add the column naive to df_full_naive_pitchsplit
    df_full_naive_pitchsplit['Naive'] = 1
    df_full_pitchsplit['Naive'] = 0
    combined_df = df_full_naive_pitchsplit.append(df_full_pitchsplit)
    #now run the lightgbm function
    runlgbmmodel_score(combined_df, optimization=False)


    #now plot by animal:
    for animal in ['F1901_Crumble', 'F1902_Eclair','F2003_Orecchiette', 'F1812_Nala']:
        df_full_naive_ps_animal = df_full_naive_pitchsplit[df_full_naive_pitchsplit['ID'].str.contains(animal)]
        if len(df_full_naive_ps_animal) == 0:
            continue
        fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
        df_above_chance = df_full_naive_ps_animal[df_full_naive_ps_animal['Below-chance'] == 0]
        df_below_chance = df_full_naive_ps_animal[df_full_naive_ps_animal['Below-chance'] == 1]

        sns.stripplot(x='ProbeWord', y='Score', data=df_above_chance, ax=ax, size=3, dodge=True, palette='Set3',
                      hue='PitchShift')
        sns.stripplot(x='ProbeWord', y='Score', data=df_below_chance, ax=ax, size=3, dodge=True, color='lightgray',
                      alpha=0.5, jitter=False, hue='PitchShift')

        sns.violinplot(x='ProbeWord', y='Score', data=df_full_naive_ps_animal, ax=ax, hue='PitchShift')
        plt.title(f'Naive scores over distractor word:{animal}')
        plt.savefig(f'G:/neural_chapter/figures/naive_animals_overdistractor_dividedbypitchshift_{animal}.png', dpi=300)

        plt.show()


    for animal in ['F1702_Zola', 'F1815_Cruella', 'F1604_Squinty', 'F1606_Windolene']:
        df_full_pitchsplit_animal = df_full_pitchsplit[df_full_pitchsplit['ID'].str.contains(animal)]
        if len(df_full_pitchsplit_animal) == 0:
            continue
        fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)

        df_above_chance = df_full_pitchsplit_animal[df_full_pitchsplit_animal['Below-chance'] == 0]
        df_below_chance = df_full_pitchsplit_animal[df_full_pitchsplit_animal['Below-chance'] == 1]

        sns.stripplot(x='ProbeWord', y='Score', data=df_above_chance, ax=ax, size=3, dodge=True, palette='Spectral',
                      hue='PitchShift')
        sns.stripplot(x='ProbeWord', y='Score', data=df_below_chance, ax=ax, size=3, dodge=True, color='lightgray',
                      alpha=0.5, jitter=False, hue='PitchShift')
        sns.violinplot(x='ProbeWord', y='Score', data=df_full_pitchsplit_animal, ax=ax, hue='PitchShift', palette='Spectral')
        plt.title(f'Trained scores over distractor word:{animal}')
        plt.savefig(f'G:/neural_chapter/figurestrained_{animal}_overdistractor_dividedbypitchshift.png', dpi=300)

        plt.show()


def plot_general_distributions(dictlist, dictlist_naive, dictlist_trained):
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

    #plot scatter data in a loop
    # for i, (data_dict, label, color) in enumerate(zip(dictlist, labels, colors)):
    #     ax.scatter(data_dict['mu_list']['nonpitchshift']['female_talker'],data_dict['mu_list']['pitchshift']['female_talker'], marker='P',
    #                facecolors =color, edgecolors = color, alpha=0.5)
    #     ax.scatter(data_dict['su_list']['nonpitchshift']['female_talker'],data_dict['su_list']['pitchshift']['female_talker'], marker='P', color=color, alpha=0.5)


    if bigconcatenatenaive_nonps.size > bigconcatenatenaive_ps.size:
        len(bigconcatenatenaive_ps)
        bigconcatenatenaive_nonps = bigconcatenatenaive_nonps[:bigconcatenatenaive_ps.size]
    elif bigconcatenatenaive_nonps.size < bigconcatenatenaive_ps.size:
        bigconcatenatenaive_ps = bigconcatenatenaive_ps[:bigconcatenatenaive_nonps.size]

    if bigconcatenatetrained_nonps.size > bigconcatenatetrained_ps.size:
        bigconcatenatetrained_nonps = bigconcatenatetrained_nonps[:bigconcatenatetrained_ps.size]
    elif bigconcatenatetrained_nonps.size < bigconcatenatetrained_ps.size:
        bigconcatenatetrained_ps = bigconcatenatetrained_ps[:bigconcatenatetrained_nonps.size]

    fig, ax = plt.subplots(1, figsize=(9, 9), dpi=300)
    ax.scatter(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, marker='P', color='purple', alpha=0.8, label='trained', s=0.1)
    plt.title('trained animals, number of points: ' + str(len(bigconcatenatetrained_ps)))
    plt.show()
    unique_scores = np.unique(bigconcatenatetrained_ps)
    len(unique_scores)


    fig, ax = plt.subplots(1, figsize=(9,9), dpi=300)

    ax.scatter(bigconcatenatenaive_nonps, bigconcatenatenaive_ps, marker='P', color='darkcyan', alpha=0.5, label='naive')
    ax.scatter(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, marker='P', color='purple', alpha=0.5, label='trained')
    x = np.linspace(0.4, 1, 101)
    ax.plot(x, x, color='black', linestyle = '--')  # identity line

    slope, intercept, r_value, pv, se = stats.linregress(bigconcatenatetrained_nonps, bigconcatenatetrained_ps)

    sns.regplot(x=bigconcatenatetrained_nonps, y=bigconcatenatetrained_ps, scatter=False, color='purple',
                label=' $y=%3.7s*x+%3.7s$' % (slope, intercept), ax=ax, line_kws={'label': ' $y=%3.7s*x+%3.7s$' % (slope, intercept)})
    slope, intercept, r_value, pv, se = stats.linregress(bigconcatenatenaive_nonps, bigconcatenatenaive_ps)

    sns.regplot(x=bigconcatenatenaive_nonps, y=bigconcatenatenaive_ps, scatter=False, color='darkcyan', label=' $y=%3.7s*x+%3.7s$' % (slope, intercept),
                ax=ax, line_kws={'label': '$y=%3.7s*x+%3.7s$' % (slope, intercept)})

    ax.set_ylabel('LSTM decoding score, F0 roved', fontsize=18)
    ax.set_xlabel('LSTM decoding score, F0 control', fontsize=18)
    ax.set_title('LSTM decoder scores for' + ' F0 control vs. roved,\n ' + ' trained and naive animals', fontsize=20)
    plt.legend( fontsize=12, ncol=2)
    fig.tight_layout()
    plt.show()

    #histogram distribution of the trained and naive animals
    fig, ax = plt.subplots(1, figsize=(8, 8))
    #relativescoretrained = abs(bigconcatenatetrained_nonps - bigconcatenatetrained_ps)/ bigconcatenatetrained_ps

    relativescoretrained = [bigconcatenatetrained_nonps - bigconcatenatetrained_ps for bigconcatenatetrained_nonps, bigconcatenatetrained_ps in zip(bigconcatenatetrained_nonps, bigconcatenatetrained_ps)]
    relativescorenaive = [bigconcatenatenaive_nonps - bigconcatenatenaive_ps for bigconcatenatenaive_nonps, bigconcatenatenaive_ps in zip(bigconcatenatenaive_ps, bigconcatenatenaive_nonps)]
    relativescoretrainedfrac = [relativescoretrained / (bigconcatenatetrained_nonps + bigconcatenatenaive_nonps) for relativescoretrained, bigconcatenatetrained_nonps, bigconcatenatenaive_nonps in zip(relativescoretrained, bigconcatenatetrained_nonps, bigconcatenatenaive_nonps)]
    relativescorenaivefrac = [relativescorenaive / (bigconcatenatenaive_nonps + bigconcatenatetrained_nonps) for relativescorenaive, bigconcatenatenaive_nonps, bigconcatenatetrained_nonps in zip(relativescorenaive, bigconcatenatenaive_nonps, bigconcatenatetrained_nonps)]



    sns.distplot(relativescoretrained, bins = 20, label='trained',ax=ax, color='purple')
    sns.distplot(relativescorenaive, bins = 20, label='naive', ax=ax, color='darkcyan')
    plt.axvline(x=0, color='black')
    #man whiteney test score


    manwhitscore = mannwhitneyu(relativescoretrained, relativescorenaive, alternative = 'greater')
    sample1 = np.random.choice(relativescoretrained, size=10000, replace=True)

    # Generate a random sample of size 100 from data2 with replacement
    sample2 = np.random.choice(relativescorenaive, size=10000, replace=True)

    # Perform a t-test on the samples
    t_stat, p_value = stats.ttest_ind(sample1, sample2, alternative='greater')

    # Print the t-statistic and p-value
    print(t_stat, p_value)
    plt.title('Control - roved F0 \n LSTM decoder scores between trained and naive animals', fontsize = 18)
    plt.xlabel('Control - roved F0 \n LSTM decoder scores', fontsize = 20)
    plt.ylabel('Density', fontsize = 20)
    #ax.legend()
    #plt.savfig('G:/neural_chapter/figures/diffF0distribution_20062023.png', dpi=1000)
    plt.show()

    #plot sns histogram of the relative score and with the displot function overlaid
    fig, ax = plt.subplots(1, figsize=(8, 8))
    # ax = sns.displot(relativescoretrainedfrac, bins = 20, label='trained',ax=ax, color='purple')
    sns.histplot(relativescoretrainedfrac, bins=20, label='trained', color='purple', kde = True)
    sns.histplot(relativescorenaivefrac, bins = 20, label='naive', color='darkcyan', kde = True)

    #plt.savfig('G:/neural_chapter/figures/diffF0distribution_relfrac_histplotwithkde_20062023.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax = sns.distplot(relativescoretrainedfrac, bins = 20, label='trained',ax=ax, color='purple')
    x = ax.lines[-1].get_xdata()  # Get the x data of the distribution
    y = ax.lines[-1].get_ydata()  # Get the y data of the distribution
    maxidtrained_idx = np.argmax(y)
    x_coord_trained = x[maxidtrained_idx]
    ax2 = sns.distplot(relativescorenaivefrac, bins = 20, label='naive', ax=ax, color='darkcyan')

    x2 = ax2.lines[-1].get_xdata()  # Get the x data of the distribution
    y2 = ax2.lines[-1].get_ydata()  # Get the y data of the distribution
    maxidnaive_idx = np.argmax(y2)  # The id of the peak (maximum of y data)

    x_coord_naive = x2[maxidnaive_idx]
    plt.axvline(x=0, color='black')
    kstestnaive = scipy.stats.kstest(relativescorenaivefrac,  stats.norm.cdf)
    leveneteststat = scipy.stats.levene(relativescorenaivefrac, relativescoretrainedfrac)
    manwhitscorefrac = mannwhitneyu(relativescorenaivefrac, relativescoretrainedfrac, alternative = 'less')
    #caclulate medians of distribution

    sample1_trained = np.random.choice(relativescoretrainedfrac, size=10000, replace=True)

    # Generate a random sample of size 100 from data2 with replacement
    sample2_naive = np.random.choice(relativescorenaive, size=10000, replace=True)

    # Perform a t-test on the samples
    t_statfrac, p_valuefrac = stats.ttest_ind(sample2_naive, sample1_trained, alternative='less')

    # Print the t-statistic and p-value
    print(t_statfrac, p_valuefrac)
    plt.title('Control - roved F0 \n LSTM decoder scores between trained and naive animals', fontsize = 18)
    plt.xlabel('Control - roved F0 \n LSTM decoder scores divided by control F0', fontsize = 20)
    plt.ylabel('Density', fontsize = 20)
    #ax.legend(fontsize = 18)


    #plt.savfig('G:/neural_chapter/figures/diffF0distribution_frac_20062023wlegendintertrialroving.png', dpi=1000)
    plt.show()



    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=800)
    ax.set_xlim([0,1])

    sns.distplot(bigconcatenatetrained_nonps,  label='trained',ax=ax, color='purple')
    sns.distplot(bigconcatenatenaive_nonps, label='naive', ax=ax, color='darkcyan')
    #plt.axvline(x=0, color='black')
    #man whiteney test score
    plt.title('Control F0 LSTM decoder scores between  \n trained and naive animals', fontsize = 18)
    plt.xlabel('Control F0 LSTM decoder scores', fontsize = 20)

    plt.ylabel('Density', fontsize = 20)
    manwhitscorecontrolf0 = mannwhitneyu(bigconcatenatetrained_nonps, bigconcatenatenaive_nonps, alternative = 'greater')

    #ax.legend()
    #plt.savfig('G:/neural_chapter/figures/controlF0distribution20062023intertrialroving.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=800)
    ax.set_xlim([0,1])
    sns.distplot(bigconcatenatetrained_ps,  label='trained',ax=ax, color='purple')
    sns.distplot(bigconcatenatenaive_ps, label='naive', ax=ax, color='darkcyan')
    #man whiteney test score
    #manwhitscore = mannwhitneyu(relativescoretrained, relativescorenaive, alternative = 'greater')
    plt.title('Roved F0 LSTM decoder scores between  \n trained and naive animals', fontsize = 18)
    plt.xlabel('Roved F0 LSTM decoder scores', fontsize = 20)
    plt.ylabel('Density', fontsize = 20)
    manwhitscorerovedf0 = mannwhitneyu(bigconcatenatetrained_ps, bigconcatenatenaive_ps, alternative = 'greater')

    ax.legend(fontsize=18)
    #plt.savfig('G:/neural_chapter/figures/rovedF0distribution_20062023intertrialroving.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi = 800)
    ax.set_xlim([0,1])
    sns.distplot(bigconcatenatetrained_ps,  label='trained roved',ax=ax, color='purple')
    sns.distplot(bigconcatenatetrained_nonps,  label='trained control',ax=ax, color='magenta')
    ax.legend(fontsize=18)
    plt.title('Roved and Control F0 Distributions for the Trained Animals', fontsize = 18)
    plt.xlabel(' LSTM decoder scores', fontsize = 20)

    #plt.savfig('G:/neural_chapter/figures/rovedF0vscontrolF0traineddistribution_20062023intertrialroving.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi = 800)
    ax.set_xlim([0,1])
    sns.distplot(bigconcatenatenaive_ps,  label='naive roved',ax=ax, color='darkcyan')
    sns.distplot(bigconcatenatenaive_nonps,  label='naive control',ax=ax, color='cyan')
    ax.legend(fontsize=18)
    plt.xlabel(' LSTM decoder scores', fontsize = 20)
    plt.title('Roved and Control F0 Distributions for the Naive Animals', fontsize = 18)

    plt.savfig('G:/neural_chapter/figures/rovedF0vscontrolF0naivedistribution_20062023intertrialroving.png', dpi=1000)
    plt.show()
    kstestcontrolf0vsrovedtrained = scipy.stats.kstest(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, alternative = 'two-sided')

    kstestcontrolf0vsrovednaive = scipy.stats.kstest(bigconcatenatenaive_nonps, bigconcatenatenaive_ps, alternative='two-sided')

    naivearray=np.concatenate((np.zeros((len(bigconcatenatetrained_nonps)+len(bigconcatenatetrained_ps),1)), np.ones((len(bigconcatenatenaive_nonps)+len(bigconcatenatenaive_ps),1))))
    trainedarray=np.concatenate((np.ones((len(bigconcatenatetrained_nonps)+len(bigconcatenatetrained_ps),1)), np.zeros((len(bigconcatenatenaive_nonps)+len(bigconcatenatenaive_ps),1))))
    controlF0array=np.concatenate((np.ones((len(bigconcatenatetrained_nonps),1)), np.zeros((len(bigconcatenatetrained_ps),1)), np.ones((len(bigconcatenatenaive_nonps),1)), np.zeros((len(bigconcatenatenaive_ps),1))))
    rovedF0array = np.concatenate((np.zeros((len(bigconcatenatetrained_nonps),1)), np.ones((len(bigconcatenatetrained_ps),1)), np.zeros((len(bigconcatenatenaive_nonps),1)), np.ones((len(bigconcatenatenaive_ps),1))))
    scores = np.concatenate((bigconcatenatetrained_nonps, bigconcatenatetrained_ps, bigconcatenatenaive_nonps, bigconcatenatenaive_ps))





if __name__ == '__main__':
    main()