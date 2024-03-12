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
                              multiunitlist=[0,1,2,3], noiselist=[], stream = 'BB_2', fullid = 'F1901_Crumble'):
    singleunitlist = [x - 1 for x in singleunitlist]
    multiunitlist = [x - 1 for x in multiunitlist]
    noiselist = [x - 1 for x in noiselist]
    original_cluster_list = np.empty([0])


    su_highf0_list_male = np.empty([0])
    mu_highf0_list_male = np.empty([0])

    mu_lowf0_list_male = np.empty([0])
    su_lowf0_list_male = np.empty([0])

    mu_lowf0_list_female = np.empty([0])
    su_lowf0_list_female = np.empty([0])
    su_highf0_list_female = np.empty([0])
    mu_highf0_list_female = np.empty([0])


    su_control_list_female = np.empty([0])
    mu_control_list_female = np.empty([0])

    su_control_list_male = np.empty([0])
    mu_control_list_male = np.empty([0])



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
                try:
                    scores = np.load(
                        saveDir + r'scores_2022_' + ferretname +'_'+  stringprobewordindex + '_' + ferretname + '_probe_bs.npy',
                        allow_pickle=True)[()]
                except:
                    print('file not found: ' + saveDir + r'scores_2022_' + ferretname + stringprobewordindex + '_' + ferretname + '_probe_bs.npy')
                    continue
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
                saveDir  + r'scores_2022_' + ferretname  +'_'+ stringprobewordindex + '_' + ferretname + '_probe_bs.npy',
                allow_pickle=True)[()]
        except:
            print('file not found: ' + saveDir + r'scores_2022_' + ferretname + '_'+ stringprobewordindex + '_' + ferretname + '_probe_bs.npy')
            continue

        for talker in [1]:
            comparisons = [comp for comp in scores[f'talker{talker}']]

            for comp in comparisons:
                for cond in ['nopitchshiftvspitchshift']:
                    for i, clus in enumerate(scores[f'talker{talker}'][comp][cond]['cluster_id']):
                        #check if clus is greater than 100
                        if 200 > clus >= 100:
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
        if fullid == 'F2003_Orecchiette' or fullid == 'F1901_Crumble' or fullid == 'F1812_Nala' or fullid == 'F1902_Eclair':
            high_units = pd.read_csv(f'G:/neural_chapter/figures/unit_ids_naive_topgenindex_{fullid}.csv')
        else:
            high_units = pd.read_csv(f'G:/neural_chapter/figures/unit_ids_trained_topgenindex_{fullid}.csv')


        high_units = high_units[(high_units['rec_name'] == recname) & (high_units['stream'] == stream_id)]
        #continue if high_units is empty,
        if high_units.empty:
            continue


        clust_ids = high_units['ID'].to_list()
        brain_area = high_units['BrainArea'].to_list()
        #remove single and multi units that are NOT in clust_ids
        singleunitlist_copy = [x for x in singleunitlist_copy if x in clust_ids]
        multiunitlist_copy = [x for x in multiunitlist_copy if x in clust_ids]


        for talker in [1]:
            comparisons = [comp for comp in scores[f'talker{talker}']]
            for comp in comparisons:
                for cond in ['nopitchshiftvspitchshift']:
                    for i, clus in enumerate(scores[f'talker{talker}'][comp][cond]['cluster_id']):
                        print(i, clus)
                        if clus in singleunitlist_copy:
                            if talker == 1:
                                if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:
                                    su_control_list_female = np.append(su_control_list_female,
                                                                         scores[f'talker{talker}'][comp][cond][
                                                                             'lstm_balanced_avg'][i])
                                    su_highf0_list_female = np.append(su_highf0_list_female,
                                                                         scores[f'talker{talker}'][comp][cond][
                                                                             'high_pitch_bal_ac'][i])
                                    su_lowf0_list_female =  np.append(su_lowf0_list_female,
                                                                         scores[f'talker{talker}'][comp][cond][
                                                                             'low_pitch_bal_ac'][i])

                            elif talker == 2:
                                if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:
                                    su_control_list_male = np.append(su_control_list_male,
                                                                       scores[f'talker{talker}'][comp][cond][
                                                                           'lstm_balanced_avg'][i])
                                    su_highf0_list_male = np.append(su_highf0_list_male,
                                                                      scores[f'talker{talker}'][comp][cond][
                                                                          'high_pitch_bal_ac'][i])
                                    su_lowf0_list_male = np.append(su_lowf0_list_male,
                                                                     scores[f'talker{talker}'][comp][cond][
                                                                         'low_pitch_bal_ac'][i])


                        elif clus in multiunitlist_copy:
                            if talker == 1:
                                if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:
                                    mu_control_list_female = np.append(mu_control_list_female,
                                                                       scores[f'talker{talker}'][comp][cond][
                                                                           'lstm_balanced_avg'][i])
                                    mu_highf0_list_female = np.append(mu_highf0_list_female,
                                                                      scores[f'talker{talker}'][comp][cond][
                                                                          'high_pitch_bal_ac'][i])
                                    mu_lowf0_list_female = np.append(mu_lowf0_list_female,
                                                                     scores[f'talker{talker}'][comp][cond][
                                                                         'low_pitch_bal_ac'][i])

                            elif talker == 2:
                                if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:
                                    mu_control_list_male = np.append(mu_control_list_male,
                                                                       scores[f'talker{talker}'][comp][cond][
                                                                           'lstm_balanced_avg'][i])
                                    mu_highf0_list_male = np.append(mu_highf0_list_male,
                                                                      scores[f'talker{talker}'][comp][cond][
                                                                          'high_pitch_bal_ac'][i])
                                    mu_lowf0_list_male = np.append(mu_lowf0_list_male,
                                                                     scores[f'talker{talker}'][comp][cond][
                                                                         'low_pitch_bal_ac'][i])





                        elif clus in noiselist:
                            pass

    # su_highf0_list_male = np.empty([0])
    # mu_highf0_list_male = np.empty([0])
    #
    # mu_lowf0_list_male = np.empty([0])
    # su_lowf0_list_male = np.empty([0])
    #
    # mu_lowf0_list_female = np.empty([0])
    # su_lowf0_list_female = np.empty([0])
    # su_highf0_list_female = np.empty([0])
    # mu_highf0_list_female = np.empty([0])
    #
    # su_control_list_female = np.empty([0])
    # mu_control_list_female = np.empty([0])
    #
    # su_control_list_male = np.empty([0])
    # mu_control_list_male = np.empty([0])

    dictofsortedscores = {'su_list': {'highf0': {'female_talker': {},
                                                     'male_talker': {}},
                                      'controlf0': {'female_talker': {},
                                                        'male_talker': {}},
                                         'lowf0': {'female_talker': {},
                                                        'male_talker': {}}
                                      },
                          'mu_list': {'highf0': {'female_talker': {},
                                                     'male_talker': {}},
                                      'controlf0': {'female_talker': {},
                                                        'male_talker': {}},
                                         'lowf0': {'female_talker': {},
                                                        'male_talker': {}}
                                      },}

    dictofsortedscores['su_list']['highf0']['female_talker'] = su_highf0_list_female
    dictofsortedscores['su_list']['highf0']['male_talker'] = su_highf0_list_male
    dictofsortedscores['su_list']['controlf0']['female_talker'] = su_control_list_female
    dictofsortedscores['su_list']['controlf0']['male_talker'] = su_control_list_male
    dictofsortedscores['su_list']['lowf0']['female_talker'] = su_lowf0_list_female
    dictofsortedscores['su_list']['lowf0']['male_talker'] = su_lowf0_list_male

    dictofsortedscores['mu_list']['highf0']['female_talker'] = mu_highf0_list_female
    dictofsortedscores['mu_list']['highf0']['male_talker'] = mu_highf0_list_male
    dictofsortedscores['mu_list']['controlf0']['female_talker'] = mu_control_list_female
    dictofsortedscores['mu_list']['controlf0']['male_talker'] = mu_control_list_male
    dictofsortedscores['mu_list']['lowf0']['female_talker'] = mu_lowf0_list_female
    dictofsortedscores['mu_list']['lowf0']['male_talker'] = mu_lowf0_list_male





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
    :param path: path to the classified report
    :return: classified report
    '''
    #join the path to the report
    if 'F2003_Orecchiette' in path:
        report_path = os.path.join(path, 'quality metrics.csv')
        #combine the paths

        report = pd.read_csv(report_path)
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
    probewordlist = [(2, 2), (5, 6), (42, 49), (32, 38), (20, 22)]
    probewordlist_l74 = [(2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
                             (14, 14)]
    animal_list = [ 'F1901_Crumble', 'F1604_Squinty', 'F1606_Windolene', 'F1702_Zola','F1815_Cruella', 'F1902_Eclair', 'F1812_Nala']


    #load the report for each animal in animal-list
    report = {}
    singleunitlist = {}
    multiunitlist = {}
    noiselist = {}
    path_list = {}

    for animal in animal_list:
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
            stream_name = str(stream_name).split('\\')[-1]
            #maybe just modify this to include the recname
            # stream_name = str(stream_name)[-4:]

            #check if stream name exists
            # if stream_name in report[animal].keys():
            #     stream_name = path.parent.absolute()
            #     stream_name = stream_name.parent.absolute()
            #     #find myriad number
            #     stream_name = str(stream_name)[-6:]
            #load the report for that stream
            # try:
            report[animal][stream_name], singleunitlist[animal][stream_name], multiunitlist[animal][stream_name], noiselist[animal][stream_name] = load_classified_report(f'{path}')
            # except:
            #     print('no report for this stream:' + str(path))
            #     pass
    # now create a dictionary of dictionaries, where the first key is the animal name, and the second key is the stream name
    #the value is are the decoding scores for each cluster


    dictoutput = {}
    dictoutput_trained = []
    dictoutput_naive = []
    dictoutput_all = []
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
                    streamtext = 'BB_2'
                elif 'BB_3' in stream:
                    streamtext = 'BB_3'
                elif 'BB_4' in stream:
                    streamtext = 'BB_4'
                elif 'BB_5' in stream:
                    streamtext = 'BB_5'
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
                animal_text = animal_text.lower()

            if animal == 'F1604_Squinty':
                # try:
                dictoutput_instance = scatterplot_and_visualise(probewordlist_l74,
                                                                saveDir=f'G:/testonroveresults/results_testonrove_inter_28102023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream = stream, fullid = animal)
                dictoutput_all.append(dictoutput_instance)

            elif animal == 'F1606_Windolene':

                # try:F:\results_testonrove_inter_28102023
                dictoutput_instance = scatterplot_and_visualise(probewordlist_l74,
                                                                saveDir=f'G:/testonroveresults/results_testonrove_inter_28102023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream = stream, fullid = animal)
                dictoutput_all.append(dictoutput_instance)
                # except:
                #     #print the exception
                #     print(f'no scores for this stream:{stream}, and {animal}')
                #     pass
            elif animal == 'F1815_Cruella' or animal == 'F1902_Eclair' or animal =='F1702_Zola':
                # try:
                dictoutput_instance = scatterplot_and_visualise(probewordlist, saveDir= f'G:/testonroveresults/results_testonrove_inter_28102023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal)
                dictoutput_all.append(dictoutput_instance)
                # except:
                #     #print the exception
                #     print(f'no scores for this stream:{stream}, and {animal}')
                #     pass
            elif animal == 'F2003_Orecchiette':
                # try:
                dictoutput_instance = scatterplot_and_visualise(probewordlist,
                                                                saveDir=f'G:/testonroveresults/results_testonrove_inter_28102023/{animal}/{rec_name_unique}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream=stream,
                                                                fullid=animal)
                dictoutput_all.append(dictoutput_instance)

            else:
                # try:
                dictoutput_instance = scatterplot_and_visualise(probewordlist, saveDir= f'G:/testonroveresults/results_testonrove_inter_28102023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal)
                dictoutput_all.append(dictoutput_instance)
                # except:
                #     #print the exception
                #     print(f'no scores for this stream:{stream}, and {animal}')
                #     pass
            try:
                if animal == 'F1604_Squinty' or animal == 'F1606_Windolene' or animal == 'F1702_Zola' or animal == 'F1815_Cruella':
                    print('trained animal'+ animal)
                    dictoutput_trained.append(dictoutput_instance)
                else:
                    print('naive animal:'+ animal)
                    dictoutput_naive.append(dictoutput_instance)
            except:
                print('no scores for this stream')
                pass


    labels = [ 'F1901_Crumble', 'F1604_Squinty', 'F1606_Windolene', 'F1702_Zola','F1815_Cruella', 'F1902_Eclair', 'F1812_Nala']

    colors = ['purple', 'magenta', 'darkturquoise', 'olivedrab', 'steelblue', 'darkcyan', 'darkorange']

    generate_plots(dictoutput_all, dictoutput_trained, dictoutput_naive, labels, colors)

    return


def generate_plots(dictlist, dictlist_trained, dictlist_naive, labels, colors):
    # labels = ['cruella', 'zola', 'nala', 'crumble', 'eclair', 'ore']
    # colors = ['purple', 'magenta', 'darkturquoise', 'olivedrab', 'steelblue', 'darkcyan']


    fig, ax = plt.subplots(1, figsize=(5, 8))
    emptydict = {}
    count = 0
    for dictoutput in dictlist:
        for sutype in dictoutput.keys():
            for pitchshiftornot in dictoutput[sutype].keys():
                for talker in dictoutput[sutype][pitchshiftornot].keys():
                    for item in dictoutput[sutype][pitchshiftornot][talker]:
                        if count == 0 or count ==1:
                            emptydict['trained'] = emptydict.get('trained', []) + [1]
                        else:
                            emptydict['trained'] = emptydict.get('trained', []) + [0]
                        emptydict['ferret']= emptydict.get('ferret', []) + [count]
                        emptydict['score'] = emptydict.get('score', []) + [item]
                        if talker == 'female_talker':
                            emptydict['male_talker'] = emptydict.get('male_talker', []) + [0]
                        else:
                            emptydict['male_talker'] = emptydict.get('male_talker', []) + [1]
                        if pitchshiftornot == 'pitchshift':
                            emptydict['pitchshift'] = emptydict.get('pitchshift', []) + [1]
                        else:
                            emptydict['pitchshift'] = emptydict.get('pitchshift', []) + [0]
                        if sutype == 'su_list':
                            emptydict['su'] = emptydict.get('su', []) + [1]
                        else:
                            emptydict['su'] = emptydict.get('su', []) + [0]
        count += 1
    for keys in emptydict.keys():
        emptydict[keys] = np.asarray(emptydict[keys])


    # for dictoutput in dictlist:
    #     for key in dictoutput.keys():
    #         for key3 in dictoutput[key]['highf0'].keys():
    #             if len(dictoutput[key]['controlf0'][key3]) < len(
    #                     dictoutput[key]['pitchshift'][key3]):
    #                 dictoutput[key]['pitchshift'][key3] = \
    #                     dictoutput[key]['pitchshift'][key3][
    #                     :len(dictoutput[key]['nonpitchshift'][key3])]
    #             elif len(dictoutput[key]['nonpitchshift'][key3]) > len(
    #                     dictoutput[key]['pitchshift'][key3]):
    #                 dictoutput[key]['nonpitchshift'][key3] = \
    #                     dictoutput[key]['nonpitchshift'][key3][
    #                     :len(dictoutput[key]['pitchshift'][key3])]

    bigconcatenatetrained_control_f0 = np.empty(0)
    bigconcatenatetrained_high_f0 = np.empty(0)
    bigconcatenatetrained_low_f0 = np.empty(0)

    for dictouput in dictlist_trained:
        for key in dictouput.keys():
            for key3 in dictouput[key]['highf0'].keys():
                bigconcatenatetrained_control_f0 = np.concatenate(
                    (bigconcatenatetrained_control_f0, dictouput[key]['controlf0'][key3]))
                bigconcatenatetrained_high_f0 = np.concatenate(
                    (bigconcatenatetrained_high_f0, dictouput[key]['highf0'][key3]))
                bigconcatenatetrained_low_f0 = np.concatenate(
                    (bigconcatenatetrained_low_f0, dictouput[key]['lowf0'][key3]))

    bigconcatenatenaive_control_f0 = np.empty(0)
    bigconcatenatenaive_high_f0 = np.empty(0)
    bigconcatenatenaive_low_f0 = np.empty(0)

    for dictouput in dictlist_naive:
        for key in dictouput.keys():
            for key3 in dictouput[key]['highf0'].keys():
                bigconcatenatenaive_control_f0 = np.concatenate(
                    (bigconcatenatenaive_control_f0, dictouput[key]['controlf0'][key3]))
                bigconcatenatenaive_high_f0 = np.concatenate(
                    (bigconcatenatenaive_high_f0, dictouput[key]['highf0'][key3]))
                bigconcatenatenaive_low_f0 = np.concatenate(
                    (bigconcatenatenaive_low_f0, dictouput[key]['lowf0'][key3]))

    fig, ax = plt.subplots(1, figsize=(5, 8))
    #plot with error bars


    ax.bar(0, np.mean(bigconcatenatetrained_high_f0), color = 'red', label = 'high f0')
    std_dev = np.std(bigconcatenatetrained_high_f0)
    ax.errorbar(0, np.mean(bigconcatenatetrained_high_f0), yerr = std_dev, color = 'black')

    ax.bar(1, np.mean(bigconcatenatetrained_control_f0), color = 'blue', label = 'control')
    std_dev = np.std(bigconcatenatetrained_control_f0)
    ax.errorbar(1, np.mean(bigconcatenatetrained_control_f0), yerr = std_dev, color = 'black')

    ax.bar(2, np.mean(bigconcatenatetrained_low_f0), color = 'green', label = 'low f0')
    std_dev = np.std(bigconcatenatetrained_low_f0)
    ax.errorbar(2, np.mean(bigconcatenatetrained_low_f0), yerr = std_dev, color = 'black')


    ax.bar(3, np.mean(bigconcatenatenaive_high_f0), color = 'red')
    std_dev = np.std(bigconcatenatenaive_high_f0)
    ax.errorbar(3, np.mean(bigconcatenatenaive_high_f0), yerr = std_dev, color = 'black')
    ax.bar(4, np.mean(bigconcatenatenaive_control_f0), color = 'blue')
    std_dev = np.std(bigconcatenatenaive_control_f0)
    ax.errorbar(4, np.mean(bigconcatenatenaive_control_f0), yerr = std_dev, color = 'black')
    ax.bar(5, np.mean(bigconcatenatenaive_low_f0), color = 'green')
    std_dev = np.std(bigconcatenatenaive_low_f0)
    ax.errorbar(5, np.mean(bigconcatenatenaive_low_f0), yerr = std_dev, color = 'black')

    ax.set_xticks([0,1,2,3,4,5])
    ax.set_xticklabels(['high f0', 'control', 'low f0',  'high f0 - naive', 'control - naive','low f0 - naive'], rotation = 45)
    ax.set_ylabel('lstm decoding score')
    ax.set_title('mean lstm decoding score for trained and naive animals')
    ax.legend()
    plt.savefig('G:/testonroveresults/results_testonrove_inter_28102023/mean_lstm_decoding_score_for_trained_and_naive_animals.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    #combine into a dataframe
    bigconcatenatetrained_control_f0 = pd.DataFrame(bigconcatenatetrained_control_f0, columns = ['score'])
    bigconcatenatetrained_control_f0['trained'] = 1
    bigconcatenatetrained_control_f0['pitch'] = 1
    bigconcatenatetrained_control_f0['ferret'] = 1
    bigconcatenatetrained_control_f0['su'] = 1

    bigconcatenatetrained_high_f0 = pd.DataFrame(bigconcatenatetrained_high_f0, columns = ['score'])
    bigconcatenatetrained_high_f0['trained'] = 1
    bigconcatenatetrained_high_f0['pitch'] = 2
    bigconcatenatetrained_high_f0['ferret'] = 1
    bigconcatenatetrained_high_f0['su'] = 1

    bigconcatenatetrained_low_f0 = pd.DataFrame(bigconcatenatetrained_low_f0, columns = ['score'])
    bigconcatenatetrained_low_f0['trained'] = 1
    bigconcatenatetrained_low_f0['pitch'] = 0
    bigconcatenatetrained_low_f0['ferret'] = 1
    bigconcatenatetrained_low_f0['su'] = 1

    bigconcatenatenaive_control_f0 = pd.DataFrame(bigconcatenatenaive_control_f0, columns = ['score'])
    bigconcatenatenaive_control_f0['trained'] = 0
    bigconcatenatenaive_control_f0['pitch'] = 1
    bigconcatenatenaive_control_f0['ferret'] = 1
    bigconcatenatenaive_control_f0['su'] = 1

    bigconcatenatenaive_high_f0 = pd.DataFrame(bigconcatenatenaive_high_f0, columns = ['score'])
    bigconcatenatenaive_high_f0['trained'] = 0
    bigconcatenatenaive_high_f0['pitch'] = 2
    bigconcatenatenaive_high_f0['ferret'] = 1
    bigconcatenatenaive_high_f0['su'] = 1

    bigconcatenatenaive_low_f0 = pd.DataFrame(bigconcatenatenaive_low_f0, columns = ['score'])
    bigconcatenatenaive_low_f0['trained'] = 0
    bigconcatenatenaive_low_f0['pitch'] = 0
    bigconcatenatenaive_low_f0['ferret'] = 1
    bigconcatenatenaive_low_f0['su'] = 1

    big_df = pd.concat([bigconcatenatetrained_control_f0, bigconcatenatetrained_high_f0, bigconcatenatetrained_low_f0, bigconcatenatenaive_control_f0, bigconcatenatenaive_high_f0, bigconcatenatenaive_low_f0])

    #make a violin plot
    fig, ax = plt.subplots(1, figsize=(5, 8))
    #plot with error bars
    ax = sns.violinplot(x="pitch", y="score", hue="trained", data=big_df, palette = 'Set2')
    # ax.set_xticklabels(['low f0', 'control', 'high f0', 'low f0', 'control', 'high f0'])
    ax.set_ylabel('lstm decoding score')
    ax.set_title('lstm decoding score for trained and naive animals')
    #get the legend labels
    handles, labels = ax.get_legend_handles_labels()
    labels = ['naive', 'trained']
    ax.legend(handles, labels)
    ax.set_xticklabels(['144 Hz', '191 Hz (control)', '251 Hz'])
    ax.set_xlabel('pitch control F0 data tested on')

    plt.savefig('G:/testonroveresults/results_testonrove_inter_28102023/VIOLIN_lstm_decoding_score_for_trained_and_naive_animals.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

    #plot as a scatter plot, with decoding scores for 144 hz on the x axis, and 191 hz on the y axis
    decoding_scores_144_trained = big_df[(big_df['pitch'] == 0) & (big_df['trained'] == 1)]
    decoding_scores_191_trained = big_df[(big_df['pitch'] == 1) & (big_df['trained'] == 1)]
    decoding_scores_251_trained = big_df[(big_df['pitch'] == 2) & (big_df['trained'] == 1)]

    decoding_scores_144_naive = big_df[(big_df['pitch'] == 0) & (big_df['trained'] == 0)]
    decoding_scores_191_naive = big_df[(big_df['pitch'] == 1) & (big_df['trained'] == 0)]
    decoding_scores_251_naive = big_df[(big_df['pitch'] == 2) & (big_df['trained'] == 0)]

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.scatter(decoding_scores_144_trained['score'], decoding_scores_191_trained['score'], color = 'purple', label = 'trained', s = 10)
    ax.scatter(decoding_scores_144_naive['score'], decoding_scores_191_naive['score'], color = 'cyan', label = 'naive', s= 10)
    #put a lingress line
    slope, intercept, r_value, p_value, std_err = stats.linregress(decoding_scores_144_trained['score'], decoding_scores_191_trained['score'])
    slope_naive, intercept_naive, r_value_naive, p_value_naive, std_err_naive = stats.linregress(decoding_scores_144_naive['score'], decoding_scores_191_naive['score'])
    ax.set_xlabel('144 Hz')
    ax.set_xlim(0, 1)
    ax.set_ylabel('191 Hz')
    ax.set_ylim(0, 1)
    ax.set_title('lstm decoding score for trained and naive animals')
    ax.legend()
    plt.savefig('G:/testonroveresults/results_testonrove_inter_28102023/scatter_lstm_decoding_score_for_trained_and_naive_animals_144versus_control.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.scatter(decoding_scores_251_trained['score'], decoding_scores_191_trained['score'], color = 'purple', label = 'trained', s = 10)
    ax.scatter(decoding_scores_251_naive['score'], decoding_scores_191_naive['score'], color = 'cyan', label = 'naive', s = 10)
    slope, intercept, r_value, p_value, std_err = stats.linregress(decoding_scores_251_trained['score'], decoding_scores_191_trained['score'])
    slope_naive, intercept_naive, r_value_naive, p_value_naive, std_err_naive = stats.linregress(decoding_scores_251_naive['score'], decoding_scores_191_naive['score'])
    ax.set_ylabel('191 Hz')
    ax.set_xlim(0,1)
    ax.set_xlabel('251 Hz')
    ax.set_ylim(0,1)
    ax.set_title('lstm decoding score for trained and naive animals')
    ax.legend()
    plt.savefig('G:/testonroveresults/results_testonrove_inter_28102023/scatter_lstm_decoding_score_for_trained_and_naive_animals_251versus_control.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    #now only repeat with the high performing units
    for ferret_id_fancy in ['F1702_Zola', 'F1815_Cruella', 'F1604_Squinty', 'F1606_Windolene']:

        high_units = pd.read_csv(f'G:/neural_chapter/figures/unit_ids_trained_topgenindex_{ferret_id_fancy}.csv')
        # remove trailing steam



        high_units = high_units[(high_units['rec_name'] == rec_name) & (high_units['stream'] == stream)]
        clust_ids = high_units['ID'].to_list()
        brain_area = high_units['BrainArea'].to_list()

    #plot with error b


    return



if __name__ == '__main__':
    main()