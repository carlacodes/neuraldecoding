import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import seaborn as sns
from scipy.stats import gaussian_kde

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
    for probeword in probewordlist:
        singleunitlist_copy = singleunitlist.copy()
        multiunitlist_copy = multiunitlist.copy()
        json_file_path = f'F:\split_cluster_jsons/{fullid}/cluster_split_list.json'


        with open(json_file_path, "r") as json_file:
            loaded_data = json.load(json_file)
        #get the recname
        recname = saveDir.split('/')[-3]
        #get the recname from the json file
        stream_id = stream[-4:]
        recname_json =  loaded_data.get(recname)
        #get the cluster ids from the json file
        original_to_split_cluster_ids = recname_json.get(stream_id)
        original_to_split_cluster_ids = original_to_split_cluster_ids.get('cluster_to_split_list')
        if original_to_split_cluster_ids == 'clust_ids':
            #get all the unique clusters ids
            original_to_split_cluster_ids = np.unique(scores['talker1']['target_vs_probe']['pitchshift']['cluster_id']+scores['talker1']['target_vs_probe']['nopitchshift']['cluster_id'])
            #make sure they are all less than 100
            original_to_split_cluster_ids = [x for x in original_to_split_cluster_ids if x < 100]
        elif original_to_split_cluster_ids == None:
            original_to_split_cluster_ids = np.array([])




        probewordindex = probeword[0]
        print(probewordindex)
        stringprobewordindex = str(probewordindex)

        scores = np.load(
            saveDir  + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_bs.npy',
            allow_pickle=True)[()]

        #print all the cluster ids for the scores
        print(f'cluster ids for animal:{ferretname}, and stream:{saveDir}')
        print(scores['talker1']['target_vs_probe']['pitchshift']['cluster_id'])
        print(scores['talker1']['target_vs_probe']['nopitchshift']['cluster_id'])

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





        for talker in [1]:
            comparisons = [comp for comp in scores[f'talker{talker}']]

            for comp in comparisons:
                for cond in ['pitchshift', 'nopitchshift']:
                    for i, clus in enumerate(scores[f'talker{talker}'][comp][cond]['cluster_id']):

                        print(i, clus)

                        if ferretname == 'Orecchiette':
                            #read csv file and get cluster id
                            print('checking clusters are in AC')
                            #read numpy from csv
                            #read pickle file

                            channelpositions = pd.read_pickle(r'D:\spkvisanddecodeproj2/analysisscriptsmodcg/visualisation\channelpositions\F2003_Orecchiette/channelpos.pkl')
                            #remove rows that are not the cluster id, represented by the first column in the np array out of three columns

                            channelpositions = channelpositions[channelpositions[:, 0] == clus, :]
                            #get the x and y coordinates
                            y_pos = channelpositions[:].tolist()[0][2]
                            y_post = int(y_pos)
                            if y_pos < 3200:
                                print('selected cluster below auditory cortex')
                                pass

                        if clus in singleunitlist_copy:
                            print('in single unit list')
                            if cond == 'pitchshift':
                                if talker == 1:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:
                                        su_pitchshiftlist_female = np.append(su_pitchshiftlist_female,
                                                                             scores[f'talker{talker}'][comp][cond][
                                                                                 'lstm_balanced_avg'][i])
                                elif talker == 2:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        su_pitchshiftlist_male = np.append(su_pitchshiftlist_male,
                                                                           scores[f'talker{talker}'][comp][cond][
                                                                               'lstm_balanced_avg'][i])
                                # print(pitchshiftlist.size)
                            elif cond == 'nopitchshift':
                                if talker == 1:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        su_nonpitchshiftlist_female = np.append(su_nonpitchshiftlist_female,
                                                                                scores[f'talker{talker}'][comp][cond][
                                                                                    'lstm_balanced_avg'][i])
                                elif talker == 2:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        su_nonpitchshiftlist_male = np.append(su_nonpitchshiftlist_male,
                                                                              scores[f'talker{talker}'][comp][cond][
                                                                                  'lstm_balanced_avg'][i])

                        elif clus in multiunitlist_copy:
                            if cond == 'pitchshift':
                                if talker == 1:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        mu_pitchshiftlist_female = np.append(mu_pitchshiftlist_female,
                                                                             scores[f'talker{talker}'][comp][cond][
                                                                                 'lstm_balanced_avg'][
                                                                                 i])

                                elif talker == 2:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        mu_pitchshiftlist_male = np.append(mu_pitchshiftlist_male,
                                                                           scores[f'talker{talker}'][comp][cond][
                                                                               'lstm_balanced_avg'][
                                                                               i])
                                        cluster_list_male_mu = np.append(cluster_list_male_mu, clus)


                            if cond == 'nopitchshift':
                                if talker == 1:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        mu_nonpitchshiftlist_female = np.append(mu_nonpitchshiftlist_female,
                                                                                scores[f'talker{talker}'][comp][cond][
                                                                                    'lstm_balanced_avg'][i])

                                elif talker == 2:
                                    if scores[f'talker{talker}'][comp][cond]['lstm_balanced_avg'][i] > scores[f'talker{talker}'][comp][cond]['perm_bal_ac'][i]:

                                        mu_nonpitchshiftlist_male = np.append(mu_nonpitchshiftlist_male,
                                                                              scores[f'talker{talker}'][comp][cond][
                                                                                  'lstm_balanced_avg'][i])
                                        cluster_list_male_mu_nops= np.append(cluster_list_male_mu_nops, clus)


                        elif clus in noiselist:
                            pass


    dictofsortedscores = {'su_list': {'pitchshift': {'female_talker': {},
                                                     'male_talker': {}},
                                      'nonpitchshift': {'female_talker': {},
                                                        'male_talker': {}}},
                          'mu_list': {'pitchshift': {'female_talker': {},
                                                     'male_talker': {}},
                                      'nonpitchshift': {'female_talker': {},
                                                        'male_talker': {}}}}

    dictofsortedscores['su_list']['pitchshift']['female_talker'] = su_pitchshiftlist_female
    dictofsortedscores['su_list']['pitchshift']['male_talker'] = su_pitchshiftlist_male
    dictofsortedscores['su_list']['nonpitchshift']['female_talker'] = su_nonpitchshiftlist_female
    dictofsortedscores['su_list']['nonpitchshift']['male_talker'] = su_nonpitchshiftlist_male

    dictofsortedscores['mu_list']['pitchshift']['female_talker'] = mu_pitchshiftlist_female
    dictofsortedscores['mu_list']['pitchshift']['male_talker'] = mu_pitchshiftlist_male
    dictofsortedscores['mu_list']['nonpitchshift']['female_talker'] = mu_nonpitchshiftlist_female
    dictofsortedscores['mu_list']['nonpitchshift']['male_talker'] = mu_nonpitchshiftlist_male

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
    probewordlist_l74 = [(6, 6), None, (3,3), None, None]
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
    count = 0
    for probeword in probewordlist:


        for animal in animal_list:
            dictoutput[animal] = {}
            for stream in report[animal]:
                dictoutput[animal][stream] = {}
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
                animal_text = animal.split('_')[1]

                max_length = len(stream) // 2

                for length in range(1, max_length + 1):
                    for i in range(len(stream) - length):
                        substring = stream[i:i + length]
                        if stream.count(substring) > 1:
                            repeating_substring = substring
                            break

                print(repeating_substring)
                rec_name_unique = repeating_substring[0:-1]

                if animal == 'F1604_Squinty':
                    try:
                        dictoutput_instance = scatterplot_and_visualise([probewordlist_l74[count]],
                                                                        saveDir=f'F:/results_24102023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                        ferretname=animal_text,
                                                                        singleunitlist=singleunitlist[animal][stream],
                                                                        multiunitlist=multiunitlist[animal][stream],
                                                                        noiselist=noiselist[animal][stream], stream = stream, fullid = animal)
                        dictoutput_all.append(dictoutput_instance)
                    except:
                        #print the exception
                        print(f'no scores for this stream:{stream}, and {animal}')
                        pass
                elif animal == 'F1606_Windolene':

                    try:
                        dictoutput_instance = scatterplot_and_visualise([probewordlist_l74[count]],
                                                                        saveDir=f'F:/results_24102023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                        ferretname=animal_text,
                                                                        singleunitlist=singleunitlist[animal][stream],
                                                                        multiunitlist=multiunitlist[animal][stream],
                                                                        noiselist=noiselist[animal][stream], stream = stream, fullid = animal)
                        dictoutput_all.append(dictoutput_instance)
                    except:
                        #print the exception
                        print(f'no scores for this stream:{stream}, and {animal}')
                        pass
                elif animal == 'F1815_Cruella' or animal == 'F1902_Eclair' or animal =='F1702_Zola':
                    try:
                        dictoutput_instance = scatterplot_and_visualise([probeword], saveDir= f'F:/results_24102023/{animal}/{rec_name_unique}/{streamtext}/',
                                                                        ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                        multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal)
                        dictoutput_all.append(dictoutput_instance)
                    except:
                        #print the exception
                        print(f'no scores for this stream:{stream}, and {animal}')
                        pass
                else:
                    # try:
                    dictoutput_instance = scatterplot_and_visualise([probeword], saveDir= f'F:/results_24102023/{animal}/{rec_name_unique}/{streamtext}/',
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

        generate_plots(dictoutput_all, dictoutput_trained, dictoutput_naive, labels, colors, probeword = probeword)
        count += 1

    return


def generate_plots(dictlist, dictlist_trained, dictlist_naive, labels, colors, probeword = (2,2)):
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


    for dictoutput in dictlist:
        for key in dictoutput.keys():
            for key3 in dictoutput[key]['pitchshift'].keys():
                if len(dictoutput[key]['nonpitchshift'][key3]) < len(
                        dictoutput[key]['pitchshift'][key3]):
                    dictoutput[key]['pitchshift'][key3] = \
                        dictoutput[key]['pitchshift'][key3][
                        :len(dictoutput[key]['nonpitchshift'][key3])]
                elif len(dictoutput[key]['nonpitchshift'][key3]) > len(
                        dictoutput[key]['pitchshift'][key3]):
                    dictoutput[key]['nonpitchshift'][key3] = \
                        dictoutput[key]['nonpitchshift'][key3][
                        :len(dictoutput[key]['pitchshift'][key3])]

    bigconcatenatetrained_ps = np.empty(0)
    bigconcatenatetrained_nonps = np.empty(0)
    for dictouput in dictlist_trained:
        for key in dictouput.keys():
            for key3 in dictouput[key]['pitchshift'].keys():
                bigconcatenatetrained_ps = np.concatenate(
                    (bigconcatenatetrained_ps, dictouput[key]['pitchshift'][key3]))
                bigconcatenatetrained_nonps = np.concatenate(
                    (bigconcatenatetrained_nonps, dictouput[key]['nonpitchshift'][key3]))


    bigconcatenatenaive_ps = np.empty(0)
    bigconcatenatenaive_nonps = np.empty(0)

    for dictouput in dictlist_naive:
        for key in dictouput.keys():
            # print(key, 'key')
            for key3 in dictouput[key]['pitchshift'].keys():
                # print(key3, 'key3')
                bigconcatenatenaive_ps = np.concatenate((bigconcatenatenaive_ps, dictouput[key]['pitchshift'][key3]))
                bigconcatenatenaive_nonps = np.concatenate(
                    (bigconcatenatenaive_nonps, dictouput[key]['nonpitchshift'][key3]))


    # Define labels and colors for scatter plots
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

    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)

    # Create a 2D density plot
    sns.kdeplot(x=bigconcatenatetrained_nonps, y=bigconcatenatetrained_ps, cmap="viridis", fill=True, ax=ax)

    # You can customize the colormap and other parameters as needed

    # Set the title
    ax.set_title('Trained animals, number of points: ' + str(len(bigconcatenatetrained_ps)))

    # Show the plot
    plt.show()
    unique_scores = np.unique(bigconcatenatetrained_ps)
    len(unique_scores)

    fig, ax = plt.subplots(1, figsize=(9, 9), dpi=300)


    # ax.scatter(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, marker='P', color='purple', alpha=0.5, label='trained')
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
    plt.savefig(f'D:/scattermuaandsuregplot_mod_21062023_probeword{probeword[0]}.png', dpi=1000)
    plt.savefig(f'D:/scattermuaandsuregplot_mod_21062023_probeword{probeword[0]}.pdf', dpi=1000)
    plt.show()

    #histogram distribution of the trained and naive animals
    fig, ax = plt.subplots(1, figsize=(8, 8))
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
    plt.savefig(f'D:/diffF0distribution_20062023_probeword{probeword[0]}.png', dpi=1000)
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
    plt.savefig(f'D:/diffF0distribution_frac_20062023wlegendintertrialroving_probeword{probeword[0]}.png', dpi=1000)
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
    plt.savefig(f'D:/controlF0distribution20062023intertrialroving_probeword_{probeword[0]}.png', dpi=1000)
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
    plt.savefig(f'D:/rovedF0distribution_20062023intertrialroving_probeword{probeword[0]}.png', dpi=1000)
    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi = 800)
    ax.set_xlim([0,1])
    sns.distplot(bigconcatenatetrained_ps,  label='trained roved',ax=ax, color='purple')
    sns.distplot(bigconcatenatetrained_nonps,  label='trained control',ax=ax, color='magenta')
    ax.legend(fontsize=18)
    plt.title('Roved and Control F0 Distributions for the Trained Animals', fontsize = 18)
    plt.xlabel(' LSTM decoder scores', fontsize = 20)
    plt.savefig(f'D:/rovedF0vscontrolF0traineddistribution_20062023intertrialroving_probeword{probeword[0]}.png', dpi=1000)
    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi = 800)
    ax.set_xlim([0,1])
    sns.distplot(bigconcatenatenaive_ps,  label='naive roved',ax=ax, color='darkcyan')
    sns.distplot(bigconcatenatenaive_nonps,  label='naive control',ax=ax, color='cyan')
    ax.legend(fontsize=18)
    plt.xlabel(' LSTM decoder scores', fontsize = 20)
    plt.title('Roved and Control F0 Distributions for the Naive Animals', fontsize = 18)
    plt.savefig(f'D:/rovedF0vscontrolF0naivedistribution_20062023intertrialroving_probeword{probeword[0]}.png', dpi=1000)
    plt.show()
    kstestcontrolf0vsrovedtrained = scipy.stats.kstest(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, alternative = 'two-sided')

    kstestcontrolf0vsrovednaive = scipy.stats.kstest(bigconcatenatenaive_nonps, bigconcatenatenaive_ps, alternative='two-sided')

    naivearray=np.concatenate((np.zeros((len(bigconcatenatetrained_nonps)+len(bigconcatenatetrained_ps),1)), np.ones((len(bigconcatenatenaive_nonps)+len(bigconcatenatenaive_ps),1))))
    trainedarray=np.concatenate((np.ones((len(bigconcatenatetrained_nonps)+len(bigconcatenatetrained_ps),1)), np.zeros((len(bigconcatenatenaive_nonps)+len(bigconcatenatenaive_ps),1))))
    controlF0array=np.concatenate((np.ones((len(bigconcatenatetrained_nonps),1)), np.zeros((len(bigconcatenatetrained_ps),1)), np.ones((len(bigconcatenatenaive_nonps),1)), np.zeros((len(bigconcatenatenaive_ps),1))))
    rovedF0array = np.concatenate((np.zeros((len(bigconcatenatetrained_nonps),1)), np.ones((len(bigconcatenatetrained_ps),1)), np.zeros((len(bigconcatenatenaive_nonps),1)), np.ones((len(bigconcatenatenaive_ps),1))))
    scores = np.concatenate((bigconcatenatetrained_nonps, bigconcatenatetrained_ps, bigconcatenatenaive_nonps, bigconcatenatenaive_ps))

    dataset = pd.DataFrame({'trained': trainedarray[:,0], 'naive': naivearray[:,0], 'controlF0': controlF0array[:,0], 'rovedF0': rovedF0array[:,0], 'scores': scores})

    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    model = ols('scores ~ C(trained) + C(controlF0) ', data=dataset).fit()
    print(model.summary())
    table = sm.stats.anova_lm(model, typ=2)
    print(table)
    from statsmodels.iolib.summary2 import summary_col
    res = summary_col([model], regressor_order=model.params.index.tolist())
    print(res)

    model = ols('scores ~ C(trained) + C(controlF0) ', data=dataset).fit()
    print(model.summary())
    table = sm.stats.anova_lm(model, typ=2)



if __name__ == '__main__':
    main()