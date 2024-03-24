import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import shap
import statsmodels as sm
import lightgbm as lgb
from pathlib import Path
import scipy
from helpers.GeneratePlotsConcise import *
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


def load_scores_and_filter(probewordlist,
                           saveDir='D:/Users/cgriffiths/resultsms4/lstm_output_frommyriad_15012023/lstm_kfold_14012023_crumble',
                           ferretname='Crumble',
                           singleunitlist=[0,1,2],
                           multiunitlist=[0,1,2,3], noiselist=[], stream = 'BB_2', fullid = 'F1901_Crumble', report =[], permutation_scores=False, pitchshift_text = 'nopitchshift'):
    if permutation_scores == False:
        score_key = 'lstm_balanced_avg'
    else:
        score_key = 'perm_bal_ac'
    singleunitlist = [x - 1 for x in singleunitlist]
    multiunitlist = [x - 1 for x in multiunitlist]
    noiselist = [x - 1 for x in noiselist]
    original_cluster_list = np.empty([0])
    with open('D:\spkvisanddecodeproj2/analysisscriptsmodcg/json_files\electrode_positions.json') as f:
        electrode_position_data = json.load(f)




    #declare a dataframe to store the scores
    sorted_df_of_scores = pd.DataFrame({'probeword1': [], 'probeword2': [], 'cluster_id': [], 'score': [], 'unit_type': [], 'animal': [], 'stream': [], 'recname': [], 'clus_id_report': [], 'brain_area': []})
    for probeword1 in probewordlist:
        for probeword2 in probewordlist:
            singleunitlist_copy = singleunitlist.copy()
            multiunitlist_copy = multiunitlist.copy()
            key_text = f'{probeword1[0]}_{probeword1[0]}_{probeword2[0]}_{probeword2[0]}'

            #load the original clusters to split from the json file
            json_file_path = f'F:\split_cluster_jsons/{fullid}/cluster_split_list.json'
            if ferretname == 'Orecchiette':
                original_to_split_cluster_ids = np.array([])
                probewordindex_1 = str(probeword1[0])
                probewordindex_2 = str(probeword2[0])
                try:
                    if probewordindex_1 == '1':
                        scores = np.load(
                            saveDir + r'scores_' + ferretname + '_2022_' + probewordindex_2+ '_' + ferretname + '_probe_bs.npy',
                            allow_pickle=True)[()]
                    elif probewordindex_2 == '1':
                        scores = np.load(
                            saveDir + r'scores_' + ferretname + '_2022_' + probewordindex_1 + '_' + ferretname + '_probe_bs.npy',
                            allow_pickle=True)[()]
                    else:
                        try:
                            scores = np.load(
                                f'{saveDir}/scores_{probewordindex_1}_vs_{probewordindex_2}_{ferretname}_probe_{pitchshift_text}_bs.npy',
                                allow_pickle=True)[()]
                        except:
                            #load the reverse as it's the same
                            scores = np.load(
                                f'{saveDir}/scores_{probewordindex_2}_vs_{probewordindex_1}_{ferretname}_probe_{pitchshift_text}_bs.npy',
                                allow_pickle=True)[()]
                except Exception as e:
                    print(e)
                    continue

            else:
                with open(json_file_path, "r") as json_file:
                    loaded_data = json.load(json_file)
                recname = saveDir.split('/')[-3]
                stream_id = stream[-4:]

                if 'BB_3' in stream_id and ferretname != 'Squinty':
                    side_of_implant = 'right'
                elif 'BB_2' in stream_id and ferretname != 'Squinty':
                    side_of_implant = 'right'
                elif 'BB_4' in stream_id:
                    side_of_implant = 'left'
                elif 'BB_5' in stream_id:
                    side_of_implant = 'left'
                else:
                    side_of_implant = 'left'
                if recname == '01_03_2022_cruellabb4bb5':
                    recname = '01_03_2022_cruella'
                elif recname == '25_01_2023_cruellabb4bb5':
                    recname = '25_01_2023_cruella'
                recname_json = loaded_data.get(recname)

                #get the cluster ids from the json file
                original_to_split_cluster_ids = recname_json.get(stream_id)
                original_to_split_cluster_ids = original_to_split_cluster_ids.get('cluster_to_split_list')
                if original_to_split_cluster_ids == 'clust_ids':
                    probewordindex_1 = str(probeword1[0])
                    probewordindex_2 = str(probeword2[0])
                    try:
                        if probewordindex_1 == '1':
                            scores = np.load(
                                saveDir + r'scores_' + ferretname + '_2022_' + probewordindex_2 + '_' + ferretname + '_probe_bs.npy',
                                allow_pickle=True)[()]
                        elif probewordindex_2 == '1':
                            scores = np.load(
                                saveDir + r'scores_' + ferretname + '_2022_' + probewordindex_1 + '_' + ferretname + '_probe_bs.npy',
                                allow_pickle=True)[()]
                        else:
                            scores = np.load(
                                f'{saveDir}/scores_{probewordindex_1}_vs_{probewordindex_2}_{ferretname}_probe_{pitchshift_text}_bs.npy',
                                allow_pickle=True)[()]
                    except Exception as e:
                        print(e)
                        continue
                    original_to_split_cluster_ids = scores['talker1']['target_vs_probe'][key_text]['cluster_id']
                    #if all of them need splitting
                elif original_to_split_cluster_ids:
                    #TODO: not sure if this elif needed
                    #get all the unique clusters ids
                    probewordindex_1 = str(probeword1[0])
                    probewordindex_2 = str(probeword2[0])
                    try:
                        if probewordindex_1 == '1':
                            scores = np.load(
                                saveDir + r'scores_' + ferretname + '_2022_' + probewordindex_2 + '_' + ferretname + '_probe_bs.npy',
                                allow_pickle=True)[()]
                        elif probewordindex_2 == '1':
                            scores = np.load(
                                saveDir + r'scores_' + ferretname + '_2022_' + probewordindex_1 + '_' + ferretname + '_probe_bs.npy',
                                allow_pickle=True)[()]
                        else:
                            scores = np.load(
                                f'{saveDir}/scores_{probewordindex_1}_vs_{probewordindex_2}_{ferretname}_probe_{pitchshift_text}_bs.npy',
                                allow_pickle=True)[()]
                    except Exception as e:
                        print(e)
                        continue
                    original_to_split_cluster_ids = [x for x in original_to_split_cluster_ids if x < 100]

                elif original_to_split_cluster_ids == None or not original_to_split_cluster_ids:
                    original_to_split_cluster_ids = np.array([])
                    probewordindex_1 = str(probeword1[0])
                    probewordindex_2 = str(probeword2[0])
                    try:
                        scores = np.load(
                            f'{saveDir}/scores_{probewordindex_1}_vs_{probewordindex_2}_{ferretname}_probe_{pitchshift_text}_bs.npy',
                            allow_pickle=True)[()]
                    except Exception as e:
                        print(e)
                        continue

            for talker in [1]:
                comparisons = [comp for comp in scores[f'talker{talker}']]
                for comp in comparisons:
                    if probewordindex_1 == '1' or probewordindex_2 == '1':
                        key_text = f'{pitchshift_text}'
                    else:
                        key_text = f'{probeword1[0]}_{probeword1[0]}_{probeword2[0]}_{probeword2[0]}'
                        #check if the key_text is in the scores
                        if key_text not in scores[f'talker{talker}'][comp] :
                            key_text = f'{probeword2[0]}_{probeword2[0]}_{probeword1[0]}_{probeword1[0]}'

                    for i, clus in enumerate(scores[f'talker{talker}'][comp][key_text]['cluster_id']):
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
                    report['tdt'] = np.zeros((len(report), 1))

            for talker in [1]:
                comparisons = [comp for comp in scores[f'talker{talker}']]
                for comp in comparisons:
                    if probewordindex_1 == '1' or probewordindex_2 == '1':
                        key_text = f'{pitchshift_text}'
                    else:
                        key_text = f'{probeword1[0]}_{probeword1[0]}_{probeword2[0]}_{probeword2[0]}'
                        if key_text not in scores[f'talker{talker}'][comp] :
                            key_text = f'{probeword2[0]}_{probeword2[0]}_{probeword1[0]}_{probeword1[0]}'
                    for i, clus in enumerate(scores[f'talker{talker}'][comp][key_text]['cluster_id']):
                        stream_small = stream[-4:]
                        clust_text = str(clus)+'_'+fullid+'_'+recname+'_'+stream_small
                        print(i, clus)
                        if fullid == 'F1702_Zola':
                            probeword_map = {
                                (2, 2): (4, 4),
                                (5, 6): (2, 2),
                                (20, 20): (3, 3),
                                (42, 49): (5, 5),
                                (32, 38): (7, 7)
                            }
                            probeword1_input_text = probeword_map.get(probeword1, probeword1)
                            probeword2_input_text = probeword_map.get(probeword2, probeword2)

                        elif fullid == 'F1604_Squinty' or fullid == 'F1606_Windolene':
                            probeword_map = {
                                (3, 3): (5, 5),
                                (6, 6): (4, 4),
                                (2, 2): (13, 13),
                                (4, 4): (15, 15),
                                (5, 5): (16, 16),
                                (7, 7): (18, 18),
                                (8, 8): (19, 19),
                                (9, 9): (20, 20),
                                (10, 10): (21, 21),
                                (11, 11): (22, 22),
                                (12, 12): (23, 23),
                                (14, 14): (7, 7)
                            }
                            probeword1_input_text = probeword_map.get(probeword1, probeword1)
                            probeword2_input_text = probeword_map.get(probeword2, probeword2)
                        else:
                            probeword1_input_text = probeword1
                            probeword2_input_text = probeword2


                        if 200 > clus >= 100 and fullid != 'F2003_Orecchiette':
                            clus_id_report = clus - 100
                        elif 300> clus >= 200 and fullid != 'F2003_Orecchiette':
                            clus_id_report = clus - 200
                        elif 400 > clus >= 300 and fullid != 'F2003_Orecchiette':
                            clus_id_report = clus - 300
                        else:
                            clus_id_report = clus
                        if clus in singleunitlist_copy:
                            unit_type = 'su'
                            #append to the dataframe
                            electrode_position_dict = electrode_position_data.get(fullid)
                            if electrode_position_dict:
                                tdt_position = report['tdt'][clus_id_report]
                                side_of_implant_list = electrode_position_dict.get(side_of_implant)
                                #convert list of dicts to a dataframe
                                side_of_implant_df = pd.DataFrame(side_of_implant_list)
                                #get the brain area
                                channel_id_and_brain_area = side_of_implant_df[side_of_implant_df['TDT_NUMBER'] == tdt_position]
                                brain_area = channel_id_and_brain_area['area'].values[0]

                                sorted_df_of_scores = sorted_df_of_scores.append(
                                        {'probeword1': probeword1_input_text[0], 'probeword2': probeword2_input_text[0],
                                         'cluster_id': clus,
                                         'score': scores[f'talker{talker}'][comp][key_text][score_key][i],
                                         'unit_type': unit_type, 'animal': fullid, 'stream': stream_id, 'recname': recname,
                                         'clus_id_report': clus_id_report, 'tdt_electrode_num': tdt_position,
                                         'brain_area': brain_area}, ignore_index=True)
                            elif fullid == 'F2003_Orecchiette':
                                if 'mod' in stream:
                                    brain_area = 'PEG'
                                elif 's2' in stream:
                                    brain_area = 'PEG'
                                elif 's3' in stream:
                                    brain_area = 'MEG'
                                tdt_position = -1
                                #NEED TO FIGURE OUT WHAT stream NG_0 IS ON MYRIAD TODO
                                try:
                                    sorted_df_of_scores = sorted_df_of_scores.append(
                                        {'probeword1': probeword1_input_text[0], 'probeword2': probeword2_input_text[0],
                                         'cluster_id': clus,
                                         'score': scores[f'talker{talker}'][comp][key_text][score_key][i],
                                         'unit_type': unit_type, 'animal': fullid, 'stream': stream_id, 'recname': recname,
                                         'clus_id_report': clus_id_report, 'tdt_electrode_num': tdt_position, 'brain_area': brain_area}, ignore_index=True)
                                except Exception as e:
                                    print(e)
                                    continue



                        elif clus in multiunitlist_copy:
                            unit_type = 'mua'
                            #append to the dataframe
                            electrode_position_dict = electrode_position_data.get(fullid)
                            if electrode_position_dict:
                                tdt_position = report['tdt'][clus_id_report]
                                side_of_implant_list = electrode_position_dict.get(side_of_implant)
                                #convert list of dicts to a dataframe
                                side_of_implant_df = pd.DataFrame(side_of_implant_list)
                                #get the brain area
                                channel_id_and_brain_area = side_of_implant_df[side_of_implant_df['TDT_NUMBER'] == tdt_position]
                                print(fullid, clus, tdt_position, channel_id_and_brain_area)
                                try:
                                    brain_area = channel_id_and_brain_area['area'].values[0]
                                except Exception as e:
                                    print(e)
                                    continue

                                sorted_df_of_scores = sorted_df_of_scores.append(
                                        {'probeword1': probeword1_input_text[0], 'probeword2': probeword2_input_text[0],
                                         'cluster_id': clus,
                                         'score': scores[f'talker{talker}'][comp][key_text][score_key][i],
                                         'unit_type': unit_type, 'animal': fullid, 'stream': stream_id, 'recname': recname,
                                         'clus_id_report': clus_id_report, 'tdt_electrode_num': tdt_position,
                                         'brain_area': brain_area}, ignore_index=True)
                            elif fullid == 'F2003_Orecchiette':
                                if 'mod' in stream:
                                    brain_area = 'PEG'
                                elif 's2' in stream:
                                    brain_area = 'PEG'
                                elif 's3' in stream:
                                    brain_area = 'MEG'
                                tdt_position = -1
                                stream_id = stream[-2:]

                                sorted_df_of_scores = sorted_df_of_scores.append(
                                    {'probeword1': probeword1_input_text[0], 'probeword2': probeword2_input_text[0],
                                     'cluster_id': clus,
                                     'score': scores[f'talker{talker}'][comp][key_text][score_key][i],
                                     'unit_type': unit_type, 'animal': fullid, 'stream': stream_id, 'recname': recname,
                                     'clus_id_report': clus_id_report, 'tdt_electrode_num': tdt_position, 'brain_area': brain_area}, ignore_index=True)

                        elif clus in noiselist:
                            pass




    return sorted_df_of_scores


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


def load_animal_electrode_data(animal, stream):
    with open('D:\spkvisanddecodeproj2/analysisscriptsmodcg/json_files\electrode_positions.json') as f:
        electrode_position_data = json.load(f)

    #load the corresponding channel_id
    if 'F1604_Squinty' in animal:
        animal = 'F1604_Squinty'
        side = 'left'
    elif 'F1606_Windolene' in animal:
        animal = 'F1606_Windolene'
    elif 'F1702_Zola' in animal:
        animal = 'F1702_Zola'
    elif 'F1815_Cruella' in animal:
        animal = 'F1815_Cruella'
    elif 'F1901_Crumble' in animal:
        animal = 'F1901_Crumble'
    elif 'F1902_Eclair' in animal:
        animal = 'F1902_Eclair'
    elif 'F1812_Nala' in animal:
        animal = 'F1812_Nala'
    elif 'F2003_Orecchiette' in animal:
        animal = 'F2003_Orecchiette'


    if 'BB_3' in side and animal!='F1604_Squinty':
        side = 'right'
    elif 'BB_2' in unit_id and animal!='F1604_Squinty':
        side = 'right'
    elif 'BB_4' in unit_id:
        side = 'left'
    elif 'BB_5' in unit_id:
        side = 'left'
    return
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
    probewordlist_zola = [(1,1), (2, 2), (5, 6), (42, 49), (32, 38), (20, 22)]
    probewordlist =[ (1,1), (2,2), (3,3), (4,4),(5,5), (6,6), (7,7), (8,8), (9,9), (10,10)]
    probewordlist_l74 = [(1,1), (10, 10), (2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (8, 8), (9, 9), (11, 11), (12, 12),
                             (14, 14)]
    animal_list = [ 'F1604_Squinty', 'F1901_Crumble', 'F1606_Windolene', 'F1702_Zola','F1815_Cruella', 'F1902_Eclair', 'F1812_Nala',  'F2003_Orecchiette',]
    # animal_list = ['F1902_Eclair']
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

    for pitchshift_option in ['nopitchshift', 'pitchshift']:
        dictoutput = {}
        df_all_trained = pd.DataFrame()
        df_all_trained_permutation = pd.DataFrame()
        df_all_naive = pd.DataFrame()
        df_all_naive_permutation = pd.DataFrame()


        df_all = pd.DataFrame()
        df_all_permutation = pd.DataFrame()
        for animal in animal_list:
            dictoutput[animal] = {}
            for stream in report[animal]:
                dictoutput[animal][stream] = {}
                animal_text = animal.split('_')[1]
                if animal =='F2003_Orecchiette':
                    rec_name_unique = stream
                else:
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
                if animal == 'F1604_Squinty':
                    df_instance = load_scores_and_filter(probewordlist_l74,
                                                                 saveDir=f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                 ferretname=animal_text,
                                                                 singleunitlist=singleunitlist[animal][stream],
                                                                 multiunitlist=multiunitlist[animal][stream],
                                                                 noiselist=noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], pitchshift_text=pitchshift_option)
                    df_all.append(df_instance)
                    df_instance_permutation = load_scores_and_filter(probewordlist_l74,
                                                                             saveDir=f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                             ferretname=animal_text,
                                                                             singleunitlist=singleunitlist[animal][stream],
                                                                             multiunitlist=multiunitlist[animal][stream],
                                                                             noiselist=noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], permutation_scores=True, pitchshift_text=pitchshift_option)
                    df_all_permutation.append(df_instance_permutation)

                elif animal == 'F1606_Windolene':
                    df_instance = load_scores_and_filter(probewordlist_l74,
                                                                 saveDir=f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                 ferretname=animal_text,
                                                                 singleunitlist=singleunitlist[animal][stream],
                                                                 multiunitlist=multiunitlist[animal][stream],
                                                                 noiselist=noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], pitchshift_text=pitchshift_option)
                    df_all = pd.concat([df_all, df_instance])

                    df_instance_permutation = load_scores_and_filter(probewordlist_l74,
                                                                             saveDir=f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                             ferretname=animal_text,
                                                                             singleunitlist=singleunitlist[animal][stream],
                                                                             multiunitlist=multiunitlist[animal][stream],
                                                                             noiselist=noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], permutation_scores=True, pitchshift_text=pitchshift_option)
                    df_all_permutation = pd.concat([df_all_permutation, df_instance_permutation])

                elif animal =='F1702_Zola':
                    df_instance = load_scores_and_filter(probewordlist_zola, saveDir=f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                 ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                 multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], pitchshift_text=pitchshift_option)
                    df_all.append(df_instance)

                    df_instance_permutation = load_scores_and_filter(probewordlist_zola, saveDir=f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                             ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                             multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream]
                                                                             , permutation_scores=True, pitchshift_text=pitchshift_option)
                    df_all_permutation = pd.concat([df_all_permutation, df_instance_permutation])

                elif animal == 'F1815_Cruella' or animal == 'F1902_Eclair':
                    df_instance = load_scores_and_filter(probewordlist, saveDir=f'G:/results_distvdist_21032024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                 ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                 multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], pitchshift_text=pitchshift_option)
                    df_all = pd.concat([df_all, df_instance])

                    df_instance_permutation = load_scores_and_filter(probewordlist, saveDir=f'G:/results_distvdist_21032024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                             ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                             multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream]
                                                                             , permutation_scores=True, pitchshift_text=pitchshift_option)
                    df_all_permutation = pd.concat([df_all_permutation, df_instance_permutation])

                elif animal == 'F2003_Orecchiette':
                    # try:
                    df_instance = load_scores_and_filter(probewordlist,
                                                                 saveDir=f'G:/results_distvdist_21032024/{animal}/{rec_name_unique}/',
                                                                 ferretname=animal_text,
                                                                 singleunitlist=singleunitlist[animal][stream],
                                                                 multiunitlist=multiunitlist[animal][stream],
                                                                 noiselist=noiselist[animal][stream], stream=stream,
                                                                 fullid=animal,
                                                                 report=report[animal][stream], pitchshift_text=pitchshift_option)

                    df_all = pd.concat([df_all, df_instance])
                    df_instance_permutation = load_scores_and_filter(probewordlist,
                                                                             saveDir=f'G:/results_distvdist_21032024/{animal}/{rec_name_unique}/',
                                                                             ferretname=animal_text,
                                                                             singleunitlist=singleunitlist[animal][stream],
                                                                             multiunitlist=multiunitlist[animal][stream],
                                                                             noiselist=noiselist[animal][stream], stream=stream,
                                                                             fullid=animal,
                                                                             report=report[animal][stream], permutation_scores=True, pitchshift_text=pitchshift_option)

                    df_all_permutation = pd.concat([df_all_permutation, df_instance_permutation])

                else:
                    df_instance = load_scores_and_filter(probewordlist, saveDir=f'G:/results_distvdist_21032024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                 ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                 multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], pitchshift_text=pitchshift_option)
                    df_all = pd.concat([df_all, df_instance])

                    df_instance_permutation = load_scores_and_filter(probewordlist, saveDir=f'G:/results_distvdist_21032024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                             ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                             multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream]
                                                                             , permutation_scores=True, pitchshift_text=pitchshift_option)

                    df_all_permutation = pd.concat([df_all_permutation, df_instance_permutation])

                try:
                    if animal == 'F1604_Squinty' or animal == 'F1606_Windolene' or animal == 'F1702_Zola' or animal == 'F1815_Cruella':
                        print('trained animal'+ animal)
                        df_all_trained = pd.concat([df_all_trained, df_instance])
                        df_all_trained_permutation = pd.concat([df_all_trained_permutation, df_instance_permutation])
                    else:
                        print('naive animal:'+ animal)
                        df_all_naive = pd.concat([df_all_naive, df_instance])
                        df_all_naive_permutation = pd.concat([df_all_naive_permutation, df_instance_permutation])
                except:
                    print('no scores for this stream')
                    pass


        labels = [ 'F1901_Crumble', 'F1604_Squinty', 'F1606_Windolene', 'F1702_Zola','F1815_Cruella', 'F1902_Eclair', 'F1812_Nala']

        colors = ['purple', 'magenta', 'darkturquoise', 'olivedrab', 'steelblue', 'darkcyan', 'darkorange']
        # plot_heatmap(df_all_trained, trained = True)
        # plot_heatmap(df_all_naive, trained=False)

        data_trained_filtered = filter_for_units_used_in_first_analysis(df_all_trained, trained = True)
        data_naive_filtered = filter_for_units_used_in_first_analysis(df_all_naive, trained = False)
        df_all_trained_filtered_permutation = filter_for_units_used_in_first_analysis(df_all_trained_permutation, trained = True)
        df_all_naive_filtered_permutation = filter_for_units_used_in_first_analysis(df_all_naive_permutation, trained = False)

        plot_heatmap_with_comparison(data_trained_filtered, df_all_trained_filtered_permutation, trained=True, pitchshift_option=pitchshift_option)
        plot_heatmap_with_comparison(data_naive_filtered, df_all_naive_filtered_permutation, trained=False, pitchshift_option=pitchshift_option)
        plot_heatmap(data_trained_filtered, trained = True, pitchshift_option=pitchshift_option)
        plot_heatmap(data_naive_filtered, trained=False, pitchshift_option=pitchshift_option)
    return


def plot_heatmap(df_in, trained = True, pitchshift_option = 'nopitchshift'):
    '''plot a heatmap of the data for each probe word pair
    :param data: the data to be plotted
    :return: None
    '''

    # pivot_df = df_in.pivot(index='probeword1', columns='probeword2', values='score')
    df_in = df_in.groupby(['probeword1', 'probeword2']).mean().reset_index() # taking the mean of the scores across clusters for each probeword pair
    pivot_df = df_in.pivot(index='probeword1', columns='probeword2', values='score')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, cmap="YlGnBu", vmin=0.3, vmax = 0.7)

    if trained == True and pitchshift_option == 'nopitchshift':
        plt.title('Trained Animal LSTM decoding scores, control F0')
    elif trained == True and pitchshift_option == 'pitchshift':
        plt.title('Trained Animal LSTM decoding scores, roved F0')
    elif trained == False and pitchshift_option == 'nopitchshift':
        plt.title('Naive Animal LSTM decoding scores, control F0')
    elif trained == False and pitchshift_option == 'pitchshift':
        plt.title('Naive Animal LSTM decoding scores, roved F0')
    plt.savefig(f'G:/neural_chapter/figures/heatmap_dist_v_dist_trained_{trained}_{pitchshift_option}.png', dpi = 300)

    # Show the plot
    map_of_words = {1: 'instruments',  2: 'craft',
    3: 'in contrast to',
    4: 'when a',
    5: 'accurate',
    6: 'pink noise',
    7: 'of science',
    8: 'rev. instruments',
    9: 'boats',
    10: 'today',
    13: 'sailor',
    15: 'but',
    16: 'researched',
    18: 'took',
    19: 'the vast',
    20: 'today',
    21: 'he takes',
    22: 'becomes',
    23: 'any',
    24: 'more'
    }
    #use the map of words to replace the xticks and yticks with the words
    xticks = pivot_df.columns
    yticks = pivot_df.index
    xticks = [map_of_words.get(xtick, xtick) for xtick in xticks]
    yticks = [map_of_words.get(ytick, ytick) for ytick in yticks]
    plt.xticks(ticks = np.arange(0.5, len(xticks), 1), labels = xticks, rotation = 45)
    plt.yticks(ticks = np.arange(0.5, len(yticks), 1), labels = yticks, rotation = 45)
    plt.savefig(f'G:/neural_chapter/figures/heatmap_dist_v_dist_trained_raw_{trained}.png', dpi = 300)
    plt.show()
    return

def plot_heatmap_with_comparison(df_in, df_in_perm, trained = True, pitchshift_option = 'nopitchshift'):
    #plot the difference in the df_in and df_perm scores

    df_in = df_in.groupby(['probeword1', 'probeword2']).mean().reset_index() # taking the mean of the scores across clusters for each probeword pair
    pivot_df = df_in.pivot(index='probeword1', columns='probeword2', values='score')
    df_in_perm = df_in_perm.groupby(['probeword1', 'probeword2']).mean().reset_index() # taking the mean of the scores across clusters for each probeword pair
    pivot_df_perm = df_in_perm.pivot(index='probeword1', columns='probeword2', values='score')
    pivot_df = pivot_df - pivot_df_perm
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, cmap="YlGnBu", vmin=-0.1, vmax=0.2)
    if trained == True and pitchshift_option == 'nopitchshift':
        plt.title('Trained Animal LSTM decoding - permutation scores, control F0')
    elif trained == True and pitchshift_option == 'pitchshift':
        plt.title('Trained Animal LSTM decoding - permutation scores, roved F0')
    elif trained == False and pitchshift_option == 'nopitchshift':
        plt.title('Naive Animal LSTM decoding - permutation scores, control F0')
    elif trained == False and pitchshift_option == 'pitchshift':
        plt.title('Naive Animal LSTM decoding - permutation scores, roved F0')

    #use the labels to label the xticks and y ticks

    # df_full_pitchsplit_violinplot['ProbeWord'] = df_full_pitchsplit_violinplot['ProbeWord'].replace(
    #     {'(2,2)': 'craft', '(3,3)': 'in contrast to', '(4,4)': 'when a', '(5,5)': 'accurate', '(6,6)': 'pink noise',
    #      '(7,7)': 'of science', '(8,8)': 'rev. instruments', '(9,9)': 'boats', '(10,10)': 'today',
    #      '(13,13)': 'sailor', '(15,15)': 'but', '(16,16)': 'researched', '(18,18)': 'took', '(19,19)': 'the vast',
    #      '(20,20)': 'today', '(21,21)': 'he takes', '(22,22)': 'becomes', '(23,23)': 'any', '(24,24)': 'more'})
    map_of_words = {1: 'instruments',  2: 'craft',
    3: 'in contrast to',
    4: 'when a',
    5: 'accurate',
    6: 'pink noise',
    7: 'of science',
    8: 'rev. instruments',
    9: 'boats',
    10: 'today',
    13: 'sailor',
    15: 'but',
    16: 'researched',
    18: 'took',
    19: 'the vast',
    20: 'today',
    21: 'he takes',
    22: 'becomes',
    23: 'any',
    24: 'more'
    }
    #use the map of words to replace the xticks and yticks with the words
    xticks = pivot_df.columns
    yticks = pivot_df.index
    xticks = [map_of_words.get(xtick, xtick) for xtick in xticks]
    yticks = [map_of_words.get(ytick, ytick) for ytick in yticks]
    plt.xticks(ticks = np.arange(0.5, len(xticks), 1), labels = xticks, rotation = 45)
    plt.yticks(ticks = np.arange(0.5, len(yticks), 1), labels = yticks, rotation = 45)

    plt.savefig(f'G:/neural_chapter/figures/heatmap_dist_v_dist_trained_difference_from_perm_{trained}_{pitchshift_option}.png', dpi = 300)
    plt.show()
    return


def filter_for_units_used_in_first_analysis(data_in, trained = True):
    if trained == True:
        #filter for the units used in the first analysis
        #read the input CSV data
        decoding_scores = pd.read_csv('G:/neural_chapter/trained_animals_decoding_scores.csv')
    else:
        decoding_scores = pd.read_csv('G:/neural_chapter/naive_animals_decoding_scores.csv')
    #get the unique UNIT ids from the input data
    data_in['cluster_id_int'] = data_in['cluster_id'].astype(int)
    #round the cluster id to the nearest 1
    data_in['cluster_id'] = data_in['cluster_id_int'].round()
    data_in['long_unit_id'] = data_in['cluster_id'].astype(str)+'_' +data_in['animal']+'_'+ data_in['recname'] + '_' + data_in['stream']
    #filter so only units in decoding scores are in data_in
    data_in_filtered = data_in[data_in['long_unit_id'].isin(decoding_scores['ID'])]
    return data_in_filtered




if __name__ == '__main__':
    main()