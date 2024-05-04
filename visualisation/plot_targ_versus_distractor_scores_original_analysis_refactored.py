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
from helpers.vis_stats_helpers import run_mixed_effects_on_dataframe, run_anova_on_dataframe, create_gen_frac_variable, runlgbmmodel_score, create_gen_frac_and_index_variable

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
    sorted_df_of_scores = pd.DataFrame({'probeword1': [], 'pitchshift': [], 'cluster_id': [], 'score': [], 'unit_type': [], 'animal': [], 'stream': [], 'recname': [], 'clus_id_report': [], 'brain_area': []})
    for probeword in probewordlist:
        probewordindex = probeword[0]
        stringprobewordindex = str(probewordindex)
        singleunitlist_copy = singleunitlist.copy()
        multiunitlist_copy = multiunitlist.copy()

        #load the original clusters to split from the json file
        json_file_path = f'F:\split_cluster_jsons/{fullid}/cluster_split_list.json'
        if ferretname == 'orecchiette':
            original_to_split_cluster_ids = np.array([])
            try:
                scores = np.load(
                    saveDir + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_bs.npy',
                    allow_pickle=True)[()]
            except Exception as e:
                print(e)
                continue

        else:
            with open(json_file_path, "r") as json_file:
                loaded_data = json.load(json_file)
                recname = saveDir.split('/')[-3]
                stream_id = stream[-4:]

                if 'BB_3' in stream_id and ferretname != 'squinty':
                    side_of_implant = 'right'
                elif 'BB_2' in stream_id and ferretname != 'squinty':
                    side_of_implant = 'right'
                elif 'BB_4' in stream_id:
                    side_of_implant = 'left'
                elif 'BB_5' in stream_id:
                    side_of_implant = 'left'
                elif ferretname == 'Squinty':
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
                    stringprobewordindex = str(probeword[0])
                    try:
                        scores = np.load(
                        saveDir + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_bs.npy',
                        allow_pickle=True)[()]
                    except Exception as e:
                        print(e)
                        continue
                    original_to_split_cluster_ids = np.unique(scores['talker1']['target_vs_probe']['pitchshift']['cluster_id']+scores['talker1']['target_vs_probe']['nopitchshift']['cluster_id'])
                    #if all of them need splitting
                elif original_to_split_cluster_ids:
                    #TODO: not sure if this elif needed
                    #get all the unique clusters ids
                    stringprobewordindex = str(probeword[0])
                    try:

                        scores = np.load(
                            saveDir + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_bs.npy',
                            allow_pickle=True)[()]
                    except Exception as e:
                        print(e)
                        continue
                    original_to_split_cluster_ids = [x for x in original_to_split_cluster_ids if x < 100]

                elif original_to_split_cluster_ids == None or not original_to_split_cluster_ids:
                    original_to_split_cluster_ids = np.array([])
                    stringprobewordindex = str(probeword[0])
                    try:
                        scores = np.load(
                            saveDir + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_bs.npy',
                            allow_pickle=True)[()]
                    except Exception as e:
                        print(e)
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
                for cond in ['pitchshift', 'nopitchshift']:
                    for i, clus in enumerate(scores[f'talker{talker}'][comp][cond]['cluster_id']):
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
                            probeword1_input_text = probeword_map.get(probeword, probeword)

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
                            probeword1_input_text = probeword_map.get(probeword, probeword)
                        else:
                            probeword1_input_text = probeword


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
                                        {'probeword1': probeword1_input_text[0], 'pitchshift': cond,
                                         'cluster_id': clus,
                                         'score': scores[f'talker{talker}'][comp][cond][score_key][i],
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
                                        {'probeword1': probeword1_input_text[0], 'pitchshift': cond,
                                         'cluster_id': clus,
                                         'score': scores[f'talker{talker}'][comp][cond][score_key][i],
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
                                        {'probeword1': probeword1_input_text[0], 'pitchshift': cond,
                                         'cluster_id': clus,
                                         'score': scores[f'talker{talker}'][comp][cond][score_key][i],
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
                                    {'probeword1': probeword1_input_text[0],
                                     'cluster_id': clus,
                                     'score': scores[f'talker{talker}'][comp][cond][score_key][i], 'pitchshift': cond,
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
                if row[1] >= 3200 and report['d_prime'][i] > 4:
                    singleunitlist.append(i+1)
                elif row[1] >= 3200 and report['d_prime'][i] <= 4:
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
    probewordlist_zola = [ (2, 2), (5, 6), (42, 49), (32, 38), (20, 22)]
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
            path = Path('G:/F2003_Orecchiette/')
        else:
            path = Path('D:/ms4output_16102023/' + animal + '/')
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
                #make it lowercase
                animal_text = animal_text.lower()
                if animal =='F2003_Orecchiette':
                    rec_name_unique = stream
                else:
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
                if animal == 'F1604_Squinty':
                    df_instance = load_scores_and_filter(probewordlist_l74,
                                                                 saveDir=f'F:/results_13112023//{animal}/{rec_name_unique}/{streamtext}/',
                                                                 ferretname=animal_text,
                                                                 singleunitlist=singleunitlist[animal][stream],
                                                                 multiunitlist=multiunitlist[animal][stream],
                                                                 noiselist=noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], pitchshift_text=pitchshift_option)
                    df_all.append(df_instance)
                    df_instance_permutation = load_scores_and_filter(probewordlist_l74,
                                                                             saveDir=f'F:/results_13112023//{animal}/{rec_name_unique}/{streamtext}/',
                                                                             ferretname=animal_text,
                                                                             singleunitlist=singleunitlist[animal][stream],
                                                                             multiunitlist=multiunitlist[animal][stream],
                                                                             noiselist=noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], permutation_scores=True, pitchshift_text=pitchshift_option)
                    df_all_permutation.append(df_instance_permutation)

                elif animal == 'F1606_Windolene':
                    df_instance = load_scores_and_filter(probewordlist_l74,
                                                                 saveDir=f'F:/results_13112023//{animal}/{rec_name_unique}/{streamtext}/',
                                                                 ferretname=animal_text,
                                                                 singleunitlist=singleunitlist[animal][stream],
                                                                 multiunitlist=multiunitlist[animal][stream],
                                                                 noiselist=noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], pitchshift_text=pitchshift_option)
                    df_all = pd.concat([df_all, df_instance])

                    df_instance_permutation = load_scores_and_filter(probewordlist_l74,
                                                                             saveDir=f'F:/results_13112023//{animal}/{rec_name_unique}/{streamtext}/',
                                                                             ferretname=animal_text,
                                                                             singleunitlist=singleunitlist[animal][stream],
                                                                             multiunitlist=multiunitlist[animal][stream],
                                                                             noiselist=noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], permutation_scores=True, pitchshift_text=pitchshift_option)
                    df_all_permutation = pd.concat([df_all_permutation, df_instance_permutation])

                elif animal =='F1702_Zola':
                    df_instance = load_scores_and_filter(probewordlist_zola, saveDir=f'F:/results_13112023//{animal}/{rec_name_unique}/{streamtext}/',
                                                                 ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                 multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], pitchshift_text=pitchshift_option)
                    df_all.append(df_instance)

                    df_instance_permutation = load_scores_and_filter(probewordlist_zola, saveDir=f'F:/results_13112023//{animal}/{rec_name_unique}/{streamtext}/',
                                                                             ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                             multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream]
                                                                             , permutation_scores=True, pitchshift_text=pitchshift_option)
                    df_all_permutation = pd.concat([df_all_permutation, df_instance_permutation])

                elif animal == 'F1815_Cruella' or animal == 'F1902_Eclair':
                    df_instance = load_scores_and_filter(probewordlist, saveDir=f'F:/results_13112023//{animal}/{rec_name_unique}/{streamtext}/',
                                                                 ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                 multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], pitchshift_text=pitchshift_option)
                    df_all = pd.concat([df_all, df_instance])

                    df_instance_permutation = load_scores_and_filter(probewordlist, saveDir=f'F:/results_13112023//{animal}/{rec_name_unique}/{streamtext}/',
                                                                             ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                             multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream]
                                                                             , permutation_scores=True, pitchshift_text=pitchshift_option)
                    df_all_permutation = pd.concat([df_all_permutation, df_instance_permutation])

                elif animal == 'F2003_Orecchiette':
                    # try:
                    df_instance = load_scores_and_filter(probewordlist,
                                                                 saveDir=f'F:/results_13112023//{animal}/{rec_name_unique}/',
                                                                 ferretname=animal_text,
                                                                 singleunitlist=singleunitlist[animal][stream],
                                                                 multiunitlist=multiunitlist[animal][stream],
                                                                 noiselist=noiselist[animal][stream], stream=stream,
                                                                 fullid=animal,
                                                                 report=report[animal][stream], pitchshift_text=pitchshift_option)

                    df_all = pd.concat([df_all, df_instance])
                    df_instance_permutation = load_scores_and_filter(probewordlist,
                                                                             saveDir=f'F:/results_13112023//{animal}/{rec_name_unique}/',
                                                                             ferretname=animal_text,
                                                                             singleunitlist=singleunitlist[animal][stream],
                                                                             multiunitlist=multiunitlist[animal][stream],
                                                                             noiselist=noiselist[animal][stream], stream=stream,
                                                                             fullid=animal,
                                                                             report=report[animal][stream], permutation_scores=True, pitchshift_text=pitchshift_option)

                    df_all_permutation = pd.concat([df_all_permutation, df_instance_permutation])

                else:
                    df_instance = load_scores_and_filter(probewordlist, saveDir=f'F:/results_13112023//{animal}/{rec_name_unique}/{streamtext}/',
                                                                 ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                 multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], pitchshift_text=pitchshift_option)
                    df_all = pd.concat([df_all, df_instance])

                    df_instance_permutation = load_scores_and_filter(probewordlist, saveDir=f'F:/results_13112023//{animal}/{rec_name_unique}/{streamtext}/',
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
        #merge the information of df_all_trained_permutation and df_all_trained
        df_all_permutation_renamed = df_all_permutation.rename(columns={'score': 'score_permutation'})

        # Set 'cluster_id', 'stream', and 'recname' as index for both dataframes
        #make a unique column that is the cluster id, stream, and recnam

        # Create a new column 'unique_id' that concatenates 'cluster_id', 'stream', and 'recname'
        df_all['unique_id'] = df_all['cluster_id'].astype(str) + '_' + df_all['stream'].astype(str) + '_' + df_all[
            'recname'].astype(str) + '_' + df_all['probeword1'].astype(str) + '_' + df_all['pitchshift'].astype(str)

        df_all_permutation['unique_id'] = df_all_permutation['cluster_id'].astype(str) + '_' + df_all_permutation['stream'].astype(str) + '_' + df_all_permutation['recname'].astype(str) + '_' + df_all_permutation['probeword1'].astype(str) + '_' + df_all_permutation['pitchshift'].astype(str)


        # Set 'unique_id' as the index
        df_all_reindexed = df_all.set_index('unique_id')
        df_all_permutation_reindexed = df_all_permutation.set_index('unique_id')
        df_all_permutation_reindexed = df_all_permutation_reindexed.rename(columns={'score': 'score_permutation'})
        df_all_permutation_reindexed = df_all_permutation_reindexed.drop(columns = ['probeword1', 'pitchshift', 'cluster_id', 'stream', 'recname', 'unit_type', 'animal', 'clus_id_report', 'brain_area', 'tdt_electrode_num'])
        # Use join to merge the dataframes, just select the subset of score
        df_merged = df_all_reindexed.join(df_all_permutation_reindexed, how='left')
        #create a new column of naive based on the animal name
        df_merged['Naive'] = df_merged['animal'].apply(lambda x: False if x in ['F1604_Squinty', 'F1606_Windolene', 'F1702_Zola', 'F1815_Cruella'] else True)
        #make an ID column that combines the cluster id, stream, and recname
        df_merged['ID'] = df_merged['cluster_id'].astype(str) + '_' + df_merged['stream'].astype(str) + '_' + df_merged['recname'].astype(str)
        #also make a Below-chance column that is True if the score is below permutation score
        df_merged['Below-chance'] = df_merged['score'] < df_merged['score_permutation']
        #rename brain area to BrainArea
        df_merged = df_merged.rename(columns={'brain_area': 'BrainArea'})
        df_merged = df_merged.rename(columns={'score': 'Score'})
        df_merged = df_merged.rename(columns={'probeword1': 'ProbeWord'})
        df_merged = df_merged.rename(columns={'pitchshift': 'PitchShift'})
        df_merged['SingleUnit'] = df_merged['unit_type'].apply(lambda x: 1 if x == 'su' else 0)
        #if the value is nopitchshift replace with 0, if it is pitchshift replace with 1
        df_merged['PitchShift'] = df_merged['PitchShift'].apply(lambda x: 0 if x == 'nopitchshift' else 1)

    plot_major_analysis(df_merged)
    return df_merged
def color_by_probeword(probeword):
    return probe_word_palette[df_full_naive['ProbeWord'].unique().tolist().index(probeword)]

def plot_major_analysis(df_merged):
    # df_full represents trained animals, df_full_naive represents naive animals
    df_full = df_merged[df_merged['Naive'] == False]
    df_full_naive = df_merged[df_merged['Naive'] == True]
    for unit_id in df_full_naive['ID']:
        df_full_unit_naive = df_full_naive[df_full_naive['ID'].str.contains(unit_id)]
        # check if all the probe words are below chance
        if np.sum(df_full_unit_naive['Below-chance']) == len(df_full_unit_naive['Below-chance']):
            df_full_naive = df_full_naive[df_full_naive['ID'] != unit_id]
    #apply the same thing to the trained animals
    for unit_id in df_full['ID']:
        df_full_unit_trained = df_full[df_full['ID'].str.contains(unit_id)]
        # check if all the probe words are below chance
        if np.sum(df_full_unit_trained['Below-chance']) == len(df_full_unit_trained['Below-chance']):
            df_full = df_full[df_full['ID'] != unit_id]


    fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
    sns.stripplot(x='BrainArea', y='Score', hue='Below-chance', data=df_full_naive, ax=ax, alpha=0.5)
    sns.violinplot(x='BrainArea', y='Score', data=df_full_naive, ax=ax, inner=None, color='lightgray')
    plt.title('Naive animals')
    plt.savefig(f'G:/neural_chapter/figures/violinplot_by_area_score_naiveanimals_04052024.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a custom color palette for ProbeWords
    probe_word_palette = sns.color_palette("Set2", n_colors=len(df_full['ProbeWord'].unique()))

    # Define a function to apply different colors for ProbeWords

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
    plt.savefig(f'G:/neural_chapter/figures/violinplot_ofdecodingscores_bybrainarea_naiveanimals_04052024.png', dpi=300)
    # Show the plot
    plt.show()
    anova_table_naive, anova_model_naive = run_anova_on_dataframe(df_full_naive)
    # now plot by animal for the trained animals
    for animal in ['F1901_Crumble', 'F1902_Eclair',
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
        plt.savefig(f'G:/neural_chapter/figures/violinplot_by_area_score_naive_{animal}_04052024.png')

        # Show the plot
        plt.show()

    # export to csv the amount of units per animal
    df_of_neural_yield = pd.DataFrame(columns=['Animal', 'Number of single units', 'Number of multi units'])
    for animal in ['F1901_Crumble', 'F1902_Eclair', 'F1812_Nala', 'F2003_Orecchiette']:
        animal_units = df_full_naive[df_full_naive['ID'].str.contains(animal)]
        # drop all non unique units
        animal_units = animal_units.drop_duplicates(subset=['ID'])
        print(animal, len(animal_units))
        df_of_neural_yield = df_of_neural_yield.append(
            {'Animal': animal, 'Number of single units': np.sum(animal_units['SingleUnit']),
             'Number of multi units': len(animal_units) - np.sum(animal_units['SingleUnit'])}, ignore_index=True)
    df_of_neural_yield.to_csv('G:/neural_chapter/neural_yield_naive.csv')

    # export to csv the amount of units per animal
    df_of_neural_yield = pd.DataFrame(columns=['Animal', 'Number of single units', 'Number of multi units'])
    for animal in ['F1604_Squinty', 'F1606_Windolene', 'F1702_Zola', 'F1815_Cruella', ]:
        animal_units = df_full[df_full['ID'].str.contains(animal)]
        animal_units = animal_units.drop_duplicates(subset=['ID'])
        print(animal, len(animal_units))
        df_of_neural_yield = df_of_neural_yield.append(
            {'Animal': animal, 'Number of single units': np.sum(animal_units['SingleUnit']),
             'Number of multi units': len(animal_units) - np.sum(animal_units['SingleUnit'])}, ignore_index=True)
    df_of_neural_yield.to_csv('G:/neural_chapter/neural_yield_trained.csv')

    for unit_id in df_full_naive['ID']:
        df_full_unit_naive = df_full_naive[df_full_naive['ID'].str.contains(unit_id)]
        # check if all the probe words are below chance
        if np.sum(df_full_unit_naive['Below-chance']) == len(df_full_unit_naive['Below-chance']):
            df_full_naive = df_full_naive[df_full_naive['ID'] != unit_id]

    for unit_id in df_full['ID']:
        df_full_unit = df_full[df_full['ID'].str.contains(unit_id)]
        # check if all the probe words are below chance
        if np.sum(df_full_unit['Below-chance']) == len(df_full_unit['Below-chance']):
            df_full = df_full[df_full['ID'] != unit_id]

    df_of_neural_yield = pd.DataFrame(columns=['Animal', 'Number of single units', 'Number of multi units'])
    for animal in ['F1901_Crumble', 'F1902_Eclair', 'F1812_Nala', 'F2003_Orecchiette']:
        animal_units = df_full_naive[df_full_naive['ID'].str.contains(animal)]
        animal_units = animal_units.drop_duplicates(subset=['ID'])
        # save to csv the unit IDs

        print(animal, len(animal_units))
        df_of_neural_yield = df_of_neural_yield.append(
            {'Animal': animal, 'Number of single units': np.sum(animal_units['SingleUnit']),
             'Number of multi units': len(animal_units) - np.sum(animal_units['SingleUnit'])}, ignore_index=True)
    df_of_neural_yield.to_csv('G:/neural_chapter/neural_yield_naive_after_pruning.csv')
    df_full_naive.to_csv('G:/neural_chapter/naive_animals_decoding_scores.csv')

    df_of_neural_yield = pd.DataFrame(columns=['Animal', 'Number of single units', 'Number of multi units'])
    for animal in ['F1604_Squinty', 'F1606_Windolene', 'F1702_Zola', 'F1815_Cruella', ]:
        animal_units = df_full[df_full['ID'].str.contains(animal)]
        animal_units = animal_units.drop_duplicates(subset=['ID'])
        print(animal, len(animal_units))
        df_of_neural_yield = df_of_neural_yield.append(
            {'Animal': animal, 'Number of single units': np.sum(animal_units['SingleUnit']),
             'Number of multi units': len(animal_units) - np.sum(animal_units['SingleUnit'])}, ignore_index=True)
    df_of_neural_yield.to_csv('G:/neural_chapter/neural_yield_trained_after_pruning.csv')
    df_full.to_csv('G:/neural_chapter/trained_animals_decoding_scores.csv')

    ##export the high genfrac units
    df_full_pitchsplit_highsubset = create_gen_frac_and_index_variable(df_full, high_score_threshold=False,
                                                                       sixty_score_threshold=False,
                                                                       need_ps=True)

    df_full_pitchsplit_plot = df_full_pitchsplit_highsubset[df_full_pitchsplit_highsubset['GenFrac'].notna()]
    df_full_pitchsplit_plot = df_full_pitchsplit_plot.drop_duplicates(subset=['ID'])

    df_full_pitchsplit_csv_save = df_full_pitchsplit_highsubset[df_full_pitchsplit_highsubset['GenFrac'].notna()]
    df_full_pitchsplit_csv_save = df_full_pitchsplit_csv_save[df_full_pitchsplit_csv_save['GenIndex'] <= 0.2]

    # decoding score over all words vs generalisability

    # export the unit ids of the units that are in the top 25% of genfrac scores
    df_full_naive_pitchsplit_plot = create_gen_frac_and_index_variable(df_full_naive,
                                                                       high_score_threshold=False,
                                                                       sixty_score_threshold=False,
                                                                       need_ps=True)
    df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit_plot[df_full_naive_pitchsplit_plot['GenFrac'].notna()]
    df_full_pitchsplit_csv_naive_save = df_full_naive_pitchsplit_plot[df_full_naive_pitchsplit_plot['GenFrac'].notna()]
    df_full_pitchsplit_csv_naive_save = df_full_pitchsplit_csv_naive_save[
        df_full_pitchsplit_csv_naive_save['GenIndex'] <= 0.2]
    # reset the index
    df_full_pitchsplit_csv_naive_save = df_full_pitchsplit_csv_naive_save.reset_index(drop=True)
    df_full_pitchsplit_csv_save = df_full_pitchsplit_csv_save.reset_index(drop=True)

    df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit_plot.drop_duplicates(subset=['ID'])

    # only include units with genfrac scores less than 0.33
    df_full_pitchsplit_plot = df_full_pitchsplit_plot[df_full_pitchsplit_plot['GenIndex'] <= 0.2]
    df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit_plot[df_full_naive_pitchsplit_plot['GenIndex'] <= 0.2]
    # make sure the mean score is over 60%
    df_full_pitchsplit_plot = df_full_pitchsplit_plot[df_full_pitchsplit_plot['MeanScore'] >= 0.60]
    df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit_plot[df_full_naive_pitchsplit_plot['MeanScore'] >= 0.60]

    # add the unit ID and stream name to the dataframe
    for i in range(0, len(df_full_pitchsplit_csv_save)):
        full_id = df_full_pitchsplit_csv_save.iloc[i]['ID']
        components = full_id.split('_')
        unit_id = components[0]
        # remove the unit id from the full_id for the rec_name
        rec_name = components[3:-2]
        # concatenate the rec_name
        rec_name = '_'.join(rec_name)
        stream = full_id[-4:]
        # append to a dataframe
        df_full_pitchsplit_csv_save.at[i, 'ID_small'] = unit_id
        # df_full_pitchsplit_csv_save.at[i, 'rec_name'] = rec_name
        # df_full_pitchsplit_csv_save.at[i, 'stream'] = stream
    for i in range(0, len(df_full_pitchsplit_csv_naive_save)):
        full_id = df_full_pitchsplit_csv_naive_save.iloc[i]['ID']
        components = full_id.split('_')
        unit_id = components[0]
        # remove the unit id from the full_id for the rec_name
        rec_name = components[3:-2]
        # # concatenate the rec_name
        # rec_name = '_'.join(rec_name)
        # stream = full_id[-4:]
        # append to a dataframe#
        df_full_pitchsplit_csv_naive_save.at[i, 'ID_small'] = unit_id
        # df_full_pitchsplit_csv_naive_save.at[i, 'rec_name'] = rec_name
        # df_full_pitchsplit_csv_naive_save.at[i, 'stream'] = stream

    # order df_full_pitchsplit_csv_save by score
    df_full_pitchsplit_csv_save = df_full_pitchsplit_csv_save.sort_values(by='MeanScore', ascending=False)
    df_full_pitchsplit_csv_naive_save = df_full_pitchsplit_csv_naive_save.sort_values(by='MeanScore', ascending=False)

    df_full_pitchsplit_csv_save.to_csv('G:/neural_chapter/csvs/units_topgenindex_allanimalstrained.csv')
    df_full_pitchsplit_csv_naive_save.to_csv('G:/neural_chapter/csvs/units_topgenindex_allanimalsnaive.csv')

    # export the unit IDs

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='GenFrac', y='MaxScore', data=df_full_pitchsplit_plot, ax=ax, color='purple')
    sns.regplot(x='GenFrac', y='MaxScore', data=df_full_pitchsplit_plot, ax=ax, color='purple')
    # get the r2 value
    x = df_full_pitchsplit_plot['GenFrac']
    y = df_full_pitchsplit_plot['MaxScore']
    r2 = stats.pearsonr(x, y)[0] ** 2
    # put it on the plot
    plt.text(0.05, 0.95, f'r2 = {r2}', transform=ax.transAxes)
    plt.title('Trained animals'' max score over generalization  frac')
    plt.xlabel('Generalization fraction')
    plt.show()

    # decoding score over all words vs generalisability, UNCOMMENT 04052024
    # fig2, ax2 = plt.subplots(figsize=(10, 6))
    # sns.scatterplot(x='GenFrac', y='MaxScore', data=df_full_naive_pitchsplit_plot, ax=ax2, color = 'darkcyan')
    # sns.regplot(x='GenFrac', y='MaxScore', data=df_full_naive_pitchsplit_plot, ax=ax2, color = 'darkcyan')
    # #get the r2 value
    # x = df_full_naive_pitchsplit_plot['GenFrac']
    # y = df_full_naive_pitchsplit_plot['MaxScore']
    # r2 = stats.pearsonr(x, y)[0] ** 2
    # plt.text(0.05, 0.95, f'r2 = {r2}', transform=ax2.transAxes)
    # plt.title('Naive animals'' max score over generalization  frac')
    # plt.xlabel('Generalization frac')
    # plt.show()

    for animal in ['F1815_Cruella', 'F1702_Zola', 'F1604_Squinty', 'F1606_Windolene']:
        # isolate the data for this animal
        df_full_pitchsplit_plot_animal = df_full_pitchsplit_plot[df_full_pitchsplit_plot['ID'].str.contains(animal)]
        # export the unit IDs for this animal
        animal_dataframe = pd.DataFrame(columns=['ID', 'rec_name', 'stream', 'BrainArea', 'GenScore', 'MeanScore'])
        for i in range(0, len(df_full_pitchsplit_plot_animal)):
            full_id = df_full_pitchsplit_plot_animal.iloc[i]['ID']
            components = full_id.split('_')
            unit_id = components[0]
            # remove the unit id from the full_id for the rec_name
            rec_name = components[3:-2]
            # concatenate the rec_name
            rec_name = df_full_pitchsplit_plot_animal.iloc[i]['recname']
            stream = df_full_pitchsplit_plot_animal.iloc[i]['stream']
            genfrac = df_full_pitchsplit_plot_animal.iloc[i]['GenFrac']
            brainarea = df_full_pitchsplit_plot_animal.iloc[i]['BrainArea']
            # append to a dataframe
            animal_dataframe = animal_dataframe.append(
                {'ID': unit_id, 'rec_name': rec_name, 'stream': stream, 'BrainArea': brainarea, 'GenScore': genfrac,
                 'MeanScore': df_full_pitchsplit_plot_animal.iloc[i]['MeanScore']},
                ignore_index=True)
        # export the dataframe to csv
        animal_dataframe.to_csv(f'G:/neural_chapter/figures/unit_ids_trained_topgenindex_{animal}.csv')
    for animal in ['F1815_Cruella', 'F1702_Zola', 'F1604_Squinty', 'F1606_Windolene']:
        # isolate the data for this animal
        df_full_pitchsplit_plot_animal = df_full[df_full['ID'].str.contains(animal)]
        # export the unit IDs for this animal
        animal_dataframe = pd.DataFrame(columns=['ID', 'rec_name', 'stream', 'BrainArea', 'Score'])
        for i in range(0, len(df_full_pitchsplit_plot_animal)):
            full_id = df_full_pitchsplit_plot_animal.iloc[i]['ID']
            components = full_id.split('_')
            unit_id = components[0]
            # remove the unit id from the full_id for the rec_name
            rec_name = df_full_pitchsplit_plot_animal.iloc[i]['recname']
            # concatenate the rec_name
            # rec_name = '_'.join(rec_name)
            stream = df_full_pitchsplit_plot_animal.iloc[i]['stream']
            # genfrac = df_full_pitchsplit_plot_animal.iloc[i]['GenFrac']
            brainarea = df_full_pitchsplit_plot_animal.iloc[i]['BrainArea']
            # append to a dataframe
            animal_dataframe = animal_dataframe.append(
                {'ID': unit_id, 'rec_name': rec_name, 'stream': stream, 'BrainArea': brainarea,
                 'Score': df_full_pitchsplit_plot_animal.iloc[i]['Score']},
                ignore_index=True)
        # export the dataframe to csv
        animal_dataframe.to_csv(f'G:/neural_chapter/csvs/unit_ids_trained_all_{animal}.csv')
    # get the ratio of highgenindex to low genindex units
    ratio_trained = len(df_full_pitchsplit_plot) / len(df_full)
    ratio_naive = len(df_full_naive_pitchsplit_plot) / len(df_full_naive)
    # make a dataframe and export to csv
    df_ratio = pd.DataFrame(columns=['Animal', 'Ratio'])
    df_ratio = df_ratio.append({'Animal': 'Trained', 'Ratio': ratio_trained}, ignore_index=True)
    df_ratio = df_ratio.append({'Animal': 'Naive', 'Ratio': ratio_naive}, ignore_index=True)
    df_ratio.to_csv('G:/neural_chapter/csvs/ratio_highgenindex_lowgenindex.csv')
    for animal in ['F1902_Eclair', 'F1901_Crumble', 'F1812_Nala', 'F2003_Orecchiette']:
        # isolate the data for this animal
        df_full_pitchsplit_plot_animal = df_full_naive[
            df_full_naive['ID'].str.contains(animal)]
        # export the unit IDs for this animal
        animal_dataframe = pd.DataFrame(columns=['ID', 'rec_name', 'stream', 'Score'])
        for i in range(0, len(df_full_pitchsplit_plot_animal)):
            full_id = df_full_pitchsplit_plot_animal.iloc[i]['ID']
            components = full_id.split('_')
            unit_id = components[0]
            # remove the unit id from the full_id for the rec_name
            rec_name = df_full_pitchsplit_plot_animal.iloc[i]['recname']
            # concatenate the rec_name
            # rec_name = '_'.join(rec_name)
            stream = df_full_pitchsplit_plot_animal.iloc[i]['stream']
            # append to a dataframe
            brainarea = df_full_pitchsplit_plot_animal.iloc[i]['BrainArea']

            animal_dataframe = animal_dataframe.append(
                {'ID': unit_id, 'rec_name': rec_name, 'stream': stream, 'BrainArea': brainarea,
                 'Score': df_full_pitchsplit_plot_animal.iloc[i]['Score']},
                ignore_index=True)
        # export the dataframe to csv
        animal_dataframe.to_csv(f'G:/neural_chapter/csvs/unit_ids_all_naive_{animal}.csv')
    for animal in ['F1902_Eclair', 'F1901_Crumble', 'F1812_Nala', 'F2003_Orecchiette']:
        # isolate the data for this animal
        df_full_pitchsplit_plot_animal = df_full_naive_pitchsplit_plot[
            df_full_naive_pitchsplit_plot['ID'].str.contains(animal)]
        # export the unit IDs for this animal
        animal_dataframe = pd.DataFrame(columns=['ID', 'rec_name', 'stream', 'BrainArea', 'GenScore', 'MeanScore'])
        for i in range(0, len(df_full_pitchsplit_plot_animal)):
            full_id = df_full_pitchsplit_plot_animal.iloc[i]['ID']
            components = full_id.split('_')
            unit_id = components[0]
            # remove the unit id from the full_id for the rec_name
            # concatenate the rec_name
            rec_name = df_full_pitchsplit_plot_animal.iloc[i]['recname']
            stream = df_full_pitchsplit_plot_animal.iloc[i]['stream']
            # append to a dataframe
            genfrac = df_full_pitchsplit_plot_animal.iloc[i]['GenFrac']
            animal_dataframe = animal_dataframe.append(
                {'ID': unit_id, 'rec_name': rec_name, 'stream': stream, 'BrainArea': brainarea, 'GenScore': genfrac,
                 'MeanScore': df_full_pitchsplit_plot_animal.iloc[i]['MeanScore']},
                ignore_index=True)
        # export the dataframe to csv
        animal_dataframe.to_csv(f'G:/neural_chapter/csvs/unit_ids_trained_topgenindex_{animal}.csv')
        ##do the roved - control f0 score divided by the control f0 score plot
        # first get the data into a format that can be analysed
    rel_frac_list_naive = []
    bigconcatenatenaive_ps = []
    bigconcatenatenaive_nonps = []
    bigconcatenatetrained_nonps = []
    bigconcatenatetrained_ps = []
    rel_frac_list_trained = []
    for unit_id in df_full_naive['ID']:
        df_full_unit_naive = df_full_naive[df_full_naive['ID'] == unit_id]
        # get all the scores where pitchshift is 1 for each probe word
        for probeword in df_full_unit_naive['ProbeWord'].unique():
            try:
                control_df = df_full_unit_naive[
                    (df_full_unit_naive['ProbeWord'] == probeword) & (df_full_unit_naive['PitchShift'] == 0) & (
                            df_full_unit_naive['Below-chance'] == 0)]
                roved_df = df_full_unit_naive[
                    (df_full_unit_naive['ProbeWord'] == probeword) & (df_full_unit_naive['PitchShift'] == 1) & (
                            df_full_unit_naive['Below-chance'] == 0)]
                if len(control_df) == 0 and len(roved_df) == 0:
                    continue

                control_score = df_full_unit_naive[
                    (df_full_unit_naive['ProbeWord'] == probeword) & (df_full_unit_naive['PitchShift'] == 0)][
                    'Score'].values[0]
                pitchshift_score = df_full_unit_naive[
                    (df_full_unit_naive['ProbeWord'] == probeword) & (df_full_unit_naive['PitchShift'] == 1)][
                    'Score'].values[0]
            except:
                continue
            if control_score is not None and pitchshift_score is not None:
                rel_score = (pitchshift_score - control_score) / control_score
                rel_frac_list_naive.append(rel_score)
                bigconcatenatenaive_ps.append(pitchshift_score)
                bigconcatenatenaive_nonps.append(control_score)
    for unit_id in df_full['ID']:
        df_full_unit = df_full[df_full['ID'] == unit_id]
        # get all the scores where pitchshift is 1 for the each probe word
        for probeword in df_full_unit['ProbeWord'].unique():
            try:
                control_df = df_full_unit[
                    (df_full_unit['ProbeWord'] == probeword) & (df_full_unit['PitchShift'] == 0) & (
                            df_full_unit['Below-chance'] == 0)]
                roved_df = df_full_unit[
                    (df_full_unit['ProbeWord'] == probeword) & (df_full_unit['PitchShift'] == 1) & (
                            df_full_unit['Below-chance'] == 0)]
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
        # check if all the probe words are below chance
    fig, ax = plt.subplots(1, figsize=(10, 10), dpi=300)
    # sns.distplot(rel_frac_list_trained, bins=20, label='trained', ax=ax, color='purple')
    # sns.distplot(rel_frac_list_naive, bins=20, label='naive', ax=ax, color='darkcyan')
    sns.histplot(rel_frac_list_trained, label='trained', color='purple', ax=ax, kde=True)
    sns.histplot(rel_frac_list_naive, label='naive', color='darkcyan', ax=ax, kde=True)

    # get the peak of the distribution on the y axis

    # get the peak of the distribution on the y axis
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
    plt.savefig('G:/neural_chapter/figures/diffF0distribution_04052024_04052024.png', dpi=1000)
    plt.show()
    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=300)
    ax.scatter(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, marker='P', color='purple', alpha=0.5,
               label='trained')
    ax.scatter(bigconcatenatenaive_nonps, bigconcatenatenaive_ps, marker='P', color='darkcyan', alpha=0.5,
               label='naive')
    x = np.linspace(0.4, 1, 101)
    ax.plot(x, x, color='black', linestyle='--')  # identity line

    slope, intercept, r_value, pv, se = stats.linregress(bigconcatenatetrained_nonps, bigconcatenatetrained_ps)

    sns.regplot(x=bigconcatenatetrained_nonps, y=bigconcatenatetrained_ps, scatter=False, color='purple',
                label=' $y=%3.7s*x+%3.7s$' % (slope, intercept), ax=ax,
                line_kws={'label': ' $y=%3.7s*x+%3.7s$' % (slope, intercept)})
    slope, intercept, r_value, pv, se = stats.linregress(bigconcatenatenaive_nonps, bigconcatenatenaive_ps)

    sns.regplot(x=bigconcatenatenaive_nonps, y=bigconcatenatenaive_ps, scatter=False, color='darkcyan',
                label=' $y=%3.7s*x+%3.7s$' % (slope, intercept),
                ax=ax, line_kws={'label': '$y=%3.7s*x+%3.7s$' % (slope, intercept)})

    ax.set_ylabel('LSTM decoding score, F0 roved', fontsize=30)
    ax.set_xlabel('LSTM decoding score, F0 control', fontsize=30)

    ax.set_title('LSTM decoder scores for' + ' F0 control vs. roved,\n ' + ' trained and naive animals', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(fontsize=15, ncol=2)
    fig.tight_layout()
    plt.savefig('G:/neural_chapter/figures/scattermuaandsuregplot_mod_21062023_04052024.png', dpi=1000,
                bbox_inches='tight')
    plt.savefig('G:/neural_chapter/figures/scattermuaandsuregplot_mod_21062023.pdf', dpi=1000, bbox_inches='tight')
    plt.show()

    unique_unit_ids_naive = df_full_naive['ID'].unique()
    unique_unit_ids_trained = df_full['ID'].unique()
    # make a kde plot
    # makea  dataframe
    df_naive = pd.DataFrame({'F0_control': bigconcatenatenaive_nonps, 'F0_roved': bigconcatenatenaive_ps})
    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=300)
    sns.kdeplot(df_naive, x='F0_control', y='F0_roved', shade=True, shade_lowest=False, ax=ax, label='naive')
    plt.xlim([0.25, 0.95])

    plt.xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('F0 control vs. roved, naive animals', fontsize=30)
    plt.ylabel('F0 roved score', fontsize=30)
    plt.xlabel('F0 control score', fontsize=30)
    plt.savefig('G:/neural_chapter/figures/kdeplot_naiveanimals_2_04052024.png', dpi=300, bbox_inches='tight')
    plt.show()

    df_trained_kde = pd.DataFrame({'F0_control': bigconcatenatetrained_nonps, 'F0_roved': bigconcatenatetrained_ps})
    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=300)
    sns.kdeplot(df_trained_kde, x='F0_control', y='F0_roved', cmap="Reds", shade=True, shade_lowest=False, ax=ax,
                label='trained')
    plt.xlim([0.25, 0.95])
    plt.xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('F0 control vs. roved, trained animals', fontsize=30)
    plt.ylabel('F0 roved score', fontsize=30)
    plt.xlabel('F0 control score', fontsize=30)
    plt.savefig('G:/neural_chapter/figures/kdeplot_trainedanimals_2_04052024.png', dpi=300, bbox_inches='tight')
    plt.show()
    manwhitscorecontrolf0 = mannwhitneyu(bigconcatenatetrained_nonps, bigconcatenatenaive_nonps, alternative='greater')

    n1 = len(bigconcatenatetrained_nonps)
    n2 = len(bigconcatenatenaive_nonps)
    r_controlf0 = 1 - (2 * manwhitscorecontrolf0.statistic) / (n1 * n2)
    # ax.legend()
    plt.savefig('G:/neural_chapter/figures/controlF0distribution20062023intertrialroving_04052024.png', dpi=1000)
    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=800)
    ax.set_xlim([0, 1])
    sns.distplot(bigconcatenatetrained_ps, label='trained', ax=ax, color='purple')
    sns.distplot(bigconcatenatenaive_ps, label='naive', ax=ax, color='darkcyan')
    plt.xlim([0.25, 0.95])
    plt.xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
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
    plt.savefig('G:/neural_chapter/figures/rovedF0distribution_20062023intertrialroving_04052024.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=800)
    sns.distplot(bigconcatenatetrained_ps, label='trained roved', ax=ax, color='purple')
    sns.distplot(bigconcatenatetrained_nonps, label='trained control', ax=ax, color='magenta')
    ax.legend(fontsize=25)
    plt.title('Roved and Control F0 Distributions \n for the Trained Animals', fontsize=30)
    plt.xlabel(' LSTM decoder scores', fontsize=30)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=20)
    plt.yticks(fontsize=20)
    # ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=20)
    # ax.set_yticklabels(labels = ax.get_yticklabels(), fontsize=20)
    plt.ylabel('Density', fontsize=30)
    plt.xlim([0.35, 1])

    plt.savefig('G:/neural_chapter/figures/rovedF0vscontrolF0traineddistribution_20062023intertrialroving_04052024.png',
                dpi=400, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=800)
    ax.set_xlim([0, 1])
    sns.distplot(bigconcatenatenaive_ps, label='naive roved', ax=ax, color='darkcyan')
    sns.distplot(bigconcatenatenaive_nonps, label='naive control', ax=ax, color='cyan')
    plt.xlim([0.35, 1])

    ax.legend(fontsize=25)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=20)
    ax.set_yticks([5, 10, 15, 20, 25], labels=[5, 10, 15, 20, 25], fontsize=20)
    plt.xlabel(' LSTM decoder scores', fontsize=30)
    plt.title('Roved and Control F0 Distributions \n for the Naive Animals', fontsize=30)
    plt.ylabel('Density', fontsize=30)
    plt.savefig('G:/neural_chapter/figures/rovedF0vscontrolF0naivedistribution_20062023intertrialroving_04052024.png',
                dpi=1000)
    plt.show()
    kstestcontrolf0vsrovedtrained = scipy.stats.kstest(bigconcatenatetrained_nonps, bigconcatenatetrained_ps,
                                                       alternative='greater')

    # do levene's test
    leveneteststat = scipy.stats.levene(bigconcatenatetrained_nonps, bigconcatenatetrained_ps)
    kstestcontrolf0vsrovednaive = scipy.stats.kstest(bigconcatenatenaive_nonps, bigconcatenatenaive_ps,
                                                     alternative='greater')
    # do a wilcoxon signed rank test
    wilcoxon_stat = scipy.stats.wilcoxon(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, alternative='two-sided',
                                         method='approx')

    # do a wilcoxon signed rank test
    wilcoxon_statnaive = scipy.stats.wilcoxon(bigconcatenatenaive_nonps, bigconcatenatenaive_ps, alternative='greater',
                                              method='approx')
    # export bigconcatenatenaive_nonps and bigconcatenatenaive_ps to csv
    df_naive = pd.DataFrame({'F0_control': bigconcatenatenaive_nonps, 'F0_roved': bigconcatenatenaive_ps})
    df_naive.to_csv('G:/neural_chapter/figures/rovedF0vscontrolF0naivedistribution_20062023intertrialroving.csv')
    df_trained = pd.DataFrame({'F0_control': bigconcatenatetrained_nonps, 'F0_roved': bigconcatenatetrained_ps})
    df_trained.to_csv('G:/neural_chapter/figures/rovedF0vscontrolF0traineddistribution_20062023intertrialroving.csv')

    z = wilcoxon_stat.zstatistic
    n = len(bigconcatenatetrained_nonps)
    effect_size_trained = z / np.sqrt(2 * n)
    print(effect_size_trained)

    # calculate the effect size
    # effect size = z/sqrt(n)
    z = wilcoxon_statnaive.zstatistic
    n = len(bigconcatenatenaive_nonps)
    effect_size_naive = z / np.sqrt(2 * n)

    # Calculating Cramér's V for effect size
    def cramers_v(n, ks_statistic):
        return np.sqrt(ks_statistic / n)

    n = len(bigconcatenatenaive_nonps) * len(bigconcatenatenaive_ps) / (
            len(bigconcatenatenaive_nonps) + len(bigconcatenatenaive_ps))
    n = len(bigconcatenatenaive_nonps) + len(bigconcatenatenaive_ps)
    # effect_size_naive = cramers_v(n, kstestcontrolf0vsrovednaive.statistic)
    k = 2
    eta_squared_naive = (kstestcontrolf0vsrovednaive.statistic - k + 1) / (n - k)
    n = len(bigconcatenatetrained_nonps) + len(bigconcatenatetrained_ps)
    eta_squared_trained = (kstestcontrolf0vsrovedtrained.statistic - k + 1) / (n - k)

    n_trained = len(bigconcatenatetrained_nonps) * len(bigconcatenatetrained_ps) / (
            len(bigconcatenatetrained_nonps) + len(bigconcatenatetrained_ps))
    effect_size_trained = cramers_v(n_trained, kstestcontrolf0vsrovedtrained.statistic)

    # run mann whitney u test
    manwhitscore_stat, manwhitescore_pvalue = mannwhitneyu(bigconcatenatetrained_nonps, bigconcatenatetrained_ps,
                                                           alternative='greater')
    manwhitscore_statnaive, manwhitescore_pvaluenaive = mannwhitneyu(bigconcatenatenaive_nonps, bigconcatenatenaive_ps,
                                                                     alternative='two-sided')
    manwhitscore_statnaive2, manwhitescore_pvaluenaive2 = mannwhitneyu(bigconcatenatenaive_nonps,
                                                                       bigconcatenatenaive_ps,
                                                                       alternative='greater')

    # Calculate rank-biserial correlation coefficient

    n1 = len(bigconcatenatetrained_nonps)
    n2 = len(bigconcatenatetrained_ps)
    r = 1 - (2 * manwhitscore_stat) / (n1 * n2)

    n1 = len(bigconcatenatenaive_nonps)
    n2 = len(bigconcatenatenaive_ps)
    r_naive = 1 - (2 * manwhitscore_statnaive) / (n1 * n2)
    # put these stats into a table and export to csv
    # create a dataframe
    dataframe_stats = pd.DataFrame({'effect sizes r value': [r_controlf0, r_rovef0, r, r_naive],
                                    'trained animals p value': [manwhitscorecontrolf0.pvalue,
                                                                manwhitscorerovedf0.pvalue, manwhitescore_pvalue,
                                                                manwhitescore_pvaluenaive]},
                                   index=['control naive vs. trained(alt = trained > naive) ',
                                          'roved naive vs trained (alt = trained > naive)',
                                          'control vs. roved trained (two sided)',
                                          'control naive vs. roved naive (two sided)'])

    # export to csv
    dataframe_stats.to_csv(
        'G:/neural_chapter/figures/stats_16112023_comparingdistributions_generalintertrialroving.csv')

    for options in ['index', 'frac']:
        df_full_pitchsplit_highsubset = create_gen_frac_variable(df_full, high_score_threshold=True,
                                                                 index_or_frac=options)
        # remove all rows where GenFrac is nan
        df_full_pitchsplit_plot = df_full_pitchsplit_highsubset[df_full_pitchsplit_highsubset['GenFrac'].notna()]
        df_full_pitchsplit_plot = df_full_pitchsplit_plot.drop_duplicates(subset=['ID'])
        # export the unit ids of the units that are in the top 25% of genfrac scores
        df_full_pitchsplit_plot.to_csv(f'G:/neural_chapter/figures/unit_ids_trained_highthreshold_{options}.csv')
        df_full_naive_pitchsplit_plot = create_gen_frac_variable(df_full_naive, high_score_threshold=True,
                                                                 index_or_frac=options)
        df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit_plot[df_full_naive_pitchsplit_plot['GenFrac'].notna()]
        df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit_plot.drop_duplicates(subset=['ID'])
        df_full_naive_pitchsplit_plot.to_csv(f'G:/neural_chapter/figures/unit_ids_naive_highthreshold_{options}.csv')
        for animal in ['F1815_Cruella', 'F1702_Zola', 'F1604_Squinty', 'F1606_Windolene']:
            # isolate the data for this animal
            df_full_pitchsplit_plot_animal = df_full_pitchsplit_plot[df_full_pitchsplit_plot['ID'].str.contains(animal)]
            # export the unit IDs for this animal
            animal_dataframe = pd.DataFrame(columns=['ID', 'rec_name', 'stream', 'BrainArea'])
            for i in range(0, len(df_full_pitchsplit_plot_animal)):
                full_id = df_full_pitchsplit_plot_animal.iloc[i]['ID']
                components = full_id.split('_')
                unit_id = components[0]
                # remove the unit id from the full_id for the rec_name
                rec_name = components[3:-2]
                # concatenate the rec_name
                rec_name = '_'.join(rec_name)
                stream = full_id[-4:]
                brainarea = df_full_pitchsplit_plot_animal.iloc[i]['BrainArea']
                # append to a dataframe
                animal_dataframe = animal_dataframe.append(
                    {'ID': unit_id, 'rec_name': rec_name, 'stream': stream, 'BrainArea': brainarea},
                    ignore_index=True)
            # export the dataframe to csv
            animal_dataframe.to_csv(f'G:/neural_chapter/figures/unit_ids_trained_highthreshold_{options}_{animal}.csv')

        for animal in ['F1902_Eclair', 'F1901_Crumble', 'F1812_Nala', 'F2003_Orecchiette']:
            # isolate the data for this animal
            df_full_pitchsplit_plot_animal = df_full_naive_pitchsplit_plot[
                df_full_naive_pitchsplit_plot['ID'].str.contains(animal)]
            # export the unit IDs for this animal
            animal_dataframe = pd.DataFrame(columns=['ID', 'rec_name', 'stream', 'BrainArea'])
            for i in range(0, len(df_full_pitchsplit_plot_animal)):
                full_id = df_full_pitchsplit_plot_animal.iloc[i]['ID']
                components = full_id.split('_')
                unit_id = components[0]
                # remove the unit id from the full_id for the rec_name
                rec_name = components[3:-2]
                # concatenate the rec_name
                rec_name = '_'.join(rec_name)
                stream = full_id[-4:]
                # append to a dataframe
                animal_dataframe = animal_dataframe.append(
                    {'ID': unit_id, 'rec_name': rec_name, 'stream': stream, 'BrainArea': brainarea},
                    ignore_index=True)
            # export the dataframe to csv
            animal_dataframe.to_csv(f'G:/neural_chapter/figures/unit_ids_trained_highthreshold_{options}_{animal}.csv')

        # plot the distplot of these scores overlaid with the histogram
        fig, ax = plt.subplots(1, dpi=300)
        sns.histplot(df_full_pitchsplit_plot['GenFrac'], ax=ax, kde=True, bins=20, color='purple', label='Trained')
        sns.histplot(df_full_naive_pitchsplit_plot['GenFrac'], ax=ax, kde=True, bins=20, color='cyan', label='Naive')
        plt.legend()
        # plt.title(f'Distribution of generalizability scores for the trained and naive animals, upper quartile threshold, index or frac:{options}')
        if options == 'index':
            plt.xlabel('Generalizability Index of Top 25% of Units', fontsize=20)
        elif options == 'frac':
            plt.xlabel('Generalizability Fraction of Top 25% of Units', fontsize=20)
            plt.xlim(0, 1.2)
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], labels=[0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.ylabel('Count', fontsize=20)
        plt.savefig(f'G:/neural_chapter/figures/GenFrac_highthreshold_{options}_04052024.png')
        plt.show()

        # plot as a violin plot with brainarea on the x axis
        fig, ax = plt.subplots(1, dpi=300)
        sns.violinplot(x='BrainArea', y='GenFrac', data=df_full_pitchsplit_plot, ax=ax, inner=None, color='lightgray')
        sns.stripplot(x='BrainArea', y='GenFrac', data=df_full_pitchsplit_plot, ax=ax, size=3, color='purple',
                      dodge=False)
        # plt.title(f'Generalizability scores for the trained animals, upper quartile threshold, index or frac:{options}')
        if options == 'index':
            plt.xlabel('Generalizability Index of Top 25% of Units', fontsize=20)
        elif options == 'frac':
            plt.xlabel('Generalizability Fraction of Top 25% of Units', fontsize=20)
            plt.xlim(0, 1)

        plt.ylabel('Count', fontsize=20)
        plt.savefig(f'G:/neural_chapter/figures/GenFrac_highthreshold_violin_{options}_04052024.png')
        plt.show()

        fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
        sns.violinplot(x='BrainArea', y='GenFrac', data=df_full_naive_pitchsplit_plot, ax=ax, inner=None,
                       color='lightgray')
        sns.stripplot(x='BrainArea', y='GenFrac', data=df_full_naive_pitchsplit_plot, ax=ax, size=3, dodge=False)
        # plt.title(f'Generalizability scores for the naive animals, 60% mean score threshold, index or frac:{options}')
        if options == 'index':
            plt.xlabel('Generalizability Index of Top 25% of Units', fontsize=20)
        elif options == 'frac':
            plt.xlabel('Generalizability Fraction of Top 25% of Units', fontsize=20)
        plt.savefig(f'G:/neural_chapter/figures/GenFrac_highthreshold_violin_naive_{options}_04052024.png')
        plt.show()

        # do the mann whitney u test between genfrac scores from PEG and MEG
        df_full_pitchsplit_plot_peg = df_full_pitchsplit_plot[df_full_pitchsplit_plot['BrainArea'] == 'PEG']
        df_full_pitchsplit_plot_meg = df_full_pitchsplit_plot[df_full_pitchsplit_plot['BrainArea'] == 'MEG']
        df_full_naive_pitchsplit_plot_peg = df_full_naive_pitchsplit_plot[
            df_full_naive_pitchsplit_plot['BrainArea'] == 'PEG']
        df_full_naive_pitchsplit_plot_meg = df_full_naive_pitchsplit_plot[
            df_full_naive_pitchsplit_plot['BrainArea'] == 'MEG']

        stat_peg, p_peg = mannwhitneyu(df_full_pitchsplit_plot_peg['GenFrac'], df_full_pitchsplit_plot_meg['GenFrac'],
                                       alternative='less')
        # plot the brain area loation of the units that are significantly different
        fig, ax = plt.subplots(1, dpi=300)
        sns.violinplot(x='BrainArea', y='GenFrac', data=df_full_pitchsplit_plot, ax=ax, inner=None, color='lightgray')
        sns.stripplot(x='BrainArea', y='GenFrac', data=df_full_pitchsplit_plot, ax=ax, color='purple', dodge=False)
        plt.title(f'Generalizability scores for the trained animals, upper quartile threshold, index or frac:{options}')
        if options == 'index':

            plt.xlabel('Generalizability Index of Top 25% of Units', fontsize=20)
        elif options == 'frac':

            plt.xlabel('Generalizability Fraction of Top 25% of Units', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.savefig(f'G:/neural_chapter/figures/GenFrac_highthreshold_violin_bybrainarea_{options}_04052024.png')

        fig, ax = plt.subplots(1, dpi=300)
        sns.violinplot(x='BrainArea', y='MeanScore', data=df_full_pitchsplit_plot, ax=ax, inner=None, label=None,
                       color='lightgray')
        sns.stripplot(x='BrainArea', y='MeanScore', data=df_full_pitchsplit_plot, color='purple', label='trained',
                      ax=ax, dodge=False)
        # plot the naive data adjacent, shift the x axis by 0.2
        x_shift = 4
        # remove AEG values from df_full_naive_pitchsplit_plot
        df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit_plot[
            df_full_naive_pitchsplit_plot['BrainArea'] != 'AEG']
        sns.violinplot(x='BrainArea', y='MeanScore', data=df_full_naive_pitchsplit_plot, ax=ax, inner=None, label=None,
                       color='lightgray',
                       position=[p + x_shift for p in range(len(df_full_naive_pitchsplit_plot['BrainArea'].unique()))])
        # Apply the x-shift manually to the strip plot points for the second set of data
        sns.stripplot(x='BrainArea', y='MeanScore', data=df_full_naive_pitchsplit_plot, ax=ax, color='cyan',
                      label='naive', dodge=False, jitter=True, linewidth=1, edgecolor='gray', marker='o', size=5,
                      alpha=0.7)

        num_categories = len(df_full_naive_pitchsplit_plot['BrainArea'].unique())
        for stripplot in ax.collections[num_categories:]:
            stripplot.set_offsets(stripplot.get_offsets() + [x_shift, 0])

        # plt.xlim(-0.5, 1.5)

        # do a mann whitney u test between the meanscores for PEG and MEG
        stat_peg, p_peg = mannwhitneyu(df_full_pitchsplit_plot_peg['MeanScore'],
                                       df_full_pitchsplit_plot_meg['MeanScore'], alternative='two-sided')
        stat_peg_index, p_peg_index = mannwhitneyu(df_full_pitchsplit_plot_peg['GenFrac'],
                                                   df_full_pitchsplit_plot_meg['GenFrac'], alternative='two-sided')

        if options == 'index':
            plt.xlabel(f'Mean Decoding Score of Top 25% of Units', fontsize=20)
        elif options == 'frac':
            plt.xlabel(f'Mean Decoding Score of Top 25% of Units', fontsize=20)
        plt.ylabel('Mean Score', fontsize=20)
        # get labels and handles for legeend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=[handles[0]], labels=['trained'], title=None, fontsize=18)

        # reinsert the handles and labeels:

        plt.savefig(f'G:/neural_chapter/figures/meanscore_highthreshold_violin_bybrainarea_{options}_04052024.png')

        df_full_pitchsplit_plot['Group'] = 'Trained'
        df_full_naive_pitchsplit_plot['Group'] = 'Naive'
        combined_df = pd.concat([df_full_pitchsplit_plot, df_full_naive_pitchsplit_plot])

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        sns.violinplot(x='BrainArea', y='MeanScore', hue='Group', data=combined_df, ax=ax, split=True,
                       inner='quartiles', palette={"Trained": "purple", "Naive": "cyan"})
        sns.stripplot(x='BrainArea', y='MeanScore', hue='Group', data=combined_df, ax=ax, dodge=True, jitter=True,
                      linewidth=1, edgecolor='gray', palette={"Trained": "purple", "Naive": "cyan"})
        if options == 'index':
            plt.xlabel(f'Mean Decoding Score of Top 25% of Units', fontsize=20)
        elif options == 'frac':
            plt.xlabel(f'Mean Decoding Score of Top 25% of Units', fontsize=20)
        plt.ylabel('Mean Score', fontsize=20)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=[handles[0], handles[1]], labels=['trained', 'naive'], title=None, fontsize=18)
        plt.savefig(
            f'G:/neural_chapter/figures/meanscore_highthreshold_naive_trained_violin_bybrainarea_{options}_04052024.png')

        plt.show()

        fig, ax = plt.subplots(1, dpi=300)
        sns.violinplot(x='BrainArea', y='GenFrac', data=df_full_naive_pitchsplit_plot, ax=ax, inner=None,
                       color='lightgray')
        sns.stripplot(x='BrainArea', y='GenFrac', data=df_full_naive_pitchsplit_plot, ax=ax, color='cyan',
                      label='naive', dodge=False)
        plt.title(f'Generalizability scores for the naive animals, upper quartile threshold, index or frac:{options}')
        if options == 'index':
            plt.xlabel('Generalizability Index of Top 25% of Units', fontsize=20)
        elif options == 'frac':
            plt.xlabel('Generalizability Fraction of Top 25% of Units', fontsize=20)
        plt.ylabel('GenFrac', fontsize=20)

        plt.savefig(f'G:/neural_chapter/figures/GenFrac_highthreshold_violin_naive_bybrainarea_{options}_04052024.png')

        # stat_peg, p_peg = mannwhitneyu(df_full_naive_pitchsplit_plot_peg['GenFrac'], df_full_naive_pitchsplit_plot_meg['GenFrac'], alternative = 'less')
        # plot the brain area loation of the units that are significantly different
        fig, ax = plt.subplots(1, dpi=300)
        # make the dataframe in the order of PEG, MEG, AEG
        df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit_plot.sort_values(by=['BrainArea'])

        sns.violinplot(x='BrainArea', y='MeanScore', data=df_full_naive_pitchsplit_plot, ax=ax, inner=None, label=None,
                       color='lightgray', order=['MEG', 'PEG', 'AEG'])
        sns.stripplot(x='BrainArea', y='MeanScore', data=df_full_naive_pitchsplit_plot, ax=ax, label='naive',
                      color='cyan', dodge=False, order=['MEG', 'PEG', 'AEG'])
        plt.xlim(-0.5, 1.5)
        # do a mann whitney u test between the meanscores for PEG and MEG
        stat_peg_naive, p_peg_naive = mannwhitneyu(df_full_naive_pitchsplit_plot_peg['MeanScore'],
                                                   df_full_naive_pitchsplit_plot_meg['MeanScore'], alternative='less')
        plt.xlabel(f'Mean Decoding Score of Top 25% of Units', fontsize=20)
        handles, labels = ax.get_legend_handles_labels()
        # reinsert the legend
        ax.legend(handles=[handles[0]], labels=['naive'], title=None, fontsize=18)
        plt.ylabel('Mean Score', fontsize=20)
        plt.savefig(f'G:/neural_chapter/figures/meanscore_highthreshold_violin_naive_bybrainarea_04052024.png')

        n1 = len(df_full_naive_pitchsplit_plot_peg)
        n2 = len(df_full_naive_pitchsplit_plot_meg)
        r_naive = 1 - (2 * stat_peg_naive) / (n1 * n2)

        n1 = len(df_full_pitchsplit_plot_peg)
        n2 = len(df_full_pitchsplit_plot_meg)
        r_trained = 1 - (2 * stat_peg) / (n1 * n2)

        # export the p values to a csv file
        df_pvalues = pd.DataFrame(columns=['Trained/naive', 'pvalue', 'statistic', 'effectsize'])
        df_pvalues = df_pvalues.append(
            {'Trained/naive': 'Trained', 'pvalue': p_peg, 'statistic': stat_peg, 'effectsize': r_trained},
            ignore_index=True)
        df_pvalues = df_pvalues.append(
            {'Trained/naive': 'Naive', 'pvalue': p_peg_naive, 'statistic': stat_peg_naive, 'effectsize': r_naive},
            ignore_index=True)
        df_pvalues.to_csv(f'G:/neural_chapter/figures/pvalues_highthreshold_manwhittest_{options}.csv')

        # man whitney u test
        stat, p = mannwhitneyu(df_full_pitchsplit_plot['GenFrac'], df_full_naive_pitchsplit_plot['GenFrac'],
                               alternative='less')
        print(f'Generalizability scores, high threshold untis, index method: {options}')
        print(stat)
        print(p)

        df_full_pitchsplit_allsubset = create_gen_frac_variable(df_full, high_score_threshold=False,
                                                                index_or_frac=options)
        # remove all rows where GenFrac is nan
        df_full_pitchsplit_plot = df_full_pitchsplit_allsubset[df_full_pitchsplit_allsubset['GenFrac'].notna()]
        df_full_pitchsplit_plot = df_full_pitchsplit_plot.drop_duplicates(subset=['ID'])
        # get the subset of the data where the meanscore is above 0.75
        df_full_pitchsplit_plot_highsubset = df_full_pitchsplit_plot[df_full_pitchsplit_plot['MeanScore'] > 0.75]

        df_full_naive_pitchsplit_plot = create_gen_frac_variable(df_full_naive, high_score_threshold=False,
                                                                 index_or_frac=options)
        df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit_plot[df_full_naive_pitchsplit_plot['GenFrac'].notna()]
        df_full_naive_pitchsplit_plot = df_full_naive_pitchsplit_plot.drop_duplicates(subset=['ID'])

        fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
        sns.histplot(df_full_pitchsplit_plot['GenFrac'], ax=ax, kde=True, bins=20, label='Trained')
        sns.histplot(df_full_pitchsplit_plot_highsubset['GenFrac'], ax=ax, kde=True, bins=20,
                     label='Trained, 75% mean score threshold')
        sns.histplot(df_full_naive_pitchsplit_plot['GenFrac'], ax=ax, kde=True, bins=20, label='Naive')
        # sns.histplot(df_full_naive_pitchsplit_plot_highsubset['GenFrac'], ax=ax, kde=True, bins=20, label='Naive, 75% mean score threshold')
        plt.legend()
        plt.title(
            f'Distribution of generalisability scores for the trained and naive animals, all units, index method: {options}')

        stat_general, p_general = mannwhitneyu(df_full_pitchsplit_plot['GenFrac'],
                                               df_full_naive_pitchsplit_plot['GenFrac'], alternative='two-sided')
        print(f'Generalizability scores, all units, index method: {options}')
        print(stat_general)
        print(p_general)

        plt.savefig(f'G:/neural_chapter/figures/GenFrac_allthreshold_{options}_04052024.png')
        plt.show()

        # plot as a violin plot with brainarea on the x-axis
        fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)

        sns.violinplot(x='BrainArea', y='GenFrac', data=df_full_pitchsplit_plot, ax=ax, inner=None, color='lightgray')
        sns.stripplot(x='BrainArea', y='GenFrac', data=df_full_pitchsplit_plot, ax=ax, size=3, dodge=False)
        plt.title(f'Generalizability scores for the trained animals, all units, method: {options}')
        if options == 'index':
            plt.xlabel('Generalizability Index ', fontsize=20)
        elif options == 'frac':
            plt.xlabel('Generalizability Fraction', fontsize=20)

        plt.savefig(f'G:/neural_chapter/figures/GenFrac_allthreshold_violin_{options}_04052024.png')
        plt.show()

        fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
        sns.violinplot(x='BrainArea', y='GenFrac', data=df_full_naive_pitchsplit_plot, ax=ax, inner=None,
                       color='lightgray')
        sns.stripplot(x='BrainArea', y='GenFrac', data=df_full_naive_pitchsplit_plot, ax=ax, size=3, dodge=False)

        if options == 'index':
            plt.xlabel('Generalizability Index', fontsize=20)
        elif options == 'frac':
            plt.xlabel('Generalizability Fraction', fontsize=20)
        plt.title(f'Generalizability scores for the naive animals, all units, method: {options}')
        plt.savefig(f'G:/neural_chapter/figures/GenFrac_allthreshold_violin_naive_{options}_04052024.png')
        # do the mann whitney u test between genfrac scores from PEG and MEG
        df_full_pitchsplit_plot_peg = df_full_pitchsplit_plot[df_full_pitchsplit_plot['BrainArea'] == 'PEG']
        df_full_pitchsplit_plot_meg = df_full_pitchsplit_plot[df_full_pitchsplit_plot['BrainArea'] == 'MEG']
        df_full_naive_pitchsplit_plot_peg = df_full_naive_pitchsplit_plot[
            df_full_naive_pitchsplit_plot['BrainArea'] == 'PEG']
        df_full_naive_pitchsplit_plot_meg = df_full_naive_pitchsplit_plot[
            df_full_naive_pitchsplit_plot['BrainArea'] == 'MEG']

        stat_peg, p_peg = mannwhitneyu(df_full_pitchsplit_plot_peg['GenFrac'], df_full_pitchsplit_plot_meg['GenFrac'],
                                       alternative='less')
        print(f'p value for PEG vs MEG genfrac scores for trained animals, {options} , {p_peg}')
        stat_peg_naive, p_peg_naive = mannwhitneyu(df_full_naive_pitchsplit_plot_peg['GenFrac'],
                                                   df_full_naive_pitchsplit_plot_meg['GenFrac'], alternative='less')
        print(f'p value for PEG vs MEG genfrac scores for naive animals, {options} , {p_peg}')

    # now plot by the probe word for the trained animals
    fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
    # Plot the data points color-coded by ProbeWord for above chance scores
    sns.stripplot(x='ProbeWord', y='Score', data=df_full, ax=ax, size=3, dodge=True,
                  palette='Set1',
                  hue='Below-chance', alpha=1, jitter=True)
    sns.violinplot(x='ProbeWord', y='Score', data=df_full, ax=ax, color='white')
    plt.title('Trained animals'' scores over distractor word')
    plt.savefig(f'G:/neural_chapter/figurestrained_animals_overdistractor_04052024.png', dpi=300)
    plt.show()

    # plot strip plot split by pitch shift
    df_full_pitchsplit_violinplot = df_full
    df_full_pitchsplit_violinplot['ProbeWord'] = df_full_pitchsplit_violinplot['ProbeWord'].replace(
        {'(2,2)': 'craft', '(3,3)': 'in contrast to', '(4,4)': 'when a', '(5,5)': 'accurate', '(6,6)': 'pink noise',
         '(7,7)': 'of science', '(8,8)': 'rev. instruments', '(9,9)': 'boats', '(10,10)': 'today',
         '(13,13)': 'sailor', '(15,15)': 'but', '(16,16)': 'researched', '(18,18)': 'took', '(19,19)': 'the vast',
         '(20,20)': 'today', '(21,21)': 'he takes', '(22,22)': 'becomes', '(23,23)': 'any', '(24,24)': 'more'})
    upper_quartile = np.percentile(df_full_pitchsplit_violinplot['Score'], 75)
    df_full_pitchsplit_violinplot = df_full_pitchsplit_violinplot[
        df_full_pitchsplit_violinplot['ProbeWord'] != 'he takes']
    df_full_pitchsplit_violinplot = df_full_pitchsplit_violinplot[
        df_full_pitchsplit_violinplot['ProbeWord'] != 'becomes']

    df_full_pitchsplit_violinplot['MeanScore'] = df_full_pitchsplit_violinplot.groupby('ID')[
        'Score'].transform('mean')
    # get the top quartile of units
    df_full_pitchsplit_violinplot_highsubset = df_full_pitchsplit_violinplot[
        df_full_pitchsplit_violinplot['MeanScore'] > upper_quartile]

    # create a custom palette`
    custom_colors = ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"]
    fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
    # remove he takes and becomes from the plot

    df_above_chance_ps = df_full_pitchsplit_violinplot[df_full_pitchsplit_violinplot['Below-chance'] == 0]
    df_below_chance_ps = df_full_pitchsplit_violinplot[df_full_pitchsplit_violinplot['Below-chance'] == 1]

    sns.stripplot(x='ProbeWord', y='Score', data=df_above_chance_ps, ax=ax, size=3, dodge=True, palette=custom_colors,
                  edgecolor='k', linewidth=0.2, hue='PitchShift', jitter=True)
    sns.stripplot(x='ProbeWord', y='Score', data=df_below_chance_ps, ax=ax, size=3, dodge=True, edgecolor='k',
                  linewidth=0.2,
                  alpha=0.25, jitter=False, hue='PitchShift', palette=custom_colors)

    sns.violinplot(x='ProbeWord', y='Score', data=df_full_pitchsplit_violinplot, ax=ax, color='white', hue='PitchShift')
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=30)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=30)
    # ax.set_ylim([0, 1])
    # get legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:2], labels=['Control', 'Pitch-shifted'], title=None, fontsize=18)
    plt.ylim([0, 1])

    plt.ylabel('Decoding Score', fontsize=40)
    plt.xlabel(None)

    plt.title("Trained animals' scores over probe word", fontsize=40)
    plt.savefig(f'G:/neural_chapter/figures/trained_animals_overdistractor_dividedbypitchshift_04052024.png', dpi=300,
                bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
    df_above_chance_ps = df_full_pitchsplit_violinplot_highsubset[
        df_full_pitchsplit_violinplot_highsubset['Below-chance'] == 0]

    sns.stripplot(x='ProbeWord', y='Score', data=df_above_chance_ps, ax=ax, size=3, dodge=True, palette=custom_colors,
                  edgecolor='k', linewidth=0.2, hue='PitchShift', jitter=True)
    # sns.stripplot(x='ProbeWord', y='Score', data=df_below_chance_ps, ax=ax, size=3, dodge=True, edgecolor='k',
    #               linewidth=0.2,
    #               alpha=0.25, jitter=False, hue='PitchShift', palette=custom_colors)

    sns.violinplot(x='ProbeWord', y='Score', data=df_above_chance_ps, ax=ax, color='white', hue='PitchShift')
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=30)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=30)
    # ax.set_ylim([0, 1])
    # get legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:2], labels=['Control', 'Pitch-shifted'], title=None, fontsize=18)
    plt.ylim([0, 1])

    plt.ylabel('Decoding Score', fontsize=40)
    plt.xlabel(None)

    plt.title("Trained animals' scores, \n top performing units, over probe word", fontsize=40)
    plt.savefig(f'G:/neural_chapter/figures/trained_animals_topunits_overdistractor_dividedbypitchshift_04052024.png',
                dpi=300,
                bbox_inches='tight')
    plt.show()

    df_kruskal = pd.DataFrame(columns=['ProbeWord', 'Kruskal_pvalue_trained', 'less than 0.05', 'epsilon_squared'])
    # Perform Kruskal-Wallis test for each ProbeWord
    for probe_word in df_full_pitchsplit_violinplot['ProbeWord'].unique():
        subset_data = df_full_pitchsplit_violinplot[df_full_pitchsplit_violinplot['ProbeWord'] == probe_word]

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
        # df_T = N - 1
        # Calculate epsilon-squared (effect size)
        epsilon_squared = (result_kruskal.statistic - df_H) / (N - k)

        print(
            f"ProbeWord: {probe_word}, Kruskal-Wallis p-value: {result_kruskal.pvalue}, epsilon-squared: {epsilon_squared}")
        # append to a dataframe
        df_kruskal = df_kruskal.append({'ProbeWord': probe_word, 'Kruskal_pvalue_trained': result_kruskal.pvalue,
                                        'less than 0.05': less_than_alpha, 'epsilon_squared': epsilon_squared},
                                       ignore_index=True)
    # export the dataframe
    df_kruskal.to_csv('G:/neural_chapter/figures/kruskal_pvalues_trained.csv')

    df_full_naive_pitchsplit_violinplot = df_full_naive
    df_full_naive_pitchsplit_violinplot['ProbeWord'] = df_full_naive_pitchsplit_violinplot['ProbeWord'].replace(
        {'(2,2)': 'craft', '(3,3)': 'in contrast to', '(4,4)': 'when a', '(5,5)': 'accurate', '(6,6)': 'pink noise',
         '(7,7)': 'of science', '(8,8)': 'rev. instruments', '(9,9)': 'boats', '(10,10)': 'today',
         '(13,13)': 'sailor', '(15,15)': 'but', '(16,16)': 'researched', '(18,18)': 'took', '(19,19)': 'the vast',
         '(20,20)': 'today', '(21,21)': 'he takes', '(22,22)': 'becomes', '(23,23)': 'any', '(24,24)': 'more'})
    # calculate the top quartile mean
    upper_quartile = np.percentile(df_full_naive_pitchsplit_violinplot['Score'], 75)

    df_full_naive_pitchsplit_violinplot['MeanScore'] = df_full_naive_pitchsplit_violinplot.groupby('ID')[
        'Score'].transform('mean')
    # get the top quartile of units
    df_full_naive_pitchsplit_violinplot_highsubset = df_full_naive_pitchsplit_violinplot[
        df_full_naive_pitchsplit_violinplot['MeanScore'] > upper_quartile]

    # now plot by the probe word for the naive animals

    fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
    df_above_chance = df_full_naive_pitchsplit_violinplot[df_full_naive_pitchsplit_violinplot['Below-chance'] == 0]
    df_below_chance = df_full_naive_pitchsplit_violinplot[df_full_naive_pitchsplit_violinplot['Below-chance'] == 1]
    custom_colors_naive = ["#5ec962", "#fde725"]
    sns.stripplot(x='ProbeWord', y='Score', data=df_above_chance, ax=ax, size=3, dodge=True,
                  palette=custom_colors_naive,
                  hue='PitchShift', edgecolor='k', linewidth=0.2, jitter=True)
    sns.stripplot(x='ProbeWord', y='Score', data=df_below_chance, ax=ax, size=3, dodge=True, color='lightgray',
                  alpha=0.1, jitter=True, hue='PitchShift', palette=custom_colors_naive, edgecolor='k', linewidth=0.2)

    sns.violinplot(x='ProbeWord', y='Score', data=df_full_naive, ax=ax, color='white', hue='PitchShift')
    # get the legend handles
    plt.ylabel('Decoding Score', fontsize=40)
    plt.xlabel(None)
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=30)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=30)

    ax.legend(handles=handles[0:2], labels=['Control', 'Pitch-shifted'], title=None, fontsize=18)
    plt.title("Naive animals' scores over distractor word", fontsize=40)
    plt.savefig(f'G:/neural_chapter/figures/naive_animals_overdistractor_dividedbypitchshift_04052024.png', dpi=300,
                bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)
    df_above_chance = df_full_naive_pitchsplit_violinplot_highsubset[
        df_full_naive_pitchsplit_violinplot['Below-chance'] == 0]
    custom_colors_naive = ["#5ec962", "#fde725"]
    sns.stripplot(x='ProbeWord', y='Score', data=df_above_chance, ax=ax, size=3, dodge=True,
                  palette=custom_colors_naive,
                  hue='PitchShift', edgecolor='k', linewidth=0.2, jitter=True)
    # sns.stripplot(x='ProbeWord', y='Score', data=df_below_chance, ax=ax, size=3, dodge=True, color='lightgray',
    #               alpha=0.1, jitter=True, hue='PitchShift', palette=custom_colors_naive, edgecolor='k', linewidth=0.2)

    sns.violinplot(x='ProbeWord', y='Score', data=df_above_chance, ax=ax, color='white', hue='PitchShift')
    # get the legend handles
    plt.ylabel('Decoding Score', fontsize=40)
    plt.xlabel(None)
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=30)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=30)

    ax.legend(handles=handles[0:2], labels=['Control', 'Pitch-shifted'], title=None, fontsize=18)
    plt.title("Naive animals' scores, \n top performing units, over distractor word", fontsize=40)
    plt.savefig(f'G:/neural_chapter/figures/naive_animals_topunits_overdistractor_dividedbypitchshift_04052024.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    df_kruskal = pd.DataFrame(columns=['ProbeWord', 'Kruskal_pvalue', 'less than 0.05', 'epsilon_squared'])
    # Perform Kruskal-Wallis test for each ProbeWord
    for probe_word in df_full_naive_pitchsplit_violinplot['ProbeWord'].unique():
        subset_data = df_full_naive_pitchsplit_violinplot[
            df_full_naive_pitchsplit_violinplot['ProbeWord'] == probe_word]

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
        # df_T = N - 1
        # Calculate epsilon-squared (effect size)

        epsilon_squared = (result_kruskal.statistic - df_H) / (N - k)

        print(
            f"ProbeWord: {probe_word}, Kruskal-Wallis p-value: {result_kruskal.pvalue}, epsilon-squared: {epsilon_squared}")

        # append to a dataframe
        df_kruskal = df_kruskal.append(
            {'ProbeWord': probe_word, 'Kruskal_pvalue': result_kruskal.pvalue, 'less than 0.05': less_than_alpha,
             'epsilon_squared': epsilon_squared}, ignore_index=True)
    # export the dataframe
    df_kruskal.to_csv('G:/neural_chapter/figures/kruskal_pvalues_naive.csv')
    # run an anova to see if probe word is significant
    # first get the data into a format that can be analysed

    df_full_pitchsplit_anova = df_full.copy()

    unique_probe_words = df_full_pitchsplit_anova['ProbeWord'].unique()

    df_full_pitchsplit_anova = df_full_pitchsplit_anova.reset_index(drop=True)

    df_full_pitchsplit_anova['ProbeWord'] = pd.Categorical(df_full_pitchsplit_anova['ProbeWord'],
                                                           categories=unique_probe_words, ordered=True)
    df_full_pitchsplit_anova['ProbeWord'] = df_full_pitchsplit_anova['ProbeWord'].cat.codes

    df_full_pitchsplit_anova['BrainArea'] = df_full_pitchsplit_anova['BrainArea'].astype('category')

    # cast the probe word category as an int
    df_full_pitchsplit_anova['ProbeWord'] = df_full_pitchsplit_anova['ProbeWord'].astype('int')
    df_full_pitchsplit_anova['PitchShift'] = df_full_pitchsplit_anova['PitchShift'].astype('int')
    df_full_pitchsplit_anova['Below-chance'] = df_full_pitchsplit_anova['Below-chance'].astype('int')

    df_full_pitchsplit_anova["ProbeWord"] = pd.to_numeric(df_full_pitchsplit_anova["ProbeWord"])
    df_full_pitchsplit_anova["PitchShift"] = pd.to_numeric(df_full_pitchsplit_anova["PitchShift"])
    df_full_pitchsplit_anova["Below_chance"] = pd.to_numeric(df_full_pitchsplit_anova["Below-chance"])
    df_full_pitchsplit_anova["Score"] = pd.to_numeric(df_full_pitchsplit_anova["Score"])
    # change the columns to the correct type

    # remove all rows where the score is NaN
    df_full_pitchsplit_anova = df_full_pitchsplit_anova.dropna(subset=['Score'])
    # nest ferret as a variable ,look at the relative magnittud eo fthe coefficients for both lightgbm model and anova
    print(df_full_pitchsplit_anova.dtypes)
    # now run anova
    import statsmodels.formula.api as smf
    formula = 'Score ~ C(ProbeWord) + C(PitchShift) +C(BrainArea)+C(SingleUnit)'
    model = smf.ols(formula, data=df_full_pitchsplit_anova).fit()
    anova_table = sm.stats.anova_lm(model, typ=3)
    # get the coefficient of determination
    print(model.rsquared)
    print(anova_table)
    # combine the dataframes df_full_naive_pitchsplit and
    # add the column naive to df_full_naive_pitchsplit
    df_full_naive['Naive'] = 1
    df_full['Naive'] = 0
    combined_df = df_full_naive.append(df_full)
    # now run the lightgbm function
    run_mixed_effects_on_dataframe(combined_df)
    runlgbmmodel_score(combined_df, optimization=False)

    # now plot by animal:
    for animal in ['F1901_Crumble', 'F1902_Eclair', 'F2003_Orecchiette', 'F1812_Nala']:
        df_full_naive_ps_animal = df_full_naive[df_full_naive['ID'].str.contains(animal)]
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
        plt.savefig(f'G:/neural_chapter/figures/naive_animals_overdistractor_dividedbypitchshift_{animal}_04052024.png',
                    dpi=300)

        plt.show()

    for animal in ['F1702_Zola', 'F1815_Cruella', 'F1604_Squinty', 'F1606_Windolene']:
        df_full_pitchsplit_animal = df_full[df_full['ID'].str.contains(animal)]
        if len(df_full_pitchsplit_animal) == 0:
            continue
        fig, ax = plt.subplots(1, figsize=(20, 10), dpi=300)

        df_above_chance = df_full_pitchsplit_animal[df_full_pitchsplit_animal['Below-chance'] == 0]
        df_below_chance = df_full_pitchsplit_animal[df_full_pitchsplit_animal['Below-chance'] == 1]

        sns.stripplot(x='ProbeWord', y='Score', data=df_above_chance, ax=ax, size=3, dodge=True, palette='Spectral',
                      hue='PitchShift')
        sns.stripplot(x='ProbeWord', y='Score', data=df_below_chance, ax=ax, size=3, dodge=True, color='lightgray',
                      alpha=0.5, jitter=False, hue='PitchShift')
        sns.violinplot(x='ProbeWord', y='Score', data=df_full_pitchsplit_animal, ax=ax, hue='PitchShift',
                       palette='Spectral')
        plt.title(f'Trained scores over distractor word:{animal}')
        plt.savefig(f'G:/neural_chapter/figurestrained_{animal}_overdistractor_dividedbypitchshift_04052024.png',
                    dpi=300)

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

    # plot scatter data in a loop
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
    ax.scatter(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, marker='P', color='purple', alpha=0.8,
               label='trained', s=0.1)
    plt.title('trained animals, number of points: ' + str(len(bigconcatenatetrained_ps)))
    plt.show()
    unique_scores = np.unique(bigconcatenatetrained_ps)
    len(unique_scores)

    fig, ax = plt.subplots(1, figsize=(9, 9), dpi=300)

    ax.scatter(bigconcatenatenaive_nonps, bigconcatenatenaive_ps, marker='P', color='darkcyan', alpha=0.5,
               label='naive')
    ax.scatter(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, marker='P', color='purple', alpha=0.5,
               label='trained')
    x = np.linspace(0.4, 1, 101)
    ax.plot(x, x, color='black', linestyle='--')  # identity line

    slope, intercept, r_value, pv, se = stats.linregress(bigconcatenatetrained_nonps, bigconcatenatetrained_ps)

    sns.regplot(x=bigconcatenatetrained_nonps, y=bigconcatenatetrained_ps, scatter=False, color='purple',
                label=' $y=%3.7s*x+%3.7s$' % (slope, intercept), ax=ax,
                line_kws={'label': ' $y=%3.7s*x+%3.7s$' % (slope, intercept)})
    slope, intercept, r_value, pv, se = stats.linregress(bigconcatenatenaive_nonps, bigconcatenatenaive_ps)

    sns.regplot(x=bigconcatenatenaive_nonps, y=bigconcatenatenaive_ps, scatter=False, color='darkcyan',
                label=' $y=%3.7s*x+%3.7s$' % (slope, intercept),
                ax=ax, line_kws={'label': '$y=%3.7s*x+%3.7s$' % (slope, intercept)})

    ax.set_ylabel('LSTM decoding score, F0 roved', fontsize=18)
    ax.set_xlabel('LSTM decoding score, F0 control', fontsize=18)
    ax.set_title('LSTM decoder scores for' + ' F0 control vs. roved,\n ' + ' trained and naive animals', fontsize=30)
    plt.legend(fontsize=12, ncol=2)
    fig.tight_layout()
    plt.show()

    # histogram distribution of the trained and naive animals
    fig, ax = plt.subplots(1, figsize=(8, 8))
    # relativescoretrained = abs(bigconcatenatetrained_nonps - bigconcatenatetrained_ps)/ bigconcatenatetrained_ps

    relativescoretrained = [bigconcatenatetrained_nonps - bigconcatenatetrained_ps for
                            bigconcatenatetrained_nonps, bigconcatenatetrained_ps in
                            zip(bigconcatenatetrained_nonps, bigconcatenatetrained_ps)]
    relativescorenaive = [bigconcatenatenaive_nonps - bigconcatenatenaive_ps for
                          bigconcatenatenaive_nonps, bigconcatenatenaive_ps in
                          zip(bigconcatenatenaive_ps, bigconcatenatenaive_nonps)]
    relativescoretrainedfrac = [relativescoretrained / (bigconcatenatetrained_nonps + bigconcatenatenaive_nonps) for
                                relativescoretrained, bigconcatenatetrained_nonps, bigconcatenatenaive_nonps in
                                zip(relativescoretrained, bigconcatenatetrained_nonps, bigconcatenatenaive_nonps)]
    relativescorenaivefrac = [relativescorenaive / (bigconcatenatenaive_nonps + bigconcatenatetrained_nonps) for
                              relativescorenaive, bigconcatenatenaive_nonps, bigconcatenatetrained_nonps in
                              zip(relativescorenaive, bigconcatenatenaive_nonps, bigconcatenatetrained_nonps)]

    sns.distplot(relativescoretrained, bins=20, label='trained', ax=ax, color='purple')
    sns.distplot(relativescorenaive, bins=20, label='naive', ax=ax, color='darkcyan')
    plt.axvline(x=0, color='black')
    # man whiteney test score

    manwhitscore = mannwhitneyu(relativescoretrained, relativescorenaive, alternative='greater')
    sample1 = np.random.choice(relativescoretrained, size=10000, replace=True)

    # Generate a random sample of size 100 from data2 with replacement
    sample2 = np.random.choice(relativescorenaive, size=10000, replace=True)

    # Perform a t-test on the samples
    t_stat, p_value = stats.ttest_ind(sample1, sample2, alternative='greater')

    # Print the t-statistic and p-value
    print(t_stat, p_value)
    plt.title('Control - roved F0 \n LSTM decoder scores between trained and naive animals', fontsize=18)
    plt.xlabel('Control - roved F0 \n LSTM decoder scores', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    # ax.legend()
    # plt.savfig('G:/neural_chapter/figures/diffF0distribution_04052024_04052024.png', dpi=1000)
    plt.show()

    # plot sns histogram of the relative score and with the displot function overlaid
    fig, ax = plt.subplots(1, figsize=(8, 8))
    # ax = sns.displot(relativescoretrainedfrac, bins = 20, label='trained',ax=ax, color='purple')
    sns.histplot(relativescoretrainedfrac, bins=20, label='trained', color='purple', kde=True)
    sns.histplot(relativescorenaivefrac, bins=20, label='naive', color='darkcyan', kde=True)

    # plt.savfig('G:/neural_chapter/figures/diffF0distribution_relfrac_histplotwithkde_04052024_04052024.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax = sns.distplot(relativescoretrainedfrac, bins=20, label='trained', ax=ax, color='purple')
    x = ax.lines[-1].get_xdata()  # Get the x data of the distribution
    y = ax.lines[-1].get_ydata()  # Get the y data of the distribution
    maxidtrained_idx = np.argmax(y)
    x_coord_trained = x[maxidtrained_idx]
    ax2 = sns.distplot(relativescorenaivefrac, bins=20, label='naive', ax=ax, color='darkcyan')

    x2 = ax2.lines[-1].get_xdata()  # Get the x data of the distribution
    y2 = ax2.lines[-1].get_ydata()  # Get the y data of the distribution
    maxidnaive_idx = np.argmax(y2)  # The id of the peak (maximum of y data)

    x_coord_naive = x2[maxidnaive_idx]
    plt.axvline(x=0, color='black')
    kstestnaive = scipy.stats.kstest(relativescorenaivefrac, stats.norm.cdf)
    leveneteststat = scipy.stats.levene(relativescorenaivefrac, relativescoretrainedfrac)
    manwhitscorefrac = mannwhitneyu(relativescorenaivefrac, relativescoretrainedfrac, alternative='less')
    # caclulate medians of distribution

    sample1_trained = np.random.choice(relativescoretrainedfrac, size=10000, replace=True)

    # Generate a random sample of size 100 from data2 with replacement
    sample2_naive = np.random.choice(relativescorenaive, size=10000, replace=True)

    # Perform a t-test on the samples
    t_statfrac, p_valuefrac = stats.ttest_ind(sample2_naive, sample1_trained, alternative='less')

    # Print the t-statistic and p-value
    print(t_statfrac, p_valuefrac)
    plt.title('Control - roved F0 \n LSTM decoder scores between trained and naive animals', fontsize=18)
    plt.xlabel('Control - roved F0 \n LSTM decoder scores divided by control F0', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    # ax.legend(fontsize = 18)

    # plt.savfig('G:/neural_chapter/figures/diffF0distribution_frac_20062023wlegendintertrialroving_04052024_04052024.png', dpi=1000)
    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=800)
    ax.set_xlim([0, 1])

    sns.distplot(bigconcatenatetrained_nonps, label='trained', ax=ax, color='purple')
    sns.distplot(bigconcatenatenaive_nonps, label='naive', ax=ax, color='darkcyan')
    # plt.axvline(x=0, color='black')
    # man whiteney test score
    plt.title('Control F0 LSTM decoder scores between  \n trained and naive animals', fontsize=18)
    plt.xlabel('Control F0 LSTM decoder scores', fontsize=20)

    plt.ylabel('Density', fontsize=20)
    manwhitscorecontrolf0 = mannwhitneyu(bigconcatenatetrained_nonps, bigconcatenatenaive_nonps, alternative='greater')

    # ax.legend()
    # plt.savfig('G:/neural_chapter/figures/controlF0distribution20062023intertrialroving_04052024_04052024.png', dpi=1000)

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

    ax.legend(fontsize=18)
    # plt.savfig('G:/neural_chapter/figures/rovedF0distribution_20062023intertrialroving_04052024_04052024.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=800)
    ax.set_xlim([0, 1])
    sns.distplot(bigconcatenatetrained_ps, label='trained roved', ax=ax, color='purple')
    sns.distplot(bigconcatenatetrained_nonps, label='trained control', ax=ax, color='magenta')
    ax.legend(fontsize=18)
    plt.title('Roved and Control F0 Distributions for the Trained Animals', fontsize=18)
    plt.xlabel(' LSTM decoder scores', fontsize=20)

    # plt.savfig('G:/neural_chapter/figures/rovedF0vscontrolF0traineddistribution_20062023intertrialroving_04052024_04052024.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=800)
    ax.set_xlim([0, 1])
    sns.distplot(bigconcatenatenaive_ps, label='naive roved', ax=ax, color='darkcyan')
    sns.distplot(bigconcatenatenaive_nonps, label='naive control', ax=ax, color='cyan')
    ax.legend(fontsize=18)
    plt.xlabel(' LSTM decoder scores', fontsize=20)
    plt.title('Roved and Control F0 Distributions for the Naive Animals', fontsize=18)

    plt.savfig(
        'G:/neural_chapter/figures/rovedF0vscontrolF0naivedistribution_20062023intertrialroving_04052024_04052024.png',
        dpi=1000)
    plt.show()
    kstestcontrolf0vsrovedtrained = scipy.stats.kstest(bigconcatenatetrained_nonps, bigconcatenatetrained_ps,
                                                       alternative='two-sided')

    kstestcontrolf0vsrovednaive = scipy.stats.kstest(bigconcatenatenaive_nonps, bigconcatenatenaive_ps,
                                                     alternative='two-sided')

    naivearray = np.concatenate((np.zeros((len(bigconcatenatetrained_nonps) + len(bigconcatenatetrained_ps), 1)),
                                 np.ones((len(bigconcatenatenaive_nonps) + len(bigconcatenatenaive_ps), 1))))
    trainedarray = np.concatenate((np.ones((len(bigconcatenatetrained_nonps) + len(bigconcatenatetrained_ps), 1)),
                                   np.zeros((len(bigconcatenatenaive_nonps) + len(bigconcatenatenaive_ps), 1))))
    controlF0array = np.concatenate((np.ones((len(bigconcatenatetrained_nonps), 1)),
                                     np.zeros((len(bigconcatenatetrained_ps), 1)),
                                     np.ones((len(bigconcatenatenaive_nonps), 1)),
                                     np.zeros((len(bigconcatenatenaive_ps), 1))))
    rovedF0array = np.concatenate((np.zeros((len(bigconcatenatetrained_nonps), 1)),
                                   np.ones((len(bigconcatenatetrained_ps), 1)),
                                   np.zeros((len(bigconcatenatenaive_nonps), 1)),
                                   np.ones((len(bigconcatenatenaive_ps), 1))))
    scores = np.concatenate(
        (bigconcatenatetrained_nonps, bigconcatenatetrained_ps, bigconcatenatenaive_nonps, bigconcatenatenaive_ps))










if __name__ == '__main__':
    main()