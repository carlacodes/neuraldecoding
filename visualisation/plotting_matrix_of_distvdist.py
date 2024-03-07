import os
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


def scatterplot_and_visualise(probewordlist,
                              saveDir='D:/Users/cgriffiths/resultsms4/lstm_output_frommyriad_15012023/lstm_kfold_14012023_crumble',
                              ferretname='Crumble',
                              singleunitlist=[0,1,2],
                              multiunitlist=[0,1,2,3], noiselist=[], stream = 'BB_2', fullid = 'F1901_Crumble', report =[], permutation_scores=False):
    if permutation_scores == False:
        score_key = 'lstm_balanced_avg'
    else:
        score_key = 'perm_bal_ac'

    singleunitlist = [x - 1 for x in singleunitlist]
    multiunitlist = [x - 1 for x in multiunitlist]
    noiselist = [x - 1 for x in noiselist]
    original_cluster_list = np.empty([0])


    for probeword1 in probewordlist:
        for probeword2 in probewordlist:
            singleunitlist_copy = singleunitlist.copy()
            multiunitlist_copy = multiunitlist.copy()
            #load the original clusters to split from the json file
            json_file_path = f'F:\split_cluster_jsons/{fullid}/cluster_split_list.json'
            if ferretname == 'Orecchiette':
                original_to_split_cluster_ids = np.array([])
                scores = np.load(f'{saveDir}/scores_{probewordindex_1}_vs_{probewordindex_2}_{ferretname}_probe_bs.npy',
                                 allow_pickle=True)[()]

            else:
                with open(json_file_path, "r") as json_file:
                    loaded_data = json.load(json_file)
                recname = saveDir.split('/')[-3]
                stream_id = stream[-4:]
                if recname == '01_03_2022_cruellabb4bb5':
                    recname = '01_03_2022_cruella'
                elif recname == '25_01_2023_cruellabb4bb5':
                    recname = '25_01_2023_cruella'
                recname_json = loaded_data.get(recname)

                #get the cluster ids from the json file
                original_to_split_cluster_ids = recname_json.get(stream_id)
                original_to_split_cluster_ids = original_to_split_cluster_ids.get('cluster_to_split_list')
                if original_to_split_cluster_ids:
                    #get all the unique clusters ids
                    probewordindex_1 = str(probeword1[0])
                    probewordindex_2 = str(probeword2[0])
                    scores = np.load(f'{saveDir}/scores_{probewordindex_1}_vs_{probewordindex_2}_{ferretname}_probe_bs.npy', allow_pickle=True)[()]

                    original_to_split_cluster_ids = np.unique(scores['talker1']['target_vs_probe']['pitchshift']['cluster_id']+scores['talker1']['target_vs_probe']['nopitchshift']['cluster_id'])
                    original_to_split_cluster_ids = [x for x in original_to_split_cluster_ids if x < 100]
                elif original_to_split_cluster_ids == None:
                    original_to_split_cluster_ids = np.array([])
                    scores = np.load(f'{saveDir}/scores_{probewordindex_1}_vs_{probewordindex_2}_{ferretname}_probe_bs.npy', allow_pickle=True)[()]


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
                dictoutput_instance = scatterplot_and_visualise(probewordlist_l74,
                                                                saveDir=f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist_l74,
                                                                saveDir=f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream = stream, fullid = animal, report = report[animal][stream], permutation_scores=True)
                dictoutput_all_permutation.append(dictoutput_instance_permutation)

            elif animal == 'F1606_Windolene':
                dictoutput_instance = scatterplot_and_visualise(probewordlist_l74,
                                                                saveDir=f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist_l74,
                                                                saveDir=f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream], permutation_scores=True)
                dictoutput_all_permutation.append(dictoutput_instance_permutation)

            elif animal =='F1702_Zola':
                dictoutput_instance = scatterplot_and_visualise(probewordlist_zola, saveDir= f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist_zola, saveDir= f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream]
                                                                            , permutation_scores=True)
                dictoutput_all_permutation.append(dictoutput_instance_permutation)

            elif animal == 'F1815_Cruella' or animal == 'F1902_Eclair':
                dictoutput_instance = scatterplot_and_visualise(probewordlist, saveDir= f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist, saveDir= f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream]
                                                                            , permutation_scores=True)
                dictoutput_all_permutation.append(dictoutput_instance_permutation)

            elif animal == 'F2003_Orecchiette':
                # try:
                dictoutput_instance = scatterplot_and_visualise(probewordlist,
                                                                saveDir=f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream=stream,
                                                                fullid=animal,
                                                                report=report[animal][stream]
                                                                )
                dictoutput_all.append(dictoutput_instance)
                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist,
                                                                saveDir=f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/',
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
                dictoutput_instance = scatterplot_and_visualise(probewordlist, saveDir= f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
                                                                ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream, fullid = animal,  report = report[animal][stream])
                dictoutput_all.append(dictoutput_instance)

                dictoutput_instance_permutation = scatterplot_and_visualise(probewordlist, saveDir= f'G:/results_distvsdist_02022024/{animal}/{rec_name_unique}/{streamtext}/',
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

    return





if __name__ == '__main__':
    main()