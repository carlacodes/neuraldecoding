import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
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

scoremat = np.load(
    'D:/Users/cgriffiths/resultsms4/lstmclass_18112022/18112022_10_58_57/scores_Eclair_2022_2_eclair_probe_pitchshift_vs_not_by_talker_bs.npy',
    allow_pickle=True)[()]

oldscoremat = np.load(
    'D:/Users/juleslebert/home/phd/figures/euclidean_class_082022/eclair/17112022_16_24_15/scores_Eclair_2022_probe_earlylate_left_right_win_bs.npy',
    allow_pickle=True)[()]

testscorematzola = np.load('D:/Users/cgriffiths/resultsms4/lstmclass_CVDATA_11122022zola/11122022_13_17_29/scores_zola_2022_5_zola_probe_pitchshift_vs_not_by_talker_bs.npy', allow_pickle=True)[()]

singleunitlist_cruella = [16, 34, 25, 12, 2, 27, 21, 24, 17, 18, 13, 11, 22, 20, 26]
singleunitlist_cruella_soundonset = [13, 16, 17, 21, 22, 26, 27, 28, 34]
singleunitlist_cruella_2 = [] #unit 25+1 only fires during non pitch shift trials for male talker

multiunitlist_cruella_2 = [21, 40,  44, 29, 43, 31, 22, 5, 4, 18, 10, 13,  30, 6] #, cluster 7+1, and 2+1 doesnt fire for every word, cluster 44+1 doesnt fire during non pitch shift trials

multiunitlist_cruella = [10, 7, 31, 29, 1, 32, 15, 9, 6, 3, 19, 23, 8, 4, 33, 14, 30, 5]
multiunitlist_cruella_soundonset = [6, 8, 9, 14, 23, 29, 30, 21, 33]

singleunitlist_nala = [17, 29, 5, 19, 27, 20, 4, 28, 1, 26, 21, 37] #37
multiunitlist_nala = [10, 24, 8, 15, 12, 7, 9, 35, 2, 14, 34, 33, 32, 38, 39, 31, 40, 41, 13] #13
saveDir = 'D:/Users/cgriffiths/resultsms4/lstmclass_18112022/19112022_12_58_54/'
singlunitlistsoundonset_crumble = [6, 7, 11, 17, 21, 22, 26]
multiunitlist_soundonset_crumble = [13, 14, 23, 25, 27, 29]

singleunitlist_cruella_bb4bb5=[16, 6, 21,5, 8, 33, 27]
multiunitlist_cruella_bb4bb5 =[]


def scatterplot_and_visualise(probewordlist,
                              saveDir='D:/Users/cgriffiths/resultsms4/lstm_output_frommyriad_15012023/lstm_kfold_14012023_crumble',
                              ferretname='Crumble',
                              singleunitlist=singlunitlistsoundonset_crumble,
                              multiunitlist=multiunitlist_soundonset_crumble, noiselist=[], stream = 'BB_2'):
    singleunitlist = [x - 1 for x in singleunitlist]
    multiunitlist = [x - 1 for x in multiunitlist]
    noiselist = [x - 1 for x in noiselist]

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

        probewordindex = probeword[0]
        print(probewordindex)
        stringprobewordindex = str(probewordindex)
        # if ferretname == 'Squinty' or ferretname == 'Windolene':
            #scores_squinty_2022_2_squinty_probe_bs
        scores = np.load(
            saveDir  + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_bs.npy',
            allow_pickle=True)[()]

        #print all the cluster ids for the scores
        print(f'cluster ids for animal:{ferretname}, and stream:{saveDir}')
        print(scores['talker1']['target_vs_probe']['pitchshift']['cluster_id'])
        print(scores['talker1']['target_vs_probe']['nopitchshift']['cluster_id'])

        # else:
        #     scores = np.load(
        #         saveDir  + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_pitchshift_vs_not_by_talker_bs.npy',
        #         allow_pickle=True)[()]


        for talker in [1]:
            comparisons = [comp for comp in scores[f'talker{talker}']]

            for comp in comparisons:
                for cond in ['pitchshift', 'nopitchshift']:
                    for i, clus in enumerate(scores[f'talker{talker}'][comp][cond]['cluster_id']):
                        if isinstance(clus, float):

                            if ferretname == 'F1815_Cruella':
                                if stream == 'BB_2':
                                    if clus == 2.3 or clus == 3.3 or clus == 4.3 or clus == 5.3 or clus==8.3 or clus ==9.2 or clus == 10.3 or clus == 13.1 or clus ==12.3 or clus == 14.3 or clus ==14.1 or clus == 15.3 or clus == 15.1:
                                        noiselist.append(clus)
                                    else:
                                        clus_instance = int(round(clus))
                                        if clus_instance in singleunitlist:
                                            singleunitlist.remove(clus_instance)
                                            noiselist.append(clus_instance)
                                            singleunitlist.append(clus)
                                        elif clus_instance in multiunitlist:
                                            multiunitlist.remove(clus_instance)
                                            noiselist.append(clus_instance)
                                            multiunitlist.append(clus)
                                        else:
                                            multiunitlist.append(clus)
                                elif stream == 'BB_4':
                                    if clus == 4.1 or clus == 11.2 or clus == 14.1 or clus == 15.3:
                                        noiselist.append(clus)
                                    else:
                                        clus_instance = int(round(clus))
                                        if clus_instance in singleunitlist:
                                            singleunitlist.remove(clus_instance)
                                            noiselist.append(clus_instance)
                                            singleunitlist.append(clus)
                                        elif clus_instance in multiunitlist:
                                            multiunitlist.remove(clus_instance)
                                            noiselist.append(clus_instance)
                                            multiunitlist.append(clus)
                                        else:
                                            multiunitlist.append(clus)
                            if ferretname == 'F1901_Crumble':
                                if stream == 'BB_2':
                                    if clus == 4.1 or clus == 4.2 or clus == 7.2 or clus == 7.1 or clus == 8.2 or clus == 12.2 or clus ==14.2 or clus == 15.1:
                                        noiselist.append(clus)
                                    else:
                                        clus_instance = int(round(clus))
                                        if clus_instance in singleunitlist:
                                            singleunitlist.remove(clus_instance)
                                            noiselist.append(clus_instance)
                                            singleunitlist.append(clus)
                                        elif clus_instance in multiunitlist:
                                            multiunitlist.remove(clus_instance)
                                            noiselist.append(clus_instance)
                                            multiunitlist.append(clus)
                                        else:
                                            multiunitlist.append(clus)
                                elif stream == 'BB_3':
                                    if clus == 7.2 or clus == 16.2:
                                        noiselist.append(clus)
                                    else:
                                        clus_instance = int(round(clus))
                                        if clus_instance in singleunitlist:
                                            singleunitlist.remove(clus_instance)
                                            noiselist.append(clus_instance)
                                            singleunitlist.append(clus)
                                        elif clus_instance in multiunitlist:
                                            multiunitlist.remove(clus_instance)
                                            noiselist.append(clus_instance)
                                            multiunitlist.append(clus)
                                        else:
                                            multiunitlist.append(clus)
                                else:
                                    clus_instance = int(round(clus))
                                    if clus_instance in singleunitlist:
                                        singleunitlist.remove(clus_instance)
                                        noiselist.append(clus_instance)
                                        singleunitlist.append(clus)
                                    elif clus_instance in multiunitlist:
                                        multiunitlist.remove(clus_instance)
                                        noiselist.append(clus_instance)
                                        multiunitlist.append(clus)
                                    else:
                                        multiunitlist.append(clus)



                            else:
                                clus_instance = int(round(clus))
                                if clus_instance in singleunitlist:
                                    singleunitlist.remove(clus_instance)
                                    noiselist.append(clus_instance)
                                    singleunitlist.append(clus)
                                elif clus_instance in multiunitlist:
                                    multiunitlist.remove(clus_instance)
                                    noiselist.append(clus_instance)
                                    multiunitlist.append(clus)
                                else:
                                    multiunitlist.append(clus)








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

                        if clus in singleunitlist:
                            print('in single unit list')
                            if cond == 'pitchshift':
                                if talker == 1:
                                    su_pitchshiftlist_female = np.append(su_pitchshiftlist_female,
                                                                         scores[f'talker{talker}'][comp][cond][
                                                                             'lstm_avg'][i])
                                elif talker == 2:
                                    su_pitchshiftlist_male = np.append(su_pitchshiftlist_male,
                                                                       scores[f'talker{talker}'][comp][cond][
                                                                           'lstm_avg'][i])
                                # print(pitchshiftlist.size)
                            elif cond == 'nopitchshift':
                                if talker == 1:

                                    su_nonpitchshiftlist_female = np.append(su_nonpitchshiftlist_female,
                                                                            scores[f'talker{talker}'][comp][cond][
                                                                                'lstm_avg'][i])
                                elif talker == 2:

                                    su_nonpitchshiftlist_male = np.append(su_nonpitchshiftlist_male,
                                                                          scores[f'talker{talker}'][comp][cond][
                                                                              'lstm_avg'][i])

                        elif clus in multiunitlist:
                            if cond == 'pitchshift':
                                if talker == 1:
                                    mu_pitchshiftlist_female = np.append(mu_pitchshiftlist_female,
                                                                         scores[f'talker{talker}'][comp][cond][
                                                                             'lstm_avg'][
                                                                             i])

                                elif talker == 2:
                                    mu_pitchshiftlist_male = np.append(mu_pitchshiftlist_male,
                                                                       scores[f'talker{talker}'][comp][cond][
                                                                           'lstm_avg'][
                                                                           i])
                                    cluster_list_male_mu = np.append(cluster_list_male_mu, clus)


                            if cond == 'nopitchshift':
                                if talker == 1:
                                    mu_nonpitchshiftlist_female = np.append(mu_nonpitchshiftlist_female,
                                                                            scores[f'talker{talker}'][comp][cond][
                                                                                'lstm_avg'][i])

                                elif talker == 2:
                                    mu_nonpitchshiftlist_male = np.append(mu_nonpitchshiftlist_male,
                                                                          scores[f'talker{talker}'][comp][cond][
                                                                              'lstm_avg'][i])
                                    cluster_list_male_mu_nops= np.append(cluster_list_male_mu_nops, clus)


                        elif clus in noiselist:
                            pass

                        # pitchshiftlist = np.append(pitchshiftlist, scores[f'talker{talker}'][comp]['pitchshift']['lstm_avg'][i])
                        # nonpitchshiftlist = np.append(nonpitchshiftlist, scores[f'talker{talker}'][comp]['nopitchshift']['lstm_avg'][i])

                        # plt.title(f'cluster {clus}')
                        # plt.show()

    keys = {"su_list", "mu_list"}
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
    probewordlist_l74 = [(2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
                             (14, 14)]
    # dictoutput_windolene = scatterplot_and_visualise(probewordlist_squinty, saveDir= 'E:/results_16092023\F1606_Windolene/bb5/', ferretname='Windolene', singleunitlist=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], multiunitlist=np.arange(1,64, 1), noiselist = [])

    report_squinty = {}
    singleunitlist_squinty = {}
    multiunitlist_squinty = {}
    noiselist_squinty = {}

    #
    # for side in streams:
    #     report_squinty[side], singleunitlist_squinty[side], multiunitlist_squinty[side], noiselist_squinty[side] = load_classified_report(f'E:\ms4output2\F1604_Squinty\BB2BB3_squinty_MYRIAD2_23092023_58noiseleveledit3medthreshold\BB2BB3_squinty_MYRIAD2_23092023_58noiseleveledit3medthreshold_BB2BB3_squinty_MYRIAD2_23092023_58noiseleveledit3medthreshold_{side}/')

    # animal_list = [  'F1606_Windolene','F1702_Zola','F1604_Squinty', 'F1815_Cruella', 'F1902_Eclair', 'F1901_Crumble']
    animal_list = [ 'F1604_Squinty', 'F1606_Windolene', 'F1702_Zola','F1815_Cruella', 'F1902_Eclair', 'F1812_Nala', 'F1901_Crumble']
    # animal_list = [  'F1815_Cruella', 'F1901_Crumble',]
    # animal_list = [ 'F1604_Squinty', 'F1606_Windolene', 'F1702_Zola','F1815_Cruella', 'F1901_Crumble', 'F1812_Nala']

    #windolene's scores pulling the mean down need to check for noise
    #crumble's scores ridiculloously high, need to check for noise, probably need to check nala as well


    #load the report for each animal in animal-list
    report = {}
    singleunitlist = {}
    multiunitlist = {}
    noiselist = {}
    path_list = {}
    for animal in animal_list:
        path = Path('E:\ms4output2/' + animal + '/')
        path_list[animal] = [path for path in path.glob('**/quality metrics.csv')]
        #get the parent directory of each path
        path_list[animal] = [path.parent for path in path_list[animal]]

    for animal in animal_list:
        report[animal] = {}
        singleunitlist[animal] = {}
        multiunitlist[animal] = {}
        noiselist[animal] = {}

        for path in path_list[animal]:


            stream_name = path.parent.absolute()
            stream_name = stream_name.parent.absolute()
            stream_name = str(stream_name)[-4:]
            #check if stream name exists
            # if stream_name in report[animal].keys():
            #     stream_name = path.parent.absolute()
            #     stream_name = stream_name.parent.absolute()
            #     #find myriad number
            #     stream_name = str(stream_name)[-6:]
            #load the report for that stream
            try:
              report[animal][stream_name], singleunitlist[animal][stream_name], multiunitlist[animal][stream_name], noiselist[animal][stream_name] = load_classified_report(f'{path}')
            except:
                print('no report for this stream:' + str(path))
                pass
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
            #make lowercase
            # animal_text = animal_text.lower()
            # try:
            if animal == 'F1604_Squinty':
                dictoutput_instance = scatterplot_and_visualise(probewordlist_l74,
                                                                saveDir=f'D:/interrovingdecoding/results_16092023/{animal}/myriad3/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream = stream)
            elif animal == 'F1606_Windolene':
                dictoutput_instance = scatterplot_and_visualise(probewordlist_l74,
                                                                saveDir=f'D:/interrovingdecoding/results_16092023/{animal}/{streamtext}/',
                                                                ferretname=animal_text,
                                                                singleunitlist=singleunitlist[animal][stream],
                                                                multiunitlist=multiunitlist[animal][stream],
                                                                noiselist=noiselist[animal][stream], stream = stream)
            else:
                try:
                    dictoutput_instance = scatterplot_and_visualise(probewordlist, saveDir= f'D:/interrovingdecoding/results_16092023/{animal}/{streamtext}/',
                                                                    ferretname=animal_text, singleunitlist=singleunitlist[animal][stream],
                                                                    multiunitlist=multiunitlist[animal][stream], noiselist = noiselist[animal][stream], stream = stream)
                except:
                    #print the exception
                    print(f'no scores for this stream:{stream}, and {animal}')
                    pass
                dictoutput_all.append(dictoutput_instance)
            if animal == 'F1604_Squinty' or animal == 'F1606_Windolene' or animal == 'F1702_Zola' or animal == 'F1815_Cruella':
                print('trained animal'+ animal)
                dictoutput_trained.append(dictoutput_instance)
            else:
                print('naive animal:'+ animal)
                dictoutput_naive.append(dictoutput_instance)

    labels = ['squinty', 'squinty', 'ore']
    colors = ['purple', 'magenta', 'darkturquoise', 'olivedrab', 'steelblue', 'darkcyan']

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




    fig, ax = plt.subplots(1, figsize=(9,9), dpi=300)


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
    for i, (data_dict, label, color) in enumerate(zip(dictlist, labels, colors)):

        #all inter trial roving stuff is female talker
        ax.scatter(data_dict['mu_list']['nonpitchshift']['female_talker'],data_dict['mu_list']['pitchshift']['female_talker'], marker='P',
                   facecolors =color, edgecolors = color, alpha=0.5)
        ax.scatter(data_dict['su_list']['nonpitchshift']['female_talker'],data_dict['su_list']['pitchshift']['female_talker'], marker='P', color=color, alpha=0.5)





    x = np.linspace(0.4, 1, 101)
    ax.plot(x, x, color='black', linestyle = '--')  # identity line

    if bigconcatenatenaive_nonps.size > bigconcatenatenaive_ps.size:


        len(bigconcatenatenaive_ps)
        bigconcatenatenaive_nonps = bigconcatenatenaive_nonps[:bigconcatenatenaive_ps.size]
    elif bigconcatenatenaive_nonps.size < bigconcatenatenaive_ps.size:
        bigconcatenatenaive_ps = bigconcatenatenaive_ps[:bigconcatenatenaive_nonps.size]

    if bigconcatenatetrained_nonps.size > bigconcatenatetrained_ps.size:
        bigconcatenatetrained_nonps = bigconcatenatetrained_nonps[:bigconcatenatetrained_ps.size]
    elif bigconcatenatetrained_nonps.size < bigconcatenatetrained_ps.size:
        bigconcatenatetrained_ps = bigconcatenatetrained_ps[:bigconcatenatetrained_nonps.size]


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
    plt.savefig('D:/scattermuaandsuregplot_mod_21062023.png', dpi=1000)
    plt.savefig('D:/scattermuaandsuregplot_mod_21062023.pdf', dpi=1000)


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
    plt.savefig('D:/diffF0distribution_20062023.png', dpi=1000)
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


    plt.savefig('D:/diffF0distribution_frac_20062023wlegendintertrialroving.png', dpi=1000)
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
    plt.savefig('D:/controlF0distribution20062023intertrialroving.png', dpi=1000)

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
    plt.savefig('D:/rovedF0distribution_20062023intertrialroving.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi = 800)
    ax.set_xlim([0,1])
    sns.distplot(bigconcatenatetrained_ps,  label='trained roved',ax=ax, color='purple')
    sns.distplot(bigconcatenatetrained_nonps,  label='trained control',ax=ax, color='magenta')
    ax.legend(fontsize=18)
    plt.title('Roved and Control F0 Distributions for the Trained Animals', fontsize = 18)
    plt.xlabel(' LSTM decoder scores', fontsize = 20)

    plt.savefig('D:/rovedF0vscontrolF0traineddistribution_20062023intertrialroving.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi = 800)
    ax.set_xlim([0,1])
    sns.distplot(bigconcatenatenaive_ps,  label='naive roved',ax=ax, color='darkcyan')
    sns.distplot(bigconcatenatenaive_nonps,  label='naive control',ax=ax, color='cyan')
    ax.legend(fontsize=18)
    plt.xlabel(' LSTM decoder scores', fontsize = 20)
    plt.title('Roved and Control F0 Distributions for the Naive Animals', fontsize = 18)

    plt.savefig('D:/rovedF0vscontrolF0naivedistribution_20062023intertrialroving.png', dpi=1000)
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

    df = pd.DataFrame(model.summary().tables[1])
    res.tables[0].to_csv("D:/trainedrovescores.csv")

    df2= pd.DataFrame(table)

    csvexport2 = df2.to_csv('D:/trainedrovescores2.csv')

    # Save the DataFrame to a CSV file
    df.to_csv('anova_results.csv', index=False)

    model = ols('scores ~ C(trained) + C(controlF0) ', data=dataset).fit()
    print(model.summary())
    table = sm.stats.anova_lm(model, typ=2)
    # plotting both mu sound driven and single unit units
    # for sutype in mergednaiveanimaldict.keys():



if __name__ == '__main__':
    main()