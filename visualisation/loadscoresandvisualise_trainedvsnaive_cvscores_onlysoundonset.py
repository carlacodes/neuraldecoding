import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import sys
import mergedeep
import seaborn as sns
import scipy.stats as stats
import shap
import lightgbm as lgb
from sklearn.inspection import permutation_importance
import scipy
from scipy.stats import mannwhitneyu

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

scoremat = np.load(
    'D:/Users/cgriffiths/resultsms4/lstmclass_18112022/18112022_10_58_57/scores_Eclair_2022_2_eclair_probe_pitchshift_vs_not_by_talker_bs.npy',
    allow_pickle=True)[()]

oldscoremat = np.load(
    'D:/Users/juleslebert/home/phd/figures/euclidean_class_082022/eclair/17112022_16_24_15/scores_Eclair_2022_probe_earlylate_left_right_win_bs.npy',
    allow_pickle=True)[()]

testscorematzola = np.load('D:/Users/cgriffiths/resultsms4/lstmclass_CVDATA_11122022zola/11122022_13_17_29/scores_zola_2022_5_zola_probe_pitchshift_vs_not_by_talker_bs.npy', allow_pickle=True)[()]
# probewordlist = [(20, 22), (2, 2), (5, 6), (42, 49), (32, 38)]
# probewordlist = [(20, 22)]
# # single unit list and multi unit list differs by animal
# singleunitlist = [8, 11, 17, 15, 27, 29, 31]
# singleunitlist = [6, 10, 30, 5, 26, 3, 11, 12, 21]
# singleunitlist = [19, 18, 6, 10, 30, 5, 2,11, 30]
# multiunitlist = [4, 7, 13, 14, 15, 18, 22]
# multiunitlist = [22, 8, 32, 17, 25, 35, 31, 7, 23, 4, 29, 34, 9, 16, 1, 15, 13, 33, 27, 2, 28, 14, 24]
# noiselist = [1, 2, 5, 9, 19, 24, 6, 20, 21, 23, 16, 28, 12, 30, 32, 33]
singleunitlist_cruella = [16, 34, 25, 12, 2, 27, 21, 24, 17, 18, 13, 11, 22, 20, 26]
singleunitlist_cruella_soundonset = [13, 16, 17, 21, 22, 26, 27, 28, 34]
multiunitlist_cruella = [10, 7, 31, 29, 1, 32, 15, 9, 6, 3, 19, 23, 8, 4, 33, 14, 30, 5]
multiunitlist_cruella_soundonset = [6, 8, 9, 14, 23, 29, 30, 21, 33]

# exclusionlistnala=[25
# 16
# 36
# 6
# 11
# 8
# 3
# 23
# 30
# 22
# ]
singleunitlist_nala = [17, 29, 5, 19, 27, 20, 4, 28, 1, 26, 21, 37] #37
multiunitlist_nala = [10, 24, 8, 15, 12, 7, 9, 35, 2, 14, 34, 33, 32, 38, 39, 31, 40, 41, 13] #13
saveDir = 'D:/Users/cgriffiths/resultsms4/lstmclass_18112022/19112022_12_58_54/'
singlunitlistsoundonset_crumble = [6, 7, 11, 17, 21, 22, 26]
multiunitlist_soundonset_crumble = [13, 14, 23, 25, 27, 29]


def scatterplot_and_visualise(probewordlist,
                              saveDir='D:/Users/cgriffiths/resultsms4/lstm_output_frommyriad_15012023/lstm_kfold_14012023_crumble',
                              ferretname='Crumble',
                              singleunitlist=singlunitlistsoundonset_crumble,
                              multiunitlist=multiunitlist_soundonset_crumble, noiselist=[19, 18, ]):
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

    for probeword in probewordlist:

        probewordindex = probeword[0]
        print(probewordindex)
        stringprobewordindex = str(probewordindex)
        # scores_Eclair_2022_2_eclair_probe_pitchshift_vs_not_by_talker_bs
        # if ferretname == 'Zola':
        #     scores = np.load(
        #         saveDir + '/' + r'scores_Trifle_June_2022_' + stringprobewordindex + '_trifle_probe_pitchshift_vs_not_by_talker_bs.npy',
        #         allow_pickle=True)[()]
        # else:
        scores = np.load(
            saveDir + '/' + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_pitchshift_vs_not_by_talker_bs.npy',
            allow_pickle=True)[()]

        # if ferretname=='Crumble':
        #     scores['lstm_avg'] =[]
        #     # scores = np.load(
        #     #     saveDir + '/' + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_pitchshift_vs_not_by_talker_bs.npy',
        #     #     allow_pickle=True)[()]
        #     for talker in [1, 2]:
        #         comparisons = [comp for comp in scores[f'talker{talker}']]
        #         for comp in comparisons:
        #             for cond in ['pitchshift', 'nopitchshift']:
        #                 scores[f'talker{talker}'][comp][cond]['lstm_avg'] =np.mean(scores[f'talker{talker}'][comp][cond]['lstm_accuracylist'])
        #                 # for i, clus in enumerate(scores[f'talker{talker}'][comp][cond]['cluster_id']):
        #                 #     scores[f'talker{talker}'][comp][cond]['lstm_avg'][i] = np.mean(scores[f'talker{talker}'][comp][cond]['lstm_accuracylist'][i])



        for talker in [1, 2]:
            comparisons = [comp for comp in scores[f'talker{talker}']]

            for comp in comparisons:
                for cond in ['pitchshift', 'nopitchshift']:
                    for i, clus in enumerate(scores[f'talker{talker}'][comp][cond]['cluster_id']):

                        print(i, clus)
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
                                    if scores[f'talker{talker}'][comp][cond][
                                        'lstm_avg'][i] <= 0.4:
                                        print('suspiciously low, check if noise:', clus)
                                elif talker == 2:
                                    mu_pitchshiftlist_male = np.append(mu_pitchshiftlist_male,
                                                                       scores[f'talker{talker}'][comp][cond][
                                                                           'lstm_avg'][
                                                                           i])
                                    if scores[f'talker{talker}'][comp][cond][
                                        'lstm_avg'][i] <= 0.4:
                                        print('suspiciously low check if noise:', clus)

                            if cond == 'nopitchshift':
                                if talker == 1:
                                    mu_nonpitchshiftlist_female = np.append(mu_nonpitchshiftlist_female,
                                                                            scores[f'talker{talker}'][comp][cond][
                                                                                'lstm_avg'][i])
                                    if scores[f'talker{talker}'][comp][cond][
                                        'lstm_avg'][i] <= 0.4:
                                        print('suspiciously low check if noise:', clus)
                                elif talker == 2:
                                    mu_nonpitchshiftlist_male = np.append(mu_nonpitchshiftlist_male,
                                                                          scores[f'talker{talker}'][comp][cond][
                                                                              'lstm_avg'][i])
                                    if scores[f'talker{talker}'][comp][cond][
                                        'lstm_avg'][i] <= 0.4:
                                        print('suspiciously low  check if noise:', clus)
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


def main():
    probewordlist = [(20, 22), (2, 2), (5, 6), (42, 49), (32, 38)]

    dictoutput = scatterplot_and_visualise(
        probewordlist)

    return dictoutput


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
    # fig, ax = plt.gcf(), plt.gca()
    # plt.title('SHAP values for the LGBM Correct Release Times model')
    # for fc in plt.gcf().get_children():
    #     for fcc in fc.get_children():
    #         if hasattr(fcc, "set_cmap"):
    #             fcc.set_cmap(newcmp)

    labels = [item.get_text() for item in ax.get_yticklabels()]
    print(labels)

if __name__ == '__main__':
    probewordlist = [(2, 2), (5, 6), (42, 49), (32, 38), (20, 22)]
    dictoutput_crumble = scatterplot_and_visualise(
        probewordlist)

    dictoutput_eclair = scatterplot_and_visualise(probewordlist,
                                                  saveDir='D:/Users/cgriffiths/resultsms4/lstm_output_frommyriad_15012023/lstm_kfold_14012023_eclair/',
                                                  ferretname='Eclair',
                                                  singleunitlist=[20, 21, 28, 35, 37, 39],
                                                  multiunitlist=[3, 4, 8, 9, 10, 11, 16, 17, 18, 19, 21, 22, 23, 24, 25,
                                                                 26, 27, 38, 21, 21, 33, 34, 40]
                                                  , noiselist=[23]
                                                  )
    dictoutput_zola = scatterplot_and_visualise(probewordlist,
                                                #saveDir='D:/Users/cgriffiths/resultsms4/lstmclass_CVDATA_08122022/08122022_14_40_02/',
                                                saveDir = 'D:/Users/cgriffiths/resultsms4/lstm_output_frommyriad_15012023/lstm_kfold_14012023_zola/',
                                                ferretname='Zola',

                                                singleunitlist=[13, 18, 37, 39],
                                                multiunitlist=[7, 8, 9, 10, 12, 15, 20, 21, 38]
                                                , noiselist=[29, 15, 36, ])
    # D:\Users\cgriffiths\resultsms4\lstmclass_18112022\27112022_21_54_08
    dictoutput_cruella = scatterplot_and_visualise(probewordlist,
                                                   #saveDir='D:/Users/cgriffiths/resultsms4/lstmclass_CVDATA_05122022/06122022_00_40_15/',
                                                   saveDir = 'D:/Users/cgriffiths/resultsms4/lstm_output_frommyriad_15012023/lstm_kfold_14012023_cruella/',
                                                   ferretname='Cruella',
                                                   singleunitlist=singleunitlist_cruella_soundonset,
                                                   multiunitlist=multiunitlist_cruella_soundonset
                                                   , noiselist=[])

    dictoutput_nala = scatterplot_and_visualise(probewordlist,
                                                saveDir='D:/Users/cgriffiths/resultsms4/lstm_output_frommyriad_15012023/lstm_kfold_14012023_nala/',
                                                ferretname='Nala',
                                                singleunitlist=singleunitlist_nala,
                                                multiunitlist=multiunitlist_nala
                                                , noiselist=[])

    meregedtest = cool_dict_merge([dictoutput_crumble, dictoutput_eclair])
    meregedtest = cool_dict_merge([meregedtest, dictoutput_nala])

    mergedtrained = cool_dict_merge([dictoutput_cruella, dictoutput_zola])

    mergedtrained['su_list'] = cool_dict_merge([mergedtrained['su_list'][0], mergedtrained['su_list'][1]])
    mergedtrained['su_list']['pitchshift'] = cool_dict_merge(
        [mergedtrained['su_list']['pitchshift'][0], mergedtrained['su_list']['pitchshift'][1]])
    mergedtrained['su_list']['pitchshift']['female_talker'] = np.concatenate((mergedtrained['su_list']['pitchshift'][
                                                                                  'female_talker'][0],
                                                                              mergedtrained['su_list']['pitchshift'][
                                                                                  'female_talker'][1]))
    mergedtrained['su_list']['pitchshift']['male_talker'] = np.concatenate((mergedtrained['su_list']['pitchshift'][
                                                                                'male_talker'][0],
                                                                            mergedtrained['su_list']['pitchshift'][
                                                                                'male_talker'][1]))

    mergedtrained['su_list']['nonpitchshift'] = cool_dict_merge(
        [mergedtrained['su_list']['nonpitchshift'][0], mergedtrained['su_list']['nonpitchshift'][1]])
    mergedtrained['su_list']['nonpitchshift']['female_talker'] = np.concatenate(
        (mergedtrained['su_list']['nonpitchshift'][
             'female_talker'][0],
         mergedtrained['su_list']['nonpitchshift'][
             'female_talker'][1]))
    mergedtrained['su_list']['nonpitchshift']['male_talker'] = np.concatenate(
        (mergedtrained['su_list']['nonpitchshift'][
             'male_talker'][0],
         mergedtrained['su_list']['nonpitchshift'][
             'male_talker'][1]))

    mergedtrained['mu_list'] = cool_dict_merge([mergedtrained['mu_list'][0], mergedtrained['mu_list'][1]])

    mergedtrained['mu_list']['nonpitchshift'] = cool_dict_merge(
        [mergedtrained['mu_list']['nonpitchshift'][0], mergedtrained['mu_list']['nonpitchshift'][1]])
    mergedtrained['mu_list']['nonpitchshift']['female_talker'] = np.concatenate(
        (mergedtrained['mu_list']['nonpitchshift'][
             'female_talker'][0],
         mergedtrained['mu_list']['nonpitchshift'][
             'female_talker'][1]))
    mergedtrained['mu_list']['nonpitchshift']['male_talker'] = np.concatenate(
        (mergedtrained['mu_list']['nonpitchshift'][
             'male_talker'][0],
         mergedtrained['mu_list']['nonpitchshift'][
             'male_talker'][1]))

    mergedtrained['mu_list']['pitchshift'] = cool_dict_merge(
        [mergedtrained['mu_list']['pitchshift'][0], mergedtrained['mu_list']['pitchshift'][1]])
    mergedtrained['mu_list']['pitchshift']['female_talker'] = np.concatenate((mergedtrained['mu_list']['pitchshift'][
                                                                                  'female_talker'][0],
                                                                              mergedtrained['mu_list']['pitchshift'][
                                                                                  'female_talker'][1]))
    mergedtrained['mu_list']['pitchshift']['male_talker'] = np.concatenate((mergedtrained['mu_list']['pitchshift'][
                                                                                'male_talker'][0],
                                                                            mergedtrained['mu_list']['pitchshift'][
                                                                                'male_talker'][1]))

    ##
    meregedtest['su_list'] = cool_dict_merge(
        [meregedtest['su_list'][0], meregedtest['su_list'][1], meregedtest['su_list'][2]])
    meregedtest['su_list']['pitchshift'] = cool_dict_merge(
        [meregedtest['su_list']['pitchshift'][0], meregedtest['su_list']['pitchshift'][1],
         meregedtest['su_list']['pitchshift'][2]])
    meregedtest['su_list']['pitchshift']['female_talker'] = np.concatenate((meregedtest['su_list']['pitchshift'][
                                                                                'female_talker'][0],
                                                                            meregedtest['su_list']['pitchshift'][
                                                                                'female_talker'][1],
                                                                            meregedtest['su_list']['pitchshift'][
                                                                                'female_talker'][2]))
    meregedtest['su_list']['pitchshift']['male_talker'] = np.concatenate((meregedtest['su_list']['pitchshift'][
                                                                              'male_talker'][0],
                                                                          meregedtest['su_list']['pitchshift'][
                                                                              'male_talker'][1],
                                                                          meregedtest['su_list']['pitchshift'][
                                                                              'male_talker'][2]))

    meregedtest['su_list']['nonpitchshift'] = cool_dict_merge(
        [meregedtest['su_list']['nonpitchshift'][0], meregedtest['su_list']['nonpitchshift'][1],
         meregedtest['su_list']['nonpitchshift'][2]])
    meregedtest['su_list']['nonpitchshift']['female_talker'] = np.concatenate((meregedtest['su_list']['nonpitchshift'][
                                                                                   'female_talker'][0],
                                                                               meregedtest['su_list']['nonpitchshift'][
                                                                                   'female_talker'][1],
                                                                               meregedtest['su_list']['nonpitchshift'][
                                                                                   'female_talker'][2]))
    meregedtest['su_list']['nonpitchshift']['male_talker'] = np.concatenate((meregedtest['su_list']['nonpitchshift'][
                                                                                 'male_talker'][0],
                                                                             meregedtest['su_list']['nonpitchshift'][
                                                                                 'male_talker'][1],
                                                                             meregedtest['su_list']['nonpitchshift'][
                                                                                 'male_talker'][2]))

    meregedtest['mu_list'] = cool_dict_merge(
        [meregedtest['mu_list'][0], meregedtest['mu_list'][1], meregedtest['mu_list'][2]])

    meregedtest['mu_list']['nonpitchshift'] = cool_dict_merge(
        [meregedtest['mu_list']['nonpitchshift'][0], meregedtest['mu_list']['nonpitchshift'][1],
         meregedtest['mu_list']['nonpitchshift'][2]])
    meregedtest['mu_list']['nonpitchshift']['female_talker'] = np.concatenate((meregedtest['mu_list']['nonpitchshift'][
                                                                                   'female_talker'][0],
                                                                               meregedtest['mu_list']['nonpitchshift'][
                                                                                   'female_talker'][1],
                                                                               meregedtest['mu_list']['nonpitchshift'][
                                                                                   'female_talker'][2]))
    meregedtest['mu_list']['nonpitchshift']['male_talker'] = np.concatenate((meregedtest['mu_list']['nonpitchshift'][
                                                                                 'male_talker'][0],
                                                                             meregedtest['mu_list']['nonpitchshift'][
                                                                                 'male_talker'][1],
                                                                             meregedtest['mu_list']['nonpitchshift'][
                                                                                 'male_talker'][2]))

    meregedtest['mu_list']['pitchshift'] = cool_dict_merge(
        [meregedtest['mu_list']['pitchshift'][0], meregedtest['mu_list']['pitchshift'][1],
         meregedtest['mu_list']['pitchshift'][2]])
    meregedtest['mu_list']['pitchshift']['female_talker'] = np.concatenate((meregedtest['mu_list']['pitchshift'][
                                                                                'female_talker'][0],
                                                                            meregedtest['mu_list']['pitchshift'][
                                                                                'female_talker'][1],
                                                                            meregedtest['mu_list']['pitchshift'][
                                                                                'female_talker'][2]))
    meregedtest['mu_list']['pitchshift']['male_talker'] = np.concatenate((meregedtest['mu_list']['pitchshift'][
                                                                              'male_talker'][0],
                                                                          meregedtest['mu_list']['pitchshift'][
                                                                              'male_talker'][1],
                                                                          meregedtest['mu_list']['pitchshift'][
                                                                              'male_talker'][2]))

    mergednaiveanimaldict = {**dictoutput_eclair, **dictoutput_crumble}
    pdoutput_eclair = pd.DataFrame.from_dict(dictoutput_eclair)
    pdoutput_crumble = pd.DataFrame.from_dict(dictoutput_crumble)

    for sutype in mergednaiveanimaldict.keys():
        fig, ax = plt.subplots(2, figsize=(5, 6))
        count = 0
        print(sutype)
        for pitchshiftornot in mergednaiveanimaldict[sutype].keys():

            print('mean score for zola male ' + pitchshiftornot + sutype + 'is' + str(
                np.mean(dictoutput_zola[sutype][pitchshiftornot]['male_talker'])) + 'and std is' + str(
                np.std(dictoutput_zola[sutype][pitchshiftornot]['male_talker'])))
            print('mean score for zola female ' + pitchshiftornot + sutype + 'is' + str(
                np.mean(dictoutput_zola[sutype][pitchshiftornot]['female_talker'])) + 'and std is' + str(
                np.std(dictoutput_zola[sutype][pitchshiftornot]['female_talker'])))

            print('mean score for zola for both talkers' + pitchshiftornot + sutype + 'is' + str(np.mean(np.concatenate(
                (dictoutput_zola[sutype][pitchshiftornot]['male_talker'],
                 dictoutput_zola[sutype][pitchshiftornot]['female_talker'])))) + 'and std is' + str(np.std(
                np.concatenate((dictoutput_zola[sutype][pitchshiftornot]['male_talker'],
                                dictoutput_zola[sutype][pitchshiftornot]['female_talker'])))))

            ax[count].boxplot([dictoutput_zola[sutype][pitchshiftornot]['female_talker'],
                               dictoutput_zola[sutype][pitchshiftornot]['male_talker']])
            ax[count].legend()
            if count == 1:
                ax[count].set_xlabel('talker', fontsize=12)
                ax[count].set_xticklabels(['female', 'male'])
            else:
                ax[count].tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)
            ax[count].set_ylabel('LSTM decoding score', fontsize=12)
            if sutype == 'su_list':
                stringtitle = 'single'
            else:
                stringtitle = 'multi'
            if pitchshiftornot == 'pitchshift':
                stringtitlepitch = 'F0-roved'
            else:
                stringtitlepitch = 'control F0'
            ax[count].set_title('Trained LSTM scores for ' + stringtitle + ' units, \n' + stringtitlepitch + ' trials')
            y = dictoutput_zola[sutype][pitchshiftornot]['female_talker']
            # Add some random "jitter" to the x-axis
            x = np.random.normal(1, 0.04, size=len(y))
            ax[count].plot(x, y, 'b.', alpha=0.2)

            y2 = dictoutput_zola[sutype][pitchshiftornot]['male_talker']
            x = np.random.normal(2, 0.04, size=len(y2))
            ax[count].plot(x, y2, 'g.', alpha=0.2)
            ax[count].set_ylim([0, 1])

            if count == 1:
                ax[count].legend()
            count += 1
    fig.tight_layout()
    plt.ylim(0, 1)
    plt.show()

    for sutype in mergednaiveanimaldict.keys():
        fig, ax = plt.subplots(2, figsize=(5, 6))
        count = 0

        for pitchshiftornot in mergednaiveanimaldict[sutype].keys():
            ax[count].boxplot([meregedtest[sutype][pitchshiftornot]['female_talker'],
                               meregedtest[sutype][pitchshiftornot]['male_talker']])
            ax[count].legend()
            ax[count].set_ylabel('LSTM decoding score')
            if sutype == 'su_list':
                stringtitle = 'single'
            else:
                stringtitle = 'multi'
            if pitchshiftornot == 'pitchshift':
                stringtitlepitch = 'F0-roved'
            else:
                stringtitlepitch = 'control F0'
            ax[count].set_title('Naive LSTM scores for ' + stringtitle + ' units,\n ' + stringtitlepitch + ' trials')
            y_crum = dictoutput_crumble[sutype][pitchshiftornot]['female_talker']
            y_eclair = dictoutput_eclair[sutype][pitchshiftornot]['female_talker']
            # Add some random "jitter" to the x-axis
            x = np.random.normal(1, 0.04, size=len(y_crum))
            x2 = np.random.normal(1, 0.04, size=len(y_eclair))
            ax[count].plot(x, y_crum, ".", color='mediumblue', alpha=0.2, label='F1901')
            ax[count].plot(x2, y_eclair, ".", color='darkorange', alpha=0.2, label='F1902')
            print('mean score for naive male ' + pitchshiftornot + sutype + 'is' + str(
                np.mean(meregedtest[sutype][pitchshiftornot]['male_talker'])) + 'and std is' + str(
                np.std(meregedtest[sutype][pitchshiftornot]['male_talker'])))
            print('mean score for naive female ' + pitchshiftornot + sutype + 'is' + str(
                np.mean(meregedtest[sutype][pitchshiftornot]['female_talker'])) + 'and std is' + str(
                np.std(meregedtest[sutype][pitchshiftornot]['female_talker'])))
            print('mean score for naive for both talkers' + pitchshiftornot + sutype + 'is' + str(np.mean(
                np.concatenate((meregedtest[sutype][pitchshiftornot]['male_talker'],
                                meregedtest[sutype][pitchshiftornot]['female_talker'])))) + 'and std is' + str(np.std(
                np.concatenate((meregedtest[sutype][pitchshiftornot]['male_talker'],
                                meregedtest[sutype][pitchshiftornot]['female_talker'])))))

            y2_crum = dictoutput_crumble[sutype][pitchshiftornot]['male_talker']
            y2_eclair = dictoutput_eclair[sutype][pitchshiftornot]['male_talker']

            x = np.random.normal(2, 0.04, size=len(y2_crum))
            x2 = np.random.normal(2, 0.04, size=len(y2_eclair))
            if count == 1:
                ax[count].set_xlabel('talker', fontsize=12)
                ax[count].set_xticklabels(['female', 'male'])
            else:
                ax[count].tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)

            ax[count].plot(x, y2_crum, ".", color='mediumblue', alpha=0.2, )
            ax[count].plot(x2, y2_eclair, ".", color='darkorange', alpha=0.2)
            ax[count].set_ylim([0, 1])
            if count == 1:
                ax[count].legend()
            count += 1
    fig.tight_layout()
    plt.ylim(0, 1)

    plt.show()
    # plotting both mu sound driven and single unit units FOR TRAINED ANIMALS
    # for sutype in mergednaiveanimaldict.keys():
    fig, ax = plt.subplots(2, figsize=(5, 8))
    count = 0

    for pitchshiftornot in mergedtrained[sutype].keys():
        ax[count].boxplot([mergedtrained['su_list'][pitchshiftornot]['female_talker'],
                           mergedtrained['mu_list'][pitchshiftornot]['female_talker'],
                           mergedtrained['su_list'][pitchshiftornot]['male_talker'],
                           mergedtrained['mu_list'][pitchshiftornot]['male_talker']])
        ax[count].legend()
        ax[count].set_ylabel('LSTM decoding score (%)')

        # as @ali14 pointed out, for python3, use this
        # for sp in ax2.spines.values():
        # and for python2, use this

        if sutype == 'su_list':
            stringtitle = 'single'
        else:
            stringtitle = 'multi'
        if pitchshiftornot == 'pitchshift':
            stringtitlepitch = 'F0-roved'
        else:
            stringtitlepitch = 'control F0'
        ax[count].set_title('Trained LSTM scores for' + ' single and multi-units,\n ' + stringtitlepitch + ' trials')
        y_zola_su = dictoutput_zola['su_list'][pitchshiftornot]['female_talker']
        y_zola_mu = dictoutput_zola['mu_list'][pitchshiftornot]['female_talker']

        y_cruella_su = dictoutput_cruella['su_list'][pitchshiftornot]['female_talker']
        y_cruella_mu = dictoutput_cruella['mu_list'][pitchshiftornot]['female_talker']

        y2_zola_su_male = dictoutput_zola['su_list'][pitchshiftornot]['male_talker']
        y2_cruella_su_male = dictoutput_cruella['su_list'][pitchshiftornot]['male_talker']

        y2_zola_mu_male = dictoutput_zola['mu_list'][pitchshiftornot]['male_talker']
        y2_cruella_mu_male = dictoutput_cruella['mu_list'][pitchshiftornot]['male_talker']

        # Add some random "jitter" to the x-axis
        x_su = np.random.normal(1, 0.04, size=len(y_zola_su))
        x2_su = np.random.normal(1, 0.04, size=len(y_cruella_su))

        x_mu = np.random.normal(2, 0.04, size=len(y_zola_mu))
        x2_mu = np.random.normal(2, 0.04, size=len(y_cruella_mu))

        x_su_male = np.random.normal(3, 0.04, size=len(y2_zola_su_male))
        x2_su_male = np.random.normal(3, 0.04, size=len(y2_cruella_su_male))

        x_mu_male = np.random.normal(4, 0.04, size=len(y2_zola_mu_male))
        x2_mu_male = np.random.normal(4, 0.04, size=len(y2_cruella_mu_male))

        ax[count].plot(x_su, y_zola_su, ".", color='hotpink', alpha=0.2, )
        ax[count].plot(x2_su, y_cruella_su, ".", color='olivedrab', alpha=0.2, )

        ax[count].plot(x_mu, y_zola_mu, ".", color='hotpink', alpha=0.2)
        ax[count].plot(x2_mu, y_cruella_mu, ".", color='olivedrab', alpha=0.2)

        ax[count].plot(x_su_male, y2_zola_su_male, ".", color='hotpink', alpha=0.2)
        ax[count].plot(x2_su_male, y2_cruella_su_male, ".", color='olivedrab', alpha=0.2)

        ax[count].plot(x_mu_male, y2_zola_mu_male, ".", color='hotpink', alpha=0.2, label='F1702')
        ax[count].plot(x2_mu_male, y2_cruella_mu_male, ".", color='olivedrab', alpha=0.2, label='F1815')

        print('mean score for TRAINED male ' + pitchshiftornot + sutype + 'is' + str(
            np.mean(mergedtrained[sutype][pitchshiftornot]['male_talker'])) + 'and std is' + str(
            np.std(mergedtrained[sutype][pitchshiftornot]['male_talker'])))
        print('mean score for TRAINED female ' + pitchshiftornot + sutype + 'is' + str(
            np.mean(mergedtrained[sutype][pitchshiftornot]['female_talker'])) + 'and std is' + str(
            np.std(mergedtrained[sutype][pitchshiftornot]['female_talker'])))
        print('mean score forTRAINED for both talkers' + pitchshiftornot + sutype + 'is' + str(np.mean(np.concatenate((
            mergedtrained[
                sutype][
                pitchshiftornot][
                'male_talker'],
            mergedtrained[
                sutype][
                pitchshiftornot][
                'female_talker'])))) + 'and std is' + str(
            np.std(np.concatenate((mergedtrained[sutype][pitchshiftornot]['male_talker'],
                                   mergedtrained[sutype][pitchshiftornot]['female_talker'])))))

        x = np.random.normal(2, 0.04, size=len(y2_crum))
        x2 = np.random.normal(2, 0.04, size=len(y2_eclair))
        if count == 1:
            ax2 = ax[count].twiny()
            # Offset the twin axis below the host
            ax2.xaxis.set_ticks_position("bottom")
            ax2.xaxis.set_label_position("bottom")

            # Offset the twin axis below the host
            ax2.spines["bottom"].set_position(("axes", -0.15))

            # Turn on the frame for the twin axis, but then hide all
            # but the bottom spine
            ax2.set_frame_on(True)
            ax2.patch.set_visible(False)

            # as @ali14 pointed out, for python3, use this
            # for sp in ax2.spines.values():
            # and for python2, use this
            for sp in ax2.spines.values():
                sp.set_visible(False)
            ax2.spines["bottom"].set_visible(True)

            ax[count].set_xticklabels(['SU', 'MUA', 'SU', 'MUA'], fontsize=12)
            ax2.set_xlabel("talker", fontsize=12)
            ax2.set_xticks([0.2, 0.8])
            ax2.set_xticklabels(["female", "male"], fontsize=12)

        #            ax[count].set_xticklabels(['female', 'male'])
        else:
            ax[count].tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)

        # ax[count].plot(x, y2_crum, ".", color='mediumturquoise', alpha=0.2, )
        # ax[count].plot(x2, y2_eclair, ".", color='darkorange', alpha=0.2)
        ax[count].set_ylim([0, 1])
        if count == 1:
            ax[count].legend(prop={'size': 12})
        count += 1
    fig.tight_layout()
    plt.ylim(0, 1)

    plt.show()

    ##now plot this, but flattened
    # plotting both mu sound driven and single unit units FOR TRAINED ANIMALS
    # for sutype in mergednaiveanimaldict.keys():
    mergedtrained_converted = pd.DataFrame.from_dict(mergedtrained)
    p = 'D:/dfformixedmodels/'
    from pathlib import Path
    filepath = Path('D:/dfformixedmodels/mergedtrained.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)



    fig, ax = plt.subplots(1, figsize=(5, 8))
    count = 0
    emptydict = {}
    dictlist = [dictoutput_cruella, dictoutput_zola, dictoutput_nala, dictoutput_crumble, dictoutput_eclair]
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
                            emptydict['su'] =  emptydict.get('su', []) + [0]
        count += 1
    for keys in emptydict.keys():
        emptydict[keys] = np.asarray(emptydict[keys])

    mergedtrainedandnaive=pd.DataFrame.from_dict(emptydict)
    runboostedregressiontreeforlstmscore(mergedtrainedandnaive)

    from pathlib import Path
    filepath = Path('D:/dfformixedmodels/mergedtrainedandnaive.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    mergedtrainedandnaive.to_csv(filepath)

    female_talker_nopitchshift = np.concatenate((mergedtrained['su_list']['nonpitchshift']['female_talker'],
                                                 mergedtrained['mu_list']['nonpitchshift']['female_talker']))

    female_talker_pitchshift = np.concatenate((mergedtrained['su_list']['pitchshift']['female_talker'],
                                               mergedtrained['mu_list']['pitchshift']['female_talker']))

    male_talker_nopitchshift = np.concatenate((mergedtrained['su_list']['nonpitchshift']['male_talker'],
                                               mergedtrained['mu_list']['nonpitchshift']['female_talker']))

    male_talker_pitchshift = np.concatenate((mergedtrained['su_list']['pitchshift']['male_talker'],
                                             mergedtrained['mu_list']['pitchshift']['female_talker']))

    ax.boxplot([female_talker_nopitchshift,
                female_talker_pitchshift, male_talker_nopitchshift, male_talker_pitchshift])
    ax.legend()
    ax.set_ylabel('LSTM decoding score (%)')
    # ax[count].set_yticklabels([0, 20, 40, 60, 80, 100])

    # if sutype == 'su_list':
    #     stringtitle = 'single'
    # else:
    #     stringtitle = 'multi'
    # if pitchshiftornot == 'pitchshift':
    #     stringtitlepitch = 'F0-roved'
    # else:
    #     stringtitlepitch = 'control F0'
    ax.set_title('Trained LSTM scores for' + ' F0 control vs. roved,\n ' + ' trained animals', fontsize=12)
    y_zola_su = dictoutput_zola['su_list']['nonpitchshift']['female_talker']
    y_zola_mu = dictoutput_zola['mu_list']['nonpitchshift']['female_talker']

    y_cruella_su = dictoutput_cruella['su_list']['nonpitchshift']['female_talker']
    y_cruella_mu = dictoutput_cruella['mu_list']['nonpitchshift']['female_talker']

    y2_zola_su_male = dictoutput_zola['su_list']['nonpitchshift']['male_talker']
    y2_cruella_su_male = dictoutput_cruella['su_list']['nonpitchshift']['male_talker']

    y2_zola_mu_male = dictoutput_zola['mu_list']['nonpitchshift']['male_talker']
    y2_cruella_mu_male = dictoutput_cruella['mu_list']['nonpitchshift']['male_talker']

    # Add some random "jitter" to the x-axis
    x_su = np.random.normal(1, 0.04, size=len(y_zola_su))
    x2_su = np.random.normal(1, 0.04, size=len(y_cruella_su))

    x_mu = np.random.normal(1, 0.04, size=len(y_zola_mu))
    x2_mu = np.random.normal(1, 0.04, size=len(y_cruella_mu))

    x_su_male = np.random.normal(3, 0.04, size=len(y2_zola_su_male))
    x2_su_male = np.random.normal(3, 0.04, size=len(y2_cruella_su_male))

    x_mu_male = np.random.normal(3, 0.04, size=len(y2_zola_mu_male))
    x2_mu_male = np.random.normal(3, 0.04, size=len(y2_cruella_mu_male))

    ax.plot(x_su, y_zola_su, ".", color='hotpink', alpha=0.5, )
    ax.plot(x2_su, y_cruella_su, ".", color='olivedrab', alpha=0.5, )

    ax.plot(x_mu, y_zola_mu, "2", color='hotpink', alpha=0.5)
    ax.plot(x2_mu, y_cruella_mu, "2", color='olivedrab', alpha=0.5)

    ax.plot(x_su_male, y2_zola_su_male, ".", color='hotpink', alpha=0.5, label='F1702 - SU')
    ax.plot(x2_su_male, y2_cruella_su_male, ".", color='olivedrab', alpha=0.5, label='F1815 - SU')

    ax.plot(x_mu_male, y2_zola_mu_male, "2", color='hotpink', alpha=0.5, label='F1702 - MUA')
    ax.plot(x2_mu_male, y2_cruella_mu_male, "2", color='olivedrab', alpha=0.5, label='F1815 - MUA')

    x = np.random.normal(2, 0.04, size=len(y2_crum))
    x2 = np.random.normal(2, 0.04, size=len(y2_eclair))

    if count == 0:
        ax2 = ax.twiny()
        # Offset the twin axis below the host
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        # Offset the twin axis below the host
        ax2.spines["bottom"].set_position(("axes", -0.15))

        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)

        # as @ali14 pointed out, for python3, use this
        # for sp in ax2.spines.values():
        # and for python2, use this
        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.spines["bottom"].set_visible(True)

        ax.set_xticklabels(['F0 Control', 'F0 Roved', 'F0 Control', 'F0 Roved'], fontsize=12)
        ax2.set_xlabel("talker", fontsize=12)
        ax2.set_xticks([0.2, 0.8])
        ax2.set_xticklabels(["female", "male"], fontsize=12)

    #            ax[count].set_xticklabels(['female', 'male'])
    else:
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)

    # ax[count].plot(x, y2_crum, ".", color='mediumturquoise', alpha=0.2, )
    # ax[count].plot(x2, y2_eclair, ".", color='darkorange', alpha=0.2)
    ax.set_ylim([0, 1])
    if count == 0:
        ax.legend(prop={'size': 12})
    count += 1
    fig.tight_layout()
    plt.ylim(0, 1)

    fig, ax = plt.subplots(1, figsize=(9,9), dpi=300)
    count = 0
    emptydict = {}

    female_talker_nopitchshift = np.concatenate((mergedtrained['su_list']['nonpitchshift']['female_talker'],
                                                 mergedtrained['mu_list']['nonpitchshift']['female_talker']))

    female_talker_pitchshift = np.concatenate((mergedtrained['su_list']['pitchshift']['female_talker'],
                                               mergedtrained['mu_list']['pitchshift']['female_talker']))

    male_talker_nopitchshift = np.concatenate((mergedtrained['su_list']['nonpitchshift']['male_talker'],
                                               mergedtrained['mu_list']['nonpitchshift']['female_talker']))

    male_talker_pitchshift = np.concatenate((mergedtrained['su_list']['pitchshift']['male_talker'],
                                             mergedtrained['mu_list']['pitchshift']['female_talker']))

    bigconcatenatetrained = np.concatenate(
        (female_talker_nopitchshift, female_talker_pitchshift, male_talker_nopitchshift, male_talker_pitchshift))
    # if len(dictoutput_cruella['su_list']['nonpitchshift']['female_talker']) < len(dictoutput_cruella['su_list']['pitchshift']['female_talker']):
    #     dictoutput_cruella['su_list']['pitchshift']['female_talker'] = dictoutput_cruella['su_list']['pitchshift']['female_talker'][:len(dictoutput_cruella['su_list']['nonpitchshift']['female_talker'])]
    # elif len(dictoutput_cruella['su_list']['nonpitchshift']['female_talker']) > len(dictoutput_cruella['su_list']['pitchshift']['female_talker']):
    #     dictoutput_cruella['su_list']['nonpitchshift']['female_talker'] = dictoutput_cruella['su_list']['nonpitchshift']['female_talker'][:len(dictoutput_cruella['su_list']['pitchshift']['female_talker'])]
    dictlist = [dictoutput_cruella, dictoutput_zola, dictoutput_nala, dictoutput_crumble, dictoutput_eclair]
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
    dictlist = [dictoutput_cruella, dictoutput_zola]
    bigconcatenatetrained_ps = np.empty(0)
    bigconcatenatetrained_nonps = np.empty(0)
    for dictouput in dictlist:
        for key in dictouput.keys():
            for key3 in dictouput[key]['pitchshift'].keys():
                bigconcatenatetrained_ps = np.concatenate(
                    (bigconcatenatetrained_ps, dictouput[key]['pitchshift'][key3]))
                bigconcatenatetrained_nonps = np.concatenate(
                    (bigconcatenatetrained_nonps, dictouput[key]['nonpitchshift'][key3]))

    dictlist = [dictoutput_nala, dictoutput_crumble, dictoutput_eclair]

    bigconcatenatenaive_ps = np.empty(0)
    bigconcatenatenaive_nonps = np.empty(0)
    for dictouput in dictlist:
        for key in dictouput.keys():
            print(key, 'key')

            for key3 in dictouput[key]['pitchshift'].keys():
                print(key3, 'key3')
                bigconcatenatenaive_ps = np.concatenate((bigconcatenatenaive_ps, dictouput[key]['pitchshift'][key3]))
                bigconcatenatenaive_nonps = np.concatenate(
                    (bigconcatenatenaive_nonps, dictouput[key]['nonpitchshift'][key3]))

    ax.plot(dictoutput_cruella['su_list']['nonpitchshift']['female_talker'],
            dictoutput_cruella['su_list']['pitchshift']['female_talker'], 'o', color='purple', alpha=0.5, label = 'F1815')

    ax.plot(dictoutput_zola['su_list']['nonpitchshift']['female_talker'],
            dictoutput_zola['su_list']['pitchshift']['female_talker'], 'o', color='magenta', alpha=0.5,label = 'F1702')
    ax.plot(dictoutput_cruella['mu_list']['nonpitchshift']['female_talker'],
            dictoutput_cruella['mu_list']['pitchshift']['female_talker'], 'P', color='purple', alpha=0.5)
    ax.plot(dictoutput_zola['mu_list']['nonpitchshift']['female_talker'],
            dictoutput_zola['mu_list']['pitchshift']['female_talker'], 'P', color='magenta', alpha=0.5)

    ax.scatter(dictoutput_cruella['su_list']['nonpitchshift']['male_talker'],
               dictoutput_cruella['su_list']['pitchshift']['male_talker'], marker='o', facecolors='none',
               edgecolors='purple', alpha=0.5)
    ax.scatter(dictoutput_zola['su_list']['nonpitchshift']['male_talker'],
               dictoutput_zola['su_list']['pitchshift']['male_talker'], marker='o', facecolors='none',
               edgecolors='magenta', alpha=0.5)

    ax.scatter(dictoutput_cruella['mu_list']['nonpitchshift']['male_talker'],
               dictoutput_cruella['mu_list']['pitchshift']['male_talker'], marker='P', facecolors='none',
               edgecolors='purple', alpha=0.5)
    ax.scatter(dictoutput_zola['mu_list']['nonpitchshift']['male_talker'],
               dictoutput_zola['mu_list']['pitchshift']['male_talker'], marker='P', facecolors='none',
               edgecolors='magenta', alpha=0.5)



    ax.scatter(dictoutput_eclair['su_list']['nonpitchshift']['female_talker'],
               dictoutput_eclair['su_list']['pitchshift']['female_talker'], marker='o', color='steelblue', alpha=0.5)

    ax.scatter(dictoutput_eclair['mu_list']['nonpitchshift']['female_talker'],
               dictoutput_eclair['mu_list']['pitchshift']['female_talker'], marker='P', color='steelblue', alpha=0.5, label = 'F1902')

    ax.scatter(dictoutput_nala['mu_list']['nonpitchshift']['female_talker'],
               dictoutput_nala['mu_list']['pitchshift']['female_talker'], marker='P', facecolors='none',
               edgecolors='darkturquoise', alpha=0.5)

    ax.scatter(dictoutput_eclair['su_list']['nonpitchshift']['male_talker'],
               dictoutput_eclair['su_list']['pitchshift']['male_talker'], marker='o', facecolors='none',
               edgecolors='steelblue', alpha=0.5)

    ax.scatter(dictoutput_crumble['su_list']['nonpitchshift']['male_talker'],
               dictoutput_crumble['su_list']['pitchshift']['male_talker'], marker='o', facecolors='darkcyan',
               edgecolors='darkcyan', alpha=0.5)

    ax.scatter(dictoutput_crumble['mu_list']['nonpitchshift']['male_talker'],
               dictoutput_crumble['mu_list']['pitchshift']['male_talker'], marker='P', facecolors='none',
               edgecolors='darkcyan', alpha=0.5)

    ax.scatter(dictoutput_crumble['su_list']['nonpitchshift']['female_talker'],
               dictoutput_crumble['su_list']['pitchshift']['female_talker'], marker='o', facecolors='darkcyan',
               edgecolors='darkcyan', alpha=0.5, label = 'F1901')

    ax.scatter(dictoutput_crumble['mu_list']['nonpitchshift']['female_talker'],
               dictoutput_crumble['mu_list']['pitchshift']['female_talker'], marker='P', color='darkcyan', alpha=0.5)

    ax.scatter(dictoutput_nala['su_list']['nonpitchshift']['male_talker'],
               dictoutput_nala['su_list']['pitchshift']['male_talker'], marker='o', facecolors='darkturquoise',
               edgecolors='darkturquoise', alpha=0.5, )

    ax.scatter(dictoutput_eclair['mu_list']['nonpitchshift']['male_talker'],
               dictoutput_eclair['mu_list']['pitchshift']['male_talker'], marker='P', facecolors='none',
               edgecolors='steelblue', alpha=0.5)

    ax.plot(dictoutput_nala['su_list']['nonpitchshift']['female_talker'],
            dictoutput_nala['su_list']['pitchshift']['female_talker'], 'o', color='darkturquoise', alpha=0.5, label='F1812 SU, F')
    ax.scatter(dictoutput_nala['mu_list']['nonpitchshift']['male_talker'],
               dictoutput_nala['mu_list']['pitchshift']['male_talker'], marker='P', facecolors='none',
               edgecolors='darkturquoise', alpha=0.5, label='F1812 MU, M')

    #x.set_ylim([0, 1])
    #ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', linestyle='--')
    x = np.linspace(0.4, 1, 101)
    ax.plot(x, x, color='black', linestyle = '--')  # identity line
    #ax.set_xlim([0, 1])


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
    plt.savefig('D:/scattermuaandsuregplot_mod_17012023.png', dpi=1000)

    plt.show()

    #histogram distribution of the trained and naive animals
    fig, ax = plt.subplots(1, figsize=(8, 8))
    #relativescoretrained = abs(bigconcatenatetrained_nonps - bigconcatenatetrained_ps)/ bigconcatenatetrained_ps

    relativescoretrained = [bigconcatenatetrained_nonps - bigconcatenatetrained_ps for bigconcatenatetrained_nonps, bigconcatenatetrained_ps in zip(bigconcatenatetrained_nonps, bigconcatenatetrained_ps)]
    relativescorenaive = [bigconcatenatenaive_nonps - bigconcatenatenaive_ps for bigconcatenatenaive_nonps, bigconcatenatenaive_ps in zip(bigconcatenatenaive_ps, bigconcatenatenaive_nonps)]
    relativescoretrainedfrac = [relativescoretrained / bigconcatenatetrained_nonps for relativescoretrained, bigconcatenatetrained_nonps in zip(relativescoretrained, bigconcatenatetrained_nonps)]
    relativescorenaivefrac = [relativescorenaive / bigconcatenatenaive_nonps for relativescorenaive, bigconcatenatenaive_nonps in zip(relativescorenaive, bigconcatenatenaive_nonps)]


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
    plt.savefig('D:/diffF0distribution_11122022.png', dpi=1000)
    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax = sns.distplot(relativescoretrainedfrac, bins = 20, label='trained',ax=ax, color='purple')
    x = ax.lines[0].get_xdata()  # Get the x data of the distribution
    y = ax.lines[0].get_ydata()  # Get the y data of the distribution
    maxidtrained_idx = np.argmax(y)
    x_coord_trained = x[maxidtrained_idx]


    ax2 = sns.distplot(relativescorenaivefrac, bins = 20, label='naive', ax=ax, color='darkcyan')

    x = ax2.lines[0].get_xdata()  # Get the x data of the distribution
    y = ax2.lines[0].get_ydata()  # Get the y data of the distribution
    maxidnaive_idx = np.argmax(y)  # The id of the peak (maximum of y data)

    x_coord_naive = x[maxidnaive_idx]
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


    plt.savefig('D:/diffF0distribution_frac_17012023wlegend.png', dpi=1000)
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
    plt.savefig('D:/controlF0distribution17012023.png', dpi=1000)

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
    plt.savefig('D:/rovedF0distribution_17012023.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi = 800)
    ax.set_xlim([0,1])
    sns.distplot(bigconcatenatetrained_ps,  label='trained roved',ax=ax, color='purple')
    sns.distplot(bigconcatenatetrained_nonps,  label='trained control',ax=ax, color='magenta')
    ax.legend(fontsize=18)
    plt.title('Roved and Control F0 Distributions for the Trained Animals', fontsize = 18)
    plt.xlabel(' LSTM decoder scores', fontsize = 20)

    plt.savefig('D:/rovedF0vscontrolF0traineddistribution_17012023.png', dpi=1000)

    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi = 800)
    ax.set_xlim([0,1])
    sns.distplot(bigconcatenatenaive_ps,  label='naive roved',ax=ax, color='darkcyan')
    sns.distplot(bigconcatenatenaive_nonps,  label='naive control',ax=ax, color='cyan')
    ax.legend(fontsize=18)
    plt.xlabel(' LSTM decoder scores', fontsize = 20)
    plt.title('Roved and Control F0 Distributions for the Naive Animals', fontsize = 18)

    plt.savefig('D:/rovedF0vscontrolF0naivedistribution_17012023.png', dpi=1000)
    plt.show()
    manwhitscorecontrolf0vsrovedtrained = scipy.stats.kstest(bigconcatenatetrained_nonps, bigconcatenatetrained_ps, alternative = 'two-sided')

    manwhitscorecontrolf0vsrovednaive = scipy.stats.kstest(bigconcatenatenaive_nonps, bigconcatenatenaive_ps, alternative='two-sided')

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
    fig, ax = plt.subplots(2, figsize=(5, 8))
    count = 0

    for pitchshiftornot in mergednaiveanimaldict[sutype].keys():
        ax[count].boxplot([meregedtest['su_list'][pitchshiftornot]['female_talker'],
                           meregedtest['mu_list'][pitchshiftornot]['female_talker'],
                           meregedtest['su_list'][pitchshiftornot]['male_talker'],
                           meregedtest['mu_list'][pitchshiftornot]['male_talker']])
        ax[count].legend()
        ax[count].set_ylabel('LSTM decoding score (%)')
        ax[count].set_yticklabels([0, 20, 40, 60, 80, 100])

        # as @ali14 pointed out, for python3, use this
        # for sp in ax2.spines.values():
        # and for python2, use this

        if sutype == 'su_list':
            stringtitle = 'single'
        else:
            stringtitle = 'multi'
        if pitchshiftornot == 'pitchshift':
            stringtitlepitch = 'F0-roved'
        else:
            stringtitlepitch = 'control F0'
        ax[count].set_title('Naive LSTM scores for' + ' single and multi-units,\n ' + stringtitlepitch + ' trials')
        y_crum_su = dictoutput_crumble['su_list'][pitchshiftornot]['female_talker']
        y_crum_mu = dictoutput_crumble['mu_list'][pitchshiftornot]['female_talker']

        y_eclair_su = dictoutput_eclair['su_list'][pitchshiftornot]['female_talker']
        y_eclair_mu = dictoutput_eclair['mu_list'][pitchshiftornot]['female_talker']

        y_nala_su = dictoutput_nala['su_list'][pitchshiftornot]['female_talker']
        y_nala_mu = dictoutput_nala['mu_list'][pitchshiftornot]['female_talker']

        y2_crum_su_male = dictoutput_crumble['su_list'][pitchshiftornot]['male_talker']
        y2_eclair_su_male = dictoutput_eclair['su_list'][pitchshiftornot]['male_talker']
        y2_nala_su_male = dictoutput_nala['su_list'][pitchshiftornot]['male_talker']

        y2_crum_mu_male = dictoutput_crumble['mu_list'][pitchshiftornot]['male_talker']
        y2_eclair_mu_male = dictoutput_eclair['mu_list'][pitchshiftornot]['male_talker']
        y2_nala_mu_male = dictoutput_nala['mu_list'][pitchshiftornot]['male_talker']

        # Add some random "jitter" to the x-axis
        x_su = np.random.normal(1, 0.04, size=len(y_crum_su))
        x2_su = np.random.normal(1, 0.04, size=len(y_eclair_su))
        x3_su = np.random.normal(1, 0.04, size=len(y_nala_su))

        x_mu = np.random.normal(2, 0.04, size=len(y_crum_mu))
        x2_mu = np.random.normal(2, 0.04, size=len(y_eclair_mu))
        x3_mu = np.random.normal(2, 0.04, size=len(y_nala_mu))

        x_su_male = np.random.normal(3, 0.04, size=len(y2_crum_su_male))
        x2_su_male = np.random.normal(3, 0.04, size=len(y2_eclair_su_male))
        x3_su_male = np.random.normal(3, 0.04, size=len(y2_nala_su_male))

        x_mu_male = np.random.normal(4, 0.04, size=len(y2_crum_mu_male))
        x2_mu_male = np.random.normal(4, 0.04, size=len(y2_eclair_mu_male))
        x3_mu_male = np.random.normal(4, 0.04, size=len(y2_nala_mu_male))

        ax[count].plot(x_su, y_crum_su, ".", color='mediumturquoise', alpha=0.2, )
        ax[count].plot(x2_su, y_eclair_su, ".", color='darkorange', alpha=0.2, )
        ax[count].plot(x3_su, y_nala_su, ".", color='slategray', alpha=0.2, )

        ax[count].plot(x2_mu, y_eclair_mu, ".", color='darkorange', alpha=0.2)
        ax[count].plot(x_mu, y_crum_mu, ".", color='mediumturquoise', alpha=0.2)
        ax[count].plot(x3_mu, y_nala_mu, ".", color='slategray', alpha=0.2, )

        ax[count].plot(x_su_male, y2_crum_su_male, ".", color='mediumturquoise', alpha=0.2)
        ax[count].plot(x2_su_male, y2_eclair_su_male, ".", color='darkorange', alpha=0.2)
        ax[count].plot(x3_su_male, y2_nala_su_male, ".", color='slategray', alpha=0.2)

        ax[count].plot(x_mu_male, y2_crum_mu_male, ".", color='mediumturquoise', alpha=0.2, label='F1901')
        ax[count].plot(x2_mu_male, y2_eclair_mu_male, ".", color='darkorange', alpha=0.2, label='F1902')
        ax[count].plot(x3_mu_male, y2_nala_mu_male, ".", color='slategray', alpha=0.2, label='F1812')

        print('mean score for naive male ' + pitchshiftornot + sutype + 'is' + str(
            np.mean(meregedtest[sutype][pitchshiftornot]['male_talker'])) + 'and std is' + str(
            np.std(meregedtest[sutype][pitchshiftornot]['male_talker'])))
        print('mean score for naive female ' + pitchshiftornot + sutype + 'is' + str(
            np.mean(meregedtest[sutype][pitchshiftornot]['female_talker'])) + 'and std is' + str(
            np.std(meregedtest[sutype][pitchshiftornot]['female_talker'])))
        print('mean score for naive for both talkers' + pitchshiftornot + sutype + 'is' + str(np.mean(np.concatenate((
            meregedtest[
                sutype][
                pitchshiftornot][
                'male_talker'],
            meregedtest[
                sutype][
                pitchshiftornot][
                'female_talker'])))) + 'and std is' + str(
            np.std(np.concatenate((meregedtest[sutype][pitchshiftornot]['male_talker'],
                                   meregedtest[sutype][pitchshiftornot]['female_talker'])))))

        x = np.random.normal(2, 0.04, size=len(y2_crum))
        x2 = np.random.normal(2, 0.04, size=len(y2_eclair))
        if count == 1:
            ax2 = ax[count].twiny()
            # Offset the twin axis below the host
            ax2.xaxis.set_ticks_position("bottom")
            ax2.xaxis.set_label_position("bottom")

            # Offset the twin axis below the host
            ax2.spines["bottom"].set_position(("axes", -0.15))
            # Turn on the frame for the twin axis, but then hide all
            # but the bottom spine
            ax2.set_frame_on(True)
            ax2.patch.set_visible(False)

            # as @ali14 pointed out, for python3, use this
            # for sp in ax2.spines.values():
            # and for python2, use this
            for sp in ax2.spines.values():
                sp.set_visible(False)
            ax2.spines["bottom"].set_visible(True)

            ax[count].set_xticklabels(['SU', 'MUA', 'SU', 'MUA'], fontsize=12)
            ax2.set_xlabel("talker", fontsize=12)
            ax2.set_xticks([0.2, 0.8])
            ax2.set_xticklabels(["female", "male"], fontsize=12)

        #            ax[count].set_xticklabels(['female', 'male'])
        else:
            ax[count].tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)

        # ax[count].plot(x, y2_crum, ".", color='mediumturquoise', alpha=0.2, )
        # ax[count].plot(x2, y2_eclair, ".", color='darkorange', alpha=0.2)
        ax[count].set_ylim([0, 1])
        if count == 1:
            ax[count].legend(prop={'size': 12})
        count += 1
    fig.tight_layout()
    plt.ylim(0, 1)

    plt.show()

    # plotting both mu sound driven and single unit units, ZOLA
    fig, ax = plt.subplots(2, figsize=(5, 8))
    count = 0

    for pitchshiftornot in dictoutput_zola[sutype].keys():
        ax[count].boxplot([dictoutput_zola['su_list'][pitchshiftornot]['female_talker'],
                           dictoutput_zola['mu_list'][pitchshiftornot]['female_talker'],
                           dictoutput_zola['su_list'][pitchshiftornot]['male_talker'],
                           dictoutput_zola['mu_list'][pitchshiftornot]['male_talker']])
        ax[count].legend()
        ax[count].set_ylabel('LSTM decoding score (%)')
        #        ax[count].set_yticklabels([0, 20, 40, 60, 80, 100])

        if sutype == 'su_list':
            stringtitle = 'single'
        else:
            stringtitle = 'multi'
        if pitchshiftornot == 'pitchshift':
            stringtitlepitch = 'F0-roved'
        else:
            stringtitlepitch = 'control F0'
        ax[count].set_title('Trained LSTM scores for' + ' single and multi-units,\n ' + stringtitlepitch + ' trials')
        y_su_female = dictoutput_zola['su_list'][pitchshiftornot]['female_talker']
        y_su_male = dictoutput_zola['mu_list'][pitchshiftornot]['female_talker']

        y_mu_female = dictoutput_zola['su_list'][pitchshiftornot]['male_talker']
        y_mu_male = dictoutput_zola['mu_list'][pitchshiftornot]['male_talker']

        # Add some random "jitter" to the x-axis
        x_su_female = np.random.normal(1, 0.04, size=len(y_su_female))
        x2_mu_female = np.random.normal(2, 0.04, size=len(y_mu_female))

        x_su_male = np.random.normal(3, 0.04, size=len(y_su_male))
        x2_mu_male = np.random.normal(4, 0.04, size=len(y_mu_male))

        ax[count].plot(x_su_female, y_su_female, ".", color='purple', alpha=0.2, )
        ax[count].plot(x2_mu_female, y_mu_female, ".", color='green', alpha=0.2, )

        ax[count].plot(x_su_male, y_su_male, ".", color='purple', alpha=0.2)
        ax[count].plot(x2_mu_male, y_mu_male, ".", color='green', alpha=0.2)

        if count == 1:
            ax2 = ax[count].twiny()
            # Offset the twin axis below the host
            ax2.xaxis.set_ticks_position("bottom")
            ax2.xaxis.set_label_position("bottom")

            # Offset the twin axis below the host
            ax2.spines["bottom"].set_position(("axes", -0.15))

            # Turn on the frame for the twin axis, but then hide all
            # but the bottom spine
            ax2.set_frame_on(True)
            ax2.patch.set_visible(False)

            # as @ali14 pointed out, for python3, use this
            # for sp in ax2.spines.values():
            # and for python2, use this
            for sp in ax2.spines.values():
                sp.set_visible(False)
            ax2.spines["bottom"].set_visible(True)

            ax[count].set_xticklabels(['SU', 'MUA', 'SU', 'MUA'], fontsize=12)
            ax2.set_xlabel("talker", fontsize=12)
            ax2.set_xticks([0.2, 0.8])
            ax2.set_xticklabels(["female", "male"], fontsize=12)

        #            ax[count].set_xticklabels(['female', 'male'])
        else:
            ax[count].tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)

        ax[count].set_ylim([0, 1])
        if count == 1:
            ax[count].legend(prop={'size': 12})
        count += 1
    fig.tight_layout()
    plt.ylim(0, 1)

    plt.show()
