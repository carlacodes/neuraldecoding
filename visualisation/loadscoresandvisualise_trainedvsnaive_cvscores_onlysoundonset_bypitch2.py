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
singlunitlistsoundonset_crumble = [ 11, 17, 21, 22, 26] #removed  6 and 7 clus ids
multiunitlist_soundonset_crumble = [13, 14, 23, 25, 27, 29]


def scatterplot_and_visualise(probewordlist,
                              saveDir='D:/decodingbypitchoutput/lstm_kfold_18122022_crumble_bypitch/',
                              ferretname='Crumble',
                              singleunitlist=singlunitlistsoundonset_crumble,
                              multiunitlist=multiunitlist_soundonset_crumble, noiselist=[19, 18, ]):
    singleunitlist = [x - 1 for x in singleunitlist]
    multiunitlist = [x - 1 for x in multiunitlist]
    noiselist = [x - 1 for x in noiselist]

    scores_su = {}
    scores_mu = {}
    dictscores = {}
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
        # dictscores = {'su': {'1': {'female_talker': {},
        #                                                  'male_talker': {}},
        #                                   '2': {'female_talker': {},
        #                                                     'male_talker': {}},
        #                      '3': {'female_talker': {},
        #                                                     'male_talker': {}},
        #                      '4': {'female_talker': {},
        #                            'male_talker': {}},
        #                      'male_talker': {},
        #               '5': {'female_talker': {},
        #                     'male_talker': {}}},
        #               'mu': {'1': {'female_talker': {},
        #                            'male_talker': {}},
        #                      '2': {'female_talker': {},
        #                            'male_talker': {}},
        #                      '3': {'female_talker': {},
        #                            'male_talker': {}},
        #                      '4': {'female_talker': {},
        #                            'male_talker': {}},
        #                      'male_talker': {},
        #                      '5': {'female_talker': {},
        #                            'male_talker': {}}}
        #               }

        scores = np.load(
            saveDir + '/' + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_pitchshift_vs_not_by_talker_bs.npy',
            allow_pickle=True)[()]


        for talker in [1, 2]:
            comparisons = [comp for comp in scores[f'talker{talker}']]

            for comp in comparisons:
                for cond in [1,2,3,4,5]:
                    for i, clus in enumerate(scores[f'talker{talker}'][comp][cond]['cluster_id']):

                        # print(i, clus)
                        if clus in singleunitlist:
                            if clus not in scores_su:
                                scores_su[clus] = {}
                            if talker not in scores_su[clus]:
                                scores_su[clus][talker] = {}
                            if cond not in scores_su[clus][talker]:
                                scores_su[clus][talker][cond] = {}
                            # if probeword not in scores_mu[clus][cond]:
                            scores_su[clus][talker][cond][probewordindex] = {}
                            scores_su[clus][talker][cond][probewordindex] = scores[f'talker{talker}'][comp][cond]['lstm_avg'][i]
                        elif clus in multiunitlist:
                            if clus not in scores_mu:
                                scores_mu[clus] = {}
                            if talker not in scores_mu[clus]:
                                scores_mu[clus][talker] = {}
                            if cond not in scores_mu[clus][talker]:
                                scores_mu[clus][talker][cond] = {}
                            # if probeword not in scores_mu[clus][cond]:
                            #     scores_mu[clus][cond][probeword] = {}
                            scores_mu[clus][talker][cond][probewordindex] = {}

                            scores_mu[clus][talker][cond][probewordindex] = scores[f'talker{talker}'][comp][cond]['lstm_avg'][i]



    return scores_su, scores_mu


def main():
    probewordlist = [ (5, 6), (32, 38), (20, 22), (2, 2),  (42, 49)]

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
# def plot_by_pitch(scores_mu_nala, scores_su_nala, scores_mu_cruella, scores_su_cruella, scores_mu_eclair, scores_su_eclair, scores_mu_zola, scores_su_zola):
#     continue

def flatten(d):
    res = []  # Result list
    if isinstance(d, dict):
        for key, val in d.items():
            res.extend(flatten(val))
    elif isinstance(d, list):
        res = d
    else:
        raise TypeError("Undefined type for flatten: %s"%type(d))

    return res

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def organise_compressed_dict(compressed_dict_mu_nala):
    female_array = np.empty((5,0))
    male_array = np.empty((5,0))
    female_array_1= np.empty((0,))
    female_array_2= np.empty((0,))
    female_array_3 = np.empty((0,))
    female_array_4 = np.empty((0,))
    female_array_5 = np.empty((0,))

    male_array_1 = np.empty((0,))
    male_array_2 = np.empty((0,))
    male_array_3 = np.empty((0,))
    male_array_4 = np.empty((0,))
    male_array_5 = np.empty((0,))
    for keys in compressed_dict_mu_nala:
        #need to turn this into dictionary instead
        values = keys.split('.')
        if values[1] == '1':
            if values[2] == '1':
                if female_array_1.size == 0:
                    female_array_1 = np.array(compressed_dict_mu_nala[keys])
                else:
                    female_array_1 = np.append(female_array_1, np.array(compressed_dict_mu_nala[keys]))
            elif values[2] == '2':
                if female_array_2.size == 0:
                    female_array_2 = np.array(compressed_dict_mu_nala[keys])
                else:
                    female_array_2 = np.append(female_array_2, np.array(compressed_dict_mu_nala[keys]))
            elif values[2] == '3':
                if female_array_3.size == 0:
                    female_array_3 = np.array(compressed_dict_mu_nala[keys])
                else:
                    female_array_3 = np.append(female_array_3,np.array(compressed_dict_mu_nala[keys]))
            elif values[2] == '4':
                if female_array_4.size == 0:
                    female_array_4 = np.array(compressed_dict_mu_nala[keys])
                else:
                    female_array_4 = np.append(female_array_4, np.array(compressed_dict_mu_nala[keys]))
            elif values[2] == '5':
                if female_array_5.size == 0:
                    female_array_5 = np.array(compressed_dict_mu_nala[keys])
                else:
                    female_array_5 = np.append(female_array_5, np.array(compressed_dict_mu_nala[keys]))
        elif values[1] == '2':
            if values[2] == '1':
                if male_array_1.size == 0:
                    male_array_1 = np.array(compressed_dict_mu_nala[keys])
                else:
                    male_array_1 = np.append(male_array_1, np.array(compressed_dict_mu_nala[keys]))
            elif values[2] == '2':
                if male_array_2.size == 0:
                    male_array_2 = np.array(compressed_dict_mu_nala[keys])
                else:
                    male_array_2 = np.append(male_array_2, np.array(compressed_dict_mu_nala[keys]))
            elif values[2] == '3':
                if male_array_3.size == 0:
                    male_array_3 = np.array(compressed_dict_mu_nala[keys])
                else:
                    male_array_3 = np.append(male_array_3, np.array(compressed_dict_mu_nala[keys]))
            elif values[2] == '4':
                if male_array_4.size == 0:
                    male_array_4 = np.array(compressed_dict_mu_nala[keys])
                else:
                    male_array_4 = np.append(male_array_4, np.array(compressed_dict_mu_nala[keys]))
            elif values[2] == '5':
                if male_array_5.size == 0:
                    male_array_5 = np.array(compressed_dict_mu_nala[keys])
                else:
                    male_array_5 = np.append(male_array_5, np.array(compressed_dict_mu_nala[keys]))
    dict_male ={}
    dict_male['1'] = {}
    dict_male['1']=male_array_1
    dict_male['2'] = male_array_2
    dict_male['3'] = male_array_3
    dict_male['4'] = male_array_4
    dict_male['5'] = male_array_5

    dict_female ={}
    dict_female['1'] = female_array_1
    dict_female['2'] = female_array_2
    dict_female['3'] = female_array_3
    dict_female['4'] = female_array_4
    dict_female['5'] = female_array_5

    return dict_female, dict_male
def rand_jitter(arr):
    try:
        stdev = .1
        print('jitering')
        print(arr+np.random.randn(len(arr))*stdev)
        return arr + np.random.randn(len(arr)) * stdev

    except:
        return arr

def plot_by_pitch(scores_mu_nala, scores_su_nala, scores_mu_cruella, scores_su_cruella, scores_mu_eclair,
                  scores_su_eclair, scores_mu_zola, scores_su_zola, scores_su_crumble, scores_mu_crumble):

    #plot by pitch with a box and whisker plot, then a scatter plot overlaid with the raw data for each animal and talker
    # for each condition

    compressed_dict_mu_zola = flatten_dict(scores_mu_zola)
    compressed_dict_su_zola = flatten_dict(scores_su_zola)
    compressed_dict_mu_nala = flatten_dict(scores_mu_nala)
    compressed_dict_su_nala = flatten_dict(scores_su_nala)
    compressed_dict_mu_cruella = flatten_dict(scores_mu_cruella)
    compressed_dict_su_cruella = flatten_dict(scores_su_cruella)
    compressed_dict_mu_eclair = flatten_dict(scores_mu_eclair)
    compressed_dict_su_eclair = flatten_dict(scores_su_eclair)
    compressed_dict_su_crumble = flatten_dict(scores_su_crumble)
    compressed_dict_mu_crumble = flatten_dict(scores_mu_crumble)

    female_array_zola_mu, male_array_zola_mu = organise_compressed_dict(compressed_dict_mu_zola)
    female_array_zola_su, male_array_zola_su = organise_compressed_dict(compressed_dict_su_zola)

    female_array_cruella_mu, male_array_cruella_mu = organise_compressed_dict(compressed_dict_mu_cruella)
    female_array_cruella_su, male_array_cruella_su = organise_compressed_dict(compressed_dict_su_cruella)

    female_array_crumble_mu, male_array_crumble_mu = organise_compressed_dict(compressed_dict_mu_crumble)
    female_array_crumble_su, male_array_crumble_su = organise_compressed_dict(compressed_dict_su_crumble)

    female_array_eclair_mu, male_array_eclair_mu = organise_compressed_dict(compressed_dict_mu_eclair)
    female_array_eclair_su, male_array_eclair_su = organise_compressed_dict(compressed_dict_su_eclair)

    # big_merged_trained_mu = np.empty((5, 2), float)
    # for i in range(5):
    #     print(i)
    #     big_merged_trained_mu[i,:] = ([female_array_cruella_mu[i], female_array_zola_mu[i]])
    big_merged_trained_mu = {}
    big_merged_trained_su = {}

    big_merged_naive_mu = {}
    big_merged_naive_su = {}

    for i in range(1,6):
        big_merged_trained_mu[str(i)] = np.append(female_array_zola_mu[str(i)], female_array_cruella_mu[str(i)])
        big_merged_trained_su[str(i)] =np.append(female_array_zola_su[str(i)], female_array_cruella_su[str(i)])
        big_merged_naive_mu[str(i)] = np.append(female_array_eclair_mu[str(i)], female_array_crumble_mu[str(i)])
        big_merged_naive_su[str(i)] = np.append(female_array_eclair_su[str(i)], female_array_crumble_su[str(i)])
    #plot box and whisker plot for each condition
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].boxplot([big_merged_trained_mu['1'], big_merged_trained_mu['2'], big_merged_trained_mu['3'], big_merged_trained_mu['4'], big_merged_trained_mu['5']])
    ax[0,0].scatter(rand_jitter(3*np.ones(len(female_array_zola_mu['3']))), female_array_zola_mu['3'], s=1, color = 'cyan')
    ax[0,0].scatter(rand_jitter(4*np.ones(len(female_array_zola_mu['4']))), female_array_zola_mu['4'], color = 'cyan')
    ax[0,0].scatter(rand_jitter(5*np.ones(len(female_array_zola_mu['5']))), female_array_zola_mu['5'], color= 'cyan')
    for i in range(1,6):
        try:
            ax[0, 0].scatter(rand_jitter(i*np.ones(len(female_array_cruella_mu[str(i)]))), female_array_cruella_mu[str(i)])
        except:
            ax[0, 0].scatter(rand_jitter(i*np.ones(len([female_array_cruella_mu[str(i)]]))), female_array_cruella_mu[str(i)])




    ax[0, 0].set_title('Trained Mu')
    ax[0, 0].set_xticklabels(['1', '2', '3', '4', '5'])
    ax[0, 0].set_ylabel('Decoding score')
    ax[0, 0].set_xlabel('Pitch')

    ax[0, 1].boxplot([big_merged_trained_su['1'], big_merged_trained_su['2'], big_merged_trained_su['3'], big_merged_trained_su['4'], big_merged_trained_su['5']])
    ax[0, 1].set_title('Trained Su')
    ax[0, 1].set_xticklabels(['1', '2', '3', '4', '5'])
    ax[0, 1].set_ylabel('Decoding score')
    ax[0, 1].set_xlabel('Pitch')

    for i in range(3,6):
        if i == 3:
            my_label = 'F1815'
            my_label_crumble = 'F1901'
            my_label_eclair = 'F1902'
            my_label_zola = 'F1702'
        else:
            my_label = "_nolegend_"
            my_label_crumble = "_nolegend_"
            my_label_eclair = "_nolegend_"
            my_label_zola = "_nolegend_"

        ax[0, 1].scatter(rand_jitter(i * np.ones(len(female_array_cruella_su[str(i)]))), female_array_cruella_su[str(i)], color='red', label = my_label)



        ax[1, 0].scatter(rand_jitter(i*np.ones(len(female_array_eclair_mu[str(i)]))), female_array_eclair_mu[str(i)], color = 'purple', label = my_label_eclair)
        ax[1, 0].scatter(rand_jitter(i*np.ones(len(female_array_crumble_mu[str(i)]))), female_array_crumble_mu[str(i)], color ='green', label =my_label_crumble)
        ax[1, 1].scatter(rand_jitter(i*np.ones(len(female_array_crumble_su[str(i)]))), female_array_crumble_su[str(i)], color = 'green')
        ax[1, 1].scatter(rand_jitter(i*np.ones(len(female_array_eclair_su[str(i)]))), female_array_eclair_su[str(i)], color = 'purple')

        try:
            ax[0, 1].scatter(rand_jitter(i* np.ones(len(female_array_zola_su[str(i)]))), female_array_zola_su[str(i)], color = 'cyan', label = my_label_zola)
        except:
            pass

    ax[1, 0].boxplot([big_merged_naive_mu['1'], big_merged_naive_mu['2'], big_merged_naive_mu['3'], big_merged_naive_mu['4'], big_merged_naive_mu['5']])
    ax[1, 0].set_title('Naive MU')
    ax[1, 0].set_xticklabels(['1', '2', '3', '4', '5'])
    ax[1, 0].set_ylabel('Decoding score')
    ax[1, 0].set_xlabel('Pitch')

    ax[1, 1].boxplot([big_merged_naive_su['1'], big_merged_naive_su['2'], big_merged_naive_su['3'], big_merged_naive_su['4'], big_merged_naive_su['5']])
    ax[1, 1].set_title('Naive SU')
    ax[1, 1].set_xticklabels(['1', '2', '3', '4', '5'])
    ax[1, 1].set_ylabel('Decoding score')
    ax[1, 1].set_xlabel('Pitch')
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].set_ylim([0, 1])
    ax[1,0].set_ylim([0, 1])
    ax[0,1].set_ylim([0, 1])
    ax[0,0].set_ylim([0, 1])

    plt.sca(ax[1, 1])

    plt.xticks([1, 2, 3, 4, 5], labels=['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'])
    plt.sca(ax[0, 1])

    plt.xticks([1, 2, 3, 4, 5], labels=['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'])
    plt.sca(ax[0, 0])

    plt.xticks([1, 2, 3, 4, 5], labels=['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'])
    plt.sca(ax[1, 0])

    plt.xticks([1, 2, 3, 4, 5], labels=['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'])
    fig.suptitle('Female talker decoding scores by mean F0 ', fontsize = 20)  # or plt.suptitle('Main title')
    ax[0,0].set_xlim(2.5,5.5)
    ax[0,1].set_xlim(2.5,5.5)
    ax[1,0].set_xlim(2.5,5.5)
    ax[1,1].set_xlim(2.5,5.5)

    plt.savefig('D:/Data/decodingbypitchresults/LSTM_decoding_scores_by_pitch_female.png', dpi=1000)

    plt.show()





    plot_su_and_mu_by_pitch(male_array_zola_mu, male_array_zola_su, male_array_cruella_mu, male_array_cruella_su, male_array_eclair_mu, male_array_eclair_su, male_array_crumble_mu, male_array_crumble_su)

def plot_su_and_mu_by_pitch(female_array_zola_mu, female_array_zola_su, female_array_cruella_mu, female_array_cruella_su, female_array_eclair_mu, female_array_eclair_su, female_array_crumble_mu, female_array_crumble_su):
    big_merged_trained_mu = {}
    big_merged_trained_su = {}

    big_merged_naive_mu = {}
    big_merged_naive_su = {}

    for i in range(1, 6):
        big_merged_trained_mu[str(i)] = np.append(female_array_zola_mu[str(i)], female_array_cruella_mu[str(i)])
        big_merged_trained_su[str(i)] = np.append(female_array_zola_su[str(i)], female_array_cruella_su[str(i)])
        big_merged_naive_mu[str(i)] = np.append(female_array_eclair_mu[str(i)], female_array_crumble_mu[str(i)])
        big_merged_naive_su[str(i)] = np.append(female_array_eclair_su[str(i)], female_array_crumble_su[str(i)])
    # plot box and whisker plot for each condition
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].boxplot(
        [big_merged_trained_mu['1'], big_merged_trained_mu['2'], big_merged_trained_mu['3'], big_merged_trained_mu['4'],
         big_merged_trained_mu['5']])
    # ax[0, 0].scatter(3 * np.ones(len(female_array_zola_mu['3'])), female_array_zola_mu['3'], color='cyan')
    # ax[0, 0].scatter(4 * np.ones(len(female_array_zola_mu['4'])), female_array_zola_mu['4'], color='cyan')
    # ax[0, 0].scatter(5 * np.ones(len(female_array_zola_mu['5'])), female_array_zola_mu['5'], color='cyan')
    for i in range(1, 6):
        ax[0, 0].scatter(rand_jitter(i * np.ones(len(female_array_cruella_mu[str(i)]))), female_array_cruella_mu[str(i)], color = 'red')
        ax[0, 0].scatter(rand_jitter(i * np.ones(len(female_array_zola_mu[str(i)]))), female_array_zola_mu[str(i)], color ='cyan')

    ax[0, 0].set_title('Trained Mu')
    ax[0, 0].set_xticklabels(['1', '2', '3', '4', '5'])
    ax[0, 0].set_ylabel('Decoding score')
    ax[0, 0].set_xlabel('Pitch')

    ax[0, 1].boxplot(
        [big_merged_trained_su['1'], big_merged_trained_su['2'], big_merged_trained_su['3'], big_merged_trained_su['4'],
         big_merged_trained_su['5']])
    plt.xticks([1,2,3,4,5], labels=['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'])

    ax[0, 1].set_title('Trained Su')
    ax[0, 1].set_xticklabels(['1', '2', '3', '4', '5'])
    ax[0, 1].set_ylabel('Decoding score')
    ax[0, 1].set_xlabel('Pitch')

    for i in range(1, 6):
        if i == 3:
            my_label_cruella = 'F1815'
            my_label_crumble = 'F1901'
            my_label_eclair = 'F1902'
            my_label_zola = 'F1702'
        else:
            my_label_cruella = "_nolegend_"
            my_label_crumble = "_nolegend_"
            my_label_eclair = "_nolegend_"
            my_label_zola = "_nolegend_"

        ax[0, 1].scatter(rand_jitter(i * np.ones(len(female_array_cruella_su[str(i)]))), female_array_cruella_su[str(i)],
                         color='red', label=my_label_cruella)
        ax[1, 0].scatter(rand_jitter(i * np.ones(len(female_array_eclair_mu[str(i)]))), female_array_eclair_mu[str(i)],
                         color='purple', label=my_label_eclair)
        ax[1, 0].scatter(rand_jitter(i * np.ones(len(female_array_crumble_mu[str(i)]))), female_array_crumble_mu[str(i)],
                         color='green', label=my_label_crumble)
        ax[1, 1].scatter(rand_jitter(i * np.ones(len(female_array_crumble_su[str(i)]))), female_array_crumble_su[str(i)],
                         color='green')
        try:
            ax[1, 1].scatter(rand_jitter(i * np.ones(len(female_array_eclair_su[str(i)]))), [female_array_eclair_su[str(i)]], color='purple')
        except:
            ax[1, 1].scatter(rand_jitter(i * np.ones(len([female_array_eclair_su[str(i)]]))), [female_array_eclair_su[str(i)]], color='purple')

        try:
            ax[0, 1].scatter(rand_jitter(i * np.ones(len(female_array_zola_su[str(i)]))), female_array_zola_su[str(i)], color='cyan',
                             label=my_label_zola)
        except:
            pass

    ax[1, 0].boxplot(
        [big_merged_naive_mu['1'], big_merged_naive_mu['2'], big_merged_naive_mu['3'], big_merged_naive_mu['4'],
         big_merged_naive_mu['5']])
    plt.xticks([1,2,3,4,5], labels=['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'])

    ax[1, 0].set_title('Naive MU')
    ax[1, 0].set_xticklabels(['1', '2', '3', '4', '5'])
    ax[1, 0].set_ylabel('Decoding score')
    ax[1, 0].set_xlabel('Pitch')

    ax[1, 1].boxplot(
        [big_merged_naive_su['1'], big_merged_naive_su['2'], big_merged_naive_su['3'], big_merged_naive_su['4'],
         big_merged_naive_su['5']])
    ax[1, 1].set_title('Naive SU')
    ax[1, 1].set_ylabel('Decoding score')
    ax[1, 1].set_xlabel('Pitch')
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1,1].set_ylim([0, 1])
    ax[1,0].set_ylim([0, 1])
    ax[0,1].set_ylim([0, 1])
    plt.xticks([1,2,3,4,5], labels=['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'])

    ax[0,0].set_ylim([0, 1])
    plt.sca(ax[1, 1])

    plt.xticks([1,2,3,4,5], labels=['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'])
    plt.sca(ax[0, 1])

    plt.xticks([1, 2, 3, 4, 5], labels=['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'])
    plt.sca(ax[0, 0])

    plt.xticks([1, 2, 3, 4, 5], labels=['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'])
    plt.sca(ax[1, 0])

    plt.xticks([1, 2, 3, 4, 5], labels=['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'])
    fig.suptitle('Male talker decoding scores by mean F0 ', fontsize = 20)  # or plt.suptitle('Main title')
    ax[0,0].set_xlim(0.5, 3.5)
    ax[0,1].set_xlim(0.5, 3.5)
    ax[1,0].set_xlim(0.5, 3.5)
    ax[1,1].set_xlim(0.5, 3.5)
    plt.savefig('D:/Data/decodingbypitchresults/LSTM_decoding_scores_by_pitch_male.png', dpi=1000)

    plt.show()


if __name__ == '__main__':
    probewordlist = [ (5, 6),(2, 2), (42, 49), (32, 38), (20, 22)]
    scores_su_crumble, scores_mu_crumble = scatterplot_and_visualise(
        probewordlist)
    scores_su_zola, scores_mu_zola= scatterplot_and_visualise(probewordlist,
                                                #saveDir='D:/Users/cgriffiths/resultsms4/lstmclass_CVDATA_08122022/08122022_14_40_02/',
                                                saveDir = 'D:/decodingbypitchoutput/lstm_kfold_18122022_zola_bypitch/',
                                                ferretname='Zola',

                                                singleunitlist=[13, 18, 37, 39],
                                                multiunitlist=[7, 8, 9, 10, 12, 15, 20, 21, 38]
                                                , noiselist=[29, 15, 36, ])

    scores_su_eclair, scores_mu_eclair = scatterplot_and_visualise(probewordlist,
                                                  saveDir='D:/decodingbypitchoutput/lstm_kfold_18122022_eclair_bypitch/',
                                                  ferretname='Eclair',
                                                  singleunitlist=[20, 21, 28, 35, 37, 39],
                                                  multiunitlist=[3, 4, 8, 9, 10, 11, 16, 17, 18, 19, 21, 22, 23, 24, 25,
                                                                 26, 27, 38, 21, 21, 33, 34, 40]
                                                  , noiselist=[23]
                                                  )

    # D:\Users\cgriffiths\resultsms4\lstmclass_18112022\27112022_21_54_08
    scores_su_cruella, scores_mu_cruella = scatterplot_and_visualise(probewordlist,
                                                   #saveDir='D:/Users/cgriffiths/resultsms4/lstmclass_CVDATA_05122022/06122022_00_40_15/',
                                                   saveDir = 'D:/decodingbypitchoutput/lstm_kfold_18122022_cruella_bypitch/',
                                                   ferretname='Cruella',
                                                   singleunitlist=singleunitlist_cruella_soundonset,
                                                   multiunitlist=multiunitlist_cruella_soundonset
                                                   , noiselist=[])

    scores_su_nala, scores_mu_nala = scatterplot_and_visualise(probewordlist,
                                                saveDir='D:/decodingbypitchoutput/lstm_kfold_18122022_nala_bypitch//',
                                                ferretname='Nala',
                                                singleunitlist=singleunitlist_nala,
                                                multiunitlist=multiunitlist_nala
                                                , noiselist=[])

    plot_by_pitch(scores_mu_nala, scores_su_nala, scores_mu_cruella, scores_su_cruella, scores_mu_eclair, scores_su_eclair, scores_mu_zola, scores_su_zola, scores_su_crumble, scores_mu_crumble)

