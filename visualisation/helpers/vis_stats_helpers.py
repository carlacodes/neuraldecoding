import numpy as np
from statsmodels.regression import linear_model
import statsmodels as sm
import pandas as pd
import statsmodels.formula.api as smf
import os
import scipy.stats as stats
import shap
from statsmodels.regression import linear_model
import seaborn as sns
import statsmodels as sm
import lightgbm as lgb
from optuna.integration import LightGBMPruningCallback

from pathlib import Path
import scipy
import sklearn
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import json
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import PermutationImportance
import optuna
def run_optuna_study_score(X, y):
    study = optuna.create_study(direction="minimize", study_name="LGBM regressor")
    func = lambda trial: objective_releasetimes(trial, X, y)
    study.optimize(func, n_trials=1000)
    print("Number of finished trials: ", len(study.trials))
    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")
    return study.best_params


def objective_releasetimes(trial, X, y):
    param_grid = {
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.6),
        "alpha": trial.suggest_float("alpha", 5, 15),
        "n_estimators": trial.suggest_int("n_estimators", 2, 100, step=2),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.1, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 30, step=1),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMRegressor(random_state=123, **param_grid)
        model.fit(
            X_train,
            y_train,

            eval_set=[(X_test, y_test)],
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "l2")
            ],  # Add a pruning callback
        )
        preds = model.predict(X_test)
        cv_scores[idx] = sklearn.metrics.mean_squared_error(y_test, preds)

    return np.mean(cv_scores)


def run_optuna_study(dfx, df_use):
    best_study_results = run_optuna_study_score(dfx.to_numpy(), df_use[col].to_numpy())
    params = best_study_results.best_params
    print(params)
    return params


def run_anova_on_dataframe(df_full_pitchsplit):
    df_full_pitchsplit_anova = df_full_pitchsplit.copy()

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
    # nest ferret as a variable , ,look at the relative magnittud eo fthe coefficients for both lightgbm model and anova
    print(df_full_pitchsplit_anova.dtypes)
    # now run anova
    formula = 'Score ~ C(ProbeWord) + C(PitchShift) +C(BrainArea)+C(SingleUnit)'
    model = smf.ols(formula, data=df_full_pitchsplit_anova).fit()
    anova_table = sm.stats.anova.anova_lm(model, typ=3)
    # get the coefficient of determination
    print(model.rsquared)
    print(anova_table)
    return anova_table, model


def create_gen_frac_variable(df_full_pitchsplit, high_score_threshold = False, index_or_frac = 'frac'):
    df_full_pitchsplit = df_full_pitchsplit[df_full_pitchsplit['Score'] >= 0.50]
    upper_quartile = np.percentile(df_full_pitchsplit['Score'], 75)

    for unit_id in df_full_pitchsplit['ID'].unique():
        # Check how many scores for that unit are above 60%
        df_full_pitchsplit_unit = df_full_pitchsplit[df_full_pitchsplit['ID'] == unit_id]
        #limit the scores to above 50%

        #filter for the above-chance scores
        mean_scores = df_full_pitchsplit_unit['Score'].mean()

        #add the mean score to the dataframe
        df_full_pitchsplit.loc[df_full_pitchsplit['ID'] == unit_id, 'MeanScore'] = mean_scores
        #if the mean score is below 0.75, then we can't calculate the gen frac
        all_scores = df_full_pitchsplit_unit['Score'].to_numpy()
        #figure out if any of the scores are above 0.75
        skip_param = False
        for score in all_scores:
            if score < upper_quartile:
                skip_param = True
                break

        if high_score_threshold == True:
            if len(df_full_pitchsplit_unit) == 0 or skip_param == True:
                df_full_pitchsplit.loc[df_full_pitchsplit['ID'] == unit_id, 'GenFrac'] = np.nan
                continue
        else:
            if len(df_full_pitchsplit_unit) == 0 :
                df_full_pitchsplit.loc[df_full_pitchsplit['ID'] == unit_id, 'GenFrac'] = np.nan
                continue
        above_60_scores = df_full_pitchsplit_unit[
            df_full_pitchsplit_unit['Score'] >= 0.60 ]  # Replace 'score_column' with the actual column name

        # Check how many probe words are below 60%

        below_60_probe_words = df_full_pitchsplit_unit[df_full_pitchsplit_unit[
                                                           'Score'] < 0.60]  # Replace 'probe_words_column' with the actual column name
        max_score = df_full_pitchsplit_unit.max()['Score']
        min_score = df_full_pitchsplit_unit.min()['Score']
        if index_or_frac == 'index':
            gen_frac = (max_score - min_score) / (max_score+min_score)
        else:
            gen_frac = len(above_60_scores) / (len(above_60_scores) + len(below_60_probe_words))


        df_full_pitchsplit.loc[df_full_pitchsplit['ID'] == unit_id, 'GenFrac'] = gen_frac
        # Now you can do something with the counts, for example, print them
        # print(f"Unit ID: {unit_id}")
        # print(f"Number of scores above 60%: {len(above_60_scores)}")
        # print(f"Number of probe words below 60%: {len(below_60_probe_words)}")
        # print("-------------------")
    return df_full_pitchsplit

def runlgbmmodel_score(df_use, optimization = False):
    col = 'Score'

    unique_probe_words = df_use['ProbeWord'].unique()
    unique_IDs = df_use['ID'].unique()
    df_use = df_use.reset_index(drop=True)

    # df_use['ProbeWord'] = pd.Categorical(df_use['ProbeWord'],
    #                                                        categories=unique_probe_words, ordered=True)
    df_use['ID'] = pd.Categorical(df_use['ID'],categories=unique_IDs, ordered=True)

    # df_use['ProbeWord'] = df_use['ProbeWord'].cat.codes

    #relabel the probe word labels to be the same as the paper
    df_use['ProbeWord'] = df_use['ProbeWord'].replace({ '(2,2)': 'craft', '(3,3)': 'in contrast to', '(4,4)': 'when a', '(5,5)': 'accurate', '(6,6)': 'pink noise', '(7,7)': 'of science', '(8,8)': 'rev. instruments', '(9,9)': 'boats', '(10,10)': 'today',
        '(13,13)': 'sailor', '(15,15)': 'but', '(16,16)': 'researched', '(18,18)': 'took', '(19,19)': 'the vast', '(20,20)': 'today', '(21,21)': 'he takes', '(22,22)': 'becomes', '(23,23)': 'any', '(24,24)': 'more'})

    #replace probeword with number ordered by length
    #order the probewords by length
    unique_probe_words = df_use['ProbeWord'].unique()
    unique_probe_words = sorted(unique_probe_words, key=len)
    for i, probe in enumerate(unique_probe_words):
        df_use['ProbeWord'] = df_use['ProbeWord'].replace({probe: i})

    df_use['BrainArea'] = df_use['BrainArea'] .replace({'PEG': 0, 'AEG': 1, 'MEG': 2})

    # df_use['BrainArea'] = df_use['BrainArea'].astype('category')
    df_use['ID'] = df_use['ID'].astype('category')
    # df_use['ProbeWord'] = df_use['ProbeWord'].astype('category')


    # cast the probe word category as an int
    df_use['PitchShift'] = df_use['PitchShift'].astype('int')
    df_use['Below-chance'] = df_use['Below-chance'].astype('int')

    # df_use["ProbeWord"] = pd.to_numeric(df_use["ProbeWord"])
    df_use["PitchShift"] = pd.to_numeric(df_use["PitchShift"])
    df_use["Below_chance"] = pd.to_numeric(df_use["Below-chance"])
    df_use["Score"] = pd.to_numeric(df_use["Score"])
    #only remove the below chance scores
    df_use = df_use[df_use['Below-chance'] == 0]




    dfx = df_use.loc[:, df_use.columns != col]
    # remove ferret as possible feature
    col = 'ID'
    dfx = dfx.loc[:, dfx.columns != col]
    col2 = 'SingleUnit'
    dfx = dfx.loc[:, dfx.columns != col2]
    col3 = 'GenFrac'
    dfx = dfx.loc[:, dfx.columns != col3]
    col4 = 'MeanScore'
    dfx = dfx.loc[:, dfx.columns != col4]
    col5 = 'Below-chance'
    dfx = dfx.loc[:, dfx.columns != col5]
    col6 = 'Below_chance'
    dfx = dfx.loc[:, dfx.columns != col6]

    #remove any rows
    if optimization == True:
        params = run_optuna_study_score(dfx.to_numpy(), df_use['Score'].to_numpy())
        #save as npy file
        np.save('params.npy', params)
    else:
        params = np.load('params.npy', allow_pickle='TRUE').item()

    X_train, X_test, y_train, y_test = train_test_split(dfx.to_numpy(), df_use['Score'].to_numpy(), test_size=0.2,
                                                        random_state=42, shuffle=True)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test)

    param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    evallist = [(dtrain, 'train'), (dtest, 'eval')]


    # xg_reg = lgb.LGBMRegressor(colsample_bytree=0.3, learning_rate=0.1,
    #                            max_depth=10, alpha=10, n_estimators=10, verbose=1)
    xg_reg = lgb.LGBMRegressor(**params, verbose=1)

    xg_reg.fit(X_train, y_train, eval_metric='MSE', verbose=1)
    ypred = xg_reg.predict(X_test)
    lgb.plot_importance(xg_reg)
    plt.title('feature importances for the lstm decoding score  model')
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
    explainer = shap.Explainer(xg_reg, X_train, feature_names=dfx.columns)
    shap_values2 = explainer(X_train)

    fig, ax = plt.subplots(figsize=(15, 15))
    # title kwargs still does nothing so need this workaround for summary plots

    fig, ax = plt.subplots(1, figsize=(10, 10), dpi = 300)

    shap.summary_plot(shap_values,dfx,  max_display=20, show = False, cmap = 'cool')
    plt.savefig(f'G:/neural_chapter/figures/summary_plot_lightgbm.png', dpi = 300)
    plt.show()
    #partial dependency plot of the pitch shift versus naive color coded by naive

    shap.dependence_plot("PitchShift", shap_values, dfx, interaction_index='Naive', show=False)
    plt.savefig(f'G:/neural_chapter/figures/naiveandpitchshift_lightgbmdependencyplot.png', dpi = 300)
    plt.show()

    naive = shap_values2[:, "Naive"].data
    pitchshift = shap_values2[:, "PitchShift"].data
    shap_values = shap_values2[:, "Naive"].values
    data_df = pd.DataFrame({
        "pitchshift": pitchshift,
        "naive": naive,
        "SHAP value": shap_values
    })
    custom_colors = ['cyan', 'hotpink', "purple"]  # Add more colors as needed

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.violinplot(x="naive", y="SHAP value", hue="pitchshift", data=data_df, split=True, inner="quart",
                   palette=custom_colors, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['True', 'False'], fontsize=18, rotation=45)
    ax.set_xlabel('Naive', fontsize=18)
    ax.set_ylabel('Impact on decoding score', fontsize=18)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    #reinsert the legend_hanldes and labels
    plt.savefig(f'G:/neural_chapter/figures/violinplot_naive.png', dpi = 300, bbox_inches = 'tight')
    plt.show()


    #F0 by talker violin plot, supp.
    pitchshift = shap_values2[:, "PitchShift"].data
    naive_values = shap_values2[:, "Naive"].data
    shap_values = shap_values2[:, "PitchShift"].values

    # Create a DataFrame with the necessary data
    data_df = pd.DataFrame({
        "pitchshift": pitchshift,
        "naive": naive_values,
        "SHAP value": shap_values
    })
    custom_colors = ['cyan', 'hotpink', "purple"]  # Add more colors as needed

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="pitchshift", y="SHAP value", hue="naive", data=data_df, split=True, inner="quart",
                   palette=custom_colors, ax=ax)

    ax.set_xticks([ 0, 1])
    ax.set_xticklabels(['True', 'False'], fontsize=18, rotation=45)
    ax.set_xlabel('Pitch Shift', fontsize=18)
    ax.set_ylabel('Impact on decoding score', fontsize=18)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    #reinsert the legend_hanldes and labels
    ax.legend(legend_handles, ['Trained', 'Naive'], loc='upper right', fontsize=13)
    plt.savefig(f'G:/neural_chapter/figures/violinplot_pitchshift.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    MEG = shap_values2[:, "BrainArea"].data
    naive_values = shap_values2[:, "Naive"].data
    shap_values = shap_values2[:, "BrainArea"].values
    data_df = pd.DataFrame({
        "MEG": MEG,
        "naive": naive_values,
        "SHAP value": shap_values
    })
    # custom_colors = ['blue', 'hotpink', "purple"]  # Add more colors as needed
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="MEG", y="SHAP value", hue="naive", data=data_df, split=True, inner="quart",
                     palette=custom_colors, ax=ax, order = ['MEG', 'PEG','AEG'])
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['MEG', 'PEG'], fontsize=18, rotation=45)
    plt.xlabel('Brain Area', fontsize=18)
    ax.set_ylabel('Impact on decoding score', fontsize=18)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    #reinsert the legend_hanldes and labels
    ax.legend(legend_handles, ['Trained', 'Naive'], loc='upper right', fontsize=13)
    plt.xlabel('Brain Area', fontsize=18)
    plt.savefig(f'G:/neural_chapter/figures/lightgbm_violinplot_brainarea.png', dpi = 300, bbox_inches='tight')
    plt.show()


    #plot the pro0be word
    fig, ax = plt.subplots(figsize=(10, 6))
    probe_word = shap_values2[:, "ProbeWord"].data
    naive_values = shap_values2[:, "Naive"].data

    shap_values = shap_values2[:, "ProbeWord"].values
    data_df = pd.DataFrame({
        "ProbeWord": probe_word,
        "naive": naive_values,
        "SHAP value": shap_values
    })
    # custom_colors = ['blue', 'hotpink', "purple"]  # Add more colors as needed
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="ProbeWord", y="SHAP value", hue="naive", data=data_df, split=True, inner="quart",
                        palette=custom_colors, ax=ax)
    # ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    # ax.set_xticklabels(['craft', 'in contrast to', 'when a', 'accurate', 'pink noise', 'of science', 'rev. instruments', 'boats', 'today'], fontsize=18, rotation=45)
    #
    ax.legend(legend_handles, ['Trained', 'Naive'], loc='upper right', fontsize=13)
    plt.xlabel('Probe Word', fontsize=18)
    plt.ylabel('Impact on decoding score', fontsize=18)
    plt.savefig(f'G:/neural_chapter/figures/lightgbm_violinplot_probeword.png', dpi = 300, bbox_inches='tight')
    plt.show()









    from sklearn.inspection import permutation_importance

    # Assuming X_test and y_test are your test data
    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=10, random_state=0)

    # Extract and print the importances
    importances = result.importances_mean

    # Map feature names to their importances
    feature_importance_dict = dict(zip(X_test.columns, importances))
    importances = result.importances_mean

    # Map feature names to their importances
    feature_importance_dict = dict(zip(dfx.columns, importances))

    # Sort the features by importance in descending order
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Print feature importances
    for feature, importance in feature_importance_dict.items():
        print(f"{feature}: {importance}")

    features, importance = zip(*sorted_feature_importance)
    plt.barh(features, importance)
    plt.xlabel('Permutation Importance')
    plt.title('Permutation Importances of Features')
    plt.show()






    labels = [item.get_text() for item in ax.get_yticklabels()]
    print(labels)
