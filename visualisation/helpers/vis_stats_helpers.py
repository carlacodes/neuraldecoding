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

def run_mixed_effects_on_dataframe(combined_df):
    #first remove all the below chance scores
    #now run the mixed effects model
    for unit_id in combined_df['ID']:
        #get the ferret ID
        if 'zola' in unit_id:
            ferret_id = 1
        elif 'cruella' in unit_id:
            ferret_id = 2
        elif 'windolene' in unit_id:
            ferret_id = 3
        elif 'squinty' in unit_id:
            ferret_id = 4
        elif 'Crumble' in unit_id:
            ferret_id = 5
        elif 'Eclair' in unit_id:
            ferret_id = 6
        elif 'Ore' in unit_id:
            ferret_id = 7
        elif 'nala' in unit_id:
            ferret_id = 8
        #add the ferret ID to the dataframe
        combined_df.loc[combined_df['ID'] == unit_id, 'FerretID'] = ferret_id
        #add unique recording ID to the dataframe, remove the first integer and then add the recording ID

    combined_df['Recording_ID'] = combined_df['ID'].str.split('_').str[1:]
    combined_df['Recording_ID'] = combined_df['Recording_ID'].str.join('_')

    #make recording ID a category
    # combined_df['Recording_ID'] = combined_df['Recording_ID'].astype('category')

    # Assign a unique number to each recording ID
    unique_recording_ids = combined_df['Recording_ID'].unique()
    for i, recording_id in enumerate(unique_recording_ids):
        combined_df.loc[combined_df['Recording_ID'] == recording_id, 'Recording_ID_Num'] = i


    model_formula = "Score ~ Naive +  PitchShift + (Naive * PitchShift )+ ProbeWord"
    model_formula2 = "Score ~ Naive +  PitchShift + (Naive * PitchShift)"
    model_formula3 = "Score ~ Naive +  PitchShift + (Naive * PitchShift) +BrainArea + ProbeWord"

    mixed_model = smf.mixedlm(model_formula, data=combined_df, groups=combined_df['Recording_ID_Num'])

    # Fit the model
    result = mixed_model.fit()

     #order the result.tables[1] by the p value
    result.summary().tables[1] =  result.summary().tables[1].sort_values(by=['P>|z|'])
    # Print model summary
    print(result.summary())
    var_resid = result.scale
    var_random_effect = float(result.cov_re.iloc[0])
    var_fixed_effect = result.predict(combined_df).var()

    total_var = var_fixed_effect + var_random_effect + var_resid
    marginal_r2 = var_fixed_effect / total_var
    conditional_r2 = (var_fixed_effect + var_random_effect) / total_var

    #export the results to a csv file
    result.summary().tables[0].to_csv('G:/neural_chapter/figures/mixed_effects_model_pg1.csv')
    result.summary().tables[1].sort_values(by=['P>|z|']).to_csv('G:/neural_chapter/figures/mixed_effects_model_pg2.csv')
    #export the marginal and conditional r2 to a csv file
    marginal_and_conditional_r2 = pd.DataFrame([marginal_r2, conditional_r2], columns = ['R2'], index = ['Marginal', 'Conditional'])
    marginal_and_conditional_r2.to_csv('G:/neural_chapter/figures/marginal_and_conditional_r2.csv')

    #make a barplot of the coefficients and their confidence intervals
    #first get the coefficients

    coefficients = result.params

    coefficients = result.summary().tables[1]
    coefficients['Coef.'] = pd.to_numeric(coefficients['Coef.'], errors='coerce')
    coefficients['Std.Err.'] = pd.to_numeric(coefficients['Std.Err.'], errors='coerce')
    coefficients['P>|z|'] = pd.to_numeric(coefficients['P>|z|'], errors='coerce')


    #sort the coefficients by ascending coefficient value
    coefficients = coefficients.sort_values(by=['Coef.'])
    #get the confidence intervals
    ci = result.conf_int()
    p_values = result.pvalues
    #get the standard errors
    se = result.bse
    #reorganise the confidence intervals
    ci = ci.reindex(coefficients.index)
    #reorganise the standard errors
    se = se.reindex(coefficients.index)
    #reorganise the p values
    p_values = p_values.reindex(coefficients.index)
    #now plot the coefficients
    fig, ax = plt.subplots(dpi = 300, figsize=(20, 10))
    #sort coefficients index by ascending coefficient value
    #convert the coefficients column to a float
    ax.bar(coefficients.index, coefficients['Coef.'], yerr = coefficients['Std.Err.'], color = 'forestgreen', edgecolor = 'black')
    #if the p value is less than 0.05, then add a star
    for i, p in enumerate(coefficients.index):
        try:
            if float(coefficients['P>|z|'][i]) < 0.05:
                ax.text(i, float(coefficients['Coef.'][i]) + 0.01, '*', fontsize = 20)
        except:
            continue
    ax.set_xticks(np.arange(0, len(coefficients.index), 1))
    ax.set_xticklabels(coefficients.index, rotation = 90, fontsize = 16)
    ax.set_ylabel('Coefficient value', fontsize = 20)
    ax.set_xlabel('Feature', fontsize = 20)
    ax.set_title('Coefficients of the mixed effects model predicting decoding score', fontsize = 25)
    plt.savefig(f'G:/neural_chapter/figures/mixed_effects_model_coefficients.png', dpi = 300, bbox_inches = 'tight')
    plt.show()



    return result
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

def create_gen_frac_and_index_variable(df_full_pitchsplit, high_score_threshold = False, need_ps = False, sixty_score_threshold = False):
    df_full_pitchsplit = df_full_pitchsplit[df_full_pitchsplit['Score'] >= 0.50]
    upper_quartile = np.percentile(df_full_pitchsplit['Score'], 75)

    for unit_id in df_full_pitchsplit['ID'].unique():
        # Check how many scores for that unit are above 60%
        df_full_pitchsplit_unit = df_full_pitchsplit[df_full_pitchsplit['ID'] == unit_id]
        if need_ps == True:
            #isolate the pitch shifted trials
            df_full_pitchsplit_unit_ps = df_full_pitchsplit_unit[df_full_pitchsplit_unit['PitchShift'] == 1]
            df_full_pitchsplit_unit_ns = df_full_pitchsplit_unit[df_full_pitchsplit_unit['PitchShift'] == 0]
            if len(df_full_pitchsplit_unit_ps) == 0 or len(df_full_pitchsplit_unit_ns) == 0:
                df_full_pitchsplit.loc[df_full_pitchsplit['ID'] == unit_id, 'GenFrac'] = np.nan
                continue

        #limit the scores to above 50%

        #filter for the above-chance scores
        mean_scores = df_full_pitchsplit_unit['Score'].mean()

        #add the mean score to the dataframe
        df_full_pitchsplit.loc[df_full_pitchsplit['ID'] == unit_id, 'MeanScore'] = mean_scores
        #if the mean score is below 0.75, then we can't calculate the gen frac
        all_scores = df_full_pitchsplit_unit['Score'].to_numpy()
        #figure out if any of the scores are above 0.75
        skip_param = False
        skip_param_60   = False
        for score in all_scores:
            if score < upper_quartile:
                skip_param = True
                break
        for score in all_scores:
            if score < 0.60:
                skip_param_60 = True
                break

        if high_score_threshold == True:
            if len(df_full_pitchsplit_unit) == 0 or skip_param == True:
                df_full_pitchsplit.loc[df_full_pitchsplit['ID'] == unit_id, 'GenFrac'] = np.nan
                continue
        elif sixty_score_threshold == True:
            if len(df_full_pitchsplit_unit) == 0 or skip_param_60 == True:
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

        gen_index = (max_score - min_score) / (max_score+min_score)

        gen_frac = len(above_60_scores) / (len(above_60_scores) + len(below_60_probe_words))


        df_full_pitchsplit.loc[df_full_pitchsplit['ID'] == unit_id, 'GenFrac'] = gen_frac
        df_full_pitchsplit.loc[df_full_pitchsplit['ID'] == unit_id, 'GenIndex'] = gen_index
        df_full_pitchsplit.loc[df_full_pitchsplit['ID'] == unit_id, 'MaxScore'] = max_score

        # Now you can do something with the counts, for example, print them
        # print(f"Unit ID: {unit_id}")
        # print(f"Number of scores above 60%: {len(above_60_scores)}")
        # print(f"Number of probe words below 60%: {len(below_60_probe_words)}")
        # print("-------------------")
    return df_full_pitchsplit
def create_gen_frac_variable(df_full_pitchsplit, high_score_threshold = False, index_or_frac = 'frac', need_ps = False, sixty_score_threshold = False):
    df_full_pitchsplit = df_full_pitchsplit[df_full_pitchsplit['Score'] >= 0.50]
    upper_quartile = np.percentile(df_full_pitchsplit['Score'], 75)

    for unit_id in df_full_pitchsplit['ID'].unique():
        # Check how many scores for that unit are above 60%
        df_full_pitchsplit_unit = df_full_pitchsplit[df_full_pitchsplit['ID'] == unit_id]
        if need_ps == True:
            #isolate the pitch shifted trials
            df_full_pitchsplit_unit_ps = df_full_pitchsplit_unit[df_full_pitchsplit_unit['PitchShift'] == 1]
            df_full_pitchsplit_unit_ns = df_full_pitchsplit_unit[df_full_pitchsplit_unit['PitchShift'] == 0]
            if len(df_full_pitchsplit_unit_ps) == 0 or len(df_full_pitchsplit_unit_ns) == 0:
                df_full_pitchsplit.loc[df_full_pitchsplit['ID'] == unit_id, 'GenFrac'] = np.nan
                continue

        #limit the scores to above 50%

        #filter for the above-chance scores
        mean_scores = df_full_pitchsplit_unit['Score'].mean()

        #add the mean score to the dataframe
        df_full_pitchsplit.loc[df_full_pitchsplit['ID'] == unit_id, 'MeanScore'] = mean_scores
        #if the mean score is below 0.75, then we can't calculate the gen frac
        all_scores = df_full_pitchsplit_unit['Score'].to_numpy()
        #figure out if any of the scores are above 0.75
        skip_param = False
        skip_param_60   = False
        for score in all_scores:
            if score < upper_quartile:
                skip_param = True
                break
        for score in all_scores:
            if score < 0.60:
                skip_param_60 = True
                break

        if high_score_threshold == True:
            if len(df_full_pitchsplit_unit) == 0 or skip_param == True:
                df_full_pitchsplit.loc[df_full_pitchsplit['ID'] == unit_id, 'GenFrac'] = np.nan
                continue
        elif sixty_score_threshold == True:
            if len(df_full_pitchsplit_unit) == 0 or skip_param_60 == True:
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
        df_full_pitchsplit.loc[df_full_pitchsplit['ID'] == unit_id, 'MaxScore'] = max_score

        # Now you can do something with the counts, for example, print them
        # print(f"Unit ID: {unit_id}")
        # print(f"Number of scores above 60%: {len(above_60_scores)}")
        # print(f"Number of probe words below 60%: {len(below_60_probe_words)}")
        # print("-------------------")
    return df_full_pitchsplit

def runlgbmmodel_score(df_use, optimization = False):
    col = 'Score'

    unique_IDs = df_use['ID'].unique()
    df_use = df_use.reset_index(drop=True)
    df_use['ID'] = pd.Categorical(df_use['ID'],categories=unique_IDs, ordered=True)

    #relabel the probe word labels to be the same as the paper
    #check if the probe word is a string
    if isinstance(df_use['ProbeWord'][0], str):
        df_use['ProbeWord'] = df_use['ProbeWord'].replace({ '(2,2)': 'craft', '(3,3)': 'in contrast to', '(4,4)': 'when a', '(5,5)': 'accurate', '(6,6)': 'pink noise', '(7,7)': 'of science', '(8,8)': 'rev. instruments', '(9,9)': 'boats', '(10,10)': 'today',
            '(13,13)': 'sailor', '(15,15)': 'but', '(16,16)': 'researched', '(18,18)': 'took', '(19,19)': 'the vast', '(20,20)': 'today', '(21,21)': 'he takes', '(22,22)': 'becomes', '(23,23)': 'any', '(24,24)': 'more'})
    # df_use['ProbeWord'] = df_use['ProbeWord'].replace(
    #     {2.0: 'craft', '(3,3)': 'in contrast to', '(4,4)': 'when a', '(5,5)': 'accurate', '(6,6)': 'pink noise',
    #      '(7,7)': 'of science', '(8,8)': 'rev. instruments', '(9,9)': 'boats', '(10,10)': 'today',
    #      '(13,13)': 'sailor', '(15,15)': 'but', '(16,16)': 'researched', '(18,18)': 'took', '(19,19)': 'the vast',
    #      '(20,20)': 'today', '(21,21)': 'he takes', '(22,22)': 'becomes', '(23,23)': 'any', '(24,24)': 'more'})
    else:
        df_use['ProbeWord'] = df_use['ProbeWord'].replace(
        {2.0: 'craft', 3.0: 'in contrast to', 4.0: 'when a', 5.0: 'accurate', 6.0: 'pink noise',
         7.0: 'of science', 8.0: 'rev. instruments', 9.0: 'boats', 10.0: 'today',
         13.0: 'sailor', 15.0: 'but', 16.0: 'researched', 18.0: 'took', 19.0: 'the vast',
         20.0: 'today', 21.0: 'he takes', 22.0: 'becomes', 23.0: 'any', 24.0: 'more'})

    #replace probeword with number ordered by length
    #order the probewords by length
    unique_probe_words = df_use['ProbeWord'].unique()
    #convert unique_probe_words to int
    # unique_probe_words = [int(unique_probe_word) for unique_probe_word in unique_probe_words]
    unique_probe_words = sorted(unique_probe_words, key=len)
    for i, probe in enumerate(unique_probe_words):
        df_use['ProbeWord'] = df_use['ProbeWord'].replace({probe: i})


    df_use['BrainArea'] = df_use['BrainArea'] .replace({'PEG': 2, 'AEG': 1, 'MEG': 0, 'aeg': 1, 'peg': 2, 'meg': 0})
    #remove all string value rows for brain area from the dataframe
    df_use = df_use[(df_use['BrainArea'] == 2) | (df_use['BrainArea'] == 0) | (df_use['BrainArea'] == 1)]
    #remove all AEG units
    df_use = df_use[df_use['BrainArea'] != 1]

    # df_use['BrainArea'] = df_use['BrainArea'].astype('category')
    df_use['ID'] = df_use['ID'].astype('category')
    # df_use['ProbeWord'] = df_use['ProbeWord'].astype('category')
    #if unit_type exists as a column name drop it
    # if 'unit_type' in df_use.columns:
    #     df_use = df_use.drop(columns = 'unit_type')


    # cast the probe word category as an int
    df_use['PitchShift'] = df_use['PitchShift'].astype('int')
    df_use['Below-chance'] = df_use['Below-chance'].astype('int')

    # df_use["ProbeWord"] = pd.to_numeric(df_use["ProbeWord"])
    df_use["PitchShift"] = pd.to_numeric(df_use["PitchShift"])
    df_use['BrainArea'] = pd.to_numeric(df_use['BrainArea'])
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
    col7 = 'FerretID'
    dfx = dfx.loc[:, dfx.columns != col7]
    col8 = 'animal'
    dfx = dfx.loc[:, dfx.columns != col8]
    col9 = 'recname'
    dfx = dfx.loc[:, dfx.columns != col9]
    col10 = 'stream'
    dfx = dfx.loc[:, dfx.columns != col10]
    col11 = 'tdt_electrode_num'
    dfx = dfx.loc[:, dfx.columns != col11]
    col12 = 'MeanScore'
    dfx = dfx.loc[:, dfx.columns != col12]
    col13 = 'MaxScore'
    dfx = dfx.loc[:, dfx.columns != col13]
    col14 = 'clus_id_report'
    dfx = dfx.loc[:, dfx.columns != col14]
    col15 = 'score_permutation'
    dfx = dfx.loc[:, dfx.columns != col15]
    col16 = 'unit_type'
    dfx = dfx.loc[:, dfx.columns != col16]
    col17 = 'cluster_id'
    dfx = dfx.loc[:, dfx.columns != col17]

    #remove any rows
    if optimization == True:
        params = run_optuna_study_score(dfx.to_numpy(), df_use['Score'].to_numpy())
        #save as npy file
        np.save('params0412noaeg.npy', params)
    else:
        params = np.load('params0412noaeg.npy', allow_pickle='TRUE').item()

    X_train, X_test, y_train, y_test = train_test_split(dfx.to_numpy(), df_use['Score'].to_numpy(), test_size=0.2,
                                                        random_state=42, shuffle=True)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test)

    param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    evallist = [(dtrain, 'train'), (dtest, 'eval')]


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
    shap_values_summary = shap.TreeExplainer(xg_reg).shap_values(dfx)
    explainer = shap.Explainer(xg_reg, X_train, feature_names=dfx.columns)
    shap_values2 = explainer(X_train)

    fig, ax = plt.subplots(1, figsize=(10, 10), dpi = 300)
    shap.summary_plot(shap_values_summary,dfx,  max_display=20, show = False, cmap = 'cool')
    plt.savefig(f'G:/neural_chapter/figures/summary_plot_lightgbm.png', dpi = 300)
    plt.show()
    #partial dependency plot of the pitch shift versus naive color coded by naive



    naive = shap_values2[:, "Naive"].data
    pitchshift = shap_values2[:, "PitchShift"].data
    shap_values = shap_values2[:, "Naive"].values
    data_df = pd.DataFrame({
        "pitchshift": pitchshift,
        "naive": naive,
        "SHAP value": shap_values
    })
    custom_colors = ['hotpink', 'cyan',]  # Add more colors as needed

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.violinplot(x="naive", y="SHAP value", hue="pitchshift", data=data_df, split=True, inner="quart",
                   palette=custom_colors, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Trained', 'Naive'], fontsize=20, rotation=45)
    # ax.set_xlabel('Naive', fontsize=18)
    ax.set_ylabel('Impact on decoding score', fontsize=20)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    ax.set_xlabel(None)
    #reinsert the legend_hanldes and labels
    # ax.legend()
    ax.legend(legend_handles, ['Control', 'Pitch-shifted'], loc='upper right', fontsize=13)
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

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="pitchshift", y="SHAP value", hue="naive", data=data_df, split=True, inner="quart",
                   palette=custom_colors, ax=ax)

    ax.set_xticks([ 0, 1])
    ax.set_xticklabels(['Control', 'Pitch Shifted'], fontsize=18, rotation=45)
    # ax.set_xlabel('Pitch Shift', fontsize=18)
    ax.set_xlabel(None)
    ax.set_ylabel('Impact on decoding score', fontsize=20)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    #reinsert the legend_hanldes and labels
    ax.legend(legend_handles, ['Trained', 'Naive'], loc='upper right', fontsize=13)
    plt.savefig(f'G:/neural_chapter/figures/violinplot_pitchshift.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

    fig, ax = plt.subplots(dpi = 300)
    BrainArea = shap_values2[:, "BrainArea"].data
    naive_values = shap_values2[:, "Naive"].data
    shap_values = shap_values2[:, "Naive"].values
    data_df = pd.DataFrame({
        "BrainArea": BrainArea,
        "naive": naive_values,
        "SHAP value": shap_values
    })
    sns.violinplot(x="naive", y="SHAP value", hue="BrainArea", data=data_df, split=True, inner="quart",
                     palette=custom_colors, ax=ax,)
    # ax.set_xlim(-0.5, 1.5)
    # ax.set_ylim(-0.02, 0.02)
    # ax.set_xticks([0, 1])
    ax.set_xticklabels(['Trained', 'Naive'], fontsize=18, rotation=45)
    plt.xlabel('Brain Area', fontsize=18)
    ax.set_ylabel('Impact on decoding score', fontsize=18)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    #reinsert the legend_hanldes and labels
    ax.legend(legend_handles, ['MEG', 'PEG'], loc='upper right', fontsize=13)
    plt.xlabel(None, fontsize=18)
    plt.savefig(f'G:/neural_chapter/figures/lightgbm_violinplot_naive_brainarea.png', dpi = 300, bbox_inches='tight')
    plt.show()


    #plot the pro0be word
    probe_word = shap_values2[:, "ProbeWord"].data
    naive_values = shap_values2[:, "Naive"].data
    shap_values = shap_values2[:, "ProbeWord"].values
    data_df = pd.DataFrame({
        "ProbeWord": probe_word,
        "naive": naive_values,
        "SHAP value": shap_values
    })
    fig, ax = plt.subplots(figsize=(10, 6), dpi = 300)


    # #convert back to the original labels based on the length
    # for i, probe in enumerate(unique_probe_words):
    #     print(i, probe)
    #     data_df['ProbeWord'] = data_df['ProbeWord'].replace({i: probe})
    #find all becomes
    data_df2 = data_df[data_df['ProbeWord'] == 'becomes']

    sns.violinplot(x="ProbeWord", y="SHAP value", hue="naive", data=data_df, split=True, inner="quart",
                        palette=custom_colors, ax=ax)

    # xticks = np.arange(0, 17, 1)
    # plt.xticks( xticks,labels = unique_probe_words, rotation=45)
    ax.set_xticklabels(unique_probe_words, rotation=45, fontsize=16)
    # plt.xticks(rotation = 90, fontsize = 16)
    ax.legend(legend_handles, ['Trained', 'Naive'], loc='upper right', fontsize=13)
    plt.xlabel('Probe Word', fontsize=20)
    plt.ylabel('Impact on decoding score', fontsize=20)
    plt.savefig(f'G:/neural_chapter/figures/lightgbm_violinplot_probeword.png', dpi = 300, bbox_inches='tight')
    plt.show()

    #make a bar plot of the probe word importance
    probe_word = shap_values2[:, "ProbeWord"].data
    naive_values = shap_values2[:, "Naive"].data
    shap_values = shap_values2[:, "ProbeWord"].values
    data_df = pd.DataFrame({
        "ProbeWord": probe_word,
        "naive": naive_values,
        "SHAP value": shap_values
    })
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    data_df_naive = data_df[data_df['naive'] == 1]
    data_df_trained = data_df[data_df['naive'] == 0]
    # data_df_trained['ProbeWord'] = data_df_trained['ProbeWord'].replace({ 2: 'craft', 3: 'in contrast to', 4: 'when a', 5: 'accurate', 6: 'pink noise', 7: 'of science', 8: 'rev. instruments', 9: 'boats', 10: 'today',
    #     13: 'sailor', 15: 'but', 16: 'researched', 18: 'took',19: 'the vast', 20: 'today', 21: 'he takes',22: 'becomes', 23: 'any', 24: 'more'})
    #reverse convert the probe word to the original labels
    #convert back to the original labels based on the length
    data_df_trained = data_df_trained.groupby('ProbeWord').mean().reset_index()

    for i, probe in enumerate(unique_probe_words):
        print(i, probe)
        data_df_trained['ProbeWord'] = data_df_trained['ProbeWord'].replace({i: probe})
    #plot by descending order
    data_df_trained = data_df_trained.sort_values(by=['SHAP value'], ascending=False)


    ax.barh(np.arange(0,len(data_df_trained['SHAP value']), 1), data_df_trained['SHAP value'], color='hotpink', label='Trained')
    #get the matching index for the probe words
    plt.yticks(np.arange(0, len(data_df_trained['SHAP value']), 1), data_df_trained['ProbeWord'], fontsize=16, rotation = 45)

    # ax.legend(legend_handles, ['Trained', 'Naive'], loc='upper right', fontsize=13)
    plt.xlabel('Mean impact on decoding score, trained units', fontsize=20)
    plt.ylabel('Probe Word', fontsize=20)

    plt.savefig('G:/neural_chapter/figures/lightgbm_barplot_probeword.png',
                dpi=300, bbox_inches='tight')
    plt.show()


    from sklearn.inspection import permutation_importance

    # Assuming X_test and y_test are your test data
    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=10, random_state=0)

    # Extract and print the importances

    # Map feature names to their importances
    importances = result.importances_mean

    # Map feature names to their importances
    feature_importance_dict = dict(zip(dfx.columns, importances))

    # Sort the features by importance in descending order
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Print feature importances
    for feature, importance in feature_importance_dict.items():
        print(f"{feature}: {importance}")

    features, importance = zip(*sorted_feature_importance)
    fig, ax = plt.subplots(dpi = 300, figsize=(10, 3))
    plt.barh(features, importance, color = 'skyblue', edgecolor = 'black')
    ax.set_xticks(np.arange(0, 0.12, 0.02))
    ax.set_xticklabels(ax.get_xticks(), fontsize=16)
    ax.set_yticklabels(features, fontsize=20, rotation = 45)
    plt.xlabel('Permutation Importance', fontsize = 20)
    plt.title('Permutation Importances of Features', fontsize = 20)
    plt.savefig(f'G:/neural_chapter/figures/permutation_importance_lightgbm.png', dpi = 300, bbox_inches='tight')
    plt.show()






    labels = [item.get_text() for item in ax.get_yticklabels()]
    print(labels)
