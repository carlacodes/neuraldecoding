import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Define a function to generate scatter plots
def generate_scatter_plot(ax, data, x, y, label, color):
    ax.scatter(data[x], data[y], label=label, color=color, alpha=0.5)

# Define a function to generate box plots
def generate_box_plot(ax, data, x, y, labels, colors):
    box_data = [data[y][data[x] == label] for label in labels]
    ax.boxplot(box_data, labels=labels, patch_artist=True, boxprops=dict(facecolor=colors), showfliers=False)
    ax.legend()

# Define a function to calculate Mann-Whitney U test
def mann_whitney_test(data1, data2):
    return mannwhitneyu(data1, data2, alternative='greater')

# Your data dictionaries
def generate_plots_concise(data_dicts):
    # Create a DataFrame to store the data
    data_frames = []

    for data_dict in data_dicts:
        for su_type, su_data in data_dict.items():
            for pitch_shift, ps_data in su_data.items():
                for talker, scores in ps_data.items():
                    data_frames.append(pd.DataFrame({
                        'su_type': [1 if su_type == 'su_list' else 0] * len(scores),
                        'pitch_shift': [1 if pitch_shift == 'pitchshift' else 0] * len(scores),
                        'talker': [1 if talker == 'male_talker' else 0] * len(scores),
                        'score': scores
                    }))

    # Combine all data frames into a single data frame
    merged_data = pd.concat(data_frames, ignore_index=True)

    # Create scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Scatter Plots for Different Datasets', fontsize=16)

    # Define labels and colors for scatter plots
    labels = ['cruella', 'zola', 'nala', 'crumble', 'eclair', 'ore']
    colors = ['purple', 'magenta', 'darkturquoise', 'olivedrab', 'steelblue', 'darkcyan']

    for i, (ax, data_dict, label, color) in enumerate(zip(axes.flatten(), data_dicts, labels, colors)):
        generate_scatter_plot(ax, merged_data, 'su_type', 'score', 'su', color)
        generate_scatter_plot(ax, merged_data, 'pitch_shift', 'score', 'pitchshift', 'black')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Control', 'Roved'])
        ax.set_xlabel('Talker', fontsize=12)
        ax.set_ylabel('LSTM Decoding Score', fontsize=12)
        ax.set_title(label, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Create box plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Box Plots for Different Datasets', fontsize=16)

    su_types = ['su_list', 'mu_list']
    pitch_shifts = ['nonpitchshift', 'pitchshift']

    for ax, su_type, color in zip(axes, su_types, colors):
        generate_box_plot(ax, merged_data, 'talker', 'score', ['female_talker', 'male_talker'], color)
        ax.set_xlabel('Talker', fontsize=12)
        ax.set_ylabel('LSTM Decoding Score', fontsize=12)
        ax.set_title(f'{su_type.capitalize()} - Control vs. Roved', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Perform Mann-Whitney U test
    trained_scores = merged_data[merged_data['su_type'] == 1]['score']
    naive_scores = merged_data[merged_data['su_type'] == 0]['score']
    mann_whitney_result = mann_whitney_test(trained_scores, naive_scores)
    print("Mann-Whitney U Test Result:", mann_whitney_result)

    # Create an ANOVA model
    model = ols('score ~ C(su_type) + C(pitch_shift) + C(talker)', data=merged_data).fit()
    print(model.summary())

    # Perform ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("ANOVA Table:\n", anova_table)
