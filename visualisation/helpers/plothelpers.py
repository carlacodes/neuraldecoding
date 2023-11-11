import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class PlotHelpers():

    def make_boxplots(mergedtrained, mergednaive, mergednaiveanimaldict, dictoutput_zola, dictoutput_crumble,
                      dictoutput_eclair, dictoutput_cruella, dictoutput_nala):
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

                print('mean score for zola for both talkers' + pitchshiftornot + sutype + 'is' + str(
                    np.mean(np.concatenate(
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
                ax[count].set_title(
                    'Trained LSTM scores for ' + stringtitle + ' units, \n' + stringtitlepitch + ' trials')
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
                ax[count].boxplot([mergednaive[sutype][pitchshiftornot]['female_talker'],
                                   mergednaive[sutype][pitchshiftornot]['male_talker']])
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
                ax[count].set_title(
                    'Naive LSTM scores for ' + stringtitle + ' units,\n ' + stringtitlepitch + ' trials')
                y_crum = dictoutput_crumble[sutype][pitchshiftornot]['female_talker']
                y_eclair = dictoutput_eclair[sutype][pitchshiftornot]['female_talker']
                # Add some random "jitter" to the x-axis
                x = np.random.normal(1, 0.04, size=len(y_crum))
                x2 = np.random.normal(1, 0.04, size=len(y_eclair))
                ax[count].plot(x, y_crum, ".", color='mediumblue', alpha=0.2, label='F1901')
                ax[count].plot(x2, y_eclair, ".", color='darkorange', alpha=0.2, label='F1902')
                print('mean score for naive male ' + pitchshiftornot + sutype + 'is' + str(
                    np.mean(mergednaive[sutype][pitchshiftornot]['male_talker'])) + 'and std is' + str(
                    np.std(mergednaive[sutype][pitchshiftornot]['male_talker'])))
                print('mean score for naive female ' + pitchshiftornot + sutype + 'is' + str(
                    np.mean(mergednaive[sutype][pitchshiftornot]['female_talker'])) + 'and std is' + str(
                    np.std(mergednaive[sutype][pitchshiftornot]['female_talker'])))
                print('mean score for naive for both talkers' + pitchshiftornot + sutype + 'is' + str(np.mean(
                    np.concatenate((mergednaive[sutype][pitchshiftornot]['male_talker'],
                                    mergednaive[sutype][pitchshiftornot]['female_talker'])))) + 'and std is' + str(
                    np.std(
                        np.concatenate((mergednaive[sutype][pitchshiftornot]['male_talker'],
                                        mergednaive[sutype][pitchshiftornot]['female_talker'])))))

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
            ax[count].set_title(
                'Trained LSTM scores for' + ' single and multi-units,\n ' + stringtitlepitch + ' trials')
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
            print('mean score forTRAINED for both talkers' + pitchshiftornot + sutype + 'is' + str(
                np.mean(np.concatenate((
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

    def general_plots(dictlist_naive, dictlist_trained, probeword_to_text):
        ## do the same for the naive animals
        scoredict_naive = {}
        scoredict = {}
        for talker in [1]:
            if talker == 1:
                talker_key = 'female_talker'
            for i, dict in enumerate(dictlist_naive):
                for key in dict['su_list_probeword']:
                    probewords = dict['su_list_probeword'][key][talker_key]
                    count = 0
                    for probeword in probewords:
                        probeword_range = int(probeword)
                        probewordtext = probeword_to_text.get(probeword_range)
                        if probewordtext:
                            scoredict_naive[probewordtext][talker_key][key]['su_list'].append(
                                dict['su_list'][key][talker_key][count])
                        count = count + 1
            for key in dict['mu_list_probeword']:
                probewords = dict['mu_list_probeword'][key][talker_key]
                count = 0
                for probeword in probewords:
                    probeword_range = int(probeword)
                    probewordtext = probeword_to_text.get(probeword_range)
                    if probewordtext:
                        scoredict_naive[probewordtext][talker_key][key]['mu_list'].append(
                            dict['mu_list'][key][talker_key][count])
                    count = count + 1

        # plot each mean across probeword as a bar plot
        fig, ax = plt.subplots(1, figsize=(10, 10), dpi=300)
        plot_count = 0
        for probeword in scoredict.keys():
            su_list_nops = scoredict_naive[probeword]['female_talker']['nonpitchshift']['su_list']
            mu_list_nops = scoredict_naive[probeword]['female_talker']['nonpitchshift']['mu_list']
            # get the mean of the su_list and mu_list
            total_control = su_list_nops + mu_list_nops
            mean = np.mean(total_control)
            std = np.std(total_control)

            # do the same for the pitchshift
            su_list_ps = scoredict_naive[probeword]['female_talker']['pitchshift']['su_list']
            mu_list_ps = scoredict_naive[probeword]['female_talker']['pitchshift']['mu_list']
            total_rove = su_list_ps + mu_list_ps
            mean_rove = np.mean(total_rove)
            std_rove = np.std(total_rove)
            # plot the bar plot
            if plot_count == 0:
                ax.bar(plot_count, mean, yerr=std, color='cyan', alpha=0.5, label='control')
                plot_count += 1
                ax.bar(plot_count, mean_rove, yerr=std_rove, color='blue', alpha=0.5, label='rove')
            else:

                ax.bar(plot_count, mean, yerr=std, color='cyan', alpha=0.5, label=None)
                plot_count += 1
                ax.bar(plot_count, mean_rove, yerr=std_rove, color='blue', alpha=0.5, label=None)

            plot_count += 1
        plt.legend(fontsize=8)
        plt.xlabel('probe word')
        # ax.set_xticks(np.arange(0, 16, 1))
        ax.set_xticklabels(
            [(2, 2), (5, 6), (42, 49), (32, 38), (20, 22), (15, 15), (42, 49), (4, 4), (16, 16), (7, 7), (8, 8), (9, 9),
             (10, 10), (11, 11), (12, 12),
             (14, 14)])
        plt.show()

        # swarm plot equivalent
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        data = []
        x_positions = []
        hue = []
        # Iterate over the keys in your dictionary
        for probeword in scoredict.keys():
            su_list_nops = scoredict_naive[probeword]['female_talker']['nonpitchshift']['su_list']
            mu_list_nops = scoredict_naive[probeword]['female_talker']['nonpitchshift']['mu_list']
            total_control = su_list_nops + mu_list_nops

            su_list_ps = scoredict_naive[probeword]['female_talker']['pitchshift']['su_list']
            mu_list_ps = scoredict_naive[probeword]['female_talker']['pitchshift']['mu_list']
            total_rove = su_list_ps + mu_list_ps

            # Create a DataFrame for seaborn
            control_df = pd.DataFrame({'Data': total_control, 'Probe Word': probeword, 'Category': 'Control'})
            rove_df = pd.DataFrame({'Data': total_rove, 'Probe Word': probeword, 'Category': 'Rove'})

            # Append the data and category
            data.extend(total_control)
            data.extend(total_rove)
            x_positions.extend([probeword] * (len(total_control) + len(total_rove)))
            hue.extend(['Control'] * len(total_control))
            hue.extend(['Rove'] * len(total_rove))

        # Create the violin plot
        sns.violinplot(x=x_positions, y=data, hue=hue, palette={"Control": "cyan", "Rove": "blue"}, split=True, ax=ax)
        # Customize the plot
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.xlabel('Probe Word')
        plt.ylabel('Data')  # Update with your actual data label
        plt.legend(title='Category')
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        data = []
        x_positions = []
        hue = []

        # Iterate over the keys in your dictionary
        for probeword in scoredict.keys():
            su_list_nops = scoredict_naive[probeword]['female_talker']['nonpitchshift']['su_list']
            mu_list_nops = scoredict_naive[probeword]['female_talker']['nonpitchshift']['mu_list']
            total_control = su_list_nops + mu_list_nops

            su_list_ps = scoredict_naive[probeword]['female_talker']['pitchshift']['su_list']
            mu_list_ps = scoredict_naive[probeword]['female_talker']['pitchshift']['mu_list']
            total_rove = su_list_ps + mu_list_ps

            # Create a DataFrame for seaborn
            control_df = pd.DataFrame({'Data': total_control, 'Probe Word': probeword, 'Category': 'Control'})
            rove_df = pd.DataFrame({'Data': total_rove, 'Probe Word': probeword, 'Category': 'Rove'})

            # Append the data and category
            data.extend(total_control)
            data.extend(total_rove)
            x_positions.extend([probeword] * (len(total_control) + len(total_rove)))
            hue.extend(['Control'] * len(total_control))
            hue.extend(['Rove'] * len(total_rove))

            # Create the violin plot
        sns.violinplot(x=x_positions, y=data, hue=hue, palette={"Control": "cyan", "Rove": "blue"}, split=True, ax=ax)

        # Scatter plot for raw data
        scatter_data = pd.DataFrame({'x_positions': x_positions, 'Data': data, 'Category': hue})

        sns.scatterplot(x="x_positions", y="Data", hue="Category", data=scatter_data, s=20, ax=ax,
                        palette={"Control": "white", "Rove": "yellow"})

        # Customize the plot
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.xlabel('Probe Word')
        plt.ylabel('Data')  # Update with your actual data label
        plt.legend(title='Category')

        plt.show()

        # now do the same for the trained animals
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

        data = []
        x_positions = []
        hue = []

        # Iterate over the keys in your dictionary
        for probeword in scoredict.keys():
            su_list_nops = scoredict[probeword]['female_talker']['nonpitchshift']['su_list']
            mu_list_nops = scoredict[probeword]['female_talker']['nonpitchshift']['mu_list']
            total_control = su_list_nops + mu_list_nops

            su_list_ps = scoredict[probeword]['female_talker']['pitchshift']['su_list']
            mu_list_ps = scoredict[probeword]['female_talker']['pitchshift']['mu_list']
            total_rove = su_list_ps + mu_list_ps

            # Create a DataFrame for seaborn
            control_df = pd.DataFrame({'Data': total_control, 'Probe Word': probeword, 'Category': 'Control'})
            rove_df = pd.DataFrame({'Data': total_rove, 'Probe Word': probeword, 'Category': 'Rove'})

            # Append the data and category
            data.extend(total_control)
            data.extend(total_rove)
            x_positions.extend([probeword] * (len(total_control) + len(total_rove)))

            hue.extend(['Control'] * len(total_control))
            hue.extend(['Rove'] * len(total_rove))

            # Create the violin plot
        sns.violinplot(x=x_positions, y=data, hue=hue, palette={"Control": "purple", "Rove": "pink"}, split=True, ax=ax)

        # Scatter plot for raw data
        scatter_data = pd.DataFrame({'x_positions': x_positions, 'Data': data, 'Category': hue})

        sns.scatterplot(x="x_positions", y="Data", hue="Category", data=scatter_data, s=15, ax=ax,
                        palette={"Control": "white", "Rove": "black"})

        # Customize the plot
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.xlabel('Probe Word')
        plt.ylabel('Data')  # Update with your actual data label
        plt.legend(title='Category')

        plt.show()

        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        plot_count = 0
        x_offset = 0  # Initialize x-coordinate offset
        xtick_labels = []

        for probeword in scoredict.keys():
            su_list_nops = scoredict[probeword]['female_talker']['nonpitchshift']['su_list']
            mu_list_nops = scoredict[probeword]['female_talker']['nonpitchshift']['mu_list']
            total_control = su_list_nops + mu_list_nops

            su_list_ps = scoredict[probeword]['female_talker']['pitchshift']['su_list']
            mu_list_ps = scoredict[probeword]['female_talker']['pitchshift']['mu_list']
            total_rove = su_list_ps + mu_list_ps

            sns.swarmplot(x=np.array([plot_count] * len(total_control)) + x_offset, y=total_control, color='purple',
                          alpha=0.5, label='control')
            x_offset += 0.2  # Adjust the offset to separate points

            sns.swarmplot(x=np.array([plot_count] * len(total_rove)) + x_offset, y=total_rove, color='pink', alpha=0.5,
                          label='rove')
            x_offset += 0.4  # Adjust the offset for the next category

            plot_count += 1
            # xtick_labels.append(str((2, 2)))  # Adjust this line to add appropriate labels

        plt.legend(fontsize=8)
        plt.xlabel('probe word')
        ax.set_xticks(np.arange(0, plot_count))
        # ax.set_xticklabels([(2, 2), (5, 6), (42, 49), (32, 38), (20, 22), (15, 15), (42, 49), (4, 4), (16, 16), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
        #                       (14, 14)])
        plt.show()
