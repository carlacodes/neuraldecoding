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