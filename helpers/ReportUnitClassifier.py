import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ReportUnitClassifier:
    def __init__(self, path):
        self.path = path
        self.data_conversion = {
            'tdt': list(range(1, 33)),
            'warp': [0, 4, 1, 5, 2, 6, 3, 7, 12, 8, 13, 9, 14, 10, 15, 11, 16, 20, 17, 21, 18, 22, 19, 23, 28, 24, 29,
                     25,
                     30, 26, 31, 27]
        }

    def classify_report(self):
        ''' Classify the units in the report.csv file into mua and su
        :return: report with unit_type column
        '''
        # Read the report.csv file
        report = pd.read_csv(self.path / 'quality metrics.csv')
        path_parent = self.path.parent

        # Read cluster info tsv file
        cluster_info = pd.read_csv(path_parent / 'phy' / 'cluster_info.tsv', delimiter='\t')
        #if any l_ratio or d_prime is nan, set to 0
        report['l_ratio'] = report['l_ratio'].fillna(0)
        report['d_prime'] = report['d_prime'].fillna(0)

        report['unit_type'] = np.nan
        report.loc[(report['l_ratio'] > 4) | (report['d_prime'] > 4), 'unit_type'] = 'mua'
        report.loc[(report['l_ratio'] <= 4) & (report['d_prime'] <= 4), 'unit_type'] = 'su'
        report.loc[(report['l_ratio'] >= 5) | (report['d_prime'] > 4.5), 'unit_type'] = 'trash'

        # Give the warp number to the channel
        report['channel_id'] = cluster_info['cluster_id'].map(lambda x: cluster_info['ch'][x])
        report['warp'] = report['channel_id'].map(lambda x: self.data_conversion['warp'][x])
        report['tdt'] = report['channel_id'].map(lambda x: self.data_conversion['tdt'][x])

        # Save the results in a new csv file
        output_file = str(self.path / 'quality_metrics_classified.csv')
        report.to_csv(output_file, index=False)
        print('Classification done. Results saved to:', output_file)
        return report

    @staticmethod
    def plot_results(report):
        ''' Plot the results of the classification
        :param report: report with unit_type column
        :return:none
        '''

        plt.figure(figsize=(10, 6))
        plt.scatter(report['tdt'], report['warp'], c=report['unit_type'].map({'su': 'blue', 'mua': 'red', 'trash': 'black'}),
                    alpha=0.5)
        plt.xlabel('TDT')
        plt.ylabel('WARP')
        plt.title('Unit Classification')
        plt.show()


if __name__ == '__main__':
    #thank you to my past self for writing this script
    path = Path('E:\ms4output2\F1604_Squinty\BB2BB3_squinty_MYRIAD3_23092023_58noiseleveledit3medthreshold\BB2BB3_squinty_MYRIAD3_23092023_58noiseleveledit3medthreshold_BB2BB3_squinty_MYRIAD3_23092023_58noiseleveledit3medthreshold_BB_3\mountainsort4/report/')
    classifier = ReportUnitClassifier(path)
    report_data = classifier.classify_report()
    classifier.plot_results(report_data)
