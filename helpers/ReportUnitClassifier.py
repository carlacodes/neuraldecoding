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


        report['unit_type'] = np.nan
        report.loc[report['l_ratio'] > 4, 'unit_type'] = 'mua'
        report.loc[report['l_ratio'] < 4, 'unit_type'] = 'su'

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
        plt.scatter(report['tdt'], report['warp'], c=report['unit_type'].map({'su': 'blue', 'mua': 'red'}))
        plt.xlabel('TDT')
        plt.ylabel('WARP')
        plt.title('Unit Classification')
        plt.show()


if __name__ == '__main__':
    path = Path('E:/report/')
    classifier = ReportUnitClassifier(path)
    report_data = classifier.classify_report()
    classifier.plot_results(report_data)
