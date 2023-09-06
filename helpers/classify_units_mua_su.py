
import pandas as pd
import numpy as np
import os
from pathlib import Path



def classify_report(path):
    '''   a function that is going to automatically read a report.csv file, extract the l ratio and then classify each unit as mua or su based on it
    then save the results in a new csv file
    :param path: the path to the directory containing the report.csv file
    :return: report with the updated csv file that sorts the units based on the l_ratio '''

    data_conversion = {
        'tdt': list(range(1, 33)),
        'warp': [0, 4, 1, 5, 2, 6, 3, 7, 12, 8, 13, 9, 14, 10, 15, 11, 16, 20, 17, 21, 18, 22, 19, 23, 28, 24, 29, 25,
                 30, 26, 31, 27]
    }

    #read the report.csv file
    report = pd.read_csv(path / 'quality metrics.csv')
    #get the level above ad read the phy directory
    path_parent = path.parent
    #read cluster info tsv file
    cluster_info = pd.read_csv(path_parent / 'phy' / 'cluster_info.tsv', delimiter='\t')

    report['unit_type'] = np.nan

    report.loc[report['l_ratio'] > 4 , 'unit_type'] = 'mua'
    report.loc[report['l_ratio'] < 4, 'unit_type'] = 'su'
    #give the warp number to the channel
    #get the corresponding channel id from the unit list
    report['channel_id'] = cluster_info['cluster_id'].map(lambda x: cluster_info['ch'][x])
    report['warp'] = report['channel_id'].map(lambda x: data_conversion['warp'][x])
    report['tdt'] = report['channel_id'].map(lambda x: data_conversion['tdt'][x])


    #save the results in a new csv file
    #get the path to the directory
    path_to_dir = os.path.dirname(path)
    #get the name of the directory
    dir_name = os.path.basename(path_to_dir)
    #save the results in a new csv file
    report.to_csv(str(path)+'/' + 'quality_metrics_classified.csv')
    print('done')
    return report










if __name__ == '__main__':
    #create a function that is going to automatically read a report.csv file, extract the l ratio and then classify each unit as mua or su based on it
    #then save the results in a new csv file
    #then create a function that is going to read the csv file and plot the results
    path = Path('E:/report/')

    classify_report(path)