#read a filepath and organise the files into diretories based ont heir dates#
import mat73
import os
import datetime
from datetime import timedelta
from datetime import datetime
import numpy as np
import scipy.io
def remove_useless_struct_from_data(path):
    #read the matlab files and extract their dates    for i in os.listdir(path):
        #check if the file is a .mat file
    for i in os.listdir(path):

        if i.endswith('.mat'):

            data = scipy.io.loadmat(path + '/' + i)['data']
            column_index_to_remove = 64
            #remove the useless struct
            data_without_column = np.delete(data, column_index_to_remove, axis=0)

            #isolate data

            #save the data


            scipy.io.savemat(path + '/' + i, {'data': data_without_column})

    return


if __name__ == '__main__':
    remove_useless_struct_from_data('D:\Data\F1604_Squinty/')