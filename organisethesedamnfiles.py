#read a filepath and organise the files into diretories based ont heir dates#
import mat73
import os
import datetime
from datetime import timedelta
from datetime import datetime

import scipy.io


def organise_files_into_directories(path, dates):
    #read the matlab files and extract their dates

    #get all the files in the path directory
    for i in os.listdir(path):
        #check if the file is a .mat file
        if i.endswith('.mat'):
            #load the file
            try:
                data = mat73.loadmat(path + '/' + i)
                date = datetime.fromordinal(int(date)) + timedelta(days=date % 1) - timedelta(days=366)
                date = datetime.date(date)
            except:
                #extract the date from the file name
                date = i.split('_')[0:3]
                #remove all characters after level
                #if month or day is only one digit, add a 0 to the front
                if len(date[0]) == 1:
                    date[0] = '0' + date[0]
                if len(date[1]) == 1:
                    date[1] = '0' + date[1]

                date = date[0] + date[1] + date[2][0:4]
                #convert the date to a datetime object
                date = datetime.strptime(date, '%d%m%Y')
                #remove the time
                date = datetime.date(date)


            #convert this matlab date to a python date

            #check if the date is between the dates
            if date > dates[0] and date < dates[-1]:

                #figure out between which date the mat file belongs in
                for j in range(len(dates)-1):
                    if date > dates[j] and date < dates[j+1]:
                        #if it is, check if the directory exists
                        if not os.path.exists(path + '/' + str(dates[j])):
                            #if it doesn't, create it
                            os.mkdir(path + '/' + str(dates[j]))
                        #move the file to the appropriate directory
                        os.rename(path + '/' + i, path + '/' + str(dates[j]) + '/' + i)

            elif date < dates[0]:
                #if it isn't, check if the directory exists
                if not os.path.exists(path + '/' + str(dates[0])):
                    #if it doesn't, create it
                    os.mkdir(path + '/' + str(dates[0]))
                #move the file to the appropriate directory
                os.rename(path + '/' + i, path + '/' + str(dates[0]) + '/' + i)
            elif date > dates[-1]:
                #if it isn't, check if the directory exists
                if not os.path.exists(path + '/' + str(dates[1])):
                    #if it doesn't, create it
                    os.mkdir(path + '/' + str(dates[1]))
                #move the file to the appropriate directory
                os.rename(path + '/' + i, path + '/' + str(dates[1]) + '/' + i)



















if __name__ == '__main__':
    path = 'D:\Data\F1306_Firefly'
    date_strings = [
        "02/10/2014",
        "05/10/2014",
        "10/10/2023",
        "15/10/2014",
        "22/10/2014",
        "23/01/2015",
        "11/05/2015",
        "17/08/2015",
        "13/10/2015",
        "07/02/2016",
        "09/03/2016",
        "15/04/2016"
    ]

    date_objects = [datetime.strptime(date_string, "%d/%m/%Y") for date_string in date_strings]
    #remove the time from the date objects
    date_objects = [datetime.date(date_object) for date_object in date_objects]
    organise_files_into_directories(path, date_objects)