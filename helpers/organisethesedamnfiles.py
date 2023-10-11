#read a filepath and organise the files into diretories based ont heir dates#
import mat73
import os
import datetime
from datetime import timedelta
from datetime import datetime

import scipy.io


def organise_files_into_directories(path, dates):
    '''    Organise the files in the path directory into subdirectories based on their dates
        :param path: the path to the directory containing the files
        :param dates: a list of dates, in datetime.date format, between which the files should be organised
        :return: None '''

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
                #if the date is in the format 2014-10-22, convert it to 20141022

                #remove all characters after level
                #if month or day is only one digit, add a 0 to the front
                try:
                    if len(date[0]) == 1:
                        date[0] = '0' + date[0]
                    if len(date[1]) == 1:
                        date[1] = '0' + date[1]
                    date = date[0] + date[1] + date[2][0:4]

                except:
                    date = i.split('-')[0:3]
                    if len(date[0]) == 1:
                        date[0] = '0' + date[0]
                    if date[1] == 'Jan':
                        date[1] = '01'
                    elif date[1] == 'Feb':
                        date[1] = '02'
                    elif date[1] == 'Mar':
                        date[1] = '03'

                    elif date[1] == 'Apr':
                        date[1] == '04'
                    elif date[1] == 'May':
                        date[1] = '05'
                    elif date[1] == 'Jun':
                        date[1] = '06'
                    elif date[1] == 'Jul':
                        date[1] = '07'
                    elif date[1] == 'Aug':
                        date[1] = '08'
                    elif date[1] == 'Sep':
                        date[1] = '09'
                    elif date[1] == 'Oct':
                        date[1] = '10'
                    elif date[1] == 'Nov':
                        date[1] = '11'
                    elif date[1] == 'Dec':
                        date[1] = '12'
                    if len(date[1]) == 1:
                        #check if the month is written as a string not number

                        date[1] = '0' + date[1]
                    date = date[0] + date[1] + date[2][0:4]


                #convert the date to a datetime object
                try:
                    date = datetime.strptime(date, '%d%m%Y')
                except:
                    print('cannot organise file ' + i)
                    continue
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
                if not os.path.exists(path + '/' + str(dates[-1])):
                    #if it doesn't, create it
                    os.mkdir(path + '/' + str(dates[-1]))
                #move the file to the appropriate directory
                os.rename(path + '/' + i, path + '/' + str(dates[-1]) + '/' + i)





def get_list_of_recblocks(path):
    '''    Get a list of the recording blocks in the path directory in ascending order
        :param path: the path to the directory containing the files
        :return: a list of recording blocks in ascending order
        '''
    #get all the files in the path directory
    recblocks = []
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
                date = i.split('_')
                #get the block number
                block = date[-1].split('.')[0]
                #get the recording number, which is the string of numbers after block
                rec = block.split('Block')
                rec = rec[1]
                #convert to an integer, remove the - sign
                rec = rec.split('-')
                #combine the two numbers
                rec = rec[0] + rec[1]
                #convert to an integer
                rec = int(rec)
                #append to a big list
                recblocks.append(rec)
    #sort the list
    recblocks.sort()
    print('done')
    return recblocks
















if __name__ == '__main__':
    path = 'D:\Data/F1815_Cruella/'
    # date_strings = [
    #     "02/10/2014",
    #     "05/10/2014",
    #     "10/10/2023",
    #     "15/10/2014",
    #     "22/10/2014",
    #     "23/01/2015",
    #     "11/05/2015",
    #     "17/08/2015",
    #     "13/10/2015",
    #     "07/02/2016",
    #     "09/03/2016",
    #     "15/04/2016"
    # ]

    date_strings = [
        "31/05/2020",
        "8/3/2020",
        "7/4/2019",
        "16/02/2020",
        "23/03/2020",
        "5/1/2019",
        "18/03/2019",
        "14/08/2018",
        "19/06/2020"
    ]

    date_strings_ivy =[
    "01/06/2015",
    "10/06/2015",
    "22/06/2015",
    "28/09/2015",
    "29/10/2015",
    "15/02/2016"]


    date_strings_cruella = [ "23/08/2023",
                             "28/04/2023",
                             "25/01/2023",
                             "16/11/2022",
                             "16/09/2022",
                             "05/07/2022",
                             "01/03/2022",
                             "24/02/2022",
                             "17/02/2022",
                             "09/02/2022",
                             "04/02/2022",
                             "01/02/2022",
                             "25/01/2022",
                             "14/01/2022",
                             "07/01/2022",
                             "21/12/2021",
                             "21/08/2023",
                             "24/04/2023",
                             "23/01/2023",
                             "15/11/2022",
                             "15/09/2022",
                             "30/06/2022",
                             "25/02/2022",
                             "22/02/2022",
                             "14/02/2022",
                             "07/02/2022",
                             "02/02/2022",
                             "28/01/2022",
                             "21/01/2022",
                             "19/12/2021",
                             "17/12/2021",
                             "16/12/2021"]


    date_objects = [datetime.strptime(date_string, "%d/%m/%Y") for date_string in date_strings_cruella]
    #remove the time from the date objects
    date_objects = [datetime.date(date_object) for date_object in date_objects]
    #organise the date_objects in ascencind order
    date_objects.sort()
    #
    organise_files_into_directories(path, date_objects)
    get_list_of_recblocks('D:\Data\F1306_Firefly/2014-10-22')