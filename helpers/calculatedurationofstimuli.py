import mat73
import scipy
import pickle

def load_files_and_calc_duration():
    stimuli = scipy.io.loadmat('D:/FemaleSounds24k.mat')
    audio = stimuli['s']
    # audio = audio[0]
    talker_duration_list = []
    for i in range(len(audio)):
        print(len(audio[i]))
        #access each word in the stimuli
        talker = audio[i]
        talker = talker[0]
        talker = talker[0]
        duration_list = []
        for j in range(len(talker)):
            word = talker[j]

            #get the duration of the word
            duration = len(word)
            print(duration)
            duration_list.append(duration)
        print(duration_list)
        talker_duration_list.append(duration_list)


        #get the duration of the word
    print('stim')
    #save this list as a pickle file
    with open('D:/talker_duration_list.pkl', 'wb') as f:
        pickle.dump(talker_duration_list, f)





if __name__ == '__main__':
    load_files_and_calc_duration()