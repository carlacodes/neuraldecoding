import numpy as np



def run_scores_and_plot(file_path, pitchshift, output_folder, ferretname,  stringprobewordindex=str(2), talker='female'):
    if talker == 'female':
        talker_string = 'onlyfemaletalker'
    scores = np.load(
        file_path + '/' + r'scores_' + ferretname + '_2022_' + stringprobewordindex + '_' + ferretname + '_probe_'+pitchshift +'_'+talker_string+'_bs.npy',
        allow_pickle=True)[()]

    return scores








if __name__ == '__main__':
    print('hello')
    file_path = 'E:\\lstm_kfold_01082023_oreovertimes2'
    output_folder = 'E:\\lstm_kfold_01082023_oreovertimes2'
    ferretname = 'orecchiette'
    pitchshift = 'nopitchshift'
    run_scores_and_plot(file_path, pitchshift, output_folder, ferretname, stringprobewordindex=str(2), talker = 'female' )

