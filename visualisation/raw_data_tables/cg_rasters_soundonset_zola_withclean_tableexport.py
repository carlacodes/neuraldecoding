import pickle
from pathlib import Path
import tensorflow as tf
import neo
import numpy as np
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from numba import njit, prange
# import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
from keras import backend as K
from viziphant.rasterplot import rasterplot


from instruments.helpers.neural_analysis_helpers import get_soundonset_alignedraster, split_cluster_base_on_segment_zola, get_soundonset_alignedraster_tabular, get_word_aligned_raster
import pickle





def run_cleaning_of_rasters(blocks, datapath):
    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']
    for cluster_id in clust_ids:
        new_blocks = split_cluster_base_on_segment_zola(blocks, cluster_id, num_clusters=2)
    with open(datapath / 'new_blocks.pkl', 'wb') as f:
        pickle.dump(new_blocks, f)
    return new_blocks
def get_spike_times_tabular(blocks):
    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']

    for st in blocks[0].segments[0].spiketrains:
        print(f"Cluster ID: {st.annotations['cluster_id']}, Group: {st.annotations['group']}")

    cluster_spiketime_dict = {cluster_id: {} for cluster_id in
                              clust_ids}  # Initialize the dictionary with cluster_id keys

    for cluster_id in clust_ids:
        print('now starting cluster')
        print(cluster_id)

        filter = ['No Level Cue']  # , 'Non Correction Trials']

        # try:
        spike_times_list, df_behavior_cluster = get_soundonset_alignedraster_tabular(blocks, cluster_id, df_filter=filter)
        cluster_spiketime_dict[cluster_id]['spike_times'] = spike_times_list
        cluster_spiketime_dict[cluster_id]['behavior'] = df_behavior_cluster

    return cluster_spiketime_dict


def generate_rasters(save_dir='D:/spkvisanddecodeproj2/analysisscriptsmodcg/visualisation/data'):
    base_path = Path('E:/ms4output2')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for subfolder in base_path.glob('**/phy'):
        if not subfolder.name.endswith('.phy'):
            datapath = subfolder
            print('now on data path: {}'.format(datapath))
            stream = str(datapath).split('\\')[-3]
            stream = stream[-4:]
            ferret_name = str(datapath).split('\\')[2]  # Assuming the ferret name is the third element in the path
            print(f"Processing stream: {stream} for ferret: {ferret_name}")

            try:
                with open(datapath / 'new_blocks.pkl', 'rb') as f:
                    blocks = pickle.load(f)
            except:
                try:
                    with open(datapath / 'blocks.pkl', 'rb') as f:
                        blocks = pickle.load(f)
                except:
                    print(f"No blocks found for {datapath}")
                    continue

            spike_time_dict = get_spike_times_tabular(blocks)

            # Save the spike_time_dict with the ferret name and stream in the filename
            folder_name = str(datapath).split('\\')[2]
            save_filename = f"{ferret_name}_{stream}_{folder_name}_spike_times.pkl"
            save_path = save_dir / save_filename
            with open(save_path, 'wb') as f:
                pickle.dump(spike_time_dict, f)
            print(f"Saved spike times to {save_path}")


def main():
    generate_rasters()

if __name__ == '__main__':
    main()

