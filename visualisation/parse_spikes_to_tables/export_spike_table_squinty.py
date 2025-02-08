import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import neo
import matplotlib.pyplot as plt
from viziphant.rasterplot import rasterplot
from instruments.helpers.neural_analysis_helpers import get_word_aligned_raster_squinty
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
TARGET_FILTER = ['Target trials', 'No Level Cue']
PROBE_FILTER = ['No Level Cue']
BINSIZE = 0.01
WINDOW = [0, 0.6]

def load_blocks(data_path: Path) -> List[Any]:
    """Load blocks from a pickle file."""
    try:
        with open(data_path / 'blocks.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load blocks: {e}")
        raise

def process_raster(blocks: List[Any], cluster_id: int, word: int, pitchshift: bool, df_filter: List[str]) -> np.ndarray:
    """Process raster data for a given cluster and word."""
    try:
        raster = get_word_aligned_raster_squinty(blocks, cluster_id, word=word, pitchshift=pitchshift,
                                                correctresp=False, df_filter=df_filter)
        return raster.reshape(raster.shape[0], )
    except Exception as e:
        logging.warning(f"No relevant firing for cluster {cluster_id}, word {word}: {e}")
        return np.array([])

def extract_spike_data(raster: np.ndarray, cluster_id: int, word_id: int, pitchshift: bool) -> List[Dict[str, Any]]:
    """Extract spike data from raster and return a list of dictionaries, grouped by trial."""
    spike_data = []
    unique_trials = np.unique(raster['trial_num'])

    for trial in unique_trials:
        trial_spikes = raster[raster['trial_num'] == trial]['spike_time']
        spike_data.append({
            'unit_id': cluster_id,
            'distractor_word_id': word_id if trial > np.max(raster['trial_num']) // 2 else 1,
            'spike_times': trial_spikes.tolist(),  # Store spike times as a list
            'pitch_shift': pitchshift,
            'trial_id': trial
        })
    return spike_data

def plot_raster(spiketrains: List[neo.SpikeTrain], title: str, save_path: Path, color: str = 'black'):
    """Plot raster and save the figure."""
    fig, ax = plt.subplots(2, figsize=(10, 5))
    rasterplot(spiketrains, c=color, histogram_bins=100, axes=ax, s=3)
    ax[0].set_ylabel('Trial')
    ax[0].set_xlabel('Time relative to word presentation (s)')
    custom_xlim = (-0.1, 0.6)
    plt.setp(ax, xlim=custom_xlim)
    plt.suptitle(title, fontsize=12)
    plt.savefig(save_path)
    plt.close()

def target_vs_probe_with_raster(blocks: List[Any], talker: int = 1, probewords: Tuple[int, int] = (20, 22),
                                pitchshift: bool = True) -> pd.DataFrame:
    """Process target and probe trials, extract spike data, and return a DataFrame."""
    results = []
    probeword = probewords[0] if talker == 1 else probewords[1]

    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains if
                 st.annotations['group'] != 'noise']

    for cluster_id in clust_ids:
        logging.info(f'Processing cluster {cluster_id}')

        # # Process target trials
        # raster_target = process_raster(blocks, cluster_id, word=1, pitchshift=pitchshift, df_filter=TARGET_FILTER)
        # if raster_target.size == 0:
        #     continue

        # Process probe trials
        raster_probe = process_raster(blocks, cluster_id, word=probeword, pitchshift=pitchshift, df_filter=PROBE_FILTER)
        if raster_probe.size == 0:
            continue
        # raster_probe['trial_num'] += np.max(raster_target['trial_num'])

        # Combine target and probe trials
        raster_combined = np.concatenate([raster_probe])

        # Extract spike data (grouped by trial)
        spike_data = extract_spike_data(raster_probe, cluster_id, probeword, pitchshift)
        results.extend(spike_data)

        # Group spikes by trial for raster plot
        unique_trials = np.unique(raster_target['trial_num'])
        spiketrains = []
        for trial in unique_trials:
            trial_spikes = raster_target[raster_target['trial_num'] == trial]['spike_time']
            if trial_spikes.size > 0:  # Skip trials with no spikes
                spiketrain = neo.SpikeTrain(trial_spikes, units='s', t_start=WINDOW[0], t_stop=WINDOW[1])
                spiketrains.append(spiketrain)

        # Plot raster
        if spiketrains:
            plot_raster(
                spiketrains,
                title=f'Target firings for cluster {cluster_id}, pitchshift={pitchshift}, talker={talker}',
                save_path=Path(f'targ_clusterid_{cluster_id}_probeword_{probeword}_pitch_{pitchshift}_talker_{talker}.png')
            )

    return pd.DataFrame(results)

def run_classification(data_path: Path) -> pd.DataFrame:
    """Run classification for all probewords and talkers."""
    blocks = load_blocks(data_path)
    probewords_list = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14)]

    all_results = []
    for probeword in probewords_list:
        for talker in [1]:
            results_df = target_vs_probe_with_raster(blocks, talker=talker, probewords=probeword, pitchshift=False)
            all_results.append(results_df)

    return pd.concat(all_results, ignore_index=True)

def main():
    """Main function to run the analysis."""
    directories = ['zola_2022']
    for dir in directories:
        data_path = Path(f'E:/ms4output2/F1604_Squinty/BB2BB3_squinty_MYRIAD3_23092023_58noiseleveledit3medthreshold/BB2BB3_squinty_MYRIAD3_23092023_58noiseleveledit3medthreshold_BB2BB3_squinty_MYRIAD3_23092023_58noiseleveledit3medthreshold_BB_2/mountainsort4/phy')
        results_df = run_classification(data_path)
        results_df.to_csv(f'{dir}_spike_data.csv', index=False)
        logging.info(f"Results saved to {dir}_spike_data.csv")

if __name__ == '__main__':
    main()