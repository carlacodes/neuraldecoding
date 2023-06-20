from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from elephant import statistics
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from probeinterface.plotting import plot_probe

from instruments.io.phyconcatrecio import PhyConcatRecIO
from instruments.helpers.neural_analysis_helpers import NeuralDataset, align_times, generate_warp32_probe
from instruments.helpers.extract_helpers import load_bhv_data
from instruments.config import warpDataPath, figure_output
from instruments.helpers import util


def apply_filter(df, filter):
    if filter == 'Target trials':
        return df[df['catchTrial'] != 1]
    if filter == 'Catch trials':
        return df[df['catchTrial'] == 1]
    if filter == 'Level cue':
        return df[df['currAtten'] > 0]
    if filter == 'No Level Cue':
        return df[df['currAtten'] == 0]
    if filter == 'Non Correction Trials':
        return df[df['correctionTrial'] == 0]
    if filter == 'Correction Trials':
        return df[df['correctionTrial'] == 1]
    if filter == 'Sound Right':
        return df[df['side'] == 1]
    if filter == 'Sound Left':
        return df[df['side'] == 0]
    else:
        return f'Filter "{filter}" not found'


@dataclass
class concatenatedNeuralData:
    dp: str
    currNeuralDataPath: str = warpDataPath
    datatype: str = 'warp'  # Either 'neuropixel' or 'warp'
    overwrite_pkl: bool = False

    def load(self):
        print("Loading data from:", self.dp)
        phy_folder = Path(self.dp)
        self.evtypes = {'Trial Start': 'startTrialLick', 'Target Time': 'absoluteTargTimes',
                        'Release Time': 'absoluteRealLickRelease'}

        if (phy_folder / 'blocks.pkl').exists() and not self.overwrite_pkl:
            with open(phy_folder / 'blocks.pkl', 'rb') as f:
                self.blocks = pickle.load(f)
        else:
            self.reader = PhyConcatRecIO(dirname=phy_folder,
                                         currNeuralDataPath=self.currNeuralDataPath,
                                         datatype=self.datatype)

            self.blocks = self.reader.read()


            for seg in self.blocks[0].segments:
                if seg.annotations['bhv_file'] is not None:
                    seg.df_bhv = load_bhv_data(seg.annotations['bhv_file'])
                else:
                    seg.df_bhv = None

            with open(phy_folder / 'blocks.pkl', 'wb') as f:
                print('save pickle blocks file in:', phy_folder / 'blocks.pkl')
                pickle.dump(self.blocks, f)

    def align_neuron_to_ev(self,
                           cluster_id,
                           evtype,
                           filter_trials={},
                           window=[-1, 2],
                           fr_threshold=1):

        aligned_spikes = []
        for seg in self.blocks[0].segments:
            unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == cluster_id][0]

            # Only keep unit for this session if firing rate > 0.5/s
            # if statistics.mean_firing_rate(unit) < 0.5:
            #     continue

            filtered_bhv = seg.df_bhv.copy()
            if len(filter_trials) > 0:
                for filter in filter_trials:
                    filtered_bhv = apply_filter(filtered_bhv, filter)

            # Get event times
            ev_times = filtered_bhv[self.evtypes[evtype]].to_numpy()

            # Get spike times
            seg_aligned_spikes = align_times(unit.times.magnitude, ev_times, window)
            if len(aligned_spikes) == 0:
                aligned_spikes = seg_aligned_spikes
            else:
                seg_aligned_spikes[:, 0] = seg_aligned_spikes[:, 0] + aligned_spikes[-1, 0]
                aligned_spikes = np.concatenate((aligned_spikes, seg_aligned_spikes))

        return aligned_spikes

        # seg.spikes_array = np.concatenate((dataset.reader._spike_clusters, dataset.reader._spike_times/seg.annotations['sampling_frequency']), axis=1, dtype=object)

    def create_summary_pdf(self,
                           saveDir,
                           title='summary_pdf'):

        block = self.blocks[0]

        units = [st.annotations['cluster_id'] for st in block.segments[0].spiketrains if
                 st.annotations['group'] == 'good']
        self._summary_pdf(units, title, saveDir)

    def _summary_pdf(self, units, title, savedir):
        events_args = {1: {'Trial Start': ['No Level Cue'],
                           'ax_to_plot': ['A', 'D']},

                       2: {'Trial Start': ['No Level Cue', 'Sound Left'],
                           'ax_to_plot': ['B', 'E']},
                       3: {'Trial Start': ['No Level Cue', 'Sound Right'],
                           'ax_to_plot': ['C', 'F']},

                       4: {'Target Time': ['No Level Cue', 'Target trials'],
                           'ax_to_plot': ['G', 'J']},
                       5: {'Target Time': ['No Level Cue', 'Target trials', 'Sound Left'],
                           'ax_to_plot': ['H', 'K']},
                       6: {'Target Time': ['No Level Cue', 'Target trials', 'Sound Right'],
                           'ax_to_plot': ['I', 'L']},

                       7: {'Release Time': ['No Level Cue', 'Target trials'],
                           'ax_to_plot': ['M', 'P']},
                       8: {'Release Time': ['No Level Cue', 'Target trials', 'Sound Left'],
                           'ax_to_plot': ['N', 'Q']},
                       9: {'Release Time': ['No Level Cue', 'Target trials', 'Sound Right'],
                           'ax_to_plot': ['O', 'R']}, }

        # cluster_ids = [st.annotations['cluster_id'] for st in self.blocks[0].segments[0].spiketrains if st.annotations['group'] != 'noise']
        # cluster_ids = units
        saveDir = Path(savedir)
        with PdfPages(saveDir / f'{title}.pdf') as pdf:
            print(f'Saving summary figures as pdf for {self.dp}')
            for clus in tqdm(units):
                fig = self._unit_summary_figure(clus, events_args)
                pdf.savefig(fig)
                plt.close(fig)

    def _unit_summary_figure(self, cluster_id, events_args):

        colors = {'Trial Start': 'red',
                  'Target Time': 'green',
                  'Release Time': 'blue'}

        mosaic = """
            ABC
            DEF
            GHI
            JKL
            MNO
            PQR
            STT
            STT
            """
        fig = plt.figure(figsize=(20, 15), dpi=300)
        ax_dict = fig.subplot_mosaic(mosaic)
        ax_keys = list(ax_dict.keys())

        for i, (fig_num, params) in enumerate(events_args.items()):
            evtype = list(params.keys())[0]
            filters = params[evtype]
            aligned_spikes = self.align_neuron_to_ev(cluster_id, evtype, filters)

            axes = [ax_dict[k] for k in params['ax_to_plot']]
            # 0, 1, 2 when i == 0, 1, 2 and 6, 7, 8 when i == 3, 4, 5

            axes[0].scatter(x=aligned_spikes[:, 1], y=aligned_spikes[:, 0],
                            s=1, c=colors[evtype], alpha=0.8, edgecolors='none'
                            )
            binsize = 0.025
            bins = np.arange(-1, 2, binsize)
            psth, edges = np.histogram(aligned_spikes[:, 1], bins)
            psthfr = (psth / len(np.unique(aligned_spikes[:, 0]))) / binsize
            zscore = (psthfr - np.mean(psthfr)) / np.std(psthfr)

            # axes[1].hist(aligned_spikes[:,1], bins = 100,
            #     color = colors[evtype], alpha = 0.8)
            axes[1].plot(edges[:-1], zscore, color=colors[evtype], alpha=0.8)

            for ax in axes:
                ax.axvline(0, color=colors[evtype], linestyle='--')
                ax.set_xlabel('Time (s)')
                util.simple_xy_axes(ax)

            axes[0].set_title(f'Unit {cluster_id} {evtype} {" ".join(filters)}')

        unit = [st for st in self.blocks[0].segments[0].spiketrains if st.annotations['cluster_id'] == cluster_id][0]

        # self.plot_waveform(ax_dict['S'], unit)
        # self.plot_channel_map(ax_dict['T'], unit)

        quality = unit.annotations['group']
        fig.suptitle(f'Unit {cluster_id} {quality}')

        fig.tight_layout()

        return fig

    def plot_waveform(self, ax, unit):
        waveform_path = self.dp.parents[0] / 'waveforms/waveforms'
        wv_file = f'waveforms_{unit.annotations["si_unit_id"]}.npy'

        wv_data = np.load(waveform_path / wv_file)
        avg_wv = np.mean(wv_data, axis=0)
        peak_channel = int(unit.annotations['peak_info']['max_on_channel_id'])
        ax.plot(avg_wv[:, peak_channel], '-', color=3 * [.2], linewidth=5)

        util.simple_xy_axes(ax)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel(u'Amplitude (\u03bcV)')
        ax.set_xlabel('Samples')

    def plot_channel_map(self, ax, unit):
        peak_channel = int(unit.annotations['peak_info']['max_on_channel_id'])

        probe = generate_warp32_probe(radius=30)
        values = np.zeros(len(probe.device_channel_indices))
        values[int(peak_channel)] = 1
        plot_probe(probe, ax=ax, with_channel_index=False,
                   with_device_index=True, contacts_values=values)


def run_concatenated(neuraldata, dp, datatype, saveDir):
    filter_trials = {'No Level Cue'}

    # dp = Path('/Users/juleslebert/home/phd/fens_data/warp_data/Trifle_June_2022/Trifle_week_16_05_22/mountainsort4/phy')
    # neural_data = Path('/mnt/storage/WarpData/behaving/raw_june_2022')
    # datatype = 'warp'


    saveDir.mkdir(parents=False, exist_ok=True)

    dataset = concatenatedNeuralData(dp, currNeuralDataPath=neuraldata, datatype=datatype,
                                     overwrite_pkl=True)

    dataset.load()
    # dataset.create_summary_pdf(saveDir, title='summary_data_34_good')

    print(dataset)


def run_single(session_path):
    filter_trials = {'No Level Cue'}
    # ferret = 'F1903_Trifle'
    # session = 'catgt_240622_F1903_Trifle_AM_g0'
    # # session = 'catgt_180522_Trifle_PM_g0'
    # dp = neuropixelDataPath / ferret / session / f'{session[6:]}_imec0' / 'imec0_ks2'
    # session_path = Path('/media/jules/jules_SSD/data/neural_data/Neuropixels/spikesorted/')
    # session = 'catgt_190522_F1903_Trifle_AM_g0'
    # session = 'catgt_180522_Trifle_PM_g0'
    # dp = session_path / f'{session_path.name[6:]}_imec0' / 'phy_postprocessing'
    dp = session_path / f'{session_path.name[6:]}_imec0' / 'phy_postprocessing'

    saveDir = Path('/home/jules/Dropbox/Jules/Figures/Neuropixel/single_session_analysis')
    saveDir.mkdir(parents=False, exist_ok=True)

    dataset = NeuralDataset(dp, datatype='neuropixel')
    dataset.load()

    dataset.create_summary_pdf(saveDir, title=f'summary_data_firing_rate_{session_path.name}')


def main():
    data_path = Path('/media/jules/jules_SSD/data/neural_data/Neuropixels/spikesorted/')
    ferret = 'F2003_Orecchiette'
    warpData = Path('E:\Electrophysiological_Data\F2003_Orecchiette\S2/')
    saveDir = Path('D:/Data/spkfigs/ore/')

    neural_data = Path('E:\Electrophysiological_Data\F2003_Orecchiette\s2cgmod/')
    dp = Path('E:/resultskilosort\F2003_Orecchiette\phy_folder/')
    datatype = 'neuropixels'


    run_concatenated(neural_data, dp, datatype, saveDir)



    # sessions = [sess for sess in data_path.glob(f'{ferret}/catgt_*')]
    # for session_path in sessions:
    #     print(f'Summary for {session_path.name}')
    #     # try:
    #     run_single(session_path)
    #     # except:
    #     #     pass


if __name__ == '__main__':
    main()