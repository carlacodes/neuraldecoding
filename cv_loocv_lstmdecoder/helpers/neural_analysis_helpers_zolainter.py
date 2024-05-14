import numpy as np
import mat73
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from math import isinf
from numba import njit
from elephant import statistics

from neo.core import SpikeTrain, Segment
from sklearn.cluster import KMeans
import numpy as np
import copy
import quantities as pq

from collections import Counter
from probeinterface import generate_multi_columns_probe
from instruments.io.archive.phywarpio import PhyWarpIO
from instruments.io.archive.phynpxlio import PhyNpxlIO
from helpers.analysis_helpers import apply_filter


def extractDataForTrialStruct(file):
    spikedat = mat73.loadmat(file)
    fs = spikedat['allInfo'][1]['fs']

    extractUnits = [u[0] for u in spikedat['spikesPerChan'] if u[2] is not None]
    units = []
    for unit in extractUnits:
        if len(unit) > 1:
            for su in unit:
                units.append(su)
        else:
            units.append(unit)

    bHvData = spikedat['allInfo'][2]

    spikeDataobj = np.array(units, dtype=object) / fs
    spikeData = [sp[0] for sp in spikeDataobj]

    trialStruct = createTrialStruct(bHvData, spikeData)
    trialStruct['BlockName'] = spikedat['allInfo'][0]

    return trialStruct


# def createTrialStruct(bHvData,
#                       spikeData,
#                       trialRange=[-2, 6]):

#     trialStruct = bHvData
#     spikeTimings = []
#     for i, startTrial in enumerate(bHvData['startTrialLick']):
#         sptime = []
#         for j, spikeUnit in enumerate(spikeData):
#             spikeTrial = [sp for sp in spikeUnit if (sp > (startTrial + trialRange[0])) if (sp < (startTrial + trialRange[1]))]
#             sptime.append(spikeTrial - startTrial)
#         spikeTimings.append(sptime)
#     trialStruct['spikeTimings']=spikeTimings

#     return trialStruct


def sptimes2binary(spTimes, fs, samplesize):
    spikematrix = np.zeros(samplesize)
    for sp in spTimes:
        spikematrix[int(sp * fs)] = 1

    return spikematrix


def createTrialStruct(block):
    groups = block.groups
    trials = [tr.annotations['trial'] for tr in block.segments[0].events]
    BhvDataDict = defaultdict(list)
    trialSpikes = []
    probeSpikes = []
    probetimings = []
    targetSpikes = []
    for i, trial in enumerate(trials):
        for key, value in trial.items():
            BhvDataDict[key].append(value)

        probes = BhvDataDict['probes'][i]

        if np.shape(probes) == ():
            probes = [probes]

        sptr = []
        trialprobespikes = []
        trialtargetspikes = []
        for unit in groups:
            sptr.append(unit.spiketrains[i])
            probespk = []
            probetm = []
            for probe in probes:
                probespiketrain, probetiming = gettrialprobespiketrain(unit.spiketrains[i], trial, probe)
                probespk.append(probespiketrain)
                probetm.append(probetiming)
            trialprobespikes.append(probespk)

            targetspk, targettiming = gettrialprobespiketrain(unit.spiketrains[i], trial, 1)
            trialtargetspikes.append(targetspk)

        trialSpikes.append(sptr)
        probeSpikes.append(trialprobespikes)
        probetimings.append(probetm)

        targetSpikes.append(trialtargetspikes)

    allTrialStruct = pd.DataFrame(BhvDataDict)
    allTrialStruct['trialSpikes'] = trialSpikes
    allTrialStruct['probeSpikes'] = probeSpikes
    allTrialStruct['probetimings'] = probetimings
    allTrialStruct['targetSpikes'] = targetSpikes

    probestart = {}
    for pr, probe in enumerate(probes):
        probe = int(probe)
        probestart[probe] = []
        for index, trial in allTrialStruct.iterrows():
            try:
                probestart[probe].append(trial['probetimings'][pr][0])
            except:
                probestart[probe].append(trial['probetimings'][pr])

        allTrialStruct['probestart{}'.format(probe)] = probestart[probe]

    return allTrialStruct


def gettrialprobespiketrain(spiketrain, trial, probe):
    if probe in trial['distractors']:
        probeposition = np.where(trial['distractors'] == probe)[0]
        dDurs = np.array(trial['dDurs'])
        probetiming = np.array([np.sum(dDurs[:probeposition[0]]), np.sum(dDurs[:probeposition[0] + 1])])
        probetiming = (probetiming / fs) * pq.s
        result = spiketrain.times - probetiming[0]
        mask = (spiketrain.times > probetiming[0]) & (spiketrain.times < probetiming[1])
        out = np.where(mask, result, np.nan)
        probespiketimes = out[~np.isnan(out)]
        probespiketrain = neo.SpikeTrain(times=probespiketimes,
                                         units='sec',
                                         t_stop=probetiming[1] - probetiming[0])

    else:
        probespiketrain = np.nan
        probetiming = np.nan

    return probespiketrain, probetiming


def gettrialonsetprobespiketrain(spiketrain, trial, probe):
    if probe in trial['distractors']:
        probeposition = np.where(trial['distractors'] == probe)[0]
        dDurs = np.array(trial['dDurs'])
        probetiming = np.array([np.sum(dDurs[:probeposition[0]]), np.sum(dDurs[:probeposition[0] + 1])])
        probetiming = (probetiming / fs) * pq.s
        result = spiketrain.times  # - probetiming[0]
        mask = (spiketrain.times > probetiming[0]) & (spiketrain.times < probetiming[1])
        out = np.where(mask, result, np.nan)
        probespiketimes = out[~np.isnan(out)]
        probespiketrain = neo.SpikeTrain(times=probespiketimes,
                                         units='sec',
                                         t_stop=8)

    else:
        probespiketrain = np.nan
        probetiming = np.nan

    return probespiketrain, probetiming


def get_spiketrains(seg, quality='all'):
    assert quality in ['all', 'good', 'mua', 'noise']

    if quality == 'all':
        spiketrains = [st for st in seg.spiketrains]
    else:
        spiketrains = [st for st in seg.spiketrains \
                       if st.annotations['quality'] == quality]

    return spiketrains


def align_times(times, events, window=[0, 1]):
    """
    Aligns times to events.

    Parameters
    ----------
    times : np.array
        Spike times (in seconds).
    events : np.array
        Event times (in seconds).
    window : list
        Window around event (in seconds).

    Returns
    -------
    aligned_times : np.array
        Aligned spike times.
    """

    t = np.sort(times)
    aligned_times = np.array([])
    for i, e in enumerate(events):
        ts = t - e  # ts: t shifted
        tsc = ts[(ts >= window[0]) & (ts <= window[1])]  # tsc: ts clipped
        al_t = np.full((tsc.size, 2), i, dtype='float')
        al_t[:, 1] = tsc
        if len(aligned_times) == 0:
            aligned_times = al_t
        else:
            aligned_times = np.concatenate((aligned_times, al_t))

    return aligned_times


def align_time_event_array(times, events, window=[-1000, 1000]):
    '''
    Parameters:
        - times: numpy array of times
        - events: numpy array of events
        - window: numpy array of window size
    Returns:
        - aligned_times: numpy array of aligned times
            aligned_times[:,0]: event index
            aligned_times[:,1]: time relative to event
    '''

    aligned_times = np.full((times.shape[0], 2), np.nan, dtype='float')
    times = times.astype('float')
    window = np.array(window, dtype='float')
    return align_time_event_array_jit(aligned_times, times, events, window)


@njit
def align_time_event_array_jit(aligned_times, times, events, window):
    # for i, ev in enumerate(events):
    for i in range(len(events)):
        if not (np.isnan(events[i]) | isinf(events[i])):
            ts = times - events[i]  # t shifted
            mask = (ts >= window[0] / 1000) & (ts <= window[1] / 1000)
            aligned_times[mask, 0] = i
            aligned_times[mask, 1] = ts[mask]

    return aligned_times


def compute_ISI(times, bin_size: float = 1.0, max_time: float = 50.0):
    '''
    @param_times (numpy array):
        array of times
	@param bin_size (float):
		Size of bin for the histogram (in ms).
	@param max_time (float):
		Stop the ISI histogram at this value (in ms).

	@return ISI (np.ndarray[uint32]) [time]:
		ISI histogram of the unit.
	@return bins (np.ndarray) [time+1]:
		Bins of the histogram.
	"""
  '''
    bin_size_c = round(bin_size / 1000)
    max_time_c = round(max_time / 1000)
    n_bins = int(max_time_c / bin_size_c)

    bins = np.arange(0, max_time_c + bin_size_c, bin_size_c)
    ISI = np.zeros(n_bins, dtype='uint64')
    return _compute_ISI


def _compute_ISI():
    pass


@dataclass
class NeuralDataset:
    # dp = path to spikesorted wrap data in phy format
    dp: str
    datatype: str  # Either 'neuropixel' or 'warp'

    def load(self):

        assert self.datatype in ['neuropixel', 'warp'], 'Unknown datatype'

        self.dp = Path(self.dp)
        if self.datatype == 'neuropixel':
            phy_folder = self.dp
            self.reader = PhyNpxlIO(dirname=phy_folder)
        elif self.datatype == 'warp':
            phy_folder = self.dp / 'phy'
            self.reader = PhyWarpIO(dirname=phy_folder)
        block = self.reader.read()
        self.seg = block[0].segments[0]
        self.spikes_array = np.concatenate(
            (self.reader._spike_clusters, self.reader._spike_times / self.seg.annotations['sampling_frequency']),
            axis=1, dtype=object)
        # self.spikes_df = pd.DataFrame({'cluster': self.reader._spike_clusters, 'spike_time': self.reader._spike_times/self.seg.annotations['sampling_frequency']}, index=np.arange(self.reader._spike_times.size))

        self.df_bhv = self.reader.load_bhv_data(self.seg.annotations['bhv_file'])
        self.quality_metrics = pd.DataFrame(self.reader._quality_metrics)
        self.quality_metrics.cluster_id = self.quality_metrics.cluster_id.astype('int')
        if self.datatype == 'warp':
            self.quality_metrics.cluster_id = self.quality_metrics.cluster_id.astype('int') - 1


def spikes_around_events(spikes_array, eventname, df_bhv, window=[-1000, 2000]):
    events = df_bhv[eventname].to_numpy(dtype='float')
    events_array = align_time_event_array(spikes_array[:, 1], events, window=[window[0] - 500, window[1] + 500])

    return events_array


def create_event_array(spikes_array, listofevnames, df_bhv, window=[-1000, 2000]):
    '''
    Parameters:
        - spikes_array: array of spikes, with columns: cluster, time, event, relative_spike_time
        - listofevnames: list of event names to be included in the array
        - df_bhv: dataframe with bhv data
        - window: window around event to be included in the array
            window can be a list of of list (with one for each event type), or one single window for all event types
    Return:
        - dataframe with columns: cluster, time, (event number, relative_spike_time to the event) (for each event type)
        '''
    if len(np.shape(window)) > 1:
        multiple_windows = True
        assert np.shape(window)[0] == len(listofevnames), 'window must have the same length as listofevnames'
    else:
        multiple_windows = False

    events_array = []
    colnames = ['cluster', 'time']
    for i, evname in enumerate(listofevnames):
        ev_array = spikes_around_events(spikes_array, evname,
                                        df_bhv, window=window[i] if multiple_windows else window)
        if len(events_array) == 0:
            events_array = ev_array
        else:
            events_array = np.concatenate((events_array, ev_array), axis=1)
        colnames.append(f'{evname}_num')
        colnames.append(f'{evname}_spike_times')

    spikes_array = np.concatenate((spikes_array, events_array), axis=1)
    df_spikes = pd.DataFrame(spikes_array, columns=colnames)
    # for evname in listofevnames:
    #     df_spikes[f'{evname}_spike_times'] = df_spikes[f'{evname}_spike_times'].apply(pd.to_numeric)

    return df_spikes


def generate_warp32_probe(radius=6):
    probe = generate_multi_columns_probe(num_columns=8,
                                         num_contact_per_column=4,
                                         xpitch=350, ypitch=350,
                                         contact_shapes='circle',
                                         contact_shape_params={'radius': radius})
    probe.create_auto_shape('rect')

    channel_indices = np.array([29, 31, 13, 15,
                                25, 27, 9, 11,
                                30, 32, 14, 16,
                                26, 28, 10, 12,
                                24, 22, 8, 6,
                                20, 18, 4, 2,
                                23, 21, 7, 5,
                                19, 17, 3, 1])

    probe.set_device_channel_indices(channel_indices - 1)

    return probe


def get_absolute_word_timing(df_bhv, word_token, passive=False):
    """
    Get the absolute word timing of a behavior dataframe.
    """
    fs = 24414.0625
    dtype = [('start', 'float64'), ('end', 'float64'), ('rel_start', 'float64')]
    word_timings = np.zeros(len(df_bhv), dtype=dtype)
    word_timings[:] = np.nan
    for index, trial in df_bhv.iterrows():
        if word_token in trial.distractors:
            wordposition = np.where(trial['distractors'] == word_token)[0]
            dDurs = trial.dDurs
            wordtiming = np.array([np.sum(dDurs[:wordposition[0]]), np.sum(dDurs[:wordposition[0] + 1])])
            wordtiming = wordtiming / fs
            if wordtiming[0] < trial.centreRelease or trial.centreRelease == 0:
                if passive == True:
                    word_timings['start'][index] = wordtiming[0] + trial.startSoundTime
                    word_timings['end'][index] = wordtiming[1] + trial.startSoundTime
                else:
                    word_timings['start'][index] = wordtiming[0] + trial.startTrialLick
                    word_timings['end'][index] = wordtiming[1] + trial.startTrialLick
                word_timings['rel_start'][index] = wordtiming[0]

    return word_timings


def get_absolute_trial_times(df_bhv):
    """
    Get the absolute TRIAL timing of a behavior dataframe.
    """
    fs = 24414.0625
    dtype = [('start', 'float64'), ('end', 'float64'), ('rel_start', 'float64')]
    word_timings = np.zeros(len(df_bhv), dtype=dtype)
    word_timings[:] = np.nan
    for index, trial in df_bhv.iterrows():
        word_timings['start'][index] = trial.startTrialLick
        word_timings['end'][index] = trial.startTrialLick + 3
        word_timings['rel_start'][index] = 0.00

    return word_timings


# def return_word_aligned_array(spiketrain, df_bhv, word_token):
#     wordtimings = get_absolute_word_timing(df_bhv, word_token)
#     df_bhv = df_bhv.assign(probestarts=wordtimings['start'], probeends=wordtimings['end'],
#                            rel_probestarts=wordtimings['rel_start'])
#
#     dtype = [('trial_num', 'float64'), ('spike_time', 'float64'),
#              ('start', 'float64'), ('end', 'float64'),
#              ('relStart', 'float64'), ('talker', 'float64')]
#
#     al_t = align_times(spiketrain.times.magnitude, df_bhv.probestarts, [0 , 1])
#     al_t = np.concatenate((al_t, np.zeros((al_t.shape[0], 4))), axis=1)
#     al_t.dtype = dtype
#     al_t['start'] = df_bhv.probestarts.to_numpy()[al_t['trial_num'].astype(int)]
#     al_t['end'] = df_bhv.probeends.to_numpy()[al_t['trial_num'].astype(int)]
#     al_t['relStart'] = df_bhv.rel_probestarts.to_numpy()[al_t['trial_num'].astype(int)]
#     al_t['talker'] = df_bhv.talker.to_numpy()[al_t['trial_num'].astype(int)]
#     return al_t
def return_word_aligned_array(spiketrain,
                              df_bhv,
                              word_token,
                              before_release_only=False
                              ):
    wordtimings = get_absolute_word_timing(df_bhv, word_token)
    df_bhv = df_bhv.assign(probestarts=wordtimings['start'], probeends=wordtimings['end'],
                           rel_probestarts=wordtimings['rel_start'])
    # 258.6533
    # 258.6789

    if before_release_only:

        df_bhv = df_bhv.loc[df_bhv.absoluteRealLickRelease > df_bhv.probeends]

        if len(df_bhv) == 0:
            return None, None

    dtype = [('trial_num', 'float64'), ('spike_time', 'float64'),
             ('start', 'float64'), ('end', 'float64'),
             ('relStart', 'float64'), ('talker', 'float64')]
    # 6468 4287
    al_t = align_times(spiketrain.times.magnitude, df_bhv.probestarts, [-1, 2])
    al_t = np.concatenate((al_t, np.zeros((al_t.shape[0], 4))), axis=1)
    al_t.dtype = dtype
    al_t['start'] = df_bhv.probestarts.to_numpy()[al_t['trial_num'].astype(int)]
    al_t['end'] = df_bhv.probeends.to_numpy()[al_t['trial_num'].astype(int)]
    al_t['relStart'] = df_bhv.rel_probestarts.to_numpy()[al_t['trial_num'].astype(int)]
    al_t['talker'] = df_bhv.talker.to_numpy()[al_t['trial_num'].astype(int)]

    return al_t

def return_before_word_aligned_array(spiketrain,
                              df_bhv,
                              word_token,
                              before_release_only=False
                              ):
    wordtimings = get_absolute_word_timing(df_bhv, word_token)
    df_bhv = df_bhv.assign(probestarts=wordtimings['start'], probeends=wordtimings['end'],
                           rel_probestarts=wordtimings['rel_start'])
    # 258.6533
    # 258.6789

    if before_release_only:

        df_bhv = df_bhv.loc[df_bhv.absoluteRealLickRelease > df_bhv.probeends]

        if len(df_bhv) == 0:
            return None, None

    dtype = [('trial_num', 'float64'), ('spike_time', 'float64'),
             ('start', 'float64'), ('end', 'float64'),
             ('relStart', 'float64'), ('talker', 'float64')]
    # 6468 4287
    al_t = align_times(spiketrain.times.magnitude, df_bhv.probestarts, [-1, 2])
    al_t = np.concatenate((al_t, np.zeros((al_t.shape[0], 4))), axis=1)
    al_t.dtype = dtype
    al_t['start'] = df_bhv.probestarts.to_numpy()[al_t['trial_num'].astype(int)]
    al_t['end'] = df_bhv.probeends.to_numpy()[al_t['trial_num'].astype(int)]
    al_t['relStart'] = df_bhv.rel_probestarts.to_numpy()[al_t['trial_num'].astype(int)]
    al_t['talker'] = df_bhv.talker.to_numpy()[al_t['trial_num'].astype(int)]

    return al_t


def return_soundonset_array(spiketrain,
                            df_bhv,
                            before_release_only=False):
    wordtimings = get_absolute_trial_times(df_bhv)
    df_bhv = df_bhv.assign(probestarts=wordtimings['start'], probeends=wordtimings['end'],
                           rel_probestarts=wordtimings['rel_start'])

    if before_release_only:

        df_bhv = df_bhv.loc[df_bhv.absoluteRealLickRelease > df_bhv.probeends]

        if len(df_bhv) == 0:
            return None, None

    dtype = [('trial_num', 'float64'), ('spike_time', 'float64'),
             ('start', 'float64'), ('end', 'float64'),
             ('relStart', 'float64'), ('talker', 'float64')]
    # 6468 4287
    al_t = align_times(spiketrain.times.magnitude, df_bhv.probestarts, [-1, 2])
    al_t = np.concatenate((al_t, np.zeros((al_t.shape[0], 4))), axis=1)
    al_t.dtype = dtype
    al_t['start'] = df_bhv.probestarts.to_numpy()[al_t['trial_num'].astype(int)]
    al_t['end'] = df_bhv.probeends.to_numpy()[al_t['trial_num'].astype(int)]
    al_t['relStart'] = df_bhv.rel_probestarts.to_numpy()[al_t['trial_num'].astype(int)]
    al_t['talker'] = df_bhv.talker.to_numpy()[al_t['trial_num'].astype(int)]

    return al_t


def return_soundonset_fra_array(spiketrain,
                                df_bhv,
                                before_release_only=False):
    wordtimings = get_absolute_trial_times(df_bhv)
    df_bhv = df_bhv.assign(probestarts=wordtimings['start'], probeends=wordtimings['end'],
                           rel_probestarts=wordtimings['rel_start'])

    if before_release_only:

        df_bhv = df_bhv.loc[df_bhv.absoluteRealLickRelease + 0.2 > df_bhv.probeends]

        if len(df_bhv) == 0:
            return None, None

    dtype = [('trial_num', 'float64'), ('spike_time', 'float64'),
             ('start', 'float64'), ('end', 'float64'),
             ('relStart', 'float64'), ('talker', 'float64')]
    # 6468 4287
    al_t = align_times(spiketrain.times.magnitude, df_bhv.probestarts, [-1, 2])
    al_t = np.concatenate((al_t, np.zeros((al_t.shape[0], 4))), axis=1)
    al_t.dtype = dtype
    al_t['start'] = df_bhv.probestarts.to_numpy()[al_t['trial_num'].astype(int)]
    al_t['end'] = df_bhv.probeends.to_numpy()[al_t['trial_num'].astype(int)]
    al_t['relStart'] = df_bhv.rel_probestarts.to_numpy()[al_t['trial_num'].astype(int)]
    # al_t['talker'] = df_bhv.talker.to_numpy()[al_t['trial_num'].astype(int)]

    return al_t


def get_neural_pop(blocks, clust_ids,
                   noise=True,
                   talker=1,
                   epochs=['Early', 'Late'],
                   binsize=0.1,
                   window=[-0.3, 1],
                   epoch_threshold=1.5):
    # If noise: return neural pop for trial with noise
    # If not noise: return neural pop for trial without noise

    neuralPop = np.array([])
    # i=0
    # clust_id = clust_ids[0]

    for i, clust_id in enumerate(clust_ids):
        n_trials = {'Early': 0, 'Late': 0}
        unit_aligned_time = np.array([])

        for s, seg in enumerate(blocks[0].segments):
            if seg.df_bhv is None:
                continue

            if noise:
                df_bhv = seg.df_bhv.loc[seg.df_bhv.currNoiseAtten <= 0]
            else:
                df_bhv = seg.df_bhv.loc[seg.df_bhv.currNoiseAtten > 60]

            df_bhv = df_bhv.reset_index(drop=True)
            if len(df_bhv) == 0:
                continue

            unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == clust_id][0]

            if statistics.mean_firing_rate(unit) < 0.5:
                continue

            if len(unit_aligned_time) == 0:
                unit_aligned_time = return_word_aligned_array(unit, df_bhv, df_bhv.probes[0])[0]
            else:
                seg_aligned_word = return_word_aligned_array(unit, df_bhv, df_bhv.probes[0])[0]
                seg_aligned_word['trial_num'] = seg_aligned_word['trial_num'] + np.max(unit_aligned_time['trial_num'])
                unit_aligned_time = np.concatenate((unit_aligned_time, seg_aligned_word))

        if len(neuralPop) == 0:
            len_probe = unit_aligned_time['end'][unit_aligned_time['talker'] == talker][0] - \
                        unit_aligned_time['start'][unit_aligned_time['talker'] == talker][0]

            # bins=np.arange(0,len_probe,binsize)
            bins = np.arange(window[0], window[1], binsize)
            # epoch * unit * bin
            neuralPop = np.zeros((len(epochs), len(clust_ids), len(bins) - 1))

        early_probe = unit_aligned_time[unit_aligned_time['relStart'] < epoch_threshold]
        late_probe = unit_aligned_time[unit_aligned_time['relStart'] > epoch_threshold]

        for epoch, ep_aligned_times in enumerate([early_probe, late_probe]):
            psth, edges = np.histogram(ep_aligned_times['spike_time'][ep_aligned_times['talker'] == talker],
                                       bins=bins)  # =np.arange(0,len_probe,binsize))
            psth_fr = (psth / len(
                np.unique(unit_aligned_time['trial_num'][unit_aligned_time['talker'] == talker]))) / binsize
            neuralPop[epoch, i, :] = psth_fr

            n_trials[epochs[epoch]] = len(
                np.unique(ep_aligned_times['trial_num'][ep_aligned_times['talker'] == talker]))

        # print(f'cluster {i}: n_trial: {n_trials}')

    return neuralPop, bins


def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10 ** precision), 10 ** precision)


def get_word_aligned_raster_zola_cruella2(blocks, clust_id, word=1, correctresp=True, pitchshift=True, df_filter=[]):
    unit_aligned_time = np.array([])
    for s, seg in enumerate(blocks[0].segments):
        if seg.df_bhv is None:
            continue
        if 'PitchShiftMat' in seg.df_bhv:
            print('This is an intra-trial pitch shift level, skipping...')
            print(f"Block name: {seg.df_bhv['recBlock']}")
            continue

        pitchshiftlist = np.array([])
        droplist = np.array([])
        for k in range(0, len(seg.df_bhv)):
            # if talker is not 1 then it is pitch shifted
            if seg.df_bhv.talker.values[k] == 1 or seg.df_bhv.talker.values[k] == 2:
                # if talker is only 1 or 2, it's not going to be pitch shifted
                pitchshiftlist = np.append(pitchshiftlist, 0)
            else:
                pitchshiftlist = np.append(pitchshiftlist, 1)

        seg.df_bhv = seg.df_bhv.drop(droplist)
        seg.df_bhv['pitchshift'] = pitchshiftlist

        if word is None:
            word = seg.df_bhv.probes[0]
        df_bhv = seg.df_bhv

        if pitchshift:
            df_bhv = df_bhv.loc[df_bhv.pitchshift == 1]
        else:
            df_bhv = df_bhv.loc[df_bhv.pitchshift == 0]

        df_bhv['targTimes'] = df_bhv['timeToTarget'] / 24414.0625

        df_bhv['centreRelease'] = df_bhv['lickRelease'] - df_bhv['startTrialLick']
        df_bhv['relReleaseTimes'] = df_bhv['centreRelease'] - df_bhv['targTimes']
        df_bhv['realRelReleaseTimes'] = df_bhv['relReleaseTimes'] - df_bhv['absentTime']

        if correctresp:
            df_bhv = df_bhv.loc[df_bhv['realRelReleaseTimes'].between(0, 2.3, inclusive=True)]

        if len(df_filter) > 0:
            for f in df_filter:
                df_bhv = apply_filter(df_bhv, f)

        df_bhv = df_bhv.reset_index(drop=True)

        if len(df_bhv) == 0:
            continue

        unit = [st for st in seg.spiketrains if st.annotations.get('cluster_id') == clust_id][0]
        unitcopy = copy.deepcopy(unit)
        # unit = [st for st in seg.spiketrains if st.annotations['cluster_id'].get() == clust_id][0]
        for st in seg.spiketrains:
            print(f"st.annotations['cluster_id'] = {st.annotations['cluster_id']}, clust_id = {clust_id}")
            if st.annotations['cluster_id'] == clust_id:
                print("Matching cluster_id found!")
                print(st.annotations['cluster_id'])
                print('original cluster_id:' + str(clust_id))

        if len(unit_aligned_time) == 0:
            unit_aligned_time = return_word_aligned_array(unit, df_bhv, word)
        else:
            try:
                seg_aligned_word = return_word_aligned_array(unit, df_bhv, word)
                seg_aligned_word['trial_num'] = seg_aligned_word['trial_num'] + np.max(unit_aligned_time['trial_num'])
                unit_aligned_time = np.concatenate((unit_aligned_time, seg_aligned_word))
            except:
                print("No instance of the word that the unit fired to found in this behavioral data file")
                continue

        # plot the distributions of unit aligned times

    return unit_aligned_time, unitcopy


def get_word_aligned_raster_ore(blocks, clust_id, word=None, pitchshift=True, correctresp=True, df_filter=[],
                                talker='female'):
    unit_aligned_time = np.array([])
    unit_aligned_time_compare = np.array([])
    for s, seg in enumerate(blocks[0].segments):
        # print(f"Processing segment {s}...")

        # check if pitchshiftmat is in annotations
        if seg.annotations['bhv_data'] is None:
            try:
                print('Loading bhv data from annotations...')
                seg.df_bhv = seg.load_bhv_data(seg.annotations['bhv_file'])
            except:
                print('Possibly corrupted or non-existent data file at iteration...' + str(s))
                continue

        if 'PitchShiftMat' in seg.annotations['bhv_data'] or 'level41' in seg.annotations['bhv_data']['fName'][
            0] or 'level48' in seg.annotations['bhv_data']['fName'][0]:
            print('This is an intra-trial pitch shift level, skipping iteration..' + str(s))
            # print(f"Block name: {seg.df_bhv['recBlock']}, iteration:" + str(s))
            continue
        df_length = pd.DataFrame(seg.annotations['bhv_data'])

        pitchshiftlist = []
        for k in range(0, len(df_length)):
            # if talker is not 1 then it is pitch shifted
            if seg.annotations['bhv_data']['talker'][k] == 1 or seg.annotations['bhv_data']['talker'][k] == 2:
                # if talker is only 1 or 2, it's not going to be pitch shifted
                pitchshiftlist.append(0)
            else:
                pitchshiftlist.append(1)

        seg.annotations['bhv_data']['pitchshift'] = pitchshiftlist

        # if word is None:
        #     word = seg.annotations['bhv_data'].probes[0]
        df_bhv = seg.annotations['bhv_data']
        df_bhv = pd.DataFrame(df_bhv)

        if pitchshift:
            # df_bhv = df_bhv[df_bhv['pitchshift'].apply(lambda x: (x == 1).all())]
            df_bhv = df_bhv[(df_bhv['pitchshift'] == 1)]
        else:
            df_bhv = df_bhv[(df_bhv['pitchshift'] == 0)]

        if talker == 'female':
            # any talker that is not 2, 13 or 8
            df_bhv = df_bhv[(df_bhv['talker'] == 1) | (df_bhv['talker'] == 3) | (df_bhv['talker'] == 5)]
        else:
            df_bhv = df_bhv[(df_bhv['talker'] == 2) | (df_bhv['talker'] == 13) | (df_bhv['talker'] == 8)]
        df_bhv['targTimes'] = df_bhv['timeToTarget'] / 24414.0625

        df_bhv['centreRelease'] = df_bhv['lickRelease'] - df_bhv['startTrialLick']
        df_bhv['relReleaseTimes'] = df_bhv['centreRelease'] - df_bhv['targTimes']
        df_bhv['realRelReleaseTimes'] = df_bhv['relReleaseTimes'] - df_bhv['absentTime']

        if correctresp:
            df_bhv = df_bhv.loc[df_bhv['realRelReleaseTimes'].between(0, 2.3, inclusive=True)]

        if len(df_filter) > 0:
            for f in df_filter:
                df_bhv = apply_filter(df_bhv, f)

        df_bhv = df_bhv.reset_index(drop=True)
        if len(df_bhv) == 0:
            print('no applicable trials found for segment:' + str(s))
            continue

        unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == clust_id][0]

        clust_id_first_digit = int(str(clust_id)[0])
        if (clust_id_first_digit == 1 or clust_id_first_digit == 2 or clust_id_first_digit == 3) and len(
                str(clust_id)) > 2:
            clust_id_compare = int(str(clust_id)[1:])
        else:
            clust_id_compare = clust_id

        unit_compare = [st for st in seg.spiketrains if st.annotations['cluster_id'] == clust_id_compare][0]

        # compare the raw spike times of unit and unit_compare
        if len(unit_aligned_time) == 0:
            unit_aligned_time = return_word_aligned_array(unit, df_bhv, word)
        else:
            seg_aligned_word = return_word_aligned_array(unit, df_bhv, word)
            seg_aligned_word['trial_num'] = seg_aligned_word['trial_num'] + np.max(unit_aligned_time['trial_num'])
            unit_aligned_time = np.concatenate((unit_aligned_time, seg_aligned_word))
            # print("no instance of word found in this behavioural data file ")

        if len(unit_aligned_time_compare) == 0:
            unit_aligned_time_compare = return_word_aligned_array(unit_compare, df_bhv, word)
        else:
            seg_aligned_word_compare = return_word_aligned_array(unit_compare, df_bhv, word)
            seg_aligned_word_compare['trial_num'] = seg_aligned_word_compare['trial_num'] + np.max(
                unit_aligned_time_compare['trial_num'])
            unit_aligned_time_compare = np.concatenate((unit_aligned_time_compare, seg_aligned_word_compare))
            # print("no instance of word found in this behavioural data file ")
            # if np.array_equal(seg_aligned_word_compare, seg_aligned_word):
            #     print('same seg extracted at iteration:'+ str(s))

        # if np.array_equal(unit_aligned_time_compare, unit_aligned_time):
        #     print('same times extracted at iteration:' + str(s))

    return unit_aligned_time, unit_aligned_time_compare


def get_word_aligned_raster_zola_cruella(blocks, clust_id, word=None, pitchshift=True, correctresp=True, df_filter=[],
                                         talker='female'):
    unit_aligned_time = np.array([])
    unit_aligned_time_compare = np.array([])
    for s, seg in enumerate(blocks[0].segments):
        # print(f"Processing segment {s}...")

        # check if pitchshiftmat is in annotations
        if seg.df_bhv is None:
            try:
                print('Loading bhv data from annotations...')
                seg.df_bhv = seg.load_bhv_data(seg.annotations['bhv_file'])
            except:
                print('Possibly corrupted or non-existent data file at iteration...' + str(s))
                continue

        if 'PitchShiftMat' in seg.df_bhv:
            print('This is an intra-trial pitch shift level, skipping iteration..' + str(s))
            # print(f"Block name: {seg.df_bhv['recBlock']}, iteration:" + str(s))
            continue

        pitchshiftlist = np.array([])
        for k in range(0, len(seg.df_bhv)):
            # if talker is not 1 then it is pitch shifted
            if seg.df_bhv.talker.values[k] == 1 or seg.df_bhv.talker.values[k] == 2:
                # if talker is only 1 or 2, it's not going to be pitch shifted
                pitchshiftlist = np.append(pitchshiftlist, 0)
            else:
                pitchshiftlist = np.append(pitchshiftlist, 1)

        seg.df_bhv['pitchshift'] = pitchshiftlist

        if word is None:
            word = seg.df_bhv.probes[0]
        df_bhv = seg.df_bhv

        if pitchshift:
            df_bhv = df_bhv.loc[df_bhv.pitchshift == 1]
        else:
            df_bhv = df_bhv.loc[df_bhv.pitchshift == 0]

        if talker == 'female':
            # any talker that is not 2, 13 or 8
            df_bhv = df_bhv[(df_bhv['talker'] == 1) | (df_bhv['talker'] == 3) | (df_bhv['talker'] == 5)]
        else:
            df_bhv = df_bhv[(df_bhv['talker'] == 2) | (df_bhv['talker'] == 13) | (df_bhv['talker'] == 8)]

        df_bhv['targTimes'] = df_bhv['timeToTarget'] / 24414.0625

        df_bhv['centreRelease'] = df_bhv['lickRelease'] - df_bhv['startTrialLick']
        df_bhv['relReleaseTimes'] = df_bhv['centreRelease'] - df_bhv['targTimes']
        df_bhv['realRelReleaseTimes'] = df_bhv['relReleaseTimes'] - df_bhv['absentTime']

        if correctresp:
            df_bhv = df_bhv.loc[df_bhv['realRelReleaseTimes'].between(0, 2.3, inclusive=True)]

        if len(df_filter) > 0:
            for f in df_filter:
                df_bhv = apply_filter(df_bhv, f)
        df_bhv = df_bhv.reset_index(drop=True)
        if len(df_bhv) == 0:
            print('no applicable trials found for segment:' + str(s))
            continue
        #print the cluster ids in seg
        # for st in seg.spiketrains:
        #     print(f"st.annotations['cluster_id'] = {st.annotations['cluster_id']}, clust_id = {clust_id}")
            # if st.annotations['cluster_id'] == clust_id:
            #     print("Matching cluster_id found!")
            #     print(st.annotations['cluster_id'])
            #     print('original cluster_id:' + str(clust_id))

        unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == clust_id][0]

        clust_id_first_digit = int(str(clust_id)[0])
        if (clust_id_first_digit == 1 or clust_id_first_digit == 2 or clust_id_first_digit == 3) and len(
                str(clust_id)) > 2:
            clust_id_compare = int(str(clust_id)[1:])
        else:
            clust_id_compare = clust_id

        unit_compare = [st for st in seg.spiketrains if st.annotations['cluster_id'] == clust_id_compare][0]

        # compare the raw spike times of unit and unit_compare
        if len(unit_aligned_time) == 0:
            unit_aligned_time = return_word_aligned_array(unit, df_bhv, word)
        else:
            seg_aligned_word = return_word_aligned_array(unit, df_bhv, word)
            seg_aligned_word['trial_num'] = seg_aligned_word['trial_num'] + np.max(unit_aligned_time['trial_num'])
            unit_aligned_time = np.concatenate((unit_aligned_time, seg_aligned_word))
            # print("no instance of word found in this behavioural data file ")

        if len(unit_aligned_time_compare) == 0:
            unit_aligned_time_compare = return_word_aligned_array(unit_compare, df_bhv, word)
        else:
            seg_aligned_word_compare = return_word_aligned_array(unit_compare, df_bhv, word)
            seg_aligned_word_compare['trial_num'] = seg_aligned_word_compare['trial_num'] + np.max(
                unit_aligned_time_compare['trial_num'])
            unit_aligned_time_compare = np.concatenate((unit_aligned_time_compare, seg_aligned_word_compare))
            # print("no instance of word found in this behavioural data file ")
            # if np.array_equal(seg_aligned_word_compare, seg_aligned_word):
            #     print('same seg extracted at iteration:'+ str(s))

        # if np.array_equal(unit_aligned_time_compare, unit_aligned_time):
        #     print('same times extracted at iteration:' + str(s))

    return unit_aligned_time, unit_aligned_time_compare



def get_before_word_raster_zola_cruella(blocks, clust_id, word=None, corresp_hit=True, df_filter=[]):
    unit_aligned_time = np.array([])
    unit_aligned_time_compare = np.array([])
    for s, seg in enumerate(blocks[0].segments):
        # print(f"Processing segment {s}...")

        # check if pitchshiftmat is in annotations
        if seg.df_bhv is None:
            try:
                print('Loading bhv data from annotations...')
                seg.df_bhv = seg.load_bhv_data(seg.annotations['bhv_file'])
            except:
                print('Possibly corrupted or non-existent data file at iteration...' + str(s))
                continue

        if 'PitchShiftMat' in seg.df_bhv:
            print('This is an intra-trial pitch shift level, skipping iteration..' + str(s))
            # print(f"Block name: {seg.df_bhv['recBlock']}, iteration:" + str(s))
            continue

        pitchshiftlist = np.array([])
        for k in range(0, len(seg.df_bhv)):
            # if talker is not 1 then it is pitch shifted
            if seg.df_bhv.talker.values[k] == 1 or seg.df_bhv.talker.values[k] == 2:
                # if talker is only 1 or 2, it's not going to be pitch shifted
                pitchshiftlist = np.append(pitchshiftlist, 0)
            else:
                pitchshiftlist = np.append(pitchshiftlist, 1)

        seg.df_bhv['pitchshift'] = pitchshiftlist

        if word is None:
            word = seg.df_bhv.probes[0]
        df_bhv = seg.df_bhv

        # if pitchshift:
        #     df_bhv = df_bhv.loc[df_bhv.pitchshift == 1]
        # else:
        #     df_bhv = df_bhv.loc[df_bhv.pitchshift == 0]

        # if talker == 'female':
        #     # any talker that is not 2, 13 or 8
        #     df_bhv = df_bhv[(df_bhv['talker'] == 1) | (df_bhv['talker'] == 3) | (df_bhv['talker'] == 5)]
        # else:
        #     df_bhv = df_bhv[(df_bhv['talker'] == 2) | (df_bhv['talker'] == 13) | (df_bhv['talker'] == 8)]

        df_bhv['targTimes'] = df_bhv['timeToTarget'] / 24414.0625

        df_bhv['centreRelease'] = df_bhv['lickRelease'] - df_bhv['startTrialLick']
        df_bhv['relReleaseTimes'] = df_bhv['centreRelease'] - df_bhv['targTimes']
        df_bhv['realRelReleaseTimes'] = df_bhv['relReleaseTimes'] - df_bhv['absentTime']

        if corresp_hit:
            df_bhv = df_bhv.loc[df_bhv['realRelReleaseTimes'].between(0, 2.2, inclusive=True)]
        else:
            df_bhv = df_bhv.loc[df_bhv['realRelReleaseTimes'].between(-6, 0, inclusive=False)]

        if len(df_filter) > 0:
            for f in df_filter:
                df_bhv = apply_filter(df_bhv, f)
        df_bhv = df_bhv.reset_index(drop=True)
        if len(df_bhv) == 0:
            print('no applicable trials found for segment:' + str(s))
            continue

        unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == clust_id][0]
        # compare the raw spike times of unit and unit_compare
        if len(unit_aligned_time) == 0:
            unit_aligned_time = return_word_aligned_array(unit, df_bhv, word)
        else:
            seg_aligned_word = return_word_aligned_array(unit, df_bhv, word)
            seg_aligned_word['trial_num'] = seg_aligned_word['trial_num'] + np.max(unit_aligned_time['trial_num'])
            unit_aligned_time = np.concatenate((unit_aligned_time, seg_aligned_word))
            # print("no instance of word found in this behavioural data file ")



    return unit_aligned_time


# def get_word_aligned_raster_zola_cruella(blocks, clust_id, word=None, pitchshift=True, correctresp=True, df_filter=[]):
#     unit_aligned_time = np.array([])
#     for s, seg in enumerate(blocks[0].segments):
#         print(f"Processing segment {s}...")
#
#         # check if pitchshiftmat is in annotations
#         if seg.df_bhv is None:
#             try:
#                 print('Loading bhv data from annotations...')
#                 seg.df_bhv = seg.load_bhv_data(seg.annotations['bhv_file'])
#             except:
#                 print('Possibly corrupted or non-existent data file.')
#                 continue
#
#         if 'PitchShiftMat' in seg.df_bhv:
#             print('This is an intra-trial pitch shift level, skipping...')
#             print(f"Block name: {seg.df_bhv['recBlock']}")
#             continue
#
#         pitchshiftlist = np.array([])
#         droplist = np.array([])
#         for k in range(0, len(seg.df_bhv)):
#             # if talker is not 1 then it is pitch shifted
#             if seg.df_bhv.talker.values[k] == 1 or seg.df_bhv.talker.values[k] == 2:
#                 # if talker is only 1 or 2, it's not going to be pitch shifted
#                 pitchshiftlist = np.append(pitchshiftlist, 0)
#             else:
#                 pitchshiftlist = np.append(pitchshiftlist, 1)
#
#         seg.df_bhv = seg.df_bhv.drop(droplist)
#         seg.df_bhv['pitchshift'] = pitchshiftlist
#
#         if word is None:
#             word = seg.df_bhv.probes[0]
#         df_bhv = seg.df_bhv
#
#         if pitchshift:
#             df_bhv = df_bhv.loc[df_bhv.pitchshift == 1]
#         else:
#             df_bhv = df_bhv.loc[df_bhv.pitchshift == 0]
#
#         df_bhv['targTimes'] = df_bhv['timeToTarget'] / 24414.0625
#
#         df_bhv['centreRelease'] = df_bhv['lickRelease'] - df_bhv['startTrialLick']
#         df_bhv['relReleaseTimes'] = df_bhv['centreRelease'] - df_bhv['targTimes']
#         df_bhv['realRelReleaseTimes'] = df_bhv['relReleaseTimes'] - df_bhv['absentTime']
#
#         if correctresp:
#             df_bhv = df_bhv.loc[df_bhv['realRelReleaseTimes'].between(0, 2.3, inclusive=True)]
#
#         if len(df_filter) > 0:
#             for f in df_filter:
#                 df_bhv = apply_filter(df_bhv, f)
#
#         df_bhv = df_bhv.reset_index(drop=True)
#
#         if len(df_bhv) == 0:
#             continue
#
#         tolerance = 0.1  # Define a tolerance level
#
#         unit = None  # Initialize unit as None
#         for st in seg.spiketrains:
#             # Compare cluster IDs using tolerance
#             if abs(st.annotations['cluster_id'] - clust_id) < tolerance:
#                 unit = st
#                 break  # Stop searching once a matching spike train is found
#
#         if unit is None:
#             print(f"No matching spike train found for clust_id {clust_id}")
#             continue
#         unit_compare =  [st for st in seg.spiketrains if st.annotations['cluster_id'] == int(clust_id)][0]
#         unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == clust_id][0]
#
#
#
#         if len(unit_aligned_time) == 0:
#             unit_aligned_time = return_soundonset_array(unit, df_bhv)
#             unit_aligned_time_compare = return_soundonset_array(unit_compare, df_bhv)
#             #compare unit aligned time with unit aligned time of unit_compare
#
#             if np.all(unit_aligned_time_compare['spike_time'] == unit_aligned_time['spike_time']):
#                 print(f'same times extracted for seg:{seg.name}, {s}')
#
#         else:
#             try:
#                 seg_aligned_word = return_soundonset_array(unit, df_bhv)
#                 seg_aligned_word['trial_num'] = seg_aligned_word['trial_num'] + np.max(unit_aligned_time['trial_num'])
#                 unit_aligned_time = np.concatenate((unit_aligned_time, seg_aligned_word))
#
#                 seg_aligned_word_compare = return_soundonset_array(unit_compare, df_bhv)
#                 seg_aligned_word_compare['trial_num'] = seg_aligned_word_compare['trial_num'] + np.max(unit_aligned_time_compare['trial_num'])
#                 unit_aligned_time_compare = np.concatenate((unit_aligned_time_compare, seg_aligned_word_compare))
#
#                 if np.all(unit_aligned_time_compare['spike_time'] == unit_aligned_time['spike_time']):
#                     print(f'same times extracted for seg:{seg.name}, {s}')
#
#
#             except:
#                 print("No instance of the word that the unit fired to found in this behavioral data file")
#                 continue
#     if np.all(unit_aligned_time_compare['spike_time'] == unit_aligned_time['spike_time']):
#         print('same times extracted')
#
#
#     return unit_aligned_time


def get_soundonset_alignedraster_squinty(blocks, clust_id, df_filter=[], fra=True):
    unit_aligned_time = np.array([])
    for s, seg in enumerate(blocks[0].segments):
        if seg.df_bhv is None:
            continue

        pitchshiftlist = np.array([])
        droplist = np.array([])
        for k in range(0, len(seg.df_bhv)):
            # if talker is not 1 then it is pitch shifted
            if seg.df_bhv.talker.values[k] == 3:
                # if talker is only 1 or 2 not going to be pitch shifted for cruella and zola
                pitchshiftlist = np.append(pitchshiftlist, 0)
            else:
                pitchshiftlist = np.append(pitchshiftlist, 1)

        seg.df_bhv = seg.df_bhv.drop(droplist)
        seg.df_bhv['pitchshift'] = pitchshiftlist

        df_bhv = seg.df_bhv

        df_bhv['targTimes'] = df_bhv['timeToTarget'] / 24414.0625

        df_bhv['centreRelease'] = df_bhv['lickRelease'] - df_bhv['startTrialLick']
        df_bhv['relReleaseTimes'] = df_bhv['centreRelease'] - df_bhv['targTimes']
        df_bhv['realRelReleaseTimes'] = df_bhv['relReleaseTimes'] - df_bhv['absentTime']

        if len(df_filter) > 0:
            for f in df_filter:
                df_bhv = apply_filter(df_bhv, f)

        df_bhv = df_bhv.reset_index(drop=True)

        if len(df_bhv) == 0:
            continue

        unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == clust_id][0]

        if statistics.mean_firing_rate(unit) < 0.001:
            print('mean firing rate too low at ', statistics.mean_firing_rate(unit) + 'skipping unit')
            continue

        if len(unit_aligned_time) == 0:
            unit_aligned_time = return_soundonset_array(unit, df_bhv)
        else:
            try:
                seg_aligned_word = return_soundonset_array(unit, df_bhv)
                seg_aligned_word['trial_num'] = seg_aligned_word['trial_num'] + np.max(unit_aligned_time['trial_num'])
                unit_aligned_time = np.concatenate((unit_aligned_time, seg_aligned_word))
                print("no instance found in this behavioural data file ")
            except:
                continue

    return unit_aligned_time


def get_word_aligned_raster_squinty(blocks, clust_id, word=None, pitchshift=True, correctresp=True, df_filter=[]):
    unit_aligned_time = np.array([])
    for s, seg in enumerate(blocks[0].segments):
        if seg.df_bhv is None:
            try:
                print('loading bhv data from annotations')
                seg.df_bhv = seg.load_bhv_data(seg.annotations['bhv_file'])

            except:
                print('possibly corrupted data file')
                continue
        pitchshiftlist = np.array([])
        droplist = np.array([])
        for k in range(0, len(seg.df_bhv)):
            # if talker is not 1 then it is pitch shifted
            if seg.df_bhv.talker.values[k] == 3:
                # if talker is only 1 or 2 not going to be pitch shifted for cruella and zola
                pitchshiftlist = np.append(pitchshiftlist, 0)
            else:
                pitchshiftlist = np.append(pitchshiftlist, 1)

        seg.df_bhv = seg.df_bhv.drop(droplist)
        seg.df_bhv['pitchshift'] = pitchshiftlist

        if word == None:
            word = seg.df_bhv.probes[0]
        df_bhv = seg.df_bhv

        if pitchshift:
            df_bhv = df_bhv.loc[df_bhv.pitchshift == 1]
        else:
            df_bhv = df_bhv.loc[df_bhv.pitchshift == 0]
        df_bhv['targTimes'] = df_bhv['timeToTarget'] / 24414.0625

        df_bhv['centreRelease'] = df_bhv['lickRelease'] - df_bhv['startTrialLick']
        df_bhv['relReleaseTimes'] = df_bhv['centreRelease'] - df_bhv['targTimes']
        df_bhv['realRelReleaseTimes'] = df_bhv['relReleaseTimes'] - df_bhv['absentTime']
        if correctresp:
            df_bhv = df_bhv.loc[df_bhv['realRelReleaseTimes'].between(0, 2.3, inclusive=True)]

        if len(df_filter) > 0:
            for f in df_filter:
                df_bhv = apply_filter(df_bhv, f)

        df_bhv = df_bhv.reset_index(drop=True)

        if len(df_bhv) == 0:
            continue

        unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == clust_id][0]

        # if statistics.mean_firing_rate(unit) < 0.5:
        #     print('mean firing rate is less than 0.2 Hz at this unit' + str(clust_id)+ 'with a mean firing rate of '+ str(statistics.mean_firing_rate(unit)))
        #     continue

        if len(unit_aligned_time) == 0:
            unit_aligned_time = return_word_aligned_array(unit, df_bhv, word)
        else:
            try:
                seg_aligned_word = return_word_aligned_array(unit, df_bhv, word)
                seg_aligned_word['trial_num'] = seg_aligned_word['trial_num'] + np.max(unit_aligned_time['trial_num'])
                unit_aligned_time = np.concatenate((unit_aligned_time, seg_aligned_word))
            except:
                print("no instance of word that unit fired to found in this behavioural data file ")

                continue

    return unit_aligned_time


def get_word_aligned_raster(blocks, clust_id, word=None, pitchshift=True, correctresp=True, df_filter=[], talker=1):
    unit_aligned_time = np.array([])
    for s, seg in enumerate(blocks[0].segments):
        if seg.df_bhv is None:
            continue
        pitchshiftlist = np.array([])
        droplist = np.array([])
        for k in range(0, len(seg.df_bhv)):
            try:
                if hasattr(seg.df_bhv, 'PitchShiftMat') == True and all(
                        [v == 0 for v in seg.df_bhv.PitchShiftMat.values[k]]):
                    pitchshiftlist = np.append(pitchshiftlist, 0)
                # also if PitchShiftMat does not exist
                elif hasattr(seg.df_bhv, 'PitchShiftMat') == False:
                    pitchshiftlist = np.append(pitchshiftlist, 0)
                else:
                    pitchshiftlist = np.append(pitchshiftlist, 1)
            except:
                droplist = np.append(droplist, k)
                # indexdrop = seg.df_bhv.iloc[droplist].columns.get_loc('PitchShiftMat')
                continue
        seg.df_bhv = seg.df_bhv.drop(droplist)
        seg.df_bhv['pitchshift'] = pitchshiftlist

        if word == None:
            word = seg.df_bhv.probes[0]
        df_bhv = seg.df_bhv

        # if attenuation:
        #     df_bhv = seg.df_bhv.loc[seg.df_bhv.currAtten == 0]
        # else:
        #     df_bhv = seg.df_bhv.loc[seg.df_bhv.currAtten > 60]

        # if len(droplist)>0:
        #     #droplist = [int(x) for x in droplist]  # drop corrupted metdata trials
        #     #indexdrop = seg.df_bhv.iloc[droplist].name
        #     df_bhvtest= seg.df_bhv
        #     indexdrop = df_bhv.iloc[droplist].name

        # df_bhv.drop(droplist)

        if pitchshift:
            df_bhv = df_bhv.loc[df_bhv.pitchshift == 1]
        else:
            df_bhv = df_bhv.loc[df_bhv.pitchshift == 0]
        df_bhv = df_bhv.loc[df_bhv.talker == talker]

        df_bhv['targTimes'] = df_bhv['timeToTarget'] / 24414.0625

        df_bhv['centreRelease'] = df_bhv['lickRelease'] - df_bhv['startTrialLick']
        df_bhv['relReleaseTimes'] = df_bhv['centreRelease'] - df_bhv['targTimes']
        df_bhv['realRelReleaseTimes'] = df_bhv['relReleaseTimes'] - df_bhv['absentTime']
        if correctresp:
            df_bhv = df_bhv.loc[df_bhv['realRelReleaseTimes'].between(0, 2.3, inclusive=True)]

        if len(df_filter) > 0:
            for f in df_filter:
                df_bhv = apply_filter(df_bhv, f)

        df_bhv = df_bhv.reset_index(drop=True)

        if len(df_bhv) == 0:
            continue

        unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == clust_id][0]

        # if statistics.mean_firing_rate(unit) < 0.5:
        #     print('mean firing rate is less than 0.2 Hz at this unit' + str(clust_id)+ 'with a mean firing rate of '+ str(statistics.mean_firing_rate(unit)))
        #     continue

        if len(unit_aligned_time) == 0:
            unit_aligned_time = return_word_aligned_array(unit, df_bhv, word)
        else:
            try:
                seg_aligned_word = return_word_aligned_array(unit, df_bhv, word)
                seg_aligned_word['trial_num'] = seg_aligned_word['trial_num'] + np.max(unit_aligned_time['trial_num'])
                unit_aligned_time = np.concatenate((unit_aligned_time, seg_aligned_word))
                print("no instance of word found in this behavioural data file ")
            except:
                continue
    time_window = 1.0

    # Calculate firing rates for each neuron
    # firing_rates = []
    # for neuron_spike_times in unit_aligned_time:
    #     num_spikes = len(neuron_spike_times)
    #     firing_rate = num_spikes / time_window
    #     firing_rates.append(firing_rate)

    # # Create a histogram of firing rates
    # plt.hist(firing_rates, bins=20, edgecolor='k')  # Adjust the number of bins as needed
    # plt.xlabel('Firing Rate (spikes per second)')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Firing Rates')
    # plt.grid(True)
    # plt.show()

    return unit_aligned_time


def get_soundonset_alignedraster(blocks, clust_id, df_filter=[]):
    unit_aligned_time = np.array([])
    unit_list = np.array([])
    for s, seg in enumerate(blocks[0].segments):
        if seg.df_bhv is None:
            continue

        df_bhv = seg.df_bhv

        df_bhv['targTimes'] = df_bhv['timeToTarget'] / 24414.0625

        df_bhv['centreRelease'] = df_bhv['lickRelease'] - df_bhv['startTrialLick']
        df_bhv['relReleaseTimes'] = df_bhv['centreRelease'] - df_bhv['targTimes']
        df_bhv['realRelReleaseTimes'] = df_bhv['relReleaseTimes'] - df_bhv['absentTime']

        if len(df_filter) > 0:
            for f in df_filter:
                df_bhv = apply_filter(df_bhv, f)

        df_bhv = df_bhv.reset_index(drop=True)

        if len(df_bhv) == 0:
            continue
        print_cluster_ids(blocks)
        matching_spike_trains = [st for st in seg.spiketrains if st.annotations.get('cluster_id') == clust_id]

        unit2 = 0
        for st in seg.spiketrains:
            print(st)
            if st.annotations.get('cluster_id') == clust_id:
                unit2 = st
                print('chosen st:')
                print(st)

        unit = copy.deepcopy([st for st in seg.spiketrains if st.annotations['cluster_id'] == clust_id][0])

        for st in seg.spiketrains:
            print(f"st.annotations['cluster_id'] = {st.annotations['cluster_id']}, clust_id = {clust_id}")
            if st.annotations['cluster_id'] == clust_id:
                print("Matching cluster_id found!")
                print(st.annotations['cluster_id'])
                print('original cluster_id:' + str(clust_id))

        unit_compare = copy.deepcopy([st for st in seg.spiketrains if st.annotations['cluster_id'] == int(clust_id)][0])

        if len(unit_aligned_time) == 0:
            unit_aligned_time = return_soundonset_array(unit, df_bhv)
            unit_aligned_time_compare = return_soundonset_array(unit_compare, df_bhv)
        else:
            try:
                seg_aligned_word = return_soundonset_array(unit, df_bhv)
                seg_aligned_word_compare = return_soundonset_array(unit_compare, df_bhv)
                seg_aligned_word['trial_num'] = seg_aligned_word['trial_num'] + np.max(unit_aligned_time['trial_num'])
                seg_aligned_word_compare['trial_num'] = seg_aligned_word_compare['trial_num'] + np.max(
                    unit_aligned_time_compare['trial_num'])
                unit_aligned_time = np.concatenate((unit_aligned_time, seg_aligned_word))
                unit_aligned_time_compare = np.concatenate((unit_aligned_time_compare, seg_aligned_word_compare))
            except:
                continue
        # plot the distributions of unit aligned times
    if (unit_aligned_time_compare == unit_aligned_time).all():
        print('same times extracted')

    return unit_aligned_time, unit_aligned_time_compare


def split_cluster_base_on_segment_zola(blocks, clust_id, num_clusters=2):
    # Create an empty list to store mean firing rates for each segment
    segment_mean_firing_rates = []
    segment_name_list = []
    original_shape = blocks[0].segments[0].spiketrains[0].shape
    print("Original Shape:", original_shape)

    # Collect mean firing rates and segment names for segments with the original cluster ID
    for s, seg in enumerate(blocks[0].segments):
        unit = [st for st in seg.spiketrains if st.annotations.get('cluster_id') == clust_id]
        if unit:
            unit = unit[0]  # Use the first spike train found
            # Calculate and store the mean firing rate for the unit in this segment
            segment_mean_firing_rate = len(unit) / (unit.t_stop - unit.t_start)
            segment_mean_firing_rates.append(segment_mean_firing_rate)
            segment_name_list.append(seg.name)

    if not segment_mean_firing_rates:
        print(f"No spike trains with cluster_id {clust_id} found.")
        return blocks

    # Combine segment names and mean firing rates into a 2D array
    data = np.array([segment_mean_firing_rates]).T

    # Perform K-means clustering
    try:
        kmeans = KMeans(n_clusters=num_clusters, tol=0.001)
        kmeans.fit(data)
    except:
        print('Not enough clusters to split.')
        return blocks

    # Get cluster labels for each segment
    cluster_labels = kmeans.labels_

    # Count the occurrences of each cluster label
    cluster_counts = Counter(cluster_labels)

    # Find the majority cluster
    majority_cluster = cluster_counts.most_common(1)[0][0]
    # Get the other clusters
    minority_clusters = cluster_counts.most_common()[1:]
    # Get the first number in every tuple
    minority_clusters = [i[0] for i in minority_clusters]

    max_cluster_id = max(max(st.annotations['cluster_id'] for st in seg.spiketrains) for seg in blocks[0].segments)

    # Create a list to store segments in each cluster
    cluster_segments = [[] for _ in range(num_clusters)]

    new_unit_id_1 = np.int32(clust_id + 100)
    new_unit_id_2 = np.int32(clust_id + 200)
    print(f"Debug - Majority Cluster: {majority_cluster}")
    print(f"Debug - Minority Clusters: {minority_clusters}")
    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains]
    clust_ids_1 = [st.annotations['cluster_id'] for st in blocks[0].segments[1].spiketrains]
    print(f"Debug - Cluster IDs: {clust_ids}")
    for i in range(0, len(blocks[0].segments)):
        clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[i].spiketrains]
        print(f"Debug - Cluster IDs: {clust_ids} for iteration {i}")

    print_cluster_ids(blocks)
    test_list = blocks[0].segments
    for seg in blocks[0].segments:
        unit = [st for st in seg.spiketrains if st.annotations.get('cluster_id') == clust_id][0]
        cluster_label = cluster_labels[segment_name_list.index(seg.name)]
        print(f"Debug - Cluster Label for {seg.name}: {cluster_label}")  # Add this line
        # Inside the loop that splits clusters based on cluster_label
        # Inside the loop that splits clusters based on cluster_label
        # ...
        # Inside the loop that splits clusters based on cluster_label
        if cluster_label == minority_clusters[0]:
            if 'cluster_id' in unit.annotations and unit.annotations['cluster_id'] == clust_id:
                new_unit_2 = copy.deepcopy(unit)
                new_unit_2.annotations['cluster_id'] = new_unit_id_2
                new_unit_2.annotations['id'] = new_unit_id_2
                new_unit_2.annotations['si_unit_id'] = new_unit_id_2 + 1
                empty_spike_train_2 = SpikeTrain([], units=new_unit_2.units, t_start=new_unit_2.t_start,
                                                 t_stop=new_unit_2.t_stop, name=new_unit_id_2)
                empty_spike_train_2.annotate(**new_unit_2.annotations)
                seg.spiketrains.append(empty_spike_train_2)

                new_unit = copy.deepcopy(unit)
                new_unit.annotations['cluster_id'] = new_unit_id_1
                new_unit.annotations['id'] = new_unit_id_1
                new_unit.annotations['si_unit_id'] = new_unit_id_1 + 1
                not_empty_spike_train = SpikeTrain(new_unit.times, units=new_unit.units, t_start=new_unit.t_start,
                                                   t_stop=new_unit.t_stop, name=new_unit_id_1)
                not_empty_spike_train.annotate(**new_unit.annotations)

                seg.spiketrains.append(not_empty_spike_train)

        # Inside the loop for majority_cluster
        elif cluster_label == majority_cluster:
            if 'cluster_id' in unit.annotations and unit.annotations['cluster_id'] == clust_id:
                new_unit_2 = copy.deepcopy(unit)
                new_unit_2.annotations['cluster_id'] = new_unit_id_2
                new_unit_2.annotations['id'] = new_unit_id_2
                new_unit_2.annotations['si_unit_id'] = new_unit_id_2 + 1
                not_empty_spike_train = SpikeTrain(new_unit_2.times, units=new_unit_2.units, t_start=new_unit_2.t_start,
                                                   t_stop=new_unit_2.t_stop, name=new_unit_2)
                not_empty_spike_train.annotate(**new_unit_2.annotations)

                seg.spiketrains.append(not_empty_spike_train)

                new_unit_1 = copy.deepcopy(unit)
                new_unit_1.annotations['cluster_id'] = new_unit_id_1
                new_unit_1.annotations['id'] = new_unit_id_1
                new_unit_1.annotations['si_unit_id'] = new_unit_id_1 + 1
                empty_spike_train_1 = SpikeTrain([], units=new_unit_1.units, t_start=new_unit_1.t_start,
                                                 t_stop=new_unit_1.t_stop, name=new_unit_1)
                empty_spike_train_1.annotate(**new_unit_1.annotations)
                seg.spiketrains.append(empty_spike_train_1)
        # ...

        # put unit back into NEO object

        # cluster_segments[cluster_label].append(seg)
    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains]
    print(f"after loop, Debug - Cluster IDs: {clust_ids}")
    print_cluster_ids(blocks)

    print("Debugging Information:")
    print(f"Cluster Labels: {cluster_labels}")
    print(f"Segment Names: {segment_name_list}")
    print(f"Majority Cluster: {majority_cluster}")

    # Plot the mean firing rates for each segment
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(segment_name_list, segment_mean_firing_rates)
    ax.set_ylabel('Mean Firing Rate (Hz)')
    ax.set_title('Mean Firing Rate Across Segments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # for st in seg.spiketrains:
    #     print(st.annotations['cluster_id'])
    print_cluster_ids(blocks)

    modified_shape = blocks[0].segments[0].spiketrains[0].shape
    print("Modified Shape:", modified_shape)

    return blocks


def split_cluster_base_on_segment(blocks, clust_id, num_clusters=3):
    # Create an empty list to store mean firing rates for each segment
    segment_mean_firing_rates = []
    segment_name_list = []
    original_shape = blocks[0].segments[0].spiketrains[0].shape
    print("Original Shape:", original_shape)

    # Collect mean firing rates and segment names for segments with the original cluster ID
    for s, seg in enumerate(blocks[0].segments):
        unit = [st for st in seg.spiketrains if st.annotations.get('cluster_id') == clust_id]
        if unit:
            unit = unit[0]  # Use the first spike train found
            # Calculate and store the mean firing rate for the unit in this segment
            segment_mean_firing_rate = len(unit) / (unit.t_stop - unit.t_start)
            segment_mean_firing_rates.append(segment_mean_firing_rate)
            segment_name_list.append(seg.name)

    if not segment_mean_firing_rates:
        print(f"No spike trains with cluster_id {clust_id} found.")
        return blocks

    # Combine segment names and mean firing rates into a 2D array
    data = np.array([segment_mean_firing_rates]).T

    # Perform K-means clustering
    try:
        kmeans = KMeans(n_clusters=num_clusters, tol=0.001)
        kmeans.fit(data)
    except:
        print('Not enough clusters to split.')
        return blocks

    # Get cluster labels for each segment
    cluster_labels = kmeans.labels_

    # Count the occurrences of each cluster label
    cluster_counts = Counter(cluster_labels)

    # Find the majority cluster
    majority_cluster = cluster_counts.most_common(1)[0][0]
    # Get the other clusters
    minority_clusters = cluster_counts.most_common()[1:]
    # Get the first number in every tuple
    minority_clusters = [i[0] for i in minority_clusters]
    if len(minority_clusters) != 2:
        print('NOT EXactly 2 minority clusters found')
        return blocks
    max_cluster_id = max(max(st.annotations['cluster_id'] for st in seg.spiketrains) for seg in blocks[0].segments)

    # Create a list to store segments in each cluster
    cluster_segments = [[] for _ in range(num_clusters)]

    # new_unit_id = max_cluster_id + 1
    # new_unit_id_2 = new_unit_id + 1

    new_unit_id_1 = clust_id + 100
    new_unit_id_2 = clust_id + 200
    new_unit_id_3 = clust_id + 300
    print(f"Debug - Majority Cluster: {majority_cluster}")
    print(f"Debug - Minority Clusters: {minority_clusters}")
    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains]
    clust_ids_1 = [st.annotations['cluster_id'] for st in blocks[0].segments[1].spiketrains]
    print(f"Debug - Cluster IDs: {clust_ids}")
    for i in range(0, len(blocks[0].segments)):
        clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[i].spiketrains]
        print(f"Debug - Cluster IDs: {clust_ids} for iteration {i}")

    print_cluster_ids(blocks)
    test_list = blocks[0].segments
    for seg in blocks[0].segments:
        unit = [st for st in seg.spiketrains if st.annotations.get('cluster_id') == clust_id][0]
        cluster_label = cluster_labels[segment_name_list.index(seg.name)]
        print(f"Debug - Cluster Label for {seg.name}: {cluster_label}")  # Add this line
        if cluster_label == minority_clusters[0]:
            if 'cluster_id' in unit.annotations and unit.annotations['cluster_id'] == clust_id:
                # unit.annotations['cluster_id'] = new_unit_id
                # append a new unit to the segment
                new_unit_3 = copy.deepcopy(unit)
                new_unit_3.annotations['cluster_id'] = new_unit_id_3
                empty_spike_train_3 = SpikeTrain([], units=new_unit_3.units, t_start=new_unit_3.t_start,
                                                 t_stop=new_unit_3.t_stop)
                empty_spike_train_3.annotate(**new_unit_3.annotations)
                seg.spiketrains.append(empty_spike_train_3)

                new_unit_2 = copy.deepcopy(unit)
                new_unit_2.annotations['cluster_id'] = new_unit_id_2

                empty_spike_train_2 = SpikeTrain([], units=new_unit_2.units, t_start=new_unit_2.t_start,
                                                 t_stop=new_unit_2.t_stop)
                empty_spike_train_2.annotate(**new_unit_2.annotations)

                seg.spiketrains.append(empty_spike_train_2)

                new_unit = copy.deepcopy(unit)
                new_unit.annotations['cluster_id'] = new_unit_id_1
                seg.spiketrains.append(new_unit)

                # append a new unit to the block
                # new_unit = unit.copy()
                # new_unit.annotations['cluster_id'] = new_unit_id
                # blocks[0].segments[0].spiketrains.append(new_unit)
                # #append a new unit to the block
                # new_unit = unit.copy()
                # new_unit.annotations['cluster_id'] = new_unit_id
                # blocks[0].spiketrains.append(new_unit)

        elif cluster_label == minority_clusters[1]:
            if 'cluster_id' in unit.annotations and unit.annotations['cluster_id'] == clust_id:
                new_unit_3 = copy.deepcopy(unit)
                new_unit_3.annotations['cluster_id'] = new_unit_id_3
                empty_spike_train_3 = SpikeTrain([], units=new_unit_3.units, t_start=new_unit_3.t_start,
                                                 t_stop=new_unit_3.t_stop)
                empty_spike_train_3.annotate(**new_unit_3.annotations)
                seg.spiketrains.append(empty_spike_train_3)

                new_unit_2 = copy.deepcopy(unit)
                new_unit_2.annotations['cluster_id'] = new_unit_id_2
                seg.spiketrains.append(new_unit_2)

                new_unit_1 = copy.deepcopy(unit)
                new_unit_1.annotations['cluster_id'] = new_unit_id_1
                empty_spike_train_1 = SpikeTrain([], units=new_unit_1.units, t_start=new_unit_1.t_start,
                                                 t_stop=new_unit_1.t_stop)
                empty_spike_train_1.annotate(**new_unit_1.annotations)
                seg.spiketrains.append(empty_spike_train_1)



        elif cluster_label == majority_cluster:
            if 'cluster_id' in unit.annotations and unit.annotations['cluster_id'] == clust_id:
                new_unit_3 = copy.deepcopy(unit)
                new_unit_3.annotations['cluster_id'] = new_unit_id_3
                seg.spiketrains.append(new_unit_3)

                new_unit_2 = copy.deepcopy(unit)
                new_unit_2.annotations['cluster_id'] = new_unit_id_2

                empty_spike_train_2 = SpikeTrain([], units=new_unit_2.units, t_start=new_unit_2.t_start,
                                                 t_stop=new_unit_2.t_stop)
                empty_spike_train_2.annotate(**new_unit_2.annotations)

                seg.spiketrains.append(empty_spike_train_2)

                new_unit_1 = copy.deepcopy(unit)
                new_unit_1.annotations['cluster_id'] = new_unit_id_1
                empty_spike_train_1 = SpikeTrain([], units=new_unit_1.units, t_start=new_unit_1.t_start,
                                                 t_stop=new_unit_1.t_stop)
                empty_spike_train_1.annotate(**new_unit_1.annotations)
                seg.spiketrains.append(empty_spike_train_1)

        # put unit back into NEO object

        # cluster_segments[cluster_label].append(seg)
    clust_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains]
    print(f"after loop, Debug - Cluster IDs: {clust_ids}")
    print_cluster_ids(blocks)

    print("Debugging Information:")
    print(f"Cluster Labels: {cluster_labels}")
    print(f"Segment Names: {segment_name_list}")
    print(f"Majority Cluster: {majority_cluster}")

    # Plot the mean firing rates for each segment
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(segment_name_list, segment_mean_firing_rates)
    ax.set_ylabel('Mean Firing Rate (Hz)')
    ax.set_title('Mean Firing Rate Across Segments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # for st in seg.spiketrains:
    #     print(st.annotations['cluster_id'])
    print_cluster_ids(blocks)

    modified_shape = blocks[0].segments[0].spiketrains[0].shape
    print("Modified Shape:", modified_shape)

    return blocks


def print_cluster_ids(blocks):
    cluster_ids = set()  # Create a set to store unique cluster IDs

    for seg in blocks[0].segments:
        print(seg)
        for st in seg.spiketrains:
            if 'cluster_id' in st.annotations:
                cluster_id = st.annotations['cluster_id']
                cluster_ids.add(cluster_id)

    # Print all unique cluster IDs found
    print("Unique Cluster IDs:")
    for cluster_id in cluster_ids:
        print(cluster_id)


def get_fra_raster(blocks, clust_id, df_filter=[]):
    unit_aligned_time = np.array([])
    for s, seg in enumerate(blocks[0].segments):
        if seg.df_bhv is None:
            continue

        df_bhv = seg.df_bhv

        # df_bhv['targTimes'] = df_bhv['timeToTarget'] / 24414.0625
        #
        # df_bhv['centreRelease'] = df_bhv['lickRelease'] - df_bhv['startTrialLick']
        # df_bhv['relReleaseTimes'] = df_bhv['centreRelease'] - df_bhv['targTimes']
        # df_bhv['realRelReleaseTimes'] = df_bhv['relReleaseTimes'] - df_bhv['absentTime']

        # if len(df_filter) > 0:
        #     for f in df_filter:
        #         df_bhv = apply_filter(df_bhv, f)

        df_bhv = df_bhv.reset_index(drop=True)

        if len(df_bhv) == 0:
            continue

        unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == clust_id][0]

        if statistics.mean_firing_rate(unit) < 0.5:
            continue

        if len(unit_aligned_time) == 0:
            unit_aligned_time = return_soundonset_fra_array(unit, df_bhv)
        else:

            seg_aligned_word = return_soundonset_fra_array(unit, df_bhv)
            seg_aligned_word['trial_num'] = seg_aligned_word['trial_num'] + np.max(unit_aligned_time['trial_num'])
            unit_aligned_time = np.concatenate((unit_aligned_time, seg_aligned_word))

    return unit_aligned_time


def get_word_aligned_raster_with_pitchshift(blocks, clust_id, word=None, correctresponse=True, selectedpitch=2,
                                            df_filter=[]):
    unit_aligned_time = np.array([])
    for s, seg in enumerate(blocks[0].segments):
        if seg.df_bhv is None:
            continue
        pitchshiftlist = np.array([])
        droplist = np.array([])
        for k in range(0, len(seg.df_bhv)):
            try:
                if all([v == 0 for v in seg.df_bhv.PitchShiftMat.values[k]]):
                    pitchshiftlist = np.append(pitchshiftlist, 0)
                else:
                    pitchshiftlist = np.append(pitchshiftlist, 1)
            except:
                droplist = np.append(droplist, k)
                # indexdrop = seg.df_bhv.iloc[droplist].columns.get_loc('PitchShiftMat')
                continue
        seg.df_bhv = seg.df_bhv.drop(droplist)
        seg.df_bhv['pitchshift'] = pitchshiftlist

        if word == None:
            word = seg.df_bhv.probes[0]
        df_bhv = seg.df_bhv

        #
        # if pitchshift == True:
        #     df_bhv = df_bhv.loc[df_bhv.pitchshift == 1]
        # elif pitchshift == False:
        #     df_bhv = df_bhv.loc[df_bhv.pitchshift == 0]
        df_bhv['targTimes'] = df_bhv['timeToTarget'] / 24414.0625

        df_bhv['centreRelease'] = df_bhv['lickRelease'] - df_bhv['startTrialLick']
        df_bhv['relReleaseTimes'] = df_bhv['centreRelease'] - df_bhv['targTimes']
        if df_bhv['absentTime'] != 0.2:
            print('DOES NOT EUQAL 0.2 SECONDS')

        df_bhv['realRelReleaseTimes'] = df_bhv['relReleaseTimes'] - df_bhv['absentTime']
        if correctresponse:
            df_bhv = df_bhv.loc[df_bhv['realRelReleaseTimes'].between(0, 2.3, inclusive=True)]

        distractors = df_bhv['distractors']
        talkermat = {}
        talkerlist = df_bhv['talker']

        for i0 in range(0, len(distractors)):
            talkermat[i0] = int(talkerlist.values[i0]) * np.ones(len(distractors.values[i0]))
        talkermat = pd.Series(talkermat, index=talkermat.keys())

        pitchshiftmat = df_bhv['PitchShiftMat']
        # if len(pitchshiftmat) == 0:
        #     pitchshiftmat = talkermat  # make array equivalent to size of pitch shift mat just like talker [3,3,3,3] # if this is inter trial roving then talker is the pitch shift

        # except:
        #     pitchshiftmat = talkermat  # make array equivalent to size of pitch shift mat just like talker [3,3,3,3] # if this is inter trial roving then talker is the pitch shift
        precursorlist = df_bhv['distractors']
        catchtriallist = df_bhv['catchTrial']
        chosenresponse = df_bhv['response']
        realrelreleasetimelist = df_bhv['realRelReleaseTimes']
        pitchoftarg = np.empty(len(pitchshiftmat))
        pitchofprecur = np.empty(len(pitchshiftmat))
        stepval = np.empty(len(pitchshiftmat))
        gradinpitch = np.empty(len(pitchshiftmat))
        gradinpitchprecur = np.empty(len(pitchshiftmat))
        timetotarglist = np.empty(len(pitchshiftmat))

        precur_and_targ_same = np.empty(len(pitchshiftmat))
        talkerlist2 = np.empty(len(pitchshiftmat))

        correctresp = np.empty(shape=(0, 0))
        pastcorrectresp = np.empty(shape=(0, 0))
        pastcatchtrial = np.empty(shape=(0, 0))
        droplist = np.empty(shape=(0, 0))
        droplistnew = np.empty(shape=(0, 0))

        for i in range(0, len(df_bhv['realRelReleaseTimes'].values)):
            chosenresponseindex = chosenresponse.values[i]

            realrelreleasetime = realrelreleasetimelist.values[i]

            chosentrial = pitchshiftmat.values[i]
            is_all_zero = np.all((chosentrial == 0))
            if isinstance(chosentrial, float) or is_all_zero:
                chosentrial = talkermat.values[i].astype(int)

            chosendisttrial = precursorlist.values[i]
            chosentalker = talkerlist.values[i]
            if chosentalker == 3:
                chosentalker = 1
            if chosentalker == 8:
                chosentalker = 2
            if chosentalker == 13:
                chosentalker = 2
            if chosentalker == 5:
                chosentalker = 1
            talkerlist2[i] = chosentalker

            targpos = np.where(chosendisttrial == word)
            if ((
                        chosenresponseindex == 0 or chosenresponseindex == 1) and realrelreleasetime >= 0) or chosenresponseindex == 3:
                correctresp = np.append(correctresp, 1)
            else:
                correctresp = np.append(correctresp, 0)

            try:

                pitchoftarg[i] = chosentrial[targpos[0][0]]  # check this for target raster function TODO
                if pitchoftarg[i] == 3:
                    pitchoftarg[i] = 3

                elif pitchoftarg[i] == 8:
                    pitchoftarg[i] = 3
                elif pitchoftarg[i] == 13:
                    pitchoftarg[i] = 1
                elif pitchoftarg[i] == 5:
                    pitchoftarg[i] = 5
                elif pitchoftarg[i] == 1:
                    pitchoftarg[i] = 4
                elif pitchoftarg[i] == 2:
                    pitchoftarg[i] = 2

                # print(pitchoftarg, 'iteration', i)






            except:
                # print(len(newdata))
                indexdrop = df_bhv.iloc[i].name
                droplist = np.append(droplist, i)
                ##arrays START AT 0, but the index starts at 1, so the index is 1 less than the array
                droplistnew = np.append(droplistnew, indexdrop)
                continue
        # # newdata.drop(0, axis=0, inplace=True)  # drop first trial for each animal
        # # accidentally dropping all catch trials?
        # df_bhv.drop(index=df_bhv.index[0],
        #              axis=0,
        #              inplace=True)
        df_bhv.drop(droplistnew, axis=0, inplace=True)

        droplist = [int(x) for x in droplist]  # drop corrupted metdata trials

        # pitchoftarg = pitchoftarg[~np.isnan(pitchoftarg)]
        # pitchoftarg = pitchoftarg.astype(int)
        # pitchofprecur = pitchofprecur[~np.isnan(pitchofprecur)]
        pitchofprecur = pitchofprecur.astype(int)

        pitchoftarg = np.delete(pitchoftarg, droplist)
        talkerlist2 = np.delete(talkerlist2, droplist)
        stepval = np.delete(stepval, droplist)

        df_bhv['pitchoftarg'] = pitchoftarg.tolist()

        pitchofprecur = np.delete(pitchofprecur, droplist)
        df_bhv['pitchofprecur'] = pitchofprecur.tolist()

        correctresp = np.delete(correctresp, droplist)

        precur_and_targ_same = np.delete(precur_and_targ_same, droplist)

        correctresp = correctresp.astype(int)
        # pastcatchtrial = pastcatchtrial.astype(int)
        # pastcorrectresp = pastcorrectresp.astype(int)

        df_bhv['correctresp'] = correctresp.tolist()
        df_bhv['talker'] = talkerlist2.tolist()
        df_bhv['stepval'] = stepval.tolist()
        precur_and_targ_same = precur_and_targ_same.astype(int)
        df_bhv['precur_and_targ_same'] = precur_and_targ_same.tolist()
        df_bhv['timeToTarget'] = df_bhv['timeToTarget'] / 24414.0625
        df_bhv = df_bhv[df_bhv['pitchoftarg'] == selectedpitch]

        if len(df_filter) > 0:
            for f in df_filter:
                df_bhv = apply_filter(df_bhv, f)

        df_bhv = df_bhv.reset_index(drop=True)

        if len(df_bhv) == 0:
            continue

        unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == clust_id][0]

        if statistics.mean_firing_rate(unit) < 0.5:
            continue

        if len(unit_aligned_time) == 0:
            unit_aligned_time = return_word_aligned_array(unit, df_bhv, word)
        else:
            try:
                seg_aligned_word = return_word_aligned_array(unit, df_bhv, word)
                seg_aligned_word['trial_num'] = seg_aligned_word['trial_num'] + np.max(unit_aligned_time['trial_num'])
                unit_aligned_time = np.concatenate((unit_aligned_time, seg_aligned_word))
                # print("no instance of word found in this behavioural data file ")
            except:
                continue

    return unit_aligned_time
