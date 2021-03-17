# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import mne # package for reading edf data
import numpy as np
import pandas as pd
from itertools import chain

import json
import matplotlib.pyplot as plt

# %%
def get_recording(subjectnum, trialnum):
    '''retrieves trial eeg recording given a subject number and trial number'''
    fileName = f"data/sub-{format(subjectnum, '02d')}/eeg/sub-{format(subjectnum, '02d')}_task-run{trialnum}_eeg.edf"
    data = mne.io.read_raw_edf(fileName)
    raw_data = data.get_data()
    return data.to_data_frame().infer_objects()

# %%
def get_events(subjectnum, trialnum):
    '''retrieves events and timings given a subject number and trial number'''
    subjectnum = '0' + (str(subjectnum)) if subjectnum < 10 else str(subjectnum)
    events = pd.read_csv("data\sub-"+str(subjectnum)+"\eeg\sub-"+str(subjectnum)+"_task-run"+str(trialnum)+"_events.tsv", sep="\t", usecols=['onset','trial_type']).infer_objects()
    eventcodes = {}
    with open('data\sub-'+str(subjectnum)+'\eeg\sub-'+str(subjectnum)+'_task-run'+str(trialnum)+'_events.json') as f:
        eventcodes = json.load(f)
    events['trial_name'] = events['trial_type'].apply(lambda x: eventcodes[str(x)]['LongName'] if str(x) in eventcodes.keys() else 'n/a')
    events['trial_desc'] = events['trial_type'].apply(lambda x: eventcodes[str(x)]['Description'] if str(x) in eventcodes.keys() else 'n/a')
    events['trial_desc_cleaned'] = events['trial_desc'].replace(".* \(The music made me feel |\)|.* \(| - The user.+", "",regex = True)
    # logic below gets song name for each of the 10 ish runs
    te_sample = events[(300<=events['trial_type'])&(events['trial_type']<=660)][['onset','trial_type']]
    songtime = te_sample['onset'].to_numpy()
    songdict = te_sample.set_index('onset').to_dict()['trial_type']
    def get_song_trial_type(searchVal, inputData):
        '''returns trial type of song for particular question sequence'''
        if searchVal < inputData.min(): return -1
        diff = inputData - searchVal
        diff[diff>=0] = -np.inf
        idx = diff.argmax()
        return songdict[inputData[idx]] -300
    events['song_clip'] = events.onset.map(lambda x: get_song_trial_type(x,songtime))

    return events
s1_2events = get_events(1,2)
s1_2events.head()


# %%
def get_recording_events(subjectnum, trialnum):    
    '''returns the recording events of each trial number of each subject'''
    trial_events = get_events(subjectnum,trialnum)
    te_sample = trial_events[(300<=trial_events['trial_type'])&(trial_events['trial_type']<=660)][['onset','trial_type']]
    songtime = te_sample['onset'].to_numpy()
    songdict = te_sample.set_index('onset').to_dict()['trial_type']
    def get_song_trial_type(searchVal, inputData):
        '''returns trial type of song for particular question sequence'''
        if searchVal < inputData.min(): return -1
        diff = inputData - searchVal
        diff[diff>=0] = -np.inf
        idx = diff.argmax()
        return songdict[inputData[idx]] -300
    trial_events['song_clip'] = trial_events.onset.map(lambda x: get_song_trial_type(x,songtime))
    trial_events['onset_time'] = (trial_events.onset*1000).astype(int)
    # trial_events
    record = get_recording(subjectnum,trialnum)
    # record
    ot = trial_events[trial_events.trial_type.isin([788,1092])].groupby('song_clip').head(2)[['song_clip','onset_time']]
    ot
    otdf = pd.DataFrame({'song':ot['song_clip'].iloc[::2].values,'start':ot['onset_time'].iloc[::2].values,'end':ot['onset_time'].iloc[1::2].values}).set_index('song')
    otdf
    df = pd.DataFrame(list(chain.from_iterable(pd.RangeIndex(otdf["start"],otdf["end"]) for _, otdf in otdf.iterrows())),columns=("time",)).merge(ot, how='left', left_on='time',right_on='onset_time').ffill().astype(int).drop(columns='onset_time')
    # df.song_clip.plot() # YES THIS IS IT THIS IS WHAT I WANT
    
    set1 = pd.read_csv('data/set1/set1/mean_ratings_set1.csv').drop(columns=['Unnamed: 10', 'Unnamed: 11'])
    return df.merge(record, how='left', on='time').merge(set1[['Number', 'TARGET']], left_on='song_clip', right_on='Number')


# %%
def get_q_a(subjectnum, trialnum):
    '''this garbage code returns the responses to the questions. positivity vs negativity of response is reflected by the score, the larger the number the stronger the agreement, zero is neutral.'''
    # get labeled trial events by adding the respective json file
    trialevents = get_events(subjectnum, trialnum)
    
    df = trialevents[trialevents.trial_name.str.contains('Question')|trialevents.trial_name.str.contains('Answer')|trialevents.trial_name.str.contains('Response')]
    df = df[df['trial_name'].str.contains('Question') & ~df['trial_name'].str.contains('hidden')|df['trial_name'].str.contains('Answer')]
    df = df[df['trial_name'].shift() != df['trial_name']]
    df = df.sort_values(['onset', 'trial_name'])
    answer_matrix = {'strongly disagree':-4,'disagree':-3,'somewhat disagree':-2,'slightly disagree':-1,'neither agree nor disagree':0,'slightly agree':1,'somewhat agree':2,'agree':3,'strongly agree':4}
    #get question answers
    qapair = df[df.trial_desc_cleaned.isin(['pleasent', 'energetic', 'tense', 'angry', 'afraid', 'happy', 'sad', 'tender']) & 
        df.shift().trial_desc_cleaned.isin(answer_matrix.keys())|df.trial_desc_cleaned.isin(answer_matrix.keys())]
    qapair['response'] = qapair.trial_desc_cleaned.shift(-1)
    qapair = qapair[qapair['trial_desc_cleaned'].isin(['pleasent', 'energetic', 'tense', 'angry', 'afraid', 'happy', 'sad', 'tender'])]
    return qapair.dropna().replace(answer_matrix)#.drop_duplicates('trial_name', 'last').sort_values('trial_name').reset_index(drop=True)

#%%
def quick_bin_power(X, Band, Fs):
    C = np.fft.fft(X)
    C = abs(C)
    Power = np.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        Power[Freq_Index] = sum(
            C[int(np.floor(Freq / Fs * len(X))): 
                int(np.floor(Next_Freq / Fs * len(X)))]
        )
    Power_Ratio = Power / sum(Power)
    return Power, Power_Ratio
# %%
def choice_bin_power(X, Band, Fs, ratio=False):
    '''does much of the stuff quick_bin_power does but 
    lets you choose with a bool to ratio or get raw power values (ratio)'''
    C = np.fft.fft(X)
    C = abs(C)
    Power = np.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        Power[Freq_Index] = sum(
            C[int(np.floor(Freq / Fs * len(X))): 
                int(np.floor(Next_Freq / Fs * len(X)))]
        )
    Power_Ratio = Power / sum(Power)
    # return Power, Power_Ratio
    return Power_Ratio if ratio else Power

# %%
def consecutive_window_powers(df_, step_size, sample_rate=1000, band = [4,8,12,30,45], band_labels = ['theta','alpha','beta','gamma'], ratio=True):
    '''does powers for non-overlapping segments of length for the dataframe at various points'''

    output = None
    for col in df_.columns:
        colvals = df_[col].groupby(df_.index //step_size).apply(lambda x: pd.Series(choice_bin_power(x, band, sample_rate, ratio), index=[str(col)+'_'+b+'_ratio' for b in band_labels])).unstack()
        if output is None:
            output = colvals #assignment
        else:
            output = output.merge(colvals, left_index=True, right_index=True) # merging
    return output