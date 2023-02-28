# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:13:32 2022

@author: nqhung
"""

import numpy as np
np.random.seed(1001)
import pandas as pd
import os
import shutil
import IPython
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook
import scipy.io.wavfile as wavfile
import stumpy as stumpy
import itertools as itertools
from matplotlib.patches import Rectangle, FancyArrowPatch
%matplotlib inline
matplotlib.style.use('ggplot')
#%% load file
import audio2numpy as a2n
x1,sr1=a2n.audio_from_file("C:/Users/nqhun/OneDrive/Desktop/CSC874/Python/CSC874/data/Audio2.mp3") 
x2,sr2=a2n.audio_from_file("C:/Users/nqhun/OneDrive/Desktop/CSC874/Python/CSC874/data/Audio1.mp3")

fig,ax = plt.subplots(2,sharex = True, gridspec_kw={'hspace': 0},figsize = (20,10))
plt.suptitle("Audio series",fontsize = 40)
ax[0].plot(x1[:,1],color = 'red')
ax[0].set_title('audio 1',fontsize = 20)
ax[1].plot(x2[:,1],color = 'green')
ax[1].set_title('audio 2',fontsize = 20)
ax[1].set_xlabel('Time',fontsize = 30)
plt.show()

m = 2200000-1650000
a=np.transpose(np.array([x1[1650000:2000000,1]]))
T1 = np.transpose(np.array([x2[0:3000000,1]]))
T2 = np.transpose(np.array([x2[3000000:len(x2[:,1]),1]]))
np.shape(T1)
T = np.row_stack((T1,a))
T = np.row_stack((T,T2))

#%% Plot file audio
fig,ax = plt.subplots(2,sharex = True, gridspec_kw={'hspace': 0},figsize = (20,10))
plt.suptitle("Audio series New",fontsize = 40)
ax[1].plot(x2[:,1])
ax[0].set_title('audio 2',fontsize = 20)
ax[0].plot(T)
ax[1].set_title('audio2_new',fontsize = 20)
ax[1].set_xlabel('Time',fontsize = 30)
plt.show()
#%%
fig,ax = plt.subplots(2,sharex = True, gridspec_kw={'hspace': 0},figsize = (20,10))
plt.suptitle("Audio Series",fontsize = 40)
ax[0].plot(x1[1650000:1651000,1])
ax[0].set_title('audio 1',fontsize = 20)
ax[1].plot(T[3000000:3001000])
ax[1].set_title('audio 2 New',fontsize = 20)
ax[1].set_xlabel('Time',fontsize = 30)
plt.show()
Ta = np.float64(T)
Tb = np.array([np.float64(x1[:,1])])
Tb = np.transpose(Tb)
Ta = pd.DataFrame(Ta)
Tb = pd.DataFrame(Tb)


fig,ax = plt.subplots(2,sharex = True, gridspec_kw={'hspace': 0},figsize = (20,10))
plt.suptitle("Audio series after reduce",fontsize = 40)
ax[1].plot(y1,color = 'red')
ax[0].set_title('audio1',fontsize = 20)
ax[0].plot(y2,color = 'green')
ax[1].set_title('audio2 new',fontsize = 20)
ax[1].set_xlabel('Time',fontsize = 30)
plt.show()
#%% 
Ta = np.array(np.float64(y1))
Tb = np.array([np.float64(y2)])
Tb = np.transpose(Tb)
Ta = pd.DataFrame(Ta)
Tb = pd.DataFrame(Tb)
#%% preprocessing
import stumpy
import time
start = time.time()
approx = stumpy.scrump(T_A = Ta.iloc[:,0],
                        m = 5500,
                        T_B = Tb.iloc[:,0],
                        percentage=0.10, pre_scrump=True, s=None)
end = time.time()

end-start

approx_P = approx.P_
approx_P
motif_index = approx_P.argmin()

plt.plot(approx_P)
plt.xlabel('Time')
plt.ylabel('Matrix Profile')

plt.scatter(motif_index,
               approx_P[motif_index],
               c='blue',
               s=100)
plt.plot(approx_P,color = 'red')
plt.show()
motif_index = approx_P.argmin()
print(f'The motif is located at index {motif_index} of "Under Pressure"')
#%% process
import time
from dask.distributed import Client
dask_client = Client()
start = time.time()
mp = stumpy.stumped(dask_client, T_A = Ta.iloc[:,0],
                        m = 5500,
                        T_B = Tb.iloc[:,0],
                        ignore_trivial = False)  # Note that a dask client is needed
end =time.time()
end -start
#%% result
motif_index = mp[:, 0].argmin()
mp[motif_index, 0]

plt.xlabel('Subsequence Time')
plt.ylabel('Matrix Profile')
plt.scatter(motif_index,
               mp[motif_index, 0],
               c='red',
               s=20)
plt.plot(mp[:,0])
np.mean(mp,axis = 0)
plt.show()

motif_index1 = mp[:,0].argmin()
motif_index2 = mp[motif_index1,1]
print(f'The motif is located at index {motif_index1} of "Audio 1"')
print(f'The motif is located at index {motif_index2} of "Audio 2"')
#%% finding the location for the minimum matrix profile
plt.plot(Ta.iloc[motif_index1 :motif_index1+1000].values, label='Uot my')
plt.plot(Tb.iloc[motif_index2:motif_index2+1000].values, label='Tayduky',alpha = 0.3)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.plot(Ta.iloc[motif_index1 :motif_index1+100].values, label='Audio 1')
plt.suptitle("Audio series after reduce",fontsize = 20)
plt.plot(Tb.iloc[motif_index2:motif_index2+100].values, label='Audio 2',alpha = 0.7)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.legend()
plt.show()
#%%
fig,ax = plt.subplots(2,sharex = True, gridspec_kw={'hspace': 0},figsize = (20,10))
plt.suptitle("Audio series after reduce",fontsize = 40)
ax[0].plot(Ta.iloc[motif_index1 :motif_index1+1000].values)
ax[0].set_title('Audio 2',fontsize = 20)
ax[1].plot(Tb.iloc[motif_index2:motif_index2+1000].values)
ax[1].set_title('Audio 1',fontsize = 20)
ax[1].set_xlabel('Time',fontsize = 30)
plt.show()

fig,ax = plt.subplots(2,sharex = True, gridspec_kw={'hspace': 0},figsize = (20,10))
plt.suptitle("Audio series after reduce",fontsize = 40)
ax[0].plot(Ta.iloc[motif_index1 :motif_index1+100].values)
ax[0].set_title('Audio 2',fontsize = 20)
ax[1].plot(Tb.iloc[motif_index2:motif_index2+100].values,color = 'blue')
ax[1].set_title('Audio 1',fontsize = 20)
ax[1].set_xlabel('Time',fontsize = 30)
plt.show()
#%% audio test
import librosa
y3, s3 = librosa.load('C:/Users/nqhun/OneDrive/Desktop/CSC874/Python/CSC874/data/audio test.wav', sr=1000) # Downsample 44.1kHz to 8kHz
Ttest = np.array(np.float64(y3))
Ttest = pd.DataFrame(Ttest)
len(Ttest)
#%% processing audio test
from dask.distributed import Client
dask_client1 = Client()
mp = stumpy.stumped(dask_client1, T_A = Ta.iloc[:,0],
                        m = 5500,
                        T_B = Ttest.iloc[:,0],
                        ignore_trivial = False)

motif_index = mp[:, 0].argmin()
plt.xlabel('Subsequence')
plt.ylabel('Matrix Profile')
plt.scatter(motif_index,
               mp[motif_index, 0],
               c='red',
               s=20)
plt.plot(mp[:,0])
plt.show()

motif_index1 = mp[:,0].argmin()
print(f'The motif is located at index {motif_index1} of "Audio 1"')
motif_index2 = mp[motif_index1,1]
print(f'The motif is located at index {motif_index2} of "Audio 2"')
#%% audio test
plt.plot(Ta.iloc[motif_index1 :motif_index1+1000].values, label='audio 2')
plt.plot(Tb.iloc[motif_index2:motif_index2+1000].values, label='audio test',alpha = 0.3)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.legend()
plt.show()
#%%
fig,ax = plt.subplots(2,sharex = True, gridspec_kw={'hspace': 0},figsize = (20,10))
plt.suptitle("Audio serie after reduce",fontsize = 40)
ax[0].plot(Ta.iloc[motif_index1 :motif_index1+1000].values)
ax[0].set_title('Audio 2',fontsize = 20)
ax[1].plot(Tb.iloc[motif_index2:motif_index2+1000].values,color = 'blue')
ax[1].set_title('Audio 1',fontsize = 20)
ax[1].set_xlabel('Time',fontsize = 30)
plt.show()

mp[motif_index, 0]
