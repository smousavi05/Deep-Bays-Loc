#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 05:47:46 2019

@author: mostafamousavi
"""

file_name = "../../STEAD/dataset/waveforms.hdf5"
csv_file = "../../STEAD/dataset/metadata.csv"


df = pd.read_csv(csv_file) 
df = df[df.trace_category == 'earthquake_local']
df = df[df.source_distance_km <= 110]
df = df[df.source_magnitude_type == 'ml']
df = df[df.p_arrival_sample >= 200]
df = df[df.p_arrival_sample+2900 <= 6000]
df = df[df.p_arrival_sample <= 1500]
df = df[df.s_arrival_sample >= 200]
df = df[df.s_arrival_sample <= 2500]
df = df[df.coda_end_sample <= 3000]
df = df[df.p_travel_sec.notnull()]
df = df[df.p_travel_sec > 0]
df = df[df.source_distance_km.notnull()]
df = df[df.source_distance_km > 0]
df = df[df.source_depth_km.notnull()]
df = df[df.source_magnitude.notnull()]
df = df[df.source_magnitude <= 2.5]
df = df[df.back_azimuth_deg.notnull()]
df.snr_db = df.snr_db.apply(lambda x: np.mean(string_convertor(x)))
df = df[df.snr_db >= 30]
df.info()
# =============================================================================
# uniq_ins = df.receiver_code.unique().tolist()
# print(uniq_ins)
# labM = []
# valM = []
# for ii in range(len(uniq_ins)):
#     print(str(uniq_ins[ii]), sum(n == str(uniq_ins[ii]) for n in df.receiver_code))
#     labM.append(str(uniq_ins[ii]))
#     valM.append(np.round(sum(n == str(uniq_ins[ii]) for n in df.receiver_code)/sum(n != None for n in df.receiver_code.dropna()), 2))
# 
# =============================================================================
ll = []
for index, row in df.iterrows():
    net = row['network_code']
    st = row['receiver_code']
    rt = row['receiver_type']+'*'
    ll.append(net+'_'+st+'_'+rt)
ll2 = set(ll) 
print(len(ll2))   

from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client
import time as tt


client_iris = Client("IRIS")
starttime = UTCDateTime('1900-01-01')
endtime = UTCDateTime('2019-01-01')
    
tr_names = []
for ro in ll2:
    ro.split('_')[0]
    
    aligned_east = False
    aligned_north = False
    net = ro.split('_')[0]
    st = ro.split('_')[1]
    rt = ro.split('_')[2]
    print(st)
    
    try:
        cat_iris = client_iris.get_stations(network=net, 
                                                starttime=starttime, 
                                                endtime=endtime, 
                                                station=st, 
                                                channel=rt,
                                                level="channel"); 
      #  print(cat_iris)
        net = cat_iris[0]
        sta = net[0]
        cha_0 = sta[0]
        if cha_0.azimuth == 90:
            aligned_east = True
        cha_1 = sta[1]
        if cha_1.azimuth == 0:
            aligned_north = True
        
        if aligned_east == True and aligned_north == True:
            tr_names.append(str(ro))
    except Exception:
        print('@@@@@@@@ Didnt get this one ... ')
        pass
        
    tt.sleep(0.2)
        
        
np.save('surface_stations', tr_names)    



sur_stio = []
for index, row in df.iterrows():
    net = row['network_code']
    st = row['receiver_code']
    rt = row['receiver_type']+'*'
    evid = net+'_'+st+'_'+rt
    if evid in tr_names:
        print('got one!')
        sur_stio.append(row['trace_name'])
        
    

