#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 20:47:28 2019

@author: mostafamousavi
"""
from keras.layers import add, ConvLSTM2D, Reshape, Dense, AveragePooling2D, Input, Conv2DTranspose, TimeDistributed, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import add, Reshape, Dense, Input, TimeDistributed, Dropout, Activation, LSTM, Conv1D, Cropping1D
from keras.layers import MaxPooling1D, UpSampling1D, Cropping1D, SpatialDropout1D, Bidirectional, BatchNormalization 
from keras.models import Model, Sequential
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras import metrics
from keras.optimizers import Adam
import locale
import os
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from os import listdir, walk
from os.path import isfile, join, isdir
import pickle
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random
from datetime import datetime
from datetime import timedelta
import os.path
from keras import Sequential
from keras.layers import Dense
import h5py
import tensorflow as tf
from keras import backend as K
import pandas as pd
import csv





def datat_reader(file_name, file_list):

    net_code = []
    rec_type = []
    eq_id = []
    eq_depth = []
    eq_mag = []
    mag_type = []
    mag_auth = []
    eq_dist = []
    snr = []
    trace_name = []
    S_P = []
    baz = []
    
    x_temp =[]
    y_temp = []

    conv_L = []
    eVa_L = []
    eVe_L = []
            
    dtfl = h5py.File(file_name, 'r')
   
    pbar = tqdm(total=len(file_list)) 
    for c, evi in enumerate(file_list):
        pbar.update()
        dataset = dtfl.get('earthquake/local/'+str(evi))   
        data = np.array(dataset)
        
        try:
            BAZ = round(float(dataset.attrs['back_azimuth_deg']), 2)
        except Exception:
            BAZ = None
            
        if BAZ == 0:
            BAZ = 360
        try:
            spt = int(dataset.attrs['p_arrival_sample'])
        except Exception:
            spt = None

        try:
            sst = int(dataset.attrs['s_arrival_sample'])
        except Exception:
            sst = None
 
        try:
            cod = int(dataset.attrs['coda_end_sample'])
        except Exception:
            cod = None
          
        try:
            dpt = round(float(dataset.attrs['source_depth_km']), 2)
        except Exception:
            dpt = 0

        try:
            mag = round(float(dataset.attrs['source_magnitude']), 2)
        except Exception:
            mag = 0

        try:
            dis = round(float(dataset.attrs['source_distance_km']), 2)
        except Exception:
            dis = 0

        try:
            SNR = dataset.attrs['snr_db']
        except Exception:
            SNR = 0
            
        try:
            sp = int(sst - spt)
        except Exception:
            sp = 0

        short_p = data[spt:spt+20, :]

        cov_p = np.cov(short_p, rowvar=False)  
        try:
            eVa, eVe = np.linalg.eig(cov_p)
            eVa /= np.max(eVa)
        except Exception:
            eVa = [None, None, None]            
            pass
  
        cov_p /= np.max(abs(cov_p)) 
        eVa /= np.max(eVa)
        
        if all(SNR >= 5) and spt >= 50 and spt <= 5750 and not np.isnan(cov_p).any() and not np.isnan(eVa).any():
            dshort = data[spt-50:spt+100, :] 
            dshort /= np.max(dshort)
            conv_L.append(cov_p)
            eVa_L.append(eVa)
            eVe_L.append(eVe)
            x_temp.append(dshort)
            y_temp.append(BAZ)
            net_code.append(dataset.attrs['network_code'])
            rec_type.append(dataset.attrs['receiver_type'])
            eq_id.append(dataset.attrs['source_id'])
            eq_depth.append(dpt)  
            eq_mag.append(mag)
            mag_type.append(dataset.attrs['source_magnitude_type'])
            mag_auth.append(dataset.attrs['source_magnitude_author'])
            eq_dist.append(dis) 
            snr.append(round(np.mean(SNR), 2))
            trace_name.append(dataset.attrs['trace_name'])
            S_P.append(sp)
            baz.append(BAZ) 
    dtfl.close()
    
    X = np.zeros([len(x_temp), 150 , 3]) 
    X2 = np.zeros([len(x_temp), 7 , 3])  
    Y = np.zeros([len(y_temp), 2])
    
    for c, dat in enumerate(x_temp):  
        X[c, :, :] = dat 
        X2[c, :3, :] = conv_L[c] 
        X2[c, 3, 0] = eVa_L[c][0]
        X2[c, 3, 1] = eVa_L[c][1]
        X2[c, 3, 2] = eVa_L[c][2]
        X2[c, 4:, :] = eVe_L[c] 
                
        tang_r = np.radians(y_temp[c])        
        Y[c, 0] = np.cos(tang_r)
        Y[c, 1] = np.sin(tang_r)

     
            
    return X, X2, Y, net_code, rec_type, eq_id, eq_depth, eq_mag, mag_type, mag_auth, eq_dist, snr, trace_name, S_P, baz




def string_convertor(dd):
    dd2 = dd.split()
    SNR = []
    for i, d in enumerate(dd2):
        if d != '[' and d != ']':
            
            dL = d.split('[')
            dR = d.split(']')
            
            if len(dL) == 2:
                dig = dL[1]
            elif len(dR) == 2:
                dig = dR[0]
            elif len(dR) == 1 and len(dR) == 1:
                dig = d
            try:
                dig = float(dig)
            except Exception:
                dig = None
                
            SNR.append(dig)
    return(SNR)

    
###############################################################################
file_name = "./dataset2/waveforms.hdf5"
csv_file = "./dataset2/metadata.csv" 

 
# file_name = "../../STEAD/dataset/waveforms.hdf5"
# csv_file = "../../STEAD/dataset/metadata.csv"


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.1

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))


limit = 10000
epochs_number=200
batch_size=128
monte_carlo_sampling = 50


# =============================================================================
# df = pd.read_csv(csv_file) 
# df = df[df.trace_category == 'earthquake_local']
# df = df[df.p_arrival_sample > 100]
# df = df[df.p_travel_sec.notnull()]
# df = df[df.p_travel_sec > 0]
# df = df[df.p_travel_sec < 50]
# df = df[df.source_distance_km.notnull()]
# df = df[df.source_distance_km > 0]
# df = df[df.source_distance_km <= 110] 
# df = df[df.back_azimuth_deg.notnull()]
# df = df[df.back_azimuth_deg > 0]
# df.snr_db = df.snr_db.apply(lambda x: np.mean(string_convertor(x)))
# df = df[df.snr_db >= 25]
#  
# df.info()
# 
# =============================================================================
# =============================================================================
# ll = []
# for index, row in df.iterrows():
#     net = str(row['network_code'])
#     st = str(row['receiver_code'])
#     rt = str(row['receiver_type']+'*')
#     ll.append(net+'_'+st+'_'+rt)
# ll2 = set(ll) 
# print(len(ll2))   
#  
# from obspy import read, UTCDateTime
# from obspy.clients.fdsn import Client
# import time as tt
# 
#  
# client_iris = Client("IRIS")
# starttime = UTCDateTime('1900-01-01')
# endtime = UTCDateTime('2019-01-01')
#     
# tr_names = []
# for ro in ll2:
#     ro.split('_')[0]
#     
#     aligned_east = False
#     aligned_north = False
#     net = ro.split('_')[0]
#     st = ro.split('_')[1]
#     rt = ro.split('_')[2]
#     print(st)
#     
#     try:
#         cat_iris = client_iris.get_stations(network=net, 
#                                                 starttime=starttime, 
#                                                 endtime=endtime, 
#                                                 station=st, 
#                                                 channel=rt,
#                                                 level="channel"); 
#       #  print(cat_iris)
#         net = cat_iris[0]
#         sta = net[0]
#         cha_0 = sta[0]
#         if cha_0.azimuth == 90:
#             aligned_east = True
#         cha_1 = sta[1]
#         if cha_1.azimuth == 0:
#             aligned_north = True
#         
#         if aligned_east == True and aligned_north == True:
#             tr_names.append(str(ro))
#     except Exception:
#         print('@@@@@@@@ Didnt get this one ... ')
#         pass
#         
#     tt.sleep(0.2)
# 
# 
# np.save('surface_stations', tr_names)
# =============================================================================

# =============================================================================
# surface = np.load('surface_stations.npy')
#   
# ev_list = []
# for index, row in df.iterrows():
#     net = str(row['network_code'])
#     st = str(row['receiver_code'])
#     rt = str(row['receiver_type']+'*')
#     evid = net+'_'+st+'_'+rt
#     if evid in surface:
#         ev_list.append(row['trace_name'])
#           
# np.random.shuffle(ev_list)  
# #ev_list = ev_list[:limit]
#      
# training = ev_list[:int(0.80*len(ev_list))]
# test =  ev_list[int(0.80*len(ev_list)):]
# =============================================================================
 
#np.save('az_trace_name_test2', test)
#np.save('az_trace_name_train2', training)


test = np.load('az_trace_name_test2.npy')
training = np.load('az_trace_name_train2.npy')

model_name = 'az_regression_cov3'
save_dir = os.path.join(os.getcwd(), str(model_name)+'_outputs')
if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)  
os.makedirs(save_dir)


x_train, x2_train, y_train, _, _, _, _, _, _, _, _, _, trace_name_train, _, baz_train = datat_reader(file_name, training) 
x_test, x2_test, y_test, net_code, rec_type, eq_id, eq_depth, eq_mag, mag_type, mag_auth, eq_dist, snr, trace_name, S_P, baz_test = datat_reader(file_name, test)  

assert not np.any(np.isnan(x_train).any())
assert not np.any(np.isnan(x_test).any())
assert not np.any(np.isnan(x2_train).any())
assert not np.any(np.isnan(x2_test).any())
assert not np.any(np.isnan(y_train).any())
assert not np.any(np.isnan(y_test).any())


drop_rate = 0.1

filters = [20, 32, 64, 128, 256] 
   
inp1 = Input(shape=(150, 3), name='input_layer1') 

e = Conv1D(filters[0], 3, padding = 'same', activation='relu')(inp1) 
e = Dropout(drop_rate)(e, training=True)
e = MaxPooling1D(2, padding='same')(e)

e = Conv1D(filters[1], 3, padding = 'same', activation='relu')(e) 
e = Dropout(drop_rate)(e, training=True)
e = MaxPooling1D(2, padding='same')(e)

e = Conv1D(filters[2], 3, padding = 'same', activation='relu')(e) 
e = Dropout(drop_rate)(e, training=True)
e = MaxPooling1D(2, padding='same')(e)

e = Conv1D(filters[0], 3, padding = 'same', activation='relu')(e) 
e = Dropout(drop_rate)(e, training=True)
e = MaxPooling1D(2, padding='same')(e)

e1 = Flatten()(e)

inp2 = Input(shape=(7, 3), name='input_layer2')  
e2 = Conv1D(filters[0], 1, padding = 'same', activation='relu')(inp2) 
e2 = Flatten()(e2)

o = keras.layers.concatenate([e1, e2])

o = Dense(100, activation='relu')(o)
o = Dropout(0.3)(o, training=True)
o = Dense(2)(o)
o = Activation('linear')(o)

model = Model(inputs=[inp1, inp2], outputs=o)
    
model.summary()


model.compile(optimizer='adam', loss='mse')


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown= 0,
                                patience= 3,
                                min_lr=0.5e-6)

m_name = str(model_name)+'_{epoch:03d}.h5' 
filepath = os.path.join(save_dir, m_name)

early_stopping_monitor = EarlyStopping(monitor= 'val_loss', patience = 5)

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             mode = 'auto',
                             verbose=1,
                             save_best_only=True)

callbacks = [early_stopping_monitor, checkpoint, lr_reducer]


history = model.fit([x_train, x2_train], y_train, epochs=epochs_number, validation_split=0.1, batch_size=batch_size, callbacks = callbacks)


#np.save(save_dir+'/history',history)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'], '--')
ax.legend(['loss', 'val_loss'], loc='upper right') 
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
fig.savefig(os.path.join(save_dir,str('X_learning_curve_loss.png'))) 


# for calculating the epistemic uncertainty 
class KerasDropoutPrediction(object):
    def __init__(self, model):
        self.model = model

    def predict(self, x, n_iter=10):
        predM = []
        
        for itr in range(n_iter):
            print('Prediction: #'+ str(itr+1))
            r = model.predict(x, batch_size=batch_size, verbose=0)
            
            pre_batch = []
            for _, xy in enumerate(r):
                pr = np.arctan2(xy[1], xy[0])            
                pred = np.degrees(pr)  
                if pred < 0:
                    pred += 360
                       
                pre_batch.append(round(pred, 2))
            
            predM.append(pre_batch)

        predM = np.array(predM).reshape(n_iter,len(predM[0]))
        
        yhat_mean = predM.mean(axis=0)        
        ep_unc = predM.std(axis=0)  
  
        return yhat_mean, ep_unc
    
kdp = KerasDropoutPrediction(model)
predic, ep_unc = kdp.predict([x_test, x2_test], monte_carlo_sampling)




def align(y_true, y_pred):
    """ Add or remove 2*pi to predicted angle to minimize difference from GT"""
    y_pred = y_pred.copy()
    y_pred[y_true-y_pred >  np.pi] += np.pi*2
    y_pred[y_true-y_pred < -np.pi] -= np.pi*2
    return y_pred

baz_ts = []
for i in range(len(y_test[:,1])):
    bazT = np.degrees(np.arctan2(y_test[i,1], y_test[i,0]))
    if bazT < 0:
        bazT += 360
    baz_ts.append(bazT)

# predic = align(baz_ts, predic)

diff = []
for i, pr in enumerate(predic):
    diff.append(baz_ts[i]-pr)
    


fig4, ax = plt.subplots()
ax.scatter(baz_ts, predic, alpha = 0.4, facecolors='none', edgecolors='r')
ax.plot([min(baz_ts), max(baz_ts)], [min(baz_ts), max(baz_ts)], 'k--', alpha=0.4, lw=2)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.title('BAZ-test')
fig4.savefig(os.path.join(save_dir,'plots4.png')) 

fig4, ax = plt.subplots()
ax.hist(diff, bins = 100)
ax.set_xlabel('degree')
plt.title('BAZ-test')
fig4.savefig(os.path.join(save_dir,'plots4hist.png')) 

fig2 = plt.figure()
plt.errorbar(predic, ep_unc, xerr= ep_unc, fmt='o', alpha=0.4, ecolor='g', capthick=2)
plt.plot(baz_ts, ep_unc, 'ro', alpha=0.4)
plt.xlabel('Magnitude')
plt.ylabel('Epistemic Uncertainty')
plt.title('Epistemic Uncertainty')
fig2.savefig(os.path.join(save_dir,'plots2.png')) 



csvfile = open(os.path.join(save_dir,'results.csv'), 'w')          
output_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
output_writer.writerow(['network_code', 'reciever_type', 'source_id', 'source_depth_km', 'source_magnitude',
                        'mag_type', 'mag_auth', 'eq_dist', 'snr', 'trace_name', 'S-P', 'back_AZ', 
                        'true_BAZ', 'predicted_BAZ', 'diff', 'ep_uncertainty'])
csvfile.flush()


for i, v in enumerate(baz_ts):
    output_writer.writerow([net_code[i], 
                            rec_type[i], 
                            eq_id[i], 
                            eq_depth[i], 
                            eq_mag[i], 
                            mag_type[i], 
                            mag_auth[i], 
                            eq_dist[i], 
                            round(snr[i], 2),
                            trace_name[i],
                            S_P[i],
                            baz_test[i],
                            round(v, 2), 
                            round(predic[i], 2), 
                            round(diff[i], 2),
                            round(ep_unc[i], 2),
                            ])           
    csvfile.flush() 



print('Writting results into: ' + str(model_name)+'_outputs')

with open(os.path.join(save_dir,'report.txt'), 'a') as the_file:
    the_file.write('file_name: '+str(file_name)+'\n')
    the_file.write('model_name: '+str(model_name)+'\n')    
    the_file.write('epoch_number: '+str(epochs_number,)+'\n')
    the_file.write('batch_size: '+str(batch_size)+'\n')
    the_file.write('total number of training: '+str(len(training))+'\n')
    the_file.write('total number of test: '+str(len(test))+'\n')
    the_file.write('average error: '+str(np.round(np.mean(diff), 2))+'\n')
    the_file.write('average error_std: '+str(np.round(np.std(diff), 2))+'\n')  
    the_file.write('stoped after epoche: '+str(len(history.history['loss']))+'\n')
    the_file.write('last loss: '+str(history.history['loss'][-1])+'\n')
    the_file.write('monte_carlo_sampling: '+str(monte_carlo_sampling)+'\n')
    the_file.write('mean Epistemic Uncertainty: '+str(np.mean(ep_unc))+'\n')





# =============================================================================
# # =============================================================================
# # 
# #     
#  baz_ts = []
#  diff = []
#  prr = []
#  ep_unc2 = []
#  for i in range(len(y_test[:,1])):
#      bazT = np.degrees(np.arctan2(y_test[i,1], y_test[i,0]))
#      if bazT < 0:
#          bazT += 360
#      if ep_unc[i] < 7:
#          baz_ts.append(bazT)
#          prr.append(predic[i])
#          diff.append(bazT-predic[i])
#          ep_unc2.append(ep_unc[i])
#  
#  fig2 = plt.figure()
#  plt.errorbar(prr, ep_unc2, xerr= ep_unc2, fmt='o', alpha=0.4, ecolor='g', capthick=2)
#  plt.plot(baz_ts, ep_unc2, 'ro', alpha=0.4)
#  plt.xlabel('Back Azimuth')
#  plt.ylabel('Epistemic Uncertainty')
#  plt.title('Epistemic Uncertainty')
#  fig2.savefig(os.path.join(save_dir,'plots2.png')) 
# 
#  fig4, ax = plt.subplots()
#  ax.scatter(baz_ts, prr, alpha = 0.4, facecolors='none', edgecolors='r')
#  ax.plot([min(baz_ts), max(baz_ts)], [min(baz_ts), max(baz_ts)], 'k--', alpha=0.4, lw=2)
#  ax.set_xlabel('Measured')
#  ax.set_ylabel('Predicted')
#  plt.title('BAZ-test')
#  fig4.savefig(os.path.join(save_dir,'plots4.png')) 
#  
#  fig4, ax = plt.subplots()
#  ax.hist(diff, bins = 100)
#  ax.set_xlabel('degree')
#  plt.title('BAZ-test')
#  fig4.savefig(os.path.join(save_dir,'plots4hist.png')) 
# # =============================================================================
# 
# =============================================================================
