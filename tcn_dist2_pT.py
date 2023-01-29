#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 20:47:28 2019

@author: mostafamousavi
"""
from keras.layers import Dense, Input
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.optimizers import Adam

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import shutil
import os
import os.path
import h5py
import tensorflow as tf
from keras import backend as K
import pandas as pd
import csv




def datat_reader(file_name, file_list):
    
    def normalize(data, mode = 'std'):
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              
        elif mode == 'std':        
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data
    
    def isfloat(x):
        try:
            a = float(x)
        except ValueError:
            return False
        else:
            return True

    def isint(x):
        try:
            a = float(x)
            b = int(a)
        except ValueError:
            return False
        else:
            return a == b

    net_code = []
    rec_type = []
    eq_id = []
    eq_depth = []
    eq_mag = []
    mag_type = []
    eq_dist = []
    snr = []
    trace_name = []
    S_P = []
    baz = []
    spt_L = []
    sst_L = []
    
    x_temp =[]
    P_travel = []
    
       
    dtfl = h5py.File(file_name, 'r')
   
    pbar = tqdm(total=len(file_list)) 
    for c, evi in enumerate(file_list):
        pbar.update()
        dataset = dtfl.get('earthquake/local/'+str(evi))   
        data = np.array(dataset)
        
        try:
            mag = round(float(dataset.attrs['source_magnitude']), 2)
        except Exception:
            mag = None

        try:
            spt = int(dataset.attrs['p_arrival_sample']);
        except Exception:
            spt = None

        try:
            sst = int(dataset.attrs['s_arrival_sample']);
        except Exception:
            sst = None
          
        try:
            dpt = round(float(dataset.attrs['source_depth_km']), 2)
        except Exception:
            dpt = None

        try:
            dis = round(float(dataset.attrs['source_distance_km']), 2)
        except Exception:
            dis = None

        try:
            SNR = dataset.attrs['snr_db']
        except Exception:
            SNR = 0
            
        try:
            sp = int(sst - spt)
        except Exception:
            sp = None
                    
        try:
            BAZ = round(float(dataset.attrs['back_azimuth_deg']), 2)
        except Exception:
            BAZ = None            

        try:
            P_t = float(dataset.attrs['p_travel_sec'])
        except Exception:
            P_t = None            
 
            
        if dis and P_t:
            x_temp.append(normalize(data, 'max'))
            P_travel.append(P_t)
            eq_dist.append(dis) 
            baz.append(BAZ) 

            spt_L.append(spt)
            sst_L.append(sst)
            
            net_code.append(dataset.attrs['network_code'])
            rec_type.append(dataset.attrs['receiver_type'])
            eq_id.append(dataset.attrs['source_id'])
            eq_depth.append(dpt)  
            eq_mag.append(mag)
            mag_type.append(dataset.attrs['source_magnitude_type'])
            snr.append(np.mean(SNR))
            trace_name.append(dataset.attrs['trace_name'])
            S_P.append(sp)
            
    dtfl.close()
    
    X = np.zeros([len(x_temp), 6000 , 4])  
    Ydis = np.zeros([len(x_temp), 1])
    Ypt = np.zeros([len(x_temp), 1])
    
    for c, dat in enumerate(x_temp):  
        X[c, :, 0:3] = dat   
        X[c, spt_L[c]:sst_L[c], 3] = 1
        Ydis[c] = eq_dist[c]
        Ypt[c] = P_travel[c]
            
    return {'input': X}, {'distance': Ydis, 'p_travel': Ypt}, net_code, rec_type, eq_id, eq_depth, eq_mag, mag_type, eq_dist, snr, trace_name, S_P, baz, P_travel



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

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))

#file_name = "../../STEAD/dataset/waveforms.hdf5"
#csv_file = "../../STEAD/dataset/metadata.csv" 

limit = 200000
epochs_number= 200
batch_size = 256
monte_carlo_sampling = 100
drop_rate = 0.1

# =============================================================================
# df = pd.read_csv(csv_file) 
# gk = df.groupby(['trace_category'])
# gk2 = gk.get_group(('earthquake_local' ))
# gk2 = gk2[gk2.source_distance_deg <= 1]
# gk2 = gk2[gk2.p_travel_sec.notnull()]
# gk2.snr_db = gk2.snr_db.apply(lambda x: np.mean(string_convertor(x)))
# ev_list = gk2[gk2.snr_db >= 20.0].trace_name.tolist()
# 
# np.random.shuffle(ev_list)  
# ev_list = ev_list[:limit]
# 
# 
# training = ev_list[:int(0.85*len(ev_list))]
# test =  ev_list[int(0.85*len(ev_list)):]
# =============================================================================

test = np.load('az_trace_name_test2.npy')
training = np.load('az_trace_name_train2.npy')

model_name = 'tcn_dist2_pT_7'
save_dir = os.path.join(os.getcwd(), str(model_name)+'_outputs')
if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)  
os.makedirs(save_dir)

   
x_train, y_train, _, _, _, _, _, _, _, _, _, trace_name_train, baz_train, _ = datat_reader(file_name, training) 
x_test, y_test, net_code, rec_type, eq_id, eq_depth, eq_mag, mag_type,  eq_dist, snr, trace_name_test, S_P, baz_test, P_travel = datat_reader(file_name, test)  

assert not np.any(np.isnan(x_train['input']).any())
assert not np.any(np.isnan(x_test['input']).any())
assert not np.any(np.isnan(y_train['distance']).any())
assert not np.any(np.isnan(y_test['distance']).any())
assert not np.any(np.isnan(y_train['p_travel']).any())
assert not np.any(np.isnan(y_test['p_travel']).any())


from tcn import TCN

batch_size, timesteps, input_dim = None, 6000, 4

i = Input(batch_shape=(batch_size, timesteps, input_dim), name='input')
o = TCN(nb_filters=20,
        kernel_size=6,
        nb_stacks=1,
        dilations=[2 ** i for i in range(11)],
        padding='causal',
        use_skip_connections=True, 
        dropout_rate = drop_rate,
        return_sequences=False
        )(i)  

do = Dense(2)(o)
do = Activation('linear', name='distance')(do)

po = Dense(2)(o)
po = Activation('linear', name='p_travel')(po)

model = Model(inputs=i, outputs=[do, po])


# costom loss for calculating aleatoric uncertainty
def customLoss(yTrue, yPred):
    y_hat = K.reshape(yPred[:, 0], [-1, 1]) 
    s = K.reshape(yPred[:, 1], [-1, 1])
    return tf.reduce_sum((0.5 * K.exp(-1 * s) * K.square(K.abs(yTrue - y_hat)) + 0.5 * s), axis=1)

model.compile(optimizer='adam', loss=[customLoss, customLoss], loss_weights=[0.1, 0.9])

model.summary()


##############################################################################
####################################################################  TRAINING
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience= 4,
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
      
history = model.fit(x_train, y_train, epochs=epochs_number, validation_split=0.1, callbacks=callbacks)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'], '--')
ax.legend(['loss', 'val_loss'], loc='upper right') 
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
fig.savefig(os.path.join(save_dir,str('X_learning_curve_loss.png'))) 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(history.history['distance_loss'])
ax.plot(history.history['val_distance_loss'], '--')
ax.legend(['distance_loss', 'val_distance_loss'], loc='upper right') 
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
fig.savefig(os.path.join(save_dir,str('X_learning_distance_loss.png')))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(history.history['p_travel_loss'])
ax.plot(history.history['val_p_travel_loss'], '--')
ax.legend(['p_travel_loss', 'val_p_travel_loss'], loc='upper right') 
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
fig.savefig(os.path.join(save_dir,str('X_learning_p_travel_loss.png')))



# for calculating the epistemic uncertainty 
print('Monte Carlo sampling ... ')
class KerasDropoutPrediction(object):
    def __init__(self, model):
        self.model = model
         
    def predict(self, x, n_iter=10):
        predM_d = []
        auM_d = []
        predM_pt = []
        auM_pt = []
         
        pbar = tqdm(total=n_iter) 
        for itr in range(n_iter):
            pbar.update()
            r = model.predict(x, batch_size=batch_size, verbose=0)
             
            pred_d = r[0][:, 0] 
            au_d = r[0][:, 1] 
            predM_d.append(pred_d.T)
            auM_d.append(au_d.T)
            
            pred_pt = r[1][:, 0] 
            au_pt = r[1][:, 1] 
            predM_pt.append(pred_pt.T)
            auM_pt.append(au_pt.T)            
         
        predM_d = np.array(predM_d).reshape(n_iter,len(predM_d[0]))
        auM_d = np.array(auM_d).reshape(n_iter, len(auM_d[0]))        
        yhat_mean_d = predM_d.mean(axis=0)
        yhat_squared_mean_d = np.square(predM_d).mean(axis=0)        
        sigma_squared_d = (auM_d) 
        sigma_squared_mean_d = sigma_squared_d.mean(axis=0)        
        ep_unc_d = predM_d.std(axis=0)          
        combibed_d = yhat_squared_mean_d - np.square(yhat_mean_d)+ sigma_squared_mean_d

        predM_pt = np.array(predM_pt).reshape(n_iter,len(predM_pt[0]))
        auM_pt = np.array(auM_pt).reshape(n_iter, len(auM_pt[0]))        
        yhat_mean_pt = predM_pt.mean(axis=0)
        yhat_squared_mean_pt = np.square(predM_pt).mean(axis=0)        
        sigma_squared_pt = (auM_pt) 
        sigma_squared_mean_pt = sigma_squared_pt.mean(axis=0)        
        ep_unc_pt = predM_pt.std(axis=0)          
        combibed_pt = yhat_squared_mean_pt - np.square(yhat_mean_pt)+ sigma_squared_mean_pt
         
        return yhat_mean_d, sigma_squared_mean_d, ep_unc_d, combibed_d, yhat_mean_pt, sigma_squared_mean_pt, ep_unc_pt, combibed_pt
     
kdp = KerasDropoutPrediction(model)
predic_d, al_unc_d, ep_unc_d, comb_d, predic_pt, al_unc_pt, ep_unc_pt, comb_pt = kdp.predict(x_test, monte_carlo_sampling)
 
 
fig1 = plt.figure()
plt.errorbar(predic_d, al_unc_d, xerr= al_unc_d, fmt='o', alpha=0.4, ecolor='g', capthick=2)
plt.plot(y_test['distance'], al_unc_d, 'ro', alpha=0.4)
plt.xlabel('Distance')
plt.ylabel('Aleatoric Uncertainty')
plt.title('Aleatoric Uncertainty_distance')
fig1.savefig(os.path.join(save_dir,'plots1.png')) 
 
fig2 = plt.figure()
plt.errorbar(predic_d, ep_unc_d, xerr= ep_unc_d, fmt='o', alpha=0.4, ecolor='g', capthick=2)
plt.plot(y_test['distance'], ep_unc_d, 'ro', alpha=0.4)
plt.xlabel('Distance')
plt.ylabel('Epistemic Uncertainty')
plt.title('Epistemic Uncertainty_distance')
fig2.savefig(os.path.join(save_dir,'plots2.png')) 
 
fig3 = plt.figure()
plt.errorbar(predic_d, comb_d, xerr= comb_d, fmt='o', alpha=0.4, ecolor='g', capthick=2)
plt.plot(y_test['distance'], comb_d, 'ro', alpha=0.4)
plt.xlabel('Distance')
plt.ylabel('Combined Uncertainty')
plt.title('Combined Uncertainty_distance')
fig3.savefig(os.path.join(save_dir,'plots3.png')) 
 
fig4, ax = plt.subplots()
ax.scatter(y_test['distance'], predic_d, alpha = 0.4, facecolors='none', edgecolors='r')
ax.plot([y_test['distance'].min(), y_test['distance'].max()], [y_test['distance'].min(), y_test['distance'].max()], 'k--', alpha=0.4, lw=2)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.title('Distance')
fig4.savefig(os.path.join(save_dir,'plots4.png')) 
 

fig5 = plt.figure()
plt.errorbar(predic_pt, al_unc_pt, xerr= al_unc_pt, fmt='o', alpha=0.4, ecolor='g', capthick=2)
plt.plot(y_test['p_travel'], al_unc_pt, 'ro', alpha=0.4)
plt.xlabel('P_travel tiem')
plt.ylabel('Aleatoric Uncertainty')
plt.title('Aleatoric Uncertainty_p-travel')
fig5.savefig(os.path.join(save_dir,'plots5.png')) 
 
fig6 = plt.figure()
plt.errorbar(predic_pt, ep_unc_pt, xerr= ep_unc_pt, fmt='o', alpha=0.4, ecolor='g', capthick=2)
plt.plot(y_test['p_travel'], ep_unc_pt, 'ro', alpha=0.4)
plt.xlabel('p_travel')
plt.ylabel('Epistemic Uncertainty')
plt.title('Epistemic Uncertainty_distance')
fig6.savefig(os.path.join(save_dir,'plots6.png')) 
 
fig7 = plt.figure()
plt.errorbar(predic_pt, comb_pt, xerr= comb_pt, fmt='o', alpha=0.4, ecolor='g', capthick=2)
plt.plot(y_test['p_travel'], comb_pt, 'ro', alpha=0.4)
plt.xlabel('p_travel')
plt.ylabel('Combined Uncertainty')
plt.title('Combined Uncertainty_distance')
fig7.savefig(os.path.join(save_dir,'plots7.png')) 
 
fig8, ax = plt.subplots()
ax.scatter(y_test['p_travel'], predic_pt, alpha = 0.4, facecolors='none', edgecolors='r')
ax.plot([y_test['p_travel'].min(), y_test['p_travel'].max()], [y_test['p_travel'].min(), y_test['p_travel'].max()], 'k--', alpha=0.4, lw=2)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.title('p_travel')
fig8.savefig(os.path.join(save_dir,'plots8.png')) 
  


csvfile = open(os.path.join(save_dir,'results.csv'), 'w')          
output_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
output_writer.writerow(['network_code', 'reciever_type', 'source_id', 'source_depth_km',
                        'source_magnitude', 'mag_type','snr', 'trace_name',  'S-P', 
                        'true_distance_km', 'predicted_distance_km', 'diffD', 'dist_al_uncertainty', 'dist_ep_uncertainty', 'dist_comb_uncertainty',
                        'true_Ptravel_time_s', 'predicted_Ptravel_time_s', 'diffPT', 'pt_al_uncertainty', 'pt_ep_uncertainty', 'pt_comb_uncertainty',
                        ])
csvfile.flush()

                
diffD = []
diffPT = []
yPrd = []
for i, v in enumerate(predic_d):
    True_dist = y_test['distance'][i][0]
    d = True_dist - v
    diffD.append(d)
    

    True_PT = y_test['p_travel'][i][0]
    dPT = True_PT - predic_pt[i]
    diffPT.append(dPT)
          
    output_writer.writerow([net_code[i], 
                            rec_type[i], 
                            eq_id[i], 
                            eq_depth[i], 
                            eq_mag[i], 
                            mag_type[i], 
                            np.round(snr[i],2), 
                            trace_name_test[i], S_P[i],
                            True_dist, 
                            np.round(v,2), 
                            np.round(d, 4), 
                            round(al_unc_d[i], 3), 
                            round(ep_unc_d[i], 3), 
                            round(comb_d[i], 3),  
                            round(True_PT, 2), 
                            np.round(predic_pt[i],2), 
                            np.round(dPT, 4), 
                            round(al_unc_pt[i], 3), 
                            round(ep_unc_pt[i], 3), 
                            round(comb_pt[i], 3),  
                            ])  
    csvfile.flush() 

print('Writting results into: ' + str(model_name)+'_outputs')    
with open(os.path.join(save_dir,'report.txt'), 'a') as the_file:
    the_file.write('file_name: '+str(file_name)+'\n')
    the_file.write('model_name: '+str(model_name)+'\n')
    the_file.write('epoch_number: '+str(epochs_number)+'\n')
    the_file.write('total number of training: '+str(len(training))+'\n')
    the_file.write('Distance average error: '+str(np.round(np.mean(diffD), 2))+'\n')
    the_file.write('Distance average error_std: '+str(np.round(np.std(diffD), 2))+'\n')
    the_file.write('P_travel average error: '+str(np.round(np.mean(diffPT), 2))+'\n')
    the_file.write('P_travel average error_std: '+str(np.round(np.std(diffPT), 2))+'\n')    



  
 


