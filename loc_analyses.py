#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:47:27 2019

@author: mostafamousavi
"""
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.metrics import r2_score
import h5py
from tqdm import tqdm

import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from datetime import datetime, timedelta
from matplotlib.patches import Ellipse
from mpl_toolkits.basemap import pyproj
import math
from matplotlib.patches import Circle
from obspy.geodetics import locations2degrees
from obspy.geodetics import degrees2kilometers


plt.style.use('ggplot')


def vinc_pt(f, a, phi1, lembda1, alpha12, s ) : 
        import math
        """ 
        Returns the lat and long of projected point and reverse azimuth 
        given a reference point and a distance and azimuth to project. 
        lats, longs and azimuths are passed in decimal degrees 
        Returns ( phi2,  lambda2,  alpha21 ) as a tuple
        Parameters:
        ===========
            f: flattening of the ellipsoid
            a: radius of the ellipsoid, meteres
            phil: latitude of the start point, decimal degrees
            lembda1: longitude of the start point, decimal degrees
            alpha12: bearing, decimal degrees
            s: Distance to endpoint, meters
        NOTE: This code could have some license issues. It has been obtained 
        from a forum and its license is not clear. I'll reimplement with
        GPL3 as soon as possible.
        The code has been taken from
        https://isis.astrogeology.usgs.gov/IsisSupport/index.php?topic=408.0
        and refers to (broken link)
        http://wegener.mechanik.tu-darmstadt.de/GMT-Help/Archiv/att-8710/Geodetic_py
        """ 
        piD4 = math.atan( 1.0 ) 
        two_pi = piD4 * 8.0 
        phi1    = phi1    * piD4 / 45.0 
        lembda1 = lembda1 * piD4 / 45.0 
        alpha12 = alpha12 * piD4 / 45.0 
        if ( alpha12 < 0.0 ) : 
            alpha12 = alpha12 + two_pi 
        if ( alpha12 > two_pi ) : 
            alpha12 = alpha12 - two_pi
        b = a * (1.0 - f) 
        TanU1 = (1-f) * math.tan(phi1) 
        U1 = math.atan( TanU1 ) 
        sigma1 = math.atan2( TanU1, math.cos(alpha12) ) 
        Sinalpha = math.cos(U1) * math.sin(alpha12) 
        cosalpha_sq = 1.0 - Sinalpha * Sinalpha 
        u2 = cosalpha_sq * (a * a - b * b ) / (b * b) 
        A = 1.0 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * \
            (320 - 175 * u2) ) ) 
        B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2) ) ) 
        # Starting with the approx 
        sigma = (s / (b * A)) 
        last_sigma = 2.0 * sigma + 2.0   # something impossible 
            
        # Iterate the following 3 eqs unitl no sig change in sigma 
        # two_sigma_m , delta_sigma 
        while ( abs( (last_sigma - sigma) / sigma) > 1.0e-9 ):
            two_sigma_m = 2 * sigma1 + sigma 
            delta_sigma = B * math.sin(sigma) * ( math.cos(two_sigma_m) \
                    + (B/4) * (math.cos(sigma) * \
                    (-1 + 2 * math.pow( math.cos(two_sigma_m), 2 ) -  \
                    (B/6) * math.cos(two_sigma_m) * \
                    (-3 + 4 * math.pow(math.sin(sigma), 2 )) *  \
                    (-3 + 4 * math.pow( math.cos (two_sigma_m), 2 )))))
            last_sigma = sigma 
            sigma = (s / (b * A)) + delta_sigma 
        phi2 = math.atan2 ( (math.sin(U1) * math.cos(sigma) +\
            math.cos(U1) * math.sin(sigma) * math.cos(alpha12) ), \
            ((1-f) * math.sqrt( math.pow(Sinalpha, 2) +  \
            pow(math.sin(U1) * math.sin(sigma) - math.cos(U1) * \
            math.cos(sigma) * math.cos(alpha12), 2))))
        lembda = math.atan2( (math.sin(sigma) * math.sin(alpha12 )),\
            (math.cos(U1) * math.cos(sigma) -  \
            math.sin(U1) *  math.sin(sigma) * math.cos(alpha12))) 
        C = (f/16) * cosalpha_sq * (4 + f * (4 - 3 * cosalpha_sq )) 
        omega = lembda - (1-C) * f * Sinalpha *  \
            (sigma + C * math.sin(sigma) * (math.cos(two_sigma_m) + \
            C * math.cos(sigma) * (-1 + 2 *\
            math.pow(math.cos(two_sigma_m), 2) ))) 
        lembda2 = lembda1 + omega 
        alpha21 = math.atan2 ( Sinalpha, (-math.sin(U1) * \
            math.sin(sigma) +
            math.cos(U1) * math.cos(sigma) * math.cos(alpha12))) 
        alpha21 = alpha21 + two_pi / 2.0 
        if ( alpha21 < 0.0 ) : 
            alpha21 = alpha21 + two_pi 
        if ( alpha21 > two_pi ) : 
            alpha21 = alpha21 - two_pi 
        phi2 = phi2 * 45.0 / piD4 
        lembda2 = lembda2 * 45.0 / piD4 
        alpha21 = alpha21 * 45.0 / piD4
        return phi2, lembda2, alpha21
    

############################### overal data set carateristics

test = np.load('az_trace_name_test2.npy')
training = np.load('az_trace_name_train2.npy')
used_dataset = np.concatenate((test, training), axis=0)
used_dataset = list(used_dataset)
df = pd.read_csv( "../../STEAD/dataset/metadata.csv") 
df_used = df[df['trace_name'].isin(used_dataset)]
df_used.info()

df.info()

df_used = df[df['source_id'] == 'nn00495902' ]
df_used.source_magnitude
df_used.source_magnitude_type 


fig = plt.figure(figsize=(5, 17)) 
ax = fig.add_subplot(511)   
df_used['source_magnitude'].plot(kind='hist', bins = 40,  facecolor='teal', alpha=0.9, edgecolor='black')
textstr = '\n'.join((
    r'Max: %.2f M' % (df_used['source_magnitude'].max(), ),
    r'Min: %.2f M' % (df_used['source_magnitude'].min(), )))
props = dict(boxstyle='round', facecolor='teal', alpha=0.99)
plt.text(4, 15000, textstr, fontsize=13, color ='w' , verticalalignment='top', bbox=props)
plt.ylabel("Frequency",fontweight='bold',fontsize=14)
plt.xlabel('Magnitude M',fontweight='bold',fontsize=14)
plt.rcParams.update({'font.size': 12})
plt.tick_params()
plt.grid(True)
plt.tight_layout()


ax = fig.add_subplot(512)   
df_used['source_distance_km'].plot(kind='hist', bins = 40,  facecolor='teal', alpha=0.9, edgecolor='black')
textstr = '\n'.join((
    r'Max: %.2f km' % (df_used['source_distance_km'].max(), ),
    r'Min: %.2f km' % (df_used['source_distance_km'].min(), )))
props = dict(boxstyle='round', facecolor='teal', alpha=0.99)
plt.text(65, 25000, textstr, fontsize=13, color ='w' , verticalalignment='top', bbox=props)
plt.ylabel("Frequency",fontweight='bold',fontsize=14)
plt.xlabel('Epicentral Distance km',fontweight='bold',fontsize=14)
plt.rcParams.update({'font.size': 12})
plt.tick_params()
plt.grid(True)
plt.tight_layout()


ax = fig.add_subplot(513)   
df_used['p_travel_sec'].plot(kind='hist', bins = 30, logy=False, facecolor='teal', alpha=0.9, edgecolor='black')
textstr = '\n'.join((
    r'Max: %.2f s' % (df_used['p_travel_sec'].max(), ),
    r'Min: %.2f s' % (df_used['p_travel_sec'].min(), )))
props = dict(boxstyle='round', facecolor='teal', alpha=0.99)
plt.text(23, 40000, textstr, fontsize=13, color ='w' , verticalalignment='top', bbox=props)
plt.ylabel("Frequency",fontweight='bold',fontsize=14)
plt.xlabel('P Travel Time s',fontweight='bold',fontsize=14)
plt.rcParams.update({'font.size': 12})
plt.tick_params()
plt.grid(True)
plt.tight_layout()


ax = fig.add_subplot(514)   
df_used['back_azimuth_deg'].plot(kind='hist', bins = 30, facecolor='teal', alpha=0.9, edgecolor='black')
plt.ylabel("Frequency",fontweight='bold',fontsize=14)
plt.xlabel('Back Azimuth degree',fontweight='bold',fontsize=14)
plt.rcParams.update({'font.size': 12})
plt.tick_params()
plt.grid(True)
plt.tight_layout()


ax = fig.add_subplot(515)   
df_used = df_used.dropna()
df_used['source_depth_km'].astype('float64').plot(kind='hist', bins = 30, logy=False,facecolor='teal', alpha=0.9, edgecolor='black')
textstr = '\n'.join((
    r'Max: %.1f km' % (df_used['source_depth_km'].astype('float64').max(), ),
    r'Min: %.1f km' % (df_used['source_depth_km'].astype('float64').min(), )))
props = dict(boxstyle='round', facecolor='teal', alpha=0.99)
plt.text(160, 61000, textstr, fontsize=12, color ='w' , verticalalignment='top', bbox=props)
plt.ylabel("Frequency",fontweight='bold',fontsize=14)
plt.xlabel('Depth km',fontweight='bold',fontsize=14)
plt.rcParams.update({'font.size': 12})
plt.grid(True)
plt.tight_layout()


plt.savefig('./Xfigs/dataset_dists3.png', dpi=500)
plt.show()









df = pd.read_csv( "../../STEAD/dataset/metadata.csv") 
df_used = df[df['trace_name'].isin(full_dataset)]
 
lon1, lat1 = (df_used.source_longitude.values, df_used.source_latitude.values)    
m = Basemap(llcrnrlon = min(lon1)-5, llcrnrlat = min(lat1)-5,
                urcrnrlon = max(lon1)+1, urcrnrlat = max(lat1)+5,
                lat_0 = 0, 
                lon_0 = 0,
                resolution='i', projection='mill')  # resolution: i, h, l     projection='lcc', "mill"

m.drawmapboundary(fill_color='#85A6D9')
m.fillcontinents(color='lavender',lake_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawcountries(color='#6D5F47', linewidth=.4)                
m.drawmeridians(np.arange(-180,180, 40), color='#bbbbbb', labels=[0,0,0,1],fontsize=8,linewidth=0.5)
m.drawparallels(np.arange(-90, 90, 20), color='#bbbbbb', labels=[1,0,0,0], fontsize=8,linewidth=0.5)                

m.scatter(lon1, 
          lat1, 
          latlon = True, 
          marker = "o", 
          color = "mediumorchid", 
          alpha = 0.5,
          s = 20,
          zorder=5,
          edgecolor='black',
          label="Earthquake Locations"
          
              )  

plt.legend(loc='upper right', borderaxespad=0.2)
plt.savefig('./Xfigs/dataset_map.png', dpi=500)
plt.show()
    
    





################################  regression results



#@@@@@@@@ baz
csv_file = "./az_regression_cov3_outputs/results.csv"
df_az = pd.read_csv(csv_file) 
df_az = df_az[df_az.ep_uncertainty < 12]
df_az.info()


tm = df_az['true_BAZ'].values
pm = df_az['predicted_BAZ'].values
corr, p_value = pearsonr(tm, pm)
textstr = ('$\mathregular{R^2}$ = %.2f' % round(r2_score(tm, pm), 2))
props = dict(boxstyle='round', facecolor='mediumorchid', alpha=0.3, linewidth=1.2, edgecolor='black')
df_az.plot(kind='scatter', x = 'true_BAZ',y='predicted_BAZ', color= 'mediumorchid', alpha = 0.4, edgecolor='black',)
plt.plot([min(tm), max(tm)], [min(tm), max(tm)], 'w--', alpha=0.8, lw=3)
plt.text(40, 340, textstr, fontsize=13, verticalalignment='top', bbox=props)
plt.title("Back Azimuth",fontweight='bold',fontsize=14)
plt.xlabel("True BAZ degree",fontweight='bold',fontsize=12)
plt.ylabel('Predicted BAZ degree',fontweight='bold',fontsize=12)
plt.rcParams.update({'font.size': 12})
plt.tick_params()
plt.grid(True)
plt.tight_layout()
plt.savefig('./Xfigs/reg_baz.png', dpi=400)
plt.show()


textstr_av = ('Mean = %.2f' % round(np.mean(df_az['diff']), 2))
textstr_std = ('Std = %.2f' % round(np.std(df_az['diff']), 2))
props = dict(boxstyle='round', facecolor='mediumorchid', alpha=0.3, edgecolor='black', linewidth=1.2)
df_az['diff'].plot(kind='hist', bins = 50,  facecolor='mediumorchid', alpha=0.9, edgecolor='black', linewidth=1.2, label = 'ml Surface')
plt.xlabel("Prediction Errors",fontweight='bold',fontsize=12,  color= 'gray' )
plt.text(-310, 7100, textstr_av, fontsize=13, verticalalignment='top', bbox=props)
plt.text(-310, 6200, textstr_std, fontsize=13, verticalalignment='top', bbox=props)
plt.title("Back Azimuth",fontweight='bold',fontsize=14)
plt.tight_layout()
plt.savefig('./Xfigs/err_baz.png', dpi=400)
plt.show()







#@@@@@@@@ dist
csv_file = "./tcn_dist2_pT_6_outputs/results.csv"
df_dist = pd.read_csv(csv_file) 
df_dist.info()

az_list = df_az.trace_name.tolist()
df_dist = df_dist[df_dist['trace_name'].isin(az_list)]

tm = df_dist['true_distance_km'].values
pm = df_dist['predicted_distance_km'].values
corr, p_value = pearsonr(tm, pm)
textstr = ('$\mathregular{R^2}$ = %.2f' % round(r2_score(tm, pm), 2))
props = dict(boxstyle='round', facecolor='mediumslateblue', alpha=0.3, edgecolor='black', linewidth=1.2)
df_dist.plot(kind='scatter', x = 'true_distance_km',y='predicted_distance_km', color= 'mediumslateblue', alpha = 0.4, edgecolor='black',)
plt.plot([min(tm), max(tm)], [min(tm), max(tm)], 'w--', alpha=0.8, lw=3)
plt.text(0.0, 120, textstr, fontsize=13, verticalalignment='top', bbox=props)
plt.title("Distance",fontweight='bold',fontsize=14)
plt.xlabel("True Distance km",fontweight='bold',fontsize=12)
plt.ylabel('Predicted Distance km',fontweight='bold',fontsize=12)
plt.rcParams.update({'font.size': 12})
plt.tick_params()
plt.grid(True)
plt.tight_layout()
plt.savefig('./Xfigs/reg_dist.png', dpi=400)
plt.show()



df_dist['diffD'] = df_dist['diffD'].astype(str).astype(float)
textstr_av = ('Mean = %.2f' % round(np.mean(df_dist['diffD']), 2))
textstr_std = ('Std = %.2f' % round(np.std(df_dist['diffD']), 2))
props = dict(boxstyle='round', facecolor='mediumslateblue', alpha=0.3, edgecolor='black', linewidth=1.2)
df_dist['diffD'].plot(kind='hist', bins = 50,  logy= False, facecolor='mediumslateblue', alpha=0.9, edgecolor='black', linewidth=1.2)
plt.title("Distance",fontweight='bold',fontsize=14)
plt.xlabel("Prediction Errors km",fontweight='bold',fontsize=12,  color= 'gray' )
plt.text(-80, 8300, textstr_av, fontsize=13, verticalalignment='top', bbox=props)
plt.text(-80, 7200, textstr_std, fontsize=13, verticalalignment='top', bbox=props)
plt.tight_layout()
plt.savefig('./Xfigs/err_dist.png', dpi=400)
plt.show()




#@@@@@@@@ pT
tm = df_dist['true_Ptravel_time_s'].values
pm = df_dist['predicted_Ptravel_time_s'].values
corr, p_value = pearsonr(tm, pm)

textstr = ('$\mathregular{R^2}$ = %.2f' % round(r2_score(tm, pm), 2))
props = dict(boxstyle='round', facecolor='mediumturquoise', alpha=0.3, edgecolor='black', linewidth=1.2)
df_dist.plot(kind='scatter', x = 'true_Ptravel_time_s',y='predicted_Ptravel_time_s', color= 'mediumturquoise', alpha = 0.4, edgecolor='black',)
plt.plot([min(tm), max(tm)], [min(tm), max(tm)], 'w--', alpha=0.8, lw=3)
plt.text(0.0, 22, textstr, fontsize=13, verticalalignment='top', bbox=props)
plt.title("P Travel-Time",fontweight='bold',fontsize=14)
plt.xlabel("True P-travel Time s",fontweight='bold',fontsize=12)
plt.ylabel('Predicted P-travel Time s',fontweight='bold',fontsize=12)
plt.rcParams.update({'font.size': 12})
plt.tick_params()
plt.xlim([-1, 25])
plt.ylim([-1, 25])
plt.grid(True)
plt.tight_layout()
plt.savefig('./Xfigs/reg_pt.png', dpi=400)
plt.show()



df_dist['diffPT'] = df_dist['diffPT'].astype(str).astype(float)
textstr_av = ('Mean = %.2f' % round(np.mean(df_dist['diffPT']), 2))
textstr_std = ('Std = %.2f' % round(np.std(df_dist['diffPT']), 2))
props = dict(boxstyle='round', facecolor='mediumturquoise', alpha=0.3, edgecolor='black', linewidth=1.2)
df_dist['diffPT'].plot(kind='hist', bins = 50, logy= False, facecolor='mediumturquoise', alpha=0.9, edgecolor='black', linewidth=1.2)
plt.title("P Travel-Time",fontweight='bold',fontsize=14)
plt.xlabel("Prediction Errors s",fontweight='bold',fontsize=12,  color= 'gray' )
plt.xlim([-20, 20])
plt.text(-17, 7000, textstr_av, fontsize=13, verticalalignment='top', bbox=props)
plt.text(-17, 6000, textstr_std, fontsize=13, verticalalignment='top', bbox=props)
plt.tight_layout()
plt.savefig('./Xfigs/err_pt.png', dpi=400)
plt.show()

















################################# correlations
#df_dist = pd.read_csv("./tcn_dist2_pT_6_outputs/results.csv") 
#df_dist.info()
df_dist.diffD.describe()
df_dist.diffPT.describe()


# =============================================================================
# test = np.load('az_trace_name_test2.npy')
# test = list(test)
# df = pd.read_csv( "../../STEAD/dataset/metadata.csv") 
# df_test = df[df['trace_name'].isin(test)]
# df_test.info()
# 
# =============================================================================


# =============================================================================
# df_dist = pd.read_csv(csv_file) 
# df_dist['dist_al_uncertainty'] = df_dist['dist_al_uncertainty'].apply(np.sqrt)
# al = df_dist['dist_al_uncertainty'].values
# hrun = df_test['source_horizontal_uncertainty_km'].values
# 
# =============================================================================
df_dist = pd.read_csv(csv_file) 
df_dist['dist_comb_uncertainty'] = df_dist['dist_comb_uncertainty'].apply(np.sqrt)
df_dist['dist_al_uncertainty'] = df_dist['dist_al_uncertainty'].apply(np.sqrt)

df_dist.info()


ec ='k'
fc = 'r'

fig, axes = plt.subplots(2, 3, figsize=(12, 6))

df_dist.plot(kind='scatter', y = 'diffD', x='dist_al_uncertainty', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[0,0])
axes[0,0].set_ylabel("Distance Error km",fontweight='bold')
axes[0,0].set_xlabel("Aleatoric Uncertainty",fontweight='bold')
df_dist.plot(kind='scatter', y = 'diffD', x='dist_ep_uncertainty', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[0,1])
axes[0,1].set_ylabel('')
axes[0,1].set_xlabel("Epistemic Uncertainty",fontweight='bold')
df_dist.plot(kind='scatter', y = 'diffD', x='dist_comb_uncertainty', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[0,2])
axes[0,2].set_ylabel('')
axes[0,2].set_xlabel("Combined Uncertainty",fontweight='bold')

ec ='k'
fc = 'b'

df_dist.plot(kind='scatter', y = 'diffPT', x='pt_al_uncertainty', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[1,0])
axes[1,0].set_ylabel("Travel Time Error s",fontweight='bold')
axes[1,0].set_xlabel("Aleatoric Uncertainty",fontweight='bold')
df_dist.plot(kind='scatter', y = 'diffPT', x='pt_ep_uncertainty', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[1, 1])
axes[1,1].set_ylabel('')
axes[1,1].set_xlabel("Epistemic Uncertainty",fontweight='bold')
df_dist.plot(kind='scatter', y = 'diffPT', x='pt_comb_uncertainty', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[1,2])
axes[1,2].set_ylabel('')
axes[1,2].set_xlabel("Combined Uncertainty",fontweight='bold')

plt.rcParams.update({'font.size': 10})
plt.tick_params()
plt.grid(True)
plt.tight_layout()
plt.savefig('./Xfigs/diff_unc.png', dpi=500)
plt.show()




df_az.info()
ec ='k'
fc = 'r'

df_az.plot(kind='scatter', y = 'diff',x='ep_uncertainty', alpha= 0.3, c=fc ,edgecolors= ec)
plt.ylabel("Back Azimuth\n Error degree",fontweight='bold',fontsize=14)
plt.xlabel('Epistemic Uncertainty',fontweight='bold',fontsize=14)
plt.rcParams.update({'font.size': 12})
plt.tick_params()
plt.grid(True)
plt.tight_layout()
plt.savefig('./Xfigs/diff_unc_baz.png', dpi=500)
plt.show()



# =============================================================================
# 
# df_az.plot(kind='scatter', y = 'back_AZ',x='diff', alpha= 0.3)
# plt.ylabel("Back Azimuth degree",fontweight='bold',fontsize=14)
# plt.xlabel('Error degree',fontweight='bold',fontsize=14)
# plt.rcParams.update({'font.size': 12})
# plt.tick_params()
# plt.grid(True)
# plt.show()
# 
# =============================================================================




ec ='k'
fc = 'turquoise'


fig, axes = plt.subplots(8, 5, figsize=(16, 21))

df_dist.plot(kind='scatter', y = 'diffD', x='true_distance_km', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[0,0])
axes[0,0].set_ylabel("Distance Error km",fontweight='bold',fontsize=14)
axes[0,0].set_xlabel('')
axes[0,0].set_xticks([])
axes[0,0].grid(False)

df_dist.plot(kind='scatter', y = 'diffD', x='snr', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[0,1])
axes[0,1].set_ylabel('')
axes[0,1].set_xlabel('')
axes[0,1].set_xticks([])
axes[0,1].set_yticks([])

df_dist.plot(kind='scatter', y = 'diffD', x='source_magnitude', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[0,2])
axes[0,2].set_ylabel('')
axes[0,2].set_xlabel('')
axes[0,2].set_xticks([])
axes[0,2].set_yticks([])

df_dist.plot(kind='scatter', y = 'diffD', x='source_depth_km', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[0,3])
axes[0,3].set_ylabel('')
axes[0,3].set_xlabel('')
axes[0,3].set_xticks([])
axes[0,3].set_yticks([])

df_dist.plot(kind='scatter', y = 'diffD', x='S-P', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[0,4])
axes[0,4].set_ylabel('')
axes[0,4].set_xlabel('')
axes[0,4].set_xticks([])
axes[0,4].set_yticks([])

df_dist.plot(kind='scatter', y = 'dist_al_uncertainty', x='true_distance_km', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[1,0])
axes[1,0].set_ylabel("Distance\n Aleatoric Uncertainty",fontweight='bold',fontsize=12)
axes[1,0].set_xlabel('')
axes[1,0].set_xticks([])
axes[1,0].grid(False)

df_dist.plot(kind='scatter', y = 'dist_al_uncertainty',  x='snr', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[1,1])
axes[1,1].set_ylabel('')
axes[1,1].set_xlabel('')
axes[1,1].set_xticks([])
axes[1,1].set_yticks([])

df_dist.plot(kind='scatter', y = 'dist_al_uncertainty',  x='source_magnitude', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[1,2])
axes[1,2].set_ylabel('')
axes[1,2].set_xlabel('')
axes[1,2].set_xticks([])
axes[1,2].set_yticks([])

df_dist.plot(kind='scatter', y = 'dist_al_uncertainty',  x='source_depth_km', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[1,3])
axes[1,3].set_ylabel('')
axes[1,3].set_xlabel('')
axes[1,3].set_xticks([])
axes[1,3].set_yticks([])

df_dist.plot(kind='scatter', y = 'dist_al_uncertainty',  x='S-P', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[1,4])
axes[1,4].set_ylabel('')
axes[1,4].set_xlabel('')
axes[1,4].set_xticks([])
axes[1,4].set_yticks([])

df_dist.plot(kind='scatter', y = 'dist_ep_uncertainty', x='true_distance_km', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[2,0])
axes[2,0].set_ylabel('Distance\n Epistemic Uncertainty',fontweight='bold',fontsize=12)
axes[2,0].set_xlabel('')
axes[2,0].set_xticks([])
axes[2,0].grid(False)

df_dist.plot(kind='scatter', y = 'dist_ep_uncertainty',  x='snr', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[2,1])
axes[2,1].set_ylabel('')
axes[2,1].set_xlabel('')
axes[2,1].set_xticks([])
axes[2,1].set_yticks([])

df_dist.plot(kind='scatter', y = 'dist_ep_uncertainty',  x='source_magnitude', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[2,2])
axes[2,2].set_ylabel('')
axes[2,2].set_xlabel('')
axes[2,2].set_xticks([])
axes[2,2].set_yticks([])

df_dist.plot(kind='scatter', y = 'dist_ep_uncertainty',  x='source_depth_km', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[2,3])
axes[2,3].set_ylabel('')
axes[2,3].set_xlabel('')
axes[2,3].set_xticks([])
axes[2,3].set_yticks([])

df_dist.plot(kind='scatter', y = 'dist_ep_uncertainty',  x='S-P', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[2,4])
axes[2,4].set_ylabel('')
axes[2,4].set_xlabel('')
axes[2,4].set_xticks([])
axes[2,4].set_yticks([])

ec ='k'
fc = 'hotpink'


df_dist.plot(kind='scatter', y = 'diffPT', x='true_distance_km', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[3,0])
axes[3,0].set_ylabel('P Travel-Time\n Error s',fontweight='bold',fontsize=14)
axes[3,0].set_xlabel('')
axes[3,0].set_xticks([])
axes[3,0].grid(False)

df_dist.plot(kind='scatter', y = 'diffPT',  x='snr', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[3,1])
axes[3,1].set_ylabel('')
axes[3,1].set_xlabel('')
axes[3,1].set_xticks([])
axes[3,1].set_yticks([])

df_dist.plot(kind='scatter', y = 'diffPT',  x='source_magnitude', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[3,2])
axes[3,2].set_ylabel('')
axes[3,2].set_xlabel('')
axes[3,2].set_xticks([])
axes[3,2].set_yticks([])

df_dist.plot(kind='scatter', y = 'diffPT',  x='source_depth_km', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[3,3])
axes[3,3].set_ylabel('')
axes[3,3].set_xlabel('')
axes[3,3].set_xticks([])
axes[3,3].set_yticks([])

df_dist.plot(kind='scatter', y = 'diffPT',  x='S-P', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[3,4])
axes[3,4].set_ylabel('')
axes[3,4].set_xlabel('')
axes[3,4].set_xticks([])
axes[3,4].set_yticks([])

df_dist.plot(kind='scatter', y = 'pt_al_uncertainty', x='true_distance_km', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[4,0])
axes[4,0].set_ylabel('P Travel-Time\n Aleatoric Uncertainty',fontweight='bold',fontsize=12)
axes[4,0].set_xlabel('')
axes[4,0].set_xticks([])
axes[4,0].grid(False)

df_dist.plot(kind='scatter', y = 'pt_al_uncertainty',  x='snr', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[4,1])
axes[4,1].set_ylabel('')
axes[4,1].set_xlabel('')
axes[4,1].set_xticks([])
axes[4,1].set_yticks([])

df_dist.plot(kind='scatter', y = 'pt_al_uncertainty',  x='source_magnitude', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[4,2])
axes[4,2].set_ylabel('')
axes[4,2].set_xlabel('')
axes[4,2].set_xticks([])
axes[4,2].set_yticks([])

df_dist.plot(kind='scatter', y = 'pt_al_uncertainty',  x='source_depth_km', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[4,3])
axes[4,3].set_ylabel('')
axes[4,3].set_xlabel('')
axes[4,3].set_xticks([])
axes[4,3].set_yticks([])

df_dist.plot(kind='scatter', y = 'pt_al_uncertainty',  x='S-P', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[4,4])
axes[4,4].set_ylabel('')
axes[4,4].set_xlabel('')
axes[4,4].set_xticks([])
axes[4,4].set_yticks([])

df_dist.plot(kind='scatter', y = 'pt_ep_uncertainty', x='true_distance_km', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[5,0])
axes[5,0].set_ylabel('P Travel-Time\n Epistemic Uncertainty',fontweight='bold',fontsize=12)
axes[5,0].set_xlabel('')
axes[5,0].set_xticks([])
axes[5,0].grid(False)

df_dist.plot(kind='scatter', y = 'pt_ep_uncertainty',  x='snr', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[5,1])
axes[5,1].set_ylabel('')
axes[5,1].set_xlabel('')
axes[5,1].set_xticks([])
axes[5,1].set_yticks([])

df_dist.plot(kind='scatter', y = 'pt_ep_uncertainty',  x='source_magnitude', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[5,2])
axes[5,2].set_ylabel('')
axes[5,2].set_xlabel('')
axes[5,2].set_xticks([])
axes[5,2].set_yticks([])

df_dist.plot(kind='scatter', y = 'pt_ep_uncertainty',  x='source_depth_km', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[5,3])
axes[5,3].set_ylabel('')
axes[5,3].set_xlabel('')
axes[5,3].set_xticks([])
axes[5,3].set_yticks([])

df_dist.plot(kind='scatter', y = 'pt_ep_uncertainty',  x='S-P', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[5,4])
axes[5,4].set_ylabel('')
axes[5,4].set_xlabel('')
axes[5,4].set_xticks([])
axes[5,4].set_yticks([])

ec ='k'
fc = 'cornflowerblue'

df_az.plot(kind='scatter', y = 'diff', x='eq_dist', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[6,0])
axes[6,0].set_ylabel('Back Azimuth\n Error degree',fontweight='bold',fontsize=14)
axes[6,0].set_xlabel('')
axes[6,0].set_xticks([])
axes[6,0].grid(False)

df_az.plot(kind='scatter', y = 'diff',  x='snr', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[6,1])
axes[6,1].set_ylabel('')
axes[6,1].set_xlabel('')
axes[6,1].set_xticks([])
axes[6,1].set_yticks([])

df_az.plot(kind='scatter', y = 'diff',  x='source_magnitude', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[6,2])
axes[6,2].set_ylabel('')
axes[6,2].set_xlabel('')
axes[6,2].set_xticks([])
axes[6,2].set_yticks([])

df_az.plot(kind='scatter', y = 'diff',  x='source_depth_km', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[6,3])
axes[6,3].set_ylabel('')
axes[6,3].set_xlabel('')
axes[6,3].set_xticks([])
axes[6,3].set_yticks([])

df_az.plot(kind='scatter', y = 'diff',  x='S-P', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[6,4])
axes[6,4].set_ylabel('')
axes[6,4].set_xlabel('')
axes[6,4].set_xticks([])
axes[6,4].set_yticks([])

df_az.plot(kind='scatter', y = 'ep_uncertainty', x='eq_dist', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[7,0])
axes[7,0].set_ylabel('Back Azimuth\n Epistemic Uncertainty',fontweight='bold',fontsize=14)
axes[7,0].set_xlabel("Distance km",fontweight='bold',fontsize=14)
axes[7,0].grid(False)

df_az.plot(kind='scatter', y = 'ep_uncertainty',  x='snr', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[7,1])
axes[7,1].set_ylabel('')
axes[7,1].set_xlabel("SNR db",fontweight='bold',fontsize=14)
axes[7,1].set_yticks([])
axes[7,1].grid(False)

df_az.plot(kind='scatter', y = 'ep_uncertainty',  x='source_magnitude', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[7,2])
axes[7,2].set_ylabel('')
axes[7,2].set_xlabel("Magnitude M",fontweight='bold',fontsize=14)
axes[7,2].set_yticks([])
axes[7,2].grid(False)

df_az.plot(kind='scatter', y = 'ep_uncertainty',  x='source_depth_km', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[7,3])
axes[7,3].set_ylabel('')
axes[7,3].set_xlabel("Earthquake Depth km",fontweight='bold',fontsize=14)
axes[7,3].set_yticks([])
axes[7,3].grid(False)

df_az.plot(kind='scatter', y = 'ep_uncertainty',  x='S-P', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[7,4])
axes[7,4].set_ylabel('')
axes[7,4].set_xlabel("S - P sample",fontweight='bold',fontsize=14)
axes[7,4].set_yticks([])
axes[7,4].grid(False)

plt.rcParams.update({'font.size': 12})
plt.tick_params()
plt.grid(False)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05, wspace = 0.05)
plt.savefig('./Xfigs/scatter_plot.png', dpi=400)
plt.show()



# =============================================================================
# df_dist.plot(kind='scatter', x='source_magnitude', y = 'pt_ep_uncertainty', alpha= 0.3, c=fc ,edgecolors= ec)
# plt.title("P Travel Time",fontweight='bold',fontsize=14)
# plt.xlabel("Magnitude M",fontweight='bold',fontsize=14)
# plt.ylabel('Epistemic Uncertainty',fontweight='bold',fontsize=14)
# plt.rcParams.update({'font.size': 12})
# plt.tick_params()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('./Xfigs/pt_ep_mag.png', dpi=500)
# plt.show()
# 
# =============================================================================

# =============================================================================
# df = pd.read_csv(csv_file) 
# df.ep_uncertainty = df.ep_uncertainty.apply(lambda x: np.log(x))
# df['diff'] = df['diff'].apply(lambda x: np.abs(x))
# df.ep_uncertainty = df.ep_uncertainty.apply(lambda x: x+4.5)# 
# =============================================================================

# =============================================================================
# df_dist.info()
# fig4, ax = plt.subplots()
# plt.errorbar(df_dist['predicted_distance_km'].values, 
#              df_dist['dist_al_uncertainty'].values, 
#              yerr= df_dist['dist_al_uncertainty'].values, 
#              fmt='ro',              
#              alpha=0.4, 
#              ecolor='black', 
#              capthick=2)
# ax.set_xlabel('Distance')
# ax.set_ylabel('Uncertainty')
# 
# fig4.savefig('7.png') 
# 
# 
# 
# fig4, ax = plt.subplots()
# plt.errorbar(df_dist['predicted_distance_km'].values, 
#              df_dist['true_distance_km'].values, 
#              yerr= df_dist['dist_comb_uncertainty'].values, 
#              fmt='ro',              
#              alpha=0.3, 
#              ecolor='black')
# ax.scatter(df_dist['true_distance_km'].values,
#            df_dist['predicted_distance_km'].values, 
#            alpha = 0.4, 
#            facecolors='none',
#            edgecolors='b')
# 
# ax.set_xlabel('Measured Magnitude')
# ax.set_ylabel('Predicted Magnitude')
# 
# fig4.savefig('8.png') 
# 
# =============================================================================








######################  Event based results
######################
df_mulistaition = pd.read_csv("Xevent_based_predictions.csv") 
df_mulistaition.info()
df_mulistaition = df_mulistaition[df_mulistaition.location_error < 100]

fig = plt.figure(figsize=(15, 4)) 
ax = fig.add_subplot(131)  
textstr_av = ('Mean = %.2f' % round(np.mean(df_mulistaition['location_error']), 2))
textstr_std = ('Std = %.2f' % round(np.std(df_mulistaition['location_error']), 2))
props = dict(boxstyle='round', facecolor='crimson', alpha=0.3)
plt.title('Location Error')
df_mulistaition['location_error'].plot(kind='hist', bins = 50,  logy= False, facecolor='crimson', alpha=0.9, edgecolor='black', linewidth=1.2)
plt.xlabel("Location Error km",fontweight='bold',fontsize=12,  color= 'gray' )
plt.text(60, 6000, textstr_av, fontsize=13, verticalalignment='top', bbox=props)
plt.text(60, 5200, textstr_std, fontsize=13, verticalalignment='top', bbox=props)
plt.tight_layout()

ax = fig.add_subplot(132)  
textstr_av = ('Mean = %.2f' % round(np.mean(df_mulistaition['origin_time_error']), 2))
textstr_std = ('Std = %.2f' % round(np.std(df_mulistaition['origin_time_error']), 2))
props = dict(boxstyle='round', facecolor='crimson', alpha=0.3)
plt.title('Origin Time Error')
df_mulistaition['origin_time_error'].plot(kind='hist', bins = 50,  logy= False, facecolor='crimson', alpha=0.9, edgecolor='black', linewidth=1.2)
plt.xlabel("Origin Time Error s",fontweight='bold',fontsize=12,  color= 'gray' )
plt.ylabel("")
plt.text(10, 7800, textstr_av, fontsize=13, verticalalignment='top', bbox=props)
plt.text(10, 6800, textstr_std, fontsize=13, verticalalignment='top', bbox=props)
plt.tight_layout()

ax = fig.add_subplot(133)  
textstr_av = ('Mean = %.2f' % round(np.mean(df_mulistaition['depth_erro']), 2))
textstr_std = ('Std = %.2f' % round(np.std(df_mulistaition['depth_erro']), 2))
props = dict(boxstyle='round', facecolor='crimson', alpha=0.3)
plt.title('Depth Error')
df_mulistaition['depth_erro'].plot(kind='hist', bins = 50,  logy= False, facecolor='crimson', alpha=0.9, edgecolor='black', linewidth=1.2)
plt.xlabel("Depth Error km",fontweight='bold',fontsize=12,  color= 'gray' )
plt.ylabel("")
plt.text(90, 5100, textstr_av, fontsize=13, verticalalignment='top', bbox=props)
plt.text(90, 4400, textstr_std, fontsize=13, verticalalignment='top', bbox=props)
plt.tight_layout()

plt.savefig('./Xfigs/MS_location_error.png', dpi=400)
plt.show()







ec ='k'
fc = 'turquoise'

csv_file = "Xevent_based_predictions.csv"
df_mulistaition = pd.read_csv(csv_file) 

fig, axes = plt.subplots(5, 3, figsize=(21, 22))
df_mulistaition = pd.read_csv(csv_file)
df_mulistaition = df_mulistaition[df_mulistaition.location_error < 100] 
df_mulistaition.plot(kind='scatter', y = 'location_error', x ='gt_mag', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[0,0])
axes[0,0].set_ylabel('Location\n Estimation Error km',fontweight='bold')
axes[0,0].set_xlabel("Magnitude M",fontweight='bold')

df_mulistaition.plot(kind='scatter', y = 'location_error', x ='gt_depth', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[1,0])
axes[1,0].set_ylabel('Location Error km',fontweight='bold')
axes[1,0].set_xlabel("Depth km",fontweight='bold')

df_mulistaition.plot(kind='scatter', y = 'location_error', x='pr_unc', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[2,0])
axes[2,0].set_ylabel("Location Error km",fontweight='bold')
axes[2,0].set_xlabel('Predicted Location Uncertainty',fontweight='bold')

df_mulistaition = pd.read_csv(csv_file)
df_mulistaition = df_mulistaition[df_mulistaition.location_error < 100] 
df_mulistaition = df_mulistaition[df_mulistaition.gt_H_unc != 'None']
df_mulistaition.gt_H_unc = df_mulistaition.gt_H_unc.apply(lambda x: float(x))
df_mulistaition = df_mulistaition[df_mulistaition.gt_H_unc != 0]
df_mulistaition.plot(kind='scatter', y = 'location_error', x='gt_H_unc', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[3,0])
axes[3,0].set_ylabel('Location Error km',fontweight='bold')
axes[3,0].set_xlabel('Horizontal Uncertainty km',fontweight='bold')

df_mulistaition.plot(kind='scatter', y = 'pr_unc', x ='gt_H_unc', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[4,0])
axes[4,0].set_ylabel("Predicted\n Location Uncertainty",fontweight='bold')
axes[4,0].set_xlabel("Horizontal Uncertainty km",fontweight='bold')


ec ='k'
fc = 'r'

df_mulistaition = pd.read_csv(csv_file)
df_mulistaition = df_mulistaition[df_mulistaition.origin_time_error < 3] 
df_mulistaition.plot(kind='scatter', y = 'origin_time_error', x ='gt_mag', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[0,1])
axes[0,1].set_ylabel('Origin-Time\n Estimation Error',fontweight='bold')
axes[0,1].set_xlabel("Magnitude M",fontweight='bold')

df_mulistaition.plot(kind='scatter', y = 'origin_time_error', x ='gt_depth', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[1,1])
axes[1,1].set_ylabel('origin_time_error',fontweight='bold')
axes[1,1].set_xlabel("Depth km",fontweight='bold')

df_mulistaition.plot(kind='scatter', y = 'origin_time_error', x='pr_origin_unc', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[2,1])
axes[2,1].set_ylabel("Origin-Time\n Estimation Error",fontweight='bold')
axes[2,1].set_xlabel('Predicted Origin Time Uncertainty',fontweight='bold')

df_mulistaition = pd.read_csv(csv_file)
df_mulistaition = df_mulistaition[df_mulistaition.origin_time_error < 3] 
df_mulistaition = df_mulistaition[df_mulistaition.gt_origin_unc != 'None']
df_mulistaition.gt_origin_unc = df_mulistaition.gt_origin_unc.apply(lambda x: float(x))
df_mulistaition = df_mulistaition[df_mulistaition.gt_origin_unc != 0]
df_mulistaition = df_mulistaition[df_mulistaition.gt_origin_unc < 6]
df_mulistaition.plot(kind='scatter', y = 'origin_time_error', x='gt_origin_unc', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[3,1])
axes[3,1].set_ylabel('Origin-Time\n Estimation Error',fontweight='bold')
axes[3,1].set_xlabel('Cataloged Origin-Time Uncertainty',fontweight='bold')

df_mulistaition.plot(kind='scatter', y = 'pr_origin_unc', x ='gt_origin_unc', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[4,1])
axes[4,1].set_ylabel("Predicted\n Origin-Time Uncertainty",fontweight='bold')
axes[4,1].set_xlabel("Cataloged Origin-Time Uncertainty",fontweight='bold')

ec ='k'
fc = 'b'
df_mulistaition = pd.read_csv(csv_file)
df_mulistaition = df_mulistaition[df_mulistaition.depth_erro < 80] 
df_mulistaition.plot(kind='scatter', y = 'depth_erro', x ='gt_mag', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[0,2])
axes[0,2].set_ylabel('Depth\n Estimation Error km',fontweight='bold')
axes[0,2].set_xlabel("Magnitude M",fontweight='bold')

df_mulistaition.plot(kind='scatter', y = 'depth_erro', x ='gt_depth', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[1,2])
axes[1,2].set_ylabel('Depth\n Estimation Error km',fontweight='bold')
axes[1,2].set_xlabel("Depth km",fontweight='bold')

df_mulistaition = pd.read_csv(csv_file)
df_mulistaition = df_mulistaition[df_mulistaition.depth_erro < 80] 
df_mulistaition = df_mulistaition[df_mulistaition.gt_depth_unc != 'None']
df_mulistaition = df_mulistaition[df_mulistaition.gt_depth_unc != 'nan']
df_mulistaition = df_mulistaition[df_mulistaition.gt_depth_unc != None]
df_mulistaition.gt_depth_unc = df_mulistaition.gt_depth_unc.astype('float')
df_mulistaition = df_mulistaition[df_mulistaition.gt_depth_unc != 0]
df_mulistaition.plot(kind='scatter', y = 'depth_erro', x='pr_depth_unc', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[2,2])
axes[2,2].set_ylabel("Depth\n Estimation Error km",fontweight='bold')
axes[2,2].set_xlabel('Predicted Depth Uncertainty',fontweight='bold')

df_mulistaition = df_mulistaition[df_mulistaition.gt_depth_unc >1] 
df_mulistaition.plot(kind='scatter', y = 'depth_erro', x='gt_depth_unc', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[3,2])
axes[3,2].set_ylabel('Depth\n Estimation Error km',fontweight='bold')
axes[3,2].set_xlabel('Cataloged Depth Uncertainty',fontweight='bold')

df_mulistaition.plot(kind='scatter', y = 'pr_depth_unc', x ='gt_depth_unc', alpha= 0.3, c=fc ,edgecolors= ec, ax=axes[4,2])
axes[4,2].set_ylabel("Predicted\n Depth Uncertainty",fontweight='bold')
axes[4,2].set_xlabel("Cataloged Depth Uncertainty",fontweight='bold')

plt.rcParams.update({'font.size': 14})

plt.savefig('./Xfigs/multistation_correlations.jpeg', dpi=400)
plt.show()







########### Global
csv_file = "Xevent_based_predictions.csv"
df_mulistaition = pd.read_csv(csv_file) 


  
lon1, lat1 = (df_mulistaition.gt_lon.values, df_mulistaition.gt_lat.values)
lon2, lat2 = (df_mulistaition.av_pr_lon.values, df_mulistaition.av_pr_lat.values)
    
fig = plt.figure(figsize=(10, 8)) 
ax = fig.add_subplot(211)             
m = Basemap(llcrnrlon = min(lon1)-1, llcrnrlat = min(lat1)-1,
                urcrnrlon = max(lon1)+1, urcrnrlat = max(lat1)+1,
                lat_0 = 0, 
                lon_0 = 0,
                resolution='i', projection='mill')  # resolution: i, h, l     projection='lcc', "mill"

m.drawmapboundary(fill_color='#85A6D9')
m.fillcontinents(color='lavender',lake_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawcountries(color='#6D5F47', linewidth=.4)                
m.drawmeridians(np.arange(-180,180, 30), color='#bbbbbb', labels=[0,0,0,1],fontsize=8,linewidth=0.5)
m.drawparallels(np.arange(-90, 90, 30), color='#bbbbbb', labels=[1,0,0,0], fontsize=8,linewidth=0.5)                
                
#    m.shadedrelief()  
#    m.etopo()
#    m.fillcontinents()

m.scatter(lon1, 
          lat1, 
          latlon = True, 
          marker = "o", 
          color = "r", 
          alpha = 0.3,
          s = 20,
          zorder=5,
          label="Catalog"
              )  

plt.legend(loc='upper right', borderaxespad=0.2)


ax = fig.add_subplot(212)             
m = Basemap(llcrnrlon = min(lon1)-1, llcrnrlat = min(lat1)-1,
                urcrnrlon = max(lon1)+1, urcrnrlat = max(lat1)+1,
                lat_0 = 0, 
                lon_0 = 0,
                resolution='i', projection='mill')  # resolution: i, h, l     projection='lcc', "mill"

m.drawmapboundary(fill_color='#85A6D9')
m.fillcontinents(color='lavender',lake_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawcountries(color='#6D5F47', linewidth=.4)
m.drawmeridians(np.arange(-180,180, 30), color='#bbbbbb', labels=[0,0,0,1],fontsize=8,linewidth=0.5)
m.drawparallels(np.arange(-90, 90, 30), color='#bbbbbb', labels=[1,0,0,0], fontsize=8,linewidth=0.5) 
#    m.shadedrelief()  
#    m.etopo()
#    m.fillcontinents()

m.scatter(lon2, 
          lat2, 
          latlon = True, 
          marker = "o", 
          color = "b", 
          alpha = 0.3,
          s = 20,
          zorder=5,
          label="Predicted"
              )  

plt.legend(loc='upper right', borderaxespad=0.2)  
plt.savefig('./Xfigs/predictions_global.png', dpi=400)
plt.show()
    
    


########### Alaska
df_mulistaition = pd.read_csv("Xevent_based_predictions.csv") 

df_mulistaition = df_mulistaition[df_mulistaition.gt_lon > -180.0]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lon < -130.0]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lat < 64.0]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lat > 50.0]

df_mulistaition['gt_bap'].plot(kind='hist')

fig = plt.figure(figsize=(10, 8)) 

ln1, lt1 = (df_mulistaition.gt_lon.values, df_mulistaition.gt_lat.values)
ln2, lt2 = (df_mulistaition.av_pr_lon.values, df_mulistaition.av_pr_lat.values)   
            
m = Basemap(llcrnrlon = min(ln1)-0.5, llcrnrlat = min(lt1)-0.5,
                urcrnrlon = max(ln1)+0.5, urcrnrlat = max(lt1)+0.5,
                lat_0 = 0, 
                lon_0 = 0,
                resolution='h', projection='mill')  # resolution: i, h, l     projection='lcc', "mill"

m.drawmapboundary(fill_color='#85A6D9')
m.fillcontinents(color='lavender',lake_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawcountries(color='#6D5F47', linewidth=.4)
m.drawmeridians(np.arange(-180,180, 5), color='#bbbbbb', labels=[0,0,0,1],fontsize=8,linewidth=0.5)
m.drawparallels(np.arange(-90, 90, 2), color='#bbbbbb', labels=[1,0,0,0], fontsize=8,linewidth=0.5) 
m.drawmapscale(-160, 53, 0, 0, 400, barstyle='fancy')
   
lon1, lat1 = m(df_mulistaition.gt_lon.values, df_mulistaition.gt_lat.values)
lon2, lat2 = m(df_mulistaition.av_pr_lon.values, df_mulistaition.av_pr_lat.values)
    
pts = np.c_[lon1, lat1, lon2, lat2].reshape(len(lon1), 2, 2)
    
plt.gca().add_collection(LineCollection(pts, color="k", linewidth=0.4, alpha=0.7))
   
m.plot(lon1, lat1, marker="o", color="r", ls="", label="Cataloged", alpha=0.4,linewidth=1, markersize=3)
m.plot(lon2, lat2, marker="o", color="b", ls="", label="Predicted", alpha=0.4,linewidth=1, markersize=3)
    
plt.legend(loc='upper right', borderaxespad=0.2)
plt.title('Alaska')
plt.savefig('./Xfigs/predictions_alaska.png', dpi=400)
plt.show()
    


def date_convertor(r):  
    try:
        new_r = int(r)
    except Exception:
        new_r = None
    return new_r


df_mulistaition['gt_bap'] = df_mulistaition['gt_bap'].apply(lambda row : date_convertor(row)) 


########### Kansas
csv_file = "Xevent_based_predictions.csv"
df_mulistaition = pd.read_csv(csv_file) 

df_mulistaition = df_mulistaition[df_mulistaition.gt_lon > -98.5]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lon < -96.0]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lat < 38.0]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lat > 36.7]
df_mulistaition.info()


fig = plt.figure(figsize=(10, 8)) 

ln1, lt1 = (df_mulistaition.gt_lon.values, df_mulistaition.gt_lat.values)
ln2, lt2 = (df_mulistaition.av_pr_lon.values, df_mulistaition.av_pr_lat.values)   
            
m = Basemap(llcrnrlon = min(ln1), llcrnrlat = min(lt1),
                urcrnrlon = max(ln1), urcrnrlat = max(lt1),
                lat_0 = 0, 
                lon_0 = 0,
                resolution='h', projection='mill')  # resolution: i, h, l     projection='lcc', "mill"

m.drawmapboundary(fill_color='#85A6D9')
m.fillcontinents(color='lavender',lake_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawcountries(color='#6D5F47', linewidth=.4)
m.drawmeridians(np.arange(-180,180, 0.5), color='#bbbbbb', labels=[0,0,0,1],fontsize=8,linewidth=0.5)
m.drawparallels(np.arange(-90, 90, 0.5), color='#bbbbbb', labels=[1,0,0,0], fontsize=8,linewidth=0.5) 
m.drawmapscale(-98.2, 36.8, 0, 0, 10, barstyle='fancy')
   
lon1, lat1 = m(df_mulistaition.gt_lon.values, df_mulistaition.gt_lat.values)
lon2, lat2 = m(df_mulistaition.av_pr_lon.values, df_mulistaition.av_pr_lat.values)
    
pts = np.c_[lon1, lat1, lon2, lat2].reshape(len(lon1), 2, 2)
    
plt.gca().add_collection(LineCollection(pts, color="k", linewidth=0.4, alpha=0.7))
   
m.plot(lon1, lat1, marker="o", color="r", ls="", label="Cataloged", alpha=0.4,linewidth=1, markersize=3)
m.plot(lon2, lat2, marker="o", color="b", ls="", label="Predicted", alpha=0.4,linewidth=1, markersize=3)
    
plt.legend(loc='upper right', borderaxespad=0.2)
plt.title('Southern Kansas')
plt.savefig('./Xfigs/predictions_Kansas.png', dpi=400)

plt.show()
    








########### North Cal
csv_file = "Xevent_based_predictions.csv"
df_mulistaition = pd.read_csv(csv_file) 

df_mulistaition.info()

df_mulistaition = df_mulistaition[df_mulistaition.gt_lon < -120.0]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lon > -123.0]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lat < 39.0]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lat > 36.0]


fig = plt.figure(figsize=(10, 8)) 

ln1, lt1 = (df_mulistaition.gt_lon.values, df_mulistaition.gt_lat.values)
ln2, lt2 = (df_mulistaition.av_pr_lon.values, df_mulistaition.av_pr_lat.values)   
            
m = Basemap(llcrnrlon = min(ln1)-0.2, llcrnrlat = min(lt1)-0.2,
                urcrnrlon = max(ln1)+0.2, urcrnrlat = max(lt1)+0.2,
                lat_0 = 0, 
                lon_0 = 0,
                resolution='h', projection='mill')  # resolution: i, h, l     projection='lcc', "mill"

m.drawmapboundary(fill_color='#85A6D9')
m.fillcontinents(color='lavender',lake_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawcountries(color='#6D5F47', linewidth=.4)
m.drawmeridians(np.arange(-180,180, 1), color='#bbbbbb', labels=[0,0,0,1],fontsize=8,linewidth=0.5)
m.drawparallels(np.arange(-90, 90, 1), color='#bbbbbb', labels=[1,0,0,0], fontsize=8,linewidth=0.5) 
                
# =============================================================================
# m.drawmeridians(np.arange(-180,180, 1), labels=[0,0,0,1],fontsize=8,linewidth=0.5)
# m.drawparallels(np.arange(-90, 90, 1),labels=[1,0,0,0],fontsize=8,linewidth=0.5)
# m.drawcoastlines(linewidth=0.5)
# m.drawcountries(linewidth=1)
# m.drawstates(linewidth=0.3)
# m.shadedrelief()  
# =============================================================================

m.drawmapscale(-122.6, 36.1, 0, 0, 50, barstyle='fancy')
   
lon1, lat1 = m(df_mulistaition.gt_lon.values, df_mulistaition.gt_lat.values)
lon2, lat2 = m(df_mulistaition.av_pr_lon.values, df_mulistaition.av_pr_lat.values)
    
pts = np.c_[lon1, lat1, lon2, lat2].reshape(len(lon1), 2, 2)
    
plt.gca().add_collection(LineCollection(pts, color="k", linewidth=0.4, alpha=0.7))
   
m.plot(lon1, lat1, marker="o", color="r", ls="", label="Cataloged", alpha=0.4,linewidth=1, markersize=3)
m.plot(lon2, lat2, marker="o", color="b", ls="", label="Predicted", alpha=0.4,linewidth=1, markersize=3)
    
plt.legend(loc='upper right', borderaxespad=0.2)
plt.title('North California')
plt.savefig('./Xfigs/predictions_NorthCal1.png', dpi=400)
plt.show()
    
    









########### North Cal
csv_file = "Xevent_based_predictions.csv"
df_mulistaition = pd.read_csv(csv_file) 

df_mulistaition.info()

df_mulistaition = df_mulistaition[df_mulistaition.gt_lon < -120.0]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lon > -123.0]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lat < 39.0]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lat > 36.0]


fig = plt.figure(figsize=(10, 8)) 

ln1, lt1 = (df_mulistaition.gt_lon.values, df_mulistaition.gt_lat.values)
ln2, lt2 = (df_mulistaition.av_pr_lon.values, df_mulistaition.av_pr_lat.values)   
            
m = Basemap(llcrnrlon = min(ln1)-0.2, llcrnrlat = min(lt1)-0.2,
                urcrnrlon = max(ln1)+0.2, urcrnrlat = max(lt1)+0.2,
                lat_0 = 0, 
                lon_0 = 0,
                resolution='h', projection='mill')  # resolution: i, h, l     projection='lcc', "mill"

# =============================================================================
# m.drawmapboundary(fill_color='#85A6D9')
# m.fillcontinents(color='lavender',lake_color='#85A6D9')
# m.drawcoastlines(color='#6D5F47', linewidth=.4)
# m.drawcountries(color='#6D5F47', linewidth=.4)
# m.drawmeridians(np.arange(-180,180, 1), color='#bbbbbb', labels=[0,0,0,1],fontsize=8,linewidth=0.5)
# m.drawparallels(np.arange(-90, 90, 1), color='#bbbbbb', labels=[1,0,0,0], fontsize=8,linewidth=0.5) 
# =============================================================================
                
m.drawmeridians(np.arange(-180,180, 1), labels=[0,0,0,1],fontsize=8,linewidth=0.5)
m.drawparallels(np.arange(-90, 90, 1),labels=[1,0,0,0],fontsize=8,linewidth=0.5)
m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.3)
m.shadedrelief()  

m.drawmapscale(-122.6, 36.1, 0, 0, 50, barstyle='fancy')
   
lon1, lat1 = m(df_mulistaition.gt_lon.values, df_mulistaition.gt_lat.values)
lon2, lat2 = m(df_mulistaition.av_pr_lon.values, df_mulistaition.av_pr_lat.values)
    
pts = np.c_[lon1, lat1, lon2, lat2].reshape(len(lon1), 2, 2)
    
plt.gca().add_collection(LineCollection(pts, color="k", linewidth=0.4, alpha=0.7))
   
m.plot(lon1, lat1, marker="o", color="r", ls="", label="Cataloged", alpha=0.4,linewidth=1, markersize=3)
m.plot(lon2, lat2, marker="o", color="b", ls="", label="Predicted", alpha=0.4,linewidth=1, markersize=3)
    
plt.legend(loc='upper right', borderaxespad=0.2)
plt.title('North California')
plt.savefig('./Xfigs/predictions_NorthCal2.png', dpi=400)
plt.show()
    
    
   



# =============================================================================
# 
# ########### Global
# csv_file = "Xevent_based_predictions.csv"
# df_mulistaition = pd.read_csv(csv_file) 
# 
#   
# lon1, lat1 = (df_mulistaition.gt_lon.values, df_mulistaition.gt_lat.values)
# lon2, lat2 = (df_mulistaition.av_pr_lon.values, df_mulistaition.av_pr_lat.values)
#     
# fig = plt.figure(figsize=(10, 8)) 
# ax = fig.add_subplot(211)             
# m = Basemap(llcrnrlon = min(lon1)-1, llcrnrlat = min(lat1)-1,
#                 urcrnrlon = max(lon1)+1, urcrnrlat = max(lat1)+1,
#                 lat_0 = 0, 
#                 lon_0 = 0,
#                 resolution='h', projection='mill')  # resolution: i, h, l     projection='lcc', "mill"
# 
# m.drawmeridians(np.arange(-180,180, 30), labels=[0,0,0,1],fontsize=8,linewidth=0.5)
# m.drawparallels(np.arange(-90, 90, 30),labels=[1,0,0,0],fontsize=8,linewidth=0.5)
# m.drawcoastlines(linewidth=0.5)
# m.drawcountries(linewidth=1)
# m.drawstates(linewidth=0.3)
# m.shadedrelief()  
# 
# m.scatter(lon1, 
#           lat1, 
#           latlon = True, 
#           marker = "o", 
#           color = "r", 
#           alpha = 0.3,
#           s = 20,
#           zorder=5,
#           label="Catalog"
#               )  
# 
# plt.legend(loc='upper right', borderaxespad=0.2)
# 
# 
# ax = fig.add_subplot(212)             
# m = Basemap(llcrnrlon = min(lon1)-1, llcrnrlat = min(lat1)-1,
#                 urcrnrlon = max(lon1)+1, urcrnrlat = max(lat1)+1,
#                 lat_0 = 0, 
#                 lon_0 = 0,
#                 resolution='h', projection='merc')  # resolution: i, h, l     projection='lcc', "mill"
# 
# m.drawmeridians(np.arange(-180,180, 30), labels=[0,0,0,1],fontsize=8,linewidth=0.5)
# m.drawparallels(np.arange(-90, 90, 30),labels=[1,0,0,0],fontsize=8,linewidth=0.5)
# m.drawcoastlines(linewidth=0.5)
# m.drawcountries(linewidth=1)
# m.drawstates(linewidth=0.3)
# m.shadedrelief()  
# 
# m.scatter(lon2, 
#           lat2, 
#           latlon = True, 
#           marker = "o", 
#           color = "b", 
#           alpha = 0.3,
#           s = 20,
#           zorder=5,
#           label="Predicted"
#               )  
# 
# plt.legend(loc='upper right', borderaxespad=0.2)  
# plt.show()
#     
# =============================================================================
