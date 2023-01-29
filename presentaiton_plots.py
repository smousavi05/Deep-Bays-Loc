#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 22:16:44 2020

@author: mostafamousavi
"""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection


df_mulistaition = pd.read_csv("Xevent_based_predictions.csv") 
df_mulistaition.info()
#df_mulistaition = df_mulistaition[df_mulistaition.location_error < 100]


########### North Cal
csv_file = "Xevent_based_predictions.csv"
df_mulistaition = pd.read_csv(csv_file) 

df_mulistaition.info()

df_mulistaition = df_mulistaition[df_mulistaition.gt_lon < -122.5]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lon > -123.0]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lat < 39.0]
df_mulistaition = df_mulistaition[df_mulistaition.gt_lat > 38.5]


fig = plt.figure(figsize=(10, 8)) 

ln1, lt1 = (df_mulistaition.gt_lon.values, df_mulistaition.gt_lat.values)
ln2, lt2 = (df_mulistaition.av_pr_lon.values, df_mulistaition.av_pr_lat.values)   
            
m = Basemap(llcrnrlon = min(ln1)-0.2, llcrnrlat = min(lt1)-0.2,
                urcrnrlon = max(ln1)+0.2, urcrnrlat = max(lt1)+0.2,
                lat_0 = 0, 
                lon_0 = 0,
                resolution='l', projection='mill')  # resolution: i, h, l     projection='lcc', "mill"

m.drawmapboundary(fill_color='#85A6D9')
m.fillcontinents(color='lavender',lake_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawcountries(color='#6D5F47', linewidth=.4)
m.drawmeridians(np.arange(-180,180, 1), color='#bbbbbb', labels=[0,0,0,1],fontsize=8,linewidth=0.5)
m.drawparallels(np.arange(-90, 90, 1), color='#bbbbbb', labels=[1,0,0,0], fontsize=8,linewidth=0.5) 
  
# =============================================================================
                
# =============================================================================
# m.drawmeridians(np.arange(-180,180, 1), labels=[0,0,0,1],fontsize=8,linewidth=0.5)
# m.drawparallels(np.arange(-90, 90, 1),labels=[1,0,0,0],fontsize=8,linewidth=0.5)
# m.drawcoastlines(linewidth=0.5)
# m.drawcountries(linewidth=1)
# m.drawstates(linewidth=0.3)
# m.shadedrelief()  
# 
# =============================================================================
m.drawmapscale(-122.6, 36.1, 0, 0, 50, barstyle='fancy')
   
lon1, lat1 = m(df_mulistaition.gt_lon.values, df_mulistaition.gt_lat.values)
lon2, lat2 = m(df_mulistaition.av_pr_lon.values, df_mulistaition.av_pr_lat.values)
    
pts = np.c_[lon1, lat1, lon2, lat2].reshape(len(lon1), 2, 2)
    
plt.gca().add_collection(LineCollection(pts, color="k", linewidth=0.4, alpha=0.7))

m.plot(-122.795303, 38.807968, marker="^", color="m", ls="", label="Cataloged", alpha=0.4,linewidth=1, markersize=4)

m.plot(lon1, lat1, marker="o", color="r", ls="", label="Cataloged", alpha=0.4,linewidth=1, markersize=1)
m.plot(lon2, lat2, marker="o", color="b", ls="", label="Predicted", alpha=0.4,linewidth=1, markersize=1)
    
plt.legend(loc='upper right', borderaxespad=0.2)
plt.title('North California')
plt.savefig('./Xfigs/presentation.png', dpi=400)
plt.show()
    
    