#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 17:44:51 2019

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
    



################################  regression results
csv_file = "./tcn_dist2_pT_6_outputs/results.csv"
df_dist = pd.read_csv(csv_file) 

#@@@@@@@@ baz
csv_file = "./az_regression_cov3_outputs/results.csv"
df_az = pd.read_csv(csv_file) 
df_az = df_az[df_az.ep_uncertainty < 12]


######################### checking single station vs multi station
    
def depth_estimator(dist, ptravel, Vp = 5.7):
    a = dist
    b = ptravel*Vp
    
    if a <= b:
        aa = a
        hh = b
    else:
        hh = a
        aa = b        
    alpha = np.arccos(aa/hh)
    return round(hh*np.sin(alpha), 2) 
               
def baz_variation(baz, baz_unc, dist, lon, lat):
    var = dist*np.cos((baz_unc/2))
    if baz <= 90:
        angl1 = 360-baz
        angl2 = baz + 90
    elif baz > 90 and baz <= 270:
        angl1 = baz - 90
        angl2 = baz + 90        
    elif baz > 270:
        angl1 = baz - 90
        angl2 = baz - 270   
        
    x1_lat, x1_lon, alpha21 = vinc_pt(f = 1/298.257223563,
                                       a = 6378137, 
                                       phi1 = lat, 
                                       lembda1 = lon, 
                                       alpha12 = angl1, 
                                       s = var*1000)    

    x2_lat, x2_lon, alpha21 = vinc_pt(f = 1/298.257223563,
                                       a = 6378137, 
                                       phi1 = lat, 
                                       lembda1 = lon, 
                                       alpha12 = angl2, 
                                       s = var*1000)  
    
    return x1_lat, x1_lon, x2_lat, x2_lon



def dist_variation(dist, dist_unc, baz, lon, lat):
    var = dist_unc/2        
    x1_lat, x1_lon, alpha21 = vinc_pt(f = 1/298.257223563,
                                       a = 6378137, 
                                       phi1 = lat, 
                                       lembda1 = lon, 
                                       alpha12 = baz, 
                                       s = abs(dist-var)*1000)    

    x2_lat, x2_lon, alpha21 = vinc_pt(f = 1/298.257223563,
                                       a = 6378137, 
                                       phi1 = lat, 
                                       lembda1 = lon, 
                                       alpha12 = baz, 
                                       s = abs(dist+var)*1000
                                       )  
    
    return x1_lat, x1_lon, x2_lat, x2_lon





df_ref = pd.read_csv("../../STEAD/dataset/metadata.csv") 

#az_list = df_mulistaition['Event id'].tolist()
#df_az = df_az[df_az['source_id'].isin(az_list)]

uniq_ins = df_az.source_id.unique()
event = []
number_of_stations = []
for ii in range(len(uniq_ins)):
      
    if sum(n == str(uniq_ins[ii]) for n in df_az.source_id) > 0:
        print(str(uniq_ins[ii]), sum(n == str(uniq_ins[ii]) for n in df_az.source_id))
        event.append(str(uniq_ins[ii]))
        number_of_stations.append(np.round(sum(n == str(uniq_ins[ii]) for n in df_az.source_id)/sum(n != None for n in df_az.source_id.dropna()), 2))
  
#     
# =============================================================================
csvfile = open('event_based_predictions.csv', 'w')          
output_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
output_writer.writerow(['Event id', 'number_of_stations','location_error', 'origin_time_error', 'depth_erro',
                        'gt_depth', 'gt_depth_unc',
                        'pr_depth', 'pr_depth_unc', 
                        'gt_H_unc', 
                        'gt_mag', 
                        'gt_sourceError', 
                        'gt_gap',                         
                        'pr_unc',
                        'av_pr_lon', 'av_pr_lat', 
                        'gt_lon', 'gt_lat',
                        'gt_origin', 'gt_origin_unc',
                        'pr_origin', 'pr_origin_unc',
                        'pr_dist_combUNC', 'pr_dist_alUNC', 'pr_dist_epUNC',
                        'pr_pt_combUNC', 'pr_pt_alUNC', 'pr_pt_epUNC', 

                        ])
csvfile.flush()





event = ['ismpkansas70166943']  

                          
for ii in range(len(event)):
    dd = df_az[df_az.source_id == event[ii]]
    stations = dd.trace_name.tolist()
      
    gt_origin_time = []
    gt_origin_uncertainty_sec = []
    gt_source_latitude = []
    gt_source_longitude = [] 
    gt_source_error_sec = []
    gt_source_gap_deg = []
    gt_source_horizontal_uncertainty_km = [] 
    gt_source_depth_km = []
    gt_source_mag = []
    gt_source_depth_uncertainty_km = [] 
    gt_receiver_latitude = []
    gt_receiver_longitude = []
    gt_receiver_elevation_m = [] 
    gt_back_azimuth_deg = []
    gt_source_distance_km = []         
        
    pr_baz = []
    pr_baz_unc = [] 
    pr_diffBAz = []
    
    pr_predicted_distance_km = []
    pr_predicted_distance_km_COMunc = []
    pr_predicted_distance_km_ALunc = []
    pr_predicted_distance_km_EPunc = []  
    pr_diffD = []
    
    pr_predicted_Ptravel_time_s = []
    pr_predicted_Ptravel_time_s_COMunc = []
    pr_predicted_Ptravel_time_s_ALunc = []
    pr_predicted_Ptravel_time_s_EPunc = []
    pr_diffPT = []
    
    pr_source_longitude = []
    pr_source_latitude = []
    gt_p_arrival_sample = []
    gt_trace_start_time = []
    pr_oringin_time = []
    pr_depth = []
    pr_depth_unc = []
    err_location = []
    err_origin_time = []
    err_depth = []
    pr_baz_var = [] 
    pr_dist_var = [] 
    pr_var = []
         
    for st in stations:
        ddd  = df_ref[df_ref.trace_name == st]
        ddd_PrDist  = df_dist[df_dist.trace_name == st]
        ddd_PrBaz  = df_az[df_az.trace_name == st]
    
        gt_p_arrival_sample.append(ddd.p_arrival_sample.values[0])
        gt_trace_start_time.append(ddd.trace_start_time.values[0])
        
        gt_origin_time.append(ddd.source_origin_time.values[0])
        gt_origin_uncertainty_sec.append(ddd.source_origin_uncertainty_sec.values[0])
        
        gt_source_latitude.append(ddd.source_latitude.values[0])
        gt_source_longitude.append(ddd.source_longitude.values[0]) 
        gt_source_error_sec.append(ddd.source_error_sec.values[0])
        gt_source_gap_deg.append(ddd.source_gap_deg.values[0])
        gt_source_horizontal_uncertainty_km.append(ddd.source_horizontal_uncertainty_km.values[0])
        gt_source_mag.append(ddd.source_magnitude.values[0])
        gt_source_depth_km.append(ddd.source_depth_km.values[0]) 
        gt_source_depth_uncertainty_km.append(ddd.source_depth_uncertainty_km.values[0])
        gt_receiver_latitude.append(ddd.receiver_latitude.values[0])
        gt_receiver_longitude.append(ddd.receiver_longitude.values[0])
        gt_receiver_elevation_m.append(ddd.receiver_elevation_m.values[0])
        gt_back_azimuth_deg.append(ddd.back_azimuth_deg.values[0])
        gt_source_distance_km.append(ddd.source_distance_km.values[0])
        
        pr_baz.append(ddd_PrBaz.predicted_BAZ.values[0])
        pr_baz_unc.append(ddd_PrBaz.ep_uncertainty.values[0]/2) 
        
        pr_predicted_distance_km.append(ddd_PrDist.predicted_distance_km.values[0])
        pr_predicted_distance_km_COMunc.append(np.sqrt(abs(ddd_PrDist.dist_comb_uncertainty.values[0])))
        pr_predicted_distance_km_ALunc.append(np.sqrt(abs(ddd_PrDist.dist_al_uncertainty.values[0])))
        pr_predicted_distance_km_EPunc.append(ddd_PrDist.dist_ep_uncertainty.values[0])
        
        pr_predicted_Ptravel_time_s.append(ddd_PrDist.predicted_Ptravel_time_s.values[0])        
        pr_predicted_Ptravel_time_s_COMunc.append(np.sqrt(abs(ddd_PrDist.pt_comb_uncertainty.values[0])))
        pr_predicted_Ptravel_time_s_ALunc.append(np.sqrt(abs(ddd_PrDist.pt_al_uncertainty.values[0])))
        pr_predicted_Ptravel_time_s_EPunc.append(ddd_PrDist.pt_ep_uncertainty.values[0]) 
        
        predicted_EQlat, predicted_EQlon, alpha21 = vinc_pt(f = 1/298.257223563,
                                                            a = 6378137, 
                                                            phi1 = ddd.receiver_latitude.values[0], 
                                                            lembda1 = ddd.receiver_longitude.values[0], 
                                                            alpha12 = ddd_PrBaz.predicted_BAZ.values[0], 
                                                            s = ddd_PrDist.predicted_distance_km.values[0]*1000 )    
    
        pr_source_longitude.append(round(predicted_EQlon, 4))
        pr_source_latitude.append(round(predicted_EQlat, 4))   
        
        
           
        x1_lat, x1_lon, x2_lat, x2_lon = baz_variation(ddd_PrBaz.predicted_BAZ.values[0], 
                                                       ddd_PrBaz.ep_uncertainty.values[0]/2, 
                                                       ddd_PrDist.predicted_distance_km.values[0], 
                                                       predicted_EQlon, 
                                                       predicted_EQlat)
        g = pyproj.Geod(ellps="WGS84")   
        _, _, baz_var = g.inv(x1_lon, x1_lat, x2_lon, x2_lat)
        pr_baz_var.append(baz_var)  
        
        baz_shift = degrees2kilometers(locations2degrees(x1_lon, x1_lat, x2_lon, x2_lat))
            
        x1_lat, x1_lon, x2_lat, x2_lon = dist_variation(ddd_PrDist.predicted_distance_km.values[0], 
                                                        np.sqrt(abs(ddd_PrDist.dist_comb_uncertainty.values[0])), 
                                                        ddd_PrBaz.predicted_BAZ.values[0], 
                                                        predicted_EQlon, 
                                                        predicted_EQlat)
        
        g = pyproj.Geod(ellps="WGS84")   
        _, _, dist_var = g.inv(x1_lon, x1_lat, x2_lon, x2_lat)    
        pr_dist_var.append(dist_var) 
        dist_shift = degrees2kilometers(locations2degrees(x1_lon, x1_lat, x2_lon, x2_lat))
        pr_var.append((baz_shift+dist_shift)/2)  
        
        start_time = datetime.strptime(ddd.trace_start_time.values[0], '%Y-%m-%d %H:%M:%S.%f')
        p_time = start_time+timedelta(seconds = ddd.p_arrival_sample.values[0]/100)
        pr_oring = p_time - timedelta(seconds= float(ddd_PrDist.predicted_Ptravel_time_s.values[0]))
        gt_origin = datetime.strptime(ddd.source_origin_time.values[0], '%Y-%m-%d %H:%M:%S.%f')
        pr_oringin_time.append(pr_oring)
    
        
        pr_depth_mean = depth_estimator(ddd_PrDist.predicted_distance_km.values[0], 
                                        ddd_PrDist.predicted_Ptravel_time_s.values[0], Vp = 5.7)
        pr_depth_upper = depth_estimator(ddd_PrDist.predicted_distance_km.values[0], 
                                         ddd_PrDist.predicted_Ptravel_time_s.values[0]+ np.sqrt(abs(ddd_PrDist.pt_comb_uncertainty.values[0])), Vp = 5.7)
        pr_depth_lower = depth_estimator(ddd_PrDist.predicted_distance_km.values[0], 
                                         ddd_PrDist.predicted_Ptravel_time_s.values[0] - np.sqrt(abs(ddd_PrDist.pt_comb_uncertainty.values[0])), Vp = 5.7)
        pr_depth_std = (pr_depth_upper - pr_depth_lower)
        
        pr_depth.append(pr_depth_mean)
        pr_depth_unc.append(abs(np.round(pr_depth_std, 2)))
        err_l = locations2degrees(ddd.source_longitude.values[0], ddd.source_latitude.values[0], round(predicted_EQlon,4), round(predicted_EQlat, 4))
        err_l = degrees2kilometers(err_l)
        err_location.append(round(err_l, 2))
        
        tdif = pr_oring - gt_origin    
        err_origin_time.append(tdif.total_seconds())
        dperr = float(ddd.source_depth_km.values[0]) - pr_depth_mean
        err_depth.append(np.round(dperr, 2))
    
        dff = pd.DataFrame({"lon1" : gt_receiver_longitude,
                            "lat1" : gt_receiver_latitude,
                            "lon2" : pr_source_longitude,
                            "lat2" : pr_source_latitude})
         
    
    
    av_location_error = '%.1f' % round(np.average(err_location), 2) 
    av_originT_error = '%.1f' % abs(round(np.average(err_origin_time), 2)) 
    av_depth_erro = '%.1f' % abs(round(np.average(err_depth), 2)) 

    av_gt_depth = gt_source_depth_km[0] 
    av_gt_depth_unc = gt_source_depth_uncertainty_km[0]

    av_pr_depth = '%.1f' % round(np.average(pr_depth), 2) 
    av_pr_depth_unc = '%.1f' % abs(round(np.average(pr_depth_unc), 2))

    av_gt_Hunc = gt_source_horizontal_uncertainty_km[0]
    av_gt_mag = gt_source_mag[0]
    av_gt_sourceError = gt_source_error_sec[0]
    av_gt_gap = gt_source_gap_deg[0]
          
    
    nrm_pr_var = 1-(pr_var/max(pr_var))  
  #  av_pr_unc = '%.1f' % round(np.average([i for i in (nrm_pr_var > 0.20)*pr_var if abs(i) > 0]), 2)
    av_pr_unc = '%.1f' % round(np.average(pr_var), 2)

#    av_pr_longitude = '%.2f' % round(np.average([i for i in (nrm_pr_var > 0.20)*pr_source_longitude if abs(i) > 0]), 2) 
#    av_pr_latitude = '%.2f' % round(np.average([i for i in (nrm_pr_var > 0.20)*pr_source_latitude if abs(i) > 0]), 2) 
    av_pr_longitude = '%.4f' % round(np.average(pr_source_longitude), 4) 
    av_pr_latitude = '%.4f' % round(np.average(pr_source_latitude), 4)     
    av_gt_longitude = '%.4f' % round(np.average(gt_source_longitude), 4) 
    av_gt_latitude = '%.4f' % round(np.average(gt_source_latitude), 4) 
   
    av_gt_origin_time = gt_origin_time[0] 
    av_gt_origin_time_unc = gt_origin_uncertainty_sec[0]
    
    av_pr_origin_time = pr_oringin_time[0].strftime('%Y-%m-%d %H:%M:%S.%f')
    av_pr_origin_time_unc = '%.1f' % round(np.average(pr_predicted_Ptravel_time_s_COMunc), 2)
    
    output_writer.writerow([event[ii], 
                            len(err_location),
                            float(av_location_error), 
                            float(av_originT_error), 
                            float(av_depth_erro), 
                            float(av_gt_depth), 
                            av_gt_depth_unc, 
                            float(av_pr_depth), 
                            float(av_pr_depth_unc),
                            av_gt_Hunc,
                            av_gt_mag,
                            av_gt_sourceError,
                            av_gt_gap,
                            float(av_pr_unc),
                            float(av_pr_longitude), 
                            float(av_pr_latitude), 
                            float(av_gt_longitude),
                            float(av_gt_latitude),
                            av_gt_origin_time,
                            av_gt_origin_time_unc, 
                            av_pr_origin_time, 
                            float(av_pr_origin_time_unc),
                            float(np.average(pr_predicted_distance_km_COMunc)),
                            float(np.average(pr_predicted_distance_km_ALunc)),
                            float(np.average(pr_predicted_distance_km_EPunc)),
                            float(np.average(pr_predicted_Ptravel_time_s_COMunc)),
                            float(np.average(pr_predicted_Ptravel_time_s_ALunc)),
                            float(np.average(pr_predicted_Ptravel_time_s_EPunc))
                                    ])           
    csvfile.flush() 


 
    
    
    
    
    
    
    
    
    
    fig = plt.figure(figsize=(10, 8)) 

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    
    rect_center = [left, 0, width, height]
    rect_right = [width + spacing, 0, 0.4, height]
    
    ax_center = plt.axes(rect_center)
    ax_center.tick_params(direction='in', labelbottom=False, labelleft=False, labelright=False)
        
    room = 1.0
    ln1, lt1 = (dff.lon1.values, dff.lat1.values)
    ln2, lt2 = (dff.lon2.values, dff.lat2.values)
    ln3, lt3 = (gt_source_longitude[0], gt_source_latitude[0]) 
    lns = np.concatenate((ln1, ln2, ln3), axis=None)
    lts = np.concatenate((lt1, lt2, lt3), axis=None)
 
    
    lndiff = max(lns) - min(lns)
    ltdiff = max(lts) - min(lts)
    
    if lndiff/ltdiff < 1.7 and lndiff/ltdiff > 1:
        scl = lndiff/ltdiff
    elif lndiff/ltdiff < 1:
        scl = ltdiff/lndiff          
    else:
        scl = 1
        
        
    m = Basemap(llcrnrlon = min(lns)-0.3, llcrnrlat = min(lts)-0.4,
                urcrnrlon = max(lns)+0.25, urcrnrlat = max(lts)+0.4,
                lat_0 = 0, 
                lon_0 = 0,
                resolution='h', projection='mill')  # resolution: i, h, l     projection='lcc', "mill"

    m.drawmapboundary(fill_color='#85A6D9')
    m.fillcontinents(color='lavender',lake_color='#85A6D9')
    m.drawcoastlines(color='#6D5F47', linewidth=.4)
    m.drawcountries(color='#6D5F47', linewidth=.4)
    m.drawmeridians(np.arange(-180,180, 0.5), color='#bbbbbb', labels=[0,0,0,1],fontsize=8,linewidth=0.5)
    m.drawparallels(np.arange(-90, 90, 0.5), color='#bbbbbb', labels=[1,0,0,0], fontsize=8,linewidth=0.5) 
#    m.shadedrelief()  
#    m.etopo()
#    m.fillcontinents()

    m.drawmapscale(min(lns)-0.0, min(lts)-0.2, 0, 0, 20, barstyle='fancy')

    
    lon1, lat1 = m(dff.lon1.values, dff.lat1.values)
    lon2, lat2 = m(dff.lon2.values, dff.lat2.values)

    
    pts = np.c_[lon1, lat1, lon2, lat2].reshape(len(lon1), 2, 2)
    
    plt.gca().add_collection(LineCollection(pts, color="k", linewidth=0.5))
   
    m.scatter(round(gt_source_longitude[0], 4), 
              round(gt_source_latitude[0], 4), 
              latlon = True, 
              marker = "*", 
              color = "r", 
              alpha = 0.7,
              s = 60,
              zorder=5,
              label="Source"
              )  

    m.plot(lon1, lat1, marker="X", color="k", ls="", label="Reciever", alpha=0.5,linewidth=2, markersize=5)
    m.plot(lon2, lat2, marker="o", color="b", ls="", label="Predictions", alpha=0.5,linewidth=2, markersize=3)
    
    plt.legend(loc= 'upper right', borderaxespad=0.2)
    plt.tight_layout()                
 

    for il in range(len(lon2)): 
        elips = Ellipse(xy=(lon2[il], lat2[il]), 
                        width= pr_baz_var[il],
                        height= pr_dist_var[il],
                        angle = -pr_baz[il],
                        edgecolor='k', 
                        fc='blue', 
                        alpha = 0.20, 
                        lw=1)

        plt.gca().add_patch(elips)
        

    ax_right = plt.axes(rect_right)
    ax_right.tick_params(direction='in', labelbottom=False, labelleft=False, labelright=False)

    textstr1 = ('Average Location Error: '+str(av_location_error)+' km' )
    props1 = dict(boxstyle='round', facecolor='palevioletred', alpha=0.2)
    plt.text(0.04, 0.96, textstr1, fontsize=13, verticalalignment='top', bbox=props1)

    textstr1 = ('Average Origin Time Error: '+str(av_originT_error)+' second' )
    props1 = dict(boxstyle='round', facecolor='palevioletred', alpha=0.2)
    plt.text(0.04, 0.89, textstr1, fontsize=13, verticalalignment='top', bbox=props1)

    textstr1 = ('Average Depth Error: '+str(av_depth_erro)+' km' )
    props1 = dict(boxstyle='round', facecolor='palevioletred', alpha=0.2)
    plt.text(0.04, 0.82, textstr1, fontsize=13, verticalalignment='top', bbox=props1)
    
    textstr = '\n'.join((
    r'         Location        ',
    r'Catalog: '+ str(av_gt_latitude) +'  '+ av_gt_longitude, 
    r'Horizontal Uncertainty: ' + str(av_gt_Hunc), 
    r'Estimation: '+ str(av_pr_latitude) +'  '+ av_pr_longitude, 
    r'Estimation Uncertainty: ' + str(av_pr_unc) + ' km',
    ))   
    props = dict(boxstyle='round', facecolor='turquoise', alpha=0.2)
    plt.text(0.04, 0.73, textstr, fontsize=13, verticalalignment='top', bbox=props)    

    textstr = '\n'.join((
    r'         Depth        ',
    r'Catalog: ' + str(av_gt_depth)+ ' km',
    r'Catalog Uncertainty: ' + str(av_gt_depth_unc),
    r'Estimation: ' + str(av_pr_depth) + ' km',
    r'Estimation Uncertainty: ' + str(av_pr_depth_unc) + ' km', 
    ))    
    props = dict(boxstyle='round', facecolor='gold', alpha=0.2)
    plt.text(0.04, 0.49, textstr, fontsize=13, verticalalignment='top', bbox=props)    
    
    
    textstr = '\n'.join((
    r'        Origin Time      ',
    r'Catalog: '+ str(av_gt_origin_time),
    r'Catalog Uncertainty: ' + str(av_gt_origin_time_unc), 
    r'Estimation: ' + str(av_pr_origin_time),
    r'Estimation Uncertainty: ' + str(av_pr_origin_time_unc) + ' s', 
    ))
    props = dict(boxstyle='round', facecolor='mediumslateblue', alpha=0.2)
    plt.text(0.04, 0.25, textstr, fontsize=13, verticalalignment='top', bbox=props)  
    
    
    plt.savefig('./plots/'+event[ii]+'.png', bbox_inches='tight', dpi=300) 

    plt.show()










########### Global

lat1 = [43.22]
lon1 = [-120.51]
 
fig = plt.figure(figsize=(10, 8)) 
m = Basemap(llcrnrlon = -180, llcrnrlat = -60,
                urcrnrlon = 180, urcrnrlat = 80,
                lat_0 = 0, 
                lon_0 = 0,
                resolution='l', projection='mill')  # resolution: i, h, l     projection='lcc', "mill"

m.drawmapboundary(fill_color='#85A6D9')
m.fillcontinents(color='lavender',lake_color='#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawcountries(color='#6D5F47', linewidth=.4)                


m.scatter(lon1, 
          lat1, 
          latlon = True, 
          marker = "o", 
          color = "r", 
          s = 200,
          zorder=5,
          label="Catalog"
              )  

#plt.savefig('./plots/uw60545811.png', dpi=400)
plt.show()
    







