#!~/software/anaconda3/bin/python
# ------------------------------------------------------------------------
#  
#   Program:  ARTMIP_Tier2_Paleo_calc_percentiles.py
#
#   Purpose: To calculate the percentiles of the IVT anomaly distributions
#            throughout the domain. This resembles a similar script used
#            previously for transmission to CW3E. 
#
# ------------------------------------------------------------------------

import calendar
import datetime
import glob
import os
import csv
import sys

import numpy as np
import numpy.ma as ma

from netCDF4 import Dataset
from netCDF4 import num2date

tags = ['cam.h2']
#runs = ['PreIndust', 'PI_21ka-CO2', '10ka-Orbital']
#time_units = ['days since 0001-01-01T00:00:00', 'days since 0001-01-01T00:00:00', 'days since 0201-01-01T00:00:00']
runs = ['10ka-Orbital']
time_units = ['days since 0201-01-01T00:00:00']

# loop through the reanalyses and types
for t_i, tag in enumerate(tags):
  for r_i, run in enumerate(runs):

    print('Now examining {}'.format(run))

    # ------------------------ Setting Path Variables ----------------------
    rawpath = '/global/cscratch1/sd/czarzyck/Paleo/{}'.format(run)
    outpath = '/global/cscratch1/sd/czarzyck/Paleo/ARTMIP_output/'
    file_name_base = tag
    
    ## If outpath doesn't exist, create it.
    if not os.path.exists(outpath):
      os.makedirs(outpath)

    # ----------------------- Obtain a Test File to Properly Set Grids ----------------
    files = os.listdir(rawpath)
    test_file_name = files[0]
    test_file = Dataset('{}/{}'.format(rawpath, test_file_name), 'r')
    test_lats = test_file.variables['lat'][:]
    test_lons = test_file.variables['lon'][:]

    # ------------------------- Establishing Grids -------------------------
    lats = test_lats
    lons = test_lons
  
    # adjust lons if they go from -180 to 180 (later code assumes 0 to 360)
    if np.nanmin(lons) < 0:
      print('This is true')
      lons = np.where(lons < 0, lons+360., lons)

    nrows = len(lats)
    ncols = len(lons)

    # ---------------------------- Function(s) -----------------------------
    def available_files(path):
      " Retrieves a list of available files "
      full_list = glob.glob(path + '/*' + tag + '*.nc')
      name_list = []
      for i in full_list:
        name_list.append(i.split('/')[-1]) # depends on given file structure
      print(len(name_list), 'files available')
      return np.sort(name_list)

    def calc_true_doy(d):
      " Calculates the day of year index 0...364 "
      doy = (d - datetime.date(d.year,1,1)).days
      if calendar.isleap(d.year) and doy == 59: 
        doy == -999
      elif calendar.isleap(d.year) and doy > 59:
        doy = doy - 1
      return doy

    # For subsetting of IVT data (lons must be between 0 and 360, lats must be between -90 and 90)
    min_lon = 180.
    max_lon = 260.
    min_lat = 0.
    max_lat = 70.

    # Obtain the indices of the sub-region
    sub_lats_indices = np.where(lats >= min_lat)[0]
    min_lat_index = min(sub_lats_indices)
    lats_filtered = lats[sub_lats_indices]
    sub_lats_indices = np.where(lats_filtered <= max_lat)[0]
    sub_lats_indices = min_lat_index + sub_lats_indices

    sub_lons_indices = np.where(lons >= min_lon)[0]
    min_lon_index = min(sub_lons_indices)
    lons_filtered = lons[sub_lons_indices]
    sub_lons_indices = np.where(lons_filtered <= max_lon)[0]
    sub_lons_indices = min_lon_index + sub_lons_indices

    sub_lats = lats[sub_lats_indices]
    sub_lons = lons[sub_lons_indices]

    # --------------------------- Begin Main Code --------------------------
  
    # Creating a list of available files
    directory = available_files(rawpath)
    print(directory[0], 'to', directory[-1])

    # Loading
    with np.load(outpath + '/ARTMIP_Tier2_Paleo.ivt_fft3_' + run + '.npz') as data:
      ivt_seasonalcycle = data['ivt_seasonalcycle'][:]
      ivt_seasonalcycle = ivt_seasonalcycle[:,sub_lats_indices,:]
      ivt_seasonalcycle = ivt_seasonalcycle[:,:,sub_lons_indices]
    print(ivt_seasonalcycle.shape)

    # Cycling through all dates to extract IVT
    all_vals = np.ones([len(sub_lats),len(sub_lons)], dtype=float) * np.NaN
    counter = 0 # dates counter
    # this will be stacked
  
    file_counter = 0
    
    # calculating the daily mean of IVT for seasonal cycle calculation (all data is hourly)
    #temp_array = np.ones([len(sub_lats), len(sub_lons), 24], dtype='float') * np.NaN
    temp_array = np.ones([len(sub_lats), len(sub_lons), 4], dtype='float') * np.NaN
    print('New daily array created!')
  
    for f_i,f in enumerate(directory):
  
      file_counter += 1
  
      # extract the IVT components for the entire file
      hourly_file = Dataset(rawpath + '/' + directory[f_i], 'r')
      time = hourly_file.variables['time']
      ivt_u = hourly_file['IVTx'][:] # time x lat x lon
      ivt_v = hourly_file['IVTy'][:] # time x lat x lon
      times = num2date(time[:], time_units[r_i], calendar='noleap')
      hourly_file.close()
  
      # calculate the date
      for t_i, t in enumerate(times):
        
        year = t.year
        month = t.month
        day = t.day
        hour = t.hour
        
        print('Now evaluating ' + str(year) + '/' + str(month) + '/' + str(day) + '/' + str(hour))
        
        # calculate the total IVT
        ivt = np.sqrt(ivt_u[t_i, :, :]**2 + ivt_v[t_i, :, :]**2)
        
        # calculate the IVT field for the chosen domain
        ivt_slice = ivt[sub_lats_indices,:] # account for any subsetting spatially
        ivt_slice = ivt_slice[:,sub_lons_indices]
      
        doy_ind = calc_true_doy(datetime.date(year, month, day))
        if doy_ind == -999: # a marker for a leap day
          print('skipped this day!')
          continue
        time_of_day = hour # the hour of the day
        
        not_emptied = (np.isnan(np.nanmean(temp_array[0,0,:])) == False) # there must be something there
    
        # assign an element to the day's temp array based on the time of day 
        #temp_array[:, :, int(hour)] = ivt_slice
        if hour == 0:
          temp_array[:, :, 0] = ivt_slice
        elif hour == 6:
          temp_array[:, :, 1] = ivt_slice
        elif hour == 12:
          temp_array[:, :, 2] = ivt_slice
        elif hour == 18:
          temp_array[:, :, 3] = ivt_slice
    
        # check to see if this is the end of the day
        #if time_of_day == 23: # the end of the day
        if time_of_day == 18:
          print('end of day!')
          daily_mean_ivt = np.nanmean(temp_array, axis=2)
          # add the IVT anomaly slice to the master array
          if counter == 0:
            anom_slice = daily_mean_ivt - ivt_seasonalcycle[doy_ind,:,:]
            all_vals[:, :] = anom_slice  
            counter += 1
          else:
            anom_slice = daily_mean_ivt - ivt_seasonalcycle[doy_ind,:,:]
            all_vals = np.dstack((all_vals, anom_slice)) # lat x lon x date
            counter += 1
          # create the new array for the next day
          #temp_array = np.ones([len(sub_lats),len(sub_lons),24], dtype='float') * np.NaN
          temp_array = np.ones([len(sub_lats),len(sub_lons),4], dtype='float') * np.NaN
          print('New daily array created!')
        elif time_of_day == 0 and not_emptied:
          print('There must be missing data! The script will terminate.')
          exit()

    # write results to output files
    np.savez(outpath + 'ARTMIP_Tier2_Paleo.all_vals_' + run + '.npz', all_vals=all_vals)
    f = open(outpath + 'ARTMIP_Tier2_Paleo.percentiles_' + run + '.txt', 'w')

    # Calculate percentiles of IVT
    print('IVTa percentile values ... 90, 91, 92, 93, 94, 95, 96, 97, 98')
    print('all: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f') 
    print(np.nanpercentile(all_vals, np.arange(90.,99.,1), interpolation='linear'))

    f.write('IVTa percentile values ... 90, 91, 92, 93, 94, 95, 96, 97, 98\n')
    f.write('all: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n') 
    pcts = np.nanpercentile(all_vals, np.arange(90.,99.,1), interpolation='linear')
    f.write(str(pcts))
    f.close()
  
    np.savez(outpath + 'ARTMIP_Tier2_Paleo.percentiles_' + run + '.npz', pct=pcts)

print('\nEnd of Program')
