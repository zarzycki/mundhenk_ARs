#!~/software/anaconda3/bin/python
# -------------------------------------------------------------------------
#   
#   Program Name:  ARTMIP_Tier2_Paleo_calc_fft.py
#
#   Purpose: Uses .nc files of IVT in order to calculate FFT for IVT.
#          This script is for the specific purpose of calculating the IVT
#            seasonal cycle for the ARTMIP Tier1 dataset.
#
#  NOTE: This script is optimized to work on ICDS.
#
# -------------------------------------------------------------------------

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

import matplotlib.pyplot as plt

tags = ['cam.h2']
#runs = ['PreIndust', 'PI_21ka-CO2', '10ka-Orbital']
#time_units = ['days since 0001-01-01T00:00:00', 'days since 0001-01-01T00:00:00', 'days since 0200-01-01T00:00:00']
runs = ['10ka-Orbital']
time_units = ['days since 0201-01-01T00:00:00']

# loop through the types
for t_i, tag in enumerate(tags):
  for r_i, run in enumerate(runs):

    print('Now examining {}'.format(run))

    # ------------------------ Setting Path Variables ----------------------
    rawpath = '/global/cscratch1/sd/czarzyck/Paleo/{}'.format(run)
    outpath = '/global/cscratch1/sd/czarzyck/Paleo/ARTMIP_output'
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
    test_file.close()

    # ------------------------- Establishing Grids -------------------------
    lats = test_lats
    lons = test_lons

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
      leap = calendar.isleap(d.year)
      doy = (d - datetime.date(d.year,1,1)).days
      if leap and doy == 59: # leap day
        doy = -999
      elif leap and doy > 59:
        doy = doy - 1
      return doy

    def perform_fft(var_3d, num_harmonics, ax):
      " Using FFT to calculate seasonal cycle of variable "
      # e.g., if num_harmonics = 3 will retain only the mean and first two harmonics
      fft_array = np.fft.fft(var_3d, axis=ax)
      if ax == 0:
        nd, nr, nc = var_3d.shape
      elif ax == 2:
        nr, nc, nd = var_3d.shape
      masked_fft_array = np.empty([nd,nr,nc], dtype=complex)
      harmonic_mask = np.zeros(nd)
      harmonic_mask[:num_harmonics] = 1
      harmonic_mask[(-1*num_harmonics):] = 1 # Symmetry required
      for i in np.arange(0,nd,1):
        if harmonic_mask[i] > 0:
          if ax == 0:
            masked_fft_array[i,:,:] = fft_array[i,:,:]
          elif ax == 2:
            masked_fft_array[i,:,:] = fft_array[:,:,i]
        else:
          masked_fft_array[i,:,:] = 0
      ffti_array = np.fft.ifft(masked_fft_array, axis=0) # ALWAYS (nd, nr, nc)
      return np.real(ffti_array)

    # --------------------------- Begin Main Code --------------------------

    # Creating a list of available files
    directory = available_files(rawpath)
    print(directory[0], 'to', directory[-1])

    # Cycling through all dates to extract IVT daily average
    ivt_3d_sum = np.zeros([nrows,ncols,365], dtype='float') # this will be a sum 
    ivt_3d_counter = np.zeros([nrows,ncols,365], dtype='float') # this will be a counter
    # 3D array will need to be used in order to avoid a memory error
    # averaging will need to be done via summation and counts

    file_counter = 0
    date_counter = 0
    
    # calculating the daily mean of IVT for seasonal cycle calculation (number of hours depends on time resolution)
    #temp_array = np.ones([nrows,ncols,24], dtype='float') * np.NaN
    temp_array = np.ones([nrows,ncols,4], dtype='float') * np.NaN
    print("New array created!")
    
    for f_i,f in enumerate(directory):
  
      file_counter += 1
  
      # extract the IVT components for the entire file
      hourly_file = Dataset(rawpath + '/' + directory[f_i], 'r')
      time = hourly_file.variables['time']
      ivt_u = hourly_file['IVTx'][:] # time x lat x lon
      ivt_v = hourly_file['IVTy'][:] # time x lat x lon
      times = num2date(time[:], time_units[r_i], calendar='noleap')
      hourly_file.close()
      
      print(f)
      
      for t_i, t in enumerate(times):
        year = t.year
        month = t.month
        day = t.day
        hour = t.hour
        
        print('Now examining ' + str(year) + '/' + str(month) + '/' + str(day) + '/' + str(hour))
  
        # calculate the total IVT
        ivt = np.sqrt(ivt_u[t_i, :, :]**2 + ivt_v[t_i, :, :]**2)
  
        # calculate the day of year
        doy_ind = calc_true_doy(datetime.date(year, month, day))
        if doy_ind == -999: # a marker for a leap day
          print('skipped this day!')
          continue
        time_of_day = hour
    
        not_emptied = (np.isnan(np.nanmean(temp_array[0,0,:])) == False) # there must be something there
    
        # assign an element to the day's temp array based on the time of day 
        #temp_array[:, :, int(hour)] = ivt
        if hour == 0:
          temp_array[:, :, 0] = ivt
        elif hour == 6:
          temp_array[:, :, 1] = ivt
        elif hour == 12:
          temp_array[:, :, 2] = ivt
        elif hour == 18:
          temp_array[:, :, 3] = ivt
    
        # check to see if this is the end of the day
        #if time_of_day == 23: # the end of the day (this is the only logic necessary if no missing data and hourly!)
        if time_of_day == 18: # the end of the day if the time is 6-hourly
          print('end of day!')
          if np.isnan(np.nanmean(temp_array[0,0,:])): # this would happen if the entire previous day is missing
            ivt_3d_sum[:,:,doy_ind] += 0 # do not add to the sum
            ivt_3d_counter[:,:,doy_ind] += 0 # do not add to the counter
          else: # this would happen if the entire day is not missing (np.nanmean would have accounted for individual hrly missing values)
            ivt_3d_sum[:,:,doy_ind] += np.nanmean(temp_array, axis=2)
            ivt_3d_counter[:,:,doy_ind] += 1
          date_counter += 1
          # create the new array for the next day
          #temp_array = np.ones([nrows,ncols,24], dtype='float') * np.NaN
          temp_array = np.ones([nrows,ncols,4], dtype='float') * np.NaN
          print('New daily array created!')
        elif time_of_day == 0 and not_emptied:
          print('There must be missing data. This script will only run for a full set of data!')
          exit()
  
    print(file_counter, date_counter, 'files/dates processed')

    # Using FFT to calculate the seasonal cycle
    ivt_3d = ivt_3d_sum / ivt_3d_counter # average over all years
    ivt_seasonalcycle = perform_fft(ivt_3d, 3, 2)

    # Saving seasonal cycle file
    np.savez('{}/ARTMIP_Tier2_Paleo.ivt_fft3_{}.npz'.format(outpath, run), ivt_seasonalcycle=ivt_seasonalcycle)
    # Use as...
    #   with np.load(fname) as data:
    #       ivt_seasonalcycle = data["ivt_seasonalcycle"]

print('\nEnd of Program')
