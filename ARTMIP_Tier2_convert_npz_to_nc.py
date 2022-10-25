#!~/software/anaconda3/bin/python
# ----------------------------------------------
#
#  Program Name: ARTMIP_Tier2R_convert_npz_to_nc.py
#
#  Purpose: To convert .npz files to .nc
#     files using netcdf4 module.
#      
#       Note: This is following specific file
#             instructions provided by ARTMIP.
#             
# ----------------------------------------------

# Step 1 (preliminaries)

import numpy as np
from netCDF4 import Dataset
import os
import sys

from datetime import datetime, timedelta
from netCDF4 import num2date, date2num

#from nco import Nco
#nco = Nco()

run = str(sys.argv[1])
run_id = str(sys.argv[1])
min_year = int(sys.argv[2])
max_year = int(sys.argv[3])

if run == 'PI_21ka-CO2':
    time_units='days since 0001-01-01T00:00:00'
elif run == 'PreIndust':
    time_units='days since 0001-01-01T00:00:00'
else:
    time_units='days since 0201-01-01T00:00:00'

input_path = '/global/cscratch1/sd/czarzyck/Paleo/ARTMIP_output/{}/'.format(run)
output_path = input_path

file_name_base = 'ARTMIP_Tier2R_detect_ars_' + run
outfile_name_base = run_id + '.ar_tag.Mundhenk_v3.6hr.'

# read-in a sample file to calculate the latitude and longitude arrays
files = os.listdir('/global/cscratch1/sd/czarzyck/Paleo/{}'.format(run))
dataset = Dataset('/global/cscratch1/sd/czarzyck/Paleo/{}/{}'.format(run, files[0]), 'r')
lats = dataset.variables['lat'][:]
lons = dataset.variables['lon'][:]
dataset.close()

# Step 2 (obtain a list of files in the input directory)

files = np.sort(os.listdir(input_path)) # This path will only contain .npz files

# all input files contain AR binary tags for a single hour

# Step 3 (begin converting the files of interest to .nc)

single_dtg = False # Are the files output for a single dtg?
grouped_year = True # Are the files being grouped? (all days in year within one file)

if single_dtg == True:
  for i in range(0,len(files)): # Loop through each file in the directory

    # Read-in the .npz file
    file_name = files[i] 

    dtg = file_name[-15:-4] # yyyymmdd
    y_str = dtg[0:4]
    m_str = dtg[4:6]
    d_str = dtg[6:8]
    t_str = dtg[9:11]
    dtg_str = dtg

    npz_file = np.load(input_path+file_name)  
    AR_BINARY_TAG = npz_file['mask']
    
    del npz_file

    print('The .npz file has been read-in for ' + dtg + '!')

    dims = AR_BINARY_TAG.shape # time steps x lats x lons
    TIME = np.array([datetime.datetime(int(y_str), int(m_str), int(d_str), int(t_str))])

    # Read-in the variables
    dataset = Dataset(output_path+file_name_base+'_'+dtg+'.nc4','w',format='NETCDF4')
    
    # Create a new set of dimensions for the variables
    time_dim = dataset.createDimension('time',len(TIME))
    lat_dim = dataset.createDimension('lat',len(lats))
    lon_dim = dataset.createDimension('lon',len(lons))
    
    print('The dimensions have been made!')
      
    # Create coordinate variables for 4-dimensions
    time = dataset.createVariable('time',np.float32,('time'))
    lat = dataset.createVariable('lat',np.float32,('lat',))  
    lon = dataset.createVariable('lon',np.float32,('lon',))
    ar_binary_tag = dataset.createVariable('ar_binary_tag',np.int8,('time', 'lat', 'lon'))

    print('The coordinate variables have been made!')

    # Assign descriptions to variables
    time.standard_name = 'time'
    time.long_name = 'Time'
    time.units = time_units
    time.calendar = 'standard'

    lat.standard_name = 'lat'
    lat.long_name = 'Latitude'
    lat.units = 'degrees_north'
    lat.axis = 'Y'

    lon.standard_name = 'lon'
    lon.long_name = 'Longitude'
    lon.units = 'degrees_east'
    lon.axis = 'X'

    ar_binary_tag.description = 'binary indicator of atmospheric river'
    ar_binary_tag.scheme = 'Mundhenk'
    ar_binary_tag.version = '3.0'

    print('The variables have been described!')

    # Fill the variables with data
    TIME = date2num(date, time_units, calendar='noleap')
    time[:] = TIME
    del TIME
    lat[:] = lats
    lon[:] = lons
    ar_binary_tag[:,:,:] = AR_BINARY_TAG
    del AR_BINARY_TAG

    print('The variables have been filled with data.')

    # Write the file
    dataset.close()

    print('The file has been written!')

if grouped_year == True:
  years = range(min_year, max_year+1) # The range of years to group
  year_tags = np.ones([len(lats),len(lons)])*np.NaN

  for y in years: # Loop through each year
    year_of_interest = y
    counter = 0
    TIME = []
  
    for i in range(0,len(files)): # Loop through each file in the directory # this assumes that the files are in chronological order

        # Read-in the .npz file
      file_name = files[i] 
      
      if '.npz' not in file_name:
        continue

      dtg = file_name[-15:-4] # yyyymmdd
      y_str = dtg[0:4]
      m_str = dtg[4:6]
      d_str = dtg[6:8]
      t_str = dtg[9:11]
      dtg_str = dtg
    
      if y_str == str(year_of_interest).rjust(4,'0'): # Is this the year being examined?
        
        TIME.append(datetime(int(y_str), int(m_str), int(d_str), int(t_str)))
      
        npz_file = np.load(input_path+file_name)
        AR_BINARY_TAG = np.asarray(npz_file['mask'], dtype='int8')
        
        if counter == 0: # Initialize the array
          year_tags = AR_BINARY_TAG
          counter += 1
        else:
          year_tags = np.dstack((year_tags, AR_BINARY_TAG))
          counter += 1

        del npz_file

        print('The .npz file has been read-in for ' + dtg + '!')
      else:
        continue
      
    dims = year_tags.shape # lats x lons x time

    # Read-in the variables
    dataset = Dataset(output_path+outfile_name_base+str(year_of_interest).rjust(4,'0')+'0101-'+str(year_of_interest).rjust(4,'0')+'1231.non_compressed.nc4','w',format='NETCDF4')

    # Create a new set of dimensions for the variables
    time_dim = dataset.createDimension('time',len(TIME))
    lat_dim = dataset.createDimension('lat',len(lats))
    lon_dim = dataset.createDimension('lon',len(lons))

    print('The dimensions have been made!')

    # Create coordinate variables for 4-dimensions
    time = dataset.createVariable('time',np.float32,('time'))
    lat = dataset.createVariable('lat',np.float32,('lat',))
    lon = dataset.createVariable('lon',np.float32,('lon',))
    ar_binary_tag = dataset.createVariable('ar_binary_tag',np.int8,('time', 'lat', 'lon'))

    print('The coordinate variables have been made!')

    # Assign descriptions to variables
    time.standard_name = 'time'
    time.long_name = 'Time'
    time.units = time_units
    time.calendar = 'noleap'

    lat.standard_name = 'lat'
    lat.long_name = 'Latitude'
    lat.units = 'degrees_north'
    lat.axis = 'Y'

    lon.standard_name = 'lon'
    lon.long_name = 'Longitude'
    lon.units = 'degrees_east'
    lon.axis = 'X'

    ar_binary_tag.description = 'binary indicator of atmospheric river'
    ar_binary_tag.scheme = 'Mundhenk'
    ar_binary_tag.version = '3.0'

    print('The variables have been described!')

    # Fill the variables with data
    TIME = date2num(TIME[:], time_units, calendar='noleap')
    time[:] = TIME
    del TIME
    lat[:] = lats
    lon[:] = lons
    year_tags = np.swapaxes(year_tags, 1, 2) # lat x lon x time -> lat x time x lon
    year_tags = np.swapaxes(year_tags, 0, 1) # lat x time x lon -> time x lat x lon
    ar_binary_tag[:,:,:] = year_tags
    del AR_BINARY_TAG
    del year_tags

    print('The variables have been filled with data.')

    # Write the file
    dataset.close()

    print('The file has been written!')
    
    pct_file = np.load('/global/cscratch1/sd/czarzyck/Paleo/ARTMIP_output/ARTMIP_Tier2_Paleo.percentiles_{}.npz'.format(run_id))
    pcts = pct_file['pct']
    thresh = pcts[4] # the 94th percentile
    
    # compress the file and add global attributes
    input_f = output_path+outfile_name_base+str(year_of_interest).rjust(4, '0') +'0101-'+str(year_of_interest).rjust(4, '0')+'1231.non_compressed.nc4'
    output_f = output_path+'nc/'+outfile_name_base+str(year_of_interest).rjust(4, '0') +'0101-'+str(year_of_interest).rjust(4,'0')+'1231.nc4'
  
    #nco.nccopy(input=input_f, output=output_f, options=[ '-d1' ])
    #cmd = 'ncks --cnk_dmn time,100 -L 1 ' + input_f + ' ' + output_f
    #os.system(cmd)
    #os.remove(input_f)

# CMZ commented out 10/24/22
#    os.system('mkdir -p {}/nc'.format(output_path))
#    os.system('ncks -h -4 -L 1 {} {}'.format(input_f, output_f))
#    os.system('ncatted -O -h -a history,global,a,c,"Created by Kyle M. Nardi using Python 3.7. Please direct questions to kmn182@psu.edu." {}'.format(output_f))
#    os.system('ncatted -O -h -a threshold_domain,global,a,c,"Anomaly intensity threshold calculated over North Pacific from 180-260E and 0-70N." {}'.format(output_f))
#    os.system('ncatted -O -h -a threshold,global,a,c,"Anomaly intensity threshold for this dataset is {} kg/m/s." {}'.format(str(thresh), output_f))
    
    
#    print('The compressed file has been written!')
    



