#!~/software/anaconda3/bin/python
# ---------------------------------------------------------------------------------
#   Program:  ARTMIP_Tier2R_detect_ars_correctorient.py
#
#   Purpose:
#       Manipulates the pre-calculated IVT files provided by CW3E to isolate
#       atmospheric rivers (ARs) from within positive IVT anomalies,
#       using the pre-processed FFT file for the long-term mean and seasonal
#       cycle.
#
# ---------------------------------------------------------------------------------

import calendar
import csv
import datetime
import glob
import sys
import os 
import time as TIME

import numpy as np
import numpy.ma as ma

from scipy import ndimage
import scipy.interpolate as interp

from netCDF4 import Dataset
from netCDF4 import num2date

from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from skimage.feature import canny

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
    ## If outpath doesn't exist, create it.
    if not os.path.exists(outpath):
      os.makedirs(outpath)
      
    file_name_base_u = tag
    file_name_base_v = tag

    # ----------------------- Obtain a Test File to Properly Set Grids ----------------
    files = os.listdir(rawpath)
    test_file_name = files[0]
    test_file = Dataset(rawpath + '/' + test_file_name, 'r')
    test_lats = test_file.variables['lat'][:]
    test_lons = test_file.variables['lon'][:]
    
    if os.path.exists(outpath + run + '/') == False:
      os.mkdir(outpath + run)
      
    existing_files = glob.glob(outpath + run + '/' + '*' + tag + '*.npz') # list of output files for the tag and run

    # ------------------------- Establishing Grids -------------------------
    lats = test_lats
    lons = test_lons

    # ------------------------- Establishing Grids -------------------------
    data_lats = test_lats
    data_lons = test_lons
    lon_deg = np.absolute(data_lons[1] - data_lons[0]) # this is the resolution, which is non-standard
    lat_deg = np.absolute(data_lats[1] - data_lats[0])
    nrows = len(data_lats)
    ncols = len(data_lons)

    grid_buffer = 0 # number of grid points around the periphery to buffer
    # set to 0 if global algorithm
    # set to nonzero if regional algorithm

    num_outrows = nrows # Removed option to subset output
    num_outcols = ncols

    # Variables approximating grid spacing in terms of kilometers (km)
    r = 6371 # Earth radius, in km
    lon_dist_array = np.empty([nrows,ncols],dtype=float)
    lat_dist_array = np.empty([nrows,ncols],dtype=float)
    area_array = np.empty([nrows,ncols],dtype=float)
    for ri in np.arange(0, nrows, 1):
      for ci in np.arange(0, ncols, 1):
        lat_deg_dist = (111132.954 - 559.822*np.cos(2*np.deg2rad(data_lats[ri]))\
              + 1.175*np.cos(4*np.deg2rad(data_lats[ri])))/1000.
        lon_deg_dist = (np.pi*r*np.cos(np.deg2rad(data_lats[ri])))/180.
        area_array[ri,ci] = (lon_deg * lon_deg_dist) * (lat_deg * lat_deg_dist)
        lon_dist_array[ri,ci] = lon_deg * lon_deg_dist
        lat_dist_array[ri,ci] = lat_deg * lat_deg_dist     
    lon_dist_1d = np.asarray(lon_dist_array[:,0]) # lon_dist_1d[y0] approx. km distance between lon
    lat_dist_1d = np.asarray(lat_dist_array[:,0]) # lat_dist_1d[y0] approx. km distance between lat
    londistFunc = interp.UnivariateSpline(np.arange(0, len(lon_dist_1d), 1), lon_dist_1d, s=0)
    latdistFunc = interp.UnivariateSpline(np.arange(0, len(lat_dist_1d), 1), lat_dist_1d, s=0)

    # Functions to interpolate center of mass locations
    latFunc = interp.UnivariateSpline(np.arange(0, len(data_lats), 1), data_lats, s=0) 
    lonFunc = interp.UnivariateSpline(np.arange(0, len(data_lons), 1), data_lons, s=0)

    # --- Step 4) --------- Set Variables for Feature Testing -------------------------
  
    # first, calculate the percentiles from the appropriate text file
    file = np.load(outpath + 'ARTMIP_Tier2_Paleo.percentiles_' + run + '.npz')
    pcts = file['pct']

    min_IVT = pcts[4]    # IVT anomaly threshold used to identify blob
                # ... this is 94th percentile of the IVTa distribution
    min_meanintensity = pcts[5]  # Lowest mean IVT anomaly intensity within a feature
                # ... this is 95th percentile 
    per_97 = pcts[7]          # 97th percentile for tropical scrutiny
    per_98 = pcts[8]           # 98th percentile for detection of TC-like features
                # ... see cam_dataset_comp.py
  
    print(min_IVT, min_meanintensity, per_97, per_98)

    size_mask = 300000.         # Minimum blob size, 2,000 km x 150 km = 300,000 km2
    min_length = 1400.          # Shortest feature length (approx. km)
    min_aspect = 1.4            # Minimum length/width ratio for retained feature
    min_eccentricity = .85      # Minimum eccentricity of a retained feature
    lat_forceorient = 16.       # Threshold for increased testing (mostly TC-focused)

    # --------------------- Functions ------------------------------------------------- 
    def available_files(path):
      " Retrieves a list of available files "
      full_list = glob.glob(path + '/*' + tag + '*.nc')
      name_list = []
      for i in full_list:
        name_list.append(i.split('/')[-1]) # depends on given file structure
      print(len(name_list), 'files available')
      return np.sort(name_list)
    
    def skeleton_length_II(array):
      " Skeletonizes the blob to calculate a more accurate length in km "
      bounds = (90.,-90.,360.,0.)
      sk = skeletonize(array)
      ir,ic = np.nonzero(sk)
      numri = np.max(ir)-np.min(ir)
      numci = np.max(ic)-np.min(ic)
      if numri > numci:
        arr_flag = 'vert'
      else:
        arr_flag = 'horiz'

      newr,newc = [],[]
      if arr_flag == 'horiz':
        duplicates = []
        for ind,col in enumerate(ic):
          if np.count_nonzero(np.where(ic == ic[ind])) > 1 and col not in duplicates:
            duplicates.append(col)
            newc.append(ic[ind])
            newr.append(np.min(ir[np.where(ic == ic[ind])]))
          elif col in duplicates:
            continue
          else:
            newr.append(ir[ind])
            newc.append(ic[ind])
        snewr, snewc = (list(el) for el in zip(*sorted(zip(newr, newc))))
        z = np.polyfit(snewc,snewr,3)
        p = np.poly1d(z)
        x = np.arange(np.min(snewc),np.max(snewc)+1,1)
        y = p(x)
        y = np.where(y >= bounds[3], bounds[3]-0.1, y)
  
      if arr_flag == 'vert':
        duplicates = []
        for ind,row in enumerate(ir): 
          if np.count_nonzero(np.where(ir == ir[ind])) > 1 and row not in duplicates:
            duplicates.append(row)
            newr.append(ir[ind])
            newc.append(np.min(ic[np.where(ir == ir[ind])]))
          elif row in duplicates:
            continue
          else:
            newr.append(ir[ind])
            newc.append(ic[ind])
        snewr, snewc = (list(el) for el in zip(*sorted(zip(newr, newc))))
        z = np.polyfit(snewr,snewc,3)
        p = np.poly1d(z)
        y = np.arange(np.min(snewr),np.max(snewr)+1,1)
        y = np.where(y >= bounds[3], bounds[3]-0.1, y)
        x = p(y)

      r = 6371 # km
      y_vals = np.copy(y)   
      piecewise_length = []
      for i in xrange(0,len(x)-1,1):
        # Visiting Pythagoras to estimate length of skeleton in km
        try:
          #x_pixel_km = lat_dist_1d[y_vals[int(i)]]
          x_pixel_km = latdistFunc(y_vals[i]).tolist()
        except IndexError:
          try:
            x_pixel_km = latdistFunc(y_vals[i-1]).tolist()
          except IndexError:
            x_pixel_km = np.mean(lat_dist_1d)
        try:
          y_pixel_km = londistFunc(y_vals[i]).tolist()
        except IndexError:
          try:
            y_pixel_km = londistFunc(y_vals[i-1]).tolist()
          except IndexError:
            y_pixel_km = np.mean(lon_dist_1d)
        len_km = np.sqrt((abs(y_vals[int(i)+1]-y_vals[int(i)])*x_pixel_km)**2 \
                 + (abs(x[int(i)+1]-x[int(i)])*y_pixel_km)**2)
        piecewise_length.append(len_km)

      return x, y, int(np.sum(piecewise_length))

    def calc_ivtdir(u, v):
      " Calculates polar angle of IVT given u & v components "
      #   WARNING:  Not in METEOROLOGICAL conventions
      #   Increases counterclockwise from the (+) x-axis
      #  0 degrees = wind vector pointing towards the east,
      #  90 degrees = wind vector pointing towards the north
      calcdir = (180./np.pi) * np.arctan2(v,u)
      return np.where(calcdir < 0, 360. + calcdir, calcdir)

    def calc_ivtmag(ivt_u, ivt_v):
      " Calculates magnitude of IVT given u & v components "
      return np.sqrt((ivt_u)**2 + (ivt_v)**2)

    def calc_true_doy(d):
      " Calculates the day of year index 0...364 "
      doy = (d - datetime.date(d.year,1,1)).days
      if calendar.isleap(d.year) and doy == 59: 
        doy == -999
      elif calendar.isleap(d.year) and doy > 59:
        doy = doy - 1
      return doy

    # --------------------- Main Code -------------------------------------------------

    # Creating a list of available files
    directory = available_files(rawpath)
    print(directory[0], 'to', directory[-1])

    # Load file w/ mean + seasonal cycle of IVT
    with np.load(outpath + 'ARTMIP_Tier2_Paleo.ivt_fft3_' + run + '.npz') as data:
      fft_array = data['ivt_seasonalcycle'] 

    # Begin detection...
    edge_cnt = 0
    aspect_cnt = 0
    length_cnt = 0
    subtropvar_cnt = 0
    intensity_cnt = 0
    tcish_cnt = 0
    tchole_cnt = 0
    tropics_cnt = 0
    multipeak_cnt = 0

    com_lats = [] # for centers of mass
    com_lons = []
    intensity = [] # for optional plotting

    blob_counter = 0
    point_ar_dtgs = []

    non_rows = 0
    ar_rows = 0
  
    file_counter = 0
  
    for f_i,f in enumerate(directory):
    
      print(f)
  
      file_counter += 1
    
      # extract the IVT components for the entire file
      hourly_file = Dataset(rawpath + '/' + directory[f_i], 'r')
    
      ivt_u = hourly_file['IVTx'][:] # time x lat x lon
      ivt_v = hourly_file['IVTy'][:] # time x lat x lon
      time = hourly_file['time']
      times = num2date(time[:], time_units[r_i], calendar='noleap')
      hourly_file.close()
      
      # loop through all times in the file
      for t_i, t in enumerate(times):
      
        # get the date from the times
        year = t.year
          
        month = t.month
        day = t.day
        hour = t.hour
        dtg_str = str(year).rjust(4,'0') + str(month).rjust(2,'0') + str(day).rjust(2,'0') + '_' + str(hour).rjust(2,'0')
      
        # calculate the day of year
        d = datetime.date(year, month, day)
        dt = datetime.datetime(year, month, day, hour)
        print('We are examining ' + str(year) + '/' + str(month).rjust(2,'0') + '/' + str(day).rjust(2,'0') + '/' + str(hour).rjust(2,'0'))
        doy_ind = calc_true_doy(d)
        if doy_ind == -999: # a marker for a leap day
          print('skipped this day!')
          continue
        time_of_day = hour # the hour of the day
      
        # if partial output already exists
        theoretical_file = outpath + run + '/ARTMIP_Tier2_Paleo_detect_ars_' + run + '_' + dtg_str + '.npz'
    
        if theoretical_file in existing_files:
          print('This file already exists!')
          continue
      
        # start the search for ARs
        point_AR_flag = False

        id_array = np.zeros([nrows,ncols], dtype='uint8')
        
        ivt_u_slice = ivt_u[t_i, :, :]
        ivt_v_slice = ivt_v[t_i, :, :]
  
        ivt_mag = calc_ivtmag(ivt_u_slice, ivt_v_slice)
        ivt_dir = calc_ivtdir(ivt_u_slice, ivt_v_slice)

        # Removing seasonal cycle to create IVT anomaly array
        ivt_anom = ivt_mag - fft_array[doy_ind, :, :]

        # Removing all IVT values less than defined "min_IVT"
        threshold_array = np.where(ivt_anom > min_IVT, 1, 0)
        label_array, num_labels = ndimage.measurements.label(threshold_array) # label each unique candidate object ("blob")
        labels = np.unique(label_array)
        #all_labels = np.copy(label_array) # <-- TESTING

        # Removing small blobs from remaining IVT field
      #   sizes = ndimage.sum(area_array, label_array, range(num_labels+1)) # sum the areas of each grid cell within each labelled feature
    #     sizes[0] = 0 # no need for "everything else"
    #     mask_size = sizes < size_mask # Applying a defined minimum "size_mask"
    #     remove_blob = mask_size[label_array] # places True at all grid cells within candidate objects whose size is less than the threshold
    #     label_array[remove_blob] = 0 # makes all of these "True" grid cells from the last line 0's, eliminating the feature from consideration
    #     labels = np.unique(label_array) # Relabeling the blobs after small ones removed
    #     label_array = np.searchsorted(labels, label_array)

        # Counting the total number of blobs interrogated (after min IVT and size applied)
        blob_counter += np.count_nonzero(labels) 

        # Using skimage to calculate blob characteristics
        blob_props = regionprops(label_array, intensity_image=ivt_anom) 

        # Testing each blob to determine if it may be an AR
        retained_count = 0  # Will use to renumber retained blobs
        max_lon_index = len(data_lons) - 1
        min_lon_index = 0
  
        cross_bound_blob = False
    
        eliminated_blobs = []

        for i,blob in enumerate(blob_props):
    
          if blob.label in eliminated_blobs:
            continue
    
          # FIX TO DOCUMENTED CUTOFF BUG (KMN February 2020)
          # ====================================================================
    
          # first, determine if the blob is located at the edge of the domain 
          blob_lat_i = blob.coords[:,0] # the latitude indices of the blob
          blob_lon_i = blob.coords[:,1] # the longitude indices of the blob
          if max_lon_index in blob_lon_i: # the blob touches the boundary
            edge_i = np.where(blob_lon_i == max_lon_index)[0]
            edge_lat_i = blob_lat_i[edge_i]
            edge_lon_i = blob_lon_i[edge_i]
            # now, see if a blob is touching the other end of the domain
            no_match = False
            for I, BLOB in enumerate(blob_props):
              BLOB_lon_i = BLOB.coords[:,1]
              BLOB_lat_i = BLOB.coords[:,0]
              blob_label = blob.label
              BLOB_label = BLOB.label
              if BLOB.label in eliminated_blobs:
                  continue
              if min_lon_index in BLOB_lon_i: # the blob touches the other boundary
                EDGE_i = np.where(BLOB_lon_i == min_lon_index)[0]
                EDGE_lat_i = BLOB_lat_i[EDGE_i]
                EDGE_lon_i = BLOB_lon_i[EDGE_i]
                edge_lat_i_shifted_p = edge_lat_i + 1
                edge_lat_i_shifted_m = edge_lat_i - 1
                A = set(EDGE_lat_i).intersection(set(edge_lat_i))
                B = set(EDGE_lat_i).intersection(set(edge_lat_i_shifted_p))
                C = set(EDGE_lat_i).intersection(set(edge_lat_i_shifted_m))
                if len(A) > 0 or len(B) > 0 or len(C) > 0: # means this blob must be touching other blob
                  # it has been established that the blobs touch and should be considered together
                  # to consider the two blobs as one, create a subset of the labels array that very likely only covers the separated blobs
                  # the chunk on the right edge needs to be concatenated with the chunk on the left edge
                  eliminated_blobs.append(int(blob_label)) # these blobs do not need to be considered again
                  eliminated_blobs.append(int(BLOB_label))
                  break
        
            label_array[BLOB_lat_i, BLOB_lon_i] = BLOB_label
            label_array[blob_lat_i, blob_lon_i] = blob_label # in case there was some prior elimination
            
            label_lat_i = np.where(label_array == blob_label)[0]
            label_lon_i = np.where(label_array == blob_label)[1]
            LABEL_lat_i = np.where(label_array == BLOB_label)[0]
            LABEL_lon_i = np.where(label_array == BLOB_label)[1]
      
            # now, determine the proper bounds around the feature
            left_bound_i = np.nanmin(label_lon_i)
            right_bound_i = np.nanmax(LABEL_lon_i)
            bottom_bound_i = np.nanmin(np.concatenate((label_lat_i, LABEL_lat_i)))
            top_bound_i = np.nanmax(np.concatenate((label_lat_i, LABEL_lat_i)))
    
            # extract the left side of the box
            left_side = label_array[bottom_bound_i:(top_bound_i+20), left_bound_i::]
            right_side = label_array[bottom_bound_i:(top_bound_i+20), 0:(right_bound_i+1)]
            iso_blobs = np.hstack((left_side,right_side))
      
            left_side = ivt_anom[bottom_bound_i:(top_bound_i+20), left_bound_i::]
            right_side = ivt_anom[bottom_bound_i:(top_bound_i+20), 0:(right_bound_i+1)]
            iso_blob_anom = np.hstack((left_side,right_side))
      
            left_side = area_array[bottom_bound_i:(top_bound_i+20), left_bound_i::]
            right_side = area_array[bottom_bound_i:(top_bound_i+20), 0:(right_bound_i+1)]
            iso_area_array = np.hstack((left_side,right_side))
      
            left_side = ivt_dir[bottom_bound_i:(top_bound_i+20), left_bound_i::]
            right_side = ivt_dir[bottom_bound_i:(top_bound_i+20), 0:(right_bound_i+1)]
            iso_ivt_dir = np.hstack((left_side,right_side))
      
            left_side = ivt_u_slice[bottom_bound_i:(top_bound_i+20), left_bound_i::]
            right_side = ivt_u_slice[bottom_bound_i:(top_bound_i+20), 0:(right_bound_i+1)]
            iso_ivt_u = np.hstack((left_side,right_side))
      
            left_side = ivt_v_slice[bottom_bound_i:(top_bound_i+20), left_bound_i::]
            right_side = ivt_v_slice[bottom_bound_i:(top_bound_i+20), 0:(right_bound_i+1)]
            iso_ivt_v = np.hstack((left_side,right_side))
      
            # filter the isolated blob array to make all non-blob values 0
            iso_blobs = np.where(iso_blobs==blob.label, 1, iso_blobs)
            iso_blobs = np.where(iso_blobs==BLOB.label, 1, iso_blobs)
            iso_blobs = np.where(iso_blobs==1, iso_blobs, 0)
        
            if np.nansum(iso_blobs) < 5: 
              label_array[label_array == blob_label] = 0
              label_array[label_array == BLOB_label] = 0
              continue
          
            iso_lats = data_lats[bottom_bound_i:(top_bound_i+20)]
            iso_lons = np.concatenate((data_lons[left_bound_i::], data_lons[0:(right_bound_i+1)]))
      
            if len(iso_lats) <= 3 or len(iso_lons) <=3: # otherwise, the box is too small for cubic interpolation
              label_array[label_array == blob_label] = 0
              label_array[label_array == BLOB_label] = 0
              continue
          
            iso_latFunc = interp.UnivariateSpline(np.arange(0, len(iso_lats), 1), iso_lats, s=0) 
      
            iso_lon_dist_1d = lon_dist_1d[bottom_bound_i:(top_bound_i+20)] # lon_dist_1d[y0] approx. km distance between lon # both functions of latitude
            iso_lat_dist_1d = lat_dist_1d[bottom_bound_i:(top_bound_i+20)] # lat_dist_1d[y0] approx. km distance between lat
            iso_londistFunc = interp.UnivariateSpline(np.arange(0, len(iso_lon_dist_1d), 1), iso_lon_dist_1d, s=0)
            iso_latdistFunc = interp.UnivariateSpline(np.arange(0, len(iso_lat_dist_1d), 1), iso_lat_dist_1d, s=0)
      
            # Using skimage to calculate blob characteristics
            iso_blob_props = regionprops(iso_blobs, intensity_image=iso_blob_anom) 
      
            # is this a cross-boundary blob?
            cross_bound_blob = True
    
          if cross_bound_blob:
        
            for iso_i, iso_blob in enumerate(iso_blob_props): # should effectively be only one iteration:
      
              logic_flag = 'na'
              y0, x0 = iso_blob.weighted_centroid

              # Orientation
              orientation = iso_blob.orientation
              orient_deg = int(-1*np.rad2deg(orientation)) # -1 since the array is actually flipped
              ivt_mean_orient = np.nanmean(np.where(iso_blobs == iso_blob.label, iso_ivt_dir, np.NaN))

              ivt_dir_means = calc_ivtdir(np.nanmean(np.where(iso_blobs == iso_blob.label, iso_ivt_u, np.NaN)),\
                          np.nanmean(np.where(iso_blobs == iso_blob.label, iso_ivt_v, np.NaN)))
              #print(orient_deg, ivt_mean_orient, ivt_dir_means)

              # Length
              length = iso_blob.major_axis_length
              min_row, min_col, max_row, max_col = iso_blob.bbox
              span = float((max_row-min_row) + (max_col-min_col))
              approx_len = length * (((max_col-min_col)/span)*iso_lon_dist_1d[int(y0)] + ((max_row-min_row)/span)*iso_lat_dist_1d[int(y0)])

              # Width, Area, Aspect Ratio, and Mean Intensity
              area = iso_blob.area
              width = iso_blob.minor_axis_length
              area_km2 = ndimage.sum(iso_area_array, iso_blobs, iso_blob.label)
              if width == 0.: width = 1.
              aspect = length/float(width)
              meanint = iso_blob.mean_intensity
      
              # remove blobs that are not of sufficient area (should have same impact as commented out code above)
              if area_km2 < size_mask:
                label_array[label_array == blob_label] = 0
                label_array[label_array == BLOB_label] = 0
                logic_flag = 'area'

              # Begin string of logic test to remove features that are not sufficiently AR-like...
              if y0 < grid_buffer or y0 > (nrows-grid_buffer):
                # Removing blobs that are centered close to domain edges (top/bottom)
                #if development: post_size[label_array == blob.label] = -1
                label_array[label_array == blob_label] = 0
                label_array[label_array == BLOB_label] = 0
                logic_flag = 'edge'
                edge_cnt += 1

              elif x0 < grid_buffer or x0 > (ncols-grid_buffer):
                # As above, but for the sides
                #if development: post_size[label_array == blob.label] = -1
                label_array[label_array == blob_label] = 0
                label_array[label_array == BLOB_label] = 0
                logic_flag = 'edge'
                edge_cnt += 1       

              elif approx_len < min_length:
                # Clearing the blobs that do not meet the minimum length threshold
                #if development: post_size[label_array == blob.label] = -1
                label_array[label_array == blob_label] = 0
                label_array[label_array == BLOB_label] = 0
                logic_flag = 'length'
                aspect_cnt += 1

              elif aspect < min_aspect:
                # Clearing the blobs that do not meet the aspect ratio threshold
                label_array[label_array == blob_label] = 0
                label_array[label_array == BLOB_label] = 0
                logic_flag = 'aspect'
                length_cnt += 1

              elif abs(latFunc(y0).tolist()) < lat_forceorient+5 and np.nanstd(np.where(iso_blobs == iso_blob.label, iso_ivt_dir, np.NaN)) > 100.:
                # Clearing low-lat blobs with a large spread in ivt direction (i.e., not a uniform river)
                label_array[label_array == blob_label] = 0
                label_array[label_array == BLOB_label] = 0
                logic_flag = 'subtropvariability'
                subtropvar_cnt += 1

              elif meanint < min_meanintensity:
                # Clearing the blobs that do not meet the mean intensity threshold
                label_array[label_array == blob_label] = 0
                label_array[label_array == BLOB_label] = 0
                logic_flag = 'intensity'
                intensity_cnt += 1

              elif abs(iso_latFunc(y0).tolist()) < lat_forceorient and iso_blob.eccentricity > min_eccentricity \
                 and iso_blob.mean_intensity > per_98 and iso_blob.extent > .50:
                # Attempting to remove solid, round, intense blobs that may be TCs or similar storms
                label_array[label_array == blob_label] = 0
                label_array[label_array == BLOB_label] = 0
                logic_flag = 'tc-like'
                tcish_cnt += 1

              elif abs(iso_latFunc(y0).tolist()) < lat_forceorient and (int(iso_blob.area) - iso_blob.filled_area) <= -5:
                # Removing blobs with sizeable holes (mostly TCs or intense ETCs)
                label_array[label_array == blob_label] = 0
                label_array[label_array == BLOB_label] = 0
                logic_flag = 'tc-hole'
                tchole_cnt += 1

              elif abs(iso_latFunc(y0).tolist()) <= lat_forceorient and ivt_mean_orient < 230. and ivt_mean_orient > 170. \
                 and (aspect < 4.5 or meanint < per_97 or area_km2 > 3*size_mask):
                # Further scrutinizing blobs in the tropics
                #   any easterly rivers must be quite river-like (vice low-intensity/broad swells)
                label_array[label_array == blob_label] = 0
                label_array[label_array == BLOB_label] = 0
                logic_flag = 'tropics'
                tropics_cnt += 1
            
              # done with cross-boundary blob examination
              cross_bound_blob = False
          
          # END OF FIX TO DOCUMENTED CUTOFF BUG (KMN February 2020)
          # ==================================================================================================
      
          else:    
                                    
            logic_flag = 'na'
            y0, x0 = blob.weighted_centroid

            # Orientation
            orientation = blob.orientation
            orient_deg = int(-1*np.rad2deg(orientation)) # -1 since the array is actually flipped
            ivt_mean_orient = np.nanmean(np.where(label_array == blob.label, ivt_dir, np.NaN))

            ivt_dir_means = calc_ivtdir(np.nanmean(np.where(label_array == blob.label, ivt_u_slice, np.NaN)),\
                        np.nanmean(np.where(label_array == blob.label, ivt_v_slice, np.NaN)))
            #print(orient_deg, ivt_mean_orient, ivt_dir_means)

            # Length
            length = blob.major_axis_length
            min_row, min_col, max_row, max_col = blob.bbox
            span = float((max_row-min_row) + (max_col-min_col))
            approx_len = length * (((max_col-min_col)/span)*lon_dist_1d[int(y0)] + ((max_row-min_row)/span)*lat_dist_1d[int(y0)])

            # Width, Area, Aspect Ratio, and Mean Intensity
            width = blob.minor_axis_length
            area = blob.area
            area_km2 = ndimage.sum(area_array, label_array, blob.label)
            if width == 0.: width = 1.
            aspect = length/float(width)
            meanint = blob.mean_intensity
      
            # remove blobs that are not of sufficient area (should have same impact as commented out code above)
            if area_km2 < size_mask:
              label_array[label_array == blob.label] = 0

            # Begin string of logic test to remove features that are not sufficiently AR-like...
            if y0 < grid_buffer or y0 > (nrows-grid_buffer):
              # Removing blobs that are centered close to domain edges (top/bottom)
              #if development: post_size[label_array == blob.label] = -1
              label_array[label_array == blob.label] = 0
              logic_flag = 'edge'
              edge_cnt += 1

            elif x0 < grid_buffer or x0 > (ncols-grid_buffer):
              # As above, but for the sides
              #if development: post_size[label_array == blob.label] = -1
              label_array[label_array == blob.label] = 0
              logic_flag = 'edge'
              edge_cnt += 1       

            elif approx_len < min_length:
              # Clearing the blobs that do not meet the minimum length threshold
              #if development: post_size[label_array == blob.label] = -1
              label_array[label_array == blob.label] = 0
              logic_flag = 'length'
              aspect_cnt += 1

            elif aspect < min_aspect:
              # Clearing the blobs that do not meet the aspect ratio threshold
              label_array[label_array == blob.label] = 0
              logic_flag = 'aspect'
              length_cnt += 1

            elif abs(latFunc(y0).tolist()) < lat_forceorient+5 and np.nanstd(np.where(label_array == blob.label, ivt_dir, np.NaN)) > 100.:
              # Clearing low-lat blobs with a large spread in ivt direction (i.e., not a uniform river)
              label_array[label_array == blob.label] = 0
              logic_flag = 'subtropvariability'
              subtropvar_cnt += 1

            elif meanint < min_meanintensity:
              # Clearing the blobs that do not meet the mean intensity threshold
              label_array[label_array == blob.label] = 0
              logic_flag = 'intensity'
              intensity_cnt += 1

            elif abs(latFunc(y0).tolist()) < lat_forceorient and blob.eccentricity > min_eccentricity \
               and blob.mean_intensity > per_98 and blob.extent > .50:
              # Attempting to remove solid, round, intense blobs that may be TCs or similar storms
              label_array[label_array == blob.label] = 0
              logic_flag = 'tc-like'
              tcish_cnt += 1

            elif abs(latFunc(y0).tolist()) < lat_forceorient and (int(blob.area) - blob.filled_area) <= -5:
              # Removing blobs with sizeable holes (mostly TCs or intense ETCs)
              label_array[label_array == blob.label] = 0
              logic_flag = 'tc-hole'
              tchole_cnt += 1

            elif abs(latFunc(y0).tolist()) <= lat_forceorient and ivt_mean_orient < 230. and ivt_mean_orient > 170. \
               and (aspect < 4.5 or meanint < per_97 or area_km2 > 3*size_mask):
              # Further scrutinizing blobs in the tropics
              #   any easterly rivers must be quite river-like (vice low-intensity/broad swells)
              label_array[label_array == blob.label] = 0
              logic_flag = 'tropics'
              tropics_cnt += 1

          # Checking for multiple peaks
          #   NOTE:  the testing of "segmented" blobs has been removed, but still counting for assessment
          temp_mask = np.zeros([nrows,ncols])
          temp_mask[label_array == blob.label] = 1
          peaks = peak_local_max(ivt_anom*temp_mask, min_distance=14, num_peaks=4)
          if len(peaks) > 1:
            multipeak_cnt += 1
    
        # save the ID and IVT arrays for the date-time
        mask = np.where(label_array != 0, 1, 0)
        ivt = np.where(label_array != 0, ivt_mag, 0)
        ivt_anom = np.where(label_array != 0, ivt_anom, 0)
        np.savez(outpath + run + '/ARTMIP_Tier2_Paleo_detect_ars_' + run + '_' + dtg_str + '.npz', mask=mask, ivt=ivt, ivt_anom=ivt_anom)

        print('file saved for ' + dtg_str)  
       
print('\nEnd of Program')

# ---------------------------------------------------------------------------------
