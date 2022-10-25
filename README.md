# The Mundhenk AR detection Algorithm

### Physical Basis of Algorithm: 

The Mundhenk algorithm detects atmospheric rivers (ARs) in global fields of integrated water vapor transport (IVT) using criteria related to intensity and geometry. A detailed description of the algorithm is provided in Mundhenk et al. (2016), and later modifications are described in Ralph et al. (2019). Fundamentally, the algorithm first identifies grid cells of highly anomalous IVT and groups adjacent grid cells into “blobs” or candidate objects. The algorithm calculates geometric characteristics of each candidate object and assesses each candidate object with a series of geometric tests to remove any features that are not true ARs. If a candidate objects fails any of the tests, it is removed from consideration and the algorithm assesses the next candidate object. Those candidate objects that remain are considered “detected” AR features. Please refer to the works referenced above for specific details related to the thresholds and criteria used.

### Instructions for Running the Code: 

The algorithm package currently requires Python 3.7. It is up to the user to verify that all necessary Python modules are available in the environment. Two nonstandard modules are `netCDF4` and `scikit-image`. A user may also find `matplotlib` and `cartopy` helpful for post-analysis purposes. These modules can be downloaded via the command line with the conda-forge. An example is shown below.

```
mamba create -n mundhenk python==3.7.* netCDF4 scikit-image matplotlib cartopy pynco -c conda-forge
conda activate mundhenk
### NOTE: the optional script requires nco
mamba install nco
```

The algorithm package has four main Python scripts and four bash runscripts. The first Python script is a preprocessing script that reads in global IVT data and calculates a smoothed seasonal cycle. The second Python script is also a preprocessing script that uses the smoothed seasonal cycle to calculate IVT anomalies. The distribution of these anomalies is then used to calculate the intensity thresholds later on. These two preprocessing scripts have accompanying runscripts that allow the user to choose the dataset and call the relevant Python script. The third Python script is comprised of the algorithm itself. This script reads in the appropriate IVT data, calculates anomalies, sets appropriate thresholds, finds candidate objects, and assesses those candidate objects. This algorithm script also has an accompanying runscript. The fourth and final Python script is a postprocessing script that takes the numpy binary files output by the algorithm script and converts them into NetCDF files. This last script is optional depending on the desired use of the output and also has a runscript. Note that adapting code to a user’s workspace will require changing file paths!

#### Script #1: calc_fft.py

1.	The first step is to define the specifications of the model (or other dataset). This step is necessary so that the script can correctly identify the dataset of interest. The contents of this CSV file are dependent on the dataset used. As a general rule, include all variable components of the input file name. Each combination of specifications should get its own row in the CSV file. Each variable specification (except for the date) should get its own list defined in the script. Then, each list is populated by reading in from the CSV file. These are then converted to numpy arrays of strings for analysis. In general, the main dataset name should be the first column, and subsequent columns should be specifications that branch off the first column. The associated runscript will tell the Python script which model/dataset to analyze. Then, the Python script will run through the various specifications to cover all possible combinations. The user is responsible for adding loops for each specification (if there are multiple possibilities). Here is an example:

Suppose you are interested in running the algorithm on reanalysis data from three reanalysis datasets (e.g., ERAI, MERRA2, and NCEP/NCAR). For each reanalysis, there is instantaneous and daily-average data (define this as “method” of storing data). Also, ERAI data comes in two resolutions (1.5 degrees and 2.5 degrees), while MERRA2 comes in three resolutions (0.5 degrees, 1 degree, and 1.5 degrees). Each file is formatted as follows: *dataset_method_res.nc* (e.g., *ERAI_inst_2.5.nc*). Then, the CSV file would look like this:

```
ERAI, inst, 1.5
ERAI, daily_avg, 1.5
ERAI, inst, 2.5
ERAI, daily_avg, 2.5
MERRA2, inst, 0.5
MERRA2, daily_avg, 0.5
MERRA2, inst, 1
MERRA2, daily_avg, 1
MERRA2, inst, 1.5
MERRA2, daily_avg, 1.5
NCEP_NCAR, inst, 2.5
NCEP_NCAR, daily_avg, 2.5
```

Within the Python script, there should be three lists (model/dataset, method, and resolution). The CSV reader should read in each row and make lists. Based on the dataset, the script should isolate the appropriate possible specifiers (in this case, the methods and resolutions associated with the given dataset). Then, the script should loop through each possible specifier (e.g., one loop for method and another for resolution). 

**If the input data is not named in this way, a “rename” command could be used to get the file names to fit this format. This is recommended practice. Alternatively, the scripts can be modified to fit the provided input data format. If doing this, it is highly recommended that you make any hardcoding modifications in a separate script.**

2.	For a given iteration of the (nested) loop, there will be a combination of specifiers that can be used to identify the appropriate input files. Here, the user should modify the input and output paths (*rawpath* and *outpath*, respectively) to reflect the file structure of the user’s system. The same should be done for the input file name base. At this point, the script takes a test file from the dataset and extracts the latitude and longitude coordinates. This is done to determine the size necessary for the array to store the IVT values. **It is important to be sure that the test file reflects the correct lat/lon grid, so specifiers should be used when finding a test file if there are expected differences in resolution within the dataset!**

3.	Next, there are three defined functions. The first function reads in a file path and obtains the files that meet the specifications. The path of interest should be modified based on the user’s file structure. Note that the *glob* function returns full file paths yet the function only retains the actual file name. To accomplish this, the function splits each file path with the slash (/) as a delimiter. Each split part of the path is placed into a list. The user must determine which index is appropriate to retain the file name.

The next two functions should not be modified. One function calculates the day of year index (0, 364) and accounts for (eliminates) leap days. The other function calculates the seasonal cycle using Fast Fourier Transform to calculate the smoothed seasonal cycle of IVT at each grid cell. This smoothed seasonal cycle consists of three harmonics (the daily mean and the first two harmonics of the time series). **The number of harmonics retained should remain the same unless there is significant justification otherwise.**

4.	For each combination of specifiers, the script finds all files that meet the specifications. The script next initializes a 3D array to store summed IVT and a counter for each grid cell and day of year. Then, the script loops through each applicable file and extracts the zonal (u) and meridional (v) IVT fields as well as the time array. After calculating the total IVT from the components, the script initializes a temporary array to store values of IVT for each time step so that a daily mean can be calculated at each grid cell. **Note that the number of time steps per day needs to be hardcoded by the user. This value depends on the temporal resolution of the data.**

5.	For each time within the time array, a date is derived using the *num2date* function, which takes the numerical time value from the netCDF file and derives the relevant date based on the attributes (e.g., units) within the input netCDF file. If a leap year tag is identified, the present iteration is skipped. Then, the temporary daily array of IVT values is populated based on the time of day. If the time step is the last of the day (assuming standard 3/6-hourly steps of 00, 06, 12, 18 UTC), a mean for the given day is derived from this temporary array, and this value is added to the array of IVT sums. The counter array is also increased by one. This process is repeated for each day in the file. **This part of the code would need to be changed if using a different time resolution!**

6.	Next, the summed IVT and counter arrays are used to calculate the average of daily mean IVT for each grid cell and day of year. The FFT function is then called to calculate the smoothed seasonal cycle of IVT for each grid cell. The last two numerical inputs of the FFT function should not be changed. This function returns a 3D array of smoothed seasonal cycle (day of year x lats x lons). This numpy array is saved as a numpy binary file.

**Runscript Note**: The runscript user is responsible for changing the batch submissions arguments to reflect the correct system structure. This is the case for the other two runscripts as well.

#### Script #2: calc_percentiles.py

1.	Like the FFT calculation script, the first step of this script is to read in the appropriate model/dataset specifications. Also like before, the script reads in a test file in order to get the correct lat/lon coordinates. 

2.	Next, the script defines two functions that were used in the FFT script. The first returns a list of files meeting the given specifications as before, while the second calculates the true day of year while accounting for leap years (again, as before).

3.	The script then defines the latitude and longitude bounds of the region over which to calculate the IVT anomaly distribution for later use as intensity thresholds in the algorithm. The longitude bounds must be between 0 and 360 degrees E, while the latitude bounds must be between -90 and 90 degrees N. The user is free to use any bounds that are deemed appropriate for the study. **Note that these bounds only apply to the region over which the intensity threshold is calculated. The algorithm is still run over the entire globe!** The script then isolates these subset latitudes and longitudes. 

4.	As with the FFT calculations, the script obtains a list of available files matching the dataset specifications and loops over all of these files. This script also reads in the smoothed seasonal cycle for each grid cell within the subset domain. A 2D array is initialized to store the IVT values at each time step. At each time step, a new 2D array will be stacked onto the existing array, resulting in a 3D array. 

5.	For each applicable input file, full IVT is calculated at each grid cell and date covered by the input file. As before, a temporary array to store daily values of IVT is initialized. Also as before, the script loops through each time step within the file, obtains the correct date, filters out leap days, and obtains the IVT field at that time step. This time slice of IVT is then subset according to the defined lat/lon subset bounds. The subset full IVT field is then placed into the temporary daily array according to the time of day. At the last time step of the day, daily averaging over the temporary daily array takes place. The subset IVT seasonal cycle value for the day of year is subtracted from the subset full daily-mean IVT field to calculate the daily-mean IVT anomaly for each subset grid cell at each time step. The end result is a 3D array of subset daily-mean IVT anomaly values for each time step.

6.	Finally, the script calculates the 90-98th percentiles of the time series distribution of IVT anomaly values for all subset grid cells combined. Some of these percentile values will be used as thresholds in the detection algorithm script. These percentiles are printed to the screen and saved as text and numpy binary files. 

#### Script #3: detect_ars_correctorient.py

1.	As with the first two, this script reads in the model specifications based on the inputs from the runscript. As before, the script loops through the model specifications (e.g., type) and defines the input/output paths depending on the specifications. The script then searches the output directory (which must already exist, either by manually creating it beforehand or by using a Python command) and compiles a list of all existing output files. This is an important step if restarting the algorithm with partial output already in the output directory.

2.	The script next defines the grid. This process involves defining the latitude/longitude coordinates and converting the grid spacing into an approximate distance in kilometers. The recommended method is to read these values in from an existing IVT NetCDF file for the dataset. This is necessary in order to measure the size of the candidate objects later in the process. Other steps include defining the number of grid points around the spatial boundary, which is needed to avoid detecting features too close to the periphery. **At this moment, this part should not be touched!**

3.	Next, the script reads in the anomaly percentiles from the appropriate numpy binary file and assigns values to the thresholds based on the necessary percentile value. IVT anomaly threshold to label a grid cell as part of a potential AR is the 94th percentile of IVT anomalies. The mean intensity threshold for a candidate object is the 95th percentile. The 97th and 98th percentiles are used to flag candidate objects that are potentially more tropical in nature (and not ARs by this algorithm’s definition). Note that the user must define the model specifications (e.g., time period or “type”) used for anomaly threshold calculations. Depending on the application, the specifications used for the anomaly thresholds do not need to be the same as the specifications over which the detection is taking place. 

  For example, suppose that you are running the algorithm to detect AR features in historical and future runs of a climate model. For consistency, you may wish to base AR detection in a future climate to IVT anomaly thresholds from a historical period. In this case, the historical percentile file must be specified here. 

  After defining intensity thresholds, the script sets the geometric criteria for object area, length, aspect ratio (length to width), eccentricity (how “circular” the is the object?), and latitude cutoff for possible labeling as tropical features. **These should not be changed unless absolutely necessary for the purposes of your study. If changing these values, be sure to document the changes, as these specific numbers are documented in prior citations.**
 
4.	The script then defines the necessary functions. The first function retrieves a list of available files of the model/dataset specifications (as before). The next function skeletonizes the candidate object (“blob”) so that the object’s features can be more accurately measured. (**Do not modify this function!**) The next two functions use the zonal and meridional IVT components and calculates the direction and magnitude of the IVT vector. The final function calculates the true day of year while accounting for leap years (as before). 

5.	At this point, the main algorithm code begins. As before, the script first compiles a list of applicable input files. Then, the smoothed seasonal cycle used for anomaly calculations is defined. The specifications used for the seasonal cycle are assumed to be the same as those used for the anomaly threshold values. Here, the script also initializes lists and counters to keep track of detected candidate objects and metrics such as reasons for candidate object rejection.

6.	The script then loops through each available input file and reads in the IVT components and the time coordinate. The script then loops through each time step within the input file and determines the true date, day of year, and time of day (excluding leap days in the process). The script will check to see if an output file for the model specifications and time step already exists. If so, the current time step is skipped. The IVT time slice is calculated for each component, and the IVT magnitude and direction are calculated from these components. From this calculated full IVT magnitude, an anomaly value is derived from the seasonal cycle array. At this point, the algorithm is ready to sweep through the IVT field and identify those grid cells that are part of candidate AR objects.

7.	The script sweeps through the field of IVT anomalies and finds those grid cells with anomaly values exceeding the threshold (i.e., the 94th percentile of IVT anomalies). Those grid cells that do exceed the threshold get marked with a 1 (0 otherwise). The next line of code groups clusters of adjacent grid cells in candidate objects and gives them a unique label. 

8.	The next block of code calculates the area of each candidate object and removes those candidate objects that have an area less than the threshold. Any grid cells that are part of a retained candidate object keep a value of 1. Grid cells of rejected candidate objects get a new value of 0. 

9.	Now, the script calculates other geometric characteristics of the candidate objects and loops through each one. For each candidate object, the intensity-weighted centroid coordinates, feature orientation, mean IVT vector orientation, length, width, etc. are calculated. The algorithm continues through a list of geometric tests described below. Note that if the candidate object fails a test, it is removed from consideration (grid cells changed to values of 0). Otherwise, the candidate object moves onto the next test.

  a.	Is the feature centered too close to the edge of the domain? This is especially important for regional studies. If true, it is likely that the feature’s characteristics cannot be properly measured because part of the feature is outside of the domain.

  b.	Same as (a).

  c.	Is the candidate object’s length above the length threshold? If not, the feature is considered too short to be a true AR plume.
  
  d.	What is the object’s aspect ratio? If the aspect ratio is below the threshold, it is considered too stout and not like a plume (e.g., more of a lake than a river).
  
  e.	Is the object’s centroid too close to the equator? And does the object have too much variation in IVT direction. If both are true, the feature is considered to be more of a meandering area of tropical moisture (i.e., not a focused, uniform AR).
  
  f.	Does the candidate object’s mean IVT anomaly value meet the threshold (i.e., the 95th percentile of IVT anomalies)? If not, the feature does not have a wide enough distribution of high IVT anomalies.
  
  g.	Is the object centered close to the equator AND fairly circular AND very intense AND fairly extensive? If all are true, then the feature is considered to be a likely tropical cyclone (thus removed from consideration). **The extent criterion is a legacy criterion but needs to be explored further in future runs.** For instance, it may lead to no failures for global regions. For now, keep it as a criterion.
  
  h.	Is the candidate object centered close to the equator? And does it contained some grid cells within its perimeter that are not actually part of the object? In this case, the object is likely a tropical cyclone with an eye hole.
  
  i.	Finally, is the candidate object centered close to the equator AND easterly in nature AND NOT extremely long/narrow and intense? This final test only keeps remaining easterly low-latitude candidate objects that are extremely long, plume-like, and intense (hence the stricter aspect ratio, area, and intensity criteria). 

  Those candidate objects that have passed all tests are considered detected AR features. Note that there is a subsequent block of code that looks for candidate objects with multiple IVT anomaly peaks. Past versions of the algorithm separated these features, but that part of the algorithm was removed in the updated version. Nonetheless, the script still counts how many detected features have the trait.

10.	For the given time step, the script next outputs fields of IVT mask, full IVT, and IVT anomaly. The IVT mask is a binary field of 1’s and 0’s showing where detected ARs exist (1 for a detected AR grid cell, 0 otherwise). These three fields are saved as numpy binary files.

