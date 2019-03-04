''' --------------------------------------------IMPORTING NECESSARY LIBRARIES------------------------------------------- '''
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from itertools import cycle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
start_time = time.time()
pd.options.mode.chained_assignment = None  # default='warn'


''' ---------------------------FUNCTIONS TO FIND NEAREST DISTANCE BETWEEN ALL NECESSARY STATIONS------------------------ '''
# Function to find nearest station between two points using Haversine Distance
def haversine_dist(lon1, lat1, lon2, lat2):
    # Calculate the great circle distance between two points on the earth
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2]) # Convert to radians
    # Haversine distance formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371   #Radius of earth in kilometers
    return c * r

# Find nearest AQ to AQ station
def near_aq_to_aq(lat, long):
    distances = station_aq.apply(lambda row: haversine_dist(lat, long, row['latitude'], row['longitude']), axis=1)
    distance = distances[distances!=0]
    return station_aq.loc[distance.idxmin(), 'station']

# Find nearest GW to GW station
def near_gw_to_gw(lat, long):
    distances = gw_station.apply(lambda row: haversine_dist(lat, long, row['latitude'], row['longitude']), axis=1)
    distance = distances[distances!=0]
    return gw_station.loc[distance.idxmin(), 'station_id']

# Find nearest OBW to OBW station
def near_obw_to_obw(lat, long):
    distances = obw_station.apply(lambda row: haversine_dist(lat, long, row['latitude'], row['longitude']), axis=1)
    distance = distances[distances!=0]
    return obw_station.loc[distance.idxmin(), 'station_id']

# Find nearest AQ to OBW station
def near_aq_to_obw(lat, long):
    distances = obw_station.apply(lambda row: haversine_dist(lat, long, row['latitude'], row['longitude']), axis=1)
    return obw_station.loc[distances.idxmin(), 'station_id']

# Find nearest AQ to GW station
def near_aq_to_gw(lat, long):
    distances = gw_station.apply(lambda row: haversine_dist(lat, long, row['latitude'], row['longitude']), axis=1)
    return gw_station.loc[distances.idxmin(), 'station_id']

# Function to calculate the model error via SMAPE
def smape(actual, predicted):
    dividend= np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator!=0, casting='unsafe'))



''' ------------------------------------------TRAIN: AIR QUALITY PREPROCESSING------------------------------------------ '''
print('Preprocessing and cleaning the train Air Quality Dataset!')
# Read all the air quality datasets
aq_2017 = pd.read_csv("airQuality_201701-201801.csv")
aq_2018 = pd.read_csv("airQuality_201802-201803.csv")
aq_2018a = pd.read_csv("aiqQuality_201804.csv")
# Renaming the header of April AQ dataset to match the other AQ datasets
aq_2018a.rename(columns={'station_id': 'stationId', 'time': 'utc_time', 'PM25_Concentration':'PM2.5'\
                   ,'PM10_Concentration':'PM10','NO2_Concentration':'NO2'\
                   ,'CO_Concentration':'CO', 'O3_Concentration':'O3'\
                   ,'SO2_Concentration':'SO2'}, inplace=True)
aq_2018a= aq_2018a.drop(columns=['id'], axis=1)
# Merge all AQ datasets together into a single dataframe
aq_train = aq_2017.append(aq_2018, ignore_index=True)
aq_train = aq_train.append(aq_2018a, ignore_index=True)
# Convert the entire 'utc_time' column into the same format
aq_train["utc_time"] = pd.to_datetime(aq_train["utc_time"])
# Delete unnecessary dataframes to save space
del(aq_2017)
del(aq_2018)
del(aq_2018a)

# Set the time column as the index of the dataframe
aq_train.set_index("utc_time", inplace = True)
# Get the entire span of the time in the AQ dataframe
min_date=aq_train.index.min()
max_date=aq_train.index.max()

# Drop any duplicates present in the AQ dataframe
aq_train.drop_duplicates(subset= None, keep= "first", inplace= True)

# Read the AQ station location file and find nearest station for each AQ station
# This dataset was created by us
station_aq = pd.read_csv("Beijing_AirQuality_Stations.csv")
station_aq["nearest_station"] = station_aq.apply(lambda row: near_aq_to_aq(row['latitude'], row['longitude']), axis=1)

# Create an empty dataframe with all hourly time stamps in the above found range
time_hours = pd.DataFrame({"date": pd.date_range(min_date, max_date, freq='H')})
# Perform a cartesian product of all AQ stations and the above dataframe
aq_all_time = pd.merge(time_hours.assign(key=0), station_aq.assign(key=0), on='key').drop('key', axis=1)

# Join the AQ dataset with the dataframe containing all the timestamps for each AQ station
aq_train1 = pd.merge(aq_train, aq_all_time,  how='right', left_on=['stationId','utc_time'], right_on = ['station','date'])
aq_train1 = aq_train1.drop('stationId', axis=1)
aq_train1.drop_duplicates(subset= None, keep= "first", inplace= True)

# Create a copy of the above dataframe keeping all required columns
# This dataframe will be used to refer all data for the nearest AQ station (same time interval)
aq_train_copy = aq_train1.copy()
aq_train_copy = aq_train_copy.drop(['nearest_station','longitude', 'latitude', 'type'], axis=1)
aq_train_copy.rename(columns={'PM2.5': 'n_PM2.5','PM10': 'n_PM10', "NO2":"n_NO2","CO":"n_CO","O3":"n_O3",
                        "SO2":"n_SO2", "date":"n_date", "station":"n_station" }, inplace=True)

# Merge original AQ data and the copy AQ data to get all attributes of a particular AQ station and its nearest AQ station
aq_train2 = pd.merge(aq_train1, aq_train_copy, how='left', left_on=['nearest_station','date'], right_on = ['n_station','n_date'])

# Sort the final dataframe based on AQ station and then time
aq_train2 = aq_train2.sort_values(by=['n_station', 'date'], ascending=[True,True])
aq_train2 = aq_train2.reset_index(drop=True)
# Drop all unncessary attributes
aq_train2.drop(['n_station', 'longitude', 'latitude', 'n_date'], axis=1, inplace=True)

# Create two attributes - month and hour
aq_train2['month'] = pd.DatetimeIndex(aq_train2['date']).month
aq_train2['hour'] = pd.DatetimeIndex(aq_train2['date']).hour

# Fill in missing values of attributes with their corresponding values in the nearest AQ station (within same time)
aq_train2['PM10'].fillna(aq_train2['n_PM10'], inplace=True)
aq_train2['PM2.5'].fillna(aq_train2['n_PM2.5'], inplace=True)
aq_train2['NO2'].fillna(aq_train2['n_NO2'], inplace=True)
aq_train2['CO'].fillna(aq_train2['n_CO'], inplace=True)
aq_train2['O3'].fillna(aq_train2['n_O3'], inplace=True)
aq_train2['SO2'].fillna(aq_train2['n_SO2'], inplace=True)
# Fill in any remaining missing value by the mean of the attribute within the same station, month and hour
aq_train2[['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']] = aq_train2.groupby(["station","month","hour"])[['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']].transform(lambda x: x.fillna(x.mean()))

# Create final AQ dataset after dropping all unnecessary attributes
aq_train_final = aq_train2.drop(['type','nearest_station','n_PM2.5','n_PM10','n_NO2','n_CO','n_O3','n_SO2'],axis=1)

# Delete unnecessary dataframes to save space
del(aq_train1)
del(aq_train2)
del(aq_train_copy)
del(aq_all_time)

print('Done!')
print('-'*50)



''' ------------------------------------------TRAIN: GRID DATASET PREPROCESSING------------------------------------------ '''
print('Preprocessing and cleaning the train Grid Weather Dataset!')
# Read all the grid weather train datasets
gw_2017 = pd.read_csv("gridWeather_201701-201803.csv")
gw_2018 = pd.read_csv("gridWeather_201804.csv")
# Renaming the headers of the GW data to match each other
gw_2017.rename(columns={'stationName': 'station_id', 'wind_speed/kph': 'wind_speed'}, inplace=True)
gw_2018.rename(columns={'station_id':'station_id', 'time':'utc_time'}, inplace=True)
# Merge all GW train datasets into a single dataframe
gw_train = gw_2017.append(gw_2018, ignore_index=True)
gw_train = gw_train.drop(columns=['id','weather'], axis=1)
# Delete unnecessary dataframes to save space
del(gw_2017)
del(gw_2018)

# Set the time column as the index of the dataframe
gw_train.set_index("utc_time", inplace = True)
# Get the entire span of the time in the GW dataframe
min_date = gw_train.index.min()
max_date = gw_train.index.max()

# Drop any duplicates present in the GW dataframe
gw_train.drop_duplicates(subset= None, keep= "first", inplace= True)

# Read the GW station location file and find nearest station for each GW station
gw_station = pd.read_csv("Beijing_grid_weather_station.csv", header=None, names=['station_id','latitude','longitude'])
gw_station["nearest_station"] = gw_station.apply(lambda row: near_gw_to_gw(row['latitude'], row['longitude']), axis=1)

# Create an empty dataframe with all hourly time stamps in the above found range
gw_time_hours = pd.DataFrame({"time": pd.date_range(min_date, max_date, freq='H')})
# Perform a cartesian product of all GW stations and the above dataframe
gw_all_time = pd.merge(gw_time_hours.assign(key=0), gw_station.assign(key=0), on='key').drop('key', axis=1)
gw_all_time['time'] = gw_all_time['time'].astype(str) # Make all time stamps in the same format
# Join the GW dataset with the dataframe containing all the timestamps for each GW station
gw_train1 = pd.merge(gw_train, gw_all_time,  how='right', left_on=['station_id','utc_time'], right_on = ['station_id','time'])
gw_train1.drop_duplicates(subset= None, keep= "first", inplace= True)

# Create a copy of the above dataframe keeping all required columns
# This dataframe will be used to refer all data for the nearest GW station (same time interval)
gw_train_copy = gw_train1.copy()
gw_train_copy.drop(['nearest_station','longitude_x', 'latitude_y','latitude_x','longitude_y'], axis=1, inplace=True)
gw_train_copy.rename(columns={'humidity': 'n_humidity','pressure': 'n_pressure', "temperature":"n_temperature",\
                              "wind_direction":"n_wind_dir","wind_speed":"n_wind_speed",\
                              "time":"n_time", "station_id":"n_station_id" }, inplace=True)

# Merge original GW data and the copy GW data to get all attributes of a particular GW station and its nearest GW station
gw_train2 = pd.merge(gw_train1, gw_train_copy, how='left', left_on=['nearest_station','time'], right_on = ['n_station_id','n_time'])

# Sort the final dataframe based on GW station and then time
gw_train2 = gw_train2.sort_values(by=['station_id', 'time'], ascending=[True,True])
gw_train2 = gw_train2.reset_index(drop=True)
# Drop all unncessary attributes
gw_train2.drop(['n_station_id', 'n_time','longitude_x', 'latitude_y','latitude_x','longitude_y'], axis=1, inplace=True)

# Create two attributes - month and hour
gw_train2['month'] = pd.DatetimeIndex(gw_train2['time']).month
gw_train2['hour'] = pd.DatetimeIndex(gw_train2['time']).hour

# Fill in missing values of attributes with their corresponding values in the nearest GW station (within same time)
gw_train2['humidity'].fillna(gw_train2['n_humidity'], inplace=True)
gw_train2['pressure'].fillna(gw_train2['n_pressure'], inplace=True)
gw_train2['temperature'].fillna(gw_train2['n_temperature'], inplace=True)
gw_train2['wind_speed'].fillna(gw_train2['n_wind_speed'], inplace=True)
gw_train2['wind_direction'].fillna(gw_train2['n_wind_dir'], inplace=True)
# Fill in any remaining missing value by the mean of the attribute within the same station, month and hour
gw_train2[['humidity', 'pressure', 'temperature', 'wind_direction', 'wind_speed']] = gw_train2.groupby(["station_id","month","hour"])[['humidity', 'pressure', 'temperature', 'wind_direction', 'wind_speed']].transform(lambda x: x.fillna(x.mean()))

# Create final GW dataset after dropping all unnecessary attributes
gw_train_final = gw_train2.drop(['nearest_station','n_humidity','n_pressure','n_temperature','n_wind_dir','n_wind_speed'],axis=1)

# Delete unnecessary dataframes to save space
del(gw_train1)
del(gw_train2)
del(gw_train_copy)
del(gw_all_time)

print('Done!')
print('-'*50)



''' -----------------------------------TRAIN: OBSERVED WEATHER DATASET PREPROCESSING------------------------------------ '''
print('Preprocessing and cleaning the train Observed Weather Dataset!')
# Read all the observed weather train datasets
obw_2017 = pd.read_csv("observedWeather_201701-201801.csv")
obw_2018 = pd.read_csv("observedWeather_201802-201803.csv")
obw_2018a = pd.read_csv("observedWeather_201804.csv")
obw_2018a.rename(columns={'time': 'utc_time'}, inplace=True)
# Read the time stamp in the April observed weather data in the same format as the other datasets
#obw_2018a['utc_time'] = pd.to_datetime(obw_2018a['utc_time'], format='%d-%m-%Y %H:%M:%S')
obw_2018a['utc_time'] = obw_2018a['utc_time'].astype(str)
# Merge all OBW train datasets into a single dataframe
obw_train = obw_2017.append(obw_2018, ignore_index=True)
obw_train = obw_train.append(obw_2018a, ignore_index=True)
obw_train.drop(['id','weather'],axis=1, inplace=True) # Drop unnecessary columns
# Delete unnecessary dataframes to save space
del(obw_2017)
del(obw_2018)
del(obw_2018a)

# Set the time column as the index of the dataframe
obw_train.set_index("utc_time", inplace = True)
# Get the entire span of the time in the OBW dataframe
min_date = obw_train.index.min()
max_date = obw_train.index.max()

# Drop any duplicates present in the OBW dataframe
obw_train.drop_duplicates(subset= None, keep= "first", inplace= True)

# Read the OBW station location file
obw_station = obw_train[["station_id","latitude","longitude"]]
obw_station = obw_station.drop_duplicates().dropna()
obw_station = obw_station.reset_index(drop=True)
# Find nearest station for each OBW station
obw_station["nearest_station"] = obw_station.apply(lambda row: near_obw_to_obw(row['latitude'], row['longitude']), axis=1)

# Create an empty dataframe with all hourly time stamps in the above found range
obw_time_hours = pd.DataFrame({"time": pd.date_range(min_date, max_date, freq='H')})
# Perform a cartesian product of all OBW stations and the above dataframe
obw_all_time = pd.merge(obw_time_hours.assign(key=0), obw_station.assign(key=0), on='key').drop('key', axis=1)
obw_all_time['time'] = obw_all_time['time'].astype(str) # Make all time stamps in the same format
# Join the OBW dataset with the dataframe containing all the timestamps for each OBW station
obw_train1 = pd.merge(obw_train, obw_all_time,  how='right', left_on=['station_id','utc_time'], right_on = ['station_id','time'])
obw_train1.drop_duplicates(subset= None, keep= "first", inplace= True)

# Create a copy of the above dataframe keeping all required columns
# This dataframe will be used to refer all data for the nearest OBW station (same time interval)
obw_train_copy = obw_train1.copy()
obw_train_copy.drop(['nearest_station','longitude_x', 'latitude_x','longitude_y', 'latitude_y'], axis=1, inplace=True)
obw_train_copy.rename(columns={'humidity': 'n_humidity','pressure': 'n_pressure', "temperature":"n_temperature",\
                              "wind_direction":"n_wind_dir","wind_speed":"n_wind_speed",\
                              "time":"n_time", "station_id":"n_station_id" }, inplace=True)

# Merge original OBW data and the copy OBW data to get all attributes of a particular OBW station and its nearest OBW station
obw_train2 = pd.merge(obw_train1, obw_train_copy, how='left', left_on=['nearest_station','time'], right_on = ['n_station_id','n_time'])

# Sort the final dataframe based on OBW station and then time
obw_train2 = obw_train2.sort_values(by=['station_id', 'time'], ascending=[True,True] )
obw_train2.drop(['n_station_id', 'n_time'], axis=1, inplace=True)
obw_train2 = obw_train2.reset_index(drop=True)

# Create two attributes - month and hour
obw_train2['month'] = pd.DatetimeIndex(obw_train2['time']).month
obw_train2['hour'] = pd.DatetimeIndex(obw_train2['time']).hour

# Fill in missing values of attributes with their corresponding values in the nearest OBW station (within same time)
obw_train2['humidity'].fillna(obw_train2['n_humidity'], inplace=True)
obw_train2['pressure'].fillna(obw_train2['n_pressure'], inplace=True)
obw_train2['temperature'].fillna(obw_train2['n_temperature'], inplace=True)
obw_train2['wind_speed'].fillna(obw_train2['n_wind_speed'], inplace=True)
obw_train2['wind_direction'].fillna(obw_train2['n_wind_dir'], inplace=True)
# Fill in any remaining missing value by the mean of the attribute within the same station, month and hour
obw_train2[['humidity', 'pressure', 'temperature', 'wind_direction', 'wind_speed']] = obw_train2.groupby(["station_id","month","hour"])[['humidity', 'pressure', 'temperature', 'wind_direction', 'wind_speed']].transform(lambda x: x.fillna(x.mean()))

# Create final OBW dataset after dropping all unnecessary attributes
obw_train_final = obw_train2.drop(['longitude_x', 'latitude_x','longitude_y', 'latitude_y','nearest_station',\
                                   'n_humidity','n_pressure','n_temperature','n_wind_dir','n_wind_speed'],axis=1)

# Delete unnecessary dataframes to save space
del(obw_train1)
del(obw_train2)
del(obw_train_copy)
del(obw_all_time)

print('Done!')
print('-'*50)



''' --------------------------MERGING ALL TRAINING DATASETS AND GETTING READY FOR MODEL TRAINING------------------------- '''
aq_train_final['date'] = aq_train_final['date'].astype(str)
print('Getting the training model ready!')
# Convert wind speed in grid weather data from kmph to m/s (observed weather data is already in m/s)
gw_train_final['wind_speed'] = (gw_train_final['wind_speed']*5)/18

# Make all start and end times equal for the training datasets
gw_train_final = gw_train_final[gw_train_final['time']>='2017-01-30 16:00:00']
aq_train_final = aq_train_final[aq_train_final['date']>='2017-01-30 16:00:00']

# Replace noise values with previous hours value in both Observed and Grid datasets
obw_train_final.replace(999999,np.NaN,inplace=True)
obw_train_final[['humidity', 'pressure','temperature','wind_direction','wind_speed']] = obw_train_final[['humidity', 'pressure','temperature','wind_direction','wind_speed']].fillna(method='ffill')
gw_train_final.replace(999999,np.NaN,inplace=True)
gw_train_final[['humidity', 'pressure','temperature','wind_direction','wind_speed']] = gw_train_final[['humidity', 'pressure','temperature','wind_direction','wind_speed']].fillna(method='ffill')

# Replace wind direction with the noise value '999017' when wind speed is less than 0.5m/s
# This value will then be replaced with data from the nearest observed or grid station for the same timestamp
obw_train_final.loc[obw_train_final.wind_speed < 0.5, 'wind_direction'] = 999017
gw_train_final.loc[gw_train_final.wind_speed < 0.5, 'wind_direction'] = 999017

# Find nearest OBW and GW station for every AQ station for proper joining of attributes
obw_station.drop(['nearest_station'],axis=1, inplace=True)
station_aq["near_obw"] = station_aq.apply(lambda row: near_aq_to_obw(row['latitude'], row['longitude']), axis=1)
gw_station.drop(['nearest_station'],axis=1, inplace=True)
station_aq["near_gw"] = station_aq.apply(lambda row: near_aq_to_gw(row['latitude'], row['longitude']), axis=1)

# Merge the AQ training dataset with the nearest OBW and GW stations for every time stamp
aq_train1 = pd.merge(aq_train_final, station_aq, how='left', on='station')
aq_train1.drop(['type','nearest_station'], axis=1, inplace=True)

# Append all GW data attributes with the AQ training set based on nearest GW station and time stamp
aq_train2 = pd.merge(aq_train1, gw_train_final, how='left', left_on=['near_gw','date'], right_on=['station_id','time'])
# Remove unnecessary columns and rename columns to prepare for merging of OBW data
aq_train2.drop(['station_id','time','month_y','hour_y'],axis=1, inplace=True)
aq_train2 = aq_train2.rename(columns={'month_x': 'month_aq', 'hour_x': 'hour_aq', 'longitude':'longitude_aq',\
                                      'latitude':'latitude_aq', 'humidity': 'humidity_gw','pressure': 'pressure_gw',\
                                     'wind_direction': 'wind_dir_gw', 'wind_speed':'wind_speed_gw',\
                                      'temperature': 'temperature_gw'})

# Append all OBW data attributes with the AQ training set based on nearest OBW station and time stamp
TRAIN = pd.merge(aq_train2, obw_train_final, how='left', left_on=['near_obw','date'], right_on=['station_id','time'])
TRAIN.drop(['station_id','time','month','hour'],axis=1, inplace=True)
TRAIN = TRAIN.rename(columns={'humidity': 'humidity_obw','pressure': 'pressure_obw',\
                                     'wind_direction': 'wind_dir_obw', 'wind_speed':'wind_speed_obw',\
                                      'temperature': 'temperature_obw'})

# Final clean of all 999017 noise from the OBW and GW for wind direction
TRAIN.loc[TRAIN.wind_dir_gw == 999017, 'wind_dir_gw'] = TRAIN['wind_dir_obw']
TRAIN.loc[TRAIN.wind_dir_obw == 999017, 'wind_dir_obw'] = TRAIN['wind_dir_gw']

# Some observed data points are very outliers (probably wrongly noted by humans)
TRAIN.loc[TRAIN.humidity_obw > 100, 'humidity_obw'] = TRAIN['humidity_gw']
TRAIN.loc[TRAIN.pressure_obw > 1040, 'pressure_obw'] = TRAIN['pressure_gw']
TRAIN.loc[TRAIN.temperature_obw > 50, 'temperature_obw'] = TRAIN['temperature_gw']
TRAIN.loc[TRAIN.wind_dir_obw > 360, 'wind_dir_obw'] = TRAIN['wind_dir_gw']
TRAIN.loc[TRAIN.wind_speed_obw > 20, 'wind_speed_obw'] = TRAIN['wind_speed_gw']

# Sort the final train set based on station and then timestamp
TRAIN = TRAIN.sort_values(by=['station', 'date'], ascending=[True,True])

print('Ready to be trained by the model!')
print('-'*50)



''' ----------------------TEST DATA: CLEANING, PREPROCESSING AND GETTING READY FOR MODEL-------------------------------- '''
print('Getting the testing data ready for the model!')
# Read the AQ test dataset for test data - This dataset was found from the Beijing meteorological datasets
# This dataset helps in getting the values for the NO2, SO2 and CO attributes for the test data timestamps
test_aq = pd.read_csv('MAY_AQ.csv')
test_aq['Time'] = pd.to_datetime(test_aq['Time'], format='%d-%m-%Y  %H:%M')
test_aq['Time'] = test_aq['Time'].astype(str)

# Merge the dataset with nearest GW and OBW stations with the AQ test dataset
test1 = pd.merge(test_aq, station_aq, how='left', left_on='station_id', right_on='station').drop(['station','longitude','latitude','type','nearest_station','AQI'],axis=1)

# Find time stamp range for test data: from 1st May 00:00 to 2nd May 23:00
test1.set_index("Time", inplace = True)
min_date_test = test1.index.min()
max_date_test = test1.index.max()
test1.reset_index(inplace=True)

# Grid Test Data Preprocessing
test_gw = pd.read_csv('gridWeather_20180501-20180502.csv') # Read GW test data
test_gw.drop(['id','weather'],axis=1, inplace=True)
# Create new dataframe with all timestamps for all GW stations
test_gw1 = pd.DataFrame({"time": pd.date_range(min_date_test, max_date_test, freq='H')})
test_gw2 = pd.merge(test_gw1.assign(key=0), gw_station.assign(key=0), on='key').drop('key', axis=1)
test_gw2['time'] = test_gw2['time'].astype(str) # Convert time in correct format
gw_test_final = pd.merge(test_gw2, test_gw,  how='left', left_on=['station_id','time'], right_on = ['station_id','time'])

# Observed Test Data Preprocessing
test_obw = pd.read_csv('observedWeather_20180501-20180502.csv') # Read OBW test data
test_obw.drop(['id','weather'],axis=1, inplace=True)
# Create new dataframe with all timestamps for all OBW stations
test_obw1 = pd.DataFrame({"time": pd.date_range(min_date, max_date, freq='H')})
test_obw2 = pd.merge(test_obw1.assign(key=0), obw_station.assign(key=0), on='key').drop('key', axis=1)
test_obw2['time'] = test_obw2['time'].astype(str) # Convert time in correct format
obw_test_final = pd.merge(test_obw2, test_obw,  how='left', left_on=['station_id','time'], right_on = ['station_id','time'])

# Join AQ Test dataframe with test GW dataframe
test_aq1 = pd.merge(test1, gw_test_final, how='left', left_on=['near_gw','Time'], right_on=['station_id','time'])
test_aq1.drop(['station_id_y','latitude','longitude'],axis=1, inplace=True)
# Rename certain columns to prepare for joining the OBW test dataframe
test_aq1 = test_aq1.rename(columns={'station_id_x':'station_id_aq',\
                                     'humidity': 'humidity_gw',\
                                     'pressure': 'pressure_gw',\
                                     'wind_direction': 'wind_dir_gw',\
                                     'wind_speed':'wind_speed_gw',\
                                     'temperature': 'temperature_gw'})

# Join the updated AQ Test dataframe with test OBW dataframe
TEST = pd.merge(test_aq1, obw_test_final, how='left', left_on=['near_obw','time'], right_on=['station_id','time'])
TEST.drop(['station_id','latitude','longitude','time'],axis=1, inplace=True)
# Rename certain columns
TEST = TEST.rename(columns={'humidity': 'humidity_obw',\
                                     'pressure': 'pressure_obw',\
                                     'wind_direction': 'wind_dir_obw',\
                                     'wind_speed':'wind_speed_obw',\
                                     'temperature': 'temperature_obw'})

# Create attributes for month and hour - to be taken as input parameters
TEST['month'] = pd.DatetimeIndex(TEST['Time']).month
TEST['hour'] = pd.DatetimeIndex(TEST['Time']).hour
# Remove missing values based on nearest GW data (as very few values are missing in OBW data)
TEST = TEST.sort_values(by=['station_id_aq', 'Time'], ascending=[True,True])
TEST['humidity_obw'] = TEST['humidity_obw'].fillna(TEST['humidity_gw'])
TEST['temperature_obw'] = TEST['temperature_obw'].fillna(TEST['temperature_gw'])
TEST['pressure_obw'] = TEST['pressure_obw'].fillna(TEST['pressure_gw'])
TEST['wind_speed_obw'] = TEST['wind_speed_obw'].fillna(TEST['wind_speed_gw'])
TEST['wind_dir_obw'] = TEST['wind_dir_obw'].fillna(TEST['wind_dir_gw'])

# Take care of noise 999017 when wind speed is less than 0.5m/s
TEST.loc[TEST.wind_dir_gw == 999017, 'wind_dir_gw'] = TEST['wind_dir_obw']
TEST.loc[TEST.wind_dir_obw == 999017, 'wind_dir_obw'] = TEST['wind_dir_gw']

print('Ready to be tested by the model!')



''' ---------------------------------TRAINING THE MODEL AND PREDICTING REQUIRED OUTPUT----------------------------------- '''
# Train the model with only April, May and June's data
TRAIN = TRAIN.loc[TRAIN['month_aq'].isin([4,5,6])]
# Extract output columns for training the model
Y = TRAIN[['PM2.5','PM10','O3']].values
# Input parameters for the model
X = TRAIN.drop(['PM2.5','PM10','O3','latitude_aq','longitude_aq'], axis=1)
# Create new features for the model
X['AQ'] = (X['SO2']*X['NO2']*X['CO'])
X['wind'] = X['wind_dir_gw']/X['wind_speed_gw']
# Final input parameters after feature engineering
X_train = X[['station','month_aq','hour_aq','temperature_gw','AQ','humidity_gw','wind','pressure_gw']].values
# One Hot encode the station column and normalize the entire input data
le = LabelEncoder()
ohe = OneHotEncoder(categorical_features=[0])
scaler = MinMaxScaler()
X_train[:,0] = le.fit_transform(X_train[:,0])
X_train = ohe.fit_transform(X_train).toarray()
X_train_sc = scaler.fit_transform(X_train)
# Use Random Forest Regressor to predict the values
model_rf = RandomForestRegressor(random_state=42)
# Use K Fold Cross Validation to check the efficiency of the model
print('-------Printing the Cross Validation SMAPE errors-------')
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X_train_sc):
    x_train, x_val = X_train_sc[train_index], X_train_sc[test_index]
    y_train, y_val = Y[train_index], Y[test_index]
    model_rf.fit(x_train, y_train)
    pred_val = model_rf.predict(x_val)
    print(smape(y_val,pred_val))

# Get the Test data ready for the model by following the above steps
TEST['AQ'] = (TEST['CO']*TEST['SO2']*TEST['NO2'])
TEST['wind'] = TEST['wind_dir_gw']/TEST['wind_speed_gw']
# Final test data input features
X_test = TEST[['station_id_aq','month','hour','temperature_gw','AQ','humidity_gw','wind','pressure_gw']].values
# One hot encode and normalize similair to train data
X_test[:,0] = le.transform(X_test[:,0])
X_test = ohe.transform(X_test).toarray()
X_test_sc = scaler.transform(X_test)

# Predict the results after training the model on the whole final train dataset
model_rf.fit(X_train_sc,Y)
pred = model_rf.predict(X_test_sc)



''' --------------------------EXPORTING THE PREDICTED RESULTS INTO THE SPECIFIED FORMAT---------------------------------- '''
index_test = TEST[['station_id_aq']]
index = list(range(0,48)) # Create a list with all the values in the range (each for one hour over a period of two days)
# Turn the above numbers into a continuous cycle
index1 = cycle(index)
index_test['index'] = [next(index1) for i in range(len(index_test))]
# Create a column with all 35 AQ station names and all time indexes
index_test['test_id'] = index_test['station_id_aq']+'#'+index_test['index'].astype(str)
# Extract the required column and join it with the predicted output
# Both test and train data are sorted by station name and time - hence predicted output will be in arranged order
index_test.drop(['index','station_id_aq'],axis=1, inplace=True)
index_test1 = index_test.values
output = np.concatenate((index_test1, pred), axis=1)
np.savetxt('submission.csv', output, delimiter=',', header='test_id,PM2.5,PM10,O3', fmt='%s,%f,%f,%f', comments='')
print('The code is complete - please find your results in the "submission.csv" file!')

print("--- %s seconds ---" % (time.time() - start_time))

'''-------------------------------------------------------END-------------------------------------------------------------'''
