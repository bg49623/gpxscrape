import gpxpy
import math
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
files = []
#Replace with local path of "activities" folder.
os.chdir("/Users/bguo/PycharmProjects/MachineLearning/activities")
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".gpx"):
        files.append(filename)
        continue
    else:
        continue

from pandas import DataFrame
pattern = '%Y-%m-%d %H:%M:%S'
columns = ['Longitude', 'Latitude', 'Altitude', 'Time', 'Speed']

df = pd.DataFrame()
for i in range(len(files)):
    gpx = gpxpy.parse(open(files[i]))
    track = gpx.tracks[0]
    segment = track.segments[0]

    data = []
    segment_length = segment.length_3d()
    for point_idx, point in enumerate(segment.points):
        data.append([point.longitude, point.latitude,
                     point.elevation, point.time, segment.get_speed(point_idx)])

    dfn = DataFrame(data, columns=columns)
    df = df.append(dfn, ignore_index = True)
    epoch = []
displacement = []

altitude = []
for i in range(len(df) - 1):
    if(i%10 == 0):
        displacement.append(math.sqrt(math.pow((df['Longitude'][i] - df['Longitude'][i+1]), 2) + math.pow((df['Latitude'][i] - df['Latitude'][i+1]),2)))

for i in range(len(df) - 1):
    if(i%10 == 0):
        altitude.append(df['Altitude'][i+1] - df['Altitude'][i])

from math import sin, cos, sqrt, atan2, radians
distance = []
elev = []
def function(lat1, lon1, lat2, lon2):
    R = 20925259.0163

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

df.to_csv(path_or_buf="ride.csv")

for i in range(len(df) - 1):
    distance.append(function(df['Latitude'][i], df['Longitude'][i], df['Latitude'][i+1], df['Longitude'][i+1]))

for i in range(len(df) - 1):
    elev.append(df['Altitude'][i+1] - df['Altitude'][i])
distance.append(1)
elev.append(1)
df['Distance'] = distance
df['Elev']=elev
df['Frac'] = df['Elev']/df['Distance']
df['Speed'] = df['Speed'] * 2.23694
df = df.drop(df.index[len(df)-1])

rando = random.sample(range(0, len(df)), 3000)

smalldf = df.ix[rando]


from scipy import stats
miles = df['Speed'] * 2.23694
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Frac'],miles)
line = slope*df['Frac']+intercept

import sklearn
import numpy as np
from pandas import DataFrame
import seaborn as sns
from matplotlib import colors
import numpy
import math

df.to_csv("jacob.csv")
x= df['Frac']
y = df['Speed'].apply(numpy.log)
plt.subplot(2, 1, 1)
plt.scatter(df['Frac'], miles, s = 3)
plt.plot(df['Frac'], line, color = 'orange')
plt.ylim(0,100)
plt.title("My Rides: Linear Fit")
plt.xlabel("Gradient")
plt.ylabel("Speed (Kilometers/hour)")
plt.subplot(2, 1, 2)
h = plt.hist2d(x,miles,bins = 150,cmap=plt.cm.BuPu)
plt.title("Heatmap of the Same Data")
plt.xlim(-0.015, 0.015)
plt.ylim(0, 75)
plt.xlabel ("Gradient")
plt.ylabel("Speed (Kilometers)")
plt.tight_layout()
cbar = plt.colorbar()
plt.show()
