#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Acer
#
# Created:     27/09/2022
# Copyright:   (c) Acer 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------


# Data Manipulation
import numpy as np
import pandas as pd
import random
#import math

# Files/OS
import os
import copy

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import logistic



# Benchmarking
import time

# Error Analysis
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


#changes text size
plt.rcParams.update({'font.size':23})


input_file=r'D:\PROJECTS\Humburg_Lake_Victoria\data\Winam_Gulf_Satellite_Data.csv'

df_original = pd.read_csv(input_file)#put in a diff folder to the py file
print(df_original.head())

#df=df_original#[(df_original > 0).all(1)]
df=df_original[(df_original["mean_monthly_temp"] > 0)]



#visualize the dataset
cols_to_plot = ["Year","month", "mean_monthly_temp",\
 'mean_monthly_rainfall', "littoral_mean_monthly_ndvi","littoral_vegetation_area","mean_monthly_ndbi","Bare_soil_area","mean_monthly_ndwi","water_area","hyacinth_mean_monthly_ndvi","hyacinth_area"]
i = 1


def time_vs_var(variable):
    year_var=dict({"year":[],"var":[]})
    for m in range(39):

        current_year=1984+m

        df_1=df[(df["Year"] == current_year)]

        df_2=df_1[(df_1[variable] != 0)]
        val=df_2[variable].tolist()
        try:
            val_mean=sum(val)/len(val)
            year_var["year"].append(current_year)
            year_var["var"].append(val_mean)
            print(current_year,)
        except:
            pass

    return year_var

attribute="hyacinth_area"
year_var=time_vs_var(attribute)
print(year_var)
##
plt.plot(year_var["year"],year_var["var"])
plt.title("Annual variation of {}".format(attribute))
plt.xlabel("year")
plt.ylabel("{} in km²".format(attribute))

plt.show()


#feature="hyacinth_mean_monthly_ndvi"
feature="hyacinth_area"



##print(df["Year"].tolist())




def func(x, a, b, c):
    return a * np.log(b * x) + c


def funcline(x, a, b):
    return a*x + b



for i in cols_to_plot:
    try:


        df_feature_1=df[(df[i]!=0)]
        df_feature_2=df_feature_1[(df_feature_1[feature] !=0)]
        on_x=df_feature_2[i].tolist()
        on_y=df_feature_2[feature].tolist()



        ydata = np.asarray(on_y, dtype=np.float32)
        xdata = np.asarray(on_x, dtype=np.float32)


        poptline, pcovline = curve_fit(funcline, xdata, ydata)



        gradient="{:.2f}".format(poptline[0])
        print(gradient)

        plt.scatter(on_x,on_y, label="data points")


        plt.plot(xdata, funcline(xdata, *poptline), 'r-', label="Fitted Curve, gadient= {}".format(gradient))

        plt.title("{} vs {}".format(feature,i))
        plt.xlabel(i)
        plt.ylabel("{} in km² ".format(feature))

        plt.legend()

        plt.show()
    except:
        pass













