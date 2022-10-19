#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Acer
#
# Created:     14/10/2022
# Copyright:   (c) Acer 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from matplotlib import pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size':23})

data=r'D:\PROJECTS\Humburg_Lake_Victoria\results\RMSE_per_input_seq.csv'
df_data = pd.read_csv(data)#put in a diff folder to the py file
print(df_data.head())



plt.plot(df_data["input_sequence"],df_data["RMSE"])
plt.title("RMSE affected by Seasonality of the data")
plt.xlabel("No of months")
plt.ylabel("RMSE")

plt.show()
