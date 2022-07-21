import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import numpy as np


def __init__(self):
    
    return

def dataframe_preparation(file_1,file_2):
    df_1=pd.read_csv(file_1,sep=',',decimal=',',index_col=0, low_memory=False)
    df_2=pd.read_csv(file_2,sep=',',decimal=',',index_col=0, low_memory=False)
    df=pd.concat([df_1,df_2])  #concatenate the 2 csv
    for i in range (len(df.DistrictCoolingFacility.values)):
        df.DistrictCoolingFacility.values[i]=float(df.DistrictCoolingFacility.values[i])
    for i in range (len(df.DistrictHeatingFacility.values)):
        df.DistrictHeatingFacility.values[i]=float(df.DistrictHeatingFacility.values[i])
    for i in range (len(df.ElectricityFacility.values)):
        df.ElectricityFacility.values[i]=float(df.ElectricityFacility.values[i])
    
    df['TotalPower1']=(df.DistrictCoolingFacility)/3.6e6 #TotalPower1 is for cooling. Summer regression
    df['TotalPower2']=(df.DistrictHeatingFacility)/3.6e6 #TotalPower2 is for heating. Winter regression


    for i in range (len(df.DrybulbTemperatureHourly.values)):
        df.DrybulbTemperatureHourly.values[i]=float(df.DrybulbTemperatureHourly.values[i])

    df['T_ext']=df.DrybulbTemperatureHourly #External Temperature
    
    #change the type of the values in float
    for i in range (len(df.ZONE1OperativeTemperature.values)):
        df.ZONE1OperativeTemperature.values[i]=float(df.ZONE1OperativeTemperature.values[i])
        
    for i in range (len(df.ZONE2OperativeTemperature.values)):
        df.ZONE2OperativeTemperature.values[i]=float(df.ZONE2OperativeTemperature.values[i])
        
    for i in range (len(df.ZONE3OperativeTemperature.values)):
        df.ZONE3OperativeTemperature.values[i]=float(df.ZONE3OperativeTemperature.values[i])
        
    for i in range (len(df.ZONE4OperativeTemperature.values)):
        df.ZONE4OperativeTemperature.values[i]=float(df.ZONE4OperativeTemperature.values[i])
        
    for i in range (len(df.ZONE5OperativeTemperature.values)):
        df.ZONE5OperativeTemperature.values[i]=float(df.ZONE5OperativeTemperature.values[i])
        
    for i in range (len(df.ZONE6OperativeTemperature.values)):
        df.ZONE6OperativeTemperature.values[i]=float(df.ZONE6OperativeTemperature.values[i])
        
    for i in range (len(df.ZONE7OperativeTemperature.values)):
        df.ZONE7OperativeTemperature.values[i]=float(df.ZONE7OperativeTemperature.values[i])
        
    for i in range (len(df.ZONE8OperativeTemperature.values)):
        df.ZONE8OperativeTemperature.values[i]=float(df.ZONE8OperativeTemperature.values[i])
        
    for i in range (len(df.ZONE10OperativeTemperature.values)):
        df.ZONE10OperativeTemperature.values[i]=float(df.ZONE10OperativeTemperature.values[i])
    
    #Calculate the internal temperature through an average
    df['T_in']=(df.ZONE1OperativeTemperature+df.ZONE2OperativeTemperature+df.ZONE3OperativeTemperature+df.ZONE4OperativeTemperature+df.ZONE5OperativeTemperature+df.ZONE6OperativeTemperature+df.ZONE7OperativeTemperature+df.ZONE8OperativeTemperature+df.ZONE10OperativeTemperature)/9

    df['delta_T']=df.T_in-df.T_ext #Delta T= Internal temperature - External Temperature

    df= df.dropna() #Eliminate all the "nan" values


    df.delta_T = np.array(df.delta_T, dtype=float)

    df.TotalPower1 = np.array(df.TotalPower1, dtype=float)
    df.TotalPower2 = np.array(df.TotalPower2, dtype=float)
    
    #Create the Daily columns 
    for i in range(730):
        if i!=0:
            if i==729:
                df['delta_T_D'][i*24:]=df.delta_T[i*24:].mean() 
                df['TotalPower1_D'][i*24:]=df.TotalPower1[i*24:].mean()
                df['TotalPower2_D'][i*24:]=df.TotalPower2[i*24:].mean()
            else:
                df['delta_T_D'][i*24:(i+1)*24]=df.delta_T[i*24:(i+1)*24].mean() #calculate the average value of 24 samples (1 day)
                df['TotalPower1_D'][i*24:(i+1)*24]=df.TotalPower1[i*24:(i+1)*24].mean()
                df['TotalPower2_D'][i*24:(i+1)*24]=df.TotalPower2[i*24:(i+1)*24].mean()
        else:
            df['delta_T_D']=df.delta_T[:24].mean()
            df['TotalPower1_D']=df.TotalPower1[:24].mean()
            df['TotalPower2_D']=df.TotalPower2[:24].mean()
            
    #Create the Weekly columns    
    for i in range(104):
        if i!=0:
            if i==103:
                df['delta_T_W'][i*168:]=df.delta_T[i*168:].mean()
                df['TotalPower1_W'][i*168:]=df.TotalPower1[i*168:].mean()
                df['TotalPower2_W'][i*168:]=df.TotalPower2[i*168:].mean()
            else:
                df['delta_T_W'][i*168:(i+1)*168]=df.delta_T[i*168:(i+1)*168].mean() #calculate the average value of 168 samples (1 week)
                df['TotalPower1_W'][i*168:(i+1)*168]=df.TotalPower1[i*168:(i+1)*168].mean()
                df['TotalPower2_W'][i*168:(i+1)*168]=df.TotalPower2[i*168:(i+1)*168].mean()
        else:
            df['delta_T_W']=df.delta_T[:168].mean()
            df['TotalPower1_W']=df.TotalPower1[:168].mean()
            df['TotalPower2_W']=df.TotalPower2[:168].mean()
           
    #Create the Monthly columns 
    for i in range(24):
        if i!=0:
            if i==23:
                df['delta_T_M'][i*730:]=df.delta_T[i*730:].mean()
                df['TotalPower1_M'][i*730:]=df.TotalPower1[i*730:].mean()
                df['TotalPower2_M'][i*730:]=df.TotalPower2[i*730:].mean()
            else:
                df['delta_T_M'][i*730:(i+1)*730]=df.delta_T[i*730:(i+1)*730].mean() #calculate the average value of 730 samples (1 month)
                df['TotalPower1_M'][i*730:(i+1)*730]=df.TotalPower1[i*730:(i+1)*730].mean()
                df['TotalPower2_M'][i*730:(i+1)*730]=df.TotalPower2[i*730:(i+1)*730].mean()
        else:
            df['delta_T_M']=df.delta_T[:730].mean()
            df['TotalPower1_M']=df.TotalPower1[:730].mean()
            df['TotalPower2_M']=df.TotalPower2[:730].mean()
    
    return df

def season(df,s=""):
    #Create winter and summer dataframe
    df_wint=pd.concat([df[:1776+744],df[7296:10533+744],df[16053:]]) #concatenate data from 01/01/19 to 15/04/19 with data from 01/11/19 to 15/04/20 and with data from 01/11/20 to 31/12/20
    df_summ=pd.concat([df[1776+744:7295],df[10533+744:16053]]) #concatenate data from 15/04/19 to 31/10/19 with data from 15/04/20 to 31/10/20
    #Filters the zero values
    df_wint= df_wint[df_wint['TotalPower2'+s] > 0.1]
    df_summ= df_summ[df_summ['TotalPower1'+s] > 0.1]
    
    return df_wint, df_summ

def plot(df_win,df_sum,s=""):
    
    #Plot the results
    
    plt.figure()
    plt.grid()
    plt.title('Energy Signature')
    plt.xlabel(" ΔT [°C]")
    plt.ylabel("Energy Consumption [kWh]")
    
    ## Winter 
    model = sm.OLS(df_win['TotalPower2'+s],sm.add_constant(df_win['delta_T'+s]))
    results=model.fit()
    print("### RESULTS WINTER REGRESSION SAMPLE"+s)
    print(results.summary())

    plt.plot(df_win['delta_T'+s],results.predict(),'r', label="Winter Regression Line")
    plt.scatter(df_win['delta_T'+s],df_win['TotalPower2'+s], s=5, label="Winter Samples")
    
    ## Summer 
    model = sm.OLS(df_sum['TotalPower1'+s],sm.add_constant(df_sum['delta_T'+s]))
    results=model.fit()
    print("### RESULTS SUMMER REGRESSION SAMPLE"+s)
    print(results.summary())


    plt.plot(df_sum['delta_T'+s],results.predict(),'g', label="Summer Regression Line")
    plt.scatter(df_sum['delta_T'+s],df_sum['TotalPower1'+s], s=5, label="Summer Samples")
    plt.legend(loc=9, prop={'size': 8})
    plt.ylim(bottom=0.0)
    
    
    
    return
    
filename_1='influx_eplusout2019.csv'
filename_2='influx_eplusout2020.csv'

df=dataframe_preparation(filename_1,filename_2)

df_win_H, df_sum_H=season(df)


plt.Figure()
plot(df_win_H,df_sum_H)


df_win_D, df_sum_D=season(df,"_D")
plot(df_win_D,df_sum_D,"_D")

df_win_W, df_sum_W=season(df,"_W")
plot(df_win_W,df_sum_W,"_W")

df_win_M, df_sum_M=season(df,"_M")
plot(df_win_M,df_sum_M,"_M")

