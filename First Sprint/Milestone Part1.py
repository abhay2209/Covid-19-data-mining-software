#!/usr/bin/env python
# coding: utf-8

# # Importing Headers

# In[23]:





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import random
import calendar
plt.style.use("ggplot")

## Importing Datasets


# In[2]:


data1 =pd.read_csv("processed_individual_cases_Sep20th2020.csv")
data2 =pd.read_csv("processed_location_Sep20th2020.csv")


# In[3]:


len(data1)


# ## Exploratory Data `Analing`

# In[59]:


describe= data2.describe()
describe.drop(['Lat','Long_'],axis=1)

describe= data2.describe()
#describe.drop(['Lat','Long_'],axis=1)
describe =describe.drop(['Incidence_Rate','Case-Fatality_Ratio'],axis=1)

describe


# In[24]:



x= 'NAN'
temp= data1['sex'].replace(np.nan, x)
plt.figure(figsize=(11, 7))
matplotlib.rcParams.update({'font.size': 22})
temp.value_counts().plot(kind='bar')


# In[47]:


newtemp=temp.value_counts()


# In[49]:


newtemp.values


# In[45]:


x= 'NAN'
temp= data1['province'].replace(np.nan, x)

plt.figure(figsize=(15, 12))
matplotlib.rcParams.update({'font.size': 5})
plt.xlabel("")
.plot(kind='bar')


# In[31]:


for i in range(len(temp)):
    print(temp[i])


# ## Data Cleaning

# ### How many values are `Null` ?

# In[70]:


data1.isnull().sum()


# In[71]:


data2.isnull().sum()


# `Age`

# In[72]:


dfAge = data1.dropna(axis=0,subset=['age']).reset_index(drop=True)


# ## PLOTS

# In[78]:


# outcomes plot
outcomeCount = data1['outcome'].value_counts()
plt.figure(figsize=(11, 7))
plt.plot(outcomeCount,"*")
plt.title("Patient Outcomes")
plt.ylabel('Number of People')
plt.savefig("outcomesCount.png")


# In[79]:


# Sex
sexCount = data1['sex'].value_counts()
plt.figure(figsize=(9, 7))
plt.plot(sexCount,"*")
plt.title("Distinct Sex Counts")
plt.ylabel("Number of Male and Females")
plt.savefig("sexCount.png")
print("Total cases by gender:")
print(sexCount)


# In[81]:


## Incidence_Rate
telly = data2['Incidence_Rate'].unique()
plt.figure(figsize=(13, 9))
plt.plot(telly)
plt.title("Incidence_Rate For All Over The World")
plt.xlabel("Countries")
plt.ylabel("Incidence Value")
plt.savefig("IncidenceRate.png")


# In[82]:


## Case-Fatality_Ratio
plt.figure(figsize=(11, 7))
plt.plot(data2['Case-Fatality_Ratio'])
plt.title("Case-Fatality_Ratio")
plt.xlabel('Countries')
plt.ylabel('Case Faculty Ratio Value')
plt.savefig("Case-Fatality_Ratio.png")

print("Case-Fatality_Ratio mean : ", np.mean(data2['Case-Fatality_Ratio']))
print("Case-Fatality_Ratio max : ", np.max(data2['Case-Fatality_Ratio']))


# In[83]:


## Deaths
plt.figure(figsize=(18, 7))
plt.subplot(1, 3, 1)
plt.plot(data2['Deaths'])
plt.title("Deaths")
plt.xlabel('Countries')
plt.ylabel('Count')
plt.subplot(1, 3, 2)
plt.plot(data2['Recovered'])
plt.xlabel('Countries')
plt.ylabel('Count')
plt.title("Recovered")
plt.subplot(1, 3,3)
plt.plot(data2['Active'])
plt.title("Active")
plt.savefig("Deaths.png")

print("Mean Deaths : ", np.mean(data2['Deaths']))
print("Max Deaths : ", np.max(data2['Deaths']))


# In[84]:


temp = data2['Incidence_Rate'].values.tolist()
plt.figure(figsize=(15,10))
N=10
area = (30 * np.random.rand(N))**2
plt.scatter(x=data2['Long_'], y=data2['Lat'], s=area, c=temp, alpha = 0.3)
plt.title('Covid-19 cases by Location')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar()
plt.savefig('HeatMap_bylocation.jpg')


# In[85]:


temp = data1.index.values.tolist()
plt.figure(figsize=(15,10))
N=5
area = (15 * np.random.rand(N))**2
plt.scatter(x=data1['longitude'], y=data1['latitude'], s=area, c=temp, alpha = 0.15)
plt.title('Covid-19 cases by Invidual')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar()
plt.savefig('HeatMap_by_individual.jpg')


# # `Date Cleaning FOR DATE FEATURE`

# In[86]:


def arrange_dates(tempdata):
    
    if(tempdata=='06.03.2020 - 09.03.2020'):
        tempdata='06.03.2020'
    elif(tempdata=='06.03.2020-13.03.2020' or tempdata == '06.03.2020 - 08.03.2020'):
        tempdata='06.03.2020'
    elif (tempdata=='07.03.2020 - 13.03.2020'):
        tempdata='07.03.2020'
    elif (tempdata=='25.02.2020 - 03.03.2020' or tempdata == '25.02.2020 - 26.02.2020') :
        tempdata='25.02.2020'
    
    elif (tempdata=='12.03.2020 - 13.03.2020' or tempdata == '12.03.2020-14.03.2020'):
        tempdata='12.03.2020'
        
    elif (tempdata=='10.03.2020 - 11.03.2020' or tempdata == '10.03.2020-13.03.2020' or tempdata == '10.03.2020 - 12.03.2020'):
        tempdata='10.03.2020'
          
    elif(tempdata== '07.03.2020 - 09.03.2020' or tempdata == '07.03.2020-09.03.2020' or tempdata == '07.03.2020 - 10.03.2020'):
        tempdata= '07.03.2020'
            
    elif (tempdata=='05.03.2020-06.03.2020'):
        tempdata='05.03.2020'
    elif (tempdata=='18.03.2020-19.03.2020'):
        tempdata='18.03.2020'
      
    return tempdata
        
data1["newdate"] = data1['date_confirmation'].apply(arrange_dates)
data1["newdate"]  = pd.to_datetime(data1["newdate"])


# In[87]:


data1["month"] = pd.to_datetime(data1['newdate'], format='%m').dt.month_name().str.slice(stop=3)


# In[89]:


# Curve plot
caseByMonth = data1["month"].value_counts()
plt.figure(figsize=(11, 7))
plt.plot(caseByMonth, "--")
plt.title("Cases By Month")
plt.xlabel('Months')
plt.ylabel('Number of Cases')
plt.savefig("CasesByMonth.png")

print("Total cases by gender:")
print(caseByMonth)


# In[ ]:




