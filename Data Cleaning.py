#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import random
plt.style.use("ggplot")


# In[2]:


data1 =pd.read_csv("processed_individual_cases_Sep20th2020.csv")


# In[3]:


data2 =pd.read_csv("processed_location_Sep20th2020.csv")


# In[4]:


#Plotting number of confirmed cases by country
groupByCountry= data2.groupby(by=["Country_Region"]).sum()
plt.figure(figsize=(10, 30))
plt.xlabel('Number of Cases')
plt.ylabel('Countries')
plt.plot(groupByCountry['Confirmed'],groupByCountry.index,'.')
plt.savefig('confirmedCases_byCountry.jpg')


# # AGE FILTERING

# In[5]:


#Age is defined in 362 different patterns
len(np.unique(data1['age'].astype(str)))
np.unique(data1['age'].astype(str))

#First we will try to solve the pattern which contains age in shape 'X-Y'
agefiltered= data1['age'].astype(str)

def getFilter1(df):
    return re.findall('\d+',df)

arrayData=  agefiltered.apply(getFilter1)

#Now we randomly assign the age for pattern X-Y between X and Y
for i in range(len(arrayData)):
    if(len(arrayData[i])>1):
        value1= int(arrayData[i][0])
        value2= int(arrayData[i][1])
        if(value2!=0 and value2>=value1):
            randomAge= random.randint(value1, value2)
            data1['age'][i]=str(randomAge)
            


# In[6]:


#Now we are still left with different pattern. We we do it manually.
for i in range(len(data1)):
    if (data1['age'][i]=='18-') :
        data1['age'][i]=str(random.randint(1, 18))
    elif (data1['age'][i]=='21-') :
        data1['age'][i]=str(random.randint(1, 21))
    elif (data1['age'][i]=='55-') :
        data1['age'][i]=str(random.randint(1, 55))
    elif (data1['age'][i]=='65-') :
        data1['age'][i]=str(random.randint(1, 65))
    elif (data1['age'][i]=='80+'):
        data1['age'][i]=str(random.randint(80, 100))
    elif (data1['age'][i]=='80-'):
        data1['age'][i]=str(random.randint(60, 80))
    elif (data1['age'][i]=='85+'):
        data1['age'][i]=str(random.randint(85, 100))
    elif(data1['age'][i]=='60-') :
        data1['age'][i]=str(random.randint(40, 60))
    elif (data1['age'][i]=='90+'):
        data1['age'][i]=str(random.randint(90, 110))   


# In[7]:


#Now we will do our last filter which contains 'months'
for i in range(len(data1)):
    if (data1['age'][i]=='11 month') :
        data1['age'][i]=str(11/12)
    elif (data1['age'][i]=='18 month') :
        data1['age'][i]=str(18/12)
    elif (data1['age'][i]=='4 months') :
        data1['age'][i]=str(4/12)
    elif (data1['age'][i]=='5 month') :
        data1['age'][i]=str(5/12)
    elif (data1['age'][i]=='6 months'):
        data1['age'][i]=str(6/12)
    elif (data1['age'][i]=='8 month'):
        data1['age'][i]=str(8/12)


# In[143]:


#data1['age']=data1['age'].astype(str).astype(float)


# # BELOW CODE WILL HELP DURING MERGE

# In[9]:


#For province value which are left 'nan' or 'State Unassigned' will be assigned the value of their country.
for i in range(len(data1)):
    if(str(data1['province'][i])=='nan' or  data1['province'][i]=='State Unassigned'):
        data1['province'][i]=data1['country'][i]
#Same approch for data2
for i in range(len(data2)):
    if(str(data2['Country_Region'][i])=='nan'):
        data2['Country_Region'][i]=data2['Province_State'][i]


# In[10]:


#Remove Two rows whose all data is NAN
data1=data1[data1['province'].notnull()]
data1=data1.reset_index(drop=True)


# In[11]:


data1['age']=data1['age'].astype(str).astype(np.float)
removedNanData = data1[data1['age'].notnull()].reset_index(drop=True)
groupByProvince= removedNanData.groupby(by=["province"]).mean()


# In[12]:


#Setting the age value by mean of that province which it belongs to.
for i in range(len(data1)):
    if str(data1['age'][i])== 'nan':
        for j in range(len(groupByProvince)):
            if(data1['province'][i]==groupByProvince.index[j]):
                data1['age'][i]=groupByProvince['age'][j]
                break;


# In[13]:


#There is still age left which who does not have mean of province then thet will be assigned mean value of their country.
data1[data1['age'].isnull()].reset_index(drop=True)
removedNanData = data1[data1['age'].notnull()].reset_index(drop=True)
groupByCountry= removedNanData.groupby(by=["country"]).mean()


# In[14]:


#Left over age 'nan' values will be assigned by mean of country.
for i in range(len(data1)):
    if(str(data1['age'][i])=='nan'):
        for j in range(len(groupByCountry)):
            if (data1['country'][i]==groupByCountry.index[j]):
                data1['age'][i]=groupByCountry['age'][j]    


# In[15]:


#Last filter for outliers where ages are greater than 150 which is not possible. 
for i in range(len(data1)):
    if(str(data1['age'][i])=='nan' or data1['age'][i]>150):
        data1['age'][i]=random.randint(0, 95)


# In[16]:


#Age histogram plot
plt.figure(figsize=(10, 10))
plt.hist(data1['age'], bins = 50,rwidth=0.9,alpha=0.9, color='#FF5733')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y',alpha=0.5)
plt.title('Age group Frequency')
plt.savefig('AgeFrequency.jpg')


# # FILTERING SEX 

# In[17]:


removedNanData = data1[data1['sex'].notnull()].reset_index(drop=True)
groupByProvince= removedNanData.groupby(by=["province"]).mean()


# In[31]:


x=data1.groupby(['province','sex']).count()
data1['sex']=data1['sex'].astype(str)


# In[32]:


#Sex value is not defined then the ratio is used for that province to set 'nan' value to male or female
for i in range(len(data1)):
    if(str(data1['sex'][i])=='nan'):
        for j in range(len(x)):
            if(str(data1['province'])==x.index[j][0]):
                if(x[i]>0 and x[i+1]>0):
                    maleRatio= (x[i]/(x[i]+x[i+1])) *100
                    randomValue= random.randint(1, 100)
                    if(randomValue<maleRatio):
                        data1['sex'][i]='male'
                    else:
                        data1['sex'][i]='female'


# # FILTERING FOR INCIDENCE RATE

# In[4]:


#Filtering of insidence_Rate
#removedNanData = data1[data1['sex'].notnull()].reset_index(drop=True)
groupByProvince= data2.groupby(by=["Province_State"]).mean()
data2[data2['Incidence_Rate'].isnull()]
for i in range(len(data2)):
    if(str(data2['Incidence_Rate'][i])=='nan' ):
        for j in range(len(groupByProvince)) :
            if data2['Province_State'][i]==groupByProvince.index[j]:
                data2['Incidence_Rate'][i]=groupByProvince['Incidence_Rate'][j]
     


# In[5]:


#Now remaining values are filtered by taking country in account
groupByCountry= data2.groupby(by=["Country_Region"]).mean()
data2[data2['Incidence_Rate'].isnull()]
for i in range(len(data2)):
    if(str(data2['Incidence_Rate'][i])=='nan' ):
        for j in range(len(groupByCountry)) :
            if data2['Country_Region'][i]==groupByCountry.index[j]:
                data2['Incidence_Rate'][i]=groupByCountry['Incidence_Rate'][j]


# In[6]:


#Filtering of Case-Fatality_Ratio
for i in range(len(data2)):
    if(str(data2['Case-Fatality_Ratio'][i])=='nan' ):
        for j in range(len(groupByProvince)) :
            if data2['Province_State'][i]==groupByProvince.index[j]:
                data2['Case-Fatality_Ratio'][i]=groupByProvince['Case-Fatality_Ratio'][j]


# In[7]:


#Manually filtering remaining data
str(data2['Incidence_Rate'][0])=='nan'
data2[data2['Case-Fatality_Ratio'].isnull()]
data2= data2[data2.index!=3071].reset_index(drop=True)
data2= data2[data2.index!=251].reset_index(drop=True)
data2= data2[data2.index!=3941].reset_index(drop=True)
data2= data2[data2.index!=594].reset_index(drop=True)
data2= data2[data2.index!=265].reset_index(drop=True)
data2= data2[data2.index!=67].reset_index(drop=True)
data2.loc[data2.index==175,'Incidence_Rate']= float('nan')
data2.loc[data2.index==359,'Incidence_Rate']= float('nan')


# In[8]:


#Giving latitude and longitude based on province
for i in range(len(data2)):
    if(str(data2['Lat'][i])=='nan' or str(data2['Long_'][i])=='nan' ):
        for j in range(len(groupByProvince)) :
            if data2['Province_State'][i]==groupByProvince.index[j]:
                data2['Lat'][i]=groupByProvince['Lat'][j]
                data2['Long_'][i]=groupByProvince['Long_'][j]


# In[9]:


#Latitude and Longitude based on country if Province is not there, we need to be near our main region
for i in range(len(data2)):
    if(str(data2['Lat'][i])=='nan' or str(data2['Long_'][i])=='nan' ):
        for j in range(len(groupByCountry)) :
            if data2['Country_Region'][i]==groupByCountry.index[j]:
                data2['Lat'][i]=groupByCountry['Lat'][j]
                data2['Long_'][i]=groupByCountry['Long_'][j]
                


# In[10]:


#We have no idea baout the rest 4, so we used unknown here'
data2[data2['Lat'].isnull()]
data2.loc[data2.index==175,'Lat']= float('nan')
data2.loc[data2.index==175,'Long_']= float('nan')
data2.loc[data2.index==359,'Lat']= float('nan')
data2.loc[data2.index==359,'Long_']=float('nan')


# In[11]:


#We calculated that only didn't have anything on active cases, and all recovered. Safe to say that 0 are active.
data2[data2['Active'].isnull()]
data2.loc[data2.index==95,'Active']= np.float(0)


# In[12]:


for i in range(len(data2)):
    if(str(data2['Province_State'][i])=='nan' or  data2['Province_State'][i]=='State Unassigned'):
        data2['Province_State'][i]=data2['Country_Region'][i]


# In[13]:


for i in range(len(data1)):
    if(str(data1['date_confirmation'][i])=='nan'):
        data1['date_confirmation'][i] = 'Unknown'

for i in range(len(data1)):
    if(str(data1['additional_information'][i])=='nan'):
        data1['additional_information'][i] = 'Unknown'

for i in range(len(data1)):
    if(str(data1['source'][i])=='nan'):
        data1['source'][i] = 'Unknown'


# # FINDING OUTLIERS

# In[ ]:


for i in range(len(data1)):
    if str(data1['age'][i]) != 'Unknown' :
        if float(data1['age'][i])  > 150 :
            print(i,"  Value hai  ",data1['age'][i])
#plt.figure(figsize=(20,20))            
#plt.plot(data2[''])  
#data2[data2.index==736]
#print(data2['Incidence_Rate'][987])
#data2['Incidence_Rate'][1048]
#data2[3272]


# # ISOLATION TREE FOR OUTLIERS

# In[ ]:


#model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
#model.fit(data2[['Confirmed']])
#data2[data2['Incidence_Rate'].isnull()]


# In[ ]:


#data2['scores']=model.decision_function(data2[['Confirmed']])
#data2['anomaly']=model.predict(data2[['Confirmed']])


# In[ ]:


#anomaly=data2.loc[data2['anomaly']==-1]
#anomaly_index=list(anomaly.index)
#print(anomaly)


# In[ ]:


#print("Accuracy percentage:", 100*list(data2['anomaly']).count(-1)/(outliers_counter))


# # Deleting latitude who are not between (-90,90) and longitude between (-180,180)

# In[44]:


for i in range(len(data2)):
    if str(data2['Lat'][i]) != 'Unknown' :
        if float(data2['Lat'][i])  > 90 :
            print(i,"  Value hai  ",data2['Lat'][i])

data2['L'].max()


# # Checking for outliers in case-Fatality_Rates

# In[45]:


for i in range(len(data2)):
    if str(data2['Case-Fatality'][i]) != 'Unknown' :
        if float(data2['Case-Fatality_Rate'][i])  > 70 :
            print(i,"  Value hai  ",data2['Case_Fatality_Rate'][i])


# In[15]:


data2= data2[data2.index!=3059].reset_index(drop=True)


# # Finding outliers for active cases as it cannot be negative

# In[14]:


for i in range(len(data2)):
    if str(data2['Active'][i]) != 'Unknown' :
        if float(data2['Active'][i])  < 0 :
            print(i,"  Value hai  ",data2['Active'][i])

data2= data2[data2.index!=2551].reset_index(drop=True)
data2= data2[data2.index!=1866].reset_index(drop=True)
data2= data2[data2.index!=1421].reset_index(drop=True)
data2= data2[data2.index!=467].reset_index(drop=True)


# # Doing 1.4 to Find data for USA

# In[16]:


dataUS=data2[data2['Country_Region']=='US']
ProvinceSum = dataUS.groupby(by=['Province_State']).sum()
ProvinceMean = dataUS.groupby(by=['Province_State']).mean()


# In[17]:


for i in range(len(ProvinceSum)):
    ProvinceSum['Case-Fatality_Ratio'][i]= (ProvinceSum['Deaths'][i] / ProvinceSum['Confirmed'][i])*100
              


# In[18]:


ProvinceSum['Incidence_Rate'] = ProvinceMean['Incidence_Rate']
ProvinceSum = ProvinceSum.drop(['Lat', 'Long_'],axis=1)
ProvinceSum


# In[19]:


individualUS = data1[data1['country']=='United States']


# In[20]:


individualUS = individualUS.merge(ProvinceSum,how='left',left_on=['province'], right_on=['Province_State'])


# In[21]:


data1[data1['country']=='United States']


# # Checking for countries that are in ocations but not in cases

# In[22]:


check1 = data1.groupby(by =['country']).mean()
check2 = data2.groupby(by=['Country_Region']).mean()


# In[23]:


count = 0

for i in range(len(check1)):
    toy="nan"
    for j in range(len(check2)):
        if check1.index[i]==check2.index[j] :
            count=count +1
            toy=check1.index[i]
    if(toy=='nan'):
        print(check1.index[i])
    


# # 8 found with distinct names in the dataset

# In[ ]:


#Czech Republic = Czechia
#Democratic republic of Congo = congo kinhasa 
#republic of COngo is congo brazzville
#Puerto rico put in province and set country US FOR CASES
#set reuinion to province and country to france in cases
#change korea,south to South Korea
#Taiwan is same as country and province with no data
#change united states to US


# In[ ]:


Checking for province


# In[24]:


check1 = data1.groupby(by =['province']).mean()
check2 = data2.groupby(by=['Province_State']).mean()


# In[25]:


count = 0

for i in range(len(check1)):
    toy="nan"
    for j in range(len(check2)):
        if str(check1.index[i]).lower()==str(check2.index[j]).lower() :
            count=count +1
            toy=check1.index[i]
    if(toy=='nan'):
        print(check1.index[i])
    


# # A Province is a country with its cities, coordinates are similar so we shifted it to province

# In[26]:


data1.loc[data1.index==98036,'province']='Puerto Rico'
data1.loc[data1.index==59920,'province']='Puerto Rico'
data1.loc[data1.index==117041,'province']='Puerto Rico'
data1.loc[data1.index==258482,'province']='Puerto Rico'
data1.loc[data1.index==292150,'province']='Puerto Rico'


# # Replaced rest of the countries

# In[27]:


data1['country']=data1['country'].replace(['United States'],['US']) 
data1['country']=data1['country'].replace(['Democratic Republic of the Congo'],['Congo (Kinshasa)']) 
data1['country']=data1['country'].replace(['Republic of Congo'],['Congo (Brazzaville)']) 
data1['country']=data1['country'].replace(['South Korea'],['Korea, South']) 
data1['country']=data1['country'].replace(['Reunion'],['France'])
data1['country']=data1['country'].replace('Puerto Rico','US')
data1['country']=data1['country'].replace('Czech Republic','Czechia')


# # making sure all strings are compared currectly

# In[28]:


def toUpper(data):
    return data.upper()

data2['Province_State']=data2['Province_State'].astype(str)
data1['province']=data1['province'].astype(str)

data1['province']=data1['province'].apply(toUpper)
data2['Province_State']=data2['Province_State'].apply(toUpper)

data2['Country_Region']=data2['Country_Region'].astype(str)
data1['country']=data1['country'].astype(str)

data1['country']=data1['country'].apply(toUpper)
data2['Country_Region']=data2['Country_Region'].apply(toUpper)


# # With our data ready we now Join it

# In[29]:


CountrySum = data2.groupby(by=['Province_State','Country_Region']).sum()
CountryMean = data2.groupby(by=['Province_State','Country_Region']).mean()


# In[30]:


for i in range(len(CountrySum)):
    CountrySum['Case-Fatality_Ratio'][i]= (CountrySum['Deaths'][i] / CountrySum['Confirmed'][i])*100


# In[31]:


CountrySum['Incidence_Rate'] = CountryMean['Incidence_Rate']
CountrySum = CountrySum.drop(['Lat', 'Long_'],axis=1)


# In[50]:


worldMerge = data1.merge(CountrySum,how='left',left_on=['province', 'country'], right_on=['Province_State','Country_Region'])


# In[71]:


for i in range(len(worldMerge)):
    if(str(worldMerge['sex'][i])=='nan' and i%2==0):
        worldMerge['sex'][i]='male'
    elif(str(worldMerge['sex'][i])=='nan' and i%2==1):
        worldMerge['sex'][i]='female'


# In[73]:


worldMerge.to_csv("mergedData.csv")


# In[77]:





# In[78]:





# In[79]:


worldMerge1 = worldMerge.head(int(len(worldMerge)/2))
worldMerge2 = worldMerge[int(len(worldMerge)/2):].reset_index()
worldMerge1.to_csv("finalDataPart1.csv", index = False)
worldMerge2.to_csv("finalDataPart2.csv", index = False)


# In[53]:


data2=data2.dropna().reset_index(drop=True)


# In[54]:


data2.isnull().sum()


# In[55]:


mergedData= pd.read_csv("mergedData.csv")


# In[56]:


mergedData.isnull().sum()


# In[57]:


mergedDataRemovedNAN= mergedData.dropna().reset_index(drop=True)


# In[58]:


is_NaN = mergedData.isnull()
row_has_NaN = is_NaN. any(axis=1)
rows_with_NaN = mergedData[row_has_NaN].reset_index(drop=True)
rows_with_NaN=rows_with_NaN.drop(columns=['Confirmed','Deaths','Recovered','Active','Incidence_Rate','Case-Fatality_Ratio'])


# In[43]:


rows_with_NaN


# In[48]:





# In[59]:


CountrySum = data2.groupby(by=['Country_Region']).sum()
CountryMean = data2.groupby(by=['Country_Region']).mean()
for i in range(len(CountrySum)):
    CountrySum['Case-Fatality_Ratio'][i]= (CountrySum['Deaths'][i] / CountrySum['Confirmed'][i])*100


# In[60]:


CountrySum['Incidence_Rate'] = CountryMean['Incidence_Rate']
CountrySum = CountrySum.drop(['Lat', 'Long_'],axis=1)


# In[36]:





# In[42]:


data1


# In[45]:


temp


# In[52]:


data1


# In[53]:





# In[55]:


rows_with_NaN=rows_with_NaN.drop(columns=['Unnamed: 0'])


# In[56]:





# In[ ]:





# In[61]:


temp=rows_with_NaN.merge(CountrySum,how='left',left_on=['country'], right_on=['Country_Region'])


# In[62]:


temp=temp.drop(columns=['Unnamed: 0'])


# In[63]:


finaldata= mergedDataRemovedNAN.append(temp).drop(columns=['Unnamed: 0']).reset_index(drop=True)


# In[69]:


worldMerge1 = finaldata.head(int(len(finaldata)/2))
worldMerge2 = finaldata[int(len(finaldata)/2):].reset_index()
worldMerge1.to_csv("finalDataPart1.csv", index = False)
worldMerge2.to_csv("finalDataPart2.csv", index = False)


# In[67]:


finaldata=finaldata.dropna().reset_index(drop=True)


# In[66]:


len(finaldata)


# In[68]:


len(finaldata)


# In[8]:





# In[11]:


data1['date_confirmation'][0].timestamp()


# In[76]:





# In[94]:


temp1=pd.read_csv("finalDataPart1.csv")
temp2=pd.read_csv("finalDataPart2.csv")

for i in range(len(temp1)):
    if(temp1['date_confirmation'][i]=='Unknown'):
        temp1['date_confirmation'][i]='15.06.2020'
        
for i in range(len(temp2)):
    if(temp2['date_confirmation'][i]=='Unknown'):
        temp2['date_confirmation'][i]='15.06.2020'


# In[95]:


import re

newtemp1=temp1['date_confirmation'].astype(str)
def getCorrectDateValue(data):
    return re.findall('\d+\.\d+\.\d+',data)

tempFirst= newtemp1.apply(getCorrectDateValue)


newtemp2=temp2['date_confirmation'].astype(str)
def getCorrectDateValue(data):
    return re.findall('\d+\.\d+\.\d+',data)

tempSecond= newtemp2.apply(getCorrectDateValue)


# In[92]:


import datetime
import re

def makeDatetime(data):
    return datetime.datetime.strptime(data[0], '%d.%m.%Y')

temp1['date_confirmation'] = tempFirst.apply(makeDatetime)
temp2['date_confirmation'] = tempSecond.apply(makeDatetime)


# In[93]:


datetime.datetime.strptime(temp1[0], '%d.%m.%Y')


# In[83]:


temp2.isnull().sum()


# In[84]:


temp1.to_csv("finalDataPart1.csv", index = False)
temp2.to_csv("finalDataPart2.csv", index = False)


# In[85]:


temp1


# In[89]:


def to_timestamp(d):
    return d.timestamp()
temp1['date_confirmation'] = temp1['date_confirmation'].apply(to_timestamp)


# In[88]:


temp1['date_confirmation'][0].timestamp()


# In[96]:


temp1


# In[97]:


tempFirst


# In[ ]:




