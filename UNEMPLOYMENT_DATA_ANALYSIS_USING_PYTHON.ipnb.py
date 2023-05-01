#!/usr/bin/env python
# coding: utf-8

# In[1]:


#**************************************UNEMPLOYMENT DATA ANALYSIS USING PYTHON**************************************************


# In[2]:


#.........................................IMPORTING REQUIRED LIBRARIES..........................................................


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[4]:


#...............................................READ THE DATASET................................................................


# In[5]:


df=pd.read_c#**************************************UNEMPLOYMENT DATA ANALYSIS USING PYTHON**************************************************sv('Unemployment in India.csv')


# In[6]:


df=pd.read_csv('Unemployment_Rate_upto_11_2020.csv')


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.describe()


# In[10]:


df.info()


# In[11]:


#.................................CHECKING WHETHER THE DATASET CONTAINS MISSING VALUES OR NOT.................................


# In[12]:


df.isnull().sum()


# In[13]:


#........................................RENAMING THE COLUMNS...................................................................


# In[14]:


df.columns=["States","Date","Frequency","Estimated Unemployment Rate","Estimated Employed","Estimated Labour Participation Rate","Region","Longitute","Latitude"]


# In[15]:


print(df)


# In[16]:


#................................FINDING CORRELATION BETWEEN THE FEATURES OF THE DATASET........................................


# In[17]:


plt.style.use("seaborn-whitegrid")


# In[18]:


plt.figure(figsize=(10,9))


# In[19]:


sns.heatmap(df.corr())


# In[20]:


#.............................ESTIMATED NUMBER OF EMPLOYEES IN DIFFERENT REGIONS OF INDIA.......................................


# In[21]:


plt.title("Indian Employees")
sns.histplot(x="Estimated Employed",hue="Region",data=df)


# In[22]:


#................................ESTIMATED NUMBER OF UNEMPLOYEES IN DIFFERENT REGIONS OF INDIA..................................


# In[23]:


plt.figure(figsize=(10,9))
plt.title("Indian Unemployees")
sns.histplot(x="Estimated Unemployment Rate",hue="Region",data=df)


# 
# 

# In[24]:


#..................CREATE A DASHBOARD TO ANALYSE THE UNEMPLOYMENT RATE OF EACH STATE IN INDIA...................................


# In[25]:


Unemployment=df[["States","Region","Estimated Unemployment Rate"]]
figure=px.sunburst(Unemployment,path=["Region","States"],values="Estimated Unemployment Rate",width=700,height=700,color_continuous_scale="RdY1Gn",title="Unemployment Rate in India")
figure.show()


# In[26]:


#******************************************************THANK YOU******************************************************#

