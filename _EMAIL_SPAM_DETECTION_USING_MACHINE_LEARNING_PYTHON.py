#!/usr/bin/env python
# coding: utf-8

# # EMAIL SPAM DETECTION WITH MACHINE LEARNING

# In[1]:


#...............................................IMPORTING THE REQUIRED  LIBRARIES...............................................


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
 


# In[3]:


import nltk
nltk.download('stopwords')
  


# In[4]:


stop_words=stopwords.words('english')
print(stop_words)


# # LOADING DATASET FROM CSV TO PANDAS DATAFRAME 

# In[5]:


raw_data=pd.read_csv('mail_data.csv')


# In[6]:


stop_words=stopwords.words('english')


# In[7]:


raw_data


# In[ ]:





# # REPLACE THE NULL VALUES WITH NULL STRING 

# In[8]:


data=raw_data.where((pd.notnull(raw_data)),'')


# In[9]:


data


# In[10]:


data.head()       #DISPLAYS FIRST FIVE ROWS


# In[11]:


data.tail()   #DISPLAYS LAST FIVE ROWS


# In[12]:


data.describe()    #DISPLAYS THE STATISTICS OF THE DATASET


# In[13]:


data.info()   #DISPLAYS INFORMATION ABOUT THE DATASET


# In[14]:


data.shape    #DISPALYS THE NUMBER OF ROWS AND NUMBER OF COLUMNS


# In[15]:


#.................................................LABEL ENCODING...............................................................


# In[16]:


#LET'S LABEL THE SPAM MAILS AS '0' AND HAM MAILS AS '1'
data.loc[data['Category']=='spam','Category',]=0
data.loc[data['Category']=='ham','Category',]=1 


# In[17]:


print(data)


# # SEPARATING THE DATA AS TEXT(INPUT) AND LABELS(OUTPUT) 

# In[18]:


x=data['Message']
y=data['Category']


# In[19]:


#...........................................SPLITTING DATA INTO TRAIN AND TEST..................................................


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[21]:


print(x_train)


# In[22]:


print(y_train)


# In[23]:


print(x_test)


# In[24]:


print(y_test)


# In[25]:


print(x_train.shape)


# In[26]:


print(y_train.shape)


# In[27]:


print(x_test.shape)


# In[28]:


print(y_test.shape)


# # FEATURE EXTRACTION 

# In[29]:


#TRANSFROM THE TEXT DATA TO FEATURED VECTORS SO THAT IT ACN BE USED AS AN INPUT FOR THE LOGISTIC REGRESSION


# In[30]:


feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)


# In[31]:


x_train_features=feature_extraction.fit_transform(x_train)


# In[32]:


print(x_train_features)


# In[33]:


x_test_features=feature_extraction.transform(x_test)


# In[34]:


print(x_test_features)


# # CONVERTING y_train, y_test VALUES AS INTEGERS 

# In[35]:


y_train_features=y_train.astype('int')


# In[36]:


y_test_features=y_test.astype('int')


# In[37]:


y_train_features


# In[38]:


y_test_features


# #  TRAINING THE MODEL USING LOGISTIC REGRESSION 

# In[39]:


model=LogisticRegression()
model.fit(x_train_features,y_train_features)


# In[40]:


pred1=model.predict(x_train_features)
print(accuracy_score(y_train_features,pred1)*100)


# In[41]:


pred2=model.predict(x_test_features)
print(accuracy_score(y_test_features,pred2)*100)


# In[42]:


input1=["Hurry Up! Only few left grab your choice"]


# In[43]:


input_features=feature_extraction.transform(input1)
prediction=model.predict(input_features)


# In[44]:


print(prediction)


# In[45]:


if(prediction==1):
    print("HAM MAIL")
else:
    print("SPAM MAIL")


# #  THANK YOU 
