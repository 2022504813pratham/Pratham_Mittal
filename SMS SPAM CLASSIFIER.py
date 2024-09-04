#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


data = pd.read_csv('spam.csv',encoding='Latin1')
data


# In[8]:


data.shape


# In[24]:


data.isnull().sum()


# In[9]:


data.info()


# In[25]:


data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)


# In[27]:


data.columns = ['Category','Messages']


# In[29]:


data.sample(5)


# In[30]:


data.info()


# In[39]:


data['Category'].value_counts()


# In[38]:


data['Category'].value_counts().plot(kind='bar')


# In[40]:


data['Spam'] = data['Category'].apply(lambda x:1 if x=='spam' else 0)


# In[43]:


data.columns
data.sample(5)


# In[ ]:





# In[57]:


x = np.array(data['Messages'])
y = np.array(data['Spam'])


# In[58]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# In[59]:


cv = CountVectorizer()
X = cv.fit_transform(x)


# In[61]:


X_train,X_test, y_train,y_test = train_test_split(X,y, test_size=0.30,random_state=2)


# In[62]:


from sklearn.naive_bayes import MultinomialNB


# In[63]:


clf = MultinomialNB()
clf.fit(X_train,y_train)


# In[68]:


sample = input('Enter a Message: ')
data = cv.transform([sample]).toarray()
print(clf.predict(data))


# In[70]:


clf.score(X_test,y_test)


# In[ ]:





# In[ ]:




