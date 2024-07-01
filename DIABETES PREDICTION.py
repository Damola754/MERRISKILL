#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import metrics


# In[6]:


df = pd.read_csv('diabetes.csv')


# In[7]:


df.head()


# In[10]:


sns.heatmap(df.isnull())


# In[11]:


correlation = df.corr()


# In[12]:


print(correlation)


# In[13]:


plt.figure(figsize=(8, 6))


# In[15]:


sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)


# In[16]:


plt.title('correlation matrix Heatmap')


# In[20]:


plt.show(block=True)


# In[22]:


target = "Outcome"
X = df.drop(columns=target)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


model = LogisticRegression()


# In[24]:


model.fit(X_train, y_train)


# In[25]:


y_pred = model.predict(X_test)


# In[26]:


print(y_pred)


# In[28]:


from sklearn.ensemble import RandomForestClassifier


# In[30]:


from sklearn.metrics import accuracy_score


# In[31]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[32]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[35]:


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[36]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)


# In[38]:


conf_matrix = confusion_matrix(y_test, y_pred)


# In[39]:


plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.title('Confusion Matrix')
plt.show()

