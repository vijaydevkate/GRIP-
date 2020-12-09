#!/usr/bin/env python
# coding: utf-8

# # GRIP - Spark Foundation

# ## Data Science and Business Analytics Internship

# # Task - 1 : Predict the percentage of an student based on the no. of study hours.

# ## By : Vijay Devkate

# # Problem Satement :

# ## we need to Predict the percentage of an student based on the no. of study hours.

# ## Project Objectives :

# ### - Make use of Linear Regression Algorthim 
# ### - Predict the score if student studies for 9.25 hrs/day

# # The Following is the project code : 

# ## Import Libraries

# In[1]:


# import the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Importing and reading the data

data = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
data.head()


# In[7]:


# Checking for the shape of the data

data.shape


# In[8]:


# Describing the data

data.describe()


# In[9]:


# Checking data information

data.info()


# In[10]:


# Checking for null values

data.isnull().sum()

# No null values present


# In[11]:


# Data Visualization

sns.pairplot(data)
plt.show()


# In[12]:


# Checking the realtion between scores and hours

data.plot(x = 'Hours', y = 'Scores', style = 'o')
plt.title("Hours vs scores")
plt.xlabel("Hours")
plt.ylabel("Scores")

plt.show()


# In[13]:


#Dviding data into input "attributes" and ouput "labels"

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# In[14]:


# Spliting the data into training and testing data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, test_size = 0.3)
print(x_train.shape)
print(x_test.shape)


# In[15]:


# Training Algorthim

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR = LR.fit(x_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)


# In[16]:


# Plotting for regression line

line = LR.coef_*x+LR.intercept_

#Plotting for test data

plt.scatter(x,y)
plt.plot(x,line);
plt.show()


# In[17]:


print(x_test)

# Predicting scores

y_pred = LR.predict(x_test)


# In[18]:


# Comparing Actual and Predicted score

df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df


# In[19]:


# finding out mean absolute error and r2 values

from sklearn import metrics
print("Mean absolute error:" , metrics.mean_absolute_error(y_test,y_pred))
print("R-squared:", metrics.r2_score(y_test,y_pred))


# In[20]:


# Finding the scores for studied hour 9.25

hours = 9.25
new_pred = LR.predict([[hours]])
print("Number of hours = {}".format(hours))
print("Predicted percentage score = {}".format(new_pred[0]))


# In[ ]:




