import numpy as np
import pandas as pd
data=pd.read_csv('Salary_Data.csv')
data


# In[2]:


x=data.YearsExperience.values
y=data.Salary.values
x


# In[3]:


x=data.YearsExperience.values.reshape(-1,1)
y=data.Salary.values.reshape(-1,1)
y


# In[5]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)


# In[10]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
new_salary=model.predict([[11]])
print(new_salary)


# In[11]:


model.score(xtrain,ytrain)

