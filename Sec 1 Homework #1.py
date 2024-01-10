#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import statsmodels.api as sm


# # 1.) Import Data from FRED

# In[4]:


data = pd.read_csv("TaylorRuleData.csv", index_col = 0)


# In[5]:


data.index = pd.to_datetime(data.index)


# In[8]:


data.dropna(inplace=True)


# In[9]:


data.head()


# # 2.) Do Not Randomize, split your data into Train, Test Holdout

# In[10]:


split_1 =int(len(data)*.6)
split_2 = int(len(data)*.9)
data_in = data[:split_1]
data_out = data[split_1:split_2]
data_hold = data[split_2:]


# In[11]:


X_in = data_in.iloc[:,1:]
y_in = data_in.iloc[:,0]
X_out = data_out.iloc[:,1:]
y_out = data_out.iloc[:,0]
X_hold = data_hold.iloc[:,1:]
y_hold = data_hold.iloc[:,0]


# In[12]:


# Add Constants
X_in = sm.add_constant(X_in)
X_out = sm.add_constant(X_out)  
X_hold = sm.add_constant(X_hold) 


# # 3.) Build a model that regresses FF~Unemp, HousingStarts, Inflation

# In[13]:


model1 = sm.OLS(y_in,X_in).fit()


# # 4.) Recreate the graph fro your model

# In[14]:


import matplotlib.pyplot as plt


# In[23]:


plt.figure(figsize = (12,5))

plt.plot(y_in)
plt.plot(y_out)
plt.plot(model1.predict(X_in))
plt.plot(model1.predict(X_out))

plt.ylabel("Fed Funds")
plt.xlabel("Time")
plt.title("Visualizing Model Accuracy")
plt.legend([])
plt.grid()
plt.show()


# ## "All Models are wrong but some are useful" - 1976 George Box

# # 5.) What are the in/out of sample MSEs

# In[16]:


from sklearn.metrics import mean_squared_error


# In[24]:


in_mse_1 = mean_squared_error(model1.predict(X_in),y_in)
out_mse_1 = mean_squared_error(model1.predict(X_out),y_out)


# In[25]:


print("Insample MSE : ", in_mse_1)
print("Outsample MSE : ", out_mse_1)


# # 6.) Using a for loop. Repeat 3,4,5 for polynomial degrees 1,2,3

# In[28]:


from sklearn.preprocessing import PolynomialFeatures


# In[38]:


max_degrees=3


# In[39]:


poly=PolynomialFeatures(degree=degrees)
X_in_poly=poly.fit_transform(X_in)
X_out_poly=poly.fit_transform(X_out)
dir(poly)#degree=2，poly gives quadratic form of x1^2 x1*x2 x2^2


# In[52]:


for degrees in range(1,max_degrees+1):
    print("DEGREE",degrees)
    
    poly=PolynomialFeatures(degree=degrees)
    X_in_poly=poly.fit_transform(X_in)
    X_out_poly=poly.transform(X_out)
   
    model1 = sm.OLS(y_in,X_in_poly).fit()
    
    pred_in=model1.predict(X_in_poly)
    pred_out=model1.predict(X_out_poly)
    
    #Aligning X_in.index to pred_in's index
    time_index_in=X_in.index
    time_index_out=X_out.index
    pred_in=pd.DataFrame(pred_in,index=time_index_in)
    pred_out=pd.DataFrame(pred_out,index=time_index_out)
    
    plt.figure(figsize = (12,5))
    plt.plot(y_in)
    plt.plot(y_out)
    plt.plot(pred_in)
    plt.plot(pred_out)
    
    plt.ylabel("Fed Funds")
    plt.xlabel("Time")
    plt.title("Visualizing Model Accuracy")
    plt.legend([])
    plt.grid()
    plt.show()
    
    in_mse_1 = mean_squared_error(model1.predict(X_in_poly),y_in)
    out_mse_1 = mean_squared_error(model1.predict(X_out_poly),y_out)
    print(in_mse_1)
    print(out_mse_1)


# # 7.) State your observations :

# 

# The predicted results are closest to the actual values when degree=1, and the Mean Squared Error (MSE) also reaches its minimum at degree=1. Therefore, the model with degree=1 is relatively better among the three models, but none of the models has been able to meet the expectations.
