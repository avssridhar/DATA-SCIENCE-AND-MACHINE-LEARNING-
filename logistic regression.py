#!/usr/bin/env python
# coding: utf-8

# In[1]:


#logistic regression 
#used for binary classification,multiclass classification 
#this is a classifictation algorithm but called regression
#here also we will create best fit line so it is called regression
#just drawing a line of seperaton that is the meaning 
#if we use linear regression some points will get wrongly classifies 
#when we have lot of points or outliers then we will get wrong best fit lien


# In[2]:


#binary classification
#logistic regression applied when two are linearly seperable
#best fit line should linearly seperate the classifcation points
#here best fit line is not calculated using linear regression
#assumptions 
#positive points-+1
#negative points--1
#y=wTx
#costfunction=sigma(1 to n)yi wT xi should be maximum
#update wi till you get maximum cost function
#cost function is optimizer
#line which maximum cost functions is the best fit line


# In[3]:


#to prevent the effect of outliers we introduce sigmoid function
#check effect of outliers in the video itself
#y=sigma(1ton)sigmoid(yi*wTxi)
#sigmoid=1/1+e^x
#sigmoid fucntion will transformall the thing between 0 and 1
#this will help balance the effect of outliers


# In[ ]:


#one vs rest or one vs all

