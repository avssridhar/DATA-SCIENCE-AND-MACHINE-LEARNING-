#!/usr/bin/env python
# coding: utf-8

# In[1]:


#R2 TEST IS GIVEN IN REGRESSION 


# In[4]:


#HYPOTHESIS TESTING
#here we evaluate two mutually exclusive statement
#steps 
#1.make initial assumption(H0)-null hypothesis
#(there is alternative hypothesis also)
#2.collect data
#3.gather evidence to reject or not
#4.if you accept null hypothesis 
#5.if you reject alternate hypotyhesis correct
#type 1 error
#type 2 error
#see the matrix which looks like confusion matrix
#             h0          h1
#donot reject ok          type2 error 
#reject       type 1error ok


# In[13]:


#t test,chi square test,anova test,p value or significance value
#we will see when to use which test
#in any analysis we will have a sample data set and based on this we will 
#p value is also called significance value alpha 
#p value should be selected before a test is decided 
#we will fix the limit of p value where to reject 
#p value limits can be compared to the gaussian distribution curve 


# In[14]:


#take a categorical variable gender
#to check whether there is a proportion in the male and female 
#h0-there is no difference 
#h1=there is a differnce 
#test-we need to apply a testto determine which hypothesis is true 
#p<=0.05 for a particular test then that is more likely to occur 
#here lets take one sample proportion test 
#so for one categorical feature best test is 
#one smaple proportion test 
#p values should be seelcted before a particular test is done 


# In[15]:


#now take two categorical features 
#for example gender and age 
#here the type of test is chi square test 
#here also we fix a p value 


# In[16]:


#if we take the numerical variable 
#type of test is T test
#if we take two numerical variable 
#we use correlation test
#correlation value tells the dependenc eof one on other 


# In[17]:


#one numerical and one categorical variable
#we take anova test (when categorical variable has more than two categories)
#we take t test(if the categorical varibale hahs only two categorical variables )


# In[18]:


#implementation in python 


# In[19]:


# t test
#type of inferential statistic to determine where there is a significant difference btween
#the means of two grops which many be related in certain features


# In[20]:


ages=[10,20,35,50,28,40,55,18,16,55,30,25,43,18,30,28,14,24,16,17,32,35,26,27,65,18,43,23,21,20,19,70]


# In[21]:


len(ages)


# In[23]:


import numpy as np
ages_mean=np.mean(ages)
print(ages_mean)


# In[33]:


#is there any signifact differecne between mean of the population and sample
#this we will check using one sample t test
#h0-no difference 
#h1-there is differenc
#we use one sample t tset to find the p value
#accordingly we determine the correct hypothesis


# In[34]:


## Lets take sample

sample_size=10
age_sample=np.random.choice(ages,sample_size)


# In[35]:


age_sample


# In[37]:


from scipy.stats import ttest_1samp


# In[40]:


ttest,p_value=ttest_1samp(age_sample,30)


# In[41]:


print(p_value)


# In[42]:


if p_value < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")


# In[43]:


#some more examples 
#Consider the age of students in a college and in Class A


# In[44]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import math
np.random.seed(6)
school_ages=stats.poisson.rvs(loc=18,mu=35,size=1500)
classA_ages=stats.poisson.rvs(loc=18,mu=30,size=60)


# In[45]:



classA_ages.mean()


# In[46]:


_,p_value=stats.ttest_1samp(a=classA_ages,popmean=school_ages.mean())


# In[47]:


p_value


# In[48]:


school_ages.mean()


# In[49]:


if p_value < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")


# In[51]:


#Two-sample T-test With Python
#compares of the two independent groups in order to determine 
#whether the statistical ecidence associated pop means are sig different


# In[52]:


#h0-there is statistical difference 
#h1-there is no statostical differnce
np.random.seed(12)
ClassB_ages=stats.poisson.rvs(loc=18,mu=33,size=60)
ClassB_ages.mean()


# In[53]:


_,p_value=stats.ttest_ind(a=classA_height,b=ClassB_ages,equal_var=False)


# In[54]:


if p_value < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")


# In[59]:


#paired t test with pyhton
#when you want to check how different sampes from the same group are
weight1=[25,30,28,35,28,34,26,29,30,26,28,32,31,30,45]
weight2=weight1+stats.norm.rvs(scale=5,loc=-1.25,size=15)
#to check if wheighs are sig different
#h0=-not different
#h1-different


# In[60]:


print(weight1)
print(weight2)


# In[61]:


weight_df=pd.DataFrame({"weight_10":np.array(weight1),
                         "weight_20":np.array(weight2),
                       "weight_change":np.array(weight2)-np.array(weight1)})


# In[62]:


weight_df


# In[63]:


_,p_value=stats.ttest_rel(a=weight1,b=weight2)


# In[64]:


print(p_value)


# In[65]:


if p_value < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")


# In[66]:


#correlation 
#when you have two numerical vairbales 
import seaborn as sns
df=sns.load_dataset('iris')


# In[67]:



df.shape


# In[69]:


df.corr()


# In[70]:


sns.pairplot(df)


# In[71]:


#chi square test 
#two categorical variables from a single population
import scipy.stats as stats


# In[72]:


import seaborn as sns
import pandas as pd
import numpy as np
dataset=sns.load_dataset('tips')


# In[73]:


dataset.head()


# In[74]:


dataset_table=pd.crosstab(dataset['sex'],dataset['smoker'])
print(dataset_table)


# In[75]:


dataset_table.values


# In[76]:


#Observed Values
Observed_Values = dataset_table.values 
print("Observed Values :-\n",Observed_Values)


# In[79]:


val=stats.chi2_contingency(dataset_table)
#this function is present in stats library


# In[80]:


val


# In[81]:


Expected_Values=val[3]


# In[82]:


no_of_rows=len(dataset_table.iloc[0:2,0])
no_of_columns=len(dataset_table.iloc[0,0:2])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)
alpha = 0.05


# In[83]:


from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]


# In[84]:


print("chi-square statistic:-",chi_square_statistic)


# In[85]:


critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)


# In[86]:


#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('p-value:',p_value)


# In[87]:


if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# In[88]:


#anova test
#refer krishna naik github for the codes 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




