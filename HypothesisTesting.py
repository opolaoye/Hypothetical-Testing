#!/usr/bin/env python
# coding: utf-8

# ### Hypothesis Testing
# Hypothesis testing in statistics is a way for you to test the results of a survey or experiment to see if you have meaningful results. You’re basically testing whether your results are valid by figuring out the odds that your results have happened by chance. If your results may have happened by chance, the experiment won’t be repeatable and so has little use. 

# In[ ]:


#import libraries
import pandas as pd
import numpy as np
from numpy import sqrt, abs, round
from scipy.stats import norm


# In[10]:


# import corona virus update data that was downloaded from John Hopkins's repository
corona = pd.read_csv('Corona_Updated.csv')
corona.drop(['Temp_Cat', 'Humid_Cat'], axis=1, inplace=True)


# In[11]:


corona.head(3)


# ### implement Two-Sample Z test for a coronavirus dataset.
# A common perception about COVID-19 is that Warm Climate is more resistant to the corona outbreak and we need to verify this using Hypothesis Testing. So what will our null and alternate hypothesis.
# 
#     * Null Hypothesis: Temperature doesn’t affect COV-19 Outbreak
#     * Alternate Hypothesis: Temperature does affect COV-19 Outbreak
# 

# In[12]:


#We are considering Temperature below 24 as Cold Climate and above 24 as Hot Climate 
corona['Temp_Cat'] = corona['Temprature'].apply(lambda x : 0 if x < 24 else 1)


# In[13]:


#Keep only features Confirmed and Temp_Cat
corona_t = corona[['Confirmed', 'Temp_Cat']]


# In[14]:


corona_t.head(3)


# In[15]:


#Can get the mean of the confirmed cases with high tempearture and low temperature 
corona_t.groupby(['Temp_Cat']).mean()


# In[22]:


#Can get the standard deviation of the confirmed cases with high tempearture and low temperature 
corona_t.groupby(['Temp_Cat']).std()


# In[16]:


#An alternative way to get the data sliced into two dataframe with one being high temperature 
#and other low temperature. Get mean and std after
ht = corona_t[(corona_t['Temp_Cat']==1)]['Confirmed']
ct = corona_t[(corona_t['Temp_Cat']==0)]['Confirmed']


# In[17]:


#get the mean, standard deviation and sample size of both cases
m1, m2 = ht.mean(), ct.mean()
sd1, sd2 = ht.std(), ct.std()
n1, n2 = ht.shape[0], ct.shape[0]


# In[18]:


m1,m2,sd1,sd2,n1,n2


# In[19]:


def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
    ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
    z = (X1 - X2)/ovr_sigma
    pval = 2*(1 - norm.cdf(abs(z)))
    return z, pval


# In[20]:


z, p = TwoSampZ(m1, m2, sd1, sd2, n1, n2)


# In[21]:


z_score = np.round(z,8)
p_val = np.round(p,6)

if (p_val<0.05):
    Hypothesis_Status = 'Reject Null Hypothesis : Significant'
else:
    Hypothesis_Status = 'Do not reject Null Hypothesis : Not Significant'

print(p_val)
print(Hypothesis_Status)


# ### Conclusion
# In conclusion, we do not have evidence to reject our Null Hypothesis that temperature doesn’t affect the COVID-19 outbreak. Although we cannot find the Temperature’s impact on COVID-19, there are certain limitations of the Z test for COVID-19 datasets:
#     
#     * The sample data may not be well representation of the population data. As we can see, we have 15% of the population have low temperature, whic indicate an imbalance target set.
#     * Early breakout in certain places.
#     * Some states could be hiding the data for geopolitical reasons.
