#!/usr/bin/env python
# coding: utf-8

# # Understaning Credit Score Algorithms
# 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score


# In[4]:


# Creating a basic DataFrame  
df = pd.DataFrame()
df['age'] = [26, 25, 20, 22, 27, 30, 36, 40, 50]
df['Employed'] = [0, 1, 0, 0, 0, 1, 1, 1, 0]
df['good'] = [0, 1, 0, 0, 1, 0, 1, 1, 1]


# In[5]:


# Features
train = df[['age', 'Employed']]
# Target
y = df['good']


# In[6]:


# Initializing the Logistic Regression model
clf = LogisticRegression(fit_intercept=True, solver='lbfgs')
clf.fit(train, y)


# In[7]:


coefficients = np.append(clf.intercept_, clf.coef_)


# In[12]:


print('Coefficients:', coefficients)
# B0, B1, B2


# In[13]:


case_one = [20,0]


# In[22]:


# Predicting probability using the model
# using predict_proba on the training data 
y_pred_proba = clf.predict_proba(train)[:, 1]


# In[23]:


#extracting the probablity for the test case
test_index = train[train['age'] == case_one[0]].index[0]  # Find the index of the test case in the training data
y_pred = y_pred_proba[test_index]
print('Predict probability of being good (proba):', y_pred)


# In[24]:


# Calculating Probability from coefficients
# ln(odds) = sum(coefficients * values)
ln_odds = sum(np.multiply(coefficients, np.array([1] + case_one)))
odds = np.exp(ln_odds)
prob_good = odds / (1 + odds)
print('Resulting probability of being good (sum):', prob_good)


# In[25]:


# Confusion Matrix to summarize the performance
y_pred_class = clf.predict(train)
cm = confusion_matrix(y, y_pred_class)
print('Confusion Matrix:')
print(cm)


# In[26]:


# ROC AUC Score
# to measure the ability of the model to distinguish 
# between the positive class ("good") and the negative class ("bad") 
roc_auc = roc_auc_score(y, y_pred_proba)
print('ROC AUC Score:', roc_auc)


# In[27]:


pdo = 20 #points to double the odds 
offset = 200 # is an offset value to calibrate the credit scores
factor = pdo / np.log(2) #scaling factor to adjust the scores based on pdo values


# In[28]:


# Calculating scores for different scenarios
score1 = offset + factor * np.log(1)  # p_bad = 0.5, bad = good, odds = 1
score2 = offset + factor * np.log(2)  # p_bad = 0.3, good = 2, bad = 1
score3 = offset + factor * np.log(4)  # p_bad = 0.2, good = 4, bad = 1


# In[30]:


print(f'scrore 1: {score1}, score 2: {score2}, score 3: {score3}')


# In[31]:


print(f'Difference 2 and 1: {score2 - score1}\nDifference 3 and 2: {score3 - score2}')


# In[32]:


# Score from Logistic Regression
score_from_regression = offset + factor * sum(np.multiply(coefficients, np.array([1] + case_one)))
print(f'Score from regression: {round(score_from_regression, 0)}')


# In[33]:


# Score from probability
score_from_probability = offset + factor * np.log(prob_good / (1 - prob_good))
print(f'Score from probability: {round(score_from_probability, 0)}')


# In[ ]:




