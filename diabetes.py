#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('diabetes.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[7]:


df.corr()


# In[8]:


diabetes_true = len(df.loc[df['Outcome']== 1 ])
diabetes_false = len(df.loc[df['Outcome']== 0 ])


# In[9]:


(diabetes_true,diabetes_false)


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
predicted = ['Outcome']


# In[12]:


X = df[features].values
y = df[predicted].values


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)


# In[14]:


print("total number of rows : {0}".format(len(df)))
print("number of rows missing Glucose: {0}".format(len(df.loc[df['Glucose'] == 0])))
print("number of rows missing BloodPressure	: {0}".format(len(df.loc[df['BloodPressure'] == 0])))
print("number of rows missing SkinThickness: {0}".format(len(df.loc[df['SkinThickness'] == 0])))
print("number of rows missing Insulin: {0}".format(len(df.loc[df['Insulin'] == 0])))
print("number of rows missing BMI: {0}".format(len(df.loc[df['BMI'] == 0])))
print("number of rows missing DiabetesPedigreeFunction: {0}".format(len(df.loc[df['DiabetesPedigreeFunction'] == 0])))
print("number of rows missing Age: {0}".format(len(df.loc[df['Age'] == 0])))


# In[15]:


from sklearn.impute import SimpleImputer
fill_val = SimpleImputer(missing_values = 0, strategy = 'mean')
X_train = fill_val.fit_transform(X_train)
X_test = fill_val.fit_transform(X_test)


# In[16]:


from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)

random_forest_model.fit(X_train, y_train.ravel())


# In[17]:


predict_train_data = random_forest_model.predict(X_test)

from sklearn import metrics

print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))


# In[18]:


# params={
#  "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
#  "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
#  "min_child_weight" : [ 1, 3, 5, 7 ],
#  "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
#  "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
# }


# In[19]:


# from sklearn.model_selection import RandomizedSearchCV
# import xgboost


# In[20]:


# classifier=xgboost.XGBClassifier()
# random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[21]:


# random_search.fit(X,y.ravel())

# random_search.best_estimator_


# In[22]:


# classifier = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.7, gamma=0.4, gpu_id=-1,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=0.05, max_delta_step=0, max_depth=12,
#               min_child_weight=3, missing=None, monotone_constraints='()',
#               n_estimators=100, n_jobs=0, num_parallel_tree=1,
#               objective='binary:logistic', random_state=0, reg_alpha=0,
#               reg_lambda=1, scale_pos_weight=1, subsample=1,
#               tree_method='exact', validate_parameters=1, verbosity=None)


# In[23]:


# from sklearn.model_selection import cross_val_score
# score=cross_val_score(classifier,X,y.ravel(),cv=10)


# In[24]:


# score


# In[25]:


# score.mean()


# In[26]:


import pickle

pickle.dump(random_forest_model, open('model.pkl','wb'))


# In[27]:


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1,89,66,23,94,28.1,0.167,21]]))


# In[31]:


X_test


# In[ ]:




