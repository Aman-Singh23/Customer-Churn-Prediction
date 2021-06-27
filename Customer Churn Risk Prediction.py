#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Risk Prediction

# ## Problem Statement : 
#     To predict whether the customer/client will terminate their relations with the company or not.

# ## Synopsis :
# ### Business Understanding ( What is happening in the Company )
#     Suppose there is a X company(whether it is telecom, healthcare, gaming, banking, e-commerce, etc any types of company)  Suppose they launced a product and customer after using that product were not satisfied due to some reason and start switching to another Y company product and using it. 
#     Here it means... that customer churn from X company and will more to Y company or in simple word people starting leaving your business.
#     
#     No business can thrive without it's customer.
#     Customer leaving the business is just like a nightmare to that business owner.    
# 
#     In order to see that which customer can possibly leave the company , business owner need the " Customer Churn Score " of each customer.
# 
# ## Now..... What is Customer Churn Score ???
# 
# ### Customer Churn Score
#     Customer Churn refers to the process of identifying all the possible customer or clients who will terminate their relations with the company. It is a very important factor for any organization as it used to estimate the growth of the organisation but also for predicting trends of future customers.
#     The task in this project is to classify the customer on whether they will stay with the company or terminate their interrelation.
#      
#     One of the key metrics to measure a business' success is by measuring its customer churn rate - lower the chrun , the    more love towards the company.

# ## Task :
#     We have to build a sophisticated Machine Learning model that predicts the churn score of a customer for a company based on multiple features.

# ## About Dataset :
#     The dataset consists of parameters such as the user’s details, membership account details, duration and frequency of their visits to the website, points after purchasing and feedback, and many more.
#     
# ### The dataset containing 2 files 
#     Training files : contains total of 36992 records(rows) with 25 featutes(columns) including target variable.
#     Testing files : contains total of 19909 records with 24 features and target variable is not present in it.
#     
# ### About Null Value
#     Dataset contains some null values in some features (i.e... columns).
#     Some null values is in the form of '?' , 'Error' , 'Unknown' & need to be converted into "NAN" values. 
#         
# ### Some important features of dataset
#     points_in_wallet : Represents the points awarded to a customer on each transaction.
#     membership_category : Represents the category of the membership that a customer is using like... Premium, Platinum etc.
#     feedback : Represents the feedback provided by a customer.
#     avg_transaction_value : Represents the average transaction value of a customer.
#     avg_time_spent : Represents the average time spent by a customer on the website.
#     
# ### About Target variable( In Training Data )
#     churn_risk_score : Represents the churn risk score that ranges from 1 to 5.
#         less churn score, more love towards the company.
#         if high churn score...like 4 or 5 then customer/client can possibly leave the business.

# In[1]:


from IPython.display import Image
Image(r"C:\Users\Ninja Clasher\Desktop\Steps.jpg")


# ##### Importing Libraries

# In[2]:


import numpy as np           # used for advanced mathematical opertion.
import pandas as pd          # used for analysing and handling data.


# In[3]:


# for visualization
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# for ignoring warning
from warnings import filterwarnings
filterwarnings('ignore')


# ##### Importing Data

# In[5]:


df = pd.read_csv(r'C:\Users\Ninja Clasher\Downloads\98efc33085a711eb\dataset\train.csv' , na_values = ['?','Error','Unknown'])


# ##### Analysing Data

# In[6]:


df.head()


# ##### Analysing target variable

# In[7]:


df['churn_risk_score'].value_counts()


# In[8]:


#index_names = df[ df['churn_risk_score'] < 1 ].index
#index_names
df.drop(df[df['churn_risk_score'] < 1].index, inplace = True)


# In[9]:


df['churn_risk_score'].value_counts()


# ##### Dropping unnecessary features which will not help while building the model

# In[10]:


df.drop(['customer_id', 'Name' , 'security_no' ,
       'referral_id', 'last_visit_time'] , axis = 1 , inplace = True)


# In[11]:


df.info()       # Summary of a DataFrame


# In[12]:


df.describe()       # Statistical data of the numerical features of a DataFrame


# In[13]:


df.describe(include='O').T


# ## EDA (Exploratory Data Analysis)

# ##### Checking for Null values

# In[14]:


df.isnull().sum() 


# #### Checking Missing Values using " Bar Plot "

# In[15]:


plt.figure(figsize=(18,6))
g = sns.barplot(x=df.columns, y=df.isna().sum(), palette='Pastel2')
plt.xticks(rotation=90)
plt.title('Missing Values', size=16, color = '#025955')

for p in g.patches:
    g.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()+20),ha='center', va='bottom',
               color= 'black')
    
plt.show()

#plt.tight_layout()


# ### From the above barplot we can see that there are 7 columns containg null value.
#     
#     Gender : 56 Null Value
#     Region_category : 5263 Null Value
#     Joined_through_referral : 5292 Null Value
#     Preferred_offer_types : 276 Null Value
#     Medium_of_operation : 5230 Null Value
#     Avg_frequency_login_days : 3419 Null Value
#     Points_in_wallet : 3341 Null Value

# ##### Handling Missing Categorical Data

# In[16]:


df['region_category'] = df['region_category'].fillna(df['region_category'].mode()[0])
df['preferred_offer_types'] = df['preferred_offer_types'].fillna(df['preferred_offer_types'].mode()[0])

df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

df['joined_through_referral'] = df['joined_through_referral'].fillna(df['joined_through_referral'].mode()[0])
df['medium_of_operation'] = df['medium_of_operation'].fillna(df['medium_of_operation'].mode()[0])


# ##### Handling Missing Numerical Data

# In[17]:


df['points_in_wallet'] = df['points_in_wallet'].fillna(df['points_in_wallet'].mean())
df['avg_frequency_login_days'] = df['avg_frequency_login_days'].fillna(df['avg_frequency_login_days'].mean())


# In[18]:


df.isnull().sum()  # checking Null Value


#     By checking again we can see from above data that we have successfully handle the missing data.

# ##### Analysing joining_date 

# In[19]:


df[['joining_date']]


# ##### Converting to DateTime and Extracting Day , Month & Year from joining_date

# In[20]:


df['joining_day'] = pd.to_datetime(df.joining_date , format = "%Y/%m/%d").dt.day           # Extracting Day
df['joining_month'] = pd.to_datetime(df.joining_date , format = "%Y/%m/%d").dt.month       # Extracting Month
df['joining_year'] = pd.to_datetime(df.joining_date , format = "%Y/%m/%d").dt.year         # Extracting Year

df.drop(['joining_date'] , axis = 1 , inplace = True)      # Dropping joining_date 


# In[21]:


df.info()


# In[22]:


df.head()


# In[23]:


import sweetviz as sv
my_repo = sv.analyze(df)
my_repo.show_html()


# In[24]:


from pandas_profiling import ProfileReport
my_repo2 = ProfileReport(df)
my_repo2.to_file(output_file = "my_repo2.html")


# In[25]:


df.to_csv('Customer_Churn_Score Cleaned_Data.csv')


# ## Visualization

# ##### Visualizing the distribution of Churn_risk_score

# In[22]:


plt.figure(figsize=(18,8))
plt.subplot(1,2,1)
vc = df['churn_risk_score'].value_counts()
g = sns.barplot(x=vc.index,y=vc, palette='Pastel2')
for p in g.patches:
    g.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()+20),ha='center', va='bottom',
               color= 'black')
plt.title('Count of different classes')
plt.subplot(1,2,2)
colors = ['#CFD6E4', '#EFCFE3', '#E4F0CF', '#F3CFB6', '#B9DCCC']
df['churn_risk_score'].value_counts().plot(kind='pie', explode=[0.1,0,0.1,0,0.1], autopct='%.2f%%', colors=colors)
plt.title('Distribution of different classes')
plt.show()


# ### Pie Chart for Internet Options Used by Client/Customer

# In[23]:


plt.figure(figsize=(18,8))
df['internet_option'].value_counts().plot(kind='pie', explode=[0.06,0,0.06], autopct='%.f%%', colors=colors)
plt.title('Distribution of Internet Options')
plt.show()


# ### Pie Chart for Medium Used by Client/Customer buying

# In[24]:


plt.figure(figsize=(18,8))
df['medium_of_operation'].value_counts().plot(kind='pie', explode=[0.06,0,0.06], autopct='%.f%%', colors=colors)
plt.title('Distribution of Medium of Operation')
plt.show()


# ### Pie Chart for from which Region customer are

# In[25]:


plt.figure(figsize=(18,8))
df['region_category'].value_counts().plot(kind='pie', explode=[0.06,0,0.06], autopct='%.f%%', colors=colors)
plt.title('Distribution of Region Category')
plt.show()


# ## Visualizing countplot for :
#     internet_option
#     medium_of_operation
#     region_category
#     gender
#     
#       w.r.t churn_risk_score

# In[26]:



plt.figure(figsize=(24,10))

col= ['internet_option','medium_of_operation','region_category','gender']
i = 1
for a in col:
    
    plt.subplot(2,2,i)
    g = sns.countplot(x=a,hue='churn_risk_score',data=df,palette='Pastel2')
    for p in g.patches:
        g.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.08, p.get_height()+20),ha='center', va='bottom',
                   color= 'black')
    i = i+1
    
plt.tight_layout()


# ## Pre-processing

# ##### Handling Categorical Data

# In[27]:


colname=[]
for x in df.columns:
    if df[x].dtype=='object':
        colname.append(x)
colname


# ## Label Encoding
#     Fitting data
#         1. Fetching the unique value.
#         2. Arrange them in ascending order.
#         3. Map the values start with 0,1,2....
#     
#     Transforming data
#         Here it actually replace the data values in the DataFrame
# ##### Coverting categorical features into numerical using Label_Encoder

# In[28]:


from sklearn import preprocessing

le=preprocessing.LabelEncoder()

for x in colname:
    df[x]=le.fit_transform(df[x])


# In[29]:


df.head()


# In[30]:


lastcolumn = df.pop('churn_risk_score')
df['churn_risk_score'] = lastcolumn
df.head()


# In[45]:


X = df.drop('churn_risk_score',axis = 1)        # Independent Variable
Y = df['churn_risk_score']                      # Dependent Variable [target variable]
Y=Y.astype(int)


# In[46]:


print(X.shape)
print(Y.shape)


# In[47]:


from sklearn.model_selection import train_test_split

#Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                    random_state=10)  


# In[48]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# ## Scaling Data

# ### Using StandardScaler to scale the train and test sets into scaled versions.
#     Scaling is used to ensure uniformity across the dataset ( means... not to get bais on certain features )
#     We need to fit as well as transform for training part
#     And for testing part we only need to transform the data

# In[49]:


from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ##### Checking whether the data is scaled successfully or not.

# In[50]:


print(X_train)
print(" ---------------------------------------------------------------------------------------------------------  ")
print(X_test)


# ## Building Model

# ## Descison Tree

# In[37]:


from sklearn.tree import DecisionTreeClassifier


# In[38]:


model_DecisionTree=DecisionTreeClassifier(criterion="gini",random_state=10)

#fit the model on the data and predict the values
model_DecisionTree.fit(X_train,Y_train)
Y_pred=model_DecisionTree.predict(X_test)

#print(list(zip(Y_test,Y_pred)))


# In[39]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

plt.figure(figsize=(6, 6))
ax = plt.subplot()
cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g' ,cmap=plt.cm.Blues)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
plt.show()

print(classification_report(Y_test,Y_pred))
print("Accuracy of The Model :",accuracy_score(Y_test,Y_pred)*100)


# ### Tuning Decision Tree Using feature_importances_

# In[40]:


print(list(zip(df.columns,model_DecisionTree.feature_importances_)))


# ### Bar Plot graph of feature importances for better visualization

# In[41]:



plt.figure(figsize = (12,8))
feat_importances = pd.Series(model_DecisionTree.feature_importances_ , index = df.columns[0:-1])
feat_importances.nlargest(17).plot(kind = 'bar')
plt.show()


# In[42]:


newData = df[['points_in_wallet','membership_category','feedback','avg_transaction_value','avg_time_spent',
            'age','avg_frequency_login_days','joining_year','joining_day','complaint_status',
            'internet_option','preferred_offer_types','region_category','medium_of_operation','joining_month','past_complaint','churn_risk_score']]


X1 = newData.drop('churn_risk_score',axis = 1)        # Independent Variable
Y1 = newData['churn_risk_score']                      # Dependent Variable [target variable]
Y1=Y.astype(int)

#Split the data into test and train
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size=0.3,
                                                    random_state=10)

X_train1 = scaler.fit_transform(X_train1)
X_test1 = scaler.transform(X_test1)


# In[43]:


from sklearn.tree import DecisionTreeClassifier

model_DecisionTree=DecisionTreeClassifier(criterion="gini",random_state=10)

#fit the model on the data and predict the values
model_DecisionTree.fit(X_train1,Y_train1)
Y_pred1=model_DecisionTree.predict(X_test1)


# In[44]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

plt.figure(figsize=(6, 6))
ax = plt.subplot()
cm = confusion_matrix(Y_test1,Y_pred1)
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g' ,cmap=plt.cm.Blues)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
plt.show()

print(classification_report(Y_test1,Y_pred1))
print("Accuracy of The Model :",accuracy_score(Y_test1,Y_pred1)*100)


# ## AdaBoost

# In[51]:


#predicting using the AdaBoost_Classifier
from sklearn.ensemble import AdaBoostClassifier

model_AdaBoost=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=10),
                                  n_estimators=10,
                                  random_state=10)
#fit the model on the data and predict the values
model_AdaBoost.fit(X_train,Y_train)
Y_pred=model_AdaBoost.predict(X_test)


# In[52]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

plt.figure(figsize=(6, 6))
ax = plt.subplot()
cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g' ,cmap=plt.cm.Blues)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
plt.show()

print(classification_report(Y_test,Y_pred))
print("Accuracy of The Model :",accuracy_score(Y_test,Y_pred)*100)


# ## Random Forest

# In[53]:


#predicting using the Random_Forest_Classifier
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(n_estimators=100, random_state=10)

#fit the model on the data and predict the values
model_RandomForest.fit(X_train,Y_train)

Y_pred=model_RandomForest.predict(X_test)


# In[54]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

plt.figure(figsize=(6, 6))
ax = plt.subplot()
cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g' ,cmap=plt.cm.Blues)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
plt.show()

print(classification_report(Y_test,Y_pred))
print("Accuracy of The Model :",accuracy_score(Y_test,Y_pred)*100)


# ### Tuning Random Forest with Hyperparameter

# In[55]:


from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(criterion='gini',n_estimators=58, random_state=10,
                                         max_features= 10,max_depth = 6,min_samples_leaf= 12,min_samples_split= 18)

#fit the model on the data and predict the values
model_RandomForest.fit(X_train,Y_train)

Y_pred=model_RandomForest.predict(X_test)


# In[56]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

plt.figure(figsize=(6, 6))
ax = plt.subplot()
cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g' ,cmap=plt.cm.Blues)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
plt.show()

print(classification_report(Y_test,Y_pred))
print("Accuracy of The Model :",accuracy_score(Y_test,Y_pred)*100)


# ## Gradient Boosting

# In[57]:


#predicting using the Gradient_Boosting_Classifier
from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting=GradientBoostingClassifier(n_estimators=150,
                                                  random_state=10)

#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train,Y_train)

Y_pred=model_GradientBoosting.predict(X_test)


# In[58]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

plt.figure(figsize=(6, 6))
ax = plt.subplot()
cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g' ,cmap=plt.cm.Blues)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
plt.show()

print(classification_report(Y_test,Y_pred))
print("Accuracy of The Model :",accuracy_score(Y_test,Y_pred)*100)


# ### Tuning Gradient Boosting with Hyperparameter

# In[59]:


from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting=GradientBoostingClassifier(n_estimators=63,max_depth= 4,max_features= 12,
                                                  min_samples_leaf= 2,min_samples_split= 11,random_state=10)

#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train,Y_train)

Y_pred=model_GradientBoosting.predict(X_test)


# In[60]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

plt.figure(figsize=(6, 6))
ax = plt.subplot()
cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g' ,cmap=plt.cm.Blues)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
plt.show()

print(classification_report(Y_test,Y_pred))
print("Accuracy of The Model :",accuracy_score(Y_test,Y_pred)*100)


# ## Xtream Gradient Boosting

# In[61]:


from xgboost import XGBClassifier

model_XGradientBoosting=XGBClassifier(n_estimators=150,leaning_rate = 0.01 ,max_depth = 15, random_state=10)

#fit the model on the data and predict the values
model_XGradientBoosting.fit(X_train,Y_train)

Y_pred=model_XGradientBoosting.predict(X_test)


# In[62]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

plt.figure(figsize=(6, 6))
ax = plt.subplot()
cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g' ,cmap=plt.cm.Blues)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
plt.show()

print(classification_report(Y_test,Y_pred))
print("Accuracy of The Model :",accuracy_score(Y_test,Y_pred)*100)


# ## RandomizedSearchCV

# ### Using RandomizedSearchCV to find best parameter for XGB Classifier (Tuning XGB Classifier)

# In[63]:


from sklearn.model_selection import RandomizedSearchCV


# In[64]:


param_grid = {
    'learning_rate' : [1 , 0.5 , 0.1 , 0.01] ,
    'max_depth' : [3 , 5 , 10 , 20] ,
    'n_estimators' : [10 , 50 , 100 , 200]
}


# In[65]:


grid = RandomizedSearchCV(XGBClassifier(objective = 'binary:logistic') , param_grid , verbose = 3)


# In[66]:


grid.fit(X_train , Y_train)


# In[67]:


grid.best_params_


# ### Best parameter we get for XGB Classifier
#     n_estimators : 100
#     max_depth : 3
#     learning_rate : 0.1

# ### Applying XGBoosting with best parameter we get from RandomizedSearchCV

# In[68]:


from xgboost import XGBClassifier

model_XGradientBoosting=XGBClassifier(n_estimators=100,learning_rate= 0.1,max_depth=3,random_state=10)

#fit the model on the data and predict the values
model_XGradientBoosting.fit(X_train,Y_train)

Y_pred=model_XGradientBoosting.predict(X_test)


# In[69]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

plt.figure(figsize=(6, 6))
ax = plt.subplot()
cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g' ,cmap=plt.cm.Blues)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
plt.show()

print(classification_report(Y_test,Y_pred))
print("Accuracy of The Model :",accuracy_score(Y_test,Y_pred)*100)


# ## Voting Classifier

# In[73]:


from sklearn.ensemble import VotingClassifier

estimator = []

estimator.append(('RF' , RandomForestClassifier(criterion='gini',n_estimators=58, random_state=10,
                                         max_features= 10,max_depth = 6,min_samples_leaf= 12,min_samples_split= 18)))

estimator.append(('GB' , GradientBoostingClassifier(n_estimators=63,max_depth= 4,max_features= 12,
                                                  min_samples_leaf= 2,min_samples_split= 11,random_state=10)))

estimator.append(('XGB' , XGBClassifier(n_estimators=100,learning_rate= 0.1,max_depth=3,random_state=10)))

votingClassifier = VotingClassifier(estimators=estimator , voting = 'hard')

votingClassifier.fit(X_train,Y_train)

Y_pred=votingClassifier.predict(X_test)


# In[74]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

plt.figure(figsize=(6, 6))
ax = plt.subplot()
cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g' ,cmap=plt.cm.Blues)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
plt.show()

print(classification_report(Y_test,Y_pred))
print("Accuracy of The Model :",accuracy_score(Y_test,Y_pred)*100)


# In[75]:


from prettytable import PrettyTable

myTable = PrettyTable(["Algorithm", "Overall Accuracy", "Tuned Accuracy",])
  
myTable.add_row(["Decision Tree", "76.10 %", "76.52 %"])
myTable.add_row(["Random Forest", "76.80 %", "76.98 %"])
myTable.add_row(["Adaptive Boosting", "76.10 %", "-"])
myTable.add_row(["Gradient Boosting", "78.73 %", "78.82 %"])
myTable.add_row(["Xtream Gradient Boosting", "77.07 %", "78.67 %"])
myTable.add_row(["Voting Classifier", "78.64 %", "-"])

print(myTable)


#                Gradient Boosting is the Optimal Model

# ## Testing Data

# ### Pre-processing on Testing Data

# In[77]:


tf = pd.read_csv(r'C:\Users\Ninja Clasher\Downloads\98efc33085a711eb\dataset\test.csv' , na_values = ['?','Error','Unknown'])

tf.drop(['customer_id', 'Name' , 'security_no' ,
       'referral_id', 'last_visit_time'] , axis = 1 , inplace = True)

tf['region_category'] = tf['region_category'].fillna(tf['region_category'].mode()[0])
tf['preferred_offer_types'] = tf['preferred_offer_types'].fillna(tf['preferred_offer_types'].mode()[0])
tf['points_in_wallet'] = tf['points_in_wallet'].fillna(tf['points_in_wallet'].mean())

tf['gender'] = tf['gender'].fillna(tf['gender'].mode()[0])

tf['joined_through_referral'] = tf['joined_through_referral'].fillna(tf['joined_through_referral'].mode()[0])
tf['medium_of_operation'] = tf['medium_of_operation'].fillna(tf['medium_of_operation'].mode()[0])
tf['avg_frequency_login_days'] = tf['avg_frequency_login_days'].fillna(tf['avg_frequency_login_days'].mean())


tf['joining_day'] = pd.to_datetime(tf.joining_date , format = "%Y/%m/%d").dt.day
tf['joining_month'] = pd.to_datetime(tf.joining_date , format = "%Y/%m/%d").dt.month
tf['joining_year'] = pd.to_datetime(tf.joining_date , format = "%Y/%m/%d").dt.year

tf.drop(['joining_date'] , axis = 1 , inplace = True)


colname=[]
for x in tf.columns:
    if tf[x].dtype=='object':
        colname.append(x)
        

for x in colname:
    tf[x]=le.fit_transform(tf[x])

                                                                       
#tf.head()                                                                       
test = scaler.transform(tf)


# ### Applying Tuned Gradient Boosting Classifier on Testing Data

# In[78]:


from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting=GradientBoostingClassifier(n_estimators=63,max_depth= 4,max_features= 12,
                                                  min_samples_leaf= 2,min_samples_split= 11,random_state=10)

#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train,Y_train)

Y_pred=model_GradientBoosting.predict(test)


# In[79]:


Y_pred


# In[80]:


testing_data = pd.read_csv(r'C:\Users\Ninja Clasher\Downloads\98efc33085a711eb\dataset\test.csv' , na_values = ['?','Error','Unknown'])

Churn_Risk_Score=testing_data[['customer_id','Name']]
Churn_Risk_Score['Churn Score'] = Y_pred


# In[81]:


Churn_Risk_Score


# In[82]:


Churn_Risk_Score.to_csv('Customer_Churn_Score Data.csv')


# ### Some suggesstion to the company
# 
#     Firstly company should more focus to that customer whose having a churn score of 4 and 5.
#     They should cash back on product or we can say give more offer to the customer.
#     Give more attention.
#     Asking them what more things added to a particular product to enchane thier product quality.
# 
