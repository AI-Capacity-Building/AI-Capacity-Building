# -*- coding: utf-8 -*-
"""how-severity-the-accidents-is.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Vfk1vMRh2WWsazHxicDh-ZiMpF98mK0l
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt 
import mlflow
import mlflow.sklearn
import xgboost as xgb 


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

train_df = pd.read_csv('US_Accidents_May19.csv')
print(train_df.shape)
print(train_df.head())

train_df.Source.unique()

states = train_df.State.unique()

count_by_state=[]
for i in train_df.State.unique():
    count_by_state.append(train_df[train_df['State']==i].count()['ID'])

fig,ax = plt.subplots(figsize=(16,10))
sns.barplot(states,count_by_state)

"""this says that California this with high accidents


lets go for EDA 

check for missing values 
"""

missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name','missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.5
fig,ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind,missing_df.missing_count.values,color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()

"""here some data pages with lat and lng"""

sns.jointplot(x=train_df.Start_Lat.values,y=train_df.Start_Lng.values,height=10)
plt.ylabel('Start_Lat', fontsize=12)
plt.xlabel('Start_Lng', fontsize=12)
plt.show()

sns.jointplot(x=train_df.End_Lat.values,y=train_df.End_Lng.values,height=10)
plt.ylabel('End_Lat', fontsize=12)
plt.xlabel('End_Lng', fontsize=12)
plt.show()

"""lets check for the top5  Weather Condition for accidents"""

fig, ax=plt.subplots(figsize=(16,7))
train_df['Weather_Condition'].value_counts().sort_values(ascending=False).head(5).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Weather_Condition',fontsize=20)
plt.ylabel('Number of Accidents',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('5 Top Weather Condition for accidents',fontsize=25)
plt.grid()
plt.ioff()

"""lets sapater the datasets based on dtype so that we can make good analysis """

dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df

dtype_df.groupby("Column Type").aggregate('count').reset_index()

"""get the ratio and the columns with more missing values above 80%"""

missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['columns_name','missing_count']
missing_df['missing_ratio'] = missing_df['missing_count'] /train_df.shape[0]
missing_df.loc[missing_df['missing_ratio']>0.777]

missin = missing_df.loc[missing_df['missing_count']>250000]
removelist = missin['columns_name'].tolist()
removelist

"""we move on making some changes ... """

train_df['Start_Time'] = pd.to_datetime(train_df['Start_Time'], errors='coerce')
train_df['End_Time'] = pd.to_datetime(train_df['End_Time'], errors='coerce')

# Extract year, month, day, hour and weekday
train_df['Year']=train_df['Start_Time'].dt.year
train_df['Month']=train_df['Start_Time'].dt.strftime('%b')
train_df['Day']=train_df['Start_Time'].dt.day
train_df['Hour']=train_df['Start_Time'].dt.hour
train_df['Weekday']=train_df['Start_Time'].dt.strftime('%a')

# Extract the amount of time in the unit of minutes for each accident, round to the nearest integer
td='Time_Duration(min)'
train_df[td]=round((train_df['End_Time']-train_df['Start_Time'])/np.timedelta64(1,'m'))

neg_outliers=train_df[td]<=0

# Set outliers to NAN
train_df[neg_outliers] = np.nan

# Drop rows with negative td
train_df.dropna(subset=[td],axis=0,inplace=True)

feature_lst=['Source','TMC','Severity','Start_Lng','Start_Lat','Distance(mi)','Side','City','County','State','Timezone','Temperature(F)','Humidity(%)','Pressure(in)', 'Visibility(mi)', 'Wind_Direction','Weather_Condition','Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop','Sunrise_Sunset','Hour','Weekday', 'Time_Duration(min)']

df = train_df[feature_lst].copy()
df.info()

"""Since there are so many variables, let us first take the 'float' variables alone and then get the correlation with the target variable to see how they are related."""

x_cols = [col for col in df.columns if col not in ['Severity'] if df[col].dtype=='float64']

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(df[col].values, df.Severity.values)[0,1])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')

ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,40))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()

"""The correlation of the target variable with the given set of variables are low overall.

there are some variable with no correlation
"""

corr_zero_columns = ['Turning_Loop','Visibility(mi)','Pressure(in)','Humidity(%)','Temperature(F)','TMC']
for col in corr_zero_columns:
    print(col,len(df[col].unique()))

"""get highly correlated columns"""

corr_df_sel = corr_df.loc[(corr_df['corr_values']>0.05) | (corr_df['corr_values'] < -0.05)]
corr_df_sel

corr_df_ = corr_df_sel.col_labels.tolist()

tem_df = df[corr_df_]

corrmat = tem_df.corr(method='spearman')
fig,ax= plt.subplots(figsize=(8,8))

sns.heatmap(corrmat,vmax=1,square = True)
plt.title('corr map',fontsize=15)
plt.show()

"""lets once check for all the vriables """

fig=plt.gcf()
fig.set_size_inches(20,20)
fig=sns.heatmap(df.corr(),annot=True,linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)

fig = plt.figure(figsize=(10,10)) 
fig_dims = (3, 2)


plt.subplot2grid(fig_dims, (0, 0))
df['Amenity'].value_counts().plot(kind='bar', 
                                     title='Amenity')
plt.subplot2grid(fig_dims, (0, 1))
df['Crossing'].value_counts().plot(kind='bar', 
                                     title='Crossing')
plt.subplot2grid(fig_dims, (1, 0))
df['Junction'].value_counts().plot(kind='bar', 
                                     title='Junction')
plt.subplot2grid(fig_dims, (1, 1))
df['Junction'].value_counts().plot(kind='bar', 
                                     title='Junction')

"""Severity of the accident oue target """

f,ax=plt.subplots(1,2,figsize=(18,8))
df['Severity'].value_counts().plot.pie(explode=[0,0.1,0.1,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Percentage Severity Distribution')
ax[0].set_ylabel('Count')
sns.countplot('Severity',data=df,ax=ax[1],order=df['Severity'].value_counts().index)
ax[1].set_title('Count of Severity')
plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(x="Severity", y="Wind_Chill(F)", data=train_df)
plt.ylabel('Wind_Chill(F)', fontsize=12)
plt.xlabel('Severity', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

train_df['Amenity'] = [1 if amenity==True else 0 for amenity in train_df['Amenity']]
train_df['Crossing']=[1 if crossing==True else 0 for crossing in train_df['Crossing']]
train_df['Junction']=[1 if junction==True else 0 for junction in train_df['Junction']]
train_df['Traffic_Signal']=[1 if traffic_signal==True else 0 for traffic_signal in train_df['Traffic_Signal']]
plt.figure(figsize=(12,8))
sns.violinplot(x='Severity', y='Amenity', data=train_df)
plt.xlabel('Severity', fontsize=12)
plt.ylabel('Amenity', fontsize=12)
plt.show()

plt.figure(figsize=(12,8))
sns.violinplot(x='Severity', y='Wind_Chill(F)', data=train_df)
plt.xlabel('Severity', fontsize=12)
plt.ylabel('Wind_Chill(F)', fontsize=12)
plt.show()

plt.figure(figsize=(12,8))
sns.violinplot(x='Severity', y='Crossing', data=train_df)
plt.xlabel('Severity', fontsize=12)
plt.ylabel('Crossing', fontsize=12)
plt.show()

plt.figure(figsize=(12,8))
sns.violinplot(x='Severity', y='Junction', data=train_df)
plt.xlabel('Severity', fontsize=12)
plt.ylabel('Junction', fontsize=12)
plt.show()

plt.figure(figsize=(12,8))
sns.violinplot(x='Severity', y='Traffic_Signal', data=train_df)
plt.xlabel('Severity', fontsize=12)
plt.ylabel('Traffic_Signal', fontsize=12)
plt.show()

"""ohhh ..... i think we have to check for the importance of the feature ... we will go with that ...."""

df.dropna(subset=df.columns[df.isnull().mean()!=0], how='any', axis=0, inplace=True)
df.shape

train_y = df['Severity'].values
x_cols = [col for col in df.columns if col not in ['Severity'] if df[col].dtype=='float64']
train_col= df[x_cols]

fearture_name = train_col.columns.values 

mlflow.set_tracking_uri("/accidents_usecase/mlruns")

from sklearn import ensemble 


model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)
mlflow.sklearn.autolog()
with mlflow.start_run():
    model.fit(train_col,train_y)

#plot imp 
importance = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
indices = np.argsort(importance)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importance[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), fearture_name[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()

"""seems like Start_lng . Traffic_signals , start,lat ,TMC are more important Feature and followed by ..

lets check with XGBoost also for Feature_importance_
"""

xgb_prames = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}

dtrain = xgb.DMatrix(train_col,train_y,feature_names=train_col.columns.values)

model = xgb.train(dict(xgb_prames, silent=0), dtrain, num_boost_round=50)


fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()

"""this gives the same includes Distance(mi)

to be contunued...soon 

**pleace upvote if you like that makes me motive... :)**
"""
print("done running")