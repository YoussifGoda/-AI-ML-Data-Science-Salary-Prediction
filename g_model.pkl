#Needed Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score , classification_report
%matplotlib inline


df = pd.read_csv('salaries.csv')
df.columns = ['work_year', 'experience_level', 'employment_type','job_title', 'salary','salary_currency', 
              'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']


df.head()

#ensuring correct read
df.shape

#checking for null values
df.isnull().sum()

df['salary'].unique()

sns.boxplot(df['work_year'])

def remove_outliers_work_year(df):
    
    IQR = df['work_year'].quantile(0.75) - df['work_year'].quantile(0.25)
    
    Lower_range = df['work_year'].quantile(0.25) - (1.5 * IQR)
    Upper_range = df['work_year'].quantile(0.75) + (1.5 * IQR)

    df.loc[df['work_year'] <= Lower_range, 'work_year'] = Lower_range
    df.loc[df['work_year'] >= Upper_range, 'work_year'] = Upper_range

remove_outliers_work_year(df)

sns.boxplot(df['work_year'])

sns.boxplot(df['remote_ratio'])

def remove_outliers_remote_ratio(df):
    
    IQR = df['remote_ratio'].quantile(0.75) - df['remote_ratio'].quantile(0.25)
    
    Lower_range = df['remote_ratio'].quantile(0.25) - (1.5 * IQR)
    Upper_range = df['remote_ratio'].quantile(0.75) + (1.5 * IQR)

    df.loc[df['remote_ratio'] <= Lower_range, 'remote_ratio'] = Lower_range
    df.loc[df['remote_ratio'] >= Upper_range, 'remote_ratio'] = Upper_range

remove_outliers_remote_ratio

sns.boxplot(df['remote_ratio'])

#removing irrelevant columns
df = df.drop('remote_ratio', axis =1)
df = df.drop('salary_in_usd', axis =1)

df.head()

#NUMERATING VALUES
def feature_engineering(df):

    #Numerating work_year
    label_encoding_work_year = {value : key  for key , value in enumerate(df['work_year'].unique())}
    df['work_year'] = df['work_year'].map(label_encoding_work_year)
    
    #Numerating experience_level
    label_encoding_experience_level = {value : key  for key , value in enumerate(df['experience_level'].unique())}
    df['experience_level'] = df['experience_level'].map(label_encoding_experience_level)
    
    #Numerating employment_type
    label_encoding_employment_type = {value : key  for key , value in enumerate(df['employment_type'].unique())}
    df['employment_type'] = df['employment_type'].map(label_encoding_employment_type)
    
    #Numerating job_title
    label_encoding_job_title = {value : key  for key , value in enumerate(df['job_title'].unique())}
    df['job_title'] = df['job_title'].map(label_encoding_job_title)
    
    #Numerating salary_currency
    label_encoding_salary_currency = {value : key  for key , value in enumerate(df['salary_currency'].unique())}
    df['salary_currency'] = df['salary_currency'].map(label_encoding_salary_currency)
    
    #Numerating employee_residence
    label_encoding_employee_residence = {value : key  for key , value in enumerate(df['employee_residence'].unique())}
    df['employee_residence'] = df['employee_residence'].map(label_encoding_employee_residence)
    
    #Numerating company_location
    label_encoding_company_location = {value : key  for key , value in enumerate(df['company_location'].unique())}
    df['company_location'] = df['company_location'].map(label_encoding_company_location)
    
    #Numerating company_size
    label_encoding_company_size = {value : key  for key , value in enumerate(df['company_size'].unique())}
    df['company_size'] = df['company_size'].map(label_encoding_company_size)
    
    return df

df = feature_engineering(df)
df.head()
sc = StandardScaler()
x = df[['work_year', 'experience_level', 'employment_type','job_title', 'salary_currency', 
               'employee_residence', 'company_location', 'company_size']]
y= df['salary']
y.value_counts()
#mean = 0, Standard Devieation = 1
x = sc.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=42) #%80 for training
lg_model = LogisticRegression()
lg_model.fit(x_train,y_train)
y_pred = lg_model.predict(x_test)
result = {'Actual': y_test, 'Prediction': y_pred}
pd.DataFrame(result)
print('Accuracy score: {}'.format(accuracy_score(y_test,y_pred)), '\n')
print('Confusion matrix:\n {}'.format(confusion_matrix(y_test,y_pred)), '\n')
print('classification report:\n {}'.format(classification_report(y_test,y_pred)), '\n')
file = open('lg_model.pkl', 'wb')  #storing data
pickle.dump(lg_model, file)
