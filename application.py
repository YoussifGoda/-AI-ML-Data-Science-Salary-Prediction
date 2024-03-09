from flask import Flask, make_response , request , render_template
import io
from io import StringIO
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

#NUMERATING VALUES
def feature_engineering(df):

    df.columns = ['work_year', 'experience_level', 'employment_type','job_title', 'salary_currency', 
               'employee_residence', 'remote_ratio', 'company_location', 'company_size']

    df = df.drop('remote_ratio', axis =1)

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
    
def scalar(df):
    sc= StandardScaler()
    x = df[['work_year', 'experience_level', 'employment_type','job_title', 'salary_currency', 
               'employee_residence', 'company_location', 'company_size']]
    x = sc.fit_transform(x)
    return(x)


@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])

def predict():
    f = request.files['data_file']
    if not f:
        return render_template('index.html', prediction_text='No File Selected')

    stream = io.StringIO(f.stream.read().decode('UTF8'), newline= None)
    
    result = stream.read()
    
    df = pd.read_csv(StringIO(result))
    
    df = feature_engineering(df)
    
    x = scalar(df)
    
    loaded_model = pickle.load(open('lg_model.pkl', 'rb'))
    
    print(loaded_model)
    
    result = loaded_model.predict(x)

    print(result)
    
    return render_template('index.html', prediction_text = 'Predicted salary is/are: {}'.format(result))

if __name__=='__main__':
    app.run(debug=False,port=8000)
