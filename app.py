from flask import Flask, jsonify, request
import numpy as np
import pickle
import pandas as pd
# from sklearn.externals import joblib
# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)

def final_prediction(data,model):
  original_cols = ['CustomerId','Surname','CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
  current_cols = data.columns.values

  for col in current_cols:
    if col not in  original_cols:
      raise ColumnNotKnownError(f"{col} not found in data on which model was trained.. Please change your data")


   
  # preprocess data like as we did on train data
  # removing unwanted columns 
  data.drop('CustomerId', axis=1,inplace=True)
  data.drop('Surname', axis=1, inplace=True)

  # adding one more feature Isoverspending as done while training

  data['isOverSpending'] = data['Balance']>data['EstimatedSalary']

  # converting  categorical columns to Numeric using label encoding

  data['Gender'].replace({'Female':1, 'Male':0}, inplace=True) 
  data['Geography'].replace({'France':0, 'Germany':1, 'Spain':2}, inplace=True)
  data['isOverSpending'].replace({False:0,True:1}, inplace=True)

  predict = model.predict(data)
  return predict


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #lr = pickle.load(open('model.pkl','rb'))
    try:
      test_data = request.form.to_dict()
      for key, value in test_data.items():
      	val = test_data[key]
      	test_data[key] = [val]
      test_df = pd.DataFrame.from_dict(test_data,orient='columns')
      
      model = pickle.load(open('finalized_model.pkl', 'rb'))
      prediction = final_prediction(test_df, model)
      
      if prediction[0]==0:
      	return f"{test_data['Surname']} will not churn"
      else:
      	return f"{test_data['Surname']} will  churn"
    except Exception as ex:
      return flask.render_template('error.html')
   
    

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)