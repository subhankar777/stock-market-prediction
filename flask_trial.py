from flask import Flask,request,jsonify,render_template
import math
import pandas as pd
import numpy as np
from textblob import TextBlob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Dense, Activation
from tensorflow.keras.models import Model,load_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler

# Defining a function which returns polarity
def detect_polarity(text):
    """Returns Polarity"""
    return TextBlob(text).sentiment.polarity

# Defining a function which returns subjectivity
def detect_subjectivity(text):
    """Returns Subjectivity"""
    return TextBlob(text).sentiment.subjectivity

def find_sentiment(text):
    polarity=detect_polarity(text)
    subjectivity=detect_subjectivity(text)
    analyzer = SentimentIntensityAnalyzer()
    compound=analyzer.polarity_scores(text)['compound']
    neutral=analyzer.polarity_scores(text)['neu']
    positive=analyzer.polarity_scores(text)['pos']
    negative=analyzer.polarity_scores(text)['neg']
    scores=[polarity,subjectivity,compound,neutral,positive,negative]
    return scores

model=load_model('stock_news_prediction.h5')

app=Flask(__name__)
@app.route('/',methods=['POST','GET'])
def trial1():
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        inputs=request.form.values()
        values=[]
        for i in inputs:
            if len(i)>0:
                values.append(i)
        print(values)
        if len(values)==3:
            open_price=float(values[0])    
            close_price=float(values[1])
            news=str(values[2])
            sentiment=find_sentiment(news)
            dataset=pd.read_csv("data.csv")
            dataset=dataset.drop('Date',axis=1)
            dataset=dataset.drop('News',axis=1)
            print(dataset.head(3))
            row=[open_price,close_price]
            row.extend(sentiment)
            dataset.loc[len(dataset)] = row
            print(dataset.tail())
            cols=['Open','Close','polarity','subjectivity','compound','neutral','positive','negative']
            scaler=MinMaxScaler(feature_range=(0,1))
            scaled_data=scaler.fit_transform(dataset[cols])
            print(scaled_data[-1])
            print(scaled_data[-2])
            data_in=[]
            #for i in range(len(scaled_data)-61, len(scaled_data)-1):
            data_in.append(scaled_data[len(scaled_data)-61:len(scaled_data)-1, 0:])
            data_in=np.array(data_in)
            print(data_in.shape)

            predictions=model.predict(data_in)
            print(predictions)
            predictions = scaler.inverse_transform(predictions)
            final_close=round(predictions[0][1],4)
            print(final_close)
            if predictions[0][1]>close_price:
                return render_template('index.html',output='<br>GIVEN <br> Open price = $ {}  &emsp;  Close Price = $ {}<br><br>NEWS ENTERED<br> {}<br><br>PREDICTIONS<br> Ploarity of News : {}  &emsp; Predicted Close Price : $ {}  &emsp;   PRICE UP : 1 '.format(open_price,close_price,news,sentiment[0],final_close))
            else:
                return render_template('index.html',output='<br>GIVEN <br> Open price = $ {}  &emsp;  Close Price = $ {}<br><br>NEWS ENTERED<br> {}<br><br>PREDICTIONS<br> Ploarity of News : {}  &emsp;  Predicted Close Price : $ {}  &emsp;   PRICE DOWN : 0'.format(open_price,close_price,news,sentiment[0],final_close))
        else:
            return render_template('index.html',output='INSUFFICIENT DATA : Please Enter details in all 3 feilds above')
    

    
    
if __name__=='__main__':
    app.run(debug=True)
