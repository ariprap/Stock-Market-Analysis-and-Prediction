# Stock-Market-Analysis-and-Prediction
Stock market prediction is one of the most researched topics in various fields like finance, economics and mathematics. But because of the volatile nature of the stock market and complexity of the stock market data it becomes difficult to predict accurate stock price. 

The stock market is basically a time series data i.e. it is a sequence of numerical data points over a specific period of time. Making predictions about the future stock prices involves handling this time series data and applying time series forecasting on it. 

**Methodology-
The following steps were followed for this project:
•	Data collection from Yahoo finance.
•	Data preprocessing on collected data.
•	Designed a simple feed forward ANN model.
•	Analyzed the model and understood its shortcoming in dealing with time series.
•	Designed a LSTM (RNN) regressor model for handling the time series data.

Data collection was done from yahoo finance. Stock prices of Apple, Amazon, Microsoft and Google from year 2010 – 2018 were used to train the models. Stock prices from January 2019 to March 2019 were used as test datasets. 

**First Model-ANN**
The first model was a simple ANN. For input layer,timestamp are used as the number of input nodes. It is followed by 2 hidden layers and a final output layer. We used TensorFlow frame work for the implementation of the above model.
After implementing ANN model, it was found that the model was good for only short periods of time. For longer periods of time, its prediction was very different from the actual value. Also, ANNs are very inefficient and inaccurate in handling time series data and also, they were incapable of capturing stock market trends. To overcome the above limitations, we implemented LSTM model. LSTMs are better in dealing with time series data and also, they can successfully capture long term and short-term stock market trends. 

**Second Model-LSTM(Long Short-Term Memory)**
The LSTM model was implemented in Keras frame work using TensorFlow backend. To increase the speed of our algorithm,cuda cores were used to compute our algorithm. Our LSTM model consist of total 6-layer, one input layer with number of input nodes equal to the timestamp, 4 hidden layer and 1 output dense layer.

The model built by us using **LSTM gave an error of approx. 3% (Accuracy-97%)** which was achieved after tuning the hyper-parameters.The  basic **ANN model gave an error of approximately 7% (Accuracy-93%).**
The robustness of our model was tested by comparing the error percentages and accuracy it gave for various stock prices of companies like Amazon, Google, Microsoft and Apple. We also compared the performance of our LSTM model with our base ANN model.

The LSTM model works very well for people who are looking for predicted prices over a short period as well as long period of time.

**The project contains the code for the 2 models and images of the final graphs plotted after prediction using ANN and LSTM**

