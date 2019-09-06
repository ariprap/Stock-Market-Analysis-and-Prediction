if __name__ == '__main__':

    # Recurrent Neural Network



    #Data Preprocessing

    # Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from keras.layers import CuDNNLSTM
    
    
    #defineing hyper parameters
    
    dataset_train = pd.read_csv('AMZN_TRAIN.csv')
    dataset_test = pd.read_csv('AMZN_TEST.csv')
    
    
    timestamps = 60
    node_l2 = 100
    node_l3 = 70
    node_l4 = 30
    node_l5 = 20
    node_output = 1
    epochs = 10
    batch_size = 32
    

    # Importing the training set
    
    
    
    training_set = dataset_train.iloc[:, 1:2].values

    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)


    # Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    for i in range(timestamps, len(training_set)):
        X_train.append(training_set_scaled[i-timestamps:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    
    #Building the RNN

    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    # Initialising the RNN
    regressor = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(CuDNNLSTM(units = node_l2, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(CuDNNLSTM(units = node_l3, return_sequences = True))
    regressor.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(CuDNNLSTM(units = node_l4, return_sequences = True))
    regressor.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(CuDNNLSTM(units = node_l5))
    regressor.add(Dropout(0.2))

    # Adding the output layer
    regressor.add(Dense(units = node_output))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)



    #Making the predictions and visualising the results

    # Getting the real stock price of 2019

    real_stock_price = dataset_test.iloc[:, 1:2].values

    # Getting the predicted stock price of 2019
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - timestamps:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(timestamps, timestamps + len(dataset_test)):
        X_test.append(inputs[i-timestamps:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    
    
    def total_accuracy(original_val, predicted_val):

            total_error = 0
            for i in range(0, original_val.size):
                err = abs(original_val[i] - predicted_val[i])/original_val[i]
                total_error += err
            print("Error: ",total_error*100/original_val.size," %")
            print("Accuracy: ",100 - total_error*100/original_val.size," %")



    import math
    from sklearn.metrics import mean_squared_error
    rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    print("rmse=",rmse)
    # Visualising the results
    plt.plot(real_stock_price, color = 'red', label = 'Real Amazon Stock Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Amazon Stock Price')
    plt.title('Amazon Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Amazon Stock Price')
    plt.legend()
    plt.savefig("graph_amazon.png")
    plt.show()
    
    
    total_accuracy(real_stock_price, predicted_stock_price)



