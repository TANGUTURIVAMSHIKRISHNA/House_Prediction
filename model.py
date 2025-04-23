import pandas as pd
import numpy as np

# read the data
house=pd.read_csv(r"C:\Users\Plhv\VS_Code\House_prediction\House_data.csv")

# Divide the data into dependent and independent variables
x=np.array(house['sqft_living']).reshape(-1,1)
y=np.array(house['price'])

# Split the data into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

# lets create the model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train, y_train)

# Let predict the values
predictions=model.predict(x_test)

# visualize the training data
import matplotlib.pyplot as plt
plt.scatter(x_train, y_train,color='red')
plt.plot(x_train,model.predict(x_train),color='blue')
plt.title('House Price Prediction(Training set)')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.show()

# visualize the testing data
plt.scatter(x_test, y_test,color='red')
plt.plot(x_train,model.predict(x_train),color='blue')
plt.title('House Price Prediction(Testing set)')
plt.xlabel('square Foot')
plt.ylabel('Price')
plt.show()



import pickle

filename='linear_regression_model_House.pkl'

with open(filename,'wb') as file:
    pickle.dump(model,file)

print("Model has been pickled and saved as linear_regression_model_House.pkl")


import os
os.getcwd()