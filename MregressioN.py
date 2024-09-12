'''
In this file we are going to develop multiple linear regression project.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error,root_mean_squared_error

class TRAINING:
    def __init__(self,location):
        try:
            self.df = pd.read_csv(location, encoding='latin-1')
            # print(self.df)
            self.df = self.df.drop(['Customer Name'], axis=1)
            self.df = self.df.drop(['Customer e-mail'], axis=1)
            self.df = self.df.drop(['Country'], axis=1)
            print(self.df)
            self.X = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f"error from the line{err_line.tb_lineno}->type {error_type}->message{error_msg}")
    def data_training(self):
        try:
            self.reg = LinearRegression()
            self.reg.fit(self.X_train, self.y_train)
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f"error from the line{err_line.tb_lineno}->type {error_type}->message{error_msg}")

    def train_performance(self):
            try:
                self.y_train_pred = self.reg.predict(self.X_train)
                print("train Accuracy :", r2_score(self.y_train,self.y_train_pred))
                print("Train Loss(mean_Squared_error): ",mean_squared_error(self.y_train,self.y_train_pred))
                print("Train Loss(mean_absolute_error): ", mean_absolute_error(self.y_train, self.y_train_pred))
                print("Train Loss:(root_mean_square_error):", root_mean_squared_error(self.y_train, self.y_train_pred))

            except Exception as e:
                error_type, error_msg, err_line = sys.exc_info()
                print(f"error from the line{err_line.tb_lineno}->type {error_type}->message{error_msg}")


    def testing(self):
        try:
            self.y_test_pred = self.reg.predict(self.X_test)
            print("test Accuracy :", r2_score(self.y_test, self.y_test_pred))
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f"error from the line{err_line.tb_lineno}->type {error_type}->message{error_msg}")

if __name__ == "__main__":
    try:
        obj = TRAINING("C:\\Users\\blessy sharon\\Downloads\\Machine learning\\Multiple_regression\\Car_Purchasing_Data.csv")
        obj.data_training()
        obj.train_performance()
        obj.testing()

    except Exception as e:
        error_type,error_msg,err_line=sys.exc_info()
        print(f"error from the line{err_line.tb_lineno}->type {error_type}->message{error_msg}")