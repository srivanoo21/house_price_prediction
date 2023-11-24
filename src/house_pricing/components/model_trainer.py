import sys
import os
import numpy as np 
import pandas as pd
from scipy.stats import skew
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from src.house_pricing.exception import CustomException
from src.house_pricing.logger import logging
from src.house_pricing.entity import ModelTrainerConfig
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from src.house_pricing.utils import save_object


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def train_val_split(self, train_data):

        X = train_data.drop(labels='SalePrice', axis=1)
        Y = train_data['SalePrice']

        xtrain, xval, ytrain, yval = train_test_split(X, Y, test_size=0.2, random_state=40)
        print(xtrain.shape)
        print(xval.shape)
        
        return xtrain, xval, ytrain, yval


    def evaluate_models(train_data, target_column, xtrain, ytrain, xval, yval, models, param):
        try:
            report_val_score = {}
            report_cross_val_score = {}
            report_adjusted_r2_score = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para = param[list(models.keys())[i]]

                gs = GridSearchCV(model, para, cv=3)
                gs.fit(xtrain, ytrain)

                model.set_params(**gs.best_params_)
                model.fit(xtrain, ytrain)
                mean_cross_val_score = np.abs(np.mean(cross_val_score(model, train_data, target_column, scoring='neg_mean_absolute_error', cv=5)))

                ytrain_pred = model.predict(xtrain)
                yval_pred = model.predict(xval)
                
                train_model_score = r2_score(ytrain, ytrain_pred)
                val_model_score = r2_score(yval, yval_pred)

                n = xval.shape[0]
                p = xval.shape[1]
                adjusted_val_score = 1-(1-val_model_score)*(n-1)/(n-p-1)

                report_val_score[list(models.keys())[i]] = val_model_score
                report_cross_val_score[list(models.keys())[i]] = mean_cross_val_score
                report_adjusted_r2_score[list(models.keys())[i]] = adjusted_val_score

            return report_val_score, report_cross_val_score, report_adjusted_r2_score

        except Exception as e:
            raise CustomException(e, sys)
    

    # def model_evaluate(self, xval, yval, ypred, model_type):
        
    #     print(f"model type is {model_type}")
    #     mean_cross_val_score = np.abs(np.mean(cross_val_score(dtr, X, Y, scoring='neg_mean_absolute_error', cv=5)))
    #     print(f"mean of cross validation score is {mean_cross_val_score}")
    #     print(f"mean absolute score for {model_type} is {mean_absolute_error(ypred, yval)}")
    #     print(f"R2 score is {r2_score(ypred, yval)}")
    #     r = r2_score(ypred, yval)
    #     n = xval.shape[0]
    #     p = xval.shape[1]
    #     adjr = 1-(1-r)*(n-1)/(n-p-1)
    #     print(f"Adjusted R2 score is {adjr}")


    # def model_train(self, xtrain, ytrain, xval, yval, model_type='RFR'):

    #     if model_type == "DTR":
    #         dtr = DecisionTreeRegressor(random_state=30, criterion='absolute_error', max_depth=10)
    #         model = dtr.fit(xtrain, ytrain)
    #     elif model_type == 'RFR':
    #         rfr = RandomForestRegressor(random_state=30, criterion='absolute_error', max_depth=10, n_estimators=20)
    #         model = rfr.fit(xtrain, ytrain)

    #     ypred = model.predict(xval)
    #     self.model_trainer_config.model_evaluate(xval, yval, ypred, model_type)

    
    def top_feature(self, model, X):
    
        imp = pd.DataFrame()
        imp['col'] = X.columns
        imp['importance'] = model.feature_importances_
        imp = imp.sort_values(by='importance', ascending=False)
        print(f"Top 10 important features are: {imp[:10]}")

    

    def initiate_model_trainer(self):
        try:
            train_data_path = self.model_trainer_config.preprocessed_train_data_path
            train_data = pd.read_csv(train_data_path)
            target_column = self.model_trainer_config.target_column
            logging.info("Pre-processed training dataset and target column is now loaded")

            xtrain, xval, ytrain, yval = self.train_val_split(train_data)
            logging.info("Splitting of training and validation data is now completed")

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor()              
            }
            params={
                "Linear Regression":{},
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                }
                # "Gradient Boosting":{
                #     # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                #     'learning_rate':[.1,.01,.05,.001],
                #     'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                #     # 'criterion':['squared_error', 'friedman_mse'],
                #     # 'max_features':['auto','sqrt','log2'],
                #     'n_estimators': [8,16,32,64,128,256]
                # },
                
                # "XGBRegressor":{
                #     'learning_rate':[.1,.01,.05,.001],
                #     'n_estimators': [8,16,32,64,128,256]
                # },
                # "CatBoosting Regressor":{
                #     'depth': [6,8,10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },
                # "AdaBoost Regressor":{
                #     'learning_rate':[.1,.01,0.5,.001],
                #     # 'loss':['linear','square','exponential'],
                #     'n_estimators': [8,16,32,64,128,256]
                # }   
            }

            model_report:dict = self.evaluate_models(train_data, target_column, xtrain, yval, xval, yval, models, params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and validation dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(xval)

            r2_square = r2_score(yval, predicted)
            return r2_square
                        
        except Exception as e:
            raise CustomException(e,sys)
        
