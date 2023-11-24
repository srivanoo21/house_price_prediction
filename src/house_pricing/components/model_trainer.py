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
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from src.house_pricing.utils import save_object


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def train_val_split(self, train_data, target_column):
        '''
        split the input data into training and validation set'''

        X = train_data.drop(labels=[target_column], axis=1)
        Y = train_data[target_column]
        logging.info("Data spitting for input and target columns are now prepared")

        xtrain, xval, ytrain, yval = train_test_split(X, Y, test_size=0.2, random_state=40)
        logging.info(f"The shape of training data is {xtrain.shape}")
        logging.info(f"The shape of validation data is {xval.shape}")
        
        return xtrain, xval, ytrain, yval



    def initiate_model_trainer(self):
        '''
        initiate the model training for the training set'''

        try:
            train_data_path = self.model_trainer_config.preprocessed_train_data_path
            train_data = pd.read_csv(train_data_path)
            target_column = self.model_trainer_config.target_column
            logging.info("Pre-processed training dataset and target column is now loaded")

            xtrain, xval, ytrain, yval = self.train_val_split(train_data, target_column)
            logging.info("Splitting of training and validation data is now completed")

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
                # "Gradient Boosting": GradientBoostingRegressor(),
                # "XGBRegressor": XGBRegressor()
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
                },
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
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }   
            }

            train_report, val_report, cross_val_report, adj_r2_report = self.evaluate_models(
                train_data, target_column, xtrain, ytrain, xval, yval, models, params)
            
            logging.info(f"The training report is {train_report}")
            logging.info(f"The validation report is {val_report}")
            logging.info(f"The cross validation report is {cross_val_report}")
            logging.info(f"The adj_r2_report report is {adj_r2_report}")
            
            ## To get best model score from dict
            best_train_score = max(sorted(train_report.values()))
            best_val_score = max(sorted(val_report.values()))
            best_cross_val_score = max(sorted(cross_val_report.values()))
            best_adj_r2_score = max(sorted(adj_r2_report.values()))

            ## To get best model name from dict on different areas
            best_model_name_train = list(train_report.keys())[
                list(train_report.values()).index(best_train_score)]
            best_model_train = models[best_model_name_train]

            best_model_name_val = list(val_report.keys())[
                list(val_report.values()).index(best_val_score)]
            best_model_val = models[best_model_name_val]

            best_model_name_cross_val = list(cross_val_report.keys())[
                list(cross_val_report.values()).index(best_cross_val_score)]
            best_model_cross_val = models[best_model_name_cross_val]

            best_model_name_adj_r2 = list(adj_r2_report.keys())[
                list(adj_r2_report.values()).index(best_adj_r2_score)]
            best_model_adj_r2 = models[best_model_name_adj_r2]

            # Displaying which model provides best score in which area
            logging.info(f"Best model for training score is: {best_model_train}")
            logging.info(f"Best model for validation score is: {best_model_val}")
            logging.info(f"Best model for cross validation score is: {best_model_cross_val}")
            logging.info(f"Best model for adjusted r2 score is: {best_model_adj_r2}")

            # Compare the best model for adjusted r2 score is for the threshold
            if best_adj_r2_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and validation dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model_adj_r2
            )
            logging.info("Best Model is now saved in the artifacts")
                        
        except Exception as e:
            raise CustomException(e, sys)
        


    def evaluate_models(self, train_data, target_column, xtrain, ytrain, xval, yval, models, param):
        '''
        evaluate all the models using various metrics'''

        try:
            report_train_score = {}
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
                mean_cross_val_score = np.abs(np.mean(cross_val_score(model, xtrain, ytrain, scoring='neg_mean_absolute_error', cv=3)))

                ytrain_pred = model.predict(xtrain)
                yval_pred = model.predict(xval)
                
                train_model_score = r2_score(ytrain, ytrain_pred)
                val_model_score = r2_score(yval, yval_pred)

                n = xval.shape[0]
                p = xval.shape[1]
                adjusted_val_score = 1-(1-val_model_score)*(n-1)/(n-p-1)

                report_train_score[list(models.keys())[i]] = train_model_score
                report_val_score[list(models.keys())[i]] = val_model_score
                report_cross_val_score[list(models.keys())[i]] = mean_cross_val_score
                report_adjusted_r2_score[list(models.keys())[i]] = adjusted_val_score

            return report_train_score, report_val_score, report_cross_val_score, report_adjusted_r2_score

        except Exception as e:
            raise CustomException(e, sys)
        