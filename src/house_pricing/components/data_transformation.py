import sys
import os
import numpy as np 
import pandas as pd
from scipy.stats import skew
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from src.house_pricing.exception import CustomException
from src.house_pricing.logger import logging
from src.house_pricing.entity import DataTransformationConfig



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def cat_con_df(self, train_data):
        '''
        Separating the categorical and continuous columns from the training dataset'''
        
        cat = []
        con = []

        for i in train_data.columns:
            if (train_data[i].dtypes=='object'):
                cat.append(i)
            else:
                con.append(i)
        con.remove('Id')
        
        return cat, con


    def fill_top_missing_values(self, train_data):
        '''
        Check and fill the top 5 missing columns with '0' in both training and testing dataset'''

        miss1 = (train_data.isna().sum()/train_data.shape[0])*100
        miss1 = pd.DataFrame(miss1, columns=['count'])
        miss1 = miss1.sort_values(by='count', ascending=False)

        logging.info(f"Top 10 missing features from training dataset are: {miss1[:10]}")

        train_miss = (miss1[:6].index).values

        for i in train_miss:
            train_data[i].fillna("0", inplace=True)
        
        logging.info(f"Missing features from training dataset are now replaced with '0'")
            
        return train_data


    def fill_missing_values(self, train_data):
        '''
        Removing the rest of the missing columns from both training and testing dataset'''

        cat, con = self.cat_con_df(train_data)
        train_data = self.fill_top_missing_values(train_data)

        si1 = SimpleImputer(strategy='mean')
        si2 = SimpleImputer(strategy='most_frequent')

        A = pd.DataFrame(si1.fit_transform(train_data[con]), columns=con)
        B = pd.DataFrame(si2.fit_transform(train_data[cat]), columns=cat)

        train_new = A.join(B)
        logging.info(f"Missing features from training dataset are now handled")
        
        return train_new, cat, con

    
    def check_skew(self, train_data):
        '''
        Check the skewness of the training data'''
        
        train_data, cat, con = self.fill_missing_values(train_data)

        con.remove('SalePrice')
        skewed = train_data[con].apply(lambda x: skew(x))
        skewed = skewed[skewed > 0.75]
        skewed = skewed.index

        train_data[skewed] = np.log1p(train_data[skewed])
        con.append('SalePrice')
        logging.info(f"skewness has now been removed from the training data")
        
        return train_data, cat, con

    def scaling(self, train_data):
        
        ss = StandardScaler()
        train_data, cat, con = self.check_skew(train_data)

        con.remove('SalePrice')
        train_data[con] = ss.fit_transform(train_data[con])
        con.append('SalePrice')
        logging.info(f"Feature scaling is now performed")
        
        return train_data, cat, con
    

    def handle_outliers(self, train_data):
        '''
        Removing the outliers from the below columns as they have more number of outliers'''
        train_data, cat, con = self.scaling(train_data)
        val1 = ['BsmtUnfSF', 'TotalBsmtSF', 'KitchenAbvGr', 'ScreenPorch']

        for i in val1:
            Q1 = train_data[i].quantile(0.05)
            Q3 = train_data[i].quantile(0.95)
            IQR = Q3 - Q1
            train_data = train_data[(train_data[i] >= Q1 - 1.5*IQR) & (train_data[i] <= Q3 + 1.5*IQR)]

        logging.info(f"Outliers are now handled in the training data")
            
        return train_data, cat, con
    

    def encode_train_data(self, train_data):
        '''
        perform one hot encoding so as to handle unseen values'''
        train_data, cat, con = self.handle_outliers(train_data)
        le = LabelEncoder()
        for i in cat:
            train_data[i] = le.fit_transform(train_data[i])
            
        logging.info(f"Missing features from training dataset are now replaced with '0'")
        return train_data
    

    def initiate_data_transformation(self):
        '''
        Start the data transformation process'''

        try:
            train_data = pd.read_csv(self.data_transformation_config.train_data_path)
            logging.info("Read training data is completed")

            train_data = self.encode_train_data(train_data)
            logging.info(f"Data pre-processing is completed for training data")
            
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessed_train_data_path), 
                        exist_ok=True)
            logging.info("The directory for the data transformation is now created")

            logging.info(f"Saving preprocessed training data.")
            train_data.to_csv(self.data_transformation_config.preprocessed_train_data_path, 
                              index=False, header=True)
            logging.info(f"Saved preprocessed training data.")

            return self.data_transformation_config.preprocessed_train_data_path
            
        except Exception as e:
            raise CustomException(e,sys)
