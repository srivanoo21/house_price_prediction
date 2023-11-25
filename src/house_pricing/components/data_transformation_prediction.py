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
from src.house_pricing.utils import save_object


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def cat_con_df(self, test_data):
        '''
        Separating the categorical and continuous columns from the testing dataset'''
        
        cat = []
        con = []

        for i in test_data.columns:
            if (test_data[i].dtypes=='object'):
                cat.append(i)
            else:
                con.append(i)
        con.remove('Id')
        
        return cat, con


    def fill_top_missing_values(self, test_data):
        '''
        Check and fill the top 5 missing columns with '0' in the testing dataset'''

        miss1 = (test_data.isna().sum()/test_data.shape[0])*100
        miss1 = pd.DataFrame(miss1, columns=['count'])
        miss1 = miss1.sort_values(by='count', ascending=False)

        logging.info(f"Top 10 missing features from testing dataset are: {miss1[:10]}")

        test_miss = (miss1[:6].index).values

        for i in test_miss:
            test_data[i].fillna("0", inplace=True)
        
        logging.info(f"Missing features from testing dataset are now replaced with '0'")
            
        return test_data


    def fill_missing_values(self, test_data):
        '''
        Removing the rest of the missing columns from the testing dataset'''

        cat, con = self.cat_con_df(test_data)
        test_data = self.fill_top_missing_values(test_data)

        si1 = SimpleImputer(strategy='mean')
        si2 = SimpleImputer(strategy='most_frequent')

        A = pd.DataFrame(si1.fit_transform(test_data[con]), columns=con)
        B = pd.DataFrame(si2.fit_transform(test_data[cat]), columns=cat)

        test_new = A.join(B)
        logging.info(f"Missing features from testing dataset are now handled")
        
        return test_new, cat, con

    
    def check_skew(self, test_data):
        '''
        Check the skewness of the testing data'''
        
        test_data, cat, con = self.fill_missing_values(test_data)

        skewed = test_data[con].apply(lambda x: skew(x))
        skewed = skewed[skewed > 0.75]
        skewed = skewed.index

        test_data[skewed] = np.log1p(test_data[skewed])
        logging.info(f"skewness has now been removed from the testing data")
        
        return test_data, cat, con
    

    def scaling(self, test_data):
        '''
        perform scaling of the test data'''
        
        ss = StandardScaler()
        test_data, cat, con = self.check_skew(test_data)
        ## SS object need to be in artifacts
        test_data[con] = ss.fit_transform(test_data[con])
        logging.info(f"Feature scaling is now performed")
        
        return test_data, cat, con
    

    def handle_outliers(self, test_data):
        '''
        Removing the outliers from the below columns as they have more number of outliers'''
        test_data, cat, con = self.scaling(test_data)
        val1 = ['BsmtUnfSF', 'TotalBsmtSF', 'KitchenAbvGr', 'ScreenPorch']

        for i in val1:
            Q1 = test_data[i].quantile(0.05)
            Q3 = test_data[i].quantile(0.95)
            IQR = Q3 - Q1
            test_data = test_data[(test_data[i] >= Q1 - 1.5*IQR) & (test_data[i] <= Q3 + 1.5*IQR)]

        logging.info(f"Outliers are now handled in the testing data")
            
        return test_data, cat, con
    

    def encode_test_data(self, test_data):
        '''
        perform one hot encoding so as to handle unseen values'''
        
        test_data, cat, con = self.handle_outliers(test_data)
        le = LabelEncoder()
        for i in cat:
            test_data[i] = le.fit_transform(test_data[i])
            
        logging.info(f"Missing features from testing dataset are now replaced with '0'")
        return test_data
    

    def initiate_data_transformation(self):
        '''
        Start the data transformation process'''

        try:
            test_data = pd.read_csv(self.data_transformation_config.test_data_path)
            logging.info("Read testing data is completed")

            test_data = self.encode_test_data(test_data)
            logging.info(f"Data pre-processing is completed for testing data")
            
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessed_test_data_path), 
                        exist_ok=True)
            logging.info("The directory for the data transformation is now created")

            logging.info(f"Saving preprocessed testing data.")

            save_object(
                file_path = self.data_transformation_config.preprocessed_test_data_path,
                obj = test_data
            )
            logging.info(f"Saved preprocessed testing data.")

            return self.data_transformation_config.preprocessed_test_data_path
            
        except Exception as e:
            raise CustomException(e,sys)
