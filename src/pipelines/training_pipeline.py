import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

#to trigger data ingestion
if __name__=='__main__':

    #obj initialization triggers the train, test and raw csv's file from data_ingestion subfile
    obj=DataIngestion()

    #return the train and test data path from ingestion
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)

    #initialize the data transformation and create the train and test files
    data_transformation =  DataTransformation()
    train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)

    #initialize the model trainer and create the data 
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)