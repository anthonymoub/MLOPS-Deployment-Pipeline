import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union
from sklearn.metrics import mean_squared_error, r2_score

class Evluation(ABC):
    '''Abstract class defining strategy for evaluating models'''

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        '''Calculates scores based on predicted vs actual values'''
        pass

class MSE(Evluation):
    '''Evaluation strategy that uses MSE'''

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true , y_pred)
            logging.info('MSE: {}'.format(mse))
            return mse
        except Exception as e:
            logging.info('Error in calculating MSE: {}'.format(e))
            raise e
        
class R2(Evluation):
    '''Evaluation strategy that uses R2 score'''
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 score")
            r2 = mean_squared_error(y_true , y_pred)
            logging.info('R2 score: {}'.format(r2))
            return r2
        except Exception as e:
            logging.info('Error in calculating MSE: {}'.format(e))
            raise e
        
class RMSE(Evluation):

    '''Evaluation strategy that uses RMSE score'''
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 score")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info('R2 score: {}'.format(rmse))
            return rmse
        except Exception as e:
            logging.info('Error in calculating MSE: {}'.format(e))
            raise e
        
