import logging
from typing import Tuple
import pandas as pd
from zenml import step
from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

@step
def evaluate_model(model: RegressorMixin, 
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> Tuple[
                       Annotated[float, 'r2_score'],
                       Annotated[float , 'rmse'],
                   ]:
    try:
        prediction = model.predict(X_test)
        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        #mlflow.log_metric("mse", mse)

        # Using the R2Score class for R2 score calculation
        r2_class = R2()
        r2_score = r2_class.calculate_score(y_test, prediction)
        #mlflow.log_metric("r2_score", r2_score)

        # Using the RMSE class for root mean squared error calculation
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        #mlflow.log_metric("rmse", rmse)

        return r2_score , rmse
    except Exception as e:
        logging.error(e)
        raise e

