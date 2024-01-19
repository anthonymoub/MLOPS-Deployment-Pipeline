import logging
from typing import Tuple
import pandas as pd
from zenml import step

@step
def train_model(df: pd.DataFrame) -> None:
    pass