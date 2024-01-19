import logging
import pandas as pd
from zenml import step


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self , data_path: str) -> None:
        """Initialize the data ingestion class."""
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        return df

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Args:
        data_path (str): Path to the data file.
    Returns:
        df (pd.DataFrame): Ingested data as a DataFrame.
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e

