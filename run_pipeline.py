from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == '__main__':
    data_path = '/Users/anthonymoubarak/Desktop/GitHub_Portfolio/MLOPS/data/olist_customers_dataset.csv'
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path)
 



#file:/Users/anthonymoubarak/Library/Application Support/zenml/local_stores/10ae6e28-0367-4c1b-a005-e89bef753999/mlruns
