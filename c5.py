from dataloader import get_training_datasets_by_client
from client import create_client
from config import FOLD_RANGE, FEATURE_RANGE

#client number
client_id = 5


if __name__ =="__main__":
    ## Collecting Datasets
    for num_feature in FEATURE_RANGE:
        for fold in FOLD_RANGE:
            print(f"Training Started for Feature Number {num_feature} and Fold {fold}")
            training, validation = get_training_datasets_by_client(client_id=client_id, fold=fold, feature_count=num_feature)
            client = create_client(training, validation, client_id, fold=fold, feature_count=num_feature)
            train_status = client.fit(training, validation)
            print(f"Training Completed for Feature Number {num_feature} and Fold {fold}")

