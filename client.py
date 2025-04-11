from config import LEARNING_RATE, EPOCHS, MOMENTUM, NUM_CLASSES, BATCH_SIZE
from model import Net
import torch
from torch.utils.data import DataLoader
from utils import (
    save_model,
    save_local_train_history_to_csv,
    save_model,
    to_tensor,
    train,
    test,
    construct_autoencoder
)


class CentralizedLearningClient:
    #initialize a client
    def __init__(self, trainloader, valloader, client_id, fold, feature_count) -> None:
       
        self.trainloader = trainloader
        self.valloader = valloader
        self.fold = fold
        self.feature_count = feature_count
        self.client_id = client_id #savign client ID
        #self.model = Net(input_size=self.feature_count, num_classes=NUM_CLASSES)
        self.model = construct_autoencoder(input_size=feature_count)

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    #main train function
    def fit(self, parameters, config={}):
        
        lr, epochs, momentum = LEARNING_RATE, EPOCHS, MOMENTUM
        
        # Define the optimizer
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)

        # do local training
        results = train(self.model, self.trainloader, self.valloader, optim, epochs=epochs, device=self.device)

        #Save the best model
        save_model(self.client_id, self.feature_count, self.fold, results['model_state'])
        
        #Saving local model train history
        save_local_train_history_to_csv(self.client_id, self.feature_count, self.fold,  results['metrics_history'])        

        return "training complete"
    
    ## Setting the parameters
    ## It is required to save the best model
    def set_parameters(self, parameters):
        """With the model parameters received from the server,
        overwrite the uninitialise model in this class with them."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        # now replace the parameters
        self.model.load_state_dict(state_dict, strict=True)


#Creates a lcient
def create_client(training_set, validation_set, client_id: int, fold, feature_count):

    # Now we apply the transform to each batch.
    trainloader = DataLoader(to_tensor(training_set), batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(to_tensor(validation_set), batch_size=BATCH_SIZE)
    
    # Create and return client
    return CentralizedLearningClient(trainloader, valloader, client_id, fold, feature_count)


if __name__ =="__main__":
    print('client.py')