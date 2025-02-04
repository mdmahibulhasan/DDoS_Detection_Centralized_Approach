from config import MODEL_PATH, NUM_CLASSES, BATCH_SIZE, FEATURE_TYPE
from config import (
    EARLY_STOP_PATIENCE_ON_EPOCH, 
    LR_ADJUSTMENT_FACTOR, 
    MIN_LR,
    METRIC_PATH 
)
from model import Net
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import csv
import timeit


## primary functions is to train the model
def train(net, 
    trainloader,
    testloader,
    optimizer, 
    epochs, 
    device: str,
    early_stop_patience=EARLY_STOP_PATIENCE_ON_EPOCH,
    factor=LR_ADJUSTMENT_FACTOR,  # Factor by which the LR will be reduced
    min_lr=MIN_LR,  # Minimum LR after reduction
    lr_patience=3
    ):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()

    # Early stopping variables
    best_val_accuracy = 0
    epochs_without_improvement = 0
    best_model_state = None  # To save the best model

    # Store metrics for each epoch
    metrics_history = {
        "epoch": [],
        "training_loss": [],
        "training_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],  # To track learning rate for each epoch
        "training_time": [] #track the trainign time
    }

    # ReduceLROnPlateau Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=lr_patience, factor=factor, min_lr=min_lr)

    for epoch in range(epochs):
        #start the timer
        start_time = timeit.default_timer()

        # Training loop
        net.train()
        correct_train, total_train, train_loss = 0, 0, 0.0
        for iteration, batch in enumerate(trainloader):
            features, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate training metrics
            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        #end time
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        metrics_history["training_time"].append( elapsed) #tracking the time

        # Calculate epoch-wise training loss and accuracy
        epoch_train_loss = train_loss / total_train
        epoch_train_accuracy = correct_train / total_train
        metrics_history["training_loss"].append(epoch_train_loss)
        metrics_history["training_accuracy"].append(epoch_train_accuracy)

        # Validation loop
        val_loss, val_accuracy = test(net, testloader, device)
        metrics_history["validation_loss"].append(val_loss)
        metrics_history["validation_accuracy"].append(val_accuracy)

        # Track current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        metrics_history["learning_rate"].append(current_lr)

        #assigning current epoch
        metrics_history['epoch'].append(epoch + 1)

        #print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {epoch_train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Learning Rate: {current_lr:.6f}")

        # Update the scheduler with the validation accuracy
        scheduler.step(val_accuracy)

        # Early stopping condition
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            best_model_state = net.state_dict()  # Save the best model state
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stop_patience: #early stop patience should be 2xpatience
            #print(f"Early Stopping Triggered at Epoch {epoch + 1}.")
            break

    # Return the best model state and metrics history
    return {
        "model_state": best_model_state if best_model_state is not None else net.state_dict(),
        "metrics_history": metrics_history,
    }


#Evaluating the Model
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            features, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


##clear the cache of the Cuda
def clear_cuda_cache():
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")


#Convert data into tensfor GPU
def to_tensor(df):
    # Separate features and labels
    X = df.drop(columns="Label").values  # Replace "label" with the actual label column name
    y = df["Label"].values
    # Convert to PyTorch tensors
    # Also reshaping it 5 by 6
    #X_tensor = torch.tensor(X.reshape(-1, 1, 5, 6), dtype=torch.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)


## Prepare File Path from Features and Folds
def prepare_file_path(path, client_id, feature_count=None, fold=None):
    if feature_count == None and fold == None:
        file_path = path.format(client_id)
    else:
        file_path = path.format(client_id, feature_count, fold)
    # Extract the directory path
    directory = os.path.dirname(file_path)
    # Check if the directory exists
    if not os.path.exists(directory):
        # If the directory does not exist, create it
        os.makedirs(directory)
    return file_path


## Save the mode based on parameters
def save_model(client_id, feature_count, fold, model_state, file_path = MODEL_PATH):

    try:
        model = Net(input_size=feature_count, num_classes=NUM_CLASSES)
        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)  # send model to device

        # set parameters to the model
        #params_dict = zip(model.state_dict().keys(), parameters)
        #state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(model_state, strict=True)
        torch.save(model.state_dict(), prepare_file_path(file_path, client_id, feature_count, fold))
    except Exception as e:
        print(f"Saving model error: {e}")


## This function will save client wise local metrics
def save_local_train_history_to_csv(client_id, feature_count, fold, metrics, file_path=METRIC_PATH):

    # Initialize CSV headers and rows
    headers = ["Client", "Feature Count", "Fold", "Epoch", "Learing_Rate", "Train Accuracy", "Train Loss", "Validation Accuracy", "Validation Loss", "Training Time (S)"]
    rows = []

    # Check if the file exists
    if os.path.exists(prepare_file_path(file_path, client_id)):
        mode = "a"  # Append mode
    else:
        mode = "w"  # Write mode

    with open(prepare_file_path(file_path, client_id), mode=mode, newline="") as file:
        writer = csv.writer(file)
        if mode == "w":
            writer.writerow(headers)

        for i, epoch in enumerate(metrics['epoch']):
            training_accuracy = metrics['training_accuracy'][i] if metrics['training_accuracy'][i] > 0 else 0
            validation_accuracy = metrics['validation_accuracy'][i] if metrics['validation_accuracy'][i] > 0 else 0
            training_loss = metrics['training_loss'][i] if metrics['training_loss'][i] > 0 else 0
            validation_loss = metrics['validation_loss'][i] if metrics['validation_loss'][i] > 0 else 0
            learning_rate = metrics['learning_rate'][i] if metrics['learning_rate'][i] > 0 else 0
            training_time = metrics['training_time'][i] if metrics['training_time'][i] > 0 else 0

            #Writing the row
            writer.writerow([client_id, feature_count, fold, epoch, learning_rate, training_accuracy, training_loss, validation_accuracy, validation_loss, training_time])

    #print(f"Local Training Metric Saved: {prepare_file_path(file_path)}")

