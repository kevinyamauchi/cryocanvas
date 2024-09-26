import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class TorchMLPClassifier:
    """
    A simple MLP classifier that mimics the scikit-learn API using PyTorch.
    Automatically splits the dataset into training and validation sets if required.
    """

    def __init__(
        self,
        input_dim,
        hidden_layers=(100,),
        output_dim=3,
        lr=0.001,
        batch_size=64,
        epochs=100,
        device="cpu",
        val_split=0.2,
    ):

        # Create the model
        self.model = nn.Sequential()
        layer_sizes = [input_dim] + list(hidden_layers) + [output_dim]
        for i in range(len(layer_sizes) - 1):
            self.model.add_module(
                f"fc{i}", nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )
            if i < len(layer_sizes) - 2:  # Don't add ReLU after the last layer
                self.model.add_module(f"relu{i}", nn.ReLU())

        # Define the loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.val_split = val_split

    def fit(self, X, y, val_data=None):
        """
        Fit the model to the data. Splits the data into training and validation
        sets if a validation split is specified.

        Parameters
        ----------
        X : np.ndarray
            The input features for training.
        y : np.ndarray
            The target values for training.
        val_data : tuple, optional
            A tuple (X_val, y_val) of validation data. If provided, the val_split
            parameter is ignored, and this data is used as the validation set.
        """
        # Automatically split the training data into train and validation sets if val_split is specified
        if val_data is None and self.val_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.val_split, random_state=42
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = val_data if val_data else (None, None)

        # Create DataLoader objects for training and validation
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float),
            torch.tensor(y_train, dtype=torch.long),
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )

        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float),
                torch.tensor(y_val, dtype=torch.long),
            )
            val_loader = DataLoader(
                dataset=val_dataset, batch_size=self.batch_size
            )

        # Training phase
        self.model.train()
        for epoch in range(self.epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Validation phase
            if X_val is not None and y_val is not None:
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(
                            self.device
                        )
                        outputs = self.model(inputs)
                        val_loss += self.criterion(outputs, labels).item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                val_loss /= len(val_loader)
                val_accuracy = correct / total
                print(
                    f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
                )
                self.model.train()

    def predict(self, X):
        self.model.eval()
        inputs = torch.tensor(X, dtype=torch.float).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs.argmax(dim=1).cpu().numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


if __name__ == "__main__":
    # Example usage
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 3, 1000)
    clf = TorchMLPClassifier(
        input_dim=10,
        hidden_layers=(100, 50),
        output_dim=3,
        lr=0.001,
        batch_size=64,
        epochs=1000,
        device="cuda",
        val_split=0.2,
    )
    clf.fit(X, y)
    print(clf.score(X, y))
    y_pred = clf.predict(X)
    print(y_pred)
