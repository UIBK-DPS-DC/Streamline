import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import logging

# Create a logger instance
logger = logging.getLogger(__name__)


class NeuralNetwork:

    def __init__(self, data, batch_size=32, lr=0.001, epochs=200, seed=42):
        self.epochs = epochs

        # Fix the seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Remove unnecessary columns from the dataset
        data = data.drop(columns=['Vertex Id', 'Busytime', 'TP_out'])

        # Encode the operator type as numerical values
        self.label_encoder = LabelEncoder()
        data['Operator Type'] = self.label_encoder.fit_transform(data['Operator Type'])
        o_types = data['Operator Type'].unique()

        self.scaler_X = {}
        self.scaler_y = {}
        X = torch.empty((0, 5))
        y = torch.empty((0, 2))

        # Process data separately for each operator type
        for o_type in o_types:
            filtered = data[data["Operator Type"] == o_type]

            # Extract features (X) and target variables (y)
            X_f = filtered[['TP_in', 'Pred Parallelism', 'Parallelism', 'Segment Size', 'Operator Type']]
            y_f = filtered[['Latency', 'CPU']]
            X_f = torch.tensor(X_f[X_f.select_dtypes(include=[np.number]).columns].values, dtype=torch.float32)
            y_f = torch.tensor(y_f[y_f.select_dtypes(include=[np.number]).columns].values, dtype=torch.float32)

            # Min-max normalization for feature scaling
            max_value_X = X_f.max(dim=0).values
            min_value_X = X_f.min(dim=0).values
            max_value_X[4] = len(o_types)

            # Ensure max > min to avoid division by zero
            for i in range(4):
                if max_value_X[i] == min_value_X[i]:
                    max_value_X[i] = min_value_X[i] + 1

            self.scaler_X[o_type] = nn.Parameter(torch.stack([min_value_X, max_value_X - min_value_X]), requires_grad=False)

            # Scale target variables
            max_value_y = y_f.max(dim=0).values
            min_value_y = y_f.min(dim=0).values
            if max_value_y[0] == min_value_y[0]:
                max_value_y[0] = min_value_y[0] + 1
            self.scaler_y[o_type] = nn.Parameter(torch.stack([min_value_y, max_value_y - min_value_y]), requires_grad=False)

            # Apply scaling
            X_f = self.scale_data(X_f, self.scaler_X[o_type])
            y_f = self.scale_data(y_f, self.scaler_y[o_type])

            # Merge the processed data
            X = torch.cat((X, X_f), dim=0)
            y = torch.cat((y, y_f), dim=0)

        # Shuffle the dataset randomly
        indices = torch.randperm(X.shape[0])
        X = X[indices]
        y = y[indices]

        # Split into training and testing sets (80/20 split)
        train_size = int(0.8 * X.shape[0])
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Create DataLoaders for training and testing
        self.train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        # Define the neural network, loss function, and optimizer
        self.model = self.build_model(input_size=X.shape[1], output_size=y.shape[1])
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.data = data

    def build_model(self, input_size, output_size):
        # Defines the neural network architecture.
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.ReLU()
        )

    def scale_data(self, data, scaler):
        # Applies min-max normalization
        min_vals, ranges = scaler
        return (data - min_vals) / ranges

    def descale_data(self, data, scaler):
        # Reverts normalized data back to original scale
        min_vals, ranges = scaler
        return data * ranges + min_vals

    def train_model(self):
        val_loader = self.test_loader
        for epoch in range(self.epochs + 1):
            self.model.train()
            train_loss = 0

            # Iterate through training batches
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(self.train_loader)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        predictions = self.model(X_batch)
                        loss = self.criterion(predictions, y_batch)
                        val_loss += loss.item()
                val_loss = val_loss / len(val_loader)
                if (epoch) % 50 == 0:
                    logger.info(f"Epoch {epoch}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if (epoch) % 50 == 0:
                    logger.info(f"Epoch {epoch}/{self.epochs}, Train Loss: {train_loss:.4f}")

        # Compute R2 score for evaluation
        r2_score_train = self.get_score(self.train_loader)
        r2_score_test = self.get_score(self.test_loader)
        logger.info(f"R^2 Score: Train={r2_score_train:.2f}, Val={r2_score_test:.2f}")

    def get_score(self, data_loader):
        # Set the model to evaluation mode
        self.model.eval()
        total_variance = 0
        explained_variance = 0

        # Disable gradient calculations for efficiency
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                # Get model predictions
                predictions = self.model(X_batch)

                # Compute total variance (sum of squared differences from the mean)
                total_variance += torch.sum((y_batch - y_batch.mean(dim=0)) ** 2)

                # Compute explained variance (sum of squared errors between predictions and actual values)
                explained_variance += torch.sum((y_batch - predictions) ** 2)

        # Compute R2 score
        r2_score = 1 - (explained_variance / total_variance)
        return r2_score.item()

    def predict(self, data):
        # Set the model to evaluation mode
        self.model.eval()

        # Convert operator type from string/category to numerical encoding
        data[0][-1] = self.label_encoder.transform([data[0][-1]])[0]

        # Scale input features based on precomputed min-max values
        new_data = self.scale_data(torch.tensor(data, dtype=torch.float32), self.scaler_X[data[0][-1]])

        # Predict output using the trained model
        with torch.no_grad():
            predicted_scaled = self.model(new_data)

        # Convert predicted values back to the original scale
        predicted_values = self.descale_data(predicted_scaled, self.scaler_y[data[0][-1]])

        return predicted_values.numpy()
