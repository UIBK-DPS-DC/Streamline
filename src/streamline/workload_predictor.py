import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
import pickle
import logging

# Create a logger instance
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()

        # Create the positional encoding matrix with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Generate the position indices (shape: max_len x 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the scaling factor (div_term) for sine and cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register the positional encoding tensor as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input tensor
        return x + self.pe[:x.size(1), :]


class TimeSeriesTransformer(nn.Module):

    def __init__(self, d_model, nhead, num_layers, input_size, output_size, max_len):
        super().__init__()

        # Define layers and components
        self.input_linear = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, output_size)

    def forward(self, x):
        # Pass through input linear layer and add positional encoding
        x = self.input_linear(x).unsqueeze(1)
        x = self.positional_encoding(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Final output transformation
        return self.output_linear(x[:, -1, :])


class WorkloadPredictor:
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def __init__(self, d_model=128, nhead=4, num_layers=4, input_size=100, output_size=30,
                 epochs=500, batch_size=128, max_len=500, lr=0.0002):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_len = max_len
        self.lr = lr
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def create_dataset(self, data):
        x, y = [], []
        for i in range(len(data) - self.input_size - self.output_size):
            # Create input-output pairs for time series data
            x.append(data[i:i + self.input_size])
            y.append(data[i + self.input_size:i + self.input_size + self.output_size])

        x = np.array(x)
        y = np.array(y)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def train_(self, load):

        # Scale the data using the scaler
        data = self.scaler.fit_transform(load.reshape(-1, 1)).flatten()

        # Create dataset for training
        x, y = self.create_dataset(data)

        # Initialize the model, loss function, and optimizer
        model = TimeSeriesTransformer(self.d_model, self.nhead, self.num_layers, self.input_size, self.output_size, self.max_len)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        best_model = model
        best_loss = 10.0
        for epoch in range(self.epochs + 1):
            model.train()
            epoch_loss = 0

            # Loop over the batches
            for i in range(0, len(x), self.batch_size):
                # Get batch of input and output data
                x_batch = x[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # Zero the gradients from the previous step
                optimizer.zero_grad()

                # Make predictions with the model
                output = model(x_batch)

                # Calculate the loss
                loss = criterion(output, y_batch)

                # Backpropagate the gradients
                loss.backward()

                # Update the model weights
                optimizer.step()

                # Accumulate the loss for the epoch
                epoch_loss += loss.item()

            # Track the best model based on loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = model

            # Log progress every 50 epochs
            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}/{self.epochs}, Loss: {epoch_loss:.4f}")

        self.model = best_model
        return best_model

    def predict(self, input):
        # Convert input to a torch tensor and add a batch dimension
        input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)

        # Make prediction without updating gradients
        with torch.no_grad():
            # Predict and remove the batch dimension
            prediction = self.model(input).squeeze().numpy()

            # Inverse scale the prediction
            return self.scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

    def plot(self, predicted, predict_at, actual, filename):
        # Plot the actual data and predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(actual)), actual, label="Actual Data")
        plt.plot(range(predict_at, predict_at + self.output_size), predicted, label="Prediction", color="red")
        plt.legend()
        plt.title("Time Series Prediction with Transformer")
        plt.xlabel("Timesteps")
        plt.ylabel("Value")
        plt.savefig('./' + filename + '.png')
