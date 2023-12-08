import numpy as np
import pandas as pd
import torch
import math
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Huber Loss
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, inputs, targets):
        loss = torch.where(torch.abs(inputs - targets) < self.delta,
                           0.5 * (inputs - targets) ** 2,
                           self.delta * (torch.abs(inputs - targets) - 0.5 * self.delta))
        return loss.mean()


# Load data from the Excel file
data_df = pd.read_excel("rainfall_data.xlsx")


# Replace NaN values with 0
data_df.fillna(0, inplace=True)

rainfall_data = data_df.iloc[:, 4:].values
river_flow_data = data_df["Data"].values

# Define short and long window sizes for DTSW
short_window = 5
long_window = 30
total_window = short_window + long_window

# Create input and target datasets based on the DTSW method
dts_input_data = []
dts_target_data = []

# Iterate through the data to create sequences based on DTSW method
for i in range(len(rainfall_data) - total_window - 6):
    short_term_data = rainfall_data[i+long_window-short_window:i+long_window]
    long_term_data = rainfall_data[i:i+long_window]
    combined_data = np.vstack((long_term_data, short_term_data))
    dts_input_data.append(combined_data)
    future_7_days_flow = river_flow_data[i + total_window: i + total_window + 7]
    dts_target_data.append(future_7_days_flow)

dts_input_data = np.array(dts_input_data)
dts_target_data = np.array(dts_target_data)

# Standardize the input data
scaler_input = StandardScaler()
dts_input_data = scaler_input.fit_transform(dts_input_data.reshape(-1, total_window * 63)).reshape(dts_input_data.shape)

# Standardize the target data
scaler_target = StandardScaler()
dts_target_data = scaler_target.fit_transform(dts_target_data)

# Split the data into training and testing sets
input_train, input_test, target_train, target_test = train_test_split(dts_input_data, dts_target_data, test_size=0.2, random_state=42)
# Convert the training data and the target data to PyTorch tensors
inputs_train = torch.tensor(input_train).float()
targets_train = torch.tensor(target_train).float()

# Define the Transformer model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Adjusted positional encoding frequency
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / (d_model * 0.5)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, nhid, nlayers, dropout=0.3):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = TransformerEncoderLayer(input_dim, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(2205, nhid)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.decoder = nn.Linear(nhid, 7)  # Modify output dimension to 7 for 7-day prediction
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.input_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output.squeeze()

# Convert the input data into the format (sequence_length, batch_size, input_dim)
inputs_train = inputs_train.view(inputs_train.shape[0], 1, 2205)

# Create the Transformer model and move it to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(64, 1, 4, 64, 5).to(device)

# Define the training loop
def train(model, inputs, targets, epochs):
    criterion = HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00048)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Create a dataset and a data loader for the training data
    dataset_train = TensorDataset(inputs, targets)
    data_loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(data_loader_train):
            # Move inputs and targets to the GPU if available
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)  # Do not use squeeze() here
            
            loss = criterion(outputs, targets)  # Both outputs and targets should be of shape (batch_size, 7)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, epochs, i+1, len(data_loader_train), loss.item()))

# Train the Transformer model for a single epoch to check if it runs without errors

# Calculate weights for the target training data to emphasize prediction accuracy on the top 20% values
sorted_indices = np.argsort(target_train, axis=0)
top_20_percent_idx = int(0.2 * len(target_train))

# Initialize weights with 1
weights = np.ones_like(target_train)

# Assign higher weight (e.g., 2) to the top 20% values
weights[sorted_indices[-top_20_percent_idx:]] = 2

# Convert weights to PyTorch tensor
weights_train = torch.tensor(weights).float()

# If using CUDA, move weights to GPU
if torch.cuda.is_available():
    weights_train = weights_train.cuda()
train(model, inputs_train, targets_train, epochs=500)

test_size = int(0.2 * len(dts_input_data))
train_data, test_data = dts_input_data[:-test_size], dts_input_data[-test_size:]
train_targets, test_targets = dts_target_data[:-test_size], dts_target_data[-test_size:]

# Convert the testing data to PyTorch tensors and move them to the GPU if available
inputs_test = torch.tensor(test_data).float()
inputs_test = inputs_test.view(inputs_test.shape[0], 1, total_window * 63)
targets_test = torch.tensor(test_targets).float()

if torch.cuda.is_available():
    inputs_test = inputs_test.cuda()
    targets_test = targets_test.cuda()

# Use the Transformer model to make predictions on the testing data
model.eval()
with torch.no_grad():
    predictions = model(inputs_test).cpu().numpy()

# Inverse transform the actual values and the predicted values
actual = scaler_target.inverse_transform(targets_test.cpu().numpy())  # No need to reshape here
predicted = scaler_target.inverse_transform(predictions)  # No need to reshape here either

# Save the actual values and the predicted values to a .xlsx file
result = pd.DataFrame(np.hstack([actual, predicted]))
result.columns = ["Actual_Day_{}".format(i) for i in range(1, 8)] + ["Predicted_Day_{}".format(i) for i in range(1, 8)]
result.to_excel("DTSW_result_transformer_7days.xlsx", index=False)
