# PYTORCH.md

This document provides **high-level instructions** for **porting** the FlightFactor methodology from the original **C-based feed-forward neural network** to a **PyTorch** (Python) implementation. PyTorch allows **GPU acceleration**, making training and inference significantly faster for large datasets. Below, we outline the key steps and best practices for replicating the airline load-factor and profitability forecasting pipeline in Python with PyTorch.

---

## 1. Why Migrate to PyTorch?

1. **GPU Acceleration**: PyTorch natively supports running tensors on GPUs (NVIDIA CUDA, ROCm), leading to major speedups for large-scale training.
2. **High-Level API**: Many utilities (e.g., `DataLoader`, `Dataset`, `nn.Module`, etc.) reduce boilerplate compared to writing forward/backprop in pure C.
3. **Extensive Ecosystem**: PyTorch integrates with the wider Python data science ecosystem (NumPy, pandas, scikit-learn, etc.).  
4. **Rapid Prototyping**: Python’s interactive REPL or Jupyter notebooks make experimentation and debugging faster.

---

## 2. Architectural Overview

### 2.1 Data and Inputs

In **C**:
- We read `nn_historical.csv` into arrays (`X`, `Y`), compute mean/std, then train using raw pointers.

In **PyTorch**:
- Typically, you’d use **pandas** (`pd.read_csv`) or **PyTorch’s Dataset** abstractions to load data.
- You can still compute mean/std to normalize data manually, or rely on `sklearn.preprocessing.StandardScaler`.

### 2.2 Neural Network Model

In **C**:
- We manually define `W1`, `b1`, `W2`, `b2`, then code the forward pass and backprop.

In **PyTorch**:
- You create a class that inherits from `nn.Module`.  
- Define layers (e.g., `nn.Linear`) and ReLU activation.  
- Forward pass is defined in the `forward()` method, and PyTorch handles backprop automatically.

Example (high-level):

```python
import torch
import torch.nn as nn

class FlightNet(nn.Module):
    def __init__(self, input_size=13, hidden_size=8, output_size=2):
        super(FlightNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: [batch_size, input_size]
        h = self.relu(self.hidden(x))
        y = self.out(h)  # shape: [batch_size, output_size]
        return y
```

### 2.3 Training Loop

In **C**:
- We do a manual loop over all samples, compute MSE, do partial derivatives, etc.

In **PyTorch**:
- Use an `nn.MSELoss` or custom multi-output MSE.  
- An optimizer like `torch.optim.Adam` or `SGD`.  
- A typical training loop:

```python
import torch.optim as optim

model = FlightNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    # Optional: shuffle data or use a DataLoader
    optimizer.zero_grad()
    outputs = model(x_train)  # forward pass
    loss = criterion(outputs, y_train)
    loss.backward()           # backprop
    optimizer.step()          # update weights

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss={loss.item():.6f}")
```

**Key difference**: PyTorch handles backprop once you call `loss.backward()`, so you no longer manually do partial derivatives as in the C code.

### 2.4 Multi-Output MSE

Because we predict **two outputs** (Load Factor, Profit), the default `nn.MSELoss()` can handle a 2-dimensional output. Just ensure your `Y` is shaped `[num_samples, 2]`, matching `[batch_size, output_size]`.

---

## 3. Steps to Migrate

1. **Data Loading**  
   - Use pandas to read `nn_historical.csv`.  
   - Convert relevant columns to `X` (13 features) and `Y` (2 targets).  
   - Potentially use `torch.utils.data.TensorDataset` + `DataLoader` for mini-batch training.

2. **Normalization**  
   - If you want to replicate exactly the C logic, compute means and std devs for each feature.  
   - Use either:
     - `sklearn.preprocessing.StandardScaler` (common approach in Python), or
     - custom code with `pandas` or `NumPy`.
   - Save these statistics to transform future data consistently.

3. **Model Definition**  
   - Create a PyTorch `nn.Module` that replicates the single hidden layer (8 ReLU neurons) and 2 linear outputs.

4. **Loss & Optimizer**  
   - Typically `nn.MSELoss` is fine.  
   - `Adam` or `SGD` can be used for parameter updates.  
   - GPU usage: move model and data to CUDA with `.cuda()` (or `.to(device)`) calls.

5. **Training**  
   - If you want GPU acceleration, do something like:
     ```python
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     model.to(device)
     X_train = X_train.to(device)
     Y_train = Y_train.to(device)
     ```
   - Then do your forward/backward pass. PyTorch automatically runs on the GPU if the tensors are on `device='cuda'`.

6. **Inference & Menu**  
   - If you want an interactive approach (like the C menu), you can write a Python script that:
     1. Loads the trained model (saved with `torch.save` / `torch.load`).  
     2. Presents a console-based menu (e.g., via `input()` prompts).  
     3. Reads user data, normalizes it, does `model.forward` to predict, then prints the results.

---

## 4. GPU Acceleration

### 4.1 Basic CUDA Workflow

- PyTorch detects a GPU if installed with CUDA support.  
- You explicitly move model and tensors to GPU. For example:
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = FlightNet().to(device)
  x_train_tensor = x_train_tensor.to(device)
  y_train_tensor = y_train_tensor.to(device)
  ```

### 4.2 Performance Gains

- Large or frequent matrix multiplications in the forward/backprop pass run on the GPU, often resulting in order-of-magnitude speedups for big data.  
- For very small datasets, GPU overhead might not be beneficial. Typically, airlines have enough flight data for GPU gains to matter.

### 4.3 Multi-GPU

- PyTorch also supports multi-GPU training via `DataParallel` or `DistributedDataParallel`. This is more advanced but can further reduce training time if you have multiple GPUs.

---

## 5. Practical Example Outline

Below is a **condensed** example showing the typical flow in PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Load Data (e.g., nn_historical.csv)
df = pd.read_csv('nn_historical.csv')

# For demonstration, let's assume these columns:
feature_cols = [
    "Seats","Booked_Pax","Connecting_Pax","Local_Pax","Average_Fare",
    "Fuel_Cost_per_leg","Crew_Cost_per_leg","Landing_Fee","Overhead_Cost",
    "Distance","Flight_Time","Competitor_Capacity","Competitor_Fare"
]
X = df[feature_cols].values   # shape: [N, 13]

# Observed load factor & profit
df['LoadFactor'] = df['Booked_Pax'] / df['Seats']
df['Profit'] = (df['Booked_Pax'] * df['Average_Fare']) - (
    df['Fuel_Cost_per_leg'] + df['Crew_Cost_per_leg'] + df['Landing_Fee'] + df['Overhead_Cost']
)
Y = df[['LoadFactor','Profit']].values  # shape: [N, 2]

# 2. Normalize
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X)
Y_scaled = y_scaler.fit_transform(Y)

# 3. Convert to torch Tensors
X_torch = torch.tensor(X_scaled, dtype=torch.float32)
Y_torch = torch.tensor(Y_scaled, dtype=torch.float32)

# 4. Define Model
class FlightNet(nn.Module):
    def __init__(self, input_size=13, hidden_size=8, output_size=2):
        super(FlightNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h = self.relu(self.hidden(x))
        return self.out(h)

model = FlightNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Optionally run on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
X_torch = X_torch.to(device)
Y_torch = Y_torch.to(device)

# 5. Training
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_torch)
    loss = criterion(outputs, Y_torch)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss={loss.item():.6f}")

# 6. Save Model
torch.save(model.state_dict(), "flightfactor_model.pt")

# Save scalers for future inference
import joblib
joblib.dump(x_scaler, "x_scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")
```

To **infer** on new data or write a **menu** tool in Python, you’d load `flightfactor_model.pt` into a `FlightNet` instance, plus reload the scalers. Then:

1. Normalize the new flight inputs with `x_scaler.transform(...)`.  
2. Pass them into `model(...)`.  
3. Denormalize the outputs with `y_scaler.inverse_transform(...)`.

---

## 6. Tips & Best Practices

1. **Minibatch Training**  
   - Instead of a single pass over all samples, use PyTorch `DataLoader` with batches to improve scalability.  
2. **Validation Split**  
   - Keep some data aside to check for overfitting.  
3. **Hyperparameter Tuning**  
   - Number of epochs, hidden layer size, learning rate can all significantly impact performance.  
4. **Reproducibility**  
   - For consistent results, set random seeds in Python, NumPy, PyTorch, and ensure your GPU environment is reproducible.  
5. **Logging & Visualization**  
   - Libraries like `tensorboardX` or `matplotlib` can track losses over time, making debug & analysis easier.

---

## 7. Conclusion

Migrating the FlightFactor methodology from C to **PyTorch** involves:

1. Adopting **Python** data loading (pandas, scikit-learn).  
2. Building a **`nn.Module`** to replicate the feed-forward structure.  
3. Letting PyTorch handle **backprop** via `loss.backward()` and `optimizer.step()`.  
4. Optionally **accelerating** training on the GPU for large datasets.  

This approach retains all the core logic (load factor + profit as outputs) while significantly **reducing code complexity** and enabling **faster** or **more flexible** model development. For large-scale airline forecasting, PyTorch’s deep learning ecosystem opens the door to advanced neural architectures, distributed training, and integration with data science workflows in Python.