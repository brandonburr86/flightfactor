# METHODOLOGY

This document outlines a **detailed breakdown** of how the FlightFactor project uses **feed-forward neural networks** (FNNs) to address airline load-factor and profitability prediction. It is geared toward a **PhD-level expert in Airline Operations** with a strong background in **mathematics** and **software development**.

---

## 1. Introduction

Accurate forecasting of load factor and profitability is key to **airline network planning** and **revenue management**. Traditional methods (regression, time-series, discrete choice) can be effective, but may not fully capture the **nonlinear interactions** present in large-scale data. Here, we adopt a **feed-forward neural network** approach that directly learns from features like seats, fares, costs, and competitor data to output **two main targets**:

1. **Load Factor**: Booked_Pax / Seats  
2. **Profit**: (Booked_Pax × Average_Fare) – (Fuel + Crew + Landing + Overhead)

---

## 2. Data & Variables

### 2.1 Input Features

We use 13 primary inputs for each flight record:

1. Seats (flight capacity)  
2. Booked Pax (total passengers booked)  
3. Connecting Pax (subset of passengers who arrived from or continue to another flight)  
4. Local Pax (Booked Pax – Connecting Pax)  
5. Average Fare (mean fare across passengers)  
6. Fuel Cost (per leg)  
7. Crew Cost (per leg)  
8. Landing Fee  
9. Overhead Cost  
10. Distance  
11. Flight Time  
12. Competitor Capacity  
13. Competitor Fare

### 2.2 Target Outputs

1. **Load Factor** = Booked_Pax / Seats  
2. **Profit** = (Booked_Pax × Average_Fare) – (Fuel_Cost + Crew_Cost + Landing_Fee + Overhead_Cost)

The model learns a **mapping** from these 13 inputs to these 2 outputs.

---

## 3. Neural Network Architecture

We use a **feed-forward neural network** with a single hidden layer:

- **Input Layer**: 13 nodes (one per input feature).  
- **Hidden Layer**: 8 neurons with ReLU activation.  
- **Output Layer**: 2 neurons (predicting Load Factor and Profit as continuous numeric values).

This design allows the network to capture **nonlinear patterns** and produce **two outputs simultaneously**.

---

## 4. Training Methodology

### 4.1 Normalization

Because inputs vary widely in scale (e.g., seats vs. cost in thousands), we **normalize** each feature by subtracting its mean and dividing by its standard deviation. Similarly, the two outputs (Load Factor and Profit) are normalized as well. This helps stabilize the training process, avoiding large swings in gradients.

### 4.2 Loss Function (Mean Squared Error)

We use a mean squared error (MSE) across both outputs:

- If the outputs are labeled as (LoadFactor, Profit) and the network’s predictions are (LF_pred, Profit_pred), we compute the squared error for each output and sum or average them.  
- Minimizing this MSE guides the network to match **both** load factor and profit observations in the training data.

### 4.3 Backpropagation

The process involves:

1. **Forward Pass**: Given a normalized input vector, compute hidden-layer activations and final outputs.  
2. **Error Calculation**: Compare predicted outputs to actual values in the training set.  
3. **Backward Pass**: Calculate gradients of the loss with respect to each weight and bias.  
4. **Weight Update**: Subtract a fraction (learning rate) of the gradient from each parameter.

Repeating this for multiple epochs (e.g., 100) across the training set allows the network to gradually improve its predictions.

### 4.4 Data Splitting (Recommended)

Although the sample code may train on the entire dataset for demonstration, best practices involve splitting data into training, validation, and test sets to evaluate **generalization**. This ensures the model’s performance is consistent on unseen data.

---

## 5. Implementation Details

1. **Data Generation**:  
   - `ff_generate_data.c` creates synthetic flight data (nn_historical.csv).  
   - Rows include variables such as Seats, Booked_Pax, Fuel_Cost, Overhead_Cost, Competitor_Fare, etc.

2. **Training**:  
   - `ff_train.c` reads nn_historical.csv, computes (Load Factor, Profit) for each row, normalizes features and targets, then trains the network via feed-forward and backprop.  
   - The trained model weights and normalization parameters are saved to a file (nn_weights.bin).

3. **Interactive Menu**:  
   - `flightfactor.c` can load the trained weights (or train on the spot, depending on the variant) and provides options for listing flight codes, predicting average load/profit per code, outputting a CSV of all predictions, or predicting for a new/future flight scenario.

---

## 6. Airline Operations Context

### 6.1 Nonlinearities in Demand & Costs

Airline demand can be strongly affected by factors like competitor fare, route distance, time-of-day, and macroeconomic shifts. Costs similarly have dynamic components (fuel price fluctuations, overhead allocation). A feed-forward NN naturally models **nonlinear** and **interactive** effects.

### 6.2 Multi-Output Benefit

Predicting **Load Factor** and **Profit** together captures **interdependencies** (e.g., a flight might have high load factor but poor profit if fares or costs are unfavorable). A single model that understands these trade-offs can be **more robust** than running separate regressions.

### 6.3 Large-Scale Feasibility

Modern computing (potentially with GPUs) can handle thousands or millions of training samples from an airline’s historical database. Once trained, the model can quickly **infer** new flight scenarios, assisting **schedule optimization** or **revenue management**.

---

## 7. Key Takeaways

1. **End-to-End Training**: The project demonstrates how raw flight-level data (fares, costs, seats) can feed directly into a neural network to predict essential metrics.  
2. **Jointly Predicting Load Factor & Profit**: Offers a more holistic view of route performance than a single-metric model.  
3. **Extensibility**: The approach can integrate additional features (e.g., weather disruptions, loyalty data) or deeper architectures for improved accuracy.  
4. **Practical Implementation**: The code includes examples of data generation, a training script, and a user-facing menu tool, illustrating a complete pipeline from data to inference.

---

## 8. Conclusion

The **feed-forward neural network** approach in FlightFactor effectively shows how to forecast **airline load factor** and **profitability** in a cohesive, data-driven manner. From an **airline operations** standpoint, such a model can inform capacity decisions, route profitability analysis, and strategic **network planning**. By leveraging modern ML practices—normalization, backpropagation, multi-output regression—this methodology aligns well with both academic and industry-focused operational research, pointing the way toward advanced AI-driven airline forecasting.

---