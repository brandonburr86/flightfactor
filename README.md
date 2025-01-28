```markdown
# FlightFactor

**FlightFactor** is a suite of C-based command-line tools for generating sample airline data, training a neural network model, and performing interactive profitability/load forecasting. This project showcases a basic approach to **data-driven airline forecasting** in C.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Directory Layout](#2-directory-layout)
- [3. Installing / Building](#3-installing--building)
- [4. Usage](#4-usage)
  - [4.1 `ff_generate_data`](#41-ff_generate_data)
  - [4.2 `ff_train`](#42-ff_train)
  - [4.3 `flightfactor`](#43-flightfactor)
- [5. Data Formats](#5-data-formats)
  - [5.1 `nn_historical.csv`](#51-nn_historicalcsv)
  - [5.2 `nn_weights.bin`](#52-nn_weightsbin)
- [6. Extending FlightFactor](#6-extending-flightfactor)
- [7. Support and Credits](#7-support-and-credits)
- [8. Quick Reference](#8-quick-reference)

---

## 1. Project Overview

- **`ff_generate_data`**  
  Generates synthetic airline data (`nn_historical.csv`) for testing.  

- **`ff_train`**  
  Reads `nn_historical.csv`, trains a **single-hidden-layer neural network**, and saves the trained model weights and normalization stats to `nn_weights.bin`.  

- **`flightfactor`**  
  Provides an **interactive menu** to list flight codes, query average load/profit predictions by flight code, export predictions to a CSV, or forecast new/future flights.  
  - In some variants, `flightfactor` can **train on load**; in others, it **loads** pre-trained weights from `nn_weights.bin` to skip retraining.

---

## 2. Directory Layout

```
flightfactor/
├─ Makefile
├─ bin/                           # Compiled executables appear here
├─ obj/                           # Compiled object files
└─ src/                           # Source code (.c)
   ├─ ff_generate_data.c
   ├─ ff_train.c
   └─ flightfactor.c
```

- **Makefile**: Defines how to build each executable.  
- **`bin/`**: Target directory for the compiled binaries (`ff_generate_data`, `ff_train`, and `flightfactor`).  
- **`obj/`**: Holds `.o` (object) files created during compilation.  
- **`src/`**: Contains the source files in C.

---

## 3. Installing / Building

1. **Requirements**  
   - A C compiler (e.g., GCC) and standard math library (`-lm`).  

2. **Build**  
   - In the project root (`flightfactor/`), run:
     ```bash
     make
     ```
   - This will:
     1. Create `obj/` and `bin/` if not present.  
     2. Compile each `.c` in `src/` to an object file in `obj/`.  
     3. Link those object files into final executables in `bin/`.

3. **Clean**  
   - Remove executables and object files:
     ```bash
     make clean
     ```

---

## 4. Usage

### 4.1 `ff_generate_data`

- **Purpose**: Creates synthetic airline flight data.  
- **Usage**:
  ```bash
  ./bin/ff_generate_data
  ```
- **Output**: By default, generates `nn_historical.csv` in the current directory with columns like Date, Flight_No, Seats, Booked_Pax, Fuel_Cost, Overhead, etc.

### 4.2 `ff_train`

- **Purpose**: Trains the neural network on `nn_historical.csv` and saves model parameters to `nn_weights.bin`.  
- **Usage**:
  ```bash
  ./bin/ff_train
  ```
- **Details**:
  - Reads `nn_historical.csv`.  
  - Normalizes the data (calculates means/std devs).  
  - Performs feed-forward/backprop for a specified number of epochs (e.g., 100).  
  - Saves:
    - **W1**, **b1**, **W2**, **b2** (network weights)  
    - **mean_in**, **std_in**, **mean_out**, **std_out** (normalization stats)  
    into `nn_weights.bin`.

### 4.3 `flightfactor`

- **Purpose**: Main interactive tool for listing flight codes, retrieving average predictions, exporting a table of predictions, or forecasting new/future flights.  
- **Usage**:
  ```bash
  ./bin/flightfactor
  ```
- **Menu**:
  ```
  1) List all flight codes
  2) Predict load & profit for a given flight code
  3) Output table of predictions for every flight code (CSV)
  4) Predict on new/future flight
  5) Exit
  ```
- **Notes**:
  - In one version, `flightfactor` trains on load by reading and processing `nn_historical.csv`.  
  - In another version, it might **load** `nn_weights.bin` to skip training.  
  - Let’s you see **Load Factor** and **Profit** predictions, aggregated by flight code or on a new flight entry.

---

## 5. Data Formats

### 5.1 `nn_historical.csv`

- **Columns** (17 total):
  1. Date  
  2. Flight_No  
  3. Origin  
  4. Destination  
  5. Seats  
  6. Booked_Pax  
  7. Connecting_Pax  
  8. Local_Pax  
  9. Average_Fare  
  10. Fuel_Cost_per_leg  
  11. Crew_Cost_per_leg  
  12. Landing_Fee  
  13. Overhead_Cost  
  14. Distance  
  15. Flight_Time  
  16. Competitor_Capacity  
  17. Competitor_Fare  

- **Observed** load factor and profit are typically computed as:  
  - `LoadFactor = Booked_Pax / Seats`  
  - `Profit = (Booked_Pax * Average_Fare) - (Fuel + Crew + Landing + Overhead)`

### 5.2 `nn_weights.bin`

- **Binary** file storing the neural network parameters and normalization stats:
  1. `W1` (size `INPUT_SIZE x HIDDEN_SIZE`)  
  2. `b1` (size `HIDDEN_SIZE`)  
  3. `W2` (size `HIDDEN_SIZE x OUTPUT_SIZE`)  
  4. `b2` (size `OUTPUT_SIZE`)  
  5. `mean_in` (size `INPUT_SIZE`)  
  6. `std_in` (size `INPUT_SIZE`)  
  7. `mean_out` (size `OUTPUT_SIZE`)  
  8. `std_out` (size `OUTPUT_SIZE`)

Any application that loads this file must read in the **same order**.

---

## 6. Extending FlightFactor

1. **Multiple Object Files**  
   - If the project grows, consider splitting each tool into more `.c` files and adjusting the Makefile accordingly.  
2. **GPU Acceleration**  
   - For very large datasets, or advanced neural networks, consider libraries (e.g., C++ with CUDA) or a high-level ML framework.  
3. **Train/Test Splits**  
   - Currently, the examples train on all data. Real-world usage requires validation splits.  
4. **Refining the Menu**  
   - Could add subcommands, date-based queries, or real-time booking curve logic.

---

## 7. Support and Credits

- **Authors**: Created for demonstration of a baseline neural network approach to airline forecasting in C.  
- **License**: Provided “as is.” Consult the repository’s license file if present.

---

## 8. Quick Reference

1. **Build**  
   ```bash
   cd flightfactor
   make
   ```
2. **Generate**  
   ```bash
   ./bin/ff_generate_data   # -> outputs nn_historical.csv
   ```
3. **Train**  
   ```bash
   ./bin/ff_train          # -> reads nn_historical.csv, outputs nn_weights.bin
   ```
4. **Interact**  
   ```bash
   ./bin/flightfactor      # -> launches menu
   ```
5. **Clean**  
   ```bash
   make clean
   ```

Enjoy exploring **FlightFactor** for airline flight forecasting!
```