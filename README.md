# **FlightFactor**

**FlightFactor** is a suite of C-based command-line tools that help generate sample airline data, train a neural network model to predict flight load factor and profitability, and then interact with predictions. These tools are illustrative of how an airline might implement a basic data-driven forecasting workflow in C.

---

## **1. Project Overview**

- **ff_generate_data**: Generates synthetic airline data (`nn_historical.csv`), suitable for testing and experimentation.  
- **ff_train**: Reads `nn_historical.csv`, trains a neural network, and saves model weights to disk (`nn_weights.bin`).  
- **flightfactor**: Application loads pre-trained weights on start. Provides an interactive menu to list flight codes, query average predictions, output a CSV table, or predict for new/future flights.

The tools **all** reside in the **FlightFactor** project, with source code in `./src/`, object files in `./obj/`, and compiled executables in `./bin/`.

---

## **2. Directory Layout**

```
flightfactor/                     (Project root)
├─ Makefile                       (Build instructions)
├─ bin/                           (Final executables go here)
├─ obj/                           (Compiled object files go here)
└─ src/                           (Source directory)
   ├─ ff_generate_data.c          (Sample data generator)
   ├─ ff_train.c                  (Training tool that saves model)
   └─ flightfactor.c              (Main menu tool for predictions)
```

---

## **3. Installing / Building**

1. **Install Dependencies**  
   - You only need a C compiler (e.g. GCC) and standard libraries (`libm` for math).  
   - On Linux/Unix systems, you typically already have GCC and `-lm`.

2. **Compile**  
   - From the `flightfactor/` directory, run:
     ```bash
     make
     ```
   - This invokes the **Makefile**, which:
     1. Creates `obj/` and `bin/` if they don’t exist.  
     2. Compiles `.c` source files from `./src/` into `.o` objects in `./obj/`.  
     3. Links those object files into executables in `./bin/`.

3. **Executables**  
   - After building, you should see:
     ```
     bin/ff_generate_data
     bin/ff_train
     bin/flightfactor
     ```

4. **Cleanup**  
   - To remove build artifacts:
     ```bash
     make clean
     ```
   - This deletes `.o` files in `obj/` and the executables in `bin/`.

---

## **4. Usage**

### **4.1 `ff_generate_data`**

- **Purpose**: Creates synthetic airline data in `nn_historical.csv`.  
- **Usage**:
  ```bash
  ./bin/ff_generate_data
  ```
  - By default, it may create 10,000 rows of data, but you can often change that in the source file (`#define ROWS 10000`).  
  - Outputs `nn_historical.csv` in the current directory with random flight info columns.

### **4.2 `ff_train`**

- **Purpose**: Trains the neural network using `nn_historical.csv` and saves the learned model parameters to `nn_weights.bin`.  
- **Usage**:
  ```bash
  ./bin/ff_train
  ```
  - Expects `nn_historical.csv` to be in the same directory.  
  - Performs feed-forward neural network training (one hidden layer).  
  - After finishing, writes **`nn_weights.bin`** (a binary file) containing:
    - **W1**, **b1**, **W2**, **b2** (the network weights)  
    - **mean_in**, **std_in**, **mean_out**, **std_out** (normalization stats)

### **4.3 `flightfactor`**

- **Purpose**: Provides an interactive menu for forecasting flight load factor and profitability.  
- **Two variants** exist:
  1. **Trains On Load** version: Where `flightfactor` itself reads `nn_historical.csv`, trains, and then offers the menu.  
  2. **Load Pretrained** version: Where `flightfactor` (or a similarly named executable) **loads** `nn_weights.bin` so it **skips** training.  

In either case, once launched, you’ll see a menu like:

```
============================================
   Neural Network Flight Forecasting Tool
============================================
1) List all flight codes
2) Predict load & profit for a given flight code
3) Output table of predictions for every flight code (CSV)
4) Predict on new/future flight
5) Exit
Select an option:
```

- **Option 1**: Lists unique flight codes found in your historical data.  
- **Option 2**: Prompts for a code (e.g., `UA123`), then displays **average** predicted load factor & profit across rows that share that code.  
- **Option 3**: Writes a CSV (`all_flight_predictions.csv`) summarizing each code’s average predictions.  
- **Option 4**: Lets you enter data for a brand-new flight (seats, fare, competitor capacity, etc.) and returns immediate predictions.  
- **Option 5**: Exits the program.

**Usage Example**:
```bash
./bin/flightfactor
```

---

## **5. Data Formats**

### **5.1 `nn_historical.csv`**  
Typically has 17 columns (header row + flight data rows):

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

The training or inference code uses these columns to compute or predict:

- **Load Factor** = Booked_Pax / Seats  
- **Profit** = (Booked_Pax * Average_Fare) - (Fuel + Crew + Landing + Overhead)

### **5.2 `nn_weights.bin`**  
Binary file storing:

1. `W1` (size `INPUT_SIZE x HIDDEN_SIZE`)  
2. `b1` (size `HIDDEN_SIZE`)  
3. `W2` (size `HIDDEN_SIZE x OUTPUT_SIZE`)  
4. `b2` (size `OUTPUT_SIZE`)  
5. `mean_in` (size `INPUT_SIZE`)  
6. `std_in` (size `INPUT_SIZE`)  
7. `mean_out` (size `OUTPUT_SIZE`)  
8. `std_out` (size `OUTPUT_SIZE`)

Any code that **loads** this file must read in the exact same order.


---

## **6. Support and Credits**

- **Authors**: Written by Brandon Burr - brandon@jetlogiq.com
- **License**: Provided as-is; no warranties. Check the repository’s license file if present.

---

## **7. Quick Reference**

### **Build**

```bash
cd /code/flightfactor
make
```

### **Run Generators**

```bash
./bin/ff_generate_data
# -> outputs nn_historical.csv
```

### **Train**

```bash
./bin/ff_train
# -> reads nn_historical.csv, saves nn_weights.bin
```

### **Inference / Menu**

```bash
./bin/flightfactor
# -> provides interactive menu
```
