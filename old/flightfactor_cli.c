#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE_LEN 1024
#define MAX_SAMPLES  100000

#define INPUT_SIZE   13
#define HIDDEN_SIZE  8
#define OUTPUT_SIZE  2
#define EPOCHS       100
#define LEARNING_RATE 0.0000005

// Arrays to store training data
static double X[MAX_SAMPLES][INPUT_SIZE];
static double Y[MAX_SAMPLES][OUTPUT_SIZE];
static int    n_samples = 0;

// Flight info columns for reference
static char dateArr[MAX_SAMPLES][16];
static char flightNoArr[MAX_SAMPLES][16];
static char originArr[MAX_SAMPLES][8];
static char destArr[MAX_SAMPLES][8];

// We'll store final predicted values for each training sample
static double predictedLoadArr[MAX_SAMPLES];
static double predictedProfitArr[MAX_SAMPLES];

// Neural Net Weights/Biases
static double W1[INPUT_SIZE][HIDDEN_SIZE];
static double b1[HIDDEN_SIZE];
static double W2[HIDDEN_SIZE][OUTPUT_SIZE];
static double b2[OUTPUT_SIZE];

// Normalization params: mean/std for each input dimension
static double mean_in[INPUT_SIZE];
static double std_in[INPUT_SIZE];

// For output scaling (especially for Profit)
static double mean_out[OUTPUT_SIZE];
static double std_out[OUTPUT_SIZE];

// -----------------------------------------------------------------------------
// Utility Functions
// -----------------------------------------------------------------------------
double rand_uniform(double min, double max) {
    return min + (max - min) * ((double)rand() / (double)RAND_MAX);
}

double relu(double x) { return (x > 0.0) ? x : 0.0; }
double relu_derivative(double x) { return (x > 0.0) ? 1.0 : 0.0; }

// Forward Pass
void forward_pass(const double *input, double *hidden_out, double *output_out) {
    // Hidden layer
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double sum = 0.0;
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += input[i] * W1[i][j];
        }
        sum += b1[j];
        hidden_out[j] = relu(sum);
    }

    // Output layer
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        double sum = 0.0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += hidden_out[j] * W2[j][k];
        }
        sum += b2[k];
        output_out[k] = sum;  // linear for regression
    }
}

// Backprop for single sample
void backward_pass(const double *input, const double *hidden_out,
                   const double *output, const double *target) {
    double error[OUTPUT_SIZE];
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        error[k] = output[k] - target[k];
    }

    // dW2, dB2
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double grad = error[k] * hidden_out[j];
            W2[j][k] -= LEARNING_RATE * grad;
        }
        b2[k] -= LEARNING_RATE * error[k];
    }

    // dHidden
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double dHidden = 0.0;
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            dHidden += error[k] * W2[j][k];
        }
        dHidden *= relu_derivative(hidden_out[j]);

        for (int i = 0; i < INPUT_SIZE; i++) {
            double grad = dHidden * input[i];
            W1[i][j] -= LEARNING_RATE * grad;
        }
        b1[j] -= LEARNING_RATE * dHidden;
    }
}

// Normalize a single input vector in-place
void normalize_input(double *features) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        if (fabs(std_in[i]) < 1e-12) {
            features[i] = (features[i] - mean_in[i]);
        } else {
            features[i] = (features[i] - mean_in[i]) / std_in[i];
        }
    }
}

// Inverse-transform a 2D output vector (LoadFactor, Profit)
void denormalize_output(double *outputs) {
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        outputs[k] = (outputs[k] * std_out[k]) + mean_out[k];
    }
}

// -----------------------------------------------------------------------------
// Menu Functions
// -----------------------------------------------------------------------------

// 1) List all unique flight codes
void menu_list_flight_codes() {
    printf("\nAvailable Flight Codes:\n");
    for (int i = 0; i < n_samples; i++) {
        // Check if we've already printed this flight code
        int already_printed = 0;
        for (int j = 0; j < i; j++) {
            if (strcmp(flightNoArr[i], flightNoArr[j]) == 0) {
                already_printed = 1;
                break;
            }
        }
        if (!already_printed) {
            printf("  %s\n", flightNoArr[i]);
        }
    }
    printf("\n");
}

// 2) Prompt user for flight code, compute average predicted load & profit
void menu_predict_flight_code() {
    char userCode[32];
    printf("\nEnter the flight code (e.g. UA123): ");
    scanf("%s", userCode);

    double sumLoad = 0.0;
    double sumProfit = 0.0;
    int count = 0;

    for (int i = 0; i < n_samples; i++) {
        if (strcmp(userCode, flightNoArr[i]) == 0) {
            sumLoad   += predictedLoadArr[i];
            sumProfit += predictedProfitArr[i];
            count++;
        }
    }

    if (count == 0) {
        printf("No flights found with code %s.\n\n", userCode);
        return;
    }

    double avgLoad = sumLoad / (double)count;
    double avgProfit = sumProfit / (double)count;

    printf("Average Predictions for Flight Code %s:\n", userCode);
    printf("  Load Factor:  %.4f\n", avgLoad);
    printf("  Profit:       %.2f\n\n", avgProfit);
}

// 3) Output table of predictions for *every* flight code into a CSV
void menu_output_all_codes_csv() {
    char filename[128] = "all_flight_predictions.csv";
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Cannot open CSV for writing");
        return;
    }
    fprintf(fp, "FlightCode,AvgPredictedLoadFactor,AvgPredictedProfit\n");

    // We'll collect unique codes and average their predictions
    for (int i = 0; i < n_samples; i++) {
        // check if we've processed this code already
        int already_done = 0;
        for (int j = 0; j < i; j++) {
            if (strcmp(flightNoArr[i], flightNoArr[j]) == 0) {
                already_done = 1;
                break;
            }
        }
        if (already_done) continue;

        // sum predictions
        double sumLoad   = 0.0;
        double sumProfit = 0.0;
        int count        = 0;
        for (int k = i; k < n_samples; k++) {
            if (strcmp(flightNoArr[i], flightNoArr[k]) == 0) {
                sumLoad   += predictedLoadArr[k];
                sumProfit += predictedProfitArr[k];
                count++;
            }
        }
        double avgLoad   = sumLoad / (double)count;
        double avgProfit = sumProfit / (double)count;

        fprintf(fp, "%s,%.4f,%.2f\n", flightNoArr[i], avgLoad, avgProfit);
    }

    fclose(fp);
    printf("Wrote all flight predictions to %s\n\n", filename);
}

// 4) Prompt user to create a *new/future* flight and predict load/profit
void menu_predict_new_flight() {
    // We'll gather all 13 input features from the user
    double input[INPUT_SIZE];
    printf("\nEnter flight data:\n");
    printf("Seats (int): ");
    scanf("%lf", &input[0]);

    printf("Booked Pax (double): ");
    scanf("%lf", &input[1]);

    printf("Connecting Pax (double): ");
    scanf("%lf", &input[2]);

    printf("Local Pax (double): ");
    scanf("%lf", &input[3]);

    printf("Average Fare (double): ");
    scanf("%lf", &input[4]);

    printf("Fuel Cost per leg (double): ");
    scanf("%lf", &input[5]);

    printf("Crew Cost per leg (double): ");
    scanf("%lf", &input[6]);

    printf("Landing Fee (double): ");
    scanf("%lf", &input[7]);

    printf("Overhead Cost (double): ");
    scanf("%lf", &input[8]);

    printf("Distance (double): ");
    scanf("%lf", &input[9]);

    printf("Flight Time (double): ");
    scanf("%lf", &input[10]);

    printf("Competitor Capacity (double): ");
    scanf("%lf", &input[11]);

    printf("Competitor Fare (double): ");
    scanf("%lf", &input[12]);

    // Now normalize & do forward pass
    normalize_input(input);
    double hidden[HIDDEN_SIZE], out[OUTPUT_SIZE];
    forward_pass(input, hidden, out);

    // Denormalize
    denormalize_output(out);

    double predLoad  = out[0];
    double predProfit= out[1];

    printf("\nPredicted Load Factor:  %.4f\n", predLoad);
    printf("Predicted Profit:       %.2f\n\n", predProfit);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(void) {
    srand((unsigned)time(NULL));

    // 1) Initialize Weights
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            W1[i][j] = rand_uniform(-0.01, 0.01);
        }
    }
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        b1[j] = 0.0;
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            W2[j][k] = rand_uniform(-0.01, 0.01);
        }
    }
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        b2[k] = 0.0;
    }

    // 2) Read Historical Data
    FILE *fp = fopen("nn_historical.csv", "r");
    if (!fp) {
        perror("Error opening nn_historical.csv");
        return 1;
    }

    char line[MAX_LINE_LEN];
    int isHeader = 1;
    while (fgets(line, sizeof(line), fp)) {
        if (isHeader) {
            isHeader = 0;
            continue;
        }
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';

        char *token = strtok(line, ",");
        int colIndex = 0;

        double seats=0, booked=0, conn=0, local=0;
        double avg_fare=0, fuel=0, crew=0, land=0, overhead=0;
        double dist=0, ftime=0, ccap=0, cfare=0;

        char dateStr[16]={0}, fltNo[16]={0}, orig[8]={0}, dst[8]={0};

        while (token) {
            switch (colIndex) {
                case 0: strncpy(dateStr, token, 15); break;
                case 1: strncpy(fltNo,   token, 15); break;
                case 2: strncpy(orig,    token, 7);  break;
                case 3: strncpy(dst,     token, 7);  break;
                case 4: seats     = atof(token); break;
                case 5: booked    = atof(token); break;
                case 6: conn      = atof(token); break;
                case 7: local     = atof(token); break;
                case 8: avg_fare  = atof(token); break;
                case 9: fuel      = atof(token); break;
                case 10: crew     = atof(token); break;
                case 11: land     = atof(token); break;
                case 12: overhead = atof(token); break;
                case 13: dist     = atof(token); break;
                case 14: ftime    = atof(token); break;
                case 15: ccap     = atof(token); break;
                case 16: cfare    = atof(token); break;
                default: break;
            }
            token = strtok(NULL, ",");
            colIndex++;
        }
        if (seats < 1) continue; // skip seats=0 or invalid rows

        double load_factor = (seats > 0) ? (booked / seats) : 0.0;
        double profit = (booked * avg_fare) - (fuel + crew + land + overhead);

        X[n_samples][0] = seats;
        X[n_samples][1] = booked;
        X[n_samples][2] = conn;
        X[n_samples][3] = local;
        X[n_samples][4] = avg_fare;
        X[n_samples][5] = fuel;
        X[n_samples][6] = crew;
        X[n_samples][7] = land;
        X[n_samples][8] = overhead;
        X[n_samples][9] = dist;
        X[n_samples][10] = ftime;
        X[n_samples][11] = ccap;
        X[n_samples][12] = cfare;

        Y[n_samples][0] = load_factor; 
        Y[n_samples][1] = profit;

        strncpy(dateArr[n_samples], dateStr, 15);
        strncpy(flightNoArr[n_samples], fltNo, 15);
        strncpy(originArr[n_samples], orig, 7);
        strncpy(destArr[n_samples], dst, 7);

        n_samples++;
        if (n_samples >= MAX_SAMPLES) break;
    }
    fclose(fp);

    if (n_samples == 0) {
        printf("No valid samples.\n");
        return 0;
    }

    // 3) Compute Means & Stddev for Each Column
    for (int i=0; i<INPUT_SIZE; i++) {
        mean_in[i] = 0.0;
        std_in[i] = 0.0;
    }
    for (int k=0; k<OUTPUT_SIZE; k++) {
        mean_out[k] = 0.0;
        std_out[k] = 0.0;
    }

    // Means
    for (int s=0; s<n_samples; s++) {
        for (int i=0; i<INPUT_SIZE; i++) {
            mean_in[i] += X[s][i];
        }
        for (int k=0; k<OUTPUT_SIZE; k++) {
            mean_out[k] += Y[s][k];
        }
    }
    for (int i=0; i<INPUT_SIZE; i++) {
        mean_in[i] /= n_samples;
    }
    for (int k=0; k<OUTPUT_SIZE; k++) {
        mean_out[k] /= n_samples;
    }

    // Stddev
    for (int s=0; s<n_samples; s++) {
        for (int i=0; i<INPUT_SIZE; i++) {
            double diff = X[s][i] - mean_in[i];
            std_in[i] += diff * diff;
        }
        for (int k=0; k<OUTPUT_SIZE; k++) {
            double diff = Y[s][k] - mean_out[k];
            std_out[k] += diff * diff;
        }
    }
    for (int i=0; i<INPUT_SIZE; i++) {
        std_in[i] = sqrt(std_in[i] / n_samples);
    }
    for (int k=0; k<OUTPUT_SIZE; k++) {
        std_out[k] = sqrt(std_out[k] / n_samples);
    }

    // 4) Normalize the Training Set
    for (int s=0; s<n_samples; s++) {
        // X
        for (int i=0; i<INPUT_SIZE; i++) {
            if (fabs(std_in[i]) < 1e-12) {
                X[s][i] = (X[s][i] - mean_in[i]);
            } else {
                X[s][i] = (X[s][i] - mean_in[i]) / std_in[i];
            }
        }
        // Y
        for (int k=0; k<OUTPUT_SIZE; k++) {
            if (fabs(std_out[k]) < 1e-12) {
                Y[s][k] = (Y[s][k] - mean_out[k]);
            } else {
                Y[s][k] = (Y[s][k] - mean_out[k]) / std_out[k];
            }
        }
    }

    // 5) Train
    printf("Training on %d samples...\n", n_samples);
    for (int epoch=0; epoch<EPOCHS; epoch++) {
        double mse = 0.0;
        for (int s=0; s<n_samples; s++) {
            double hidden[HIDDEN_SIZE], out[OUTPUT_SIZE];
            forward_pass(X[s], hidden, out);

            // MSE
            double e0 = out[0] - Y[s][0];
            double e1 = out[1] - Y[s][1];
            mse += (e0*e0 + e1*e1);

            // Backprop
            backward_pass(X[s], hidden, out, Y[s]);
        }
        mse /= (double)n_samples;
        if ((epoch+1) % 10 == 0) {
            printf("Epoch %d/%d - MSE: %f\n", epoch+1, EPOCHS, mse);
        }
    }
    printf("Training complete.\n\n");

    // 6) After Training, Let's Predict on the *Training* Data
    //    We'll store the predicted load/profit in predictedLoadArr / predictedProfitArr.
    for (int s = 0; s < n_samples; s++) {
        double hidden[HIDDEN_SIZE], out[OUTPUT_SIZE];
        // forward pass on the normalized X[s]
        forward_pass(X[s], hidden, out);
        // Denormalize
        denormalize_output(out);

        predictedLoadArr[s]   = out[0];  // predicted load factor
        predictedProfitArr[s] = out[1];  // predicted profit
    }

    // 7) Command Prompt Menu
    while (1) {
        printf("============================================\n");
        printf("   Neural Network Flight Forecasting Tool   \n");
        printf("============================================\n");
        printf("1) List all flight codes\n");
        printf("2) Predict load & profit for a given flight code\n");
        printf("3) Output table of predictions for every flight code (CSV)\n");
        printf("4) Predict on new/future flight\n");
        printf("5) Exit\n");
        printf("Select an option: ");

        int choice;
        if (scanf("%d", &choice) != 1) {
            // invalid input
            printf("Invalid selection.\n");
            while (getchar() != '\n') {} // clear stdin buffer
            continue;
        }

        switch (choice) {
            case 1:
                menu_list_flight_codes();
                break;
            case 2:
                menu_predict_flight_code();
                break;
            case 3:
                menu_output_all_codes_csv();
                break;
            case 4:
                menu_predict_new_flight();
                break;
            case 5:
                printf("Exiting...\n");
                return 0;
            default:
                printf("Unknown option. Please try again.\n");
                break;
        }
    }

    return 0;
}

