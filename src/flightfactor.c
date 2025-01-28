#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE_LEN 1024
#define MAX_SAMPLES  100000

// The same architecture/hyperparams as your original training code
#define INPUT_SIZE   13
#define HIDDEN_SIZE  8
#define OUTPUT_SIZE  2

// We are NO longer training, so we skip EPOCHS, LEARNING_RATE, etc.

// Arrays to store data from ff_historical_data.csv
static double X[MAX_SAMPLES][INPUT_SIZE];   // We only need these to compute predictions
static int    n_samples = 0;

// We'll store flight info for referencing in the menu
static char dateArr[MAX_SAMPLES][16];
static char flightNoArr[MAX_SAMPLES][16];
static char originArr[MAX_SAMPLES][8];
static char destArr[MAX_SAMPLES][8];

// We'll store predicted values for each row
static double predictedLoadArr[MAX_SAMPLES];
static double predictedProfitArr[MAX_SAMPLES];

// -----------------------------------------------------------
// Model Parameters (loaded from ff_weights.bin)
// -----------------------------------------------------------
static double W1[INPUT_SIZE][HIDDEN_SIZE];
static double b1[HIDDEN_SIZE];
static double W2[HIDDEN_SIZE][OUTPUT_SIZE];
static double b2[OUTPUT_SIZE];

// Normalization stats (loaded from file, used for input/output scaling)
static double mean_in[INPUT_SIZE];
static double std_in[INPUT_SIZE];
static double mean_out[OUTPUT_SIZE];
static double std_out[OUTPUT_SIZE];

// -----------------------------------------------------------------------------
// Utility Functions
// -----------------------------------------------------------------------------

// Basic random uniform (still here if needed, though we won't train)
double rand_uniform(double min, double max) {
    return min + (max - min) * ((double)rand() / (double)RAND_MAX);
}

// ReLU
double relu(double x) {
    return (x > 0.0) ? x : 0.0;
}

// Forward pass: hidden + output layers
void forward_pass(const double *input, double *hidden_out, double *output_out) {
    // Hidden layer
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double sum = 0.0;
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += input[i] * W1[i][j];
        }
        sum += b1[j];
        // ReLU
        hidden_out[j] = (sum > 0.0) ? sum : 0.0;
    }

    // Output layer
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        double sum = 0.0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += hidden_out[j] * W2[j][k];
        }
        sum += b2[k];
        output_out[k] = sum;  // linear output
    }
}

// Normalize a single input vector in-place using loaded stats
void normalize_input(double *features) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        if (fabs(std_in[i]) < 1e-12) {
            features[i] = (features[i] - mean_in[i]);
        } else {
            features[i] = (features[i] - mean_in[i]) / std_in[i];
        }
    }
}

// Denormalize the output vector in-place (LoadFactor, Profit)
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
    char filename[128] = "../data/all_flight_predictions.csv";
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Cannot open CSV for writing");
        return;
    }
    fprintf(fp, "FlightCode,AvgPredictedLoadFactor,AvgPredictedProfit\n");

    // We'll collect unique codes and average their predictions
    for (int i = 0; i < n_samples; i++) {
        int already_done = 0;
        for (int j = 0; j < i; j++) {
            if (strcmp(flightNoArr[i], flightNoArr[j]) == 0) {
                already_done = 1;
                break;
            }
        }
        if (already_done) continue;

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

    // Normalize & forward pass
    normalize_input(input);
    double hidden[HIDDEN_SIZE], out[OUTPUT_SIZE];
    forward_pass(input, hidden, out);

    // Denormalize
    denormalize_output(out);

    printf("\nPredicted Load Factor:  %.4f\n", out[0]);
    printf("Predicted Profit:       %.2f\n\n", out[1]);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(void) {
    srand((unsigned)time(NULL));

    // 1) Load the pre-trained model from ff_weights.bin (no training)
    {
        FILE *fp = fopen("../data/ff_weights.bin", "rb");
        if (!fp) {
            perror("Error opening ff_weights.bin");
            return 1;
        }
        size_t readCount;

        // 1) W1
        readCount = fread(W1, sizeof(double), INPUT_SIZE * HIDDEN_SIZE, fp);
        if (readCount != (INPUT_SIZE * HIDDEN_SIZE)) {
            fprintf(stderr, "Error reading W1 from file.\n");
        }
        // 2) b1
        readCount = fread(b1, sizeof(double), HIDDEN_SIZE, fp);
        if (readCount != HIDDEN_SIZE) {
            fprintf(stderr, "Error reading b1 from file.\n");
        }
        // 3) W2
        readCount = fread(W2, sizeof(double), HIDDEN_SIZE * OUTPUT_SIZE, fp);
        if (readCount != (HIDDEN_SIZE * OUTPUT_SIZE)) {
            fprintf(stderr, "Error reading W2 from file.\n");
        }
        // 4) b2
        readCount = fread(b2, sizeof(double), OUTPUT_SIZE, fp);
        if (readCount != OUTPUT_SIZE) {
            fprintf(stderr, "Error reading b2 from file.\n");
        }
        // 5) mean_in
        readCount = fread(mean_in, sizeof(double), INPUT_SIZE, fp);
        if (readCount != INPUT_SIZE) {
            fprintf(stderr, "Error reading mean_in.\n");
        }
        // 6) std_in
        readCount = fread(std_in, sizeof(double), INPUT_SIZE, fp);
        if (readCount != INPUT_SIZE) {
            fprintf(stderr, "Error reading std_in.\n");
        }
        // 7) mean_out
        readCount = fread(mean_out, sizeof(double), OUTPUT_SIZE, fp);
        if (readCount != OUTPUT_SIZE) {
            fprintf(stderr, "Error reading mean_out.\n");
        }
        // 8) std_out
        readCount = fread(std_out, sizeof(double), OUTPUT_SIZE, fp);
        if (readCount != OUTPUT_SIZE) {
            fprintf(stderr, "Error reading std_out.\n");
        }

        fclose(fp);
        printf("Loaded model from ff_weights.bin\n");
    }

    // 2) Read ff_historical_data.csv to gather flight codes, create predictions
    //    We'll parse each row, store in X[] for forward pass,
    //    but we do NOT compute new means or do training. We rely on loaded stats.

    FILE *fcsv = fopen("../data/ff_historical_data.csv", "r");
    if (!fcsv) {
        perror("Error opening ff_historical_data.csv");
        return 1;
    }

    char line[MAX_LINE_LEN];
    int isHeader = 1;
    while (fgets(line, sizeof(line), fcsv)) {
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
        if (seats < 1) continue;

        // Build input vector
        X[n_samples][0]  = seats;
        X[n_samples][1]  = booked;
        X[n_samples][2]  = conn;
        X[n_samples][3]  = local;
        X[n_samples][4]  = avg_fare;
        X[n_samples][5]  = fuel;
        X[n_samples][6]  = crew;
        X[n_samples][7]  = land;
        X[n_samples][8]  = overhead;
        X[n_samples][9]  = dist;
        X[n_samples][10] = ftime;
        X[n_samples][11] = ccap;
        X[n_samples][12] = cfare;

        strncpy(dateArr[n_samples],  dateStr, 15);
        strncpy(flightNoArr[n_samples], fltNo, 15);
        strncpy(originArr[n_samples],    orig, 7);
        strncpy(destArr[n_samples],      dst,  7);

        n_samples++;
        if (n_samples >= MAX_SAMPLES) break;
    }
    fclose(fcsv);

    if (n_samples == 0) {
        printf("No valid samples found in ff_historical_data.csv.\n");
        return 0;
    }

    // 3) For each row, normalize X, forward pass, store predicted load/profit
    for (int s=0; s<n_samples; s++) {
        double input[INPUT_SIZE];
        // copy
        for (int i=0; i<INPUT_SIZE; i++) {
            input[i] = X[s][i];
        }
        // normalize
        normalize_input(input);

        // forward pass
        double hidden[HIDDEN_SIZE], out[OUTPUT_SIZE];
        forward_pass(input, hidden, out);

        // denormalize
        denormalize_output(out);

        predictedLoadArr[s]   = out[0];
        predictedProfitArr[s] = out[1];
    }

    // 4) Present the command-line menu (same as before)
    while (1) {
        printf("============================================\n");
        printf("   Neural Network Flight Forecasting Tool   \n");
        printf("   (loaded from ff_weights.bin)            \n");
        printf("============================================\n");
        printf("1) List all flight codes\n");
        printf("2) Predict load & profit for a given flight code\n");
        printf("3) Output table of predictions for every flight code (CSV)\n");
        printf("4) Predict on new/future flight\n");
        printf("5) Exit\n");
        printf("Select an option: ");

        int choice;
        if (scanf("%d", &choice) != 1) {
            printf("Invalid selection.\n");
            while (getchar() != '\n') {}
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
