#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE_LEN 1024
#define MAX_SAMPLES  100000

// Hyperparameters
#define INPUT_SIZE   13
#define HIDDEN_SIZE  8
#define OUTPUT_SIZE  2
#define EPOCHS       100
#define LEARNING_RATE 0.0000005

// Global Data Arrays
static double X[MAX_SAMPLES][INPUT_SIZE];
static double Y[MAX_SAMPLES][OUTPUT_SIZE];
static int    n_samples = 0;

// Neural Net Weights/Biases
static double W1[INPUT_SIZE][HIDDEN_SIZE];
static double b1[HIDDEN_SIZE];
static double W2[HIDDEN_SIZE][OUTPUT_SIZE];
static double b2[OUTPUT_SIZE];

// Normalization parameters
static double mean_in[INPUT_SIZE];
static double std_in[INPUT_SIZE];
static double mean_out[OUTPUT_SIZE];
static double std_out[OUTPUT_SIZE];

double rand_uniform(double min, double max) {
    return min + (max - min) * ((double)rand() / (double)RAND_MAX);
}

double relu(double x) {
    return (x > 0.0) ? x : 0.0;
}

double relu_derivative(double x) {
    return (x > 0.0) ? 1.0 : 0.0;
}

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
        output_out[k] = sum;  // linear
    }
}

void backward_pass(const double *input, const double *hidden_out,
                   const double *output, const double *target) {
    // error = output - target
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

        // update W1, b1
        for (int i = 0; i < INPUT_SIZE; i++) {
            double grad = dHidden * input[i];
            W1[i][j] -= LEARNING_RATE * grad;
        }
        b1[j] -= LEARNING_RATE * dHidden;
    }
}

int main(void) {
    srand((unsigned)time(NULL));

    // 1) Initialize weights randomly
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

    // 2) Read "ff_historical_data.csv"
    FILE *fp = fopen("../data/ff_historical_data.csv", "r");
    if (!fp) {
        perror("Error opening ff_historical_data.csv");
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

        // Some columns for reference, though not stored in separate arrays
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
        if (seats < 1) continue; // skip rows with seats=0

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

        n_samples++;
        if (n_samples >= MAX_SAMPLES) break;
    }
    fclose(fp);

    if (n_samples == 0) {
        printf("No valid samples in ff_historical_data.csv.\n");
        return 0;
    }

    // 3) Compute means & stddev for each input/output
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

    // 4) Normalize training set
    for (int s=0; s<n_samples; s++) {
        // Inputs
        for (int i=0; i<INPUT_SIZE; i++) {
            if (fabs(std_in[i]) < 1e-12) {
                X[s][i] = (X[s][i] - mean_in[i]);
            } else {
                X[s][i] = (X[s][i] - mean_in[i]) / std_in[i];
            }
        }
        // Outputs
        for (int k=0; k<OUTPUT_SIZE; k++) {
            if (fabs(std_out[k]) < 1e-12) {
                Y[s][k] = (Y[s][k] - mean_out[k]);
            } else {
                Y[s][k] = (Y[s][k] - mean_out[k]) / std_out[k];
            }
        }
    }

    // 5) Train the network
    printf("Training on %d samples...\n", n_samples);
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double mse = 0.0;
        for (int s = 0; s < n_samples; s++) {
            double hidden[HIDDEN_SIZE], out[OUTPUT_SIZE];
            forward_pass(X[s], hidden, out);

            double e0 = out[0] - Y[s][0];
            double e1 = out[1] - Y[s][1];
            mse += (e0*e0 + e1*e1);

            backward_pass(X[s], hidden, out, Y[s]);
        }
        mse /= (double)n_samples;
        if ((epoch + 1) % 10 == 0) {
            printf("Epoch %d/%d - MSE: %f\n", epoch+1, EPOCHS, mse);
        }
    }
    printf("Training complete.\n\n");

    // 6) Save Weights/Biases/Normalization to disk
    // We'll save to a single binary file "ff_weights.bin"
    {
        FILE *wfp = fopen("../data/ff_weights.bin", "wb");
        if (!wfp) {
            perror("Failed to open ff_weights.bin for writing");
            return 1;
        }

        // We'll store everything in a fixed order:
        //  1) W1 (INPUT_SIZE x HIDDEN_SIZE)
        //  2) b1 (HIDDEN_SIZE)
        //  3) W2 (HIDDEN_SIZE x OUTPUT_SIZE)
        //  4) b2 (OUTPUT_SIZE)
        //  5) mean_in (INPUT_SIZE)
        //  6) std_in (INPUT_SIZE)
        //  7) mean_out (OUTPUT_SIZE)
        //  8) std_out (OUTPUT_SIZE)

        size_t written = 0;
        
        // W1
        written = fwrite(W1, sizeof(double), INPUT_SIZE * HIDDEN_SIZE, wfp);
        if (written != (INPUT_SIZE * HIDDEN_SIZE)) {
            fprintf(stderr, "Error writing W1.\n");
        }
        // b1
        written = fwrite(b1, sizeof(double), HIDDEN_SIZE, wfp);
        if (written != HIDDEN_SIZE) {
            fprintf(stderr, "Error writing b1.\n");
        }
        // W2
        written = fwrite(W2, sizeof(double), HIDDEN_SIZE * OUTPUT_SIZE, wfp);
        if (written != (HIDDEN_SIZE * OUTPUT_SIZE)) {
            fprintf(stderr, "Error writing W2.\n");
        }
        // b2
        written = fwrite(b2, sizeof(double), OUTPUT_SIZE, wfp);
        if (written != OUTPUT_SIZE) {
            fprintf(stderr, "Error writing b2.\n");
        }
        // mean_in
        written = fwrite(mean_in, sizeof(double), INPUT_SIZE, wfp);
        if (written != INPUT_SIZE) {
            fprintf(stderr, "Error writing mean_in.\n");
        }
        // std_in
        written = fwrite(std_in, sizeof(double), INPUT_SIZE, wfp);
        if (written != INPUT_SIZE) {
            fprintf(stderr, "Error writing std_in.\n");
        }
        // mean_out
        written = fwrite(mean_out, sizeof(double), OUTPUT_SIZE, wfp);
        if (written != OUTPUT_SIZE) {
            fprintf(stderr, "Error writing mean_out.\n");
        }
        // std_out
        written = fwrite(std_out, sizeof(double), OUTPUT_SIZE, wfp);
        if (written != OUTPUT_SIZE) {
            fprintf(stderr, "Error writing std_out.\n");
        }

        fclose(wfp);
        printf("Saved trained weights to ff_weights.bin\n");
    }

    return 0;
}
