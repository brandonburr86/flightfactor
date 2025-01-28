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

// Flight info columns for re-checking or debugging
static char dateArr[MAX_SAMPLES][16];
static char flightNoArr[MAX_SAMPLES][16];
static char originArr[MAX_SAMPLES][8];
static char destArr[MAX_SAMPLES][8];

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

// Utility
double rand_uniform(double min, double max) {
    return min + (max - min) * ((double)rand() / (double)RAND_MAX);
}

double relu(double x) { return (x > 0.0) ? x : 0.0; }
double relu_derivative(double x) { return (x > 0.0) ? 1.0 : 0.0; }

// Forward Pass
void forward_pass(const double *input, double *hidden_out, double *output_out) {
    // Hidden
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double sum = 0.0;
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += input[i] * W1[i][j];
        }
        sum += b1[j];
        hidden_out[j] = relu(sum);
    }
    // Output
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

    // dW2
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

// Normalize in-place
void normalize_input(double *features) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        // avoid dividing by zero if std_in[i] is extremely small
        if (fabs(std_in[i]) < 1e-12) {
            features[i] = (features[i] - mean_in[i]);
        } else {
            features[i] = (features[i] - mean_in[i]) / std_in[i];
        }
    }
}

// Inverse-transform outputs to original scale
void denormalize_output(double *outputs) {
    // outputs[0] = loadFactor (should be ~0..1 anyway)
    // outputs[1] = profit
    // so we only do the standard invert if we scaled them
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        outputs[k] = (outputs[k] * std_out[k]) + mean_out[k];
    }
}

// ---- MAIN ----
int main(void) {
    srand((unsigned)time(NULL));

    // Init weights
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

    // Read historical data
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
        if (seats < 1) continue; // skip nonsense or seats=0

        double load_factor = booked / seats;
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

    // ----------------------------
    // Calculate means & stddevs for each input & output column
    // so we can normalize
    // ----------------------------
    // 1) compute means
    for (int i=0; i<INPUT_SIZE; i++) mean_in[i] = 0.0;
    for (int k=0; k<OUTPUT_SIZE; k++) mean_out[k] = 0.0;

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

    // 2) compute stddevs
    for (int i=0; i<INPUT_SIZE; i++) std_in[i] = 0.0;
    for (int k=0; k<OUTPUT_SIZE; k++) std_out[k] = 0.0;

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
        std_in[i] = sqrt(std_in[i]/n_samples);
    }
    for (int k=0; k<OUTPUT_SIZE; k++) {
        std_out[k] = sqrt(std_out[k]/n_samples);
    }

    // 3) Normalize training set
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

    // ----------------------------
    // Train
    // ----------------------------
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

    // ----------------------------
    // Step B: Predict for future flights
    // We'll generate, say, 10 random future flights
    // and store the predictions in nn_predicted_profit.csv
    // ----------------------------
    FILE *fout = fopen("nn_predicted_profit.csv", "w");
    if (!fout) {
        perror("Cannot open nn_predicted_profit.csv for writing");
        return 1;
    }
    fprintf(fout,"Date,FlightNo,Origin,Destination,PredictedLoadFactor,PredictedProfit\n");

    const char* possibleOrigins[] = {"DFW","SFO","LAX","EWR","ORD","ATL","DSM","VCE","CDG"};
    const char* possibleDests[]   = {"DFW","SFO","LAX","EWR","ORD","ATL","DSM","VCE","CDG"};
    int nOrig = 9, nDest = 9;

    // Generate 10 random future flights
    for (int fidx=0; fidx<10; fidx++) {
        int year  = 2026; // near future
        int month = 1 + rand()%12;
        int day   = 1 + rand()%28;
        char dateBuf[16];
        snprintf(dateBuf,16,"%04d-%02d-%02d", year, month, day);

        int flightNum = 100 + rand()%900;
        char fltBuf[16];
        snprintf(fltBuf,16,"UA%d", flightNum);

        // pick random origin & destination
        const char* orig = possibleOrigins[rand()%nOrig];
        const char* dst  = possibleDests[rand()%nDest];
        while (strcmp(orig,dst)==0) {
            dst  = possibleDests[rand()%nDest];
        }

        // create random feature set
        double seats     = 100 + rand()%200;   // 100..300
        double booked    = seats * rand_uniform(0.3, 0.9); // 30-90% load
        double conn      = booked * rand_uniform(0.2, 0.5); // e.g. 20-50% are connecting
        double local     = booked - conn;
        double avgFare   = rand_uniform(150, 1000);
        double fuel      = rand_uniform(1000, 8000);
        double crew      = rand_uniform(500, 3000);
        double land      = rand_uniform(300, 1000);
        double overhead  = rand_uniform(2000,5000);
        double dist      = rand_uniform(500, 4000);
        double ftime     = rand_uniform(1, 10);
        double ccap      = 50 + rand()%200;
        double cfare     = rand_uniform(100, 1200);

        // Prepare input array
        double input[INPUT_SIZE] = {
            seats, booked, conn, local, avgFare, fuel, crew, land, overhead,
            dist, ftime, ccap, cfare
        };

        // Normalize
        for (int i=0; i<INPUT_SIZE; i++) {
            if (fabs(std_in[i]) < 1e-12) {
                input[i] = (input[i] - mean_in[i]);
            } else {
                input[i] = (input[i] - mean_in[i]) / std_in[i];
            }
        }

        // Forward pass
        double hidden[HIDDEN_SIZE], out[OUTPUT_SIZE];
        forward_pass(input, hidden, out);

        // Denormalize outputs
        denormalize_output(out);
        double predLoad = out[0];
        double predProf = out[1];

        fprintf(fout, "%s,%s,%s,%s,%.4f,%.2f\n",
                dateBuf, fltBuf, orig, dst, predLoad, predProf);
    }
    fclose(fout);
    printf("Wrote 10 future flight predictions to nn_predicted_profit.csv\n");

    return 0;
}

