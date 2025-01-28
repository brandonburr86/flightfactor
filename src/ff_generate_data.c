#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define ROWS 100000  // Change this to generate more or fewer rows

// Small pool of IATA airport codes to randomly choose from
static const char *AIRPORT_CODES[] = {
    "EWR", "SFO", "ORD", "DSM", "VCE", "LAX", "ATL", "DFW", "JFK", "CDG"
};
static const int NUM_AIRPORT_CODES = 10;

// Utility: random integer in [min, max]
int rand_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

// Utility: random double in [min, max]
double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / (double)RAND_MAX);
}

int main(void) {
    // Seed random generator
    srand((unsigned int)time(NULL));

    // Open output file
    FILE *fp = fopen("../data/ff_historical_data.csv", "w");
    if (!fp) {
        perror("Failed to open ff_historical_data.csv for writing");
        return 1;
    }

    // Write CSV header
    fprintf(fp, "Date,Flight_No,Origin,Destination,Seats,Booked_Pax,Connecting_Pax,Local_Pax,"
                "Average_Fare,Fuel_Cost_per_leg,Crew_Cost_per_leg,Landing_Fee,Overhead_Cost,"
                "Distance,Flight_Time,Competitor_Capacity,Competitor_Fare\n");

    for (int i = 0; i < ROWS; i++) {
        // 1) Random date in historical range (e.g., 2020–2024)
        int year  = rand_int(2020, 2024);
        int month = rand_int(1, 12);
        int day   = rand_int(1, 28);  // keep it simple with up to 28 days
        char dateStr[16];
        snprintf(dateStr, sizeof(dateStr), "%04d-%02d-%02d", year, month, day);

        // 2) Flight number: "UA" + random 3-digit number
        int flightNum = rand_int(100, 999);
        char flightNo[16];
        snprintf(flightNo, sizeof(flightNo), "UA%d", flightNum);

        // 3) Origin / Destination
        const char *orig = AIRPORT_CODES[rand_int(0, NUM_AIRPORT_CODES - 1)];
        const char *dest = AIRPORT_CODES[rand_int(0, NUM_AIRPORT_CODES - 1)];
        while (strcmp(orig, dest) == 0) {
            dest = AIRPORT_CODES[rand_int(0, NUM_AIRPORT_CODES - 1)];
        }

        // 4) Seats (100–300)
        int seats = rand_int(100, 300);

        // 5) Booked_Pax (0–seats)
        int booked_pax = rand_int(0, seats);

        // 6) Connecting_Pax (0–booked_pax)
        int connecting_pax = rand_int(0, booked_pax);

        // 7) Local_Pax = Booked_Pax - Connecting_Pax
        int local_pax = booked_pax - connecting_pax;

        // 8) Average Fare (100–1200)
        double avg_fare = rand_double(100.0, 1200.0);

        // 9) Fuel Cost (1000–10000)
        double fuel_cost = rand_double(1000.0, 10000.0);

        // 10) Crew Cost (500–3000)
        double crew_cost = rand_double(500.0, 3000.0);

        // 11) Landing Fee (300–1000)
        double landing_fee = rand_double(300.0, 1000.0);

        // 12) Overhead Cost (2000–5000)
        double overhead_cost = rand_double(2000.0, 5000.0);

        // 13) Distance (100–6000 miles)
        double distance = rand_double(100.0, 6000.0);

        // 14) Flight Time (1–12 hours)
        double flight_time = rand_double(1.0, 12.0);

        // 15) Competitor Capacity (50–300)
        int comp_capacity = rand_int(50, 300);

        // 16) Competitor Fare (100–1200)
        double comp_fare = rand_double(100.0, 1200.0);

        // Write the row to nn_historical.csv
        fprintf(fp,
                "%s,%s,%s,%s,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.2f\n",
                dateStr,
                flightNo,
                orig,
                dest,
                seats,
                booked_pax,
                connecting_pax,
                local_pax,
                avg_fare,
                fuel_cost,
                crew_cost,
                landing_fee,
                overhead_cost,
                distance,
                flight_time,
                comp_capacity,
                comp_fare
        );
    }

    fclose(fp);
    printf("Successfully generated %d rows of historical data in ff_historical_data.csv\n", ROWS);
    return 0;
}

