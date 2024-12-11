#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

double rand_double(unsigned int *seed) {
    return (double)rand_r(seed) / RAND_MAX;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <number_of_points>\n", argv[0]);
        return 1;
    }

    long long num_points = atoll(argv[1]);
    if (num_points <= 0) {
        printf("Anzahl Punkte.\n");
        return 1;
    }

    srand(time(NULL));

    long long points_inside_circle = 0;

    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        unsigned int seed = time(NULL) ^ omp_get_thread_num();

        long long local_count = 0;

        #pragma omp for nowait
        for (long long i = 0; i < num_points; i++) {
            double x = rand_double(&seed) * 2.0 - 1.0;
            double y = rand_double(&seed) * 2.0 - 1.0;

            if (x * x + y * y <= 1.0) {
                local_count++;
            }
        }

        #pragma omp atomic
        points_inside_circle += local_count;
    }

    double end_time = omp_get_wtime();
    double time_spent = end_time - start_time;

    double pi_estimate = 4.0 * ((double)points_inside_circle / (double)num_points);

    printf("Punkte innerhalb des Kreises: %lld\n", points_inside_circle);
    printf("Geschätzter Wert von π: %.10f\n", pi_estimate);
    printf("Abweichung von tatsächlichem π: %.10f\n", fabs(M_PI - pi_estimate));
    printf("Laufzeit: %.5f Sekunden\n", time_spent);

    return 0;
}