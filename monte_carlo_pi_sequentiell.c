
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double rand_double() {
    return (double)rand() / RAND_MAX;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <number_of_points>\n", argv[0]);
        return 1;
    }

    long long num_points = atoll(argv[1]);
    if (num_points <= 0) {
        printf("Number of Points.\n");
        return 1;
    }

    srand(time(NULL));

    long long points_inside_circle = 0;

    clock_t start_time = clock();

    // Der Anteil der Punkte, die innerhalb des Kreises liegen, zu allen generierten Punkten (\text{points_inside_circle} / \text{num_points}) entspricht dem Verhältnis der Kreisfläche (\pi r^2) zur Quadratfläche (4r^2).
    for (long long i = 0; i < num_points; i++) {
        double x = rand_double() * 2.0 - 1.0;
        double y = rand_double() * 2.0 - 1.0;

        if (x * x + y * y <= 1.0) {
            points_inside_circle++;
        }
    }

    clock_t end_time = clock();
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    double pi_estimate = 4.0 * ((double)points_inside_circle / (double)num_points);

    printf("---------------------------------------------\n");
    printf("Number of points: %lld\n", num_points);
    printf("Points within the circle: %lld\n", points_inside_circle);
    printf("Estimated Value for π:  %.10f\n", pi_estimate);
    printf("Difference from actual π: %.10f\n", fabs(M_PI - pi_estimate));
    printf("Runtime: %.5f Seconds\n", time_spent);
    printf("\n\n\n");

    return 0;
}