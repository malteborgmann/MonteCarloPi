#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Funktion zur Generierung einer zufälligen Gleitkommazahl zwischen 0 und 1
double rand_double() {
    return (double)rand() / RAND_MAX;
}

int main(int argc, char *argv[]) {
    // Überprüfen, ob die Anzahl der Punkte als Argument übergeben wurde
    if (argc != 2) {
        printf("Usage: %s <number_of_points>\n", argv[0]);
        return 1;
    }

    // Anzahl der Punkte aus den Kommandozeilenargumenten lesen
    long long num_points = atoll(argv[1]);
    if (num_points <= 0) {
        printf("Bitte geben Sie eine positive Anzahl von Punkten ein.\n");
        return 1;
    }

    // Initialisieren des Zufallszahlengenerators mit dem aktuellen Zeitstempel
    srand(time(NULL));

    long long points_inside_circle = 0;

    for (long long i = 0; i < num_points; i++) {
        // Generiere zufällige x und y Koordinaten zwischen -1 und 1
        double x = rand_double() * 2.0 - 1.0;
        double y = rand_double() * 2.0 - 1.0;

        // Überprüfen, ob der Punkt innerhalb des Kreises liegt
        if (x * x + y * y <= 1.0) {
            points_inside_circle++;
        }
    }

    // Schätzung von π berechnen
    double pi_estimate = 4.0 * ((double)points_inside_circle / (double)num_points);

    // Ergebnisse ausgeben
    printf("Anzahl der Punkte: %lld\n", num_points);
    printf("Punkte innerhalb des Kreises: %lld\n", points_inside_circle);
    printf("Geschätzter Wert von π: %.10f\n", pi_estimate);
    printf("Abweichung von tatsächlichem π: %.10f\n", fabs(M_PI - pi_estimate));

    return 0;
}