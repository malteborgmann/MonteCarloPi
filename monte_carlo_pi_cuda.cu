#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

// CUDA-Kernel zur Generierung und Überprüfung von Punkten
__global__ void monte_carlo_pi_kernel(long long num_points, unsigned int seed, long long *count) {
    // Berechne die globale Thread-ID
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    // Lokaler Zähler
    long long local_count = 0;

    // Jede Thread verarbeitet mehrere Punkte
    for (long long i = idx; i < num_points; i += stride) {
        // Einfache LCG (Linear Congruential Generator) für Zufallszahlen
        // Parameter für LCG
        unsigned int a = 1664525;
        unsigned int c = 1013904223;
        seed = a * seed + c;

        // Generiere zufällige x und y zwischen -1 und 1
        double x = ((double)(seed >> 16) / 65535.0) * 2.0 - 1.0;
        seed = a * seed + c;
        double y = ((double)(seed >> 16) / 65535.0) * 2.0 - 1.0;

        // Überprüfe, ob der Punkt innerhalb des Kreises liegt
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }

    // Atomare Addition des lokalen Zählers zum globalen Zähler
    atomicAdd(count, local_count);
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

    // Host-Zähler initialisieren
    long long h_count = 0;

    // Device-Zähler initialisieren
    long long *d_count;
    cudaMalloc((void **)&d_count, sizeof(long long));
    cudaMemcpy(d_count, &h_count, sizeof(long long), cudaMemcpyHostToDevice);

    // Seed für den Zufallszahlengenerator
    unsigned int seed = (unsigned int)time(NULL);

    // Anzahl der Threads und Blocks festlegen
    int threads_per_block = 256;
    int blocks = 1024;

    // Startzeit für die Messung der Laufzeit
    clock_t start_time = clock();

    // CUDA-Kernel aufrufen
    monte_carlo_pi_kernel<<<blocks, threads_per_block>>>(num_points, seed, d_count);

    // Warte auf die Fertigstellung des Kernels
    cudaDeviceSynchronize();

    // Endzeit nach der Berechnung
    clock_t end_time = clock();
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Ergebnis vom Device zum Host kopieren
    cudaMemcpy(&h_count, d_count, sizeof(long long), cudaMemcpyDeviceToHost);

    // Schätzung von π berechnen
    double pi_estimate = 4.0 * ((double)h_count / (double)num_points);

    // Ergebnisse ausgeben
    printf("Anzahl der Punkte: %lld\n", num_points);
    printf("Punkte innerhalb des Kreises: %lld\n", h_count);
    printf("Geschätzter Wert von π: %.10f\n", pi_estimate);
    printf("Abweichung von tatsächlichem π: %.10f\n", fabs(M_PI - pi_estimate));
    printf("Laufzeit: %.5f Sekunden\n", time_spent);

    // Speicher freigeben
    cudaFree(d_count);

    return 0;
}
