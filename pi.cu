#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>

#define BLOCK_SIZE 256
#define GRID_SIZE  256
// Punkte pro Thread - kann an gewünschte Genauigkeit angepasst werden
#define POINTS_PER_THREAD 10000

__global__ void setup_kernel(curandState *state, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void pi_kernel(curandState *state, unsigned long long *countInside, long long pointsPerThread) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[id];

    unsigned long long inside = 0ULL;
    for (long long i = 0; i < pointsPerThread; i++) {
        float x = curand_uniform(&localState) * 2.0f - 1.0f;
        float y = curand_uniform(&localState) * 2.0f - 1.0f;
        float dist = x*x + y*y;
        if (dist <= 1.0f) {
            inside++;
        }
    }

    state[id] = localState;
    atomicAdd(countInside, inside);
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

    // Anzahl Threads gesamt
    int totalThreads = BLOCK_SIZE * GRID_SIZE;
    // Berechne wie viele Punkte jeder Thread erzeugen soll
    // Falls num_points nicht durch totalThreads teilbar ist, wird abgerundet.
    long long pointsPerThread = num_points / totalThreads;
    if (pointsPerThread <= 0) {
        // Falls zu wenige Punkte für die gewählte Thread-Konfiguration:
        // Einfach mindestens 1 Punkt pro Thread testen.
        pointsPerThread = 1;
    }

    // Timer für Laufzeitmessung
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    curandState *d_state;
    cudaMalloc(&d_state, totalThreads * sizeof(curandState));

    unsigned long long *d_countInside;
    cudaMalloc(&d_countInside, sizeof(unsigned long long));
    cudaMemset(d_countInside, 0, sizeof(unsigned long long));

    // Startzeit messen
    cudaEventRecord(start, 0);

    // RNG initialisieren
    setup_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_state, 1234ULL);
    cudaDeviceSynchronize();

    // Kernel zum Zählen der Punkte innerhalb des Kreises
    pi_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_state, d_countInside, pointsPerThread);
    cudaDeviceSynchronize();

    // Ende der Berechnung
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Zeit auslesen
    float time_spent = 0.0f;
    cudaEventElapsedTime(&time_spent, start, stop);

    unsigned long long h_countInside = 0ULL;
    cudaMemcpy(&h_countInside, d_countInside, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    double pi_estimate = 4.0 * (double)h_countInside / (double)(pointsPerThread * totalThreads);

    printf("Anzahl der Punkte: %lld\n", (long long)(pointsPerThread * totalThreads));
    printf("Punkte innerhalb des Kreises: %llu\n", h_countInside);
    printf("Geschätzter Wert von π: %.10f\n", pi_estimate);
    printf("Abweichung von tatsächlichem π: %.10f\n", fabs(M_PI - pi_estimate));
    // CUDA Event-Zeit ist in Millisekunden, Umrechnung in Sekunden
    printf("Laufzeit: %.5f Sekunden\n", time_spent / 1000.0f);

    cudaFree(d_state);
    cudaFree(d_countInside);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
