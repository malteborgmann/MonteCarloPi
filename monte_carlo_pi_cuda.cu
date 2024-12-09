#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>

// CUDA-Kernel zur Initialisierung von cuRAND
__global__ void init_curand(curandState *state, unsigned long seed, long long num_points) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        // Initialisiere den cuRAND Zustand mit dem Seed, der Thread-ID und der Position
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// CUDA-Kernel zur Generierung und Überprüfung von Punkten mit cuRAND
__global__ void monte_carlo_pi_kernel(long long num_points, curandState *state, unsigned long long *count) {
    extern __shared__ unsigned long long shared_count[];
    unsigned int tid = threadIdx.x;
    shared_count[tid] = 0;
    __syncthreads();

    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for (long long i = idx; i < num_points; i += stride) {
        // Generiere zufällige x und y zwischen -1 und 1
        double x = 2.0 * curand_uniform_double(&state[i]) - 1.0;
        double y = 2.0 * curand_uniform_double(&state[i]) - 1.0;

        // Überprüfe, ob der Punkt innerhalb des Kreises liegt
        if (x * x + y * y <= 1.0) {
            shared_count[tid]++;
        }
    }

    __syncthreads();

    // Reduziere die lokalen Zähler innerhalb des Blocks
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_count[tid] += shared_count[tid + s];
        }
        __syncthreads();
    }

    // Atomare Addition nur einmal pro Block
    if (tid == 0) {
        atomicAdd(count, shared_count[0]);
    }
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
    unsigned long long h_count = 0;

    // Device-Zähler initialisieren
    unsigned long long *d_count;
    cudaError_t err = cudaMalloc((void **)&d_count, sizeof(unsigned long long));
    if (err != cudaSuccess) {
        printf("CUDA Error (cudaMalloc): %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemcpy(d_count, &h_count, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error (cudaMemcpy H2D): %s\n", cudaGetErrorString(err));
        cudaFree(d_count);
        return 1;
    }

    // cuRAND-Zustände initialisieren
    curandState *d_state;
    err = cudaMalloc((void **)&d_state, num_points * sizeof(curandState));
    if (err != cudaSuccess) {
        printf("CUDA Error (cudaMalloc d_state): %s\n", cudaGetErrorString(err));
        cudaFree(d_count);
        return 1;
    }

    // Anzahl der Threads und Blocks festlegen
    int threads_per_block = 256;
    int blocks = (num_points + threads_per_block - 1) / threads_per_block;
    // Begrenze die Anzahl der Blocks, um die maximale Anzahl zu vermeiden
    blocks = min(blocks, 1024);

    // Seed für cuRAND
    unsigned long seed = (unsigned long)time(NULL);

    // Startzeit für die Messung der Laufzeit mit CUDA-Ereignissen
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Initialisiere cuRAND-Zustände
    init_curand<<<blocks, threads_per_block>>>(d_state, seed, num_points);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (init_curand): %s\n", cudaGetErrorString(err));
        cudaFree(d_count);
        cudaFree(d_state);
        return 1;
    }
    cudaDeviceSynchronize();

    // Starte den Monte-Carlo-Kernel mit cuRAND
    size_t shared_mem_size = threads_per_block * sizeof(unsigned long long);
    monte_carlo_pi_kernel<<<blocks, threads_per_block, shared_mem_size>>>(num_points, d_state, d_count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (monte_carlo_pi_kernel): %s\n", cudaGetErrorString(err));
        cudaFree(d_count);
        cudaFree(d_state);
        return 1;
    }
    cudaDeviceSynchronize();

    // Endzeit nach der Berechnung
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_spent_ms;
    cudaEventElapsedTime(&time_spent_ms, start, stop);

    // Ergebnis vom Device zum Host kopieren
    err = cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA Error (cudaMemcpy D2H): %s\n", cudaGetErrorString(err));
        cudaFree(d_count);
        cudaFree(d_state);
        return 1;
    }

    // Schätzung von π berechnen
    double pi_estimate = 4.0 * ((double)h_count / (double)num_points);

    // Ergebnisse ausgeben
    printf("Anzahl der Punkte: %lld\n", num_points);
    printf("Punkte innerhalb des Kreises: %llu\n", h_count);
    printf("Geschätzter Wert von π: %.10f\n", pi_estimate);
    printf("Abweichung von tatsächlichem π: %.10f\n", fabs(M_PI - pi_estimate));
    printf("Laufzeit: %.5f Sekunden\n", time_spent_ms / 1000.0f); // CUDA gibt Zeit in Millisekunden

    // Speicher freigeben
    cudaFree(d_count);
    cudaFree(d_state);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
