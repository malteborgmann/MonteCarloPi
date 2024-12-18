#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <ctime>

#define BLOCK_SIZE 256
#define GRID_SIZE  256

__global__ void setup_kernel(curandState *state, unsigned long long base_seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Jeder Thread erhält einen eigenen Seed basierend auf base_seed und id.
    unsigned long long seed = base_seed ^ (unsigned long long)id;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void pi_kernel(curandState *state, unsigned long long *countInside, long long totalPoints) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    curandState localState = state[id];
    unsigned long long inside = 0ULL;

    // Jeder Thread springt in Schritten von totalThreads durch den Input-Bereich
    for (long long i = id; i < totalPoints; i += totalThreads) {
        double x = curand_uniform_double(&localState) * 2.0 - 1.0;
        double y = curand_uniform_double(&localState) * 2.0 - 1.0;
        double dist = x*x + y*y;

        if (dist <= 1.0) {
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
        printf("Number of Points must be positive.\n");
        return 1;
    }

    int totalThreads = BLOCK_SIZE * GRID_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    curandState *d_state;
    cudaMalloc(&d_state, totalThreads * sizeof(curandState));

    unsigned long long *d_countInside;
    cudaMalloc(&d_countInside, sizeof(unsigned long long));
    cudaMemset(d_countInside, 0, sizeof(unsigned long long));

    cudaEventRecord(start, 0);

    // Seed basiert auf aktueller Zeit
    unsigned long long seed = (unsigned long long)time(NULL);
    setup_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_state, seed);
    cudaDeviceSynchronize();

    pi_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_state, d_countInside, num_points);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float time_spent = 0.0f;
    cudaEventElapsedTime(&time_spent, start, stop);

    unsigned long long h_countInside = 0ULL;
    cudaMemcpy(&h_countInside, d_countInside, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    double pi_estimate = 4.0 * (double)h_countInside / (double)num_points;

    printf("---------------------------------------------\n");
    printf("Number of points: %lld\n", num_points);
    printf("Points within the circle: %llu\n", h_countInside);
    printf("Estimated Value for π: %.10f\n", pi_estimate);
    printf("Difference from actual π: %.10f\n", fabs(M_PI - pi_estimate));
    printf("Runtime: %.5f Seconds\n", time_spent / 1000.0f);
    printf("\n\n\n");

    cudaFree(d_state);
    cudaFree(d_countInside);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
