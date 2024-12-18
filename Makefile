# Verwenden Sie gcc für C-Code und nvcc für CUDA
CC = gcc
NVCC = nvcc --allow-unsupported-compiler -ccbin /opt/f39/bin/g++ 

# OpenMP-Flags für die parallele Version
OPENMP_FLAGS = -fopenmp

# Anzahl der zu simulierenden Punkte
NUM_POINTS = 1000000000

# Ausführbare Dateien
EXE_SEQ = monte_carlo_pi_sequentiell
EXE_PAR = monte_carlo_pi_parallel
EXE_CUDA = monte_carlo_pi_cuda

# Quellcode
SRC_SEQ = monte_carlo_pi_sequentiell.c
SRC_PAR = monte_carlo_pi_parallel.c
SRC_CUDA = monte_carlo_pi_cuda.cu

.PHONY: all clean run_all

all: $(EXE_SEQ) $(EXE_PAR) $(EXE_CUDA)

$(EXE_SEQ): $(SRC_SEQ)
	$(CC) -o $@ $<

$(EXE_PAR): $(SRC_PAR)
	$(CC) $(OPENMP_FLAGS) -o $@ $<

$(EXE_CUDA): $(SRC_CUDA)
	$(NVCC) -lcudart -lcurand -o $@ $<

run_all: all
	./$(EXE_SEQ) $(NUM_POINTS)
	./$(EXE_PAR) $(NUM_POINTS)
	./$(EXE_CUDA) $(NUM_POINTS)

clean:
	rm -f $(EXE_SEQ) $(EXE_PAR) $(EXE_CUDA)
