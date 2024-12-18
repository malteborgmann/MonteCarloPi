cmake_minimum_required(VERSION 3.10)

# Projektname und Sprachen setzen (inkl. CUDA)
project(MonteCarloPi C CUDA)

# Hinzufügen der Unsupported-Compiler-Option für NVCC
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler -ccbin /opt/f39/bin/g++")

# C-Standard festlegen
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Finden von OpenMP
find_package(OpenMP)

# Sequentielle Version hinzufügen
add_executable(monte_carlo_pi_sequentiell monte_carlo_pi_sequentiell.c)

# Parallele Version hinzufügen
add_executable(monte_carlo_pi_parallel monte_carlo_pi_parallel.c)

# Wenn OpenMP gefunden wurde, es zur parallelen Version hinzufügen
if(OpenMP_C_FOUND)
    target_link_libraries(monte_carlo_pi_parallel PUBLIC OpenMP::OpenMP_C)
    message(STATUS "OpenMP wurde gefunden und verlinkt.")
else()
    message(WARNING "OpenMP wurde nicht gefunden. Die parallele Version wird möglicherweise nicht korrekt kompiliert.")
endif()

# CUDA-Version hinzufügen
add_executable(monte_carlo_pi_cuda monte_carlo_pi_cuda.cu)
target_link_libraries(monte_carlo_pi_cuda PRIVATE cudart curand)
set_target_properties(monte_carlo_pi_cuda PROPERTIES 
    CUDA_STANDARD 11 
    CUDA_SEPARABLE_COMPILATION ON
)

# Variable für die Anzahl der Punkte definieren
set(NUM_POINTS 1000000000 CACHE STRING "Anzahl der zu simulierenden Punkte")

# Benutzerdefiniertes Ziel zum Ausführen aller Programme
add_custom_target(run_all
    COMMAND monte_carlo_pi_sequentiell ${NUM_POINTS}
    COMMAND monte_carlo_pi_parallel ${NUM_POINTS}
    COMMAND monte_carlo_pi_cuda ${NUM_POINTS}
    DEPENDS monte_carlo_pi_sequentiell monte_carlo_pi_parallel monte_carlo_pi_cuda
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Führe alle Monte-Carlo Pi Simulationen aus..."
)
