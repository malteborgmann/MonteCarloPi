# Mindestversion von CMake festlegen
cmake_minimum_required(VERSION 3.10)

# Projektname und Sprache
project(MonteCarloPi C)

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

# Benutzerdefiniertes Ziel zum Ausführen beider Programme
add_custom_target(run_all
        COMMAND monte_carlo_pi_sequentiell 1000000000
        COMMAND monte_carlo_pi_parallel 1000000000
        DEPENDS monte_carlo_pi_sequentiell monte_carlo_pi_parallel
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Führe beide Monte-Carlo Pi Simulationen aus..."
)