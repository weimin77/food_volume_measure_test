cmake_minimum_required(VERSION 3.28)
project(example CXX)

find_package(food_volume_measure REQUIRED)

add_executable(example main.cpp)
target_compile_features(example PRIVATE cxx_std_17)
target_link_libraries(example PRIVATE food_volume_measure::food_volume_measure)

add_executable(bench bench.cpp)
target_compile_features(bench PRIVATE cxx_std_17)
target_link_libraries(bench PRIVATE food_volume_measure::food_volume_measure)
