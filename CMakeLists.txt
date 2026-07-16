cmake_minimum_required(VERSION 3.20)
project(nnc VERSION 1.0.0 LANGUAGES CXX)

find_package(Threads REQUIRED)

option(NNC_ENABLE_COVERAGE "Enable GCC/Clang coverage instrumentation" OFF)
option(NNC_ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

function(nnc_enable_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
endfunction()

add_library(nnc src/nnc.cpp)
target_include_directories(nnc PUBLIC include)
target_link_libraries(nnc PUBLIC Threads::Threads)
nnc_enable_warnings(nnc)

if(NOT MSVC)
  if(NNC_ENABLE_COVERAGE)
    target_compile_options(nnc PUBLIC --coverage -O0 -g)
    target_link_options(nnc PUBLIC --coverage)
  endif()
  if(NNC_ENABLE_SANITIZERS AND NOT (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    target_compile_options(nnc PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(nnc PUBLIC -fsanitize=address,undefined)
  elseif(NNC_ENABLE_SANITIZERS)
    message(WARNING "Sanitizers requested, but this MinGW distribution does not ship ASan/UBSan runtimes")
  endif()
elseif(NNC_ENABLE_SANITIZERS)
  message(WARNING "Sanitizers requested, but this MSVC configuration does not enable ASan/UBSan")
endif()

add_executable(nnc_demo apps/demo.cpp)
target_link_libraries(nnc_demo PRIVATE nnc)
nnc_enable_warnings(nnc_demo)

add_executable(nnc_benchmark benchmarks/benchmark.cpp)
target_link_libraries(nnc_benchmark PRIVATE nnc)
nnc_enable_warnings(nnc_benchmark)

enable_testing()
add_executable(nnc_tests tests/test_main.cpp)
target_link_libraries(nnc_tests PRIVATE nnc)
nnc_enable_warnings(nnc_tests)
add_test(NAME nnc_unit_and_integration COMMAND nnc_tests)
add_test(NAME nnc_demo_smoke COMMAND nnc_demo)

add_test(
  NAME nnc_cli_e2e
  COMMAND ${CMAKE_COMMAND}
    -DBENCHMARK=$<TARGET_FILE:nnc_benchmark>
    -DREPORT=${CMAKE_CURRENT_BINARY_DIR}/e2e-report.json
    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cli.cmake)
