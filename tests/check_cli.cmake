execute_process(
  COMMAND "${BENCHMARK}" --threads 2 --warmup 1 --iterations 3 --json "${REPORT}"
  RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "benchmark CLI failed with ${result}")
endif()
file(READ "${REPORT}" report)
string(JSON schema GET "${report}" schema_version)
string(JSON operations_before GET "${report}" graph operations_before)
string(JSON operations_after GET "${report}" graph operations_after)
string(JSON baseline_median GET "${report}" results serial_interpreter median_ms)
string(JSON optimized_median GET "${report}" results compiled_multi_thread median_ms)
string(JSON baseline_memory GET "${report}" results serial_interpreter peak_activation_bytes)
string(JSON optimized_memory GET "${report}" results compiled_multi_thread peak_activation_bytes)
string(JSON maximum_error GET "${report}" summary max_abs_error)

if(NOT schema EQUAL 1 OR NOT operations_before EQUAL 9 OR NOT operations_after EQUAL 3)
  message(FATAL_ERROR "benchmark JSON has unexpected graph metadata")
endif()
if(NOT baseline_median GREATER 0 OR NOT optimized_median GREATER 0)
  message(FATAL_ERROR "benchmark medians must be positive")
endif()
if(optimized_memory GREATER baseline_memory)
  message(FATAL_ERROR "compiled activation arena exceeds baseline activations")
endif()
if(maximum_error GREATER 0.00001)
  message(FATAL_ERROR "compiled output exceeds correctness tolerance")
endif()

foreach(invalid_args IN ITEMS "--iterations;-1" "--warmup;10001" "--unknown;1")
  execute_process(
    COMMAND "${BENCHMARK}" ${invalid_args}
    RESULT_VARIABLE invalid_result
    OUTPUT_QUIET ERROR_QUIET)
  if(invalid_result EQUAL 0)
    message(FATAL_ERROR "benchmark accepted invalid arguments: ${invalid_args}")
  endif()
endforeach()

execute_process(COMMAND "${BENCHMARK}" --help RESULT_VARIABLE help_result
                OUTPUT_QUIET ERROR_QUIET)
if(NOT help_result EQUAL 0)
  message(FATAL_ERROR "benchmark --help failed")
endif()
