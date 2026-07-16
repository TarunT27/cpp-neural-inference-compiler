#include "nnc/nnc.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
  std::size_t threads{std::min<std::size_t>(
      64, std::max(2U, std::thread::hardware_concurrency()))};
  std::size_t warmup{10};
  std::size_t iterations{50};
  std::filesystem::path json{"artifacts/benchmark.json"};
};

struct Timing {
  double minimum_ms{};
  double mean_ms{};
  double median_ms{};
  double p95_ms{};
};

std::size_t parse_positive(const std::string& text, const char* option,
                           std::size_t maximum) {
  if (text.empty() || !std::all_of(text.begin(), text.end(), [](char value) {
        return value >= '0' && value <= '9';
      })) {
    throw std::invalid_argument(std::string(option) + " must contain only digits");
  }
  std::size_t consumed = 0;
  const auto value = std::stoull(text, &consumed);
  if (consumed != text.size() || value == 0 ||
      value > std::numeric_limits<std::size_t>::max() || value > maximum) {
    throw std::invalid_argument(std::string(option) +
                                " is outside the supported range");
  }
  return static_cast<std::size_t>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help") {
      std::cout << "Usage: nnc_benchmark [--threads N] [--warmup N] "
                   "[--iterations N] [--json PATH]\n";
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + argument);
    }
    const std::string value = argv[++index];
    if (argument == "--threads") {
      options.threads = parse_positive(value, "--threads", 64);
    } else if (argument == "--warmup") {
      options.warmup = parse_positive(value, "--warmup", 10000);
    } else if (argument == "--iterations") {
      options.iterations = parse_positive(value, "--iterations", 10000);
    } else if (argument == "--json") {
      options.json = value;
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  return options;
}

template <typename Function>
Timing measure(std::size_t warmup, std::size_t iterations, Function&& function) {
  for (std::size_t index = 0; index < warmup; ++index) {
    static_cast<void>(function());
  }
  std::vector<double> samples;
  samples.reserve(iterations);
  for (std::size_t index = 0; index < iterations; ++index) {
    const auto start = std::chrono::steady_clock::now();
    static_cast<void>(function());
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }
  std::sort(samples.begin(), samples.end());
  const auto percentile = [&](double fraction) {
    const auto position = static_cast<std::size_t>(
        fraction * static_cast<double>(samples.size() - 1));
    return samples[position];
  };
  return {samples.front(),
          std::accumulate(samples.begin(), samples.end(), 0.0) /
              static_cast<double>(samples.size()),
          percentile(0.50), percentile(0.95)};
}

void write_mode(std::ostream& stream, const char* name, const Timing& timing,
                std::size_t memory, std::size_t allocations,
                std::size_t workers) {
  stream << "    \"" << name << "\": {\n"
         << "      \"minimum_ms\": " << timing.minimum_ms << ",\n"
         << "      \"mean_ms\": " << timing.mean_ms << ",\n"
         << "      \"median_ms\": " << timing.median_ms << ",\n"
         << "      \"p95_ms\": " << timing.p95_ms << ",\n"
         << "      \"peak_activation_bytes\": " << memory << ",\n"
         << "      \"activation_buffer_allocations\": " << allocations << ",\n"
         << "      \"workers\": " << workers << "\n"
         << "    }";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    const auto graph = nnc::make_demo_graph(64);
    const auto inputs = nnc::make_demo_inputs(graph);
    const nnc::BaselineExecutor interpreter(graph);

    const auto compile_one_start = std::chrono::steady_clock::now();
    auto compiled_one = nnc::Compiler::compile(graph, {.threads = 1});
    const auto compile_one_end = std::chrono::steady_clock::now();
    const auto compile_many_start = std::chrono::steady_clock::now();
    auto compiled_many = nnc::Compiler::compile(graph, {.threads = options.threads});
    const auto compile_many_end = std::chrono::steady_clock::now();
    const auto compile_one_ms = std::chrono::duration<double, std::milli>(
                                    compile_one_end - compile_one_start)
                                    .count();
    const auto compile_many_ms = std::chrono::duration<double, std::milli>(
                                     compile_many_end - compile_many_start)
                                     .count();

    const auto baseline_result = interpreter.run(inputs);
    const auto one_result = compiled_one.run(inputs);
    const auto many_result = compiled_many.run(inputs);
    const auto one_error =
        nnc::max_abs_error(baseline_result.output.data, one_result.output.data);
    const auto many_error =
        nnc::max_abs_error(baseline_result.output.data, many_result.output.data);
    const auto maximum_error = std::max(one_error, many_error);
    if (maximum_error > 1.0e-5F) {
      throw std::runtime_error("optimized output failed correctness check");
    }

    const auto baseline = measure(options.warmup, options.iterations, [&] {
      return interpreter.run(inputs);
    });
    const auto one = measure(options.warmup, options.iterations,
                             [&] { return compiled_one.run(inputs); });
    const auto many = measure(options.warmup, options.iterations,
                              [&] { return compiled_many.run(inputs); });

    const auto speedup = baseline.median_ms / many.median_ms;
    const auto memory_reduction =
        100.0 * (1.0 - static_cast<double>(compiled_many.stats().arena_bytes) /
                           static_cast<double>(baseline_result.stats.peak_activation_bytes));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "NNC benchmark — 64x128 three-layer MLP\n";
    std::cout << "Mode                      p50 (ms)   p95 (ms)   memory (KiB)\n";
    std::cout << "Serial interpreter        " << std::setw(8)
              << baseline.median_ms << "   " << std::setw(8) << baseline.p95_ms
              << "   " << std::setw(12)
              << static_cast<double>(baseline_result.stats.peak_activation_bytes) /
                     1024.0
              << '\n';
    std::cout << "Compiled / 1 thread       " << std::setw(8) << one.median_ms
              << "   " << std::setw(8) << one.p95_ms << "   " << std::setw(12)
              << static_cast<double>(compiled_one.stats().arena_bytes) / 1024.0
              << '\n';
    std::cout << "Compiled / " << options.threads << " threads      "
              << std::setw(8) << many.median_ms << "   " << std::setw(8)
              << many.p95_ms << "   " << std::setw(12)
              << static_cast<double>(compiled_many.stats().arena_bytes) / 1024.0
              << '\n';
    std::cout << "\nOperations: " << graph.nodes().size() << " -> "
              << compiled_many.stats().compiled_nodes << " | speedup: "
              << speedup << "x | activation memory: -" << memory_reduction
              << "% | max error: " << std::scientific << maximum_error << '\n';

    if (!options.json.parent_path().empty()) {
      std::filesystem::create_directories(options.json.parent_path());
    }
    std::ofstream report(options.json);
    if (!report) {
      throw std::runtime_error("could not open JSON report path");
    }
    report << std::fixed << std::setprecision(6);
    report << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"model\": {\"batch\": 64, \"input\": 128, \"layers\": [256, 128, 64]},\n"
           << "  \"configuration\": {\"warmup\": " << options.warmup
           << ", \"iterations\": " << options.iterations
           << ", \"requested_threads\": " << options.threads << "},\n"
           << "  \"compile_ms\": {\"compiled_1_thread\": " << compile_one_ms
           << ", \"compiled_multi_thread\": " << compile_many_ms << "},\n"
           << "  \"graph\": {\"operations_before\": " << graph.nodes().size()
           << ", \"operations_after\": " << compiled_many.stats().compiled_nodes
           << ", \"fused_linear_relu\": "
           << compiled_many.stats().fused_linear_relu << "},\n"
           << "  \"results\": {\n";
    write_mode(report, "serial_interpreter", baseline,
               baseline_result.stats.peak_activation_bytes,
               baseline_result.stats.activation_buffer_allocations, 1);
    report << ",\n";
    write_mode(report, "compiled_1_thread", one,
               compiled_one.stats().arena_bytes,
               one_result.stats.activation_buffer_allocations, 1);
    report << ",\n";
    write_mode(report, "compiled_multi_thread", many,
               compiled_many.stats().arena_bytes,
               many_result.stats.activation_buffer_allocations,
               many_result.stats.workers_used);
    report << "\n  },\n"
           << "  \"summary\": {\"speedup\": " << std::fixed << speedup
           << ", \"memory_reduction_percent\": " << memory_reduction
           << ", \"max_abs_error\": " << std::scientific << maximum_error
           << "}\n}\n";
    report.flush();
    if (!report) {
      throw std::runtime_error("failed while writing JSON report");
    }
    std::cout << "JSON report: " << options.json.string() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "nnc_benchmark: " << error.what() << '\n';
    return 2;
  }
}
