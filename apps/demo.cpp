#include "nnc/nnc.hpp"

#include <iomanip>
#include <iostream>

int main() {
  try {
    const auto graph = nnc::make_demo_graph(8);
    const auto inputs = nnc::make_demo_inputs(graph);
    const auto baseline = nnc::BaselineExecutor::run(graph, inputs);
    auto compiled = nnc::Compiler::compile(graph, {.threads = 4});
    const auto optimized = compiled.run(inputs);

    std::cout << "NNC — C++ Neural Inference Compiler & Runtime\n\n";
    std::cout << "Graph: [8,128] -> [8,256] -> [8,128] -> [8,64]\n";
    std::cout << "Operations: " << compiled.stats().original_nodes << " -> "
              << compiled.stats().compiled_nodes << " ("
              << compiled.stats().fused_linear_relu << " fused kernels)\n";
    std::cout << "Activation memory: " << baseline.stats.peak_activation_bytes
              << " -> " << optimized.stats.peak_activation_bytes << " bytes\n";
    std::cout << "Activation-buffer allocations per run: "
              << optimized.stats.activation_buffer_allocations
              << "\nWorkers used: " << optimized.stats.workers_used << '\n';
    std::cout << "Max absolute error: " << std::scientific
              << nnc::max_abs_error(baseline.output.data, optimized.output.data)
              << "\n\nCompiled plan:\n";
    for (std::size_t index = 0; index < compiled.instructions().size(); ++index) {
      const auto& instruction = compiled.instructions()[index];
      const auto& buffer = compiled.memory_plan()[index];
      std::cout << "  " << index << ": " << nnc::op_name(instruction.kind)
                << " -> value " << instruction.output << " @ arena+"
                << buffer.offset << " (" << buffer.bytes << " bytes)\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "nnc_demo: " << error.what() << '\n';
    return 1;
  }
}
