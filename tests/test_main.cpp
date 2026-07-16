#include "nnc/nnc.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct TestCase {
  std::string name;
  std::function<void()> body;
};

std::vector<TestCase>& tests() {
  static std::vector<TestCase> registry;
  return registry;
}

struct Register {
  Register(std::string name, std::function<void()> body) {
    tests().push_back({std::move(name), std::move(body)});
  }
};

#define NNC_JOIN_IMPL(left, right) left##right
#define NNC_JOIN(left, right) NNC_JOIN_IMPL(left, right)
#define TEST(name)                                                           \
  static void NNC_JOIN(test_, __LINE__)();                                  \
  static Register NNC_JOIN(register_, __LINE__)(name, NNC_JOIN(test_, __LINE__)); \
  static void NNC_JOIN(test_, __LINE__)()
#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      throw std::runtime_error(std::string("check failed: ") + #condition); \
    }                                                                        \
  } while (false)
#define CHECK_NEAR(actual, expected, tolerance) \
  CHECK(std::fabs((actual) - (expected)) <= (tolerance))

nnc::Graph one_layer_graph() {
  nnc::GraphBuilder builder;
  const auto x = builder.input("x", {2, 2});
  const auto w = builder.constant("w", {{2, 2}, {2.0F, 1.0F, -1.0F, 3.0F}});
  const auto b = builder.constant("b", {{2}, {0.5F, 1.0F}});
  const auto y = builder.relu(builder.bias_add(builder.matmul(x, w), b));
  builder.output(y);
  return std::move(builder).build();
}

TEST("reference executor computes MatMul + BiasAdd + ReLU") {
  const auto graph = one_layer_graph();
  const nnc::Inputs inputs{{graph.inputs().front(), {{2, 2}, {1.0F, -2.0F, 3.0F, 4.0F}}}};
  const auto result = nnc::BaselineExecutor::run(graph, inputs);
  const std::vector<float> expected{4.5F, 0.0F, 2.5F, 16.0F};
  CHECK(result.output.shape == nnc::Shape({2, 2}));
  CHECK(result.output.data.size() == expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    CHECK_NEAR(result.output.data[index], expected[index], 1.0e-5F);
  }
}

TEST("compiler fuses a linear activation and preserves results") {
  const auto graph = one_layer_graph();
  const nnc::Inputs inputs{{graph.inputs().front(), {{2, 2}, {1.0F, -2.0F, 3.0F, 4.0F}}}};
  const auto expected = nnc::BaselineExecutor::run(graph, inputs);
  auto model = nnc::Compiler::compile(graph, {.threads = 1});
  const auto actual = model.run(inputs);
  CHECK(model.stats().original_nodes == 3);
  CHECK(model.stats().compiled_nodes == 1);
  CHECK(model.stats().fused_linear_relu == 1);
  CHECK(nnc::max_abs_error(expected.output.data, actual.output.data) <= 1.0e-5F);
}

TEST("liveness planner reuses buffers on the three-layer demo") {
  const auto graph = nnc::make_demo_graph(8);
  auto model = nnc::Compiler::compile(graph, {.threads = 1});
  CHECK(model.stats().original_nodes == 9);
  CHECK(model.stats().compiled_nodes == 3);
  CHECK(model.stats().fused_linear_relu == 3);
  CHECK(model.stats().reused_values >= 1);
  CHECK(model.stats().arena_bytes < model.stats().naive_activation_bytes);
  const auto& plan = model.memory_plan();
  CHECK(plan.size() == 3);
  CHECK(plan.front().offset == plan.back().offset);
}

TEST("four-thread execution is deterministic and equivalent") {
  const auto graph = nnc::make_demo_graph(32);
  const auto inputs = nnc::make_demo_inputs(graph, 7);
  auto serial = nnc::Compiler::compile(graph, {.threads = 1});
  auto parallel = nnc::Compiler::compile(graph, {.threads = 4});
  const auto expected = serial.run(inputs);
  const auto first = parallel.run(inputs);
  const auto second = parallel.run(inputs);
  CHECK(nnc::max_abs_error(expected.output.data, first.output.data) <= 1.0e-5F);
  CHECK(first.output.data == second.output.data);
  CHECK(first.stats.parallel_tasks >= 4);
  CHECK(first.stats.workers_used == 4);
}

TEST("invalid shapes and runtime inputs fail fast") {
  bool bad_tensor = false;
  try {
    static_cast<void>(nnc::Tensor({2, 2}, {1.0F}));
  } catch (const std::invalid_argument&) {
    bad_tensor = true;
  }
  CHECK(bad_tensor);

  const auto graph = one_layer_graph();
  bool missing_input = false;
  try {
    static_cast<void>(nnc::BaselineExecutor::run(graph, {}));
  } catch (const std::invalid_argument&) {
    missing_input = true;
  }
  CHECK(missing_input);

  bool wrong_shape = false;
  try {
    const nnc::Inputs inputs{{graph.inputs().front(), {{1, 4}, {1, 2, 3, 4}}}};
    static_cast<void>(nnc::BaselineExecutor::run(graph, inputs));
  } catch (const std::invalid_argument&) {
    wrong_shape = true;
  }
  CHECK(wrong_shape);
}

TEST("builder rejects incompatible matrix dimensions") {
  bool rejected = false;
  try {
    nnc::GraphBuilder builder;
    const auto left = builder.input("left", {2, 3});
    const auto right = builder.constant("right", {{4, 2}, std::vector<float>(8, 1.0F)});
    static_cast<void>(builder.matmul(left, right));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}

TEST("unfused execution and unique-buffer mode preserve output") {
  const auto graph = one_layer_graph();
  const nnc::Inputs inputs{{graph.inputs().front(), {{2, 2}, {1.0F, -2.0F, 3.0F, 4.0F}}}};
  const auto expected = nnc::BaselineExecutor::run(graph, inputs);
  auto model = nnc::Compiler::compile(
      graph, {.enable_fusion = false, .reuse_buffers = false, .threads = 1});
  const auto actual = model.run(inputs);
  CHECK(model.stats().compiled_nodes == 3);
  CHECK(model.stats().fused_linear_relu == 0);
  CHECK(model.stats().reused_values == 0);
  CHECK(model.memory_plan()[0].offset != model.memory_plan()[1].offset);
  CHECK(model.memory_plan()[1].offset != model.memory_plan()[2].offset);
  CHECK(nnc::max_abs_error(expected.output.data, actual.output.data) <= 1.0e-5F);
}

TEST("fusion is skipped when an intermediate is the graph output") {
  nnc::GraphBuilder builder;
  const auto x = builder.input("x", {1, 2});
  const auto w = builder.constant("w", {{2, 2}, {1, 0, 0, 1}});
  const auto b = builder.constant("b", {{2}, {0, 0}});
  const auto product = builder.matmul(x, w);
  const auto biased = builder.bias_add(product, b);
  static_cast<void>(builder.relu(biased));
  builder.output(product);
  const auto graph = std::move(builder).build();
  auto model = nnc::Compiler::compile(graph);
  CHECK(model.stats().fused_linear_relu == 0);
  CHECK(model.stats().compiled_nodes == 3);
}

TEST("builder validates names, dimensions, bias shapes, and outputs") {
  bool zero_dimension = false;
  try {
    nnc::GraphBuilder builder;
    static_cast<void>(builder.input("zero", {0, 2}));
  } catch (const std::invalid_argument&) {
    zero_dimension = true;
  }
  CHECK(zero_dimension);

  bool duplicate_name = false;
  try {
    nnc::GraphBuilder builder;
    static_cast<void>(builder.input("same", {1, 2}));
    static_cast<void>(builder.input("same", {1, 2}));
  } catch (const std::invalid_argument&) {
    duplicate_name = true;
  }
  CHECK(duplicate_name);

  bool bad_bias = false;
  try {
    nnc::GraphBuilder builder;
    const auto x = builder.input("x", {1, 2});
    const auto b = builder.constant("b", {{3}, {1, 2, 3}});
    static_cast<void>(builder.bias_add(x, b));
  } catch (const std::invalid_argument&) {
    bad_bias = true;
  }
  CHECK(bad_bias);

  bool no_output = false;
  try {
    nnc::GraphBuilder builder;
    static_cast<void>(builder.input("x", {1}));
    static_cast<void>(std::move(builder).build());
  } catch (const std::invalid_argument&) {
    no_output = true;
  }
  CHECK(no_output);
}

TEST("compiler and comparison utilities reject invalid requests") {
  const auto graph = one_layer_graph();
  bool too_many_threads = false;
  try {
    static_cast<void>(nnc::Compiler::compile(graph, {.threads = 65}));
  } catch (const std::invalid_argument&) {
    too_many_threads = true;
  }
  CHECK(too_many_threads);

  bool different_sizes = false;
  try {
    static_cast<void>(nnc::max_abs_error(std::vector<float>{1},
                                         std::vector<float>{1, 2}));
  } catch (const std::invalid_argument&) {
    different_sizes = true;
  }
  CHECK(different_sizes);

  CHECK(std::string(nnc::op_name(nnc::OpKind::MatMul)) == "MatMul");
  CHECK(std::string(nnc::op_name(nnc::OpKind::BiasAdd)) == "BiasAdd");
  CHECK(std::string(nnc::op_name(nnc::OpKind::Relu)) == "ReLU");
  CHECK(std::string(nnc::op_name(nnc::OpKind::FusedLinearRelu)) ==
        "FusedLinearReLU");

  const auto nan = std::numeric_limits<float>::quiet_NaN();
  CHECK(std::isinf(nnc::max_abs_error(std::vector<float>{nan},
                                      std::vector<float>{0.0F})));
  CHECK(std::isinf(nnc::max_abs_error(
      std::vector<float>{std::numeric_limits<float>::infinity()},
      std::vector<float>{std::numeric_limits<float>::infinity()})));
}

TEST("compiled runtime rejects a binding to a constant") {
  const auto graph = one_layer_graph();
  auto model = nnc::Compiler::compile(graph);
  nnc::Inputs bindings;
  bindings.emplace(graph.inputs().front(),
                   nnc::Tensor({2, 2}, {1, 2, 3, 4}));
  bindings.emplace(1, nnc::Tensor({2, 2}, {1, 0, 0, 1}));
  bool rejected = false;
  try {
    static_cast<void>(model.run(bindings));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}

TEST("executors reject a tensor mutated to an undersized payload") {
  const auto graph = one_layer_graph();
  nnc::Tensor corrupted;
  corrupted.shape = {2, 2};
  corrupted.data = {1.0F};
  const nnc::Inputs inputs{{graph.inputs().front(), corrupted}};

  bool baseline_rejected = false;
  try {
    static_cast<void>(nnc::BaselineExecutor::run(graph, inputs));
  } catch (const std::invalid_argument&) {
    baseline_rejected = true;
  }
  CHECK(baseline_rejected);

  auto model = nnc::Compiler::compile(graph);
  bool compiled_rejected = false;
  try {
    static_cast<void>(model.run(inputs));
  } catch (const std::invalid_argument&) {
    compiled_rejected = true;
  }
  CHECK(compiled_rejected);
}

TEST("builder rejects a constant mutated to an undersized payload") {
  nnc::Tensor corrupted;
  corrupted.shape = {2, 2};
  corrupted.data = {1.0F};
  bool rejected = false;
  try {
    nnc::GraphBuilder builder;
    static_cast<void>(builder.constant("corrupted", std::move(corrupted)));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}

TEST("compiler rejects aggregate activation-size overflow") {
  nnc::GraphBuilder builder;
  const auto huge = std::numeric_limits<std::size_t>::max() / 8;
  auto value = builder.input("huge", {huge});
  value = builder.relu(value, "relu1");
  value = builder.relu(value, "relu2");
  value = builder.relu(value, "relu3");
  builder.output(value);
  const auto graph = std::move(builder).build();
  bool rejected = false;
  try {
    static_cast<void>(nnc::Compiler::compile(
        graph, {.enable_fusion = false, .reuse_buffers = false, .threads = 1}));
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  CHECK(rejected);
}

TEST("shape arithmetic and builder boundaries reject malformed graphs") {
  bool empty_shape = false;
  try {
    static_cast<void>(nnc::Tensor({}, {}));
  } catch (const std::invalid_argument&) {
    empty_shape = true;
  }
  CHECK(empty_shape);

  bool element_overflow = false;
  try {
    static_cast<void>(nnc::Tensor(
        {std::numeric_limits<std::size_t>::max(), 2}, {}));
  } catch (const std::overflow_error&) {
    element_overflow = true;
  }
  CHECK(element_overflow);

  bool byte_overflow = false;
  try {
    static_cast<void>(nnc::Tensor(
        {std::numeric_limits<std::size_t>::max() / 2}, {}));
  } catch (const std::overflow_error&) {
    byte_overflow = true;
  }
  CHECK(byte_overflow);

  bool empty_name = false;
  try {
    nnc::GraphBuilder builder;
    static_cast<void>(builder.input("", {1}));
  } catch (const std::invalid_argument&) {
    empty_name = true;
  }
  CHECK(empty_name);

  bool unknown_value = false;
  try {
    nnc::GraphBuilder builder;
    static_cast<void>(builder.relu(999));
  } catch (const std::invalid_argument&) {
    unknown_value = true;
  }
  CHECK(unknown_value);

  bool no_input = false;
  try {
    nnc::GraphBuilder builder;
    const auto constant = builder.constant("constant", {{1}, {1.0F}});
    builder.output(constant);
    static_cast<void>(std::move(builder).build());
  } catch (const std::invalid_argument&) {
    no_input = true;
  }
  CHECK(no_input);
}

TEST("compiler rejects activation alignment overflow") {
  nnc::GraphBuilder builder;
  const auto huge = std::numeric_limits<std::size_t>::max() / sizeof(float);
  const auto input = builder.input("huge", {huge});
  const auto output = builder.relu(input);
  builder.output(output);
  const auto graph = std::move(builder).build();
  bool rejected = false;
  try {
    static_cast<void>(nnc::Compiler::compile(graph));
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  CHECK(rejected);
}

TEST("zero-instruction graph returns its input with automatic workers") {
  nnc::GraphBuilder builder;
  const auto input = builder.input("input", {2});
  builder.output(input);
  const auto graph = std::move(builder).build();
  auto model = nnc::Compiler::compile(graph, {.threads = 0});
  const nnc::Inputs inputs{{input, {{2}, {3.0F, -1.0F}}}};
  const auto result = model.run(inputs);
  CHECK(model.instructions().empty());
  CHECK(model.memory_plan().empty());
  CHECK(model.stats().arena_bytes == 0);
  CHECK(result.stats.activation_buffer_allocations == 0);
  CHECK(result.output.data == std::vector<float>({3.0F, -1.0F}));
}

}  // namespace

int main() {
  std::size_t failures = 0;
  for (const auto& test : tests()) {
    try {
      test.body();
      std::cout << "[PASS] " << test.name << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
    }
  }
  std::cout << (tests().size() - failures) << '/' << tests().size() << " tests passed\n";
  return failures == 0 ? 0 : 1;
}
