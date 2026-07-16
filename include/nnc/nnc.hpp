#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace nnc {

using Shape = std::vector<std::size_t>;
using ValueId = std::size_t;

struct Tensor {
  Shape shape;
  std::vector<float> data;

  Tensor() = default;
  Tensor(Shape tensor_shape, std::vector<float> values);
  [[nodiscard]] std::size_t elements() const;
  [[nodiscard]] std::size_t bytes() const;
};

enum class OpKind { MatMul, BiasAdd, Relu, FusedLinearRelu };

struct Value {
  ValueId id{};
  std::string name;
  Shape shape;
  bool is_input{false};
  std::optional<Tensor> constant;
};

struct Node {
  OpKind kind{};
  std::vector<ValueId> inputs;
  ValueId output{};
};

class Graph {
 public:
  [[nodiscard]] const std::vector<Value>& values() const;
  [[nodiscard]] const std::vector<Node>& nodes() const;
  [[nodiscard]] const std::vector<ValueId>& inputs() const;
  [[nodiscard]] ValueId output() const;
  [[nodiscard]] const Value& value(ValueId id) const;
  void validate() const;

 private:
  friend class GraphBuilder;
  std::vector<Value> values_;
  std::vector<Node> nodes_;
  std::vector<ValueId> inputs_;
  ValueId output_{};
  bool has_output_{false};
};

class GraphBuilder {
 public:
  ValueId input(std::string name, Shape shape);
  ValueId constant(std::string name, Tensor tensor);
  ValueId matmul(ValueId left, ValueId right, std::string name = "matmul");
  ValueId bias_add(ValueId value, ValueId bias, std::string name = "bias_add");
  ValueId relu(ValueId value, std::string name = "relu");
  void output(ValueId value);
  [[nodiscard]] Graph build() &&;

 private:
  ValueId append_value(std::string name, Shape shape);
  void require_value(ValueId id) const;
  void require_unique_name(const std::string& name) const;
  Graph graph_;
};

using Inputs = std::unordered_map<ValueId, Tensor>;

struct RunStats {
  std::size_t peak_activation_bytes{};
  std::size_t activation_buffer_allocations{};
  std::size_t parallel_tasks{};
  std::size_t workers_used{1};
};

struct RunResult {
  Tensor output;
  RunStats stats;
};

class BaselineExecutor {
 public:
  explicit BaselineExecutor(const Graph& graph);
  [[nodiscard]] RunResult run(const Inputs& inputs) const;
  [[nodiscard]] static RunResult run(const Graph& graph, const Inputs& inputs);

 private:
  Graph graph_;
};

struct CompileOptions {
  bool enable_fusion{true};
  bool reuse_buffers{true};
  std::size_t threads{1};
};

struct CompileStats {
  std::size_t original_nodes{};
  std::size_t compiled_nodes{};
  std::size_t fused_linear_relu{};
  std::size_t naive_activation_bytes{};
  std::size_t arena_bytes{};
  std::size_t reused_values{};
  std::size_t worker_threads{};
};

struct BufferAssignment {
  ValueId value{};
  std::size_t offset{};
  std::size_t bytes{};
  std::size_t first_instruction{};
  std::size_t last_instruction{};
};

class ThreadPool;

class CompiledModel {
 public:
  CompiledModel(CompiledModel&&) noexcept;
  CompiledModel& operator=(CompiledModel&&) noexcept;
  ~CompiledModel();
  CompiledModel(const CompiledModel&) = delete;
  CompiledModel& operator=(const CompiledModel&) = delete;

  [[nodiscard]] RunResult run(const Inputs& inputs) const;
  [[nodiscard]] const CompileStats& stats() const;
  [[nodiscard]] const std::vector<Node>& instructions() const;
  [[nodiscard]] const std::vector<BufferAssignment>& memory_plan() const;

 private:
  friend class Compiler;
  CompiledModel(Graph graph, std::vector<Node> instructions,
                std::vector<BufferAssignment> memory_plan,
                CompileOptions options, CompileStats stats);

  Graph graph_;
  std::vector<Node> instructions_;
  std::vector<BufferAssignment> memory_plan_;
  CompileOptions options_;
  CompileStats stats_;
  std::unique_ptr<ThreadPool> pool_;
};

class Compiler {
 public:
  [[nodiscard]] static CompiledModel compile(const Graph& graph,
                                             CompileOptions options = {});
};

[[nodiscard]] const char* op_name(OpKind kind);
[[nodiscard]] Graph make_demo_graph(std::size_t batch = 64);
[[nodiscard]] Inputs make_demo_inputs(const Graph& graph, std::size_t seed = 42);
[[nodiscard]] float max_abs_error(std::span<const float> left,
                                  std::span<const float> right);

}  // namespace nnc
