#include "nnc/nnc.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

namespace nnc {
namespace {
constexpr std::size_t kAlignment = 64;
constexpr std::size_t kMaxThreads = 64;
std::size_t checked_elements(const Shape& shape) {
  if (shape.empty()) {
    throw std::invalid_argument("tensor shape must have at least one dimension");
  }
  std::size_t total = 1;
  for (const auto dimension : shape) {
    if (dimension == 0) {
      throw std::invalid_argument("tensor dimensions must be positive");
    }
    if (total > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::overflow_error("tensor element count overflows size_t");
    }
    total *= dimension;
  }
  if (total > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error("tensor byte count overflows size_t");
  }
  return total;
}
std::size_t aligned_bytes(std::size_t bytes) {
  if (bytes > std::numeric_limits<std::size_t>::max() - (kAlignment - 1)) {
    throw std::overflow_error("aligned buffer size overflows size_t");
  }
  return ((bytes + kAlignment - 1) / kAlignment) * kAlignment;
}
std::size_t checked_add(std::size_t left, std::size_t right,
                        const char* context) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error(std::string(context) + " overflows size_t");
  }
  return left + right;
}

void require_same_shape(const Shape& actual, const Shape& expected,
                        const char* context) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(context) + " shape mismatch");
  }
}

struct Block {
  std::size_t offset{};
  std::size_t bytes{};
};

struct ActiveBlock {
  std::size_t offset{};
  std::size_t bytes{};
  std::size_t last{};
};

void coalesce(std::vector<Block>& blocks) {
  std::sort(blocks.begin(), blocks.end(),
            [](const Block& left, const Block& right) {
              return left.offset < right.offset;
            });
  std::vector<Block> merged;
  for (const auto block : blocks) {
    if (!merged.empty() && merged.back().offset + merged.back().bytes == block.offset) {
      merged.back().bytes += block.bytes;
    } else {
      merged.push_back(block);
    }
  }
  blocks = std::move(merged);
}

}  // namespace

Tensor::Tensor(Shape tensor_shape, std::vector<float> values)
    : shape(std::move(tensor_shape)), data(std::move(values)) {
  if (checked_elements(shape) != data.size()) {
    throw std::invalid_argument("tensor payload does not match its shape");
  }
}

std::size_t Tensor::elements() const { return checked_elements(shape); }

std::size_t Tensor::bytes() const { return elements() * sizeof(float); }

const std::vector<Value>& Graph::values() const { return values_; }
const std::vector<Node>& Graph::nodes() const { return nodes_; }
const std::vector<ValueId>& Graph::inputs() const { return inputs_; }
ValueId Graph::output() const {
  if (!has_output_) {
    throw std::logic_error("graph output has not been set");
  }
  return output_;
}

const Value& Graph::value(ValueId id) const {
  if (id >= values_.size()) {
    throw std::out_of_range("unknown graph value");
  }
  return values_[id];
}

void Graph::validate() const {
  if (inputs_.empty()) {
    throw std::invalid_argument("graph must have at least one input");
  }
  if (!has_output_ || output_ >= values_.size()) {
    throw std::invalid_argument("graph must have a valid output");
  }
  std::unordered_set<std::string> names;
  for (std::size_t index = 0; index < values_.size(); ++index) {
    const auto& current = values_[index];
    if (current.id != index) {
      throw std::invalid_argument("graph value IDs must be contiguous");
    }
    static_cast<void>(checked_elements(current.shape));
    if (!names.insert(current.name).second) {
      throw std::invalid_argument("graph value names must be unique");
    }
    if (current.constant.has_value()) {
      require_same_shape(current.constant->shape, current.shape, "constant");
      if (checked_elements(current.constant->shape) != current.constant->data.size()) {
        throw std::invalid_argument("constant payload does not match its shape");
      }
    }
  }

  std::vector<bool> available(values_.size(), false);
  for (const auto input : inputs_) {
    if (input >= values_.size() || !values_[input].is_input) {
      throw std::invalid_argument("graph has an invalid input");
    }
    available[input] = true;
  }
  for (const auto& current : values_) {
    if (current.constant.has_value()) {
      available[current.id] = true;
    }
  }
  for (const auto& node : nodes_) {
    for (const auto input : node.inputs) {
      if (input >= available.size() || !available[input]) {
        throw std::invalid_argument("graph nodes must be topologically ordered");
      }
    }
    if (node.output >= available.size() || available[node.output]) {
      throw std::invalid_argument("graph node has an invalid or duplicate output");
    }
    available[node.output] = true;
  }
  if (!available[output_]) {
    throw std::invalid_argument("graph output is not produced");
  }
}

void GraphBuilder::require_value(ValueId id) const {
  if (id >= graph_.values_.size()) {
    throw std::invalid_argument("operation references an unknown value");
  }
}

void GraphBuilder::require_unique_name(const std::string& name) const {
  if (name.empty()) {
    throw std::invalid_argument("value names must not be empty");
  }
  const auto duplicate = std::find_if(
      graph_.values_.begin(), graph_.values_.end(),
      [&name](const Value& value) { return value.name == name; });
  if (duplicate != graph_.values_.end()) {
    throw std::invalid_argument("value names must be unique: " + name);
  }
}

ValueId GraphBuilder::append_value(std::string name, Shape shape) {
  require_unique_name(name);
  static_cast<void>(checked_elements(shape));
  const auto id = graph_.values_.size();
  graph_.values_.push_back({id, std::move(name), std::move(shape), false, std::nullopt});
  return id;
}

ValueId GraphBuilder::input(std::string name, Shape shape) {
  const auto id = append_value(std::move(name), std::move(shape));
  graph_.values_[id].is_input = true;
  graph_.inputs_.push_back(id);
  return id;
}

ValueId GraphBuilder::constant(std::string name, Tensor tensor) {
  if (checked_elements(tensor.shape) != tensor.data.size()) {
    throw std::invalid_argument("constant payload does not match its shape");
  }
  const auto id = append_value(std::move(name), tensor.shape);
  graph_.values_[id].constant = std::move(tensor);
  return id;
}

ValueId GraphBuilder::matmul(ValueId left, ValueId right, std::string name) {
  require_value(left);
  require_value(right);
  const auto& left_shape = graph_.values_[left].shape;
  const auto& right_shape = graph_.values_[right].shape;
  if (left_shape.size() != 2 || right_shape.size() != 2 ||
      left_shape[1] != right_shape[0]) {
    throw std::invalid_argument("MatMul expects [M,K] x [K,N]");
  }
  const auto result = append_value(std::move(name), {left_shape[0], right_shape[1]});
  graph_.nodes_.push_back({OpKind::MatMul, {left, right}, result});
  return result;
}

ValueId GraphBuilder::bias_add(ValueId value_id, ValueId bias,
                               std::string name) {
  require_value(value_id);
  require_value(bias);
  const auto& value_shape = graph_.values_[value_id].shape;
  const auto& bias_shape = graph_.values_[bias].shape;
  if (value_shape.size() != 2 || bias_shape.size() != 1 ||
      bias_shape[0] != value_shape[1]) {
    throw std::invalid_argument("BiasAdd expects [M,N] + [N]");
  }
  const auto result = append_value(std::move(name), value_shape);
  graph_.nodes_.push_back({OpKind::BiasAdd, {value_id, bias}, result});
  return result;
}

ValueId GraphBuilder::relu(ValueId value_id, std::string name) {
  require_value(value_id);
  const auto result = append_value(std::move(name), graph_.values_[value_id].shape);
  graph_.nodes_.push_back({OpKind::Relu, {value_id}, result});
  return result;
}

void GraphBuilder::output(ValueId value_id) {
  require_value(value_id);
  graph_.output_ = value_id;
  graph_.has_output_ = true;
}

Graph GraphBuilder::build() && {
  graph_.validate();
  return std::move(graph_);
}

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t workers) : worker_count_(workers) {
    if (workers == 0 || workers > kMaxThreads) {
      throw std::invalid_argument("worker count must be between 1 and 64");
    }
    if (workers == 1) {
      return;
    }
    threads_.reserve(workers);
    try {
      for (std::size_t index = 0; index < workers; ++index) {
        threads_.emplace_back([this, index] { worker_loop(index); });
      }
    } catch (...) {
      {
        std::lock_guard lock(state_mutex_);
        stopping_ = true;
        ++generation_;
      }
      start_cv_.notify_all();
      for (auto& thread : threads_) {
        if (thread.joinable()) {
          thread.join();
        }
      }
      throw;
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard lock(state_mutex_);
      stopping_ = true;
      ++generation_;
    }
    start_cv_.notify_all();
    for (auto& thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void parallel_rows(
      std::size_t rows,
      const std::function<void(std::size_t, std::size_t)>& function) {
    if (worker_count_ == 1 || rows < 2) {
      function(0, rows);
      return;
    }
    std::lock_guard dispatch_lock(dispatch_mutex_);
    std::unique_lock lock(state_mutex_);
    rows_ = rows;
    task_ = function;
    completed_ = 0;
    error_ = nullptr;
    ++generation_;
    start_cv_.notify_all();
    done_cv_.wait(lock, [this] { return completed_ == worker_count_; });
    if (error_) {
      std::rethrow_exception(error_);
    }
  }

  [[nodiscard]] std::size_t workers() const { return worker_count_; }

 private:
  void worker_loop(std::size_t index) {
    std::size_t observed_generation = 0;
    while (true) {
      std::unique_lock lock(state_mutex_);
      start_cv_.wait(lock, [this, observed_generation] {
        return stopping_ || generation_ != observed_generation;
      });
      if (stopping_) {
        return;
      }
      observed_generation = generation_;
      const auto rows = rows_;
      const auto* task = &task_;
      lock.unlock();
      const auto begin = rows * index / worker_count_;
      const auto end = rows * (index + 1) / worker_count_;
      try {
        if (begin < end) {
          (*task)(begin, end);
        }
      } catch (...) {
        std::lock_guard error_lock(state_mutex_);
        if (!error_) {
          error_ = std::current_exception();
        }
      }
      lock.lock();
      ++completed_;
      if (completed_ == worker_count_) {
        done_cv_.notify_one();
      }
    }
  }

  std::size_t worker_count_;
  std::vector<std::thread> threads_;
  std::mutex dispatch_mutex_;
  std::mutex state_mutex_;
  std::condition_variable start_cv_;
  std::condition_variable done_cv_;
  bool stopping_{false};
  std::size_t generation_{0};
  std::size_t completed_{0};
  std::size_t rows_{0};
  std::function<void(std::size_t, std::size_t)> task_;
  std::exception_ptr error_;
};

namespace {

void validate_inputs(const Graph& graph, const Inputs& inputs) {
  for (const auto input_id : graph.inputs()) {
    const auto found = inputs.find(input_id);
    if (found == inputs.end()) {
      throw std::invalid_argument("missing required runtime input: " +
                                  graph.value(input_id).name);
    }
    require_same_shape(found->second.shape, graph.value(input_id).shape,
                       "runtime input");
    if (checked_elements(found->second.shape) != found->second.data.size()) {
      throw std::invalid_argument("runtime input payload does not match its shape");
    }
  }
  for (const auto& [id, tensor] : inputs) {
    if (id >= graph.values().size() || !graph.value(id).is_input) {
      throw std::invalid_argument("runtime input binding references a non-input value");
    }
    require_same_shape(tensor.shape, graph.value(id).shape, "runtime input");
    if (checked_elements(tensor.shape) != tensor.data.size()) {
      throw std::invalid_argument("runtime input payload does not match its shape");
    }
  }
}

void matmul_kernel(std::span<const float> left, const Shape& left_shape,
                   std::span<const float> right, const Shape& right_shape,
                   std::span<float> output, ThreadPool* pool, RunStats& stats,
                   std::span<const float> bias = {}, bool relu = false) {
  const auto rows = left_shape[0];
  const auto inner = left_shape[1];
  const auto columns = right_shape[1];
  const auto work = [&](std::size_t begin, std::size_t end) {
    for (auto row = begin; row < end; ++row) {
      for (std::size_t column = 0; column < columns; ++column) {
        float sum = 0.0F;
        for (std::size_t index = 0; index < inner; ++index) {
          sum += left[row * inner + index] * right[index * columns + column];
        }
        if (!bias.empty()) {
          sum += bias[column];
        }
        output[row * columns + column] = relu ? std::max(0.0F, sum) : sum;
      }
    }
  };
  if (pool != nullptr && pool->workers() > 1 && rows >= pool->workers()) {
    pool->parallel_rows(rows, work);
    stats.parallel_tasks += pool->workers();
    stats.workers_used = std::max(stats.workers_used, pool->workers());
  } else {
    work(0, rows);
  }
}

Tensor execute_node_owned(const Graph& graph, const Node& node,
                          const std::vector<const Tensor*>& values,
                          RunStats& stats) {
  const auto get = [&](ValueId id) -> const Tensor& {
    if (id >= values.size() || values[id] == nullptr) {
      throw std::logic_error("executor encountered an unavailable value");
    }
    return *values[id];
  };
  const auto& output_value = graph.value(node.output);
  Tensor output(output_value.shape,
                std::vector<float>(checked_elements(output_value.shape), 0.0F));
  if (node.kind == OpKind::MatMul) {
    const auto& left = get(node.inputs[0]);
    const auto& right = get(node.inputs[1]);
    matmul_kernel(left.data, left.shape, right.data, right.shape, output.data,
                  nullptr, stats);
  } else if (node.kind == OpKind::BiasAdd) {
    const auto& source = get(node.inputs[0]);
    const auto& bias = get(node.inputs[1]);
    const auto columns = source.shape[1];
    for (std::size_t index = 0; index < source.data.size(); ++index) {
      output.data[index] = source.data[index] + bias.data[index % columns];
    }
  } else if (node.kind == OpKind::Relu) {
    const auto& source = get(node.inputs[0]);
    std::transform(source.data.begin(), source.data.end(), output.data.begin(),
                   [](float value) { return std::max(0.0F, value); });
  } else {
    const auto& left = get(node.inputs[0]);
    const auto& right = get(node.inputs[1]);
    const auto& bias = get(node.inputs[2]);
    matmul_kernel(left.data, left.shape, right.data, right.shape, output.data,
                  nullptr, stats, bias.data, true);
  }
  return output;
}

std::vector<Node> fuse_graph(const Graph& graph, bool enabled,
                             std::size_t& fused_count) {
  if (!enabled) {
    return graph.nodes();
  }
  std::vector<std::size_t> consumers(graph.values().size(), 0);
  for (const auto& node : graph.nodes()) {
    for (const auto input : node.inputs) {
      ++consumers[input];
    }
  }
  std::vector<Node> result;
  for (std::size_t index = 0; index < graph.nodes().size();) {
    if (index + 2 < graph.nodes().size()) {
      const auto& matmul = graph.nodes()[index];
      const auto& add = graph.nodes()[index + 1];
      const auto& relu = graph.nodes()[index + 2];
      const bool matches =
          matmul.kind == OpKind::MatMul && add.kind == OpKind::BiasAdd &&
          relu.kind == OpKind::Relu && add.inputs[0] == matmul.output &&
          relu.inputs[0] == add.output && consumers[matmul.output] == 1 &&
          consumers[add.output] == 1 && graph.output() != matmul.output &&
          graph.output() != add.output;
      if (matches) {
        result.push_back({OpKind::FusedLinearRelu,
                          {matmul.inputs[0], matmul.inputs[1], add.inputs[1]},
                          relu.output});
        ++fused_count;
        index += 3;
        continue;
      }
    }
    result.push_back(graph.nodes()[index]);
    ++index;
  }
  return result;
}

std::vector<BufferAssignment> plan_memory(const Graph& graph,
                                          const std::vector<Node>& instructions,
                                          bool reuse, std::size_t& arena_bytes,
                                          std::size_t& reused_values) {
  std::unordered_map<ValueId, std::size_t> producer;
  std::unordered_map<ValueId, std::size_t> last_use;
  for (std::size_t index = 0; index < instructions.size(); ++index) {
    producer[instructions[index].output] = index;
    last_use[instructions[index].output] = index;
  }
  for (std::size_t index = 0; index < instructions.size(); ++index) {
    for (const auto input : instructions[index].inputs) {
      if (producer.contains(input)) {
        last_use[input] = std::max(last_use[input], index);
      }
    }
  }
  if (producer.contains(graph.output())) {
    last_use[graph.output()] = instructions.size();
  }

  std::vector<BufferAssignment> assignments;
  std::vector<ActiveBlock> active;
  std::vector<Block> free_blocks;
  std::size_t high_watermark = 0;
  for (std::size_t index = 0; index < instructions.size(); ++index) {
    for (auto iterator = active.begin(); iterator != active.end();) {
      if (iterator->last < index) {
        free_blocks.push_back({iterator->offset, iterator->bytes});
        iterator = active.erase(iterator);
      } else {
        ++iterator;
      }
    }
    if (reuse) {
      coalesce(free_blocks);
    }
    const auto value_id = instructions[index].output;
    const auto raw_bytes = graph.value(value_id).shape.empty()
                               ? 0
                               : checked_elements(graph.value(value_id).shape) * sizeof(float);
    const auto bytes = aligned_bytes(raw_bytes);
    std::size_t offset = high_watermark;
    bool reused = false;
    if (reuse) {
      auto best = free_blocks.end();
      for (auto iterator = free_blocks.begin(); iterator != free_blocks.end(); ++iterator) {
        if (iterator->bytes >= bytes &&
            (best == free_blocks.end() || iterator->bytes < best->bytes)) {
          best = iterator;
        }
      }
      if (best != free_blocks.end()) {
        reused = true;
        offset = best->offset;
        if (best->bytes > bytes) {
          const Block remainder{best->offset + bytes, best->bytes - bytes};
          *best = remainder;
        } else {
          free_blocks.erase(best);
        }
      } else {
        high_watermark = checked_add(high_watermark, bytes, "arena capacity");
      }
    } else {
      high_watermark = checked_add(high_watermark, bytes, "arena capacity");
    }
    if (reused) {
      ++reused_values;
    }
    const auto last = last_use[value_id];
    assignments.push_back({value_id, offset, bytes, index, last});
    active.push_back({offset, bytes, last});
  }
  arena_bytes = high_watermark;
  return assignments;
}

}  // namespace

BaselineExecutor::BaselineExecutor(const Graph& graph) : graph_(graph) {
  graph_.validate();
}

RunResult BaselineExecutor::run(const Inputs& inputs) const {
  validate_inputs(graph_, inputs);
  std::vector<std::optional<Tensor>> owned(graph_.values().size());
  std::vector<const Tensor*> values(graph_.values().size(), nullptr);
  for (const auto& [id, tensor] : inputs) {
    values[id] = &tensor;
  }
  for (const auto& value : graph_.values()) {
    if (value.constant.has_value()) {
      values[value.id] = &value.constant.value();
    }
  }
  RunStats stats;
  for (const auto& node : graph_.nodes()) {
    auto output = execute_node_owned(graph_, node, values, stats);
    stats.peak_activation_bytes = checked_add(
        stats.peak_activation_bytes, output.bytes(), "baseline activation bytes");
    ++stats.activation_buffer_allocations;
    owned[node.output] = std::move(output);
    values[node.output] = &owned[node.output].value();
  }
  return {*values[graph_.output()], stats};
}

RunResult BaselineExecutor::run(const Graph& graph, const Inputs& inputs) {
  return BaselineExecutor(graph).run(inputs);
}

CompiledModel::CompiledModel(Graph graph, std::vector<Node> instructions,
                             std::vector<BufferAssignment> memory_plan,
                             CompileOptions options, CompileStats stats)
    : graph_(std::move(graph)),
      instructions_(std::move(instructions)),
      memory_plan_(std::move(memory_plan)),
      options_(options),
      stats_(stats),
      pool_(std::make_unique<ThreadPool>(stats.worker_threads)) {}

CompiledModel::CompiledModel(CompiledModel&&) noexcept = default;
CompiledModel& CompiledModel::operator=(CompiledModel&&) noexcept = default;
CompiledModel::~CompiledModel() = default;

RunResult CompiledModel::run(const Inputs& inputs) const {
  validate_inputs(graph_, inputs);
  std::vector<std::span<const float>> values(graph_.values().size());
  for (const auto& [id, tensor] : inputs) {
    values[id] = tensor.data;
  }
  for (const auto& value : graph_.values()) {
    if (value.constant.has_value()) {
      values[value.id] = value.constant->data;
    }
  }

  std::vector<float> arena(stats_.arena_bytes / sizeof(float));
  std::vector<std::size_t> offsets(graph_.values().size(),
                                   std::numeric_limits<std::size_t>::max());
  for (const auto& assignment : memory_plan_) {
    offsets[assignment.value] = assignment.offset;
  }
  RunStats stats{stats_.arena_bytes, stats_.arena_bytes == 0 ? 0U : 1U, 0, 1};
  for (const auto& node : instructions_) {
    const auto count = checked_elements(graph_.value(node.output).shape);
    const auto offset = offsets[node.output] / sizeof(float);
    std::span<float> output(arena.data() + offset, count);
    if (node.kind == OpKind::MatMul) {
      matmul_kernel(values[node.inputs[0]], graph_.value(node.inputs[0]).shape,
                    values[node.inputs[1]], graph_.value(node.inputs[1]).shape,
                    output, pool_.get(), stats);
    } else if (node.kind == OpKind::BiasAdd) {
      const auto columns = graph_.value(node.inputs[0]).shape[1];
      for (std::size_t index = 0; index < count; ++index) {
        output[index] = values[node.inputs[0]][index] +
                        values[node.inputs[1]][index % columns];
      }
    } else if (node.kind == OpKind::Relu) {
      for (std::size_t index = 0; index < count; ++index) {
        output[index] = std::max(0.0F, values[node.inputs[0]][index]);
      }
    } else {
      matmul_kernel(values[node.inputs[0]], graph_.value(node.inputs[0]).shape,
                    values[node.inputs[1]], graph_.value(node.inputs[1]).shape,
                    output, pool_.get(), stats, values[node.inputs[2]], true);
    }
    values[node.output] = output;
  }
  const auto final_view = values[graph_.output()];
  return {Tensor(graph_.value(graph_.output()).shape,
                 std::vector<float>(final_view.begin(), final_view.end())),
          stats};
}

const CompileStats& CompiledModel::stats() const { return stats_; }
const std::vector<Node>& CompiledModel::instructions() const { return instructions_; }
const std::vector<BufferAssignment>& CompiledModel::memory_plan() const {
  return memory_plan_;
}

CompiledModel Compiler::compile(const Graph& graph, CompileOptions options) {
  graph.validate();
  if (options.threads > kMaxThreads) {
    throw std::invalid_argument("thread count cannot exceed 64");
  }
  const auto hardware_threads =
      std::max<std::size_t>(1, std::thread::hardware_concurrency());
  const auto worker_threads = options.threads == 0
                                  ? std::min(hardware_threads, kMaxThreads)
                                  : options.threads;
  CompileStats stats;
  stats.original_nodes = graph.nodes().size();
  stats.worker_threads = worker_threads;
  auto instructions = fuse_graph(graph, options.enable_fusion,
                                 stats.fused_linear_relu);
  stats.compiled_nodes = instructions.size();
  for (const auto& node : graph.nodes()) {
    stats.naive_activation_bytes = checked_add(
        stats.naive_activation_bytes,
        checked_elements(graph.value(node.output).shape) * sizeof(float),
        "naive activation bytes");
  }
  auto memory_plan = plan_memory(graph, instructions, options.reuse_buffers,
                                 stats.arena_bytes, stats.reused_values);
  return CompiledModel(graph, std::move(instructions), std::move(memory_plan),
                       options, stats);
}

const char* op_name(OpKind kind) {
  switch (kind) {
    case OpKind::MatMul:
      return "MatMul";
    case OpKind::BiasAdd:
      return "BiasAdd";
    case OpKind::Relu:
      return "ReLU";
    case OpKind::FusedLinearRelu:
      return "FusedLinearReLU";
  }
  return "Unknown";
}

Graph make_demo_graph(std::size_t batch) {
  if (batch == 0) {
    throw std::invalid_argument("demo batch size must be positive");
  }
  std::mt19937 generator(42);
  std::uniform_real_distribution<float> weights(-0.12F, 0.12F);
  const auto tensor = [&](Shape shape) {
    std::vector<float> values(checked_elements(shape));
    std::generate(values.begin(), values.end(), [&] { return weights(generator); });
    return Tensor(std::move(shape), std::move(values));
  };

  GraphBuilder builder;
  const auto input = builder.input("input", {batch, 128});
  const auto w1 = builder.constant("layer1.weights", tensor({128, 256}));
  const auto b1 = builder.constant("layer1.bias", tensor({256}));
  const auto h1 = builder.relu(
      builder.bias_add(builder.matmul(input, w1, "layer1.matmul"), b1,
                       "layer1.bias_add"),
      "layer1.relu");
  const auto w2 = builder.constant("layer2.weights", tensor({256, 128}));
  const auto b2 = builder.constant("layer2.bias", tensor({128}));
  const auto h2 = builder.relu(
      builder.bias_add(builder.matmul(h1, w2, "layer2.matmul"), b2,
                       "layer2.bias_add"),
      "layer2.relu");
  const auto w3 = builder.constant("layer3.weights", tensor({128, 64}));
  const auto b3 = builder.constant("layer3.bias", tensor({64}));
  const auto output = builder.relu(
      builder.bias_add(builder.matmul(h2, w3, "layer3.matmul"), b3,
                       "layer3.bias_add"),
      "layer3.relu");
  builder.output(output);
  return std::move(builder).build();
}

Inputs make_demo_inputs(const Graph& graph, std::size_t seed) {
  graph.validate();
  std::mt19937 generator(static_cast<std::uint32_t>(seed));
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  Inputs result;
  for (const auto input : graph.inputs()) {
    std::vector<float> values(checked_elements(graph.value(input).shape));
    std::generate(values.begin(), values.end(),
                  [&] { return distribution(generator); });
    result.emplace(input, Tensor(graph.value(input).shape, std::move(values)));
  }
  return result;
}

float max_abs_error(std::span<const float> left, std::span<const float> right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("cannot compare tensors with different sizes");
  }
  float maximum = 0.0F;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const auto difference = std::abs(left[index] - right[index]);
    if (!std::isfinite(left[index]) || !std::isfinite(right[index]) ||
        !std::isfinite(difference)) {
      return std::numeric_limits<float>::infinity();
    }
    maximum = std::max(maximum, difference);
  }
  return maximum;
}

}  // namespace nnc
