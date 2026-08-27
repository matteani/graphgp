#include "graph_sharding/planner.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace gs = graph_sharding;

namespace {

void cuda_check(cudaError_t status, const char* expression, const char* file, int line) {
  if (status == cudaSuccess) return;
  std::ostringstream message;
  message << file << ':' << line << ": " << expression << " failed: "
          << cudaGetErrorString(status);
  throw std::runtime_error(message.str());
}

void nccl_check(ncclResult_t status, const char* expression, const char* file, int line) {
  if (status == ncclSuccess) return;
  std::ostringstream message;
  message << file << ':' << line << ": " << expression << " failed: "
          << ncclGetErrorString(status);
  throw std::runtime_error(message.str());
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)
#define NCCL_CHECK(expr) nccl_check((expr), #expr, __FILE__, __LINE__)

struct Options {
  std::size_t nodes = 1U << 20U;
  std::size_t parents = 8;
  std::size_t levels = 16;
  int gpus = 0;
  std::size_t communities = 0;
  double locality = 0.0;
  std::uint64_t seed = 42;
  std::string dtype = "float";
  std::string strategy = "all";
  int warmup = 5;
  int runs = 20;
  std::string output;
};

std::string usage() {
  return R"(Native CUDA/NCCL graph sharding benchmark

Options:
  --nodes N          Number of DAG nodes (default 1048576)
  --parents K        Parents per non-root node (default 8)
  --levels L         Topological levels (default 16)
  --gpus M           First M visible GPUs; 0 means all (default 0)
  --communities C     Planted communities; 0 means M (default 0)
  --locality Q        Same-community parent probability in [0,1] (default 0)
  --seed S            Graph, partition, and device RNG seed (default 42)
  --dtype TYPE        float|double (default float)
  --strategy NAME     all|random|affinity|closure (default all)
  --warmup N          Untimed runs per strategy (default 5)
  --runs N            Timed runs per strategy (default 20)
  --output PATH       Also write JSON to PATH
  --help              Show this message
)";
}

template <typename T>
T parse_number(const std::string& text, const char* name) {
  std::size_t consumed = 0;
  try {
    if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
      if (!text.empty() && text.front() == '-') throw std::out_of_range(name);
      auto value = std::stoull(text, &consumed);
      if (consumed != text.size() || value > std::numeric_limits<T>::max()) throw std::out_of_range(name);
      return static_cast<T>(value);
    } else if constexpr (std::is_integral_v<T>) {
      auto value = std::stoll(text, &consumed);
      if (consumed != text.size() || value < std::numeric_limits<T>::min() ||
          value > std::numeric_limits<T>::max()) throw std::out_of_range(name);
      return static_cast<T>(value);
    } else {
      auto value = std::stod(text, &consumed);
      if (consumed != text.size()) throw std::invalid_argument(name);
      return static_cast<T>(value);
    }
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("invalid value for ") + name + ": " + text);
  }
}

Options parse_options(int argc, char** argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      std::cout << usage();
      std::exit(EXIT_SUCCESS);
    }
    std::string value;
    auto equals = key.find('=');
    if (equals != std::string::npos) {
      value = key.substr(equals + 1);
      key.resize(equals);
    } else {
      if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
      value = argv[++i];
    }
    if (key == "--nodes") result.nodes = parse_number<std::size_t>(value, "--nodes");
    else if (key == "--parents") result.parents = parse_number<std::size_t>(value, "--parents");
    else if (key == "--levels") result.levels = parse_number<std::size_t>(value, "--levels");
    else if (key == "--gpus") result.gpus = parse_number<int>(value, "--gpus");
    else if (key == "--communities") result.communities = parse_number<std::size_t>(value, "--communities");
    else if (key == "--locality") result.locality = parse_number<double>(value, "--locality");
    else if (key == "--seed") result.seed = parse_number<std::uint64_t>(value, "--seed");
    else if (key == "--dtype") result.dtype = value;
    else if (key == "--strategy") result.strategy = value;
    else if (key == "--warmup") result.warmup = parse_number<int>(value, "--warmup");
    else if (key == "--runs") result.runs = parse_number<int>(value, "--runs");
    else if (key == "--output") result.output = value;
    else throw std::invalid_argument("unknown option: " + key);
  }
  if (result.gpus < 0) throw std::invalid_argument("--gpus must be non-negative");
  if (result.warmup < 0) throw std::invalid_argument("--warmup must be non-negative");
  if (result.runs <= 0) throw std::invalid_argument("--runs must be positive");
  if (result.dtype != "float" && result.dtype != "double") {
    throw std::invalid_argument("--dtype must be float or double");
  }
  if (result.strategy != "all" && result.strategy != "random" &&
      result.strategy != "affinity" && result.strategy != "closure") {
    throw std::invalid_argument("--strategy must be all, random, affinity, or closure");
  }
  return result;
}

template <typename T>
__device__ T node_random(std::uint64_t seed, std::uint64_t iteration, std::uint32_t node);

template <>
__device__ float node_random<float>(std::uint64_t seed, std::uint64_t iteration,
                                    std::uint32_t node) {
  curandStatePhilox4_32_10_t state;
  curand_init(seed, node, iteration, &state);
  return curand_uniform(&state) - 0.5F;
}

template <>
__device__ double node_random<double>(std::uint64_t seed, std::uint64_t iteration,
                                      std::uint32_t node) {
  curandStatePhilox4_32_10_t state;
  curand_init(seed, node, iteration, &state);
  return curand_uniform_double(&state) - 0.5;
}

template <typename T>
__global__ void pack_kernel(const T* values, const std::uint32_t* slots, T* packed,
                            std::size_t count) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) packed[i] = values[slots[i]];
}

template <typename T>
__global__ void update_kernel(const std::uint32_t* global_ids, const std::uint32_t* level_nodes,
                              const std::int32_t* parent_refs, std::size_t row_begin,
                              std::size_t row_count, std::size_t parents_per_node,
                              std::size_t active_parents, const T* received, T* values,
                              std::uint64_t seed, std::uint64_t iteration) {
  std::size_t local_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_row >= row_count) return;
  const std::size_t row = row_begin + local_row;
  const std::uint32_t slot = level_nodes[row];
  T value = node_random<T>(seed, iteration, global_ids[slot]);
  for (std::size_t j = 0; j < active_parents; ++j) {
    const std::int32_t ref = parent_refs[row * parents_per_node + j];
    value += ref >= 0 ? values[ref] : received[-1 - ref];
  }
  values[slot] = value;
}

template <typename T>
__global__ void partial_sum_kernel(const T* values, const std::uint32_t* primary_slots,
                                   std::size_t count, T* result) {
  T value = 0;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    value += values[primary_slots[i]];
  }
  __shared__ T shared[256];
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(result, shared[0]);
}

template <typename T>
ncclDataType_t nccl_type();

template <>
ncclDataType_t nccl_type<float>() { return ncclFloat32; }

template <>
ncclDataType_t nccl_type<double>() { return ncclFloat64; }

template <typename T>
T* device_copy(const std::vector<T>& values, cudaStream_t stream) {
  if (values.empty()) return nullptr;
  T* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, values.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpyAsync(pointer, values.data(), values.size() * sizeof(T),
                             cudaMemcpyHostToDevice, stream));
  return pointer;
}

template <typename T>
T* device_allocate(std::size_t count) {
  if (count == 0) return nullptr;
  T* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
  return pointer;
}

template <typename T>
struct DeviceRank {
  int device = -1;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  T* values = nullptr;
  std::uint32_t* global_ids = nullptr;
  std::uint32_t* level_nodes = nullptr;
  std::int32_t* parent_refs = nullptr;
  std::uint32_t* send_slots = nullptr;
  std::uint32_t* primary_slots = nullptr;
  T* send_buffer = nullptr;
  T* recv_buffer = nullptr;
  T* checksum = nullptr;
};

template <typename T>
struct RunOutput {
  double milliseconds = 0;
  double checksum = 0;
  std::vector<T> values;
};

template <typename T>
class Executor {
 public:
  Executor(const gs::ExecutionPlan& plan, std::vector<int> devices)
      : plan_(plan), devices_(std::move(devices)), ranks_(plan.ranks), comms_(plan.ranks) {
    if (devices_.size() != ranks_.size()) throw std::invalid_argument("device/rank count mismatch");
    NCCL_CHECK(ncclCommInitAll(comms_.data(), static_cast<int>(ranks_.size()), devices_.data()));
    try {
      for (std::size_t rank = 0; rank < ranks_.size(); ++rank) initialize_rank(rank);
    } catch (...) {
      cleanup();
      throw;
    }
  }

  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  ~Executor() { cleanup(); }

  RunOutput<T> run(std::uint64_t seed, std::uint64_t iteration, bool collect_values) {
    constexpr int threads = 256;
    for (auto& rank : ranks_) {
      CUDA_CHECK(cudaSetDevice(rank.device));
      CUDA_CHECK(cudaEventRecord(rank.start, rank.stream));
    }

    for (std::size_t level = 0; level < plan_.levels; ++level) {
      for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
        const auto& host = plan_.rank_plans[rank];
        const auto& lp = host.levels[level];
        const auto count = lp.send_slot_end - lp.send_slot_begin;
        if (count == 0) continue;
        CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
        pack_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads, 0,
                      ranks_[rank].stream>>>(
            ranks_[rank].values, ranks_[rank].send_slots + lp.send_slot_begin,
            ranks_[rank].send_buffer, count);
        CUDA_CHECK(cudaGetLastError());
      }

      NCCL_CHECK(ncclGroupStart());
      for (std::size_t src = 0; src < ranks_.size(); ++src) {
        const auto& send = plan_.rank_plans[src].levels[level].send_offsets;
        for (std::size_t dst = 0; dst < ranks_.size(); ++dst) {
          if (src == dst) continue;
          const auto count = send[dst + 1] - send[dst];
          if (count == 0) continue;
          const auto& recv = plan_.rank_plans[dst].levels[level].recv_offsets;
          if (count != recv[src + 1] - recv[src]) {
            throw std::logic_error("runtime send/receive mismatch");
          }
          CUDA_CHECK(cudaSetDevice(ranks_[src].device));
          NCCL_CHECK(ncclSend(ranks_[src].send_buffer + send[dst], count, nccl_type<T>(),
                              static_cast<int>(dst), comms_[src], ranks_[src].stream));
          CUDA_CHECK(cudaSetDevice(ranks_[dst].device));
          NCCL_CHECK(ncclRecv(ranks_[dst].recv_buffer + recv[src], count, nccl_type<T>(),
                              static_cast<int>(src), comms_[dst], ranks_[dst].stream));
        }
      }
      NCCL_CHECK(ncclGroupEnd());

      for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
        const auto& lp = plan_.rank_plans[rank].levels[level];
        const auto count = lp.node_end - lp.node_begin;
        if (count == 0) continue;
        CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
        update_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads, 0,
                        ranks_[rank].stream>>>(
            ranks_[rank].global_ids, ranks_[rank].level_nodes, ranks_[rank].parent_refs,
            lp.node_begin, count, plan_.parents, level == 0 ? 0 : plan_.parents,
            ranks_[rank].recv_buffer, ranks_[rank].values, seed, iteration);
        CUDA_CHECK(cudaGetLastError());
      }
    }

    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
      CUDA_CHECK(cudaMemsetAsync(ranks_[rank].checksum, 0, sizeof(T), ranks_[rank].stream));
      const auto count = plan_.rank_plans[rank].primary_slots.size();
      const auto blocks = static_cast<unsigned int>(std::min<std::size_t>(1024, (count + 255) / 256));
      if (blocks > 0) {
        partial_sum_kernel<<<blocks, 256, 0, ranks_[rank].stream>>>(
            ranks_[rank].values, ranks_[rank].primary_slots, count, ranks_[rank].checksum);
        CUDA_CHECK(cudaGetLastError());
      }
    }
    NCCL_CHECK(ncclGroupStart());
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
      NCCL_CHECK(ncclAllReduce(ranks_[rank].checksum, ranks_[rank].checksum, 1, nccl_type<T>(),
                               ncclSum, comms_[rank], ranks_[rank].stream));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (auto& rank : ranks_) {
      CUDA_CHECK(cudaSetDevice(rank.device));
      CUDA_CHECK(cudaEventRecord(rank.stop, rank.stream));
    }
    double maximum_ms = 0;
    for (auto& rank : ranks_) {
      CUDA_CHECK(cudaSetDevice(rank.device));
      CUDA_CHECK(cudaEventSynchronize(rank.stop));
      float elapsed = 0;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed, rank.start, rank.stop));
      maximum_ms = std::max(maximum_ms, static_cast<double>(elapsed));
    }
    T checksum = 0;
    CUDA_CHECK(cudaSetDevice(ranks_[0].device));
    CUDA_CHECK(cudaMemcpy(&checksum, ranks_[0].checksum, sizeof(T), cudaMemcpyDeviceToHost));
    if (!std::isfinite(static_cast<double>(checksum))) {
      throw std::runtime_error("non-finite checksum; reduce k/levels or use double precision");
    }

    RunOutput<T> output{maximum_ms, static_cast<double>(checksum), {}};
    if (collect_values) output.values = gather_primary_values();
    return output;
  }

 private:
  void initialize_rank(std::size_t rank) {
    auto& device = ranks_[rank];
    const auto& host = plan_.rank_plans[rank];
    device.device = devices_[rank];
    CUDA_CHECK(cudaSetDevice(device.device));
    CUDA_CHECK(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&device.start));
    CUDA_CHECK(cudaEventCreate(&device.stop));
    device.values = device_allocate<T>(host.global_nodes.size());
    device.global_ids = device_copy(host.global_nodes, device.stream);
    device.level_nodes = device_copy(host.level_nodes, device.stream);
    device.parent_refs = device_copy(host.parent_refs, device.stream);
    device.send_slots = device_copy(host.send_local_slots, device.stream);
    device.primary_slots = device_copy(host.primary_slots, device.stream);
    device.send_buffer = device_allocate<T>(host.max_send_values);
    device.recv_buffer = device_allocate<T>(host.max_recv_values);
    device.checksum = device_allocate<T>(1);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
  }

  std::vector<T> gather_primary_values() {
    std::vector<T> global(plan_.nodes, std::numeric_limits<T>::quiet_NaN());
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      const auto& host = plan_.rank_plans[rank];
      std::vector<T> local(host.global_nodes.size());
      CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
      if (!local.empty()) {
        CUDA_CHECK(cudaMemcpy(local.data(), ranks_[rank].values, local.size() * sizeof(T),
                              cudaMemcpyDeviceToHost));
      }
      for (std::size_t slot = 0; slot < local.size(); ++slot) {
        if (host.primary[slot]) global[host.global_nodes[slot]] = local[slot];
      }
    }
    if (std::any_of(global.begin(), global.end(), [](T value) { return !std::isfinite(value); })) {
      throw std::runtime_error("gathered output contains missing or non-finite values");
    }
    return global;
  }

  void cleanup() noexcept {
    for (auto& rank : ranks_) {
      if (rank.device < 0) continue;
      cudaSetDevice(rank.device);
      if (rank.stream) cudaStreamSynchronize(rank.stream);
      cudaFree(rank.values);
      cudaFree(rank.global_ids);
      cudaFree(rank.level_nodes);
      cudaFree(rank.parent_refs);
      cudaFree(rank.send_slots);
      cudaFree(rank.primary_slots);
      cudaFree(rank.send_buffer);
      cudaFree(rank.recv_buffer);
      cudaFree(rank.checksum);
      if (rank.start) cudaEventDestroy(rank.start);
      if (rank.stop) cudaEventDestroy(rank.stop);
      if (rank.stream) cudaStreamDestroy(rank.stream);
      rank.device = -1;
    }
    for (auto& comm : comms_) {
      if (comm) ncclCommDestroy(comm);
      comm = nullptr;
    }
  }

  const gs::ExecutionPlan& plan_;
  std::vector<int> devices_;
  std::vector<DeviceRank<T>> ranks_;
  std::vector<ncclComm_t> comms_;
};

struct DeviceInfo {
  int visible_id = 0;
  std::string name;
  std::size_t total_memory = 0;
  int major = 0;
  int minor = 0;
  std::string pci_bus_id;
};

std::vector<DeviceInfo> inspect_devices(const std::vector<int>& devices) {
  std::vector<DeviceInfo> result;
  for (int device : devices) {
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    char pci[32]{};
    CUDA_CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), device));
    result.push_back({device, properties.name, properties.totalGlobalMem,
                      properties.major, properties.minor, pci});
  }
  return result;
}

template <typename T>
bool plan_fits(const gs::ExecutionPlan& plan, const std::vector<int>& devices,
               const std::vector<std::size_t>& bytes, std::string& reason) {
  for (std::size_t rank = 0; rank < devices.size(); ++rank) {
    CUDA_CHECK(cudaSetDevice(devices[rank]));
    std::size_t free = 0;
    std::size_t total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    if (bytes[rank] > static_cast<std::size_t>(0.8 * static_cast<double>(free))) {
      std::ostringstream text;
      text << "rank " << rank << " requires approximately " << bytes[rank]
           << " bytes, exceeding the 80% free-memory safety limit of "
           << static_cast<std::size_t>(0.8 * static_cast<double>(free));
      reason = text.str();
      return false;
    }
  }
  return true;
}

struct BenchmarkResult {
  std::string strategy;
  std::string status = "ok";
  std::string reason;
  gs::PlanStats stats;
  std::vector<std::size_t> allocated_bytes;
  std::vector<double> runtime_ms;
  double checksum = 0;
  bool validated = false;
  double max_abs_error = 0;
  double max_rel_error = 0;
};

double percentile(std::vector<double> values, double q) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const double position = q * (values.size() - 1);
  const auto low = static_cast<std::size_t>(std::floor(position));
  const auto high = static_cast<std::size_t>(std::ceil(position));
  const double weight = position - low;
  return values[low] * (1.0 - weight) + values[high] * weight;
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (char c : value) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << c;
    }
  }
  return out.str();
}

template <typename T>
void emit_array(std::ostringstream& out, const std::vector<T>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}

template <typename T>
struct CompletedResult {
  BenchmarkResult report;
  std::vector<T> output;
};

template <typename T>
CompletedResult<T> benchmark_plan(const gs::ExecutionPlan& plan, const Options& options,
                                  const std::vector<int>& devices) {
  CompletedResult<T> result;
  result.report.strategy = plan.strategy;
  result.report.stats = plan.stats;
  result.report.allocated_bytes = gs::estimate_device_bytes(plan, sizeof(T));
  if (!plan_fits<T>(plan, devices, result.report.allocated_bytes, result.report.reason)) {
    result.report.status = "skipped";
    return result;
  }
  Executor<T> executor(plan, devices);
  for (int i = 0; i < options.warmup; ++i) {
    (void)executor.run(options.seed, static_cast<std::uint64_t>(i), false);
  }
  for (int i = 0; i < options.runs; ++i) {
    const bool last = i + 1 == options.runs;
    auto output = executor.run(options.seed,
                               static_cast<std::uint64_t>(options.warmup + i), last);
    result.report.runtime_ms.push_back(output.milliseconds);
    result.report.checksum = output.checksum;
    if (last) result.output = std::move(output.values);
  }
  return result;
}

template <typename T>
void compare_to_reference(CompletedResult<T>& result, const std::vector<T>& reference) {
  if (result.report.status != "ok") return;
  if (result.output.size() != reference.size()) throw std::logic_error("validation output size mismatch");
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double actual = static_cast<double>(result.output[i]);
    const double expected = static_cast<double>(reference[i]);
    const double absolute = std::abs(actual - expected);
    const double relative = absolute / std::max(1.0, std::abs(expected));
    result.report.max_abs_error = std::max(result.report.max_abs_error, absolute);
    result.report.max_rel_error = std::max(result.report.max_rel_error, relative);
  }
  const double tolerance = std::is_same_v<T, float> ? 2e-5 : 1e-12;
  result.report.validated = result.report.max_rel_error <= tolerance;
  if (!result.report.validated) {
    result.report.status = "validation_failed";
    result.report.reason = "distributed output differs from the single-GPU CUDA reference";
  }
}

std::string version_string(int version, int major_scale, int minor_scale) {
  std::ostringstream out;
  out << version / major_scale << '.' << (version % major_scale) / minor_scale;
  const int patch = version % minor_scale;
  if (patch) out << '.' << patch;
  return out.str();
}

template <typename T>
std::string render_json(const Options& options, const gs::Graph& graph,
                        const std::vector<DeviceInfo>& devices, int driver_version,
                        int runtime_version, int nccl_version, double graph_setup_seconds,
                        double plan_setup_seconds,
                        const std::vector<CompletedResult<T>>& completed) {
  const auto non_root_edges = (graph.config.nodes - graph.level_offsets[1]) * graph.config.parents;
  const double local_fraction = non_root_edges == 0 ? 0.0 :
      static_cast<double>(graph.local_parent_edges) / non_root_edges;
  std::ostringstream out;
  out << std::setprecision(12);
  out << "{\n  \"configuration\": {\"nodes\":" << options.nodes
      << ",\"parents\":" << options.parents << ",\"levels\":" << options.levels
      << ",\"gpus\":" << devices.size() << ",\"communities\":" << graph.config.communities
      << ",\"locality\":" << options.locality << ",\"seed\":" << options.seed
      << ",\"dtype\":\"" << options.dtype << "\",\"warmup\":" << options.warmup
      << ",\"runs\":" << options.runs << "},\n";
  out << "  \"environment\": {\"cuda_driver\":\""
      << version_string(driver_version, 1000, 10) << "\",\"cuda_runtime\":\""
      << version_string(runtime_version, 1000, 10) << "\",\"nccl\":\""
      << version_string(nccl_version, 10000, 100) << "\",\"devices\":[";
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (i) out << ',';
    const auto& d = devices[i];
    out << "{\"visible_id\":" << d.visible_id << ",\"name\":\"" << json_escape(d.name)
        << "\",\"total_memory_bytes\":" << d.total_memory << ",\"compute_capability\":\""
        << d.major << '.' << d.minor << "\",\"pci_bus_id\":\"" << d.pci_bus_id << "\"}";
  }
  out << "]},\n";
  out << "  \"setup_seconds\": {\"graph\":" << graph_setup_seconds
      << ",\"partitions_and_plans\":" << plan_setup_seconds << "},\n";
  out << "  \"graph\": {\"root_nodes\":" << graph.level_offsets[1]
      << ",\"actual_local_parent_fraction\":" << local_fraction << "},\n";
  out << "  \"results\": [\n";
  for (std::size_t i = 0; i < completed.size(); ++i) {
    if (i) out << ",\n";
    const auto& result = completed[i].report;
    out << "    {\"strategy\":\"" << result.strategy << "\",\"status\":\""
        << result.status << "\"";
    if (!result.reason.empty()) out << ",\"reason\":\"" << json_escape(result.reason) << "\"";
    out << ",\"owned_nodes\":";
    emit_array(out, result.stats.owned_nodes);
    out << ",\"stored_nodes\":";
    emit_array(out, result.stats.stored_nodes);
    const auto evaluated_nodes = std::accumulate(result.stats.stored_nodes.begin(),
                                                 result.stats.stored_nodes.end(),
                                                 std::size_t{0});
    out << ",\"evaluated_nodes\":" << evaluated_nodes;
    out << ",\"load_imbalance\":" << result.stats.load_imbalance
        << ",\"primary_cut_edges\":" << result.stats.cut_edges
        << ",\"communicated_values\":" << result.stats.communicated_values
        << ",\"communicated_bytes\":" << result.stats.communicated_values * sizeof(T)
        << ",\"messages\":" << result.stats.messages
        << ",\"replication_factor\":" << result.stats.replication_factor
        << ",\"allocated_bytes_per_rank\":";
    emit_array(out, result.allocated_bytes);
    if (result.status != "skipped") {
      out << ",\"checksum\":" << result.checksum << ",\"validated\":"
          << (result.validated ? "true" : "false")
          << ",\"max_abs_error\":" << result.max_abs_error
          << ",\"max_rel_error\":" << result.max_rel_error
          << ",\"runtime_ms\":";
      emit_array(out, result.runtime_ms);
      const double median = percentile(result.runtime_ms, 0.5);
      out << ",\"runtime_min_ms\":" << *std::min_element(result.runtime_ms.begin(), result.runtime_ms.end())
          << ",\"runtime_median_ms\":" << median
          << ",\"runtime_p95_ms\":" << percentile(result.runtime_ms, 0.95)
          << ",\"nodes_per_second\":" << options.nodes * 1000.0 / median
          << ",\"evaluated_nodes_per_second\":" << evaluated_nodes * 1000.0 / median
          << ",\"primary_edges_per_second\":"
          << non_root_edges * 1000.0 / median;
    }
    out << '}';
  }
  out << "\n  ]\n}\n";
  return out.str();
}

template <typename T>
int run_benchmark(const Options& options, const std::vector<int>& devices,
                  const std::vector<DeviceInfo>& device_info, int driver_version,
                  int runtime_version, int nccl_version) {
  const auto graph_start = std::chrono::steady_clock::now();
  gs::GraphConfig graph_config{options.nodes, options.parents, options.levels,
                               options.communities == 0 ? devices.size() : options.communities,
                               options.locality, options.seed};
  gs::Graph graph = gs::generate_graph(graph_config);
  const auto graph_end = std::chrono::steady_clock::now();

  const auto plan_start = std::chrono::steady_clock::now();
  auto random_partition = gs::make_random_balanced_partition(graph, devices.size(), options.seed ^ 0xA5A5A5A5ULL);
  auto affinity_partition = gs::make_parent_affinity_partition(graph, devices.size(), options.seed ^ 0x5A5A5A5AULL);
  std::vector<gs::ExecutionPlan> plans;
  if (options.strategy == "all" || options.strategy == "random") {
    plans.push_back(gs::make_exchange_plan(graph, random_partition));
  }
  if (options.strategy == "all" || options.strategy == "affinity") {
    plans.push_back(gs::make_exchange_plan(graph, affinity_partition));
  }
  if (options.strategy == "all" || options.strategy == "closure") {
    plans.push_back(gs::make_ancestor_closure_plan(graph, affinity_partition));
  }
  const auto plan_end = std::chrono::steady_clock::now();

  std::vector<CompletedResult<T>> completed;
  completed.reserve(plans.size());
  for (const auto& plan : plans) {
    std::cerr << "benchmarking " << plan.strategy << " on " << devices.size() << " GPU(s)\n";
    completed.push_back(benchmark_plan<T>(plan, options, devices));
  }

  // The same kernel and RNG on one GPU provide the reference, avoiding a second
  // host implementation of cuRAND's Philox mapping.
  auto reference_partition = gs::make_random_balanced_partition(graph, 1, 0);
  auto reference_plan = gs::make_exchange_plan(graph, reference_partition);
  reference_plan.strategy = "reference";
  std::vector<int> reference_device{devices.front()};
  auto reference_bytes = gs::estimate_device_bytes(reference_plan, sizeof(T));
  std::string reference_reason;
  if (plan_fits<T>(reference_plan, reference_device, reference_bytes, reference_reason)) {
    Executor<T> reference(reference_plan, reference_device);
    auto output = reference.run(options.seed,
                                static_cast<std::uint64_t>(options.warmup + options.runs - 1), true);
    for (auto& result : completed) compare_to_reference(result, output.values);
  } else {
    for (auto& result : completed) {
      if (result.report.status == "ok") result.report.reason = "validation skipped: " + reference_reason;
    }
  }

  const double graph_seconds = std::chrono::duration<double>(graph_end - graph_start).count();
  const double plan_seconds = std::chrono::duration<double>(plan_end - plan_start).count();
  const std::string json = render_json(options, graph, device_info, driver_version,
                                       runtime_version, nccl_version, graph_seconds,
                                       plan_seconds, completed);
  std::cout << json;
  if (!options.output.empty()) {
    std::ofstream stream(options.output);
    if (!stream) throw std::runtime_error("cannot open output file: " + options.output);
    stream << json;
  }
  return std::all_of(completed.begin(), completed.end(), [](const auto& result) {
           return result.report.status == "ok" || result.report.status == "skipped";
         }) ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_options(argc, argv);
    int available = 0;
    CUDA_CHECK(cudaGetDeviceCount(&available));
    if (available == 0) throw std::runtime_error("no visible CUDA devices");
    const int count = options.gpus == 0 ? available : options.gpus;
    if (count > available) throw std::invalid_argument("--gpus exceeds visible CUDA device count");
    std::vector<int> devices(count);
    std::iota(devices.begin(), devices.end(), 0);
    auto info = inspect_devices(devices);
    int driver_version = 0;
    int runtime_version = 0;
    int nccl_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    NCCL_CHECK(ncclGetVersion(&nccl_version));
    if (options.dtype == "float") {
      return run_benchmark<float>(options, devices, info, driver_version, runtime_version, nccl_version);
    }
    return run_benchmark<double>(options, devices, info, driver_version, runtime_version, nccl_version);
  } catch (const std::exception& error) {
    std::cerr << "cuda_graph_sharding_bench: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
