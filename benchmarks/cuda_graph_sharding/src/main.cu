#include "graph_sharding/planner.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <nccl.h>

#if defined(GRAPH_SHARDING_HAS_NVTX)
#include <nvtx3/nvToolsExt.h>
#endif

#include <algorithm>
#include <array>
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
  std::string mode = "forward";
  int warmup = 5;
  int runs = 20;
  int evaluations_per_run = 1;
  double learning_rate = 0.01;
  int compute_iters = 0;
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
  --mode MODE         forward|iterative (default forward)
  --warmup N          Untimed runs per strategy (default 5)
  --runs N            Timed runs per strategy (default 20)
  --evaluations-per-run N  Complete DAG evaluations in each timing sample (default 1)
  --learning-rate X   Iterative-mode SGD step size (default 0.01)
  --compute-iters N   Extra affine transforms per node in both passes (default 0)
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
    else if (key == "--mode") result.mode = value;
    else if (key == "--warmup") result.warmup = parse_number<int>(value, "--warmup");
    else if (key == "--runs") result.runs = parse_number<int>(value, "--runs");
    else if (key == "--evaluations-per-run") {
      result.evaluations_per_run = parse_number<int>(value, "--evaluations-per-run");
    } else if (key == "--learning-rate") {
      result.learning_rate = parse_number<double>(value, "--learning-rate");
    } else if (key == "--compute-iters") {
      result.compute_iters = parse_number<int>(value, "--compute-iters");
    }
    else if (key == "--output") result.output = value;
    else throw std::invalid_argument("unknown option: " + key);
  }
  if (result.gpus < 0) throw std::invalid_argument("--gpus must be non-negative");
  if (result.warmup < 0) throw std::invalid_argument("--warmup must be non-negative");
  if (result.runs <= 0) throw std::invalid_argument("--runs must be positive");
  if (result.evaluations_per_run <= 0) {
    throw std::invalid_argument("--evaluations-per-run must be positive");
  }
  if (!(result.learning_rate >= 0.0) || !std::isfinite(result.learning_rate)) {
    throw std::invalid_argument("--learning-rate must be finite and non-negative");
  }
  if (result.compute_iters < 0) throw std::invalid_argument("--compute-iters must be non-negative");
  if (result.dtype != "float" && result.dtype != "double") {
    throw std::invalid_argument("--dtype must be float or double");
  }
  if (result.strategy != "all" && result.strategy != "random" &&
      result.strategy != "affinity" && result.strategy != "closure") {
    throw std::invalid_argument("--strategy must be all, random, affinity, or closure");
  }
  if (result.mode != "forward" && result.mode != "iterative") {
    throw std::invalid_argument("--mode must be forward or iterative");
  }
  if (result.mode == "iterative" && result.strategy == "random") {
    throw std::invalid_argument("iterative mode intentionally supports affinity and closure only");
  }
  return result;
}

class NvtxRange {
 public:
  explicit NvtxRange(const char* name) {
#if defined(GRAPH_SHARDING_HAS_NVTX)
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }
  ~NvtxRange() {
#if defined(GRAPH_SHARDING_HAS_NVTX)
    nvtxRangePop();
#endif
  }
};

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
__device__ T device_fma(T a, T b, T c);

template <>
__device__ float device_fma<float>(float a, float b, float c) {
  return fmaf(a, b, c);
}

template <>
__device__ double device_fma<double>(double a, double b, double c) {
  return fma(a, b, c);
}

template <typename T>
__device__ T synthetic_transform(T value, std::uint32_t node, int iterations) {
  const T multiplier = static_cast<T>(0.999999);
  const T offset = static_cast<T>((static_cast<int>(node % 7) - 3) * 1.0e-7);
  for (int i = 0; i < iterations; ++i) value = device_fma(value, multiplier, offset);
  return value;
}

template <typename T>
__device__ T synthetic_derivative(int iterations) {
  const T multiplier = static_cast<T>(0.999999);
  T derivative = static_cast<T>(1);
  for (int i = 0; i < iterations; ++i) {
    derivative = device_fma(derivative, multiplier, T{0});
  }
  return derivative;
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
                              std::uint64_t seed, std::uint64_t iteration,
                              int compute_iters) {
  std::size_t local_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_row >= row_count) return;
  const std::size_t row = row_begin + local_row;
  const std::uint32_t slot = level_nodes[row];
  T value = node_random<T>(seed, iteration, global_ids[slot]);
  for (std::size_t j = 0; j < active_parents; ++j) {
    const std::int32_t ref = parent_refs[row * parents_per_node + j];
    value += ref >= 0 ? values[ref] : received[-1 - ref];
  }
  values[slot] = synthetic_transform(value, global_ids[slot], compute_iters);
}

template <typename T>
__global__ void initialize_latents_kernel(const std::uint32_t* global_ids,
                                          const std::uint32_t* primary_slots,
                                          std::size_t count, T* latents, T* gradients,
                                          std::uint64_t seed) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const auto slot = primary_slots[i];
  latents[slot] = node_random<T>(seed, 0, global_ids[slot]);
  gradients[slot] = T{0};
}

template <typename T>
__global__ void latent_update_kernel(const std::uint32_t* primary_slots, std::size_t count,
                                     T learning_rate, const T* gradients, T* latents) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const auto slot = primary_slots[i];
  latents[slot] -= learning_rate * gradients[slot];
}

template <typename T>
__global__ void scatter_kernel(const T* packed, const std::uint32_t* slots, T* values,
                               std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) values[slots[i]] = packed[i];
}

template <typename T>
__global__ void scatter_add_kernel(const T* packed, const std::uint32_t* slots, T* values,
                                   std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) atomicAdd(values + slots[i], packed[i]);
}

template <typename T>
__global__ void iterative_forward_kernel(
    const std::uint32_t* global_ids, const std::uint32_t* level_nodes,
    const std::int32_t* parent_refs, std::size_t row_begin, std::size_t row_count,
    std::size_t parents_per_node, std::size_t active_parents, const T* received,
    const T* latents, T* values, int compute_iters) {
  const std::size_t local_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_row >= row_count) return;
  const std::size_t row = row_begin + local_row;
  const auto slot = level_nodes[row];
  T value = latents[slot];
  const T parent_scale = active_parents == 0 ? T{0} : T{1} / static_cast<T>(active_parents);
  for (std::size_t j = 0; j < active_parents; ++j) {
    const auto ref = parent_refs[row * parents_per_node + j];
    value += parent_scale * (ref >= 0 ? values[ref] : received[-1 - ref]);
  }
  values[slot] = synthetic_transform(value, global_ids[slot], compute_iters);
}

template <typename T>
__global__ void seed_adjoint_kernel(const T* values, const std::uint32_t* primary_slots,
                                    std::size_t count, T inverse_nodes, T* adjoints) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const auto slot = primary_slots[i];
  adjoints[slot] = values[slot] * inverse_nodes;
}

template <typename T>
__global__ void backward_kernel(const std::uint32_t* level_nodes,
                                const std::int32_t* parent_refs, std::size_t row_begin,
                                std::size_t row_count, std::size_t parents_per_node,
                                std::size_t active_parents, T* remote_contributions,
                                T* adjoints, int compute_iters) {
  const std::size_t local_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_row >= row_count) return;
  const std::size_t row = row_begin + local_row;
  const auto slot = level_nodes[row];
  const T gradient = adjoints[slot] * synthetic_derivative<T>(compute_iters);
  adjoints[slot] = gradient;
  if (active_parents == 0) return;
  const T contribution = gradient / static_cast<T>(active_parents);
  for (std::size_t j = 0; j < active_parents; ++j) {
    const auto ref = parent_refs[row * parents_per_node + j];
    if (ref >= 0) {
      atomicAdd(adjoints + ref, contribution);
    } else {
      atomicAdd(remote_contributions + (-1 - ref), contribution);
    }
  }
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
__global__ void partial_loss_kernel(const T* values, const std::uint32_t* primary_slots,
                                    std::size_t count, T scale, T* result) {
  T value = 0;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    const T output = values[primary_slots[i]];
    value += scale * output * output;
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
  Executor(const gs::ExecutionPlan& plan, std::vector<int> devices, int compute_iters = 0)
      : plan_(plan), devices_(std::move(devices)), ranks_(plan.ranks), comms_(plan.ranks),
        compute_iters_(compute_iters) {
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

  RunOutput<T> run(std::uint64_t seed, std::uint64_t iteration_begin, int evaluations,
                   bool collect_values) {
    constexpr int threads = 256;
    for (auto& rank : ranks_) {
      CUDA_CHECK(cudaSetDevice(rank.device));
      CUDA_CHECK(cudaEventRecord(rank.start, rank.stream));
    }

    {
      NvtxRange batch_range("forward/batch");
      for (int evaluation = 0; evaluation < evaluations; ++evaluation) {
        const auto iteration = iteration_begin + static_cast<std::uint64_t>(evaluation);
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
                ranks_[rank].recv_buffer, ranks_[rank].values, seed, iteration,
                compute_iters_);
            CUDA_CHECK(cudaGetLastError());
          }
        }
      }
    }

    // Reporting reduction is performed once per timing batch rather than once
    // per evaluation. This keeps --evaluations-per-run focused on DAG work.
    compute_checksum();

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
  void compute_checksum() {
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
  }

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
  int compute_iters_ = 0;
};

enum class Phase : std::size_t {
  kLatentUpdate = 0,
  kLatentSync = 1,
  kForward = 2,
  kBackward = 3,
  kGradientSync = 4,
  kCount = 5
};

constexpr std::size_t kPhaseCount = static_cast<std::size_t>(Phase::kCount);

struct PhaseTimes {
  double latent_update_ms = 0;
  double latent_sync_ms = 0;
  double forward_ms = 0;
  double backward_ms = 0;
  double gradient_sync_ms = 0;
  double total_ms = 0;
};

struct EventPair {
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
};

template <typename T>
struct IterativeDeviceRank {
  int device = -1;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  T* latents = nullptr;
  T* values = nullptr;
  T* adjoints = nullptr;
  T* send_buffer = nullptr;
  T* recv_buffer = nullptr;
  T* checksum = nullptr;
  std::uint32_t* global_ids = nullptr;
  std::uint32_t* level_nodes = nullptr;
  std::int32_t* parent_refs = nullptr;
  std::uint32_t* send_slots = nullptr;
  std::uint32_t* primary_slots = nullptr;
  std::uint32_t* replica_send_slots = nullptr;
  std::uint32_t* replica_recv_slots = nullptr;
  std::array<std::vector<EventPair>, kPhaseCount> phase_events;
  std::array<std::size_t, kPhaseCount> phase_cursor{};
};

template <typename T>
struct IterativeRunOutput {
  PhaseTimes times;
  double checksum = 0;
  std::vector<T> values;
  std::vector<T> latents;
  std::vector<T> gradients;
};

template <typename T>
class IterativeExecutor {
 public:
  IterativeExecutor(const gs::ExecutionPlan& plan, std::vector<int> devices,
                    T learning_rate, int compute_iters, std::uint64_t seed)
      : plan_(plan), devices_(std::move(devices)), ranks_(plan.ranks), comms_(plan.ranks),
        learning_rate_(learning_rate), compute_iters_(compute_iters), initialization_seed_(seed),
        closure_(plan.strategy == "closure") {
    if (devices_.size() != ranks_.size()) throw std::invalid_argument("device/rank count mismatch");
    if (plan.strategy != "affinity" && plan.strategy != "closure" &&
        plan.strategy != "reference") {
      throw std::invalid_argument("iterative executor requires affinity or closure ownership");
    }
    NCCL_CHECK(ncclCommInitAll(comms_.data(), static_cast<int>(ranks_.size()), devices_.data()));
    try {
      for (std::size_t rank = 0; rank < ranks_.size(); ++rank) initialize_rank(rank);
    } catch (...) {
      cleanup();
      throw;
    }
  }

  IterativeExecutor(const IterativeExecutor&) = delete;
  IterativeExecutor& operator=(const IterativeExecutor&) = delete;
  ~IterativeExecutor() { cleanup(); }

  IterativeRunOutput<T> run(int evaluations, bool collect_values, bool measure = true) {
    if (measure) {
      prepare_events(evaluations);
      for (auto& rank : ranks_) {
        CUDA_CHECK(cudaSetDevice(rank.device));
        CUDA_CHECK(cudaEventRecord(rank.start, rank.stream));
      }
    }

    {
      NvtxRange batch("iterative/batch");
      for (int evaluation = 0; evaluation < evaluations; ++evaluation) {
        maybe_timed(Phase::kLatentUpdate, measure, [&] { update_latents(); });
        if (closure_ && plan_.stats.replica_values > 0) {
          maybe_timed(Phase::kLatentSync, measure, [&] { synchronize_latents(); });
        }
        maybe_timed(Phase::kForward, measure, [&] { forward(); });
        backward(measure);
      }
    }

    PhaseTimes times;
    if (measure) {
      for (auto& rank : ranks_) {
        CUDA_CHECK(cudaSetDevice(rank.device));
        CUDA_CHECK(cudaEventRecord(rank.stop, rank.stream));
      }
      times = collect_times();
    }

    IterativeRunOutput<T> output;
    output.times = times;
    if (!measure && !collect_values) return output;

    compute_checksum();
    T checksum = 0;
    CUDA_CHECK(cudaSetDevice(ranks_[0].device));
    CUDA_CHECK(cudaMemcpy(&checksum, ranks_[0].checksum, sizeof(T), cudaMemcpyDeviceToHost));
    if (!std::isfinite(static_cast<double>(checksum))) {
      throw std::runtime_error("non-finite iterative checksum; reduce depth or learning rate");
    }

    output.checksum = static_cast<double>(checksum);
    if (collect_values) {
      output.values = gather_primary(&IterativeDeviceRank<T>::values);
      output.latents = gather_primary(&IterativeDeviceRank<T>::latents);
      output.gradients = gather_primary(&IterativeDeviceRank<T>::adjoints);
    }
    return output;
  }

 private:
  using ValueMember = T* IterativeDeviceRank<T>::*;

  static std::size_t phase_index(Phase phase) { return static_cast<std::size_t>(phase); }

  void ensure_event_pairs(IterativeDeviceRank<T>& rank, Phase phase, std::size_t count) {
    auto& events = rank.phase_events[phase_index(phase)];
    CUDA_CHECK(cudaSetDevice(rank.device));
    while (events.size() < count) {
      EventPair pair;
      CUDA_CHECK(cudaEventCreate(&pair.begin));
      CUDA_CHECK(cudaEventCreate(&pair.end));
      events.push_back(pair);
    }
  }

  void prepare_events(int evaluations) {
    const std::size_t evals = static_cast<std::size_t>(evaluations);
    for (auto& rank : ranks_) {
      rank.phase_cursor.fill(0);
      ensure_event_pairs(rank, Phase::kLatentUpdate, evals);
      ensure_event_pairs(rank, Phase::kLatentSync,
                         closure_ && plan_.stats.replica_values > 0 ? evals : 0);
      ensure_event_pairs(rank, Phase::kForward, evals);
      ensure_event_pairs(rank, Phase::kBackward,
                         closure_ ? evals : evals * plan_.levels);
      ensure_event_pairs(rank, Phase::kGradientSync,
                         closure_ ? (plan_.stats.replica_values > 0 ? evals : 0)
                                  : evals * plan_.levels);
    }
  }

  void begin_phase(Phase phase) {
    const auto index = phase_index(phase);
    for (auto& rank : ranks_) {
      auto& pair = rank.phase_events[index][rank.phase_cursor[index]];
      CUDA_CHECK(cudaSetDevice(rank.device));
      CUDA_CHECK(cudaEventRecord(pair.begin, rank.stream));
    }
  }

  void end_phase(Phase phase) {
    const auto index = phase_index(phase);
    for (auto& rank : ranks_) {
      auto& pair = rank.phase_events[index][rank.phase_cursor[index]];
      CUDA_CHECK(cudaSetDevice(rank.device));
      CUDA_CHECK(cudaEventRecord(pair.end, rank.stream));
      rank.phase_cursor[index]++;
    }
  }

  template <typename Function>
  void timed_all_ranks(Phase phase, Function&& function) {
    begin_phase(phase);
    function();
    end_phase(phase);
  }

  template <typename Function>
  void maybe_timed(Phase phase, bool measure, Function&& function) {
    if (measure) {
      timed_all_ranks(phase, std::forward<Function>(function));
    } else {
      function();
    }
  }

  void initialize_rank(std::size_t rank_index) {
    constexpr int threads = 256;
    auto& device = ranks_[rank_index];
    const auto& host = plan_.rank_plans[rank_index];
    device.device = devices_[rank_index];
    CUDA_CHECK(cudaSetDevice(device.device));
    CUDA_CHECK(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&device.start));
    CUDA_CHECK(cudaEventCreate(&device.stop));
    device.global_ids = device_copy(host.global_nodes, device.stream);
    device.level_nodes = device_copy(host.level_nodes, device.stream);
    device.parent_refs = device_copy(host.parent_refs, device.stream);
    device.send_slots = device_copy(host.send_local_slots, device.stream);
    device.primary_slots = device_copy(host.primary_slots, device.stream);
    device.replica_send_slots = device_copy(host.replica_send_slots, device.stream);
    device.replica_recv_slots = device_copy(host.replica_recv_slots, device.stream);
    device.latents = device_allocate<T>(host.global_nodes.size());
    device.values = device_allocate<T>(host.global_nodes.size());
    device.adjoints = device_allocate<T>(host.global_nodes.size());
    const auto send_capacity = std::max(host.max_send_values, host.replica_send_slots.size());
    const auto recv_capacity = std::max(host.max_recv_values, host.replica_recv_slots.size());
    device.send_buffer = device_allocate<T>(send_capacity);
    device.recv_buffer = device_allocate<T>(recv_capacity);
    device.checksum = device_allocate<T>(1);
    if (!host.global_nodes.empty()) {
      CUDA_CHECK(cudaMemsetAsync(device.latents, 0, host.global_nodes.size() * sizeof(T),
                                 device.stream));
      CUDA_CHECK(cudaMemsetAsync(device.values, 0, host.global_nodes.size() * sizeof(T),
                                 device.stream));
      CUDA_CHECK(cudaMemsetAsync(device.adjoints, 0, host.global_nodes.size() * sizeof(T),
                                 device.stream));
    }
    const auto primary_count = host.primary_slots.size();
    if (primary_count > 0) {
      initialize_latents_kernel<<<static_cast<unsigned int>((primary_count + threads - 1) / threads),
                                  threads, 0, device.stream>>>(
          device.global_ids, device.primary_slots, primary_count, device.latents,
          device.adjoints, initialization_seed_);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
  }

  void update_latents() {
    constexpr int threads = 256;
    NvtxRange range("iterative/latent_update");
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      const auto count = plan_.rank_plans[rank].primary_slots.size();
      if (count == 0) continue;
      CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
      latent_update_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads),
                             threads, 0, ranks_[rank].stream>>>(
          ranks_[rank].primary_slots, count, learning_rate_, ranks_[rank].adjoints,
          ranks_[rank].latents);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  void synchronize_latents() {
    constexpr int threads = 256;
    NvtxRange range("iterative/latent_sync");
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      const auto count = plan_.rank_plans[rank].replica_send_slots.size();
      if (count == 0) continue;
      CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
      pack_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads, 0,
                    ranks_[rank].stream>>>(ranks_[rank].latents,
                                           ranks_[rank].replica_send_slots,
                                           ranks_[rank].send_buffer, count);
      CUDA_CHECK(cudaGetLastError());
    }
    NCCL_CHECK(ncclGroupStart());
    for (std::size_t src = 0; src < ranks_.size(); ++src) {
      const auto& send = plan_.rank_plans[src].replica_send_offsets;
      for (std::size_t dst = 0; dst < ranks_.size(); ++dst) {
        if (src == dst) continue;
        const auto count = send[dst + 1] - send[dst];
        if (count == 0) continue;
        const auto& recv = plan_.rank_plans[dst].replica_recv_offsets;
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
      const auto count = plan_.rank_plans[rank].replica_recv_slots.size();
      if (count == 0) continue;
      CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
      scatter_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads, 0,
                       ranks_[rank].stream>>>(ranks_[rank].recv_buffer,
                                              ranks_[rank].replica_recv_slots,
                                              ranks_[rank].latents, count);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  void forward() {
    constexpr int threads = 256;
    NvtxRange range("iterative/forward");
    for (std::size_t level = 0; level < plan_.levels; ++level) {
      if (!closure_) {
        for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
          const auto& lp = plan_.rank_plans[rank].levels[level];
          const auto count = lp.send_slot_end - lp.send_slot_begin;
          if (count == 0) continue;
          CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
          pack_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads, 0,
                        ranks_[rank].stream>>>(
              ranks_[rank].values, ranks_[rank].send_slots + lp.send_slot_begin,
              ranks_[rank].send_buffer, count);
          CUDA_CHECK(cudaGetLastError());
        }
        exchange_forward_level(level);
      }
      for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
        const auto& lp = plan_.rank_plans[rank].levels[level];
        const auto count = lp.node_end - lp.node_begin;
        if (count == 0) continue;
        CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
        iterative_forward_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads),
                                   threads, 0, ranks_[rank].stream>>>(
            ranks_[rank].global_ids, ranks_[rank].level_nodes, ranks_[rank].parent_refs,
            lp.node_begin, count, plan_.parents, level == 0 ? 0 : plan_.parents,
            ranks_[rank].recv_buffer, ranks_[rank].latents, ranks_[rank].values,
            compute_iters_);
        CUDA_CHECK(cudaGetLastError());
      }
    }
  }

  void exchange_forward_level(std::size_t level) {
    NCCL_CHECK(ncclGroupStart());
    for (std::size_t src = 0; src < ranks_.size(); ++src) {
      const auto& send = plan_.rank_plans[src].levels[level].send_offsets;
      for (std::size_t dst = 0; dst < ranks_.size(); ++dst) {
        if (src == dst) continue;
        const auto count = send[dst + 1] - send[dst];
        if (count == 0) continue;
        const auto& recv = plan_.rank_plans[dst].levels[level].recv_offsets;
        CUDA_CHECK(cudaSetDevice(ranks_[src].device));
        NCCL_CHECK(ncclSend(ranks_[src].send_buffer + send[dst], count, nccl_type<T>(),
                            static_cast<int>(dst), comms_[src], ranks_[src].stream));
        CUDA_CHECK(cudaSetDevice(ranks_[dst].device));
        NCCL_CHECK(ncclRecv(ranks_[dst].recv_buffer + recv[src], count, nccl_type<T>(),
                            static_cast<int>(src), comms_[dst], ranks_[dst].stream));
      }
    }
    NCCL_CHECK(ncclGroupEnd());
  }

  void reset_and_seed_adjoints() {
    constexpr int threads = 256;
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      const auto stored = plan_.rank_plans[rank].global_nodes.size();
      CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
      if (stored > 0) {
        CUDA_CHECK(cudaMemsetAsync(ranks_[rank].adjoints, 0, stored * sizeof(T),
                                   ranks_[rank].stream));
      }
      const auto count = plan_.rank_plans[rank].primary_slots.size();
      if (count == 0) continue;
      seed_adjoint_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads),
                            threads, 0, ranks_[rank].stream>>>(
          ranks_[rank].values, ranks_[rank].primary_slots, count,
          T{1} / static_cast<T>(plan_.nodes), ranks_[rank].adjoints);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  void backward(bool measure) {
    constexpr int threads = 256;
    NvtxRange range("iterative/backward");
    if (closure_) {
      maybe_timed(Phase::kBackward, measure, [&] {
        reset_and_seed_adjoints();
        for (std::size_t level = plan_.levels; level-- > 0;) {
          for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
            const auto& lp = plan_.rank_plans[rank].levels[level];
            const auto count = lp.node_end - lp.node_begin;
            if (count == 0) continue;
            CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
            backward_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads),
                              threads, 0, ranks_[rank].stream>>>(
                ranks_[rank].level_nodes, ranks_[rank].parent_refs, lp.node_begin, count,
                plan_.parents, level == 0 ? 0 : plan_.parents, nullptr,
                ranks_[rank].adjoints, compute_iters_);
            CUDA_CHECK(cudaGetLastError());
          }
        }
      });
      if (plan_.stats.replica_values > 0) {
        maybe_timed(Phase::kGradientSync, measure, [&] { reduce_closure_gradients(); });
      }
      return;
    }

    bool first_level = true;
    for (std::size_t level = plan_.levels; level-- > 0;) {
      maybe_timed(Phase::kBackward, measure, [&] {
        if (first_level) reset_and_seed_adjoints();
        for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
          const auto& lp = plan_.rank_plans[rank].levels[level];
          const auto remote_count = lp.recv_offsets.back();
          CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
          if (remote_count > 0) {
            CUDA_CHECK(cudaMemsetAsync(ranks_[rank].recv_buffer, 0,
                                       remote_count * sizeof(T), ranks_[rank].stream));
          }
          const auto count = lp.node_end - lp.node_begin;
          if (count == 0) continue;
          backward_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads),
                            threads, 0, ranks_[rank].stream>>>(
              ranks_[rank].level_nodes, ranks_[rank].parent_refs, lp.node_begin, count,
              plan_.parents, level == 0 ? 0 : plan_.parents, ranks_[rank].recv_buffer,
              ranks_[rank].adjoints, compute_iters_);
          CUDA_CHECK(cudaGetLastError());
        }
      });
      first_level = false;
      if (level_has_exchange(level)) {
        maybe_timed(Phase::kGradientSync, measure,
                    [&] { reduce_affinity_level(level); });
      }
    }
  }

  void reduce_affinity_level(std::size_t level) {
    constexpr int threads = 256;
    NvtxRange range("iterative/gradient_sync_level");
    NCCL_CHECK(ncclGroupStart());
    for (std::size_t owner = 0; owner < ranks_.size(); ++owner) {
      const auto& send = plan_.rank_plans[owner].levels[level].send_offsets;
      for (std::size_t child_rank = 0; child_rank < ranks_.size(); ++child_rank) {
        if (owner == child_rank) continue;
        const auto count = send[child_rank + 1] - send[child_rank];
        if (count == 0) continue;
        const auto& recv = plan_.rank_plans[child_rank].levels[level].recv_offsets;
        CUDA_CHECK(cudaSetDevice(ranks_[child_rank].device));
        NCCL_CHECK(ncclSend(ranks_[child_rank].recv_buffer + recv[owner], count,
                            nccl_type<T>(), static_cast<int>(owner), comms_[child_rank],
                            ranks_[child_rank].stream));
        CUDA_CHECK(cudaSetDevice(ranks_[owner].device));
        NCCL_CHECK(ncclRecv(ranks_[owner].send_buffer + send[child_rank], count,
                            nccl_type<T>(), static_cast<int>(child_rank), comms_[owner],
                            ranks_[owner].stream));
      }
    }
    NCCL_CHECK(ncclGroupEnd());
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      const auto& lp = plan_.rank_plans[rank].levels[level];
      const auto count = lp.send_slot_end - lp.send_slot_begin;
      if (count == 0) continue;
      CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
      scatter_add_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads),
                           threads, 0, ranks_[rank].stream>>>(
          ranks_[rank].send_buffer, ranks_[rank].send_slots + lp.send_slot_begin,
          ranks_[rank].adjoints, count);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  bool level_has_exchange(std::size_t level) const {
    for (const auto& rank : plan_.rank_plans) {
      if (rank.levels[level].send_slot_end != rank.levels[level].send_slot_begin) return true;
    }
    return false;
  }

  void reduce_closure_gradients() {
    constexpr int threads = 256;
    NvtxRange range("iterative/gradient_sync");
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      const auto count = plan_.rank_plans[rank].replica_recv_slots.size();
      if (count == 0) continue;
      CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
      pack_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads, 0,
                    ranks_[rank].stream>>>(ranks_[rank].adjoints,
                                           ranks_[rank].replica_recv_slots,
                                           ranks_[rank].recv_buffer, count);
      CUDA_CHECK(cudaGetLastError());
    }
    NCCL_CHECK(ncclGroupStart());
    for (std::size_t owner = 0; owner < ranks_.size(); ++owner) {
      const auto& send = plan_.rank_plans[owner].replica_send_offsets;
      for (std::size_t replica = 0; replica < ranks_.size(); ++replica) {
        if (owner == replica) continue;
        const auto count = send[replica + 1] - send[replica];
        if (count == 0) continue;
        const auto& recv = plan_.rank_plans[replica].replica_recv_offsets;
        CUDA_CHECK(cudaSetDevice(ranks_[replica].device));
        NCCL_CHECK(ncclSend(ranks_[replica].recv_buffer + recv[owner], count,
                            nccl_type<T>(), static_cast<int>(owner), comms_[replica],
                            ranks_[replica].stream));
        CUDA_CHECK(cudaSetDevice(ranks_[owner].device));
        NCCL_CHECK(ncclRecv(ranks_[owner].send_buffer + send[replica], count,
                            nccl_type<T>(), static_cast<int>(replica), comms_[owner],
                            ranks_[owner].stream));
      }
    }
    NCCL_CHECK(ncclGroupEnd());
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      const auto count = plan_.rank_plans[rank].replica_send_slots.size();
      if (count == 0) continue;
      CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
      scatter_add_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads),
                           threads, 0, ranks_[rank].stream>>>(
          ranks_[rank].send_buffer, ranks_[rank].replica_send_slots,
          ranks_[rank].adjoints, count);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  PhaseTimes collect_times() {
    PhaseTimes result;
    for (auto& rank : ranks_) {
      CUDA_CHECK(cudaSetDevice(rank.device));
      CUDA_CHECK(cudaEventSynchronize(rank.stop));
      float total = 0;
      CUDA_CHECK(cudaEventElapsedTime(&total, rank.start, rank.stop));
      result.total_ms = std::max(result.total_ms, static_cast<double>(total));
    }
    std::array<double*, kPhaseCount> destinations{
        &result.latent_update_ms, &result.latent_sync_ms, &result.forward_ms,
        &result.backward_ms, &result.gradient_sync_ms};
    for (std::size_t phase = 0; phase < kPhaseCount; ++phase) {
      double maximum = 0;
      for (auto& rank : ranks_) {
        double sum = 0;
        CUDA_CHECK(cudaSetDevice(rank.device));
        for (std::size_t i = 0; i < rank.phase_cursor[phase]; ++i) {
          float elapsed = 0;
          CUDA_CHECK(cudaEventElapsedTime(&elapsed, rank.phase_events[phase][i].begin,
                                          rank.phase_events[phase][i].end));
          sum += elapsed;
        }
        maximum = std::max(maximum, sum);
      }
      *destinations[phase] = maximum;
    }
    return result;
  }

  void compute_checksum() {
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
      CUDA_CHECK(cudaMemsetAsync(ranks_[rank].checksum, 0, sizeof(T), ranks_[rank].stream));
      const auto count = plan_.rank_plans[rank].primary_slots.size();
      const auto blocks = static_cast<unsigned int>(std::min<std::size_t>(1024, (count + 255) / 256));
      if (blocks > 0) {
        partial_loss_kernel<<<blocks, 256, 0, ranks_[rank].stream>>>(
            ranks_[rank].values, ranks_[rank].primary_slots, count,
            T{0.5} / static_cast<T>(plan_.nodes), ranks_[rank].checksum);
        CUDA_CHECK(cudaGetLastError());
      }
    }
    NCCL_CHECK(ncclGroupStart());
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
      NCCL_CHECK(ncclAllReduce(ranks_[rank].checksum, ranks_[rank].checksum, 1,
                               nccl_type<T>(), ncclSum, comms_[rank],
                               ranks_[rank].stream));
    }
    NCCL_CHECK(ncclGroupEnd());
  }

  std::vector<T> gather_primary(ValueMember member) {
    std::vector<T> global(plan_.nodes, std::numeric_limits<T>::quiet_NaN());
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      const auto& host = plan_.rank_plans[rank];
      std::vector<T> local(host.global_nodes.size());
      CUDA_CHECK(cudaSetDevice(ranks_[rank].device));
      if (!local.empty()) {
        CUDA_CHECK(cudaMemcpy(local.data(), ranks_[rank].*member,
                              local.size() * sizeof(T), cudaMemcpyDeviceToHost));
      }
      for (std::size_t slot = 0; slot < local.size(); ++slot) {
        if (host.primary[slot]) global[host.global_nodes[slot]] = local[slot];
      }
    }
    if (std::any_of(global.begin(), global.end(), [](T value) { return !std::isfinite(value); })) {
      throw std::runtime_error("gathered iterative vector contains missing or non-finite values");
    }
    return global;
  }

  void cleanup() noexcept {
    for (auto& rank : ranks_) {
      if (rank.device < 0) continue;
      cudaSetDevice(rank.device);
      if (rank.stream) cudaStreamSynchronize(rank.stream);
      cudaFree(rank.latents);
      cudaFree(rank.values);
      cudaFree(rank.adjoints);
      cudaFree(rank.send_buffer);
      cudaFree(rank.recv_buffer);
      cudaFree(rank.checksum);
      cudaFree(rank.global_ids);
      cudaFree(rank.level_nodes);
      cudaFree(rank.parent_refs);
      cudaFree(rank.send_slots);
      cudaFree(rank.primary_slots);
      cudaFree(rank.replica_send_slots);
      cudaFree(rank.replica_recv_slots);
      for (auto& phase : rank.phase_events) {
        for (auto& pair : phase) {
          if (pair.begin) cudaEventDestroy(pair.begin);
          if (pair.end) cudaEventDestroy(pair.end);
        }
      }
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
  std::vector<IterativeDeviceRank<T>> ranks_;
  std::vector<ncclComm_t> comms_;
  T learning_rate_;
  int compute_iters_ = 0;
  bool closure_ = false;
  std::uint64_t initialization_seed_ = 0;
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
  std::vector<double> batch_runtime_ms;
  std::vector<double> latent_update_ms;
  std::vector<double> latent_sync_ms;
  std::vector<double> forward_ms;
  std::vector<double> backward_ms;
  std::vector<double> gradient_sync_ms;
  double checksum = 0;
  bool validated = false;
  double max_abs_error = 0;
  double max_rel_error = 0;
  double latent_max_rel_error = 0;
  double gradient_max_rel_error = 0;
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
  std::vector<T> latents;
  std::vector<T> gradients;
};

template <typename T>
CompletedResult<T> benchmark_plan(const gs::ExecutionPlan& plan, const Options& options,
                                  const std::vector<int>& devices) {
  CompletedResult<T> result;
  result.report.strategy = plan.strategy;
  result.report.stats = plan.stats;
  result.report.allocated_bytes = options.mode == "iterative"
                                      ? gs::estimate_iterative_device_bytes(plan, sizeof(T))
                                      : gs::estimate_device_bytes(plan, sizeof(T));
  if (!plan_fits<T>(plan, devices, result.report.allocated_bytes, result.report.reason)) {
    result.report.status = "skipped";
    return result;
  }
  if (options.mode == "forward") {
    Executor<T> executor(plan, devices, options.compute_iters);
    std::uint64_t iteration = 0;
    for (int i = 0; i < options.warmup; ++i) {
      (void)executor.run(options.seed, iteration, options.evaluations_per_run, false);
      iteration += static_cast<std::uint64_t>(options.evaluations_per_run);
    }
    for (int i = 0; i < options.runs; ++i) {
      const bool last = i + 1 == options.runs;
      auto output = executor.run(options.seed, iteration, options.evaluations_per_run, last);
      iteration += static_cast<std::uint64_t>(options.evaluations_per_run);
      result.report.batch_runtime_ms.push_back(output.milliseconds);
      result.report.runtime_ms.push_back(output.milliseconds / options.evaluations_per_run);
      result.report.checksum = output.checksum;
      if (last) result.output = std::move(output.values);
    }
  } else {
    IterativeExecutor<T> executor(plan, devices, static_cast<T>(options.learning_rate),
                                  options.compute_iters, options.seed);
    for (int i = 0; i < options.warmup; ++i) {
      (void)executor.run(options.evaluations_per_run, false, false);
    }
    for (int i = 0; i < options.runs; ++i) {
      const bool last = i + 1 == options.runs;
      auto output = executor.run(options.evaluations_per_run, last);
      const double inverse = 1.0 / options.evaluations_per_run;
      result.report.batch_runtime_ms.push_back(output.times.total_ms);
      result.report.runtime_ms.push_back(output.times.total_ms * inverse);
      result.report.latent_update_ms.push_back(output.times.latent_update_ms * inverse);
      result.report.latent_sync_ms.push_back(output.times.latent_sync_ms * inverse);
      result.report.forward_ms.push_back(output.times.forward_ms * inverse);
      result.report.backward_ms.push_back(output.times.backward_ms * inverse);
      result.report.gradient_sync_ms.push_back(output.times.gradient_sync_ms * inverse);
      result.report.checksum = output.checksum;
      if (last) {
        result.output = std::move(output.values);
        result.latents = std::move(output.latents);
        result.gradients = std::move(output.gradients);
      }
    }
  }
  return result;
}

template <typename T>
double relative_error(const std::vector<T>& actual_values, const std::vector<T>& expected_values,
                      double& maximum_absolute) {
  if (actual_values.size() != expected_values.size()) {
    throw std::logic_error("validation vector size mismatch");
  }
  double maximum_relative = 0;
  for (std::size_t i = 0; i < expected_values.size(); ++i) {
    const double actual = static_cast<double>(actual_values[i]);
    const double expected = static_cast<double>(expected_values[i]);
    const double absolute = std::abs(actual - expected);
    const double relative = absolute / std::max(1.0, std::abs(expected));
    maximum_absolute = std::max(maximum_absolute, absolute);
    maximum_relative = std::max(maximum_relative, relative);
  }
  return maximum_relative;
}

template <typename T>
void compare_to_reference(CompletedResult<T>& result, const CompletedResult<T>& reference,
                          bool iterative) {
  if (result.report.status != "ok") return;
  result.report.max_rel_error = relative_error(result.output, reference.output,
                                                result.report.max_abs_error);
  if (iterative) {
    double ignored_absolute = 0;
    result.report.latent_max_rel_error =
        relative_error(result.latents, reference.latents, ignored_absolute);
    ignored_absolute = 0;
    result.report.gradient_max_rel_error =
        relative_error(result.gradients, reference.gradients, ignored_absolute);
  }
  const double tolerance = iterative ? (std::is_same_v<T, float> ? 5e-4 : 5e-11)
                                     : (std::is_same_v<T, float> ? 2e-5 : 1e-12);
  result.report.validated = result.report.max_rel_error <= tolerance &&
                            (!iterative || (result.report.latent_max_rel_error <= tolerance &&
                                            result.report.gradient_max_rel_error <= tolerance));
  if (!result.report.validated) {
    result.report.status = "validation_failed";
    result.report.reason = "distributed state differs from the single-GPU CUDA reference";
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
  out << "{\n  \"schema_version\":2,\n  \"configuration\": {\"nodes\":" << options.nodes
      << ",\"parents\":" << options.parents << ",\"levels\":" << options.levels
      << ",\"gpus\":" << devices.size() << ",\"communities\":" << graph.config.communities
      << ",\"locality\":" << options.locality << ",\"seed\":" << options.seed
      << ",\"dtype\":\"" << options.dtype << "\",\"mode\":\"" << options.mode
      << "\",\"warmup\":" << options.warmup << ",\"runs\":" << options.runs
      << ",\"evaluations_per_run\":" << options.evaluations_per_run
      << ",\"learning_rate\":" << options.learning_rate
      << ",\"compute_iters\":" << options.compute_iters << "},\n";
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
    const bool iterative = options.mode == "iterative";
    const auto latent_values = iterative ? result.stats.replica_values : std::size_t{0};
    const auto forward_values = result.stats.communicated_values;
    const auto gradient_values = iterative
                                     ? (result.strategy == "closure"
                                            ? result.stats.replica_values
                                            : result.stats.communicated_values)
                                     : std::size_t{0};
    const auto latent_messages = iterative ? result.stats.replica_messages : std::size_t{0};
    const auto forward_messages = result.stats.messages;
    const auto gradient_messages = iterative
                                       ? (result.strategy == "closure"
                                              ? result.stats.replica_messages
                                              : result.stats.messages)
                                       : std::size_t{0};
    const auto communicated_values = latent_values + forward_values + gradient_values;
    const auto messages = latent_messages + forward_messages + gradient_messages;
    out << ",\"load_imbalance\":" << result.stats.load_imbalance
        << ",\"primary_cut_edges\":" << result.stats.cut_edges
        << ",\"communicated_values\":" << communicated_values
        << ",\"communicated_bytes\":" << communicated_values * sizeof(T)
        << ",\"communicated_bytes_per_batch\":"
        << communicated_values * sizeof(T) *
               static_cast<std::size_t>(options.evaluations_per_run)
        << ",\"messages\":" << messages
        << ",\"messages_per_batch\":"
        << messages * static_cast<std::size_t>(options.evaluations_per_run)
        << ",\"communication_per_iteration\":{\"latent_sync_values\":" << latent_values
        << ",\"latent_sync_bytes\":" << latent_values * sizeof(T)
        << ",\"latent_sync_messages\":" << latent_messages
        << ",\"forward_values\":" << forward_values
        << ",\"forward_bytes\":" << forward_values * sizeof(T)
        << ",\"forward_messages\":" << forward_messages
        << ",\"gradient_sync_values\":" << gradient_values
        << ",\"gradient_sync_bytes\":" << gradient_values * sizeof(T)
        << ",\"gradient_sync_messages\":" << gradient_messages << "}"
        << ",\"replication_factor\":" << result.stats.replication_factor
        << ",\"allocated_bytes_per_rank\":";
    emit_array(out, result.allocated_bytes);
    out << ",\"peak_benchmark_bytes_per_rank\":";
    emit_array(out, result.allocated_bytes);
    if (result.status != "skipped") {
      out << ",\"checksum\":" << result.checksum << ",\"validated\":"
          << (result.validated ? "true" : "false")
          << ",\"max_abs_error\":" << result.max_abs_error
          << ",\"max_rel_error\":" << result.max_rel_error
          << ",\"latent_max_rel_error\":" << result.latent_max_rel_error
          << ",\"gradient_max_rel_error\":" << result.gradient_max_rel_error;
      if (iterative) out << ",\"objective\":" << result.checksum;
      out << ",\"runtime_ms\":";
      emit_array(out, result.runtime_ms);
      out << ",\"batch_runtime_ms\":";
      emit_array(out, result.batch_runtime_ms);
      const double median = percentile(result.runtime_ms, 0.5);
      out << ",\"runtime_min_ms\":" << *std::min_element(result.runtime_ms.begin(), result.runtime_ms.end())
          << ",\"runtime_median_ms\":" << median
          << ",\"runtime_p95_ms\":" << percentile(result.runtime_ms, 0.95)
          << ",\"nodes_per_second\":" << options.nodes * 1000.0 / median
          << ",\"evaluated_nodes_per_second\":" << evaluated_nodes * 1000.0 / median
          << ",\"primary_edges_per_second\":"
          << non_root_edges * 1000.0 / median;
      if (iterative) {
        out << ",\"phase_runtime_ms\":{";
        out << "\"latent_update\":";
        emit_array(out, result.latent_update_ms);
        out << ",\"latent_sync\":";
        emit_array(out, result.latent_sync_ms);
        out << ",\"forward\":";
        emit_array(out, result.forward_ms);
        out << ",\"backward\":";
        emit_array(out, result.backward_ms);
        out << ",\"gradient_sync\":";
        emit_array(out, result.gradient_sync_ms);
        out << ",\"total\":";
        emit_array(out, result.runtime_ms);
        out << "},\"phase_median_ms\":{\"latent_update\":"
            << percentile(result.latent_update_ms, 0.5)
            << ",\"latent_sync\":" << percentile(result.latent_sync_ms, 0.5)
            << ",\"forward\":" << percentile(result.forward_ms, 0.5)
            << ",\"backward\":" << percentile(result.backward_ms, 0.5)
            << ",\"gradient_sync\":" << percentile(result.gradient_sync_ms, 0.5)
            << ",\"total\":" << median
            << "}";
      }
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
  std::vector<gs::ExecutionPlan> plans;
  if (options.mode == "forward" &&
      (options.strategy == "all" || options.strategy == "random")) {
    auto random_partition = gs::make_random_balanced_partition(
        graph, devices.size(), options.seed ^ 0xA5A5A5A5ULL);
    plans.push_back(gs::make_exchange_plan(graph, random_partition));
  }
  if (options.strategy == "all" || options.strategy == "affinity" ||
      options.strategy == "closure") {
    auto affinity_partition = gs::make_parent_affinity_partition(
        graph, devices.size(), options.seed ^ 0x5A5A5A5AULL);
    if (options.strategy == "all" || options.strategy == "affinity") {
      plans.push_back(gs::make_exchange_plan(graph, affinity_partition));
    }
    if (options.strategy == "all" || options.strategy == "closure") {
      plans.push_back(gs::make_ancestor_closure_plan(graph, affinity_partition));
    }
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
  auto reference_bytes = options.mode == "iterative"
                             ? gs::estimate_iterative_device_bytes(reference_plan, sizeof(T))
                             : gs::estimate_device_bytes(reference_plan, sizeof(T));
  std::string reference_reason;
  if (plan_fits<T>(reference_plan, reference_device, reference_bytes, reference_reason)) {
    CompletedResult<T> reference_result;
    if (options.mode == "forward") {
      Executor<T> reference(reference_plan, reference_device, options.compute_iters);
      const auto final_iteration =
          static_cast<std::uint64_t>((options.warmup + options.runs) *
                                     options.evaluations_per_run - 1);
      auto output = reference.run(options.seed, final_iteration, 1, true);
      reference_result.output = std::move(output.values);
    } else {
      IterativeExecutor<T> reference(reference_plan, reference_device,
                                     static_cast<T>(options.learning_rate),
                                     options.compute_iters, options.seed);
      auto output = reference.run((options.warmup + options.runs) *
                                      options.evaluations_per_run,
                                  true, false);
      reference_result.output = std::move(output.values);
      reference_result.latents = std::move(output.latents);
      reference_result.gradients = std::move(output.gradients);
    }
    for (auto& result : completed) {
      compare_to_reference(result, reference_result, options.mode == "iterative");
    }
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
