#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace graph_sharding {

using NodeId = std::uint32_t;
using Rank = std::int32_t;
constexpr NodeId kInvalidNode = ~NodeId{0};

struct GraphConfig {
  std::size_t nodes = 1U << 20U;
  std::size_t parents = 8;
  std::size_t levels = 16;
  std::size_t communities = 1;
  double locality = 0.0;
  std::uint64_t seed = 42;
};

struct Graph {
  GraphConfig config;
  std::vector<std::size_t> level_offsets;
  std::vector<std::uint32_t> communities;
  // Fixed-width rows. Root rows contain kInvalidNode.
  std::vector<NodeId> parents;
  std::size_t local_parent_edges = 0;

  std::size_t level_of(NodeId node) const;
};

Graph generate_graph(const GraphConfig& config);
void validate_graph(const Graph& graph);

struct Partition {
  std::string name;
  std::size_t ranks = 0;
  std::vector<Rank> owners;
  std::vector<std::size_t> owned_counts;
  std::vector<std::size_t> level_rank_counts;
};

Partition make_random_balanced_partition(const Graph& graph, std::size_t ranks,
                                         std::uint64_t seed);
Partition make_parent_affinity_partition(const Graph& graph, std::size_t ranks,
                                         std::uint64_t seed);
void validate_partition(const Graph& graph, const Partition& partition);

struct LevelPlan {
  std::size_t node_begin = 0;
  std::size_t node_end = 0;
  std::size_t send_slot_begin = 0;
  std::size_t send_slot_end = 0;
  std::vector<std::size_t> send_offsets;
  std::vector<std::size_t> recv_offsets;
};

struct RankPlan {
  std::vector<NodeId> global_nodes;
  std::vector<std::uint8_t> primary;
  std::vector<std::uint32_t> level_nodes;
  // Local slots are non-negative. Remote slots are encoded as -1-slot.
  std::vector<std::int32_t> parent_refs;
  std::vector<std::uint32_t> send_local_slots;
  std::vector<std::uint32_t> primary_slots;
  std::vector<LevelPlan> levels;
  std::size_t max_send_values = 0;
  std::size_t max_recv_values = 0;
};

struct PlanStats {
  std::vector<std::size_t> owned_nodes;
  std::vector<std::size_t> stored_nodes;
  std::size_t cut_edges = 0;
  std::size_t communicated_values = 0;
  std::size_t messages = 0;
  double load_imbalance = 1.0;
  double replication_factor = 1.0;
};

struct ExecutionPlan {
  std::string strategy;
  std::size_t nodes = 0;
  std::size_t parents = 0;
  std::size_t levels = 0;
  std::size_t ranks = 0;
  std::vector<RankPlan> rank_plans;
  PlanStats stats;
};

ExecutionPlan make_exchange_plan(const Graph& graph, const Partition& partition);
ExecutionPlan make_ancestor_closure_plan(const Graph& graph, const Partition& primary_partition);
void validate_execution_plan(const Graph& graph, const Partition& primary_partition,
                             const ExecutionPlan& plan);

std::vector<std::size_t> estimate_device_bytes(const ExecutionPlan& plan,
                                               std::size_t value_bytes);

}  // namespace graph_sharding
