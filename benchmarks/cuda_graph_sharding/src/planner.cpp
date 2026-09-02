#include "graph_sharding/planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace graph_sharding {
namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}

std::vector<NodeId> level_nodes(const Graph& graph, std::size_t level) {
  std::vector<NodeId> result(graph.level_offsets[level + 1] - graph.level_offsets[level]);
  std::iota(result.begin(), result.end(), static_cast<NodeId>(graph.level_offsets[level]));
  return result;
}

std::vector<std::size_t> level_quotas(std::size_t count, std::size_t ranks,
                                     std::size_t level) {
  std::vector<std::size_t> quotas(ranks, count / ranks);
  for (std::size_t i = 0; i < count % ranks; ++i) quotas[(level + i) % ranks]++;
  return quotas;
}

PlanStats base_stats(const Graph& graph, const Partition& partition) {
  PlanStats stats;
  stats.owned_nodes = partition.owned_counts;
  std::size_t minimum = *std::min_element(stats.owned_nodes.begin(), stats.owned_nodes.end());
  std::size_t maximum = *std::max_element(stats.owned_nodes.begin(), stats.owned_nodes.end());
  stats.load_imbalance = minimum == 0 ? std::numeric_limits<double>::infinity()
                                     : static_cast<double>(maximum) / minimum;
  const auto k = graph.config.parents;
  for (std::size_t node = graph.level_offsets[1]; node < graph.config.nodes; ++node) {
    for (std::size_t j = 0; j < k; ++j) {
      if (partition.owners[graph.parents[node * k + j]] != partition.owners[node]) {
        stats.cut_edges++;
      }
    }
  }
  return stats;
}

}  // namespace

std::size_t Graph::level_of(NodeId node) const {
  auto it = std::upper_bound(level_offsets.begin(), level_offsets.end(), node);
  return static_cast<std::size_t>(std::distance(level_offsets.begin(), it) - 1);
}

Graph generate_graph(const GraphConfig& config) {
  require(config.nodes > 0, "nodes must be positive");
  require(config.levels >= 2, "levels must be at least two");
  require(config.levels <= config.nodes, "levels cannot exceed nodes");
  require(config.parents > 0, "parents must be positive");
  require(config.communities > 0, "communities must be positive");
  require(config.locality >= 0.0 && config.locality <= 1.0, "locality must be in [0, 1]");
  require(config.nodes <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
          "nodes exceed the signed 32-bit local-slot range");
  require(config.parents <= config.nodes / config.communities,
          "parents * communities exceeds the node count");

  Graph graph;
  graph.config = config;
  graph.level_offsets.resize(config.levels + 1);
  for (std::size_t level = 0; level <= config.levels; ++level) {
    graph.level_offsets[level] = config.nodes * level / config.levels;
  }
  const auto roots = graph.level_offsets[1];
  require(roots >= config.parents * config.communities,
          "level zero must contain at least parents * communities nodes");

  graph.communities.resize(config.nodes);
  graph.parents.assign(config.nodes * config.parents, kInvalidNode);
  std::vector<std::vector<NodeId>> prior_by_community(config.communities);
  std::vector<NodeId> prior_all;
  prior_all.reserve(config.nodes);
  std::mt19937_64 rng(config.seed);
  std::bernoulli_distribution choose_local(config.locality);

  for (std::size_t level = 0; level < config.levels; ++level) {
    const auto begin = graph.level_offsets[level];
    const auto end = graph.level_offsets[level + 1];
    std::vector<std::uint32_t> labels(end - begin);
    for (std::size_t i = 0; i < labels.size(); ++i) labels[i] = i % config.communities;
    std::shuffle(labels.begin(), labels.end(), rng);
    for (std::size_t i = begin; i < end; ++i) graph.communities[i] = labels[i - begin];

    if (level > 0) {
      for (std::size_t node = begin; node < end; ++node) {
        const auto community = graph.communities[node];
        std::unordered_set<NodeId> selected;
        selected.reserve(config.parents * 2);
        while (selected.size() < config.parents) {
          const bool local = choose_local(rng);
          const auto& pool = local ? prior_by_community[community] : prior_all;
          require(pool.size() >= config.parents,
                  "not enough distinct preceding parents in the selected pool");
          std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
          selected.insert(pool[pick(rng)]);
        }
        std::vector<NodeId> row(selected.begin(), selected.end());
        std::sort(row.begin(), row.end());
        for (std::size_t j = 0; j < config.parents; ++j) {
          graph.parents[node * config.parents + j] = row[j];
          if (graph.communities[row[j]] == community) graph.local_parent_edges++;
        }
      }
    }

    for (std::size_t node = begin; node < end; ++node) {
      prior_all.push_back(static_cast<NodeId>(node));
      prior_by_community[graph.communities[node]].push_back(static_cast<NodeId>(node));
    }
  }
  validate_graph(graph);
  return graph;
}

void validate_graph(const Graph& graph) {
  const auto& c = graph.config;
  require(graph.level_offsets.size() == c.levels + 1, "invalid level offset count");
  require(graph.level_offsets.front() == 0 && graph.level_offsets.back() == c.nodes,
          "level offsets must span all nodes");
  require(graph.communities.size() == c.nodes, "invalid community array size");
  require(graph.parents.size() == c.nodes * c.parents, "invalid parent array size");
  for (std::size_t level = 0; level < c.levels; ++level) {
    require(graph.level_offsets[level] < graph.level_offsets[level + 1],
            "every level must be non-empty");
    for (std::size_t node = graph.level_offsets[level]; node < graph.level_offsets[level + 1]; ++node) {
      require(graph.communities[node] < c.communities, "community is out of range");
      std::unordered_set<NodeId> unique;
      for (std::size_t j = 0; j < c.parents; ++j) {
        NodeId parent = graph.parents[node * c.parents + j];
        if (level == 0) {
          require(parent == kInvalidNode, "root nodes must not have parents");
        } else {
          require(parent < graph.level_offsets[level], "parent must belong to an earlier level");
          unique.insert(parent);
        }
      }
      if (level > 0) require(unique.size() == c.parents, "parents must be distinct");
    }
  }
}

Partition make_random_balanced_partition(const Graph& graph, std::size_t ranks,
                                         std::uint64_t seed) {
  require(ranks > 0, "ranks must be positive");
  require(ranks <= graph.config.nodes, "ranks cannot exceed nodes");
  Partition result{"random", ranks, std::vector<Rank>(graph.config.nodes, -1),
                   std::vector<std::size_t>(ranks, 0),
                   std::vector<std::size_t>(graph.config.levels * ranks, 0)};
  std::mt19937_64 rng(seed);
  for (std::size_t level = 0; level < graph.config.levels; ++level) {
    auto nodes = level_nodes(graph, level);
    std::shuffle(nodes.begin(), nodes.end(), rng);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      const auto rank = static_cast<Rank>((level + i) % ranks);
      result.owners[nodes[i]] = rank;
      result.owned_counts[rank]++;
      result.level_rank_counts[level * ranks + rank]++;
    }
  }
  validate_partition(graph, result);
  return result;
}

Partition make_parent_affinity_partition(const Graph& graph, std::size_t ranks,
                                         std::uint64_t seed) {
  require(ranks > 0, "ranks must be positive");
  require(ranks <= graph.config.nodes, "ranks cannot exceed nodes");
  Partition result{"affinity", ranks, std::vector<Rank>(graph.config.nodes, -1),
                   std::vector<std::size_t>(ranks, 0),
                   std::vector<std::size_t>(graph.config.levels * ranks, 0)};
  std::mt19937_64 rng(seed);
  for (std::size_t level = 0; level < graph.config.levels; ++level) {
    auto nodes = level_nodes(graph, level);
    std::shuffle(nodes.begin(), nodes.end(), rng);
    auto quotas = level_quotas(nodes.size(), ranks, level);
    std::vector<std::size_t> used(ranks, 0);
    for (NodeId node : nodes) {
      Rank best = -1;
      std::size_t best_affinity = 0;
      for (std::size_t rank = 0; rank < ranks; ++rank) {
        if (used[rank] == quotas[rank]) continue;
        std::size_t affinity = 0;
        if (level > 0) {
          for (std::size_t j = 0; j < graph.config.parents; ++j) {
            affinity += result.owners[graph.parents[node * graph.config.parents + j]] ==
                        static_cast<Rank>(rank);
          }
        }
        if (best < 0 || affinity > best_affinity ||
            (affinity == best_affinity && result.owned_counts[rank] < result.owned_counts[best]) ||
            (affinity == best_affinity && result.owned_counts[rank] == result.owned_counts[best] &&
             rank < static_cast<std::size_t>(best))) {
          best = static_cast<Rank>(rank);
          best_affinity = affinity;
        }
      }
      require(best >= 0, "no rank has remaining level quota");
      result.owners[node] = best;
      result.owned_counts[best]++;
      result.level_rank_counts[level * ranks + best]++;
      used[best]++;
    }
  }
  validate_partition(graph, result);
  return result;
}

void validate_partition(const Graph& graph, const Partition& partition) {
  require(partition.ranks > 0, "partition has no ranks");
  require(partition.owners.size() == graph.config.nodes, "owner array has wrong size");
  require(partition.owned_counts.size() == partition.ranks, "owned count has wrong size");
  require(partition.level_rank_counts.size() == graph.config.levels * partition.ranks,
          "level/rank count has wrong size");
  std::vector<std::size_t> actual(partition.ranks, 0);
  for (Rank owner : partition.owners) {
    require(owner >= 0 && static_cast<std::size_t>(owner) < partition.ranks,
            "owner is out of range");
    actual[owner]++;
  }
  require(actual == partition.owned_counts, "owned counts do not match owners");
  for (std::size_t level = 0; level < graph.config.levels; ++level) {
    std::size_t minimum = std::numeric_limits<std::size_t>::max();
    std::size_t maximum = 0;
    for (std::size_t rank = 0; rank < partition.ranks; ++rank) {
      auto value = partition.level_rank_counts[level * partition.ranks + rank];
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
    }
    require(maximum - minimum <= 1, "partition is not balanced within a level");
  }
}

ExecutionPlan make_exchange_plan(const Graph& graph, const Partition& partition) {
  validate_graph(graph);
  validate_partition(graph, partition);
  ExecutionPlan plan{partition.name, graph.config.nodes, graph.config.parents,
                     graph.config.levels, partition.ranks,
                     std::vector<RankPlan>(partition.ranks), base_stats(graph, partition)};
  const auto n = graph.config.nodes;
  const auto k = graph.config.parents;
  std::vector<std::vector<std::int32_t>> local_maps(partition.ranks,
                                                    std::vector<std::int32_t>(n, -1));
  for (std::size_t rank = 0; rank < partition.ranks; ++rank) {
    auto& rp = plan.rank_plans[rank];
    for (std::size_t node = 0; node < n; ++node) {
      if (partition.owners[node] == static_cast<Rank>(rank)) {
        local_maps[rank][node] = static_cast<std::int32_t>(rp.global_nodes.size());
        rp.global_nodes.push_back(static_cast<NodeId>(node));
        rp.primary.push_back(1);
        rp.primary_slots.push_back(static_cast<std::uint32_t>(rp.global_nodes.size() - 1));
      }
    }
    rp.replica_send_offsets.assign(partition.ranks + 1, 0);
    rp.replica_recv_offsets.assign(partition.ranks + 1, 0);
    rp.levels.resize(graph.config.levels);
  }

  for (std::size_t level = 0; level < graph.config.levels; ++level) {
    using Sets = std::vector<std::vector<std::unordered_set<NodeId>>>;
    Sets needed(partition.ranks,
                std::vector<std::unordered_set<NodeId>>(partition.ranks));
    if (level > 0) {
      for (std::size_t node = graph.level_offsets[level]; node < graph.level_offsets[level + 1]; ++node) {
        const auto dst = static_cast<std::size_t>(partition.owners[node]);
        for (std::size_t j = 0; j < k; ++j) {
          NodeId parent = graph.parents[node * k + j];
          const auto src = static_cast<std::size_t>(partition.owners[parent]);
          if (src != dst) needed[dst][src].insert(parent);
        }
      }
    }
    std::vector<std::vector<std::vector<NodeId>>> ordered(
        partition.ranks, std::vector<std::vector<NodeId>>(partition.ranks));
    for (std::size_t dst = 0; dst < partition.ranks; ++dst) {
      for (std::size_t src = 0; src < partition.ranks; ++src) {
        ordered[dst][src].assign(needed[dst][src].begin(), needed[dst][src].end());
        std::sort(ordered[dst][src].begin(), ordered[dst][src].end());
        if (!ordered[dst][src].empty()) {
          plan.stats.communicated_values += ordered[dst][src].size();
          plan.stats.messages++;
        }
      }
    }

    for (std::size_t rank = 0; rank < partition.ranks; ++rank) {
      auto& rp = plan.rank_plans[rank];
      auto& lp = rp.levels[level];
      lp.node_begin = rp.level_nodes.size();
      lp.send_slot_begin = rp.send_local_slots.size();
      lp.send_offsets.assign(partition.ranks + 1, 0);
      lp.recv_offsets.assign(partition.ranks + 1, 0);
      for (std::size_t peer = 0; peer < partition.ranks; ++peer) {
        lp.send_offsets[peer] = rp.send_local_slots.size() - lp.send_slot_begin;
        for (NodeId parent : ordered[peer][rank]) {
          require(local_maps[rank][parent] >= 0, "send value is not locally owned");
          rp.send_local_slots.push_back(static_cast<std::uint32_t>(local_maps[rank][parent]));
        }
        lp.recv_offsets[peer + 1] = lp.recv_offsets[peer] + ordered[rank][peer].size();
      }
      lp.send_offsets[partition.ranks] = rp.send_local_slots.size() - lp.send_slot_begin;
      lp.send_slot_end = rp.send_local_slots.size();

      std::unordered_map<NodeId, std::size_t> remote_slots;
      for (std::size_t src = 0; src < partition.ranks; ++src) {
        for (std::size_t i = 0; i < ordered[rank][src].size(); ++i) {
          remote_slots[ordered[rank][src][i]] = lp.recv_offsets[src] + i;
        }
      }
      for (std::size_t node = graph.level_offsets[level]; node < graph.level_offsets[level + 1]; ++node) {
        if (partition.owners[node] != static_cast<Rank>(rank)) continue;
        rp.level_nodes.push_back(static_cast<std::uint32_t>(local_maps[rank][node]));
        for (std::size_t j = 0; j < k; ++j) {
          if (level == 0) {
            rp.parent_refs.push_back(0);
            continue;
          }
          NodeId parent = graph.parents[node * k + j];
          if (partition.owners[parent] == static_cast<Rank>(rank)) {
            rp.parent_refs.push_back(local_maps[rank][parent]);
          } else {
            auto found = remote_slots.find(parent);
            require(found != remote_slots.end(), "remote parent has no receive slot");
            rp.parent_refs.push_back(-1 - static_cast<std::int32_t>(found->second));
          }
        }
      }
      lp.node_end = rp.level_nodes.size();
      rp.max_send_values = std::max(rp.max_send_values, lp.send_offsets.back());
      rp.max_recv_values = std::max(rp.max_recv_values, lp.recv_offsets.back());
    }
  }
  plan.stats.stored_nodes = plan.stats.owned_nodes;
  validate_execution_plan(graph, partition, plan);
  return plan;
}

ExecutionPlan make_ancestor_closure_plan(const Graph& graph, const Partition& partition) {
  validate_graph(graph);
  validate_partition(graph, partition);
  ExecutionPlan plan{"closure", graph.config.nodes, graph.config.parents, graph.config.levels,
                     partition.ranks, std::vector<RankPlan>(partition.ranks),
                     base_stats(graph, partition)};
  plan.stats.communicated_values = 0;
  plan.stats.messages = 0;
  std::size_t total_stored = 0;
  std::vector<std::int32_t> primary_slot_by_node(graph.config.nodes, -1);
  for (std::size_t rank = 0; rank < partition.ranks; ++rank) {
    std::vector<std::uint8_t> included(graph.config.nodes, 0);
    for (std::size_t node = 0; node < graph.config.nodes; ++node) {
      included[node] = partition.owners[node] == static_cast<Rank>(rank);
    }
    for (std::size_t node = graph.config.nodes; node-- > graph.level_offsets[1];) {
      if (!included[node]) continue;
      for (std::size_t j = 0; j < graph.config.parents; ++j) {
        included[graph.parents[node * graph.config.parents + j]] = 1;
      }
    }
    auto& rp = plan.rank_plans[rank];
    std::vector<std::int32_t> local_map(graph.config.nodes, -1);
    for (std::size_t node = 0; node < graph.config.nodes; ++node) {
      if (!included[node]) continue;
      local_map[node] = static_cast<std::int32_t>(rp.global_nodes.size());
      rp.global_nodes.push_back(static_cast<NodeId>(node));
      const bool primary = partition.owners[node] == static_cast<Rank>(rank);
      rp.primary.push_back(primary);
      if (primary) {
        const auto slot = static_cast<std::uint32_t>(rp.global_nodes.size() - 1);
        rp.primary_slots.push_back(slot);
        primary_slot_by_node[node] = static_cast<std::int32_t>(slot);
      }
    }
    rp.levels.resize(graph.config.levels);
    for (std::size_t level = 0; level < graph.config.levels; ++level) {
      auto& lp = rp.levels[level];
      lp.node_begin = rp.level_nodes.size();
      lp.node_end = lp.node_begin;
      lp.send_slot_begin = lp.send_slot_end = 0;
      lp.send_offsets.assign(partition.ranks + 1, 0);
      lp.recv_offsets.assign(partition.ranks + 1, 0);
      for (std::size_t node = graph.level_offsets[level]; node < graph.level_offsets[level + 1]; ++node) {
        if (!included[node]) continue;
        rp.level_nodes.push_back(static_cast<std::uint32_t>(local_map[node]));
        for (std::size_t j = 0; j < graph.config.parents; ++j) {
          if (level == 0) {
            rp.parent_refs.push_back(0);
          } else {
            auto slot = local_map[graph.parents[node * graph.config.parents + j]];
            require(slot >= 0, "ancestor closure is incomplete");
            rp.parent_refs.push_back(slot);
          }
        }
      }
      lp.node_end = rp.level_nodes.size();
    }
    total_stored += rp.global_nodes.size();
    plan.stats.stored_nodes.push_back(rp.global_nodes.size());
  }

  // Build one graph-wide sparse primary-to-replica exchange. Lists are sorted
  // by global node ID so peer segments match exactly and deterministically.
  std::vector<std::vector<std::vector<NodeId>>> replicas(
      partition.ranks, std::vector<std::vector<NodeId>>(partition.ranks));
  for (std::size_t dst = 0; dst < partition.ranks; ++dst) {
    const auto& rp = plan.rank_plans[dst];
    for (std::size_t slot = 0; slot < rp.global_nodes.size(); ++slot) {
      if (rp.primary[slot]) continue;
      const NodeId node = rp.global_nodes[slot];
      const auto src = static_cast<std::size_t>(partition.owners[node]);
      replicas[dst][src].push_back(node);
    }
    for (auto& peer_nodes : replicas[dst]) std::sort(peer_nodes.begin(), peer_nodes.end());
  }
  for (std::size_t rank = 0; rank < partition.ranks; ++rank) {
    auto& rp = plan.rank_plans[rank];
    rp.replica_send_offsets.assign(partition.ranks + 1, 0);
    rp.replica_recv_offsets.assign(partition.ranks + 1, 0);
    for (std::size_t peer = 0; peer < partition.ranks; ++peer) {
      rp.replica_send_offsets[peer] = rp.replica_send_slots.size();
      for (NodeId node : replicas[peer][rank]) {
        const auto slot = primary_slot_by_node[node];
        require(slot >= 0 && rp.primary[slot], "replica source is not primary-owned");
        rp.replica_send_slots.push_back(static_cast<std::uint32_t>(slot));
      }
      rp.replica_send_offsets[peer + 1] = rp.replica_send_slots.size();

      rp.replica_recv_offsets[peer] = rp.replica_recv_slots.size();
      for (NodeId node : replicas[rank][peer]) {
        const auto found = std::lower_bound(rp.global_nodes.begin(), rp.global_nodes.end(), node);
        require(found != rp.global_nodes.end() && *found == node,
                "replica destination is not stored");
        const auto slot = static_cast<std::size_t>(found - rp.global_nodes.begin());
        require(!rp.primary[slot], "replica destination is not replicated");
        rp.replica_recv_slots.push_back(static_cast<std::uint32_t>(slot));
      }
      rp.replica_recv_offsets[peer + 1] = rp.replica_recv_slots.size();
    }
  }
  for (std::size_t dst = 0; dst < partition.ranks; ++dst) {
    for (std::size_t src = 0; src < partition.ranks; ++src) {
      if (replicas[dst][src].empty()) continue;
      plan.stats.replica_values += replicas[dst][src].size();
      plan.stats.replica_messages++;
    }
  }
  plan.stats.replication_factor = static_cast<double>(total_stored) / graph.config.nodes;
  validate_execution_plan(graph, partition, plan);
  return plan;
}

void validate_execution_plan(const Graph& graph, const Partition& partition,
                             const ExecutionPlan& plan) {
  require(plan.ranks == partition.ranks, "plan rank count differs from partition");
  require(plan.rank_plans.size() == plan.ranks, "invalid rank plan count");
  std::vector<std::size_t> primary_counts(graph.config.nodes, 0);
  for (std::size_t rank = 0; rank < plan.ranks; ++rank) {
    const auto& rp = plan.rank_plans[rank];
    require(rp.global_nodes.size() == rp.primary.size(), "primary mask has wrong size");
    require(rp.level_nodes.size() * graph.config.parents == rp.parent_refs.size(),
            "parent references have wrong size");
    require(rp.levels.size() == graph.config.levels, "rank level count is wrong");
    require(rp.replica_send_offsets.size() == plan.ranks + 1 &&
                rp.replica_recv_offsets.size() == plan.ranks + 1,
            "invalid replica peer offsets");
    require(std::is_sorted(rp.replica_send_offsets.begin(), rp.replica_send_offsets.end()) &&
                std::is_sorted(rp.replica_recv_offsets.begin(), rp.replica_recv_offsets.end()),
            "replica peer offsets must be sorted");
    require(rp.replica_send_offsets.back() == rp.replica_send_slots.size() &&
                rp.replica_recv_offsets.back() == rp.replica_recv_slots.size(),
            "replica offsets do not span their slot lists");
    std::unordered_set<NodeId> stored(rp.global_nodes.begin(), rp.global_nodes.end());
    for (std::size_t slot = 0; slot < rp.global_nodes.size(); ++slot) {
      if (rp.primary[slot]) {
        require(partition.owners[rp.global_nodes[slot]] == static_cast<Rank>(rank),
                "primary node is on the wrong rank");
        primary_counts[rp.global_nodes[slot]]++;
      }
    }
    for (std::size_t level = 0; level < graph.config.levels; ++level) {
      const auto& lp = rp.levels[level];
      require(lp.node_begin <= lp.node_end && lp.node_end <= rp.level_nodes.size(),
              "invalid node interval");
      require(lp.send_offsets.size() == plan.ranks + 1 &&
                  lp.recv_offsets.size() == plan.ranks + 1,
              "invalid peer offsets");
      require(std::is_sorted(lp.send_offsets.begin(), lp.send_offsets.end()) &&
                  std::is_sorted(lp.recv_offsets.begin(), lp.recv_offsets.end()),
              "peer offsets must be sorted");
      for (std::size_t row = lp.node_begin; row < lp.node_end; ++row) {
        auto slot = rp.level_nodes[row];
        require(slot < rp.global_nodes.size(), "level node slot is out of range");
        require(graph.level_of(rp.global_nodes[slot]) == level, "node is assigned to wrong level");
      }
    }
    for (auto slot : rp.primary_slots) {
      require(slot < rp.primary.size() && rp.primary[slot], "invalid primary output slot");
    }
    for (auto slot : rp.replica_send_slots) {
      require(slot < rp.primary.size() && rp.primary[slot],
              "replica synchronization source is not primary");
    }
    for (auto slot : rp.replica_recv_slots) {
      require(slot < rp.primary.size() && !rp.primary[slot],
              "replica synchronization destination is not replicated");
    }
    (void)stored;
  }
  require(std::all_of(primary_counts.begin(), primary_counts.end(), [](std::size_t count) { return count == 1; }),
          "every graph node must have exactly one primary owner");

  for (std::size_t level = 0; level < plan.levels; ++level) {
    for (std::size_t src = 0; src < plan.ranks; ++src) {
      for (std::size_t dst = 0; dst < plan.ranks; ++dst) {
        const auto& send = plan.rank_plans[src].levels[level].send_offsets;
        const auto& recv = plan.rank_plans[dst].levels[level].recv_offsets;
        require(send[dst + 1] - send[dst] == recv[src + 1] - recv[src],
                "send and receive counts do not match");
      }
    }
  }
  for (std::size_t src = 0; src < plan.ranks; ++src) {
    for (std::size_t dst = 0; dst < plan.ranks; ++dst) {
      const auto& send = plan.rank_plans[src].replica_send_offsets;
      const auto& recv = plan.rank_plans[dst].replica_recv_offsets;
      require(send[dst + 1] - send[dst] == recv[src + 1] - recv[src],
              "replica send and receive counts do not match");
    }
  }
}

std::vector<std::size_t> estimate_device_bytes(const ExecutionPlan& plan,
                                               std::size_t value_bytes) {
  std::vector<std::size_t> result(plan.ranks, 0);
  for (std::size_t rank = 0; rank < plan.ranks; ++rank) {
    const auto& rp = plan.rank_plans[rank];
    result[rank] = rp.global_nodes.size() * (value_bytes + sizeof(NodeId)) +
                   rp.level_nodes.size() * sizeof(std::uint32_t) +
                   rp.parent_refs.size() * sizeof(std::int32_t) +
                   rp.send_local_slots.size() * sizeof(std::uint32_t) +
                   rp.primary_slots.size() * sizeof(std::uint32_t) +
                   (rp.max_send_values + rp.max_recv_values + 1) * value_bytes;
  }
  return result;
}

std::vector<std::size_t> estimate_iterative_device_bytes(const ExecutionPlan& plan,
                                                         std::size_t value_bytes) {
  std::vector<std::size_t> result(plan.ranks, 0);
  for (std::size_t rank = 0; rank < plan.ranks; ++rank) {
    const auto& rp = plan.rank_plans[rank];
    const auto send_values = std::max(rp.max_send_values, rp.replica_send_slots.size());
    const auto recv_values = std::max(rp.max_recv_values, rp.replica_recv_slots.size());
    result[rank] = rp.global_nodes.size() * (3 * value_bytes + sizeof(NodeId)) +
                   rp.level_nodes.size() * sizeof(std::uint32_t) +
                   rp.parent_refs.size() * sizeof(std::int32_t) +
                   (rp.send_local_slots.size() + rp.primary_slots.size() +
                    rp.replica_send_slots.size() + rp.replica_recv_slots.size()) *
                       sizeof(std::uint32_t) +
                   (send_values + recv_values + 1) * value_bytes;
  }
  return result;
}

}  // namespace graph_sharding
