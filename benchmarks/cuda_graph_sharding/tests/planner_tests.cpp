#include "graph_sharding/planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace graph_sharding;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void test_graph_and_partitions() {
  GraphConfig config{257, 3, 7, 3, 0.5, 1234};
  Graph graph = generate_graph(config);
  validate_graph(graph);
  check(graph.level_offsets.back() == 257, "graph lost nodes");

  auto random = make_random_balanced_partition(graph, 4, 11);
  auto affinity = make_parent_affinity_partition(graph, 4, 11);
  validate_partition(graph, random);
  validate_partition(graph, affinity);
  auto exchange_random = make_exchange_plan(graph, random);
  auto exchange_affinity = make_exchange_plan(graph, affinity);
  auto closure = make_ancestor_closure_plan(graph, affinity);
  validate_execution_plan(graph, random, exchange_random);
  validate_execution_plan(graph, affinity, exchange_affinity);
  validate_execution_plan(graph, affinity, closure);
  check(closure.stats.replication_factor >= 1.0, "closure replication is invalid");
  check(closure.stats.communicated_values == 0, "closure should not communicate graph values");
  const auto replicated = std::accumulate(closure.stats.stored_nodes.begin(),
                                          closure.stats.stored_nodes.end(), std::size_t{0}) -
                          graph.config.nodes;
  check(closure.stats.replica_values == replicated,
        "closure replica synchronization count is wrong");
  for (std::size_t src = 0; src < closure.ranks; ++src) {
    for (std::size_t dst = 0; dst < closure.ranks; ++dst) {
      const auto& send = closure.rank_plans[src].replica_send_offsets;
      const auto& recv = closure.rank_plans[dst].replica_recv_offsets;
      check(send[dst + 1] - send[dst] == recv[src + 1] - recv[src],
            "closure replica peer counts are asymmetric");
      for (std::size_t i = 0; i < send[dst + 1] - send[dst]; ++i) {
        const auto owner_slot = closure.rank_plans[src].replica_send_slots[send[dst] + i];
        const auto replica_slot = closure.rank_plans[dst].replica_recv_slots[recv[src] + i];
        check(closure.rank_plans[src].global_nodes[owner_slot] ==
                  closure.rank_plans[dst].global_nodes[replica_slot],
              "closure replica peer ordering is inconsistent");
      }
    }
  }
  check(exchange_random.stats.stored_nodes == exchange_random.stats.owned_nodes,
        "exchange plan replicated nodes");
}

void test_locality_endpoints_and_determinism() {
  GraphConfig local_config{192, 2, 6, 3, 1.0, 99};
  Graph local = generate_graph(local_config);
  const auto non_root_edges = (local_config.nodes - local.level_offsets[1]) * local_config.parents;
  check(local.local_parent_edges == non_root_edges, "locality=1 produced a cross-community edge");
  Graph again = generate_graph(local_config);
  check(local.parents == again.parents && local.communities == again.communities,
        "graph generation is not deterministic");

  local_config.locality = 0.0;
  Graph uniform = generate_graph(local_config);
  validate_graph(uniform);
  check(uniform.parents != local.parents, "locality endpoints produced identical graphs");
}

void test_one_rank_and_memory_estimate() {
  Graph graph = generate_graph(GraphConfig{65, 1, 4, 1, 0.0, 7});
  auto partition = make_parent_affinity_partition(graph, 1, 3);
  auto exchange = make_exchange_plan(graph, partition);
  auto closure = make_ancestor_closure_plan(graph, partition);
  check(exchange.stats.cut_edges == 0 && exchange.stats.messages == 0,
        "one-rank plan contains communication");
  check(closure.stats.replication_factor == 1.0, "one-rank closure should not replicate");
  check(closure.stats.replica_values == 0 && closure.stats.replica_messages == 0,
        "one-rank closure should not synchronize replicas");
  auto bytes = estimate_device_bytes(exchange, sizeof(float));
  check(bytes.size() == 1 && bytes[0] > graph.config.nodes * sizeof(float),
        "memory estimate is too small");
  auto iterative_bytes = estimate_iterative_device_bytes(exchange, sizeof(float));
  check(iterative_bytes[0] > bytes[0], "iterative memory estimate is not larger");
}

void test_validation_errors() {
  bool failed = false;
  try {
    (void)generate_graph(GraphConfig{20, 4, 5, 2, 1.0, 1});
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  check(failed, "invalid root capacity was accepted");
}

double normalized_loss(const Graph& graph, const std::vector<double>& latents,
                       std::vector<double>* output = nullptr) {
  std::vector<double> values(graph.config.nodes, 0.0);
  for (std::size_t level = 0; level < graph.config.levels; ++level) {
    for (std::size_t node = graph.level_offsets[level]; node < graph.level_offsets[level + 1];
         ++node) {
      values[node] = latents[node];
      if (level == 0) continue;
      for (std::size_t j = 0; j < graph.config.parents; ++j) {
        values[node] += values[graph.parents[node * graph.config.parents + j]] /
                        static_cast<double>(graph.config.parents);
      }
    }
  }
  double loss = 0;
  for (double value : values) loss += 0.5 * value * value / graph.config.nodes;
  if (output) *output = std::move(values);
  return loss;
}

void test_normalized_dag_gradient() {
  Graph graph = generate_graph(GraphConfig{48, 2, 4, 1, 0.5, 81});
  std::vector<double> latents(graph.config.nodes);
  for (std::size_t i = 0; i < latents.size(); ++i) {
    latents[i] = 0.01 * static_cast<double>(static_cast<int>(i % 11) - 5);
  }
  std::vector<double> values;
  (void)normalized_loss(graph, latents, &values);
  std::vector<double> adjoints(values.size());
  std::vector<double> gradients(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) adjoints[i] = values[i] / values.size();
  for (std::size_t node = graph.config.nodes; node-- > 0;) {
    gradients[node] = adjoints[node];
    if (node < graph.level_offsets[1]) continue;
    for (std::size_t j = 0; j < graph.config.parents; ++j) {
      adjoints[graph.parents[node * graph.config.parents + j]] +=
          gradients[node] / graph.config.parents;
    }
  }
  constexpr double epsilon = 1e-6;
  for (std::size_t node : {std::size_t{0}, std::size_t{17}, std::size_t{47}}) {
    latents[node] += epsilon;
    const double plus = normalized_loss(graph, latents);
    latents[node] -= 2 * epsilon;
    const double minus = normalized_loss(graph, latents);
    latents[node] += epsilon;
    const double finite_difference = (plus - minus) / (2 * epsilon);
    check(std::abs(finite_difference - gradients[node]) < 1e-9,
          "normalized DAG gradient failed finite-difference validation");
  }
}

}  // namespace

int main() {
  try {
    test_graph_and_partitions();
    test_locality_endpoints_and_determinism();
    test_one_rank_and_memory_estimate();
    test_validation_errors();
    test_normalized_dag_gradient();
    std::cout << "planner tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "planner test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
