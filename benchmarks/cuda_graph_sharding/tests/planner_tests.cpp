#include "graph_sharding/planner.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
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
  auto bytes = estimate_device_bytes(exchange, sizeof(float));
  check(bytes.size() == 1 && bytes[0] > graph.config.nodes * sizeof(float),
        "memory estimate is too small");
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

}  // namespace

int main() {
  try {
    test_graph_and_partitions();
    test_locality_endpoints_and_determinism();
    test_one_rank_and_memory_estimate();
    test_validation_errors();
    std::cout << "planner tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "planner test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
