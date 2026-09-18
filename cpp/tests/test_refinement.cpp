#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "elementary_transformer/refinement.hpp"

namespace et = elementary_transformer;

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

et::Structure binary(int n, const std::function<bool(int, int)>& holds) {
  et::Relation r;
  r.arity = 2;
  r.table.assign(static_cast<std::size_t>(n) * n, 0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) r.table[static_cast<std::size_t>(i) * n + j] = holds(i, j) ? 1 : 0;
  }
  et::Structure s;
  s.size = n;
  s.relations.push_back(std::move(r));
  return s;
}

et::Structure linear_order(int n) {
  return binary(n, [](int i, int j) { return i < j; });
}

// Disjoint union of cycles with the given lengths.
et::Structure cycles(const std::vector<int>& lengths) {
  std::vector<int> component;
  std::vector<int> offset;
  int n = 0;
  for (int length : lengths) {
    for (int i = 0; i < length; ++i) {
      component.push_back(static_cast<int>(offset.size()));
    }
    offset.push_back(n);
    n += length;
  }
  return binary(n, [&](int i, int j) {
    if (component[i] != component[j]) return false;
    const int c = component[i];
    const int length = lengths[c];
    const int a = i - offset[c];
    const int b = j - offset[c];
    return (a + 1) % length == b || (b + 1) % length == a;
  });
}

et::Structure permuted(const et::Structure& s, const std::vector<int>& perm) {
  const int n = s.size;
  return binary(n, [&](int i, int j) {
    // perm maps old elements to new ones; find preimages.
    const int a = static_cast<int>(std::find(perm.begin(), perm.end(), i) - perm.begin());
    const int b = static_cast<int>(std::find(perm.begin(), perm.end(), j) - perm.begin());
    return s.relations[0].table[static_cast<std::size_t>(a) * n + b] != 0;
  });
}

int expected_order_rank(int m, int n) {
  // L_m and L_n agree on all sentences of rank q iff m == n or both are at
  // least 2^q - 1, so the least separating rank is the least q with
  // min(m, n) < 2^q - 1.
  if (m == n) return -1;
  const int low = std::min(m, n);
  int q = 0;
  while (low >= (1 << q) - 1) ++q;
  return q;
}

void test_linear_orders() {
  const int k = 4;
  const int q_max = 4;
  for (int m = 1; m <= 17; ++m) {
    for (int n = m; n <= 17; ++n) {
      const et::TypeTables t = et::refine_types(linear_order(m), linear_order(n), k, q_max);
      int expected = expected_order_rank(m, n);
      if (expected > q_max) expected = -1;
      check(t.q_star == expected, "linear orders L_" + std::to_string(m) + ", L_" + std::to_string(n) +
                                      ": got " + std::to_string(t.q_star) + ", expected " +
                                      std::to_string(expected));
    }
  }
}

void test_cycles() {
  const et::Structure c6 = cycles({6});
  const et::Structure two_triangles = cycles({3, 3});
  const et::TypeTables three = et::refine_types(c6, two_triangles, 3, 5);
  check(three.q_star == 3, "C6 and 2C3 are separated at rank 3 with three variables");
  const et::TypeTables two = et::refine_types(c6, two_triangles, 2, 10);
  check(two.q_star == -1 && two.stable, "C6 and 2C3 are FO^2-equivalent");
}

void test_isomorphic_copies() {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 20; ++trial) {
    const int n = 3 + trial % 6;
    std::bernoulli_distribution coin(0.4);
    std::vector<std::vector<bool>> adjacency(n, std::vector<bool>(n, false));
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) adjacency[i][j] = adjacency[j][i] = coin(rng);
    }
    const et::Structure g = binary(n, [&](int i, int j) { return adjacency[i][j]; });
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    const et::TypeTables t = et::refine_types(g, permuted(g, perm), 3, 6);
    check(t.q_star == -1, "isomorphic copies are equivalent (trial " + std::to_string(trial) + ")");
  }
}

void test_sizes() {
  check(et::num_configurations(4, 3) == 125, "number of configurations");
  const et::TypeTables t = et::refine_types(linear_order(0), linear_order(1), 1, 3);
  check(t.q_star == 1, "the empty structure is separated from a singleton at rank 1");
}

}  // namespace

int main() {
  test_linear_orders();
  test_cycles();
  test_isomorphic_copies();
  test_sizes();
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "all C++ checks passed\n";
  return EXIT_SUCCESS;
}
