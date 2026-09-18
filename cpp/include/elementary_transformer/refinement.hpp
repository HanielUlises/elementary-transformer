#pragma once

#include <cstdint>
#include <vector>

namespace elementary_transformer {

const char* version();

// A relation of a finite structure with universe {0, ..., size - 1}, stored as
// a dense table of size^arity entries in row-major order (the first argument
// is the most significant digit).
struct Relation {
  int arity = 0;
  std::vector<std::uint8_t> table;
};

struct Structure {
  int size = 0;
  std::vector<Relation> relations;
};

// Configurations of the k-pebble game on a structure with n elements are the
// tuples in {bot, 0, ..., n - 1}^k, where bot marks an unplaced pebble. A
// configuration is encoded by the integer sum_i d_i (n + 1)^i with digit
// d_i = 0 for bot and d_i = a + 1 for an element a. The empty configuration
// has index 0.
std::uint64_t num_configurations(int size, int k);

// Rank-r FO^k types of all configurations of two structures, named jointly:
// left[r][s] == right[r][t] iff Duplicator wins the r-round k-pebble game
// from the pair of configurations (s, t).
struct TypeTables {
  int k = 0;
  int size_left = 0;
  int size_right = 0;
  std::vector<std::vector<std::uint32_t>> left;
  std::vector<std::vector<std::uint32_t>> right;
  std::vector<std::uint32_t> num_types;
  // Least r <= q_max whose tables separate the empty configurations, or -1.
  int q_star = -1;
  // True when the joint partition stopped refining; with q_star == -1 this
  // means the structures are FO^k-equivalent for every quantifier rank.
  bool stable = false;
};

TypeTables refine_types(const Structure& left, const Structure& right, int k, int q_max);

}  // namespace elementary_transformer
