#include "elementary_transformer/refinement.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include "elementary_transformer/interner.hpp"

namespace elementary_transformer {

const char* version() { return "0.1.0"; }

namespace {

constexpr int kMaxPebbles = 8;
constexpr int kMaxArity = 8;
constexpr std::uint64_t kMaxConfigurations = std::uint64_t{1} << 31;

std::uint64_t checked_power(std::uint64_t base, int exponent) {
  std::uint64_t result = 1;
  for (int i = 0; i < exponent; ++i) {
    if (base != 0 && result > std::numeric_limits<std::uint64_t>::max() / base) {
      throw std::overflow_error("integer overflow in power computation");
    }
    result *= base;
  }
  return result;
}

void validate(const Structure& s, const char* name) {
  if (s.size < 0) throw std::invalid_argument(std::string(name) + ": negative size");
  for (const auto& r : s.relations) {
    if (r.arity < 1 || r.arity > kMaxArity) {
      throw std::invalid_argument(std::string(name) + ": relation arity must lie in [1, 8]");
    }
    if (r.table.size() != checked_power(static_cast<std::uint64_t>(s.size), r.arity)) {
      throw std::invalid_argument(std::string(name) + ": relation table has the wrong size");
    }
  }
}

std::vector<std::uint64_t> powers(int size, int k) {
  std::vector<std::uint64_t> pow(k + 1, 1);
  for (int i = 1; i <= k; ++i) pow[i] = pow[i - 1] * static_cast<std::uint64_t>(size + 1);
  return pow;
}

// Round 0: the atomic type of each configuration, i.e. which pebbles are
// placed, which placed pebbles coincide and which relational atoms over the
// placed pebbles hold.
std::vector<std::uint32_t> atomic_types(const Structure& s, int k, RowInterner& interner) {
  const int n = s.size;
  const std::uint64_t count = powers(n, k)[k];
  std::size_t bits = static_cast<std::size_t>(k) * (k - 1) / 2;
  for (const auto& r : s.relations) bits += checked_power(k, r.arity);
  const std::size_t words = 1 + (bits + 31) / 32;

  std::vector<std::uint32_t> types(count);
  std::vector<int> digits(k, 0);
  std::vector<int> map(kMaxArity, 0);
  std::vector<std::uint32_t> row(words, 0);

  for (std::uint64_t t = 0; t < count; ++t) {
    std::fill(row.begin(), row.end(), 0);
    std::uint32_t mask = 0;
    for (int i = 0; i < k; ++i) {
      if (digits[i] != 0) mask |= std::uint32_t{1} << i;
    }
    row[0] = mask;
    std::size_t bit = 0;
    auto set_bit = [&](bool value) {
      if (value) row[1 + bit / 32] |= std::uint32_t{1} << (bit % 32);
      ++bit;
    };
    for (int i = 0; i < k; ++i) {
      for (int j = i + 1; j < k; ++j) set_bit(digits[i] != 0 && digits[i] == digits[j]);
    }
    for (const auto& r : s.relations) {
      std::fill(map.begin(), map.begin() + r.arity, 0);
      const std::uint64_t maps = checked_power(k, r.arity);
      for (std::uint64_t m = 0; m < maps; ++m) {
        bool placed = true;
        std::uint64_t index = 0;
        for (int j = 0; j < r.arity; ++j) {
          const int d = digits[map[j]];
          if (d == 0) {
            placed = false;
            break;
          }
          index = index * static_cast<std::uint64_t>(n) + static_cast<std::uint64_t>(d - 1);
        }
        set_bit(placed && r.table[index] != 0);
        for (int j = r.arity - 1; j >= 0; --j) {
          if (++map[j] < k) break;
          map[j] = 0;
        }
      }
    }
    types[t] = interner.intern(row.data(), row.size());
    for (int i = 0; i < k; ++i) {
      if (++digits[i] <= n) break;
      digits[i] = 0;
    }
  }
  return types;
}

// One refinement step. The new type of a configuration t is its old type
// together with, for every pebble i, the set of old types of the
// configurations obtained by moving pebble i to an element. The set for
// (t, i) does not depend on the current position of pebble i, so it is
// computed once per configuration with pebble i lifted.
std::vector<std::uint32_t> refine_once(const std::vector<std::uint32_t>& prev, int n, int k,
                                       RowInterner& sets, RowInterner& signatures) {
  const std::vector<std::uint64_t> pow = powers(n, k);
  const std::uint64_t count = pow[k];
  const std::uint64_t base = static_cast<std::uint64_t>(n) + 1;
  const std::uint64_t lifted = count / base;

  std::vector<std::vector<std::uint32_t>> set_ids(k, std::vector<std::uint32_t>(lifted));
  std::vector<std::uint32_t> buffer(n);
  for (int i = 0; i < k; ++i) {
    for (std::uint64_t c = 0; c < lifted; ++c) {
      const std::uint64_t t0 = c % pow[i] + (c / pow[i]) * pow[i + 1];
      for (int e = 0; e < n; ++e) buffer[e] = prev[t0 + static_cast<std::uint64_t>(e + 1) * pow[i]];
      std::sort(buffer.begin(), buffer.end());
      const auto end = std::unique(buffer.begin(), buffer.end());
      set_ids[i][c] = sets.intern(buffer.data(), static_cast<std::size_t>(end - buffer.begin()));
    }
  }

  std::vector<std::uint32_t> next(count);
  std::vector<std::uint32_t> row(k + 1);
  for (std::uint64_t t = 0; t < count; ++t) {
    row[0] = prev[t];
    for (int i = 0; i < k; ++i) {
      const std::uint64_t d = (t / pow[i]) % base;
      const std::uint64_t t0 = t - d * pow[i];
      row[i + 1] = set_ids[i][t0 % pow[i] + (t0 / pow[i + 1]) * pow[i]];
    }
    next[t] = signatures.intern(row.data(), row.size());
  }
  return next;
}

}  // namespace

std::uint64_t num_configurations(int size, int k) {
  if (size < 0 || k < 0) throw std::invalid_argument("size and k must be non-negative");
  return checked_power(static_cast<std::uint64_t>(size) + 1, k);
}

TypeTables refine_types(const Structure& left, const Structure& right, int k, int q_max) {
  if (k < 1 || k > kMaxPebbles) throw std::invalid_argument("k must lie in [1, 8]");
  if (q_max < 0) throw std::invalid_argument("q_max must be non-negative");
  validate(left, "left");
  validate(right, "right");
  if (left.relations.size() != right.relations.size()) {
    throw std::invalid_argument("structures have different signatures");
  }
  for (std::size_t i = 0; i < left.relations.size(); ++i) {
    if (left.relations[i].arity != right.relations[i].arity) {
      throw std::invalid_argument("structures have different signatures");
    }
  }
  const std::uint64_t count_left = num_configurations(left.size, k);
  const std::uint64_t count_right = num_configurations(right.size, k);
  if (count_left > kMaxConfigurations || count_right > kMaxConfigurations) {
    throw std::invalid_argument("(n + 1)^k exceeds 2^31 configurations");
  }

  TypeTables out;
  out.k = k;
  out.size_left = left.size;
  out.size_right = right.size;
  const std::size_t expected = static_cast<std::size_t>(std::min<std::uint64_t>(count_left + count_right, 1u << 20));
  {
    RowInterner atoms(1024);
    out.left.push_back(atomic_types(left, k, atoms));
    out.right.push_back(atomic_types(right, k, atoms));
    out.num_types.push_back(static_cast<std::uint32_t>(atoms.size()));
  }
  if (out.left[0][0] != out.right[0][0]) {
    out.q_star = 0;
    return out;
  }
  for (int r = 1; r <= q_max; ++r) {
    RowInterner sets(expected);
    RowInterner signatures(expected);
    out.left.push_back(refine_once(out.left.back(), left.size, k, sets, signatures));
    out.right.push_back(refine_once(out.right.back(), right.size, k, sets, signatures));
    out.num_types.push_back(static_cast<std::uint32_t>(signatures.size()));
    if (out.left[r][0] != out.right[r][0]) {
      out.q_star = r;
      return out;
    }
    if (out.num_types[r] == out.num_types[r - 1]) {
      out.stable = true;
      return out;
    }
  }
  return out;
}

}  // namespace elementary_transformer
