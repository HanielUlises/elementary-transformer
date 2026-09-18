#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace elementary_transformer {

// Assigns consecutive identifiers to variable-length rows of 32-bit words.
// Two rows receive the same identifier iff they are equal, so the naming is
// exact (no hash collisions can merge distinct rows). Identifiers are given in
// order of first appearance, which makes the naming deterministic.
class RowInterner {
 public:
  explicit RowInterner(std::size_t expected_rows = 1024) {
    std::size_t capacity = 16;
    while (capacity < 2 * expected_rows) capacity <<= 1;
    slots_.assign(capacity, 0);
    offsets_.push_back(0);
  }

  std::uint32_t intern(const std::uint32_t* row, std::size_t length) {
    const std::uint64_t h = hash(row, length);
    std::size_t mask = slots_.size() - 1;
    std::size_t pos = static_cast<std::size_t>(h) & mask;
    while (slots_[pos] != 0) {
      const std::uint32_t id = slots_[pos] - 1;
      if (hashes_[id] == h && equal(id, row, length)) return id;
      pos = (pos + 1) & mask;
    }
    const auto id = static_cast<std::uint32_t>(hashes_.size());
    pool_.insert(pool_.end(), row, row + length);
    offsets_.push_back(pool_.size());
    hashes_.push_back(h);
    slots_[pos] = id + 1;
    if (2 * hashes_.size() > slots_.size()) grow();
    return id;
  }

  std::size_t size() const { return hashes_.size(); }

 private:
  std::vector<std::uint32_t> pool_;
  std::vector<std::size_t> offsets_;
  std::vector<std::uint64_t> hashes_;
  std::vector<std::uint32_t> slots_;

  static std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
  }

  static std::uint64_t hash(const std::uint32_t* row, std::size_t length) {
    std::uint64_t h = mix(length + 0x9e3779b97f4a7c15ULL);
    for (std::size_t i = 0; i < length; ++i) h = mix(h ^ (row[i] + 0x9e3779b97f4a7c15ULL + (h << 6)));
    return h;
  }

  bool equal(std::uint32_t id, const std::uint32_t* row, std::size_t length) const {
    const std::size_t begin = offsets_[id];
    const std::size_t end = offsets_[id + 1];
    if (end - begin != length) return false;
    return length == 0 || std::memcmp(pool_.data() + begin, row, length * sizeof(std::uint32_t)) == 0;
  }

  void grow() {
    std::vector<std::uint32_t> slots(slots_.size() * 2, 0);
    const std::size_t mask = slots.size() - 1;
    for (std::size_t id = 0; id < hashes_.size(); ++id) {
      std::size_t pos = static_cast<std::size_t>(hashes_[id]) & mask;
      while (slots[pos] != 0) pos = (pos + 1) & mask;
      slots[pos] = static_cast<std::uint32_t>(id + 1);
    }
    slots_.swap(slots);
  }
};

}  // namespace elementary_transformer
