#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "elementary_transformer/refinement.hpp"

namespace py = pybind11;
namespace et = elementary_transformer;

namespace {

using RelationInput = std::pair<int, py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>>;

et::Structure to_structure(int size, const std::vector<RelationInput>& relations) {
  et::Structure s;
  s.size = size;
  for (const auto& [arity, table] : relations) {
    et::Relation r;
    r.arity = arity;
    r.table.assign(table.data(), table.data() + table.size());
    s.relations.push_back(std::move(r));
  }
  return s;
}

py::list to_arrays(const std::vector<std::vector<std::uint32_t>>& tables) {
  py::list out;
  for (const auto& table : tables) {
    py::array_t<std::uint32_t> array(static_cast<py::ssize_t>(table.size()));
    std::copy(table.begin(), table.end(), array.mutable_data());
    out.append(std::move(array));
  }
  return out;
}

py::dict refine(int size_left, const std::vector<RelationInput>& relations_left, int size_right,
                const std::vector<RelationInput>& relations_right, int k, int q_max) {
  const et::Structure left = to_structure(size_left, relations_left);
  const et::Structure right = to_structure(size_right, relations_right);
  et::TypeTables tables;
  {
    py::gil_scoped_release release;
    tables = et::refine_types(left, right, k, q_max);
  }
  py::dict out;
  out["k"] = tables.k;
  out["q_star"] = tables.q_star < 0 ? py::object(py::none()) : py::object(py::int_(tables.q_star));
  out["stable"] = tables.stable;
  out["num_types"] = tables.num_types;
  out["left"] = to_arrays(tables.left);
  out["right"] = to_arrays(tables.right);
  return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "C++ core of elementary_transformer: FO^k type refinement over pebble configurations";
  m.def("version", &et::version);
  m.def("num_configurations", &et::num_configurations, py::arg("size"), py::arg("k"));
  m.def("refine", &refine, py::arg("size_left"), py::arg("relations_left"), py::arg("size_right"),
        py::arg("relations_right"), py::arg("k"), py::arg("q_max"),
        "Jointly named rank-r FO^k types of all pebble configurations of two structures.");
}
