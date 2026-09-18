#include <pybind11/pybind11.h>

#include "elementary_transformer/refinement.hpp"

namespace py = pybind11;
namespace et = elementary_transformer;

PYBIND11_MODULE(_core, m) {
  m.doc() = "C++ core of elementary_transformer";
  m.def("version", &et::version);
}
