#include <cstring>
#include <iostream>

#include "elementary_transformer/refinement.hpp"

int main() {
  namespace et = elementary_transformer;
  if (std::strcmp(et::version(), "0.1.0") != 0) {
    std::cerr << "unexpected version\n";
    return 1;
  }
  std::cout << "ok\n";
  return 0;
}
