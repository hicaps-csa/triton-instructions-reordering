#ifndef TRITON_DIALECT_TRITON_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITON_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "/home/pushpendram/compiler/triton/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include <memory>

namespace mlir {
namespace triton {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createDowncastReorderOptimizer();

#define GEN_PASS_REGISTRATION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
