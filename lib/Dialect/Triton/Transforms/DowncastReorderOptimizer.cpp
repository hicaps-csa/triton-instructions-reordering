// DowncastReorderOptimizer: TTIR pass that hoists element size-reducing
// casts (arith.truncf, arith.trunci, arith.fptosi, arith.fptoui, narrowing
// tt.fp_to_fp, narrowing arith.index_cast) upstream of safe data-movement
// ops (tt.cat, tt.trans, tt.broadcast, tt.reshape, tt.expand_dims,
// tt.join) so the mover runs on the smaller element type. The sections
// below cover op classification, the per-operand cast clone, the
// per-producer multi-consumer rewrite, the pass driver (eligibility,
// worklist, outer fixed-point), and registration.

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "downcast-reorder-optimizer"

using namespace mlir;

namespace mlir::triton {

//===----------------------------------------------------------------------===//
// 1. Operation Classification
//===----------------------------------------------------------------------===//

/// Classify an op as a safe data-movement op.
///
/// Safe data-movement ops preserve element-wise mapping, so a
/// size-reducing cast on the result is equivalent to the same cast
/// applied to each input before the mover. Every entry has
/// ranked-tensor operands and a ranked-tensor result; the rewrite logic
/// depends on that.
///
/// Args:
///   op: Operation to classify.
///
/// Returns:
///   True if `op` is in the safe-mover set; false otherwise.
static bool isSafeDataMovementOp(Operation *op) {
  return isa<triton::CatOp,        // Concatenation
             triton::TransOp,      // Transpose (permutation)
             triton::BroadcastOp,  // Broadcast (size-1 dims expand)
             triton::ReshapeOp,    // Reshape (same element count)
             triton::ExpandDimsOp, // Insert size-1 dim
             triton::JoinOp>(op);  // Join two tensors along new minor dim
}

/// Classify an op as a size-reducing cast.
///
/// For `triton::FpToFpOp` and `arith::IndexCastOp` the answer is
/// type-dependent: only narrowing instances qualify (widening fp_to_fp
/// or non-comparable `index` widths return false).
///
/// Args:
///   op: Operation to classify.
///
/// Returns:
///   True if `op` narrows its element bitwidth; false otherwise.
static bool isSizeReducingOp(Operation *op) {
  // Float / integer truncation (e.g. fp32 → fp16, i64 → i32).
  if (isa<arith::TruncFOp, arith::TruncIOp>(op))
    return true;

  // Float-to-integer conversions.
  if (isa<arith::FPToSIOp, arith::FPToUIOp>(op))
    return true;

  // Triton's custom FP cast (handles F8 ↔ FP16/BF16/FP32/FP64). Only treat
  // as size-reducing when the destination element bitwidth is strictly
  // smaller than the source's — widening fp_to_fp gains nothing from
  // pushing the cast into the mover.
  if (auto fp = dyn_cast<triton::FpToFpOp>(op)) {
    Type srcElem = fp.getSrc().getType();
    Type dstElem = fp.getResult().getType();
    if (auto t = dyn_cast<RankedTensorType>(srcElem))
      srcElem = t.getElementType();
    if (auto t = dyn_cast<RankedTensorType>(dstElem))
      dstElem = t.getElementType();
    auto srcF = dyn_cast<FloatType>(srcElem);
    auto dstF = dyn_cast<FloatType>(dstElem);
    if (!srcF || !dstF)
      return false;
    return srcF.getWidth() > dstF.getWidth();
  }

  // Index cast: only treat as size-reducing when both ends are concrete
  // IntegerType and the destination is strictly narrower. `index` width is
  // target-dependent and not safely comparable, so skip it.
  if (auto indexCast = dyn_cast<arith::IndexCastOp>(op)) {
    Type srcElem = indexCast.getOperand().getType();
    Type dstElem = indexCast.getType();
    if (auto srcTensor = dyn_cast<RankedTensorType>(srcElem)) {
      auto dstTensor = dyn_cast<RankedTensorType>(dstElem);
      if (!dstTensor)
        return false;
      srcElem = srcTensor.getElementType();
      dstElem = dstTensor.getElementType();
    }
    auto srcInt = dyn_cast<IntegerType>(srcElem);
    auto dstInt = dyn_cast<IntegerType>(dstElem);
    if (!srcInt || !dstInt)
      return false;
    return srcInt.getWidth() > dstInt.getWidth();
  }

  return false;
}

//===----------------------------------------------------------------------===//
// 2. Clone Size-Reducing Operations
//===----------------------------------------------------------------------===//

/// Clone a size-reducing cast onto a new operand with a new result type.
///
/// Args:
///   builder: Builder positioned at the desired insertion point.
///   originalOp: The cast whose kind (and rounding attribute, where
///     applicable) is mirrored.
///   operand: The new input value the cloned cast will consume.
///   newType: The result type the cloned cast must produce.
///
/// Returns:
///   The cloned cast's result, or nullptr if `originalOp` is not in the
///   supported set of size-reducing ops.
static Value createSizeReducingClone(OpBuilder &builder, Operation *originalOp,
                                     Value operand, Type newType) {
  Location loc = originalOp->getLoc();

  return llvm::TypeSwitch<Operation *, Value>(originalOp)
      // Float truncation (fp32 → fp16, fp64 → fp32).
      .Case<arith::TruncFOp>([&](auto op) {
        return builder.create<arith::TruncFOp>(loc, newType, operand)
            .getResult();
      })

      // Integer truncation (i64 → i32, i32 → i16, ...).
      .Case<arith::TruncIOp>([&](auto op) {
        return builder.create<arith::TruncIOp>(loc, newType, operand)
            .getResult();
      })

      // Float to signed integer.
      .Case<arith::FPToSIOp>([&](auto op) {
        return builder.create<arith::FPToSIOp>(loc, newType, operand)
            .getResult();
      })

      // Float to unsigned integer.
      .Case<arith::FPToUIOp>([&](auto op) {
        return builder.create<arith::FPToUIOp>(loc, newType, operand)
            .getResult();
      })

      // Index cast (i64 → i32).
      .Case<arith::IndexCastOp>([&](auto op) {
        return builder.create<arith::IndexCastOp>(loc, newType, operand)
            .getResult();
      })

      // Triton custom FP cast — must preserve rounding attribute, since the
      // verifier mandates one on any narrowing cast.
      .Case<triton::FpToFpOp>([&](auto op) {
        return builder
            .create<triton::FpToFpOp>(loc, newType, operand,
                                      op.getRoundingAttr())
            .getResult();
      })

      .Default([&](Operation *) -> Value {
        LLVM_DEBUG(llvm::dbgs() << "Unsupported size-reducing op for clone: "
                                << originalOp->getName() << "\n");
        return nullptr;
      });
}

//===----------------------------------------------------------------------===//
// 3. Generic Transformation Logic (Multi-Consumer Support)
//===----------------------------------------------------------------------===//

/// Build a rewritten data-movement chain for one size-reducing consumer.
///
/// Clones the consumer's cast onto each producer operand, then builds a
/// new data-movement op that produces the consumer's result type
/// directly. The original `dataMovementOp` is left in place; the caller
/// decides when to erase it.
///
/// Args:
///   builder: Builder used to insert the cloned casts and the new mover.
///   dataMovementOp: Original mover whose operands are being narrowed.
///   sizeReduceOp: Consumer whose result type defines the target element
///     type for the rewrite.
///
/// Returns:
///   The new mover's result, or nullptr if any operand could not be
///   reduced, the original result type is not a ranked tensor, or
///   verification of the new op failed.
static Value createTransformedDataMovement(OpBuilder &builder,
                                           Operation *dataMovementOp,
                                           Operation *sizeReduceOp) {
  auto finalResultType =
      dyn_cast<RankedTensorType>(sizeReduceOp->getResult(0).getType());
  if (!finalResultType)
    return nullptr;
  Type targetElementType = finalResultType.getElementType();

  // The cloned reduce ops + new data-movement op all sit just before the
  // original data-movement op. The original operands already dominate it,
  // so the cloned reduce ops will too — no per-operand insertion dance.
  builder.setInsertionPoint(dataMovementOp);

  SmallVector<Value> newOperands;
  for (Value operand : dataMovementOp->getOperands()) {
    auto inputType = dyn_cast<RankedTensorType>(operand.getType());
    if (!inputType)
      return nullptr;
    auto newType = RankedTensorType::get(
        inputType.getShape(), targetElementType, inputType.getEncoding());

    Value newReduced =
        createSizeReducingClone(builder, sizeReduceOp, operand, newType);
    if (!newReduced)
      return nullptr;
    newOperands.push_back(newReduced);
  }

  auto origResultType =
      dyn_cast<RankedTensorType>(dataMovementOp->getResult(0).getType());
  if (!origResultType)
    return nullptr;
  auto correctedResultType =
      RankedTensorType::get(origResultType.getShape(), targetElementType,
                            origResultType.getEncoding());

  OperationState state(dataMovementOp->getLoc(), dataMovementOp->getName());
  state.addOperands(newOperands);
  state.addTypes(correctedResultType);
  state.addAttributes(dataMovementOp->getAttrs());
  Operation *newDataMovementOp = builder.create(state);

  // `builder.create` never returns null — if the IR is malformed we want
  // verifier failure now rather than later in the pipeline.
  if (failed(verify(newDataMovementOp))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Rewritten data-movement op failed verification: "
               << *newDataMovementOp << "\n");
    newDataMovementOp->erase();
    return nullptr;
  }

  return newDataMovementOp->getResult(0);
}

/// Rewrite a data-movement op against its size-reducing consumers.
///
/// Consumers are deduplicated by (op kind, result type) so two reducers
/// of the same kind narrowing to the same element type share one
/// pushed-down chain; different kinds (or different result types) keep
/// their own. The original data-movement op is erased once it has no
/// remaining uses.
///
/// Args:
///   dataMovementOp: The mover to rewrite.
///   sizeReduceOps: Eligible reducer consumers of `dataMovementOp`.
///
/// Returns:
///   True if at least one consumer was redirected to a rewritten chain.
static bool
applyMultiConsumerOptimization(Operation *dataMovementOp,
                               SmallVectorImpl<Operation *> &sizeReduceOps) {
  LLVM_DEBUG(llvm::dbgs() << "  [Transforming] " << dataMovementOp->getName()
                          << " with " << sizeReduceOps.size()
                          << " size-reducing consumer(s)\n");

  OpBuilder builder(dataMovementOp);

  // Dedup by (op kind, result type): two consumers with the same kind AND
  // type can share one rewritten data-movement op. Different kinds (e.g.,
  // truncf vs fptosi) must not share even if their result types coincide.
  // This naturally supports multi-reducer-kind fan-out: a producer feeding
  // multiple distinct reducers gets one rewritten chain per (kind, type)
  // pair, with reducers of the same (kind, type) sharing.
  DenseMap<std::pair<OperationName, Type>, Value> cache;
  bool changed = false;

  for (Operation *sizeReduceOp : sizeReduceOps) {
    auto key = std::make_pair(sizeReduceOp->getName(),
                              sizeReduceOp->getResult(0).getType());

    Value newResult;
    auto it = cache.find(key);
    if (it != cache.end()) {
      newResult = it->second;
    } else {
      newResult =
          createTransformedDataMovement(builder, dataMovementOp, sizeReduceOp);
      if (!newResult) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skipping consumer due to transformation failure\n");
        continue;
      }
      cache.insert({key, newResult});
    }

    LLVM_DEBUG(llvm::dbgs() << "    - Replaced " << sizeReduceOp->getName()
                            << " consumer with pushed-down size-reduction.\n");

    sizeReduceOp->getResult(0).replaceAllUsesWith(newResult);
    sizeReduceOp->erase();
    changed = true;
  }

  // In the common path every user of the producer is a same-block reducer
  // (guaranteed by eligibleReducers) and each one was just redirected
  // above, so the producer is now dead. The use_empty guard tolerates two
  // edge cases that leave uses behind: (a) a cross-block reducer that
  // eligibleReducers chose to skip rather than rewrite across regions,
  // and (b) a per-consumer rewrite that aborted in
  // createTransformedDataMovement (nullptr return / verifier failure) and
  // left its reducer still pointing at the original mover.
  if (dataMovementOp->use_empty())
    dataMovementOp->erase();

  return changed;
}

//===----------------------------------------------------------------------===//
// 4. Pass Definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_DOWNCASTREORDEROPTIMIZER
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

struct DowncastReorderOptimizer
    : public impl::DowncastReorderOptimizerBase<DowncastReorderOptimizer> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, triton::TritonDialect>();
  }

  /// Gather the size-reducing consumers of `producer` that this pass
  /// will rewrite.
  ///
  /// Returns an empty vector if the producer fails any of the eligibility
  /// checks: not a safe mover, an operand defined outside the producer's
  /// block, non-tensor / unranked result, transpose of rank < 2, or — by
  /// design — any user that is not size-reducing (see the diamond bail-out
  /// inside the user loop). Cross-block reducer users are dropped from
  /// the returned list but do not bail the producer; same-block reducer
  /// users whose operands don't all strictly narrow in bitwidth are also
  /// dropped.
  ///
  /// Args:
  ///   producer: Candidate data-movement op.
  ///
  /// Returns:
  ///   The same-block, strictly-narrowing reducer users of `producer`;
  ///   empty if `producer` is not eligible.
  SmallVector<Operation *> eligibleReducers(Operation *producer) {
    SmallVector<Operation *> reducers;

    if (!isSafeDataMovementOp(producer))
      return reducers;

    // Producer's operands must already be defined above it (block-local).
    for (Value operand : producer->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        if (defOp->getBlock() != producer->getBlock() ||
            !defOp->isBeforeInBlock(producer))
          return {};
      }
    }

    // Transpose only makes sense for rank >= 2.
    if (auto transOp = dyn_cast<triton::TransOp>(producer)) {
      auto inputType =
          dyn_cast<RankedTensorType>(transOp.getOperand().getType());
      if (!inputType || inputType.getRank() < 2)
        return {};
    }

    if (!isa<RankedTensorType>(producer->getResult(0).getType()))
      return {};

    // Diamond / mixed-fanout bail-out: if ANY user is non-reducer, drop
    // the producer entirely. A surviving non-reducer user would keep the
    // original mover alive (the use_empty guard in
    // applyMultiConsumerOptimization), so the cloned reducer chain would
    // run in addition to — not instead of — the original, and the
    // duplicated-work cost outweighs the savings from the cast hoist.
    //
    // Chain propagation (Cat -> Trans -> Trunc) still works under this
    // rule: an upstream mover's only initial user is the downstream
    // mover (also non-reducer, so it wouldn't be seeded by itself), but
    // once the downstream mover is rewritten the upstream mover's new
    // user is the sunk/cloned reducer — at which point upstream
    // propagation in runOnce re-queues it.
    for (Operation *user : producer->getResult(0).getUsers()) {
      if (!isSizeReducingOp(user))
        return {};

      // isBeforeInBlock asserts on cross-block ops. Same-block enforcement
      // also keeps the rewrite scope-local: a cross-block reducer is left
      // attached to the original producer (handled by the use_empty guard
      // in applyMultiConsumerOptimization). Reverse-dominance can't be
      // assumed across blocks, so attempting to clone reducers above the
      // mover would be unsafe.
      if (user->getBlock() != producer->getBlock() ||
          !producer->isBeforeInBlock(user))
        continue;

      auto dstType = dyn_cast<RankedTensorType>(user->getResult(0).getType());
      if (!dstType)
        continue;

      // Every producer operand must strictly reduce in bitwidth, otherwise
      // pushing the cast across the data-movement op buys us nothing. All
      // safe movers in isSafeDataMovementOp() have ranked-tensor operands.
      bool allOperandsReduce = true;
      for (Value operand : producer->getOperands()) {
        auto rt = dyn_cast<RankedTensorType>(operand.getType());
        if (!rt) {
          allOperandsReduce = false;
          break;
        }
        Type elemTy = rt.getElementType();
        if (!elemTy.isIntOrFloat() || elemTy.getIntOrFloatBitWidth() <=
                                          dstType.getElementTypeBitWidth()) {
          allOperandsReduce = false;
          break;
        }
      }
      if (!allOperandsReduce)
        continue;

      reducers.push_back(user);
    }

    return reducers;
  }

  /// Perform one sweep of the rewrite.
  ///
  /// A worklist is seeded by a module walk, then extended by upstream
  /// propagation after every rewrite: when a mover is rewritten the
  /// cloned reducers become new users of its operands' defining ops, and
  /// any of those defining ops that are themselves safe movers are pushed
  /// back on the worklist. This collapses chained patterns like
  /// `Cat -> Trans -> Trunc` in a single sweep — rewriting Trans turns
  /// Cat's user into the cloned trunc, so Cat becomes eligible without
  /// waiting for the outer loop to re-walk.
  ///
  /// Args:
  ///   module: The module to rewrite.
  ///
  /// Returns:
  ///   True if any consumer was redirected, signalling the outer driver
  ///   to walk again.
  bool runOnce(ModuleOp module) {
    // SetVector keeps walk order while giving O(1) contains() and
    // de-duplication on insert.
    llvm::SmallSetVector<Operation *, 16> worklist;

    module.walk([&](Operation *op) {
      if (!eligibleReducers(op).empty())
        worklist.insert(op);
    });

    bool changed = false;
    while (!worklist.empty()) {
      Operation *producer = worklist.pop_back_val();

      // Re-check eligibility at pop time: an earlier rewrite in this
      // worklist may have invalidated this producer (e.g., its consumer
      // list changed) or simply made it newly-eligible.
      auto reducers = eligibleReducers(producer);
      if (reducers.empty())
        continue;

      // Capture upstream movers before the rewrite runs. Once
      // applyMultiConsumerOptimization returns, each cloned reducer is a
      // new user of one of these upstream ops; if any of those upstream
      // ops is itself a safe mover, it may now be eligible.
      llvm::SmallSetVector<Operation *, 4> upstream;
      for (Value operand : producer->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp()) {
          if (isSafeDataMovementOp(defOp))
            upstream.insert(defOp);
        }
      }

      if (applyMultiConsumerOptimization(producer, reducers)) {
        changed = true;
        // Upstream movers are never erased by
        // applyMultiConsumerOptimization (it only erases the producer
        // being rewritten and the size-reducing consumers it just
        // consumed), so these pointers remain valid here.
        for (Operation *up : upstream)
          worklist.insert(up);
      }
    }
    return changed;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // runOnce is already worklist-driven with upstream propagation, so in
    // practice a single sweep handles every chain and diamond the pass can
    // touch. The outer loop is a defensive fixed-point: it re-runs the
    // module walk to catch any producer that became newly eligible without
    // being re-queued by the worklist. Termination is guaranteed because
    // each successful sweep strictly reduces the number of (mover,
    // reducer) pairs in the module; the cap is a paranoia ceiling for
    // malformed input.
    constexpr unsigned kMaxIterations = 16;
    unsigned i = 0;
    for (; i < kMaxIterations; ++i) {
      if (!runOnce(module))
        break;
    }
    LLVM_DEBUG({
      if (i == kMaxIterations)
        llvm::dbgs() << "DowncastReorderOptimizer: hit iteration cap at "
                     << kMaxIterations
                     << " sweeps; some rewrites may not have converged.\n";
    });

    // Pass-manager runs the verifier between passes, but a local check here
    // surfaces any malformed rewrite as a pass failure rather than a later
    // crash.
    if (failed(verify(module)))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// 5. Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createDowncastReorderOptimizer() {
  return std::make_unique<DowncastReorderOptimizer>();
}

} // namespace mlir::triton