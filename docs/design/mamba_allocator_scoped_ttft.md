# Mamba/KV/Kernel-Scoped TTFT Optimization Note

## Goal
Improve TTFT for hybrid Mamba workloads while avoiding generic scheduler policy changes.

## Scope
- No scheduler fairness/admission policy changes.
- No changes to `Scheduler._mamba_block_aligned_split` behavior.
- Optimizations are constrained to KV allocation internals, Mamba cache manager behavior, and Mamba selective scan CUDA path.

## Allocator approach (scheduler-agnostic)
- `KVCacheManager.allocate_slots()` now performs an internal lightweight feasibility estimate first.
- If estimated required blocks exceed total pool capacity, allocation fast-fails immediately.
- Debug logging records:
  - `allocation_fail_reason`
  - `estimated_blocks_needed`
  - `free_blocks_at_attempt`

## Mamba tail handling approach
In Mamba align mode:
- Aligned chunks keep persistent cache semantics.
- Sub-block tails are allowed to progress with deferred persistence when safe for already-allocated requests.
- This avoids forcing extra block churn for tiny tails while preserving aligned persistence behavior.

Tail behavior categories:
1. **Cached aligned chunk**: normal aligned allocation + persistence.
2. **Uncached tail chunk**: defer additional block allocation for sub-block tails on existing requests.
3. **Final persistence point**: persistence resumes on aligned boundary or final-safe kernel writeback path.

## Kernel approach
Mamba selective scan path keeps:
- static dispatch (`template <bool IsAligned, int ChunkSize, ...>`)
- predicate masking for tail lanes
- local parameter bindings

Writeback policy is guarded:
- final-only writeback only when chunk metadata guarantees correctness
- otherwise fallback to intermediate writeback semantics

## Comparison
### Previous scheduler-based approach
- Added global scheduler knobs (progress floors/retry thresholds/preflight) with wider blast radius.
- Improved some Mamba scenarios but risked non-Mamba policy side effects.

### New allocator/kernel-scoped approach
- No scheduler policy modifications.
- Moves fast-fail and tail-efficiency work into Mamba/KV/kernel components.
- Reduces impact on non-Mamba hot paths.

## Risk table
| Risk Type | Assessment | Mitigation |
|---|---|---|
| Correctness risk | Medium | Guarded final-only writeback; fallback to intermediate semantics when metadata guarantees are absent. |
| Compatibility risk | Low-Medium | Scheduler logic unchanged; Mamba-specific behavior limited to Mamba manager + selective scan kernel path. |
| Maintainer acceptance risk | Lower than scheduler-global patch | Scoped changes, explicit behavior categories, and no generic fairness-policy modifications. |
