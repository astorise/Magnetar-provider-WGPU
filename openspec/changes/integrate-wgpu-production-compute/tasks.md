## 1. Scope and compatibility

- [ ] 1.1 Reconcile G01 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

- [ ] 1.4 Concrete contract migration: reconcile `wgpu-provider` / "WGPU Provider Real `add` Kernel" with "WGPU Provider Real `add` Kernel" and test both old baseline and new qualified profile.

## 2. G01 WGPU production execution path

- [ ] 2.1 Implement G01: WGPU SHALL expose qualified MatMul, RMSNorm, RoPE, attention, allocation and transfer operations through ProviderExecutionApi and Kernel Registry with real Device residency.
- [ ] 2.2 Add the positive conformance case: when a supported real model executes on a hardware WGPU adapter, verify results and transfer traces qualify the actual graph path rather than a standalone add demo.
- [ ] 2.3 Add the failure/isolation case: when only a software adapter or unsupported operation is available, verify the result is labeled accordingly and cannot qualify real-hardware production support.

## 3. Qualification, rollout and documentation

- [ ] 3.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 3.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 3.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 3.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 3.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 3.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 3.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
