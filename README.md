# magnetar-provider-wgpu

## Purpose

A cross-platform GPU compute Provider for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime, built
on [`wgpu`](https://wgpu.rs/) -- a mature, widely-used cross-platform
compute/graphics API that targets Vulkan, Metal, and DX12 through one
Rust API surface. This is Magnetar's real path to Apple Metal support:
rather than hand-written, unverifiable Metal FFI (see
[`providers/metal`](https://github.com/astorise/Magnetar-provider-Metal)'s
own honest, unconditionally-unavailable skeleton and its README for why
that direct approach was rejected), this crate's real device discovery
and compute Kernels are verified here, on real Vulkan-backed hardware,
and rest on `wgpu`'s own cross-backend consistency guarantees to carry
over to Metal on macOS.

## Status

**Real device-discovery baseline with one real, hardware-verified compute
Kernel (`add`), not yet a full Provider.** `WgpuProvider::new()` requests
a real `wgpu` adapter and device -- on this crate's own development
machine (Windows, a real NVIDIA GPU), `wgpu` selects the real Vulkan
backend, genuinely discovering and driving that hardware, not a mock.
`WgpuProvider::add` uploads real data to real device-resident storage
buffers, dispatches a real compiled WGSL compute shader, and reads the
real result back -- verified against hand-computed expected values and
(see its own conformance test) against `providers/cpu`'s reference `add`
implementation, on this real GPU.

**What is real and verified here** (on this crate's own Windows/NVIDIA/
Vulkan development environment):

- Real adapter/device discovery through `wgpu`'s real Vulkan backend.
- One real, hardware-executed compute Kernel (`add`), including a
  non-round input-size case proving the shader's own workgroup bounds
  check is correct.
- Graceful unavailability: `WgpuProvider::new()` always constructs
  successfully even without a compatible GPU/backend, matching
  `CudaProvider`'s own posture for CUDA.

**What rests on `wgpu`'s own guarantees, not this crate's own testing**:
real execution via the Metal backend on macOS. No macOS development
environment or CI runner exists anywhere in this repository's current
tooling -- this crate has never been compiled or run on macOS. `wgpu`
itself is verified, cross-backend-consistent software with a large real
user base spanning exactly this Vulkan-develop/Metal-deploy pattern; this
crate's own real Vulkan-backed verification is the strongest evidence
available from this development environment that the *Rust code calling
wgpu* is correct, which is the only part of the real Metal behavior this
crate's own source controls.

**What is explicitly not implemented yet**: the rest of the required
Operator set (`matmul`/`rmsnorm`/`rope`/`attention`/`silu`/`residual-add`/
...) `providers/cpu`/`providers/cuda` both implement, and the full
`ProviderExecutionApi`/Kernel Registry dispatch integration (this
baseline's `add` is directly callable, not yet reachable through the
generic dispatch contract) -- real, tracked future work, matching the
same "one real Kernel first, wired into dispatch later" sequencing
`providers/cuda`'s own half-precision work already used in this
repository's history (`add-native-cuda-half-precision-compute` then
`wire-cuda-half-precision-into-kernel-registry-dispatch`).

## Governing contract

Implements Magnetar's generic `Provider`/`Device`/`ProviderExecutionApi`
contracts from the main [Magnetar](https://github.com/astorise/Magnetar)
repository's `magnetar-runtime` crate. No dedicated OpenSpec capability
exists yet for this crate specifically; one should be scoped in the main
repository's `openspec/specs/` once the rest of the Operator set and
dispatch-contract wiring land, the same way `providers/cuda`'s own
`cuda-provider` capability was.

## Relationship to magnetar-runtime

This Provider is loaded and driven by the Runtime's own Provider registry,
never the reverse -- `magnetar-runtime` has zero compile-time dependency
on this crate (the same externalization invariant every Provider/Component/
Format module in this workspace observes). It is pinned into the main
Magnetar repository as a git submodule at `providers/wgpu`.
