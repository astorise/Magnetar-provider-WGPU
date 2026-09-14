//! Cross-platform WGPU (Vulkan/Metal/DX12) execution Provider for the
//! Magnetar local AI Runtime.
//!
//! **Status: real device discovery + one real, hardware-verified compute
//! Kernel (`add`).** See [`provider::WgpuProvider`]'s own module doc
//! comment and this crate's README for what that means, what remains
//! unimplemented, and why `wgpu` (rather than hand-written Metal FFI) is
//! this repository's real path to Apple Metal support.

mod kernels;
mod provider;

pub use provider::{WgpuProvider, wgpu_provider_metadata};
