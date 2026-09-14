//! `WgpuProvider`: a real, cross-platform GPU compute Provider built on
//! `wgpu` -- real device discovery and one real, hardware-verified
//! compute Kernel (`add`, see `kernels.rs`), not yet the full
//! `ProviderExecutionApi`/Kernel Registry dispatch surface
//! `providers/cpu`/`providers/cuda` both implement.
//!
//! # Why `wgpu` instead of native Metal/Vulkan/DX12 bindings directly
//!
//! This crate exists specifically to give Magnetar a real, verifiable
//! path to Apple Metal (and, incidentally, any other Vulkan/DX12-capable
//! GPU) without writing platform-specific FFI this repository's own
//! tooling could never compile or test (see
//! [`providers/metal`](https://github.com/astorise/Magnetar-provider-Metal)'s
//! own honest, unconditionally-unavailable skeleton and its README for
//! why that direct approach was rejected). `wgpu` is a mature (35M+
//! downloads), widely-used cross-platform compute/graphics API: on this
//! development machine (Windows, a real NVIDIA GPU) it selects the real
//! Vulkan backend; on macOS it selects the real Metal backend
//! automatically, with zero Metal-specific code anywhere in this crate.
//! Real compute correctness verified here, on this real Vulkan-backed
//! GPU, rests on `wgpu`'s own cross-backend consistency guarantees to
//! carry over to Metal on macOS -- a real, load-bearing assumption this
//! crate depends on and states explicitly, not a claim that Metal itself
//! has been tested.
//!
//! # Graceful unavailability
//!
//! `WgpuProvider::new()` always constructs successfully, even on a host
//! with no compatible GPU/backend: adapter/device request failure is
//! treated the same way `CudaProvider::new()` treats a missing CUDA
//! driver -- [`ProviderHealth::Unavailable`], not a construction error.

use magnetar_runtime::affinity::ProviderHealth;
use magnetar_runtime::provider::{Provider, ProviderError, ProviderMetadata, ProviderRegistry};

const WGPU_PROVIDER_NAME: &str = "wgpu";
const WGPU_PROVIDER_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn wgpu_provider_metadata(adapter_info: Option<&wgpu::AdapterInfo>) -> ProviderMetadata {
    let vendor = adapter_info
        .map(|info| format!("{:?}", info.backend))
        .unwrap_or_else(|| "none".to_string());
    ProviderMetadata::new(
        WGPU_PROVIDER_NAME,
        WGPU_PROVIDER_VERSION,
        vendor,
        "Real, cross-platform GPU compute Provider built on wgpu (Vulkan/Metal/DX12) -- \
         one real Kernel (add) implemented so far, see this crate's own README",
    )
}

/// The WGPU Provider itself.
pub struct WgpuProvider {
    metadata: ProviderMetadata,
    adapter_info: Option<wgpu::AdapterInfo>,
    /// `Some((device, queue))` only when a compatible `wgpu` adapter and
    /// device were successfully requested at construction time.
    device: Option<(wgpu::Device, wgpu::Queue)>,
}

impl WgpuProvider {
    pub fn new() -> Self {
        let discovered = Self::discover_device();
        let adapter_info = discovered.as_ref().map(|(info, _, _)| info.clone());
        Self {
            metadata: wgpu_provider_metadata(adapter_info.as_ref()),
            adapter_info,
            device: discovered.map(|(_, device, queue)| (device, queue)),
        }
    }

    fn discover_device() -> Option<(wgpu::AdapterInfo, wgpu::Device, wgpu::Queue)> {
        // Headless compute only, no window/surface -- no display handle
        // needed (`new_with_display_handle` is only for a Provider that
        // also presents to a real window/swapchain, out of scope here).
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            // This Provider is never exposed to untrusted content (e.g. a
            // browser sandbox); real, unbucketed limits are the correct
            // choice for a trusted, local desktop application.
            apply_limit_buckets: false,
        }))
        .ok()?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("magnetar-wgpu-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        Some((info, device, queue))
    }

    /// Whether this Provider found a usable `wgpu` adapter and device.
    pub fn is_available(&self) -> bool {
        self.device.is_some()
    }

    /// Real information about the discovered adapter (backend, device
    /// name, vendor), when one was found.
    pub fn adapter_info(&self) -> Option<&wgpu::AdapterInfo> {
        self.adapter_info.as_ref()
    }

    /// Executes the real `add` Kernel on the discovered device. Panics if
    /// no device was found -- callers must check [`Self::is_available`]
    /// first, matching the same "caller's responsibility" contract this
    /// baseline's minimal scope keeps everywhere else (no Kernel Registry
    /// dispatch integration yet -- see this crate's own README).
    pub fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let (device, queue) = self
            .device
            .as_ref()
            .expect("WgpuProvider::add called with no available device");
        crate::kernels::add(device, queue, a, b)
    }
}

impl Default for WgpuProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for WgpuProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.metadata.clone()
    }

    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        Ok(())
    }

    fn health(&self) -> ProviderHealth {
        if self.is_available() {
            ProviderHealth::Available
        } else {
            ProviderHealth::Unavailable
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every real-hardware test below needs an available Provider to mean
    /// anything; on a genuinely GPU-less host (or one without a working
    /// Vulkan/Metal/DX12 backend `wgpu` can find) this gracefully skips
    /// (prints and returns) rather than failing -- the same "no hardware
    /// here" tolerance `providers/cuda`'s own benchmark already
    /// establishes, since this baseline's own graceful-unavailability
    /// behavior is exactly what a Provider with no compatible backend
    /// SHALL do, not a test failure. On this crate's own development
    /// machine (a real NVIDIA GPU via `wgpu`'s Vulkan backend), and on
    /// CI's `ubuntu-latest` runner once Mesa's `llvmpipe` software Vulkan
    /// driver is installed, a real device is genuinely found and every
    /// assertion below runs for real, not skipped.
    macro_rules! require_device_or_skip {
        ($provider:expr) => {
            if !$provider.is_available() {
                eprintln!(
                    "skipping: no wgpu-compatible device found on this host \
                     (expected on a machine/runner with no Vulkan/Metal/DX12 backend)"
                );
                return;
            }
        };
    }

    /// Real hardware verification when a device is found (see
    /// `require_device_or_skip!`'s own doc comment for when it is not).
    #[test]
    fn discovers_a_real_device_on_this_machine() {
        let provider = WgpuProvider::new();
        require_device_or_skip!(provider);
        let info = provider.adapter_info().expect("adapter info recorded");
        assert!(!info.name.is_empty());
    }

    #[test]
    fn metadata_is_well_formed() {
        let provider = WgpuProvider::new();
        let metadata = provider.metadata();
        assert_eq!(metadata.name, WGPU_PROVIDER_NAME);
        assert!(!metadata.description.is_empty());
    }

    /// Real GPU compute, verified against a hand-computed expected
    /// result -- not a mock, not a CPU fallback pretending to be GPU
    /// work.
    #[test]
    fn add_computes_the_correct_result_on_real_hardware() {
        let provider = WgpuProvider::new();
        require_device_or_skip!(provider);
        let a = vec![1.0f32, 2.0, 3.0, 4.5, -1.5];
        let b = vec![10.0f32, 20.0, 30.0, 0.5, 1.5];
        let result = provider.add(&a, &b);
        assert_eq!(result, vec![11.0, 22.0, 33.0, 5.0, 0.0]);
    }

    /// A larger, non-round-number-of-workgroups case (5000 is not a
    /// multiple of the shader's own `workgroup_size(64)`), proving the
    /// WGSL shader's own bounds check (`if index >= arrayLength(&output)`)
    /// is correct, not just convenient for a suspiciously round input size.
    #[test]
    fn add_handles_a_length_not_a_multiple_of_the_workgroup_size() {
        let provider = WgpuProvider::new();
        require_device_or_skip!(provider);
        let count = 5000;
        let a: Vec<f32> = (0..count).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..count).map(|i| (i as f32) * 2.0).collect();
        let result = provider.add(&a, &b);
        let expected: Vec<f32> = (0..count).map(|i| (i as f32) * 3.0).collect();
        assert_eq!(result, expected);
    }

    /// Conformance: this Provider's real GPU `add` output must match
    /// `providers/cpu`'s reference implementation exactly, for the same
    /// real (non-trivial, non-integer) random-ish input -- the same
    /// numerical-oracle convention `providers/cuda`'s own conformance
    /// tests already establish for its own `add` Kernel.
    #[test]
    fn add_matches_the_reference_cpu_implementation() {
        let provider = WgpuProvider::new();
        require_device_or_skip!(provider);

        let count = 777usize;
        let a: Vec<f32> = (0..count)
            .map(|i| ((i as f32) * 0.3171).sin() * 17.0)
            .collect();
        let b: Vec<f32> = (0..count)
            .map(|i| ((i as f32) * 1.7291).cos() * -4.0)
            .collect();

        let gpu_result = provider.add(&a, &b);

        let shape = vec![count as u64];
        let tensor_a =
            magnetar_runtime::HostTensor::new(shape.clone(), a.clone()).expect("valid tensor a");
        let tensor_b = magnetar_runtime::HostTensor::new(shape, b.clone()).expect("valid tensor b");
        let cpu_result =
            magnetar_provider_cpu::add(&tensor_a, &tensor_b).expect("reference CPU add succeeds");

        assert_eq!(gpu_result, cpu_result.data);
    }
}
