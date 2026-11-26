# wgpu-provider Specification

## Purpose
TBD - created by archiving change add-rocm-metal-provider-skeletons-and-real-wgpu-provider. Update Purpose after archive.
## Requirements
### Requirement: WGPU Provider Real Device Discovery

The WGPU Provider SHALL construct successfully on every host, regardless of whether a compatible `wgpu` adapter/device is available. At construction, it SHALL request a real, headless (no display handle) `wgpu` adapter with high-performance power preference and no fingerprint-reducing limit bucketing, then request a device from it. Any failure in either request SHALL result in `ProviderHealth::Unavailable`, never a construction error or panic.

#### Scenario: Compatible adapter available

- **GIVEN** a host with a `wgpu`-compatible Vulkan, Metal, or DX12 backend
- **WHEN** the WGPU Provider is constructed
- **THEN** it reports `ProviderHealth::Available`
- **AND** real adapter information (name, backend, vendor) is recorded

#### Scenario: No compatible adapter available

- **GIVEN** a host with no `wgpu`-compatible backend
- **WHEN** the WGPU Provider is constructed
- **THEN** construction succeeds
- **AND** it reports `ProviderHealth::Unavailable`

### Requirement: WGPU Provider Real `add` Kernel

The WGPU Provider SHALL provide a directly-callable `add` operation that uploads two equal-length `f32` slices to real device-resident storage buffers, dispatches a real compiled WGSL compute shader over them, and reads the real elementwise sum back to the host. It SHALL be correct for input lengths that are not an exact multiple of the shader's workgroup size, and SHALL match `providers/cpu`'s reference `add` implementation for the same input.

This operation is not yet wired into the `ProviderExecutionApi`/Kernel Registry dispatch contract; callers invoke it directly and are responsible for checking availability first.

#### Scenario: Correct elementwise sum

- **GIVEN** two equal-length `f32` slices and an available WGPU Provider
- **WHEN** `add` is called
- **THEN** the result is the real, hardware-computed elementwise sum

#### Scenario: Non-multiple-of-workgroup-size input

- **GIVEN** an input length that is not an exact multiple of the shader's workgroup size
- **WHEN** `add` is called
- **THEN** every element is computed correctly, with no out-of-bounds access

#### Scenario: Conformance with Reference CPU

- **GIVEN** the same real input tensors run through both the WGPU Provider's `add` and `providers/cpu`'s reference `add`
- **WHEN** both results are compared
- **THEN** they match exactly

