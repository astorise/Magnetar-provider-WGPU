## MODIFIED Requirements

### Requirement: WGPU Provider Real `add` Kernel

The WGPU Provider SHALL preserve its directly callable real elementwise F32 add operation and SHALL additionally expose qualified add execution through ProviderExecutionApi and Kernel Registry. Both paths SHALL handle non-workgroup-multiple lengths without out-of-bounds access and match the Reference CPU implementation. Generic dispatch SHALL use Runtime-owned logical tensor resources, validated Device affinity and completion-safe leases; native wgpu buffers and queues SHALL remain private. Standalone upload/compute/readback remains a compatibility helper, not evidence of a complete LLM path.

#### Scenario: Existing directly callable addition

- **GIVEN** two equal-length F32 slices and an available hardware adapter
- **WHEN** add runs with a length not divisible by the workgroup size
- **THEN** every result equals the Reference CPU result without an out-of-bounds access

#### Scenario: Generic registry dispatch

- **WHEN** eligible resident add resources are dispatched through Kernel Registry
- **THEN** the Provider executes real device compute and returns opaque completion and resource metadata

#### Scenario: No adapter or incompatible affinity

- **WHEN** generic add dispatch has no eligible Device or incompatible resource affinity
- **THEN** it returns a structured unsupported/unavailable result without constructing native handles in Component code
