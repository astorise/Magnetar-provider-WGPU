## ADDED Requirements

### Requirement: G01 WGPU production execution path

WGPU SHALL expose qualified MatMul, RMSNorm, RoPE, attention, allocation and transfer operations through ProviderExecutionApi and Kernel Registry with real Device residency.

#### Scenario: G01 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a supported real model executes on a hardware WGPU adapter
- **THEN** results and transfer traces qualify the actual graph path rather than a standalone add demo

#### Scenario: G01 Ineligible or failed execution

- **WHEN** only a software adapter or unsupported operation is available
- **THEN** the result is labeled accordingly and cannot qualify real-hardware production support

### Requirement: wgpu-model-execution promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
