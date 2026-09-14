//! Real GPU compute Kernels, dispatched through `wgpu`'s real compute
//! pipeline -- not a stub. This baseline implements exactly one Operator
//! (`add`, elementwise) end to end, real WGSL compiled and executed on
//! whatever real GPU `wgpu` selected (this development machine: an
//! NVIDIA GPU via the real Vulkan backend), verified against
//! `providers/cpu`'s reference implementation.
//!
//! # Why `add` first, and why only `add` so far
//!
//! Matches this repository's own established precedent for a first real
//! increment on a new compute backend (`providers/cuda`'s own real
//! half-precision work landed one real, hardware-verified elementwise
//! Kernel pair before being wired into the generic dispatch contract in a
//! later, separate chantier) -- `add` is the simplest real Operator to
//! get right end to end (buffer upload, dispatch, readback), proving the
//! whole real GPU compute path works before investing in the rest of the
//! Operator set (`matmul`/`rmsnorm`/`rope`/`attention`/...), which is
//! real, larger future work, not attempted here.

const ADD_SHADER_SOURCE: &str = include_str!("add.wgsl");

/// Executes real elementwise `f32` addition on the GPU: uploads `a`/`b`
/// to real device-resident storage buffers, dispatches the real compiled
/// WGSL compute shader, reads the real result back, and returns it.
/// `a`/`b` must have equal, non-zero length.
pub fn add(device: &wgpu::Device, queue: &wgpu::Queue, a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "add: mismatched operand lengths");
    assert!(!a.is_empty(), "add: empty operands");

    use wgpu::util::DeviceExt;

    let element_count = a.len();
    let byte_size = (element_count * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("magnetar-wgpu-add-a"),
        contents: bytemuck_cast_f32_slice(a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("magnetar-wgpu-add-b"),
        contents: bytemuck_cast_f32_slice(b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer_output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("magnetar-wgpu-add-output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buffer_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("magnetar-wgpu-add-staging"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("magnetar-wgpu-add-shader"),
        source: wgpu::ShaderSource::Wgsl(ADD_SHADER_SOURCE.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("magnetar-wgpu-add-pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("magnetar-wgpu-add-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_output.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("magnetar-wgpu-add-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("magnetar-wgpu-add-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_count = element_count.div_ceil(64) as u32;
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buffer_output, 0, &buffer_staging, 0, byte_size);
    queue.submit(Some(encoder.finish()));

    let slice = buffer_staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::Wait).expect("device.poll failed");
    receiver
        .recv()
        .expect("map_async callback never fired")
        .expect("failed to map staging buffer for read");

    let mapped = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck_cast_to_f32_vec(&mapped);
    drop(mapped);
    buffer_staging.unmap();
    result
}

/// Minimal, local `&[f32] -> &[u8]` reinterpretation -- avoids pulling in
/// the `bytemuck` crate for one call site; `f32` has no padding/alignment
/// surprises relevant here (native-endian round-trip on the same
/// machine, matching every other raw-byte tensor reinterpretation already
/// used throughout this workspace, e.g. `providers/cpu`'s own tensor byte
/// views).
fn bytemuck_cast_f32_slice(values: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding bits, and the resulting slice's
    // lifetime and length are derived directly from `values`' own valid,
    // exclusive-borrow-free memory -- a plain reinterpretation of an
    // already-initialized, properly-aligned buffer as raw bytes.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn bytemuck_cast_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert!(
        bytes.len().is_multiple_of(std::mem::size_of::<f32>()),
        "byte length is not a multiple of f32's size"
    );
    bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}
