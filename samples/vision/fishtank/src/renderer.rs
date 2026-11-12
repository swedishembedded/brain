// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The GPU renderer on wgpu, presented to a surface.
//!
//! Once, at start-up, the material texture is generated into every mip
//! level (`shaders/bake.wgsl`). Then, every frame, two passes:
//!
//! ```text
//! scene:   ray traced, shaded and tonemapped (shaders/scene.wgsl)  [scale x output]
//! present: upscale, dither, HUD (shaders/present.wgsl)             [output]
//! ```
//!
//! The scene renders at `scale` times the output size. On a window the
//! scale follows the measured GPU time so a frame fits its budget: when the
//! GPU is throttled the scene keeps all its shading and loses pixels, not
//! the other way round. GPU time comes from timestamp queries read back a
//! few frames late, without ever waiting for the GPU.
//!
//! The target is either a window surface or, headless, an offscreen texture
//! read back once for a PNG.
//!
//! The renderer knows nothing about tracking: it is handed the inverse of an
//! off-axis view-projection (from `headpose::OffAxis`) and the eye, which is
//! the whole contract between the two halves of the sample. Each pixel's ray
//! runs from the eye through that pixel's point on the physical glass.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::scene::Tank;

/// The scene target: tonemapped, so 8 bits, sRGB-encoded for even steps.
const SCENE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// The generated texture: `shaders/bake.wgsl` declares the same size.
const MATERIAL_SIZE: u32 = 512;
const MATERIAL_MIPS: u32 = MATERIAL_SIZE.ilog2() + 1;
/// The adaptive scale's range, and the share of the frame interval the GPU
/// may use (the rest is headroom for the compositor and clock changes).
const SCALE_RANGE: (f32, f32) = (0.35, 1.0);
const BUDGET_SHARE: f64 = 0.75;

/// The per-frame uniform block; `shaders/scene.wgsl` declares the same.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    inv_view_proj: [f32; 16],
    eye: [f32; 4],
    tank: [f32; 4],
}

/// The present pass's uniform block; `shaders/present.wgsl` declares the same.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PresentUniform {
    // x, y, w, h of the HUD in output pixels; w = 0: no HUD
    hud: [f32; 4],
    // x seconds (the dither's seed)
    misc: [f32; 4],
}

/// What one frame shows.
pub struct View<'a> {
    pub inv_view_proj: [f32; 16],
    pub eye: [f32; 3],
    pub time_s: f32,
    pub grid: bool,
    /// RGBA8 (sRGB) HUD pixels and their size, drawn at the top-left.
    pub hud: Option<(&'a [u8], u32, u32)>,
}

/// GPU timestamps at the start of the scene pass, its end, and the end of
/// the frame: the only honest frame cost on a shared, throttled GPU, where
/// wall time also counts everything else it runs.
struct GpuTimer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    period_ns: f64,
    /// The read buffer: IDLE, MAPPING (in flight), MAPPED or FAILED.
    state: Arc<AtomicU8>,
}

const IDLE: u8 = 0;
const MAPPING: u8 = 1;
const MAPPED: u8 = 2;
const FAILED: u8 = 3;

impl GpuTimer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> GpuTimer {
        let buffer = |label, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: 24, usage, mapped_at_creation: false });
        GpuTimer {
            set: device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("frame"), ty: wgpu::QueryType::Timestamp, count: 3 }),
            resolve: buffer("timestamps", wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            read: buffer("timestamps-read", wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ),
            period_ns: f64::from(queue.get_timestamp_period()),
            state: Arc::new(AtomicU8::new(IDLE)),
        }
    }

    /// Start mapping the read buffer, if it is free.
    fn request(&self) {
        if self.state.compare_exchange(IDLE, MAPPING, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            let state = self.state.clone();
            self.read.slice(..).map_async(wgpu::MapMode::Read, move |r| state.store(if r.is_ok() { MAPPED } else { FAILED }, Ordering::Release));
        }
    }

    /// The scene and frame milliseconds, if the mapping has landed.
    fn take(&self) -> Option<[f64; 2]> {
        match self.state.load(Ordering::Acquire) {
            MAPPED => {
                let ts: Vec<u64> = self.read.slice(..).get_mapped_range().as_chunks::<8>().0.iter().map(|b| u64::from_le_bytes(*b)).collect();
                self.read.unmap();
                self.state.store(IDLE, Ordering::Release);
                let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * self.period_ns / 1e6;
                Some([ms(ts[0], ts[1]), ms(ts[0], ts[2])])
            }
            FAILED => {
                self.state.store(IDLE, Ordering::Release);
                None
            }
            _ => None,
        }
    }
}

struct Targets {
    width: u32,
    height: u32,
    scale: f32,
    scene: wgpu::TextureView,
    present: wgpu::BindGroup,
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    output_format: wgpu::TextureFormat,
    frame_buf: wgpu::Buffer,
    frame_bg: wgpu::BindGroup,
    present_buf: wgpu::Buffer,
    present_layout: wgpu::BindGroupLayout,
    tank: Tank,
    scene: wgpu::RenderPipeline,
    present: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    hud_tex: wgpu::Texture,
    hud_size: (u32, u32),
    targets: Targets,
    timer: Option<GpuTimer>,
    /// The frame interval to fit, seconds; `None` keeps the scale fixed.
    budget_s: Option<f64>,
    /// The last measured GPU times, milliseconds: scene, whole frame.
    last_ms: Option<[f64; 2]>,
    /// The whole frame's GPU time per pixel at scale 1, smoothed: what the
    /// scale is chosen from.
    cost: Option<f64>,
}

fn module(device: &wgpu::Device, label: &str, parts: &[&str]) -> wgpu::ShaderModule {
    let src: String = parts.concat();
    device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) })
}

const NOISE: &str = include_str!("../shaders/noise.wgsl");

fn texture(device: &wgpu::Device, label: &str, w: u32, h: u32, mips: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn view(t: &wgpu::Texture) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor::default())
}

fn pass<'e>(enc: &'e mut wgpu::CommandEncoder, label: &str, target: &wgpu::TextureView, timestamps: Option<wgpu::RenderPassTimestampWrites<'e>>) -> wgpu::RenderPass<'e> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: timestamps,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

fn pipeline(device: &wgpu::Device, label: &str, layout: &[Option<&wgpu::BindGroupLayout>], m: &wgpu::ShaderModule, vs: &str, fs: &str, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: layout, immediate_size: 0 });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState { module: m, entry_point: Some(vs), compilation_options: Default::default(), buffers: &[] },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: m,
            entry_point: Some(fs),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn uniform(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
}

impl Renderer {
    /// Build every pipeline for an output of `format` at `width` x `height`.
    /// GPU timing needs the device opened with `Features::TIMESTAMP_QUERY`.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, format: wgpu::TextureFormat, width: u32, height: u32, tank: Tank) -> Renderer {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty, count: None };
        let buffer = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let tex = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false };
        let filtering = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("frame"), entries: &[entry(0, buffer), entry(1, tex), entry(2, filtering)] });
        let present_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("present"), entries: &[entry(0, tex), entry(1, filtering), entry(2, buffer), entry(3, tex)] });

        let scene_mod = module(&device, "scene", &[include_str!("../shaders/scene.wgsl")]);
        let present_mod = module(&device, "present", &[NOISE, include_str!("../shaders/present.wgsl")]);
        let bake_mod = module(&device, "bake", &[NOISE, include_str!("../shaders/bake.wgsl")]);
        let scene = pipeline(&device, "scene", &[Some(&frame_layout)], &scene_mod, "vs_scene", "fs_scene", SCENE);
        let present = pipeline(&device, "present", &[Some(&present_layout)], &present_mod, "vs_full", "fs_present", format);

        let material = Self::bake_material(&device, &queue, &bake_mod);
        let repeat = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let frame_buf = uniform(&device, "frame", std::mem::size_of::<FrameUniform>());
        let frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame"),
            layout: &frame_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view(&material)) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&repeat) },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("linear"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        let present_buf = uniform(&device, "present", std::mem::size_of::<PresentUniform>());
        let hud_tex = texture(&device, "hud", 1, 1, 1, wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let targets = Self::targets(&device, &present_layout, &sampler, &present_buf, &hud_tex, width, height, 1.0);
        let timer = device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| GpuTimer::new(&device, &queue));
        Renderer {
            device,
            queue,
            output_format: format,
            frame_buf,
            frame_bg,
            present_buf,
            present_layout,
            tank,
            scene,
            present,
            sampler,
            hud_tex,
            hud_size: (1, 1),
            targets,
            timer,
            budget_s: None,
            last_ms: None,
            cost: None,
        }
    }

    /// Generate the material texture, every mip level drawn by its own
    /// full-screen triangle (the level is the instance index).
    fn bake_material(device: &wgpu::Device, queue: &wgpu::Queue, m: &wgpu::ShaderModule) -> wgpu::Texture {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let tex = texture(device, "material", MATERIAL_SIZE, MATERIAL_SIZE, MATERIAL_MIPS, format, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let pipe = pipeline(device, "bake", &[], m, "vs_bake", "fs_bake", format);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bake") });
        for level in 0..MATERIAL_MIPS {
            let target = tex.create_view(&wgpu::TextureViewDescriptor { base_mip_level: level, mip_level_count: Some(1), ..Default::default() });
            let mut p = pass(&mut enc, "bake", &target, None);
            p.set_pipeline(&pipe);
            p.draw(0..3, level..level + 1);
        }
        queue.submit([enc.finish()]);
        tex
    }

    #[allow(clippy::too_many_arguments)]
    fn targets(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, sampler: &wgpu::Sampler, buf: &wgpu::Buffer, hud: &wgpu::Texture, width: u32, height: u32, scale: f32) -> Targets {
        let (sw, sh) = (((width as f32 * scale) as u32).max(1), ((height as f32 * scale) as u32).max(1));
        let scene = view(&texture(device, "scene", sw, sh, 1, SCENE, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING));
        let present = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&scene) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&view(hud)) },
            ],
        });
        Targets { width, height, scale, scene, present }
    }

    fn rebuild(&mut self, width: u32, height: u32, scale: f32) {
        self.targets = Self::targets(&self.device, &self.present_layout, &self.sampler, &self.present_buf, &self.hud_tex, width, height, scale);
    }

    /// Reallocate the render targets for a new output size.
    pub fn resize(&mut self, width: u32, height: u32) {
        if (width, height) != (self.targets.width, self.targets.height) && width > 0 && height > 0 {
            self.rebuild(width, height, self.targets.scale);
        }
    }

    /// Fit each frame into `interval_s` (the display's refresh) by scaling
    /// the scene's resolution. Needs GPU timestamps; without them the scale
    /// stays at 1.
    pub fn adapt_to(&mut self, interval_s: f64) {
        self.budget_s = Some(interval_s);
    }

    /// The scene's resolution as a share of the output's, per axis.
    pub fn scale(&self) -> f32 {
        self.targets.scale
    }

    /// The last measured GPU times (scene, whole frame), milliseconds.
    pub fn gpu_ms(&self) -> Option<[f64; 2]> {
        self.last_ms
    }

    /// Wait for this frame's GPU times: for measuring, not for presenting.
    pub fn wait_gpu_ms(&mut self) -> Result<Option<[f64; 2]>, String> {
        let Some(t) = &self.timer else { return Ok(None) };
        t.request();
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).map_err(|e| format!("waiting for the GPU: {e}"))?;
        self.last_ms = t.take();
        Ok(self.last_ms)
    }

    /// Pick up timestamps that have landed, and choose the scale that fits
    /// the budget. The frame's cost goes with the scene's pixel count, so
    /// the measured time divided by scale squared is the cost of a full
    /// resolution frame; smoothing that (not the scale) keeps one slow frame
    /// from moving it, and a dead band keeps it from reallocating on noise.
    fn adapt(&mut self) {
        let Some(t) = &self.timer else { return };
        let _ = self.device.poll(wgpu::PollType::Poll);
        let Some(ms) = t.take() else { return };
        self.last_ms = Some(ms);
        let Some(budget) = self.budget_s else { return };
        let s = self.targets.scale;
        let full = ms[1] / f64::from(s * s);
        let cost = self.cost.map_or(full, |c| 0.8 * c + 0.2 * full);
        self.cost = Some(cost);
        let want = ((budget * 1e3 * BUDGET_SHARE / cost).sqrt() as f32).clamp(SCALE_RANGE.0, SCALE_RANGE.1);
        if (want - s).abs() > 0.03 || (want == SCALE_RANGE.1 && s != want) {
            self.rebuild(self.targets.width, self.targets.height, want);
        }
    }

    fn upload_hud(&mut self, rgba: &[u8], w: u32, h: u32) {
        if (w, h) != self.hud_size {
            self.hud_tex = texture(&self.device, "hud", w, h, 1, wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
            self.hud_size = (w, h);
            self.rebuild(self.targets.width, self.targets.height, self.targets.scale);
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.hud_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * w), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }

    /// Record and submit one frame into `target` (a view of the output
    /// texture, of [`Renderer::output_format`], at the renderer's size).
    pub fn render(&mut self, target: &wgpu::TextureView, v: &View) {
        self.adapt();
        let hud = match v.hud {
            Some((px, w, h)) => {
                self.upload_hud(px, w, h);
                [12.0, 12.0, w as f32, h as f32]
            }
            None => [0.0; 4],
        };
        let frame = FrameUniform {
            inv_view_proj: v.inv_view_proj,
            eye: [v.eye[0], v.eye[1], v.eye[2], v.time_s],
            tank: [self.tank.half_w, self.tank.half_h, self.tank.depth, f32::from(u8::from(v.grid))],
        };
        self.queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&frame));
        self.queue.write_buffer(&self.present_buf, 0, bytemuck::bytes_of(&PresentUniform { hud, misc: [v.time_s, 0.0, 0.0, 0.0] }));

        // timestamps are written only while the read buffer is free to take them
        let timer = self.timer.as_ref().filter(|t| t.state.load(Ordering::Acquire) == IDLE);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        {
            let ts = timer.map(|t| wgpu::RenderPassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) });
            let mut p = pass(&mut enc, "scene", &self.targets.scene, ts);
            p.set_bind_group(0, &self.frame_bg, &[]);
            p.set_pipeline(&self.scene);
            p.draw(0..3, 0..1);
        }
        {
            let ts = timer.map(|t| wgpu::RenderPassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(2) });
            let mut p = pass(&mut enc, "present", target, ts);
            p.set_bind_group(0, &self.targets.present, &[]);
            p.set_pipeline(&self.present);
            p.draw(0..3, 0..1);
        }
        if let Some(t) = timer {
            enc.resolve_query_set(&t.set, 0..3, &t.resolve, 0);
            enc.copy_buffer_to_buffer(&t.resolve, 0, &t.read, 0, 24);
        }
        self.queue.submit([enc.finish()]);
        if self.budget_s.is_some() {
            if let Some(t) = timer {
                t.request();
            }
        }
    }

    /// Render one frame offscreen and read it back as an RGB image, whatever
    /// channel order the output format stores.
    pub fn render_to_rgb(&mut self, v: &View) -> Result<imaging::Rgb8, String> {
        let (w, h) = (self.targets.width, self.targets.height);
        let tex = texture(&self.device, "offscreen", w, h, 1, self.output_format, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        self.render(&view(&tex), v);
        let row = (4 * w).div_ceil(256) * 256;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).map_err(|e| format!("waiting for the GPU: {e}"))?;
        rx.recv().map_err(|e| format!("readback callback dropped: {e}"))?.map_err(|e| format!("mapping the readback buffer: {e}"))?;
        let bgra = matches!(self.output_format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
        let data = slice.get_mapped_range();
        let mut rgb = Vec::with_capacity((3 * w * h) as usize);
        for y in 0..h as usize {
            for p in data[y * row as usize..y * row as usize + 4 * w as usize].as_chunks::<4>().0 {
                rgb.extend_from_slice(&if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] });
            }
        }
        drop(data);
        buf.unmap();
        imaging::Rgb8::new(w, h, rgb)
    }
}
