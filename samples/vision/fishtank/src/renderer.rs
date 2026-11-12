// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The GPU renderer: a real raster pipeline on wgpu, presented to a surface.
//!
//! Passes, every frame:
//!
//! ```text
//! sky (full-screen, far plane) -> tank grid -> instanced objects   [HDR + depth]
//! bloom: threshold downsample, 4 more downsamples, upsample-add    [half res ..]
//! composite: bloom + ACES tonemap + vignette + dither + HUD        [sRGB target]
//! ```
//!
//! The target is either a window surface or, headless, an offscreen texture
//! read back once for a PNG. Nothing is read back per frame on screen.
//!
//! The renderer knows nothing about tracking: it is handed a view-projection
//! (from `headpose::OffAxis`) and an eye, which is the whole contract between
//! the two halves of the sample.

use bytemuck::{Pod, Zeroable};

use crate::scene::{self, Instance, Meshes, Tank, Vertex};

const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const BLOOM_LEVELS: usize = 5;
const MAX_INSTANCES: u64 = 64;

/// The per-frame uniform block; `shaders/frame.wgsl` declares the same.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    view_proj: [f32; 16],
    inv_view_proj: [f32; 16],
    eye: [f32; 4],
    tank: [f32; 4],
    light: [f32; 4],
}

/// The post-processing uniform block; `shaders/post.wgsl` declares the same.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostUniform {
    texel: [f32; 4],
    grade: [f32; 4],
    hud: [f32; 4],
}

/// What one frame shows.
pub struct View<'a> {
    pub view_proj: [f32; 16],
    pub inv_view_proj: [f32; 16],
    pub eye: [f32; 3],
    pub time_s: f32,
    pub grid: bool,
    /// RGBA8 (sRGB) HUD pixels and their size, drawn at the top-left.
    pub hud: Option<(&'a [u8], u32, u32)>,
}

struct Targets {
    width: u32,
    height: u32,
    hdr: wgpu::TextureView,
    depth: wgpu::TextureView,
    bloom: Vec<(wgpu::TextureView, u32, u32)>,
    /// One bind group per post pass, built against these views.
    down: Vec<wgpu::BindGroup>,
    up: Vec<wgpu::BindGroup>,
    composite: wgpu::BindGroup,
    post_uniforms: Vec<wgpu::Buffer>,
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    output_format: wgpu::TextureFormat,
    frame_buf: wgpu::Buffer,
    frame_bg: wgpu::BindGroup,
    vertices: wgpu::Buffer,
    instances: wgpu::Buffer,
    meshes: Meshes,
    tank: Tank,
    sky: wgpu::RenderPipeline,
    grid: wgpu::RenderPipeline,
    objects: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    post_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    hud_tex: wgpu::Texture,
    hud_size: (u32, u32),
    targets: Targets,
}

fn module(device: &wgpu::Device, label: &str, parts: &[&str]) -> wgpu::ShaderModule {
    let src: String = parts.concat();
    device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) })
}

const NOISE: &str = include_str!("../shaders/noise.wgsl");
const FRAME: &str = include_str!("../shaders/frame.wgsl");

fn texture(device: &wgpu::Device, label: &str, w: u32, h: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
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

impl Renderer {
    /// Build every pipeline for an output of `format` at `width` x `height`.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, format: wgpu::TextureFormat, width: u32, height: u32, tank: Tank) -> Renderer {
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let post_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post"),
            entries: &[
                tex_entry(0),
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex_entry(3),
                tex_entry(4),
            ],
        });
        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame"),
            layout: &frame_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() }],
        });

        let meshes = Meshes::build(tank);
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertices"),
            size: (meshes.vertices.len() * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertices, 0, bytemuck::cast_slice(&meshes.vertices));
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: MAX_INSTANCES * std::mem::size_of::<Instance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("scene"), bind_group_layouts: &[Some(&frame_layout)], immediate_size: 0 });
        let post_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("post"), bind_group_layouts: &[Some(&post_layout)], immediate_size: 0 });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4],
        };

        let depth_state = |write: bool, compare| wgpu::DepthStencilState {
            format: DEPTH,
            depth_write_enabled: Some(write),
            depth_compare: Some(compare),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let hdr_target = [Some(wgpu::ColorTargetState { format: HDR, blend: None, write_mask: wgpu::ColorWrites::ALL })];

        let sky_mod = module(&device, "sky", &[NOISE, FRAME, include_str!("../shaders/sky.wgsl")]);
        let grid_mod = module(&device, "grid", &[NOISE, FRAME, include_str!("../shaders/grid.wgsl")]);
        let obj_mod = module(&device, "objects", &[NOISE, FRAME, include_str!("../shaders/objects.wgsl")]);
        let post_mod = module(&device, "post", &[NOISE, include_str!("../shaders/post.wgsl")]);

        let pipeline = |label: &str,
                        layout: &wgpu::PipelineLayout,
                        m: &wgpu::ShaderModule,
                        vs: &str,
                        fs: &str,
                        buffers: &[wgpu::VertexBufferLayout],
                        cull: Option<wgpu::Face>,
                        depth: Option<wgpu::DepthStencilState>,
                        targets: &[Option<wgpu::ColorTargetState>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState { module: m, entry_point: Some(vs), compilation_options: Default::default(), buffers },
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: cull, ..Default::default() },
                depth_stencil: depth,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState { module: m, entry_point: Some(fs), compilation_options: Default::default(), targets }),
                multiview_mask: None,
                cache: None,
            })
        };
        let sky = pipeline("sky", &scene_layout, &sky_mod, "vs_sky", "fs_sky", &[], None, Some(depth_state(false, wgpu::CompareFunction::Always)), &hdr_target);
        let grid = pipeline("grid", &scene_layout, &grid_mod, "vs_tank", "fs_tank", std::slice::from_ref(&vertex_layout), None, Some(depth_state(true, wgpu::CompareFunction::Less)), &hdr_target);
        let objects = pipeline(
            "objects",
            &scene_layout,
            &obj_mod,
            "vs_obj",
            "fs_obj",
            &[vertex_layout, instance_layout],
            None,
            Some(depth_state(true, wgpu::CompareFunction::Less)),
            &hdr_target,
        );
        let down = pipeline("bloom-down", &post_pipeline_layout, &post_mod, "vs_full", "fs_down", &[], None, None, &hdr_target);
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent::OVER,
        };
        let up = pipeline(
            "bloom-up",
            &post_pipeline_layout,
            &post_mod,
            "vs_full",
            "fs_up",
            &[],
            None,
            None,
            &[Some(wgpu::ColorTargetState { format: HDR, blend: Some(additive), write_mask: wgpu::ColorWrites::ALL })],
        );
        let composite = pipeline(
            "composite",
            &post_pipeline_layout,
            &post_mod,
            "vs_full",
            "fs_composite",
            &[],
            None,
            None,
            &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let hud_size = (1, 1);
        let hud_tex = texture(&device, "hud", 1, 1, wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let targets = Self::targets(&device, &post_layout, &sampler, &hud_tex, width, height);
        Renderer {
            device,
            queue,
            output_format: format,
            frame_buf,
            frame_bg,
            vertices,
            instances,
            meshes,
            tank,
            sky,
            grid,
            objects,
            down,
            up,
            composite,
            post_layout,
            sampler,
            hud_tex,
            hud_size,
            targets,
        }
    }

    fn targets(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, sampler: &wgpu::Sampler, hud: &wgpu::Texture, width: u32, height: u32) -> Targets {
        let rt = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let hdr = view(&texture(device, "hdr", width, height, HDR, rt));
        let depth = view(&texture(device, "depth", width, height, DEPTH, wgpu::TextureUsages::RENDER_ATTACHMENT));
        let mut bloom = Vec::with_capacity(BLOOM_LEVELS);
        let (mut w, mut h) = (width, height);
        for _ in 0..BLOOM_LEVELS {
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            bloom.push((view(&texture(device, "bloom", w, h, HDR, rt)), w, h));
        }
        let hud_view = view(hud);
        let uniform = |device: &wgpu::Device| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("post"),
                size: std::mem::size_of::<PostUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let group = |src: &wgpu::TextureView, aux: &wgpu::TextureView, buf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(aux) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&hud_view) },
                ],
            })
        };
        // uniforms: one per downsample, one per upsample, one for the composite
        let post_uniforms: Vec<wgpu::Buffer> = (0..2 * BLOOM_LEVELS).map(|_| uniform(device)).collect();
        let mut down = Vec::new();
        for i in 0..BLOOM_LEVELS {
            let src = if i == 0 { &hdr } else { &bloom[i - 1].0 };
            down.push(group(src, &hdr, &post_uniforms[i]));
        }
        let mut up = Vec::new();
        for i in (1..BLOOM_LEVELS).rev() {
            up.push(group(&bloom[i].0, &hdr, &post_uniforms[BLOOM_LEVELS + i]));
        }
        let composite = group(&hdr, &bloom[0].0, &post_uniforms[BLOOM_LEVELS]);
        Targets { width, height, hdr, depth, bloom, down, up, composite, post_uniforms }
    }

    /// Reallocate the render targets for a new output size.
    pub fn resize(&mut self, width: u32, height: u32) {
        if (width, height) != (self.targets.width, self.targets.height) && width > 0 && height > 0 {
            self.targets = Self::targets(&self.device, &self.post_layout, &self.sampler, &self.hud_tex, width, height);
        }
    }

    fn upload_hud(&mut self, rgba: &[u8], w: u32, h: u32) {
        if (w, h) != self.hud_size {
            self.hud_tex = texture(&self.device, "hud", w, h, wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
            self.hud_size = (w, h);
            self.targets = Self::targets(&self.device, &self.post_layout, &self.sampler, &self.hud_tex, self.targets.width, self.targets.height);
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
        let hud_on = if let Some((px, w, h)) = v.hud {
            self.upload_hud(px, w, h);
            true
        } else {
            false
        };
        let t = v.time_s;
        let light = scene::crystal_position(self.tank, t);
        let uni = FrameUniform {
            view_proj: v.view_proj,
            inv_view_proj: v.inv_view_proj,
            eye: [v.eye[0], v.eye[1], v.eye[2], t],
            tank: [self.tank.half_w, self.tank.half_h, self.tank.depth, if v.grid { 1.0 } else { 0.0 }],
            light: [light[0], light[1], light[2], 0.5 + 0.5 * (t * 2.1).sin()],
        };
        self.queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&uni));

        let inst = scene::instances(self.tank, t);
        let groups = [(&inst.crystal, self.meshes.crystal), (&inst.pillar, self.meshes.pillar), (&inst.ring, self.meshes.ring), (&inst.orb, self.meshes.orb)];
        let mut all: Vec<Instance> = Vec::new();
        let mut draws = Vec::new();
        for (list, range) in groups {
            draws.push((range, all.len() as u32, list.len() as u32));
            all.extend_from_slice(list);
        }
        assert!(all.len() as u64 <= MAX_INSTANCES, "the scene outgrew its instance buffer");
        self.queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&all));

        let tg = &self.targets;
        for (i, (_, w, h)) in tg.bloom.iter().enumerate() {
            // downsample i reads the level above it (the HDR target for 0)
            let (sw, sh) = if i == 0 { (tg.width, tg.height) } else { (tg.bloom[i - 1].1, tg.bloom[i - 1].2) };
            let p = PostUniform { texel: [1.0 / sw as f32, 1.0 / sh as f32, if i == 0 { 1.0 } else { 0.0 }, 0.0], grade: [0.0; 4], hud: [0.0; 4] };
            self.queue.write_buffer(&tg.post_uniforms[i], 0, bytemuck::bytes_of(&p));
            if i > 0 {
                let p = PostUniform { texel: [1.0 / *w as f32, 1.0 / *h as f32, 0.0, 0.0], grade: [0.0; 4], hud: [0.0; 4] };
                self.queue.write_buffer(&tg.post_uniforms[BLOOM_LEVELS + i], 0, bytemuck::bytes_of(&p));
            }
        }
        let comp = PostUniform {
            texel: [1.0 / tg.width as f32, 1.0 / tg.height as f32, 0.0, 0.0],
            grade: [1.15, 0.65, t, if hud_on { 1.0 } else { 0.0 }],
            hud: [12.0, 12.0, self.hud_size.0 as f32, self.hud_size.1 as f32],
        };
        self.queue.write_buffer(&tg.post_uniforms[BLOOM_LEVELS], 0, bytemuck::bytes_of(&comp));

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &tg.hdr,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &tg.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.frame_bg, &[]);
            pass.set_pipeline(&self.sky);
            pass.draw(0..3, 0..1);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.set_pipeline(&self.grid);
            let tr = self.meshes.tank;
            pass.draw(tr.first..tr.first + tr.count, 0..1);
            pass.set_pipeline(&self.objects);
            pass.set_vertex_buffer(1, self.instances.slice(..));
            for (r, first, n) in draws {
                if n > 0 {
                    pass.draw(r.first..r.first + r.count, first..first + n);
                }
            }
        }
        let full = |enc: &mut wgpu::CommandEncoder, label: &str, dst: &wgpu::TextureView, pipe: &wgpu::RenderPipeline, bg: &wgpu::BindGroup, load: wgpu::LoadOp<wgpu::Color>| {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: dst, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load, store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        };
        for i in 0..BLOOM_LEVELS {
            full(&mut enc, "bloom-down", &tg.bloom[i].0, &self.down, &tg.down[i], wgpu::LoadOp::Clear(wgpu::Color::BLACK));
        }
        for (k, i) in (1..BLOOM_LEVELS).rev().enumerate() {
            full(&mut enc, "bloom-up", &tg.bloom[i - 1].0, &self.up, &tg.up[k], wgpu::LoadOp::Load);
        }
        full(&mut enc, "composite", target, &self.composite, &tg.composite, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
        self.queue.submit([enc.finish()]);
    }

    /// Render one frame offscreen and read it back as an RGB image, whatever
    /// channel order the output format stores.
    pub fn render_to_rgb(&mut self, v: &View) -> Result<imaging::Rgb8, String> {
        let (w, h) = (self.targets.width, self.targets.height);
        let px = self.render_to_pixels(v)?;
        let bgra = matches!(self.output_format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
        let rgb = px.as_chunks::<4>().0.iter().flat_map(|p| if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] }).collect();
        imaging::Rgb8::new(w, h, rgb)
    }

    /// Render one frame offscreen and read it back as tightly packed 8-bit
    /// pixels in the output format's channel order.
    fn render_to_pixels(&mut self, v: &View) -> Result<Vec<u8>, String> {
        let (w, h) = (self.targets.width, self.targets.height);
        let tex = texture(&self.device, "offscreen", w, h, self.output_format, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
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
        let data = slice.get_mapped_range();
        let mut out = Vec::with_capacity((4 * w * h) as usize);
        for y in 0..h as usize {
            out.extend_from_slice(&data[y * row as usize..y * row as usize + 4 * w as usize]);
        }
        drop(data);
        buf.unmap();
        Ok(out)
    }
}
