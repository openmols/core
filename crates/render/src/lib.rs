//! Offscreen molecular renderer built on wgpu.
//!
//! The renderer is independent of any UI toolkit: it draws into its own
//! textures and hands back BGRA8 pixels that a UI (GPUI for now) can display.
//! A second `Rg32Uint` target stores `(entity + 1, atom + 1)` per pixel for
//! picking. Each entity (a structure placed in the scene) has its own
//! geometry buffers and a rigid model transform.

mod camera;
mod cartoon;
mod scene;
mod surface;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use molecule::Structure;
use std::time::{Duration, Instant};

pub use camera::Camera;
pub use scene::{Coloring, Component, Representation, Target, default_preset};
use scene::{CylinderInstance, MeshVertex, SphereInstance};

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
/// Dynamic-offset stride for per-entity uniforms (minimum alignment is 256).
const ENTITY_STRIDE: u64 = 256;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Entity index reported by [`Renderer::pick`] for pixels covered by the
/// overlay; the atom index is then the `id` the overlay shape was given.
pub const OVERLAY_ENTITY: usize = 0x00FF_0000;

/// A world-space cylinder in the overlay (measurement lines, box edges).
#[derive(Clone, Copy, Debug)]
pub struct OverlayCylinder {
    pub p0: [f32; 3],
    pub p1: [f32; 3],
    pub radius: f32,
    pub color: [u8; 4],
    /// Reported by `pick` (0 = nothing in particular).
    pub id: u32,
}

/// A world-space sphere in the overlay (handles, markers).
#[derive(Clone, Copy, Debug)]
pub struct OverlaySphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub color: [u8; 4],
    pub id: u32,
}

/// A translucent quad in the overlay (box faces); `color` carries the alpha.
#[derive(Clone, Copy, Debug)]
pub struct OverlayFace {
    pub corners: [[f32; 3]; 4],
    pub color: [u8; 4],
}

/// Everything drawn over the molecules, in world space and depth-correct.
#[derive(Clone, Debug, Default)]
pub struct Overlay {
    pub cylinders: Vec<OverlayCylinder>,
    pub spheres: Vec<OverlaySphere>,
    /// Spheres drawn on top of everything (and pickable through the molecule):
    /// drag handles.
    pub handles: Vec<OverlaySphere>,
    pub faces: Vec<OverlayFace>,
}

/// A translucent (or wireframe) surface around some of an entity's atoms.
#[derive(Clone, Debug)]
pub struct Volume {
    /// Atoms the surface wraps (same length as the structure's atoms).
    pub selected: Vec<bool>,
    /// RGBA per atom; the alpha is the surface's opacity there.
    pub colors: Vec<[u8; 4]>,
    /// Draw the mesh edges as well.
    pub wire: bool,
}

/// Background colour (sRGB, 0..1).
pub const BACKGROUND: [f32; 3] = [0.075, 0.08, 0.095];

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Globals {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    background: [f32; 4],
    fog: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EntityUniform {
    model: [[f32; 4]; 4],
    info: [u32; 4],
}

/// A rendered frame: tightly packed BGRA8 rows, top row first.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub bgra: Vec<u8>,
    /// Time spent rendering on the GPU plus copying the pixels back.
    pub elapsed: Duration,
}

struct RenderTarget {
    width: u32,
    height: u32,
    color: wgpu::Texture,
    pick: wgpu::Texture,
    depth_view: wgpu::TextureView,
    color_view: wgpu::TextureView,
    pick_view: wgpu::TextureView,
    readback: wgpu::Buffer,
    padded_row: u32,
}

struct VolumeBuffers {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    wire: Option<(wgpu::Buffer, u32)>,
}

struct OverlayBuffers {
    cylinders: Option<wgpu::Buffer>,
    cylinder_count: u32,
    spheres: Option<wgpu::Buffer>,
    sphere_count: u32,
    faces: Option<(wgpu::Buffer, wgpu::Buffer)>,
    face_index_count: u32,
    handles: Option<wgpu::Buffer>,
    handle_count: u32,
}

struct EntityBuffers {
    volume: Option<VolumeBuffers>,
    spheres: Option<wgpu::Buffer>,
    sphere_count: u32,
    cylinders: Option<wgpu::Buffer>,
    cylinder_count: u32,
    mesh: Option<(wgpu::Buffer, wgpu::Buffer)>,
    index_count: u32,
    transform: Mat4,
    visible: bool,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    sphere_pipeline: wgpu::RenderPipeline,
    cylinder_pipeline: wgpu::RenderPipeline,
    mesh_pipeline: wgpu::RenderPipeline,
    translucent_pipeline: wgpu::RenderPipeline,
    wire_pipeline: wgpu::RenderPipeline,
    ghost_pipeline: wgpu::RenderPipeline,
    gizmo_pipeline: wgpu::RenderPipeline,
    overlay: Option<OverlayBuffers>,
    #[allow(dead_code)] // kept alive for the bind group
    overlay_uniform: wgpu::Buffer,
    overlay_bind_group: wgpu::BindGroup,
    globals: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    entity_layout: wgpu::BindGroupLayout,
    entity_uniforms: wgpu::Buffer,
    entity_bind_group: wgpu::BindGroup,
    entity_capacity: usize,
    pick_readback: wgpu::Buffer,
    entities: Vec<Option<EntityBuffers>>,
    target: Option<RenderTarget>,
}

impl Renderer {
    /// Create a renderer on the highest-performance adapter available.
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("no suitable GPU adapter found")?;
        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("openmols renderer"),
                ..Default::default()
            })
            .await
            .context("failed to create GPU device")?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("impostors"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            }],
        });
        let entity_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("entity"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<EntityUniform>() as u64),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("impostors"),
            bind_group_layouts: &[&bind_group_layout, &entity_layout],
            push_constant_ranges: &[],
        });

        let make_pipeline = |label: &str,
                             vs: &str,
                             fs: &str,
                             stride: usize,
                             step_mode: wgpu::VertexStepMode,
                             attributes: &[wgpu::VertexAttribute]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: stride as u64,
                        step_mode,
                        attributes,
                    }],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[
                        Some(COLOR_FORMAT.into()),
                        Some(PICK_FORMAT.into()),
                    ],
                }),
                multiview: None,
                cache: None,
            })
        };

        let sphere_pipeline = make_pipeline(
            "spheres",
            "vs_sphere",
            "fs_sphere",
            std::mem::size_of::<SphereInstance>(),
            wgpu::VertexStepMode::Instance,
            &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Unorm8x4, 3 => Uint32],
        );
        let cylinder_pipeline = make_pipeline(
            "cylinders",
            "vs_cylinder",
            "fs_cylinder",
            std::mem::size_of::<CylinderInstance>(),
            wgpu::VertexStepMode::Instance,
            &wgpu::vertex_attr_array![
                0 => Float32x3, 1 => Float32, 2 => Float32x3,
                3 => Unorm8x4, 4 => Unorm8x4, 5 => Uint32, 6 => Uint32
            ],
        );

        let mesh_pipeline = make_pipeline(
            "mesh",
            "vs_mesh",
            "fs_mesh",
            std::mem::size_of::<MeshVertex>(),
            wgpu::VertexStepMode::Vertex,
            &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Unorm8x4, 3 => Uint32],
        );

        // Translucent surfaces and wireframes: blended, no depth writes, and
        // the pick target masked so they cannot be clicked.
        let make_blended = |label: &str, topology: wgpu::PrimitiveTopology| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_mesh_alpha"),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<MeshVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Unorm8x4, 3 => Uint32],
                    }],
                },
                primitive: wgpu::PrimitiveState { topology, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_mesh_alpha"),
                    compilation_options: Default::default(),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: COLOR_FORMAT,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: PICK_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::empty(),
                        }),
                    ],
                }),
                multiview: None,
                cache: None,
            })
        };
        // Overlay variants of the impostor pipelines: the ghost of a line behind
        // the molecule (faint, blended, only where something is in front), and
        // gizmo spheres that ignore depth entirely.
        let make_instanced = |label: &str,
                              vs: &str,
                              fs: &str,
                              stride: usize,
                              attributes: &[wgpu::VertexAttribute],
                              compare: wgpu::CompareFunction,
                              ghost: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: stride as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes,
                    }],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: compare,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: COLOR_FORMAT,
                            blend: ghost.then_some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: PICK_FORMAT,
                            blend: None,
                            write_mask: if ghost { wgpu::ColorWrites::empty() } else { wgpu::ColorWrites::ALL },
                        }),
                    ],
                }),
                multiview: None,
                cache: None,
            })
        };
        let ghost_pipeline = make_instanced(
            "ghost cylinders",
            "vs_cylinder",
            "fs_cylinder_ghost",
            std::mem::size_of::<CylinderInstance>(),
            &wgpu::vertex_attr_array![
                0 => Float32x3, 1 => Float32, 2 => Float32x3,
                3 => Unorm8x4, 4 => Unorm8x4, 5 => Uint32, 6 => Uint32
            ],
            wgpu::CompareFunction::Greater,
            true,
        );
        let gizmo_pipeline = make_instanced(
            "gizmos",
            "vs_sphere",
            "fs_sphere",
            std::mem::size_of::<SphereInstance>(),
            &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Unorm8x4, 3 => Uint32],
            wgpu::CompareFunction::Always,
            false,
        );
        let translucent_pipeline = make_blended("translucent", wgpu::PrimitiveTopology::TriangleList);
        let wire_pipeline = make_blended("wire", wgpu::PrimitiveTopology::LineList);

        // The overlay lives in world space: an identity placement that
        // reports `OVERLAY_ENTITY` when picked.
        let overlay_uniform = {
            use wgpu::util::DeviceExt;
            let uniform = EntityUniform {
                model: Mat4::IDENTITY.to_cols_array_2d(),
                info: [OVERLAY_ENTITY as u32, 0, 0, 0],
            };
            let mut bytes = vec![0u8; ENTITY_STRIDE as usize];
            bytes[..std::mem::size_of::<EntityUniform>()].copy_from_slice(bytemuck::bytes_of(&uniform));
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("overlay uniform"),
                contents: &bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let overlay_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay"),
            layout: &entity_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &overlay_uniform,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<EntityUniform>() as u64),
                }),
            }],
        });

        let pick_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick readback"),
            size: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Placeholder until the first `ensure_entity_uniforms`.
        let entity_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entity uniforms"),
            size: ENTITY_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let entity_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entity"),
            layout: &entity_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &entity_uniforms,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<EntityUniform>() as u64),
                }),
            }],
        });

        Ok(Self {
            device,
            queue,
            adapter_info,
            sphere_pipeline,
            cylinder_pipeline,
            mesh_pipeline,
            translucent_pipeline,
            wire_pipeline,
            ghost_pipeline,
            gizmo_pipeline,
            overlay: None,
            overlay_uniform,
            overlay_bind_group,
            globals,
            bind_group,
            entity_layout,
            entity_uniforms,
            entity_bind_group,
            entity_capacity: 1,
            pick_readback,
            entities: Vec::new(),
            target: None,
        })
    }

    /// e.g. `NVIDIA GeForce RTX 4070 Laptop GPU (Vulkan)`.
    pub fn adapter_name(&self) -> String {
        format!("{} ({:?})", self.adapter_info.name, self.adapter_info.backend)
    }

    /// Single-structure convenience: replaces all entities with one at identity.
    pub fn set_scene(&mut self, structure: &Structure, components: &[Component], selection: &[usize]) {
        self.entities.clear();
        self.upload_entity(0, structure, components, selection);
    }

    /// Build and upload geometry for entity `slot` (created if needed).
    /// Surfaces are computed on the CPU here, so this can take a moment for
    /// very large structures. `selection` lists atom indices matched by
    /// [`Target::Selection`].
    pub fn upload_entity(&mut self, slot: usize, structure: &Structure, components: &[Component], selection: &[usize]) {
        // Two chains per entity step keeps chain A/B of different entities apart.
        let data = scene::build(structure, components, selection, slot * 2);
        let buffers = self.upload(data);
        if self.entities.len() <= slot {
            self.entities.resize_with(slot + 1, || None);
        }
        let (transform, visible, volume) = match self.entities[slot].take() {
            Some(e) => (e.transform, e.visible, e.volume),
            None => (Mat4::IDENTITY, true, None),
        };
        self.entities[slot] = Some(EntityBuffers {
            transform,
            visible,
            volume,
            ..buffers
        });
    }

    /// Wrap the selected atoms of an entity in a translucent (or wireframe)
    /// surface that follows the entity's placement; `None` removes it.
    /// Rebuilt from `structure`'s current coordinates, so call it again after
    /// the entity's geometry is uploaded again.
    pub fn set_entity_volume(&mut self, slot: usize, structure: &Structure, volume: Option<&Volume>) {
        use wgpu::util::DeviceExt;
        let buffers = volume.and_then(|v| {
            let color = |atom: usize| v.colors.get(atom).copied().unwrap_or([128, 160, 255, 90]);
            let mesh = surface::gaussian(structure, &v.selected, &color);
            if mesh.indices.is_empty() {
                return None;
            }
            let buf = |label: &str, bytes: &[u8], usage| {
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytes, usage })
            };
            let wire = v.wire.then(|| {
                // Each triangle's three edges once (shared edges are drawn twice, which is invisible).
                let mut lines: Vec<u32> = Vec::with_capacity(mesh.indices.len() * 2);
                for t in mesh.indices.chunks_exact(3) {
                    lines.extend([t[0], t[1], t[1], t[2], t[2], t[0]]);
                }
                (buf("volume wire", bytemuck::cast_slice(&lines), wgpu::BufferUsages::INDEX), lines.len() as u32)
            });
            Some(VolumeBuffers {
                vertices: buf("volume vertices", bytemuck::cast_slice(&mesh.vertices), wgpu::BufferUsages::VERTEX),
                indices: buf("volume indices", bytemuck::cast_slice(&mesh.indices), wgpu::BufferUsages::INDEX),
                index_count: mesh.indices.len() as u32,
                wire,
            })
        });
        if let Some(Some(e)) = self.entities.get_mut(slot) {
            e.volume = buffers;
        }
    }

    /// Replace the overlay: measurement lines, the docking box, handles.
    pub fn set_overlay(&mut self, overlay: &Overlay) {
        use wgpu::util::DeviceExt;
        let buffer = |label: &str, contents: &[u8], usage| {
            (!contents.is_empty()).then(|| {
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
            })
        };
        let cylinders: Vec<CylinderInstance> = overlay
            .cylinders
            .iter()
            .map(|c| CylinderInstance {
                p0: c.p0,
                radius: c.radius,
                p1: c.p1,
                color0: c.color,
                color1: c.color,
                id0: c.id,
                id1: c.id,
            })
            .collect();
        let spheres: Vec<SphereInstance> = overlay
            .spheres
            .iter()
            .map(|s| SphereInstance { center: s.center, radius: s.radius, color: s.color, id: s.id })
            .collect();
        let handles: Vec<SphereInstance> = overlay
            .handles
            .iter()
            .map(|s| SphereInstance { center: s.center, radius: s.radius, color: s.color, id: s.id })
            .collect();
        let mut mesh = scene::Mesh::default();
        for face in &overlay.faces {
            let c = face.corners.map(glam::Vec3::from_array);
            let normal = (c[1] - c[0]).cross(c[3] - c[0]).normalize_or_zero().to_array();
            let base = mesh.vertices.len() as u32;
            for p in c {
                mesh.vertices.push(MeshVertex { position: p.to_array(), normal, color: face.color, id: 0 });
            }
            mesh.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        self.overlay = Some(OverlayBuffers {
            cylinders: buffer("overlay cylinders", bytemuck::cast_slice(&cylinders), wgpu::BufferUsages::VERTEX),
            cylinder_count: cylinders.len() as u32,
            spheres: buffer("overlay spheres", bytemuck::cast_slice(&spheres), wgpu::BufferUsages::VERTEX),
            sphere_count: spheres.len() as u32,
            faces: buffer("overlay face vertices", bytemuck::cast_slice(&mesh.vertices), wgpu::BufferUsages::VERTEX)
                .zip(buffer("overlay face indices", bytemuck::cast_slice(&mesh.indices), wgpu::BufferUsages::INDEX)),
            face_index_count: mesh.indices.len() as u32,
            handles: buffer("overlay handles", bytemuck::cast_slice(&handles), wgpu::BufferUsages::VERTEX),
            handle_count: handles.len() as u32,
        });
    }

    /// Cheap per-frame update of an entity's placement and visibility.
    pub fn set_entity_pose(&mut self, slot: usize, transform: Mat4, visible: bool) {
        if let Some(Some(e)) = self.entities.get_mut(slot) {
            e.transform = transform;
            e.visible = visible;
        }
    }

    pub fn remove_entity(&mut self, slot: usize) {
        if let Some(e) = self.entities.get_mut(slot) {
            *e = None;
        }
    }

    /// Drop entities at `len..`.
    pub fn truncate_entities(&mut self, len: usize) {
        self.entities.truncate(len);
    }

    fn upload(&self, data: scene::SceneData) -> EntityBuffers {
        use wgpu::util::DeviceExt;
        let buffer = |label: &str, contents: &[u8], usage| {
            (!contents.is_empty()).then(|| {
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                })
            })
        };
        let spheres = buffer("spheres", bytemuck::cast_slice(&data.spheres), wgpu::BufferUsages::VERTEX);
        let cylinders = buffer("cylinders", bytemuck::cast_slice(&data.cylinders), wgpu::BufferUsages::VERTEX);
        let vertices = buffer("mesh vertices", bytemuck::cast_slice(&data.mesh.vertices), wgpu::BufferUsages::VERTEX);
        let indices = buffer("mesh indices", bytemuck::cast_slice(&data.mesh.indices), wgpu::BufferUsages::INDEX);
        EntityBuffers {
            spheres,
            sphere_count: data.spheres.len() as u32,
            cylinders,
            cylinder_count: data.cylinders.len() as u32,
            mesh: vertices.zip(indices),
            index_count: data.mesh.indices.len() as u32,
            transform: Mat4::IDENTITY,
            visible: true,
            volume: None,
        }
    }

    /// Make sure the per-entity uniform buffer can hold `count` entries.
    fn ensure_entity_uniforms(&mut self, count: usize) {
        if count <= self.entity_capacity {
            return;
        }
        let capacity = count.next_power_of_two().max(8);
        self.entity_uniforms = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entity uniforms"),
            size: (capacity as u64) * ENTITY_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.entity_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entity"),
            layout: &self.entity_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.entity_uniforms,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<EntityUniform>() as u64),
                }),
            }],
        });
        self.entity_capacity = capacity;
    }

    fn ensure_target(&mut self, width: u32, height: u32) {
        if self
            .target
            .as_ref()
            .is_some_and(|t| t.width == width && t.height == height)
        {
            return;
        }
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = |label, format, usage| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let attach_copy = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        let color = texture("color", COLOR_FORMAT, attach_copy);
        let pick = texture("pick", PICK_FORMAT, attach_copy);
        let depth = texture("depth", DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let padded_row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("color readback"),
            size: (padded_row * height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.target = Some(RenderTarget {
            width,
            height,
            color_view: color.create_view(&Default::default()),
            pick_view: pick.create_view(&Default::default()),
            depth_view: depth.create_view(&Default::default()),
            color,
            pick,
            readback,
            padded_row,
        });
    }

    /// Render the current scene and read the pixels back to the CPU.
    pub fn render(&mut self, camera: &Camera, width: u32, height: u32) -> Result<Frame> {
        let start = Instant::now();
        let width = width.clamp(1, 8192);
        let height = height.clamp(1, 8192);
        self.ensure_target(width, height);
        self.ensure_entity_uniforms(self.entities.len());
        for (slot, entity) in self.entities.iter().enumerate() {
            if let Some(entity) = entity {
                let uniform = EntityUniform {
                    model: entity.transform.to_cols_array_2d(),
                    info: [slot as u32, 0, 0, 0],
                };
                self.queue
                    .write_buffer(&self.entity_uniforms, slot as u64 * ENTITY_STRIDE, bytemuck::bytes_of(&uniform));
            }
        }
        let target = self.target.as_ref().expect("target was just created");

        let fog_start = camera.distance - camera.scene_radius * 0.3;
        let globals = Globals {
            view: camera.view().to_cols_array_2d(),
            proj: camera.projection(width as f32 / height as f32).to_cols_array_2d(),
            background: [BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 1.0],
            fog: [fog_start, camera.distance + camera.scene_radius, 0.75, 0.0],
        };
        self.queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let bg = wgpu::Color {
                r: BACKGROUND[0] as f64,
                g: BACKGROUND[1] as f64,
                b: BACKGROUND[2] as f64,
                a: 1.0,
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("molecule"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &target.color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(bg),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &target.pick_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            for (slot, scene) in self.entities.iter().enumerate() {
                let Some(scene) = scene.as_ref().filter(|e| e.visible) else {
                    continue;
                };
                pass.set_bind_group(1, &self.entity_bind_group, &[(slot as u64 * ENTITY_STRIDE) as u32]);
                if let Some(cylinders) = &scene.cylinders {
                    pass.set_pipeline(&self.cylinder_pipeline);
                    pass.set_vertex_buffer(0, cylinders.slice(..));
                    pass.draw(0..6, 0..scene.cylinder_count);
                }
                if let Some(spheres) = &scene.spheres {
                    pass.set_pipeline(&self.sphere_pipeline);
                    pass.set_vertex_buffer(0, spheres.slice(..));
                    pass.draw(0..6, 0..scene.sphere_count);
                }
                if let Some((vertices, indices)) = &scene.mesh {
                    pass.set_pipeline(&self.mesh_pipeline);
                    pass.set_vertex_buffer(0, vertices.slice(..));
                    pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..scene.index_count, 0, 0..1);
                }
            }

            // Overlay shapes that are solid: lines, boxes' edges, handles.
            pass.set_bind_group(1, &self.overlay_bind_group, &[0]);
            if let Some(o) = &self.overlay {
                if let Some(cylinders) = &o.cylinders {
                    pass.set_pipeline(&self.cylinder_pipeline);
                    pass.set_vertex_buffer(0, cylinders.slice(..));
                    pass.draw(0..6, 0..o.cylinder_count);
                }
                if let Some(spheres) = &o.spheres {
                    pass.set_pipeline(&self.sphere_pipeline);
                    pass.set_vertex_buffer(0, spheres.slice(..));
                    pass.draw(0..6, 0..o.sphere_count);
                }
            }

            // Where the molecule hides an overlay line, a faint ghost of it.
            if let Some(o) = &self.overlay
                && let Some(cylinders) = &o.cylinders
            {
                pass.set_pipeline(&self.ghost_pipeline);
                pass.set_vertex_buffer(0, cylinders.slice(..));
                pass.draw(0..6, 0..o.cylinder_count);
            }

            // Then everything translucent, over the opaque scene.
            for (slot, scene) in self.entities.iter().enumerate() {
                let Some(scene) = scene.as_ref().filter(|e| e.visible) else { continue };
                let Some(volume) = &scene.volume else { continue };
                pass.set_bind_group(1, &self.entity_bind_group, &[(slot as u64 * ENTITY_STRIDE) as u32]);
                pass.set_vertex_buffer(0, volume.vertices.slice(..));
                pass.set_pipeline(&self.translucent_pipeline);
                pass.set_index_buffer(volume.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..volume.index_count, 0, 0..1);
                if let Some((lines, count)) = &volume.wire {
                    pass.set_pipeline(&self.wire_pipeline);
                    pass.set_index_buffer(lines.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                }
            }
            if let Some((vertices, indices)) = self.overlay.as_ref().and_then(|o| o.faces.as_ref()) {
                pass.set_bind_group(1, &self.overlay_bind_group, &[0]);
                pass.set_pipeline(&self.translucent_pipeline);
                pass.set_vertex_buffer(0, vertices.slice(..));
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                let count = self.overlay.as_ref().map(|o| o.face_index_count).unwrap_or(0);
                pass.draw_indexed(0..count, 0, 0..1);
            }

            // Handles last and on top, so they can always be seen and grabbed.
            if let Some(o) = &self.overlay
                && let Some(handles) = &o.handles
            {
                pass.set_bind_group(1, &self.overlay_bind_group, &[0]);
                pass.set_pipeline(&self.gizmo_pipeline);
                pass.set_vertex_buffer(0, handles.slice(..));
                pass.draw(0..6, 0..o.handle_count);
            }
        }
        encoder.copy_texture_to_buffer(
            target.color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &target.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(target.padded_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let bytes = read_buffer(&self.device, &target.readback)?;
        let row = (width * 4) as usize;
        let mut bgra = Vec::with_capacity(row * height as usize);
        for y in 0..height as usize {
            let offset = y * target.padded_row as usize;
            bgra.extend_from_slice(&bytes[offset..offset + row]);
        }
        drop(bytes);
        target.readback.unmap();

        Ok(Frame {
            width,
            height,
            bgra,
            elapsed: start.elapsed(),
        })
    }

    /// `(entity slot, atom index)` under pixel `(x, y)` of the most recently
    /// rendered frame.
    pub fn pick(&self, x: u32, y: u32) -> Result<Option<(usize, usize)>> {
        let Some(target) = &self.target else {
            return Ok(None);
        };
        if x >= target.width || y >= target.height {
            return Ok(None);
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.pick,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.pick_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let bytes = read_buffer(&self.device, &self.pick_readback)?;
        let entity = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        let atom = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        drop(bytes);
        self.pick_readback.unmap();
        Ok(entity
            .checked_sub(1)
            .zip(atom.checked_sub(1))
            .map(|(e, a)| (e as usize, a as usize)))
    }
}

/// Map a buffer for reading and block until the GPU is done with it.
fn read_buffer<'a>(device: &wgpu::Device, buffer: &'a wgpu::Buffer) -> Result<wgpu::BufferView<'a>> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait)
        .context("waiting for the GPU")?;
    rx.recv()
        .context("map callback dropped")?
        .context("mapping readback buffer")?;
    Ok(slice.get_mapped_range())
}
