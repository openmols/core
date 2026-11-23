// Ray-cast sphere and cylinder impostors.
//
// Each atom/bond is a single camera-facing quad. The fragment shader
// intersects the view ray with the exact sphere/cylinder, shades the hit
// point and writes its true depth, so overlapping atoms intersect correctly.
// All math is in view space (camera at the origin, looking down -Z).

struct Globals {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    background: vec4<f32>,
    // x = fog start, y = fog end (view-space distance), z = fog strength
    fog: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;

// Per-entity placement (rigid transform) and its index for picking.
struct EntityUniform {
    model: mat4x4<f32>,
    // x = entity index
    info: vec4<u32>,
};

@group(1) @binding(0) var<uniform> e: EntityUniform;

fn model_view() -> mat4x4<f32> {
    return g.view * e.model;
}

fn pick_id(atom: u32) -> vec2<u32> {
    return vec2(e.info.x + 1u, atom + 1u);
}

struct FragOut {
    @location(0) color: vec4<f32>,
    @location(1) pick: vec2<u32>,
    @builtin(frag_depth) depth: f32,
};

fn corner(i: u32) -> vec2<f32> {
    var c = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );
    return c[i];
}

fn shade(base_srgb: vec3<f32>, n: vec3<f32>, p: vec3<f32>) -> vec4<f32> {
    let base = pow(base_srgb, vec3(2.2));
    let l = normalize(vec3(0.35, 0.55, 1.0));
    let v = normalize(-p);
    let h = normalize(l + v);
    let diffuse = max(dot(n, l), 0.0);
    let specular = pow(max(dot(n, h), 0.0), 56.0) * 0.3;
    // Darken grazing angles a little: gives atoms a soft outline.
    let facing = max(dot(n, v), 0.0);
    let rim = 1.0 - 0.45 * pow(1.0 - facing, 3.0);
    var c = base * (0.22 + 0.78 * diffuse) * rim + vec3(specular);
    c = pow(c, vec3(1.0 / 2.2));
    // Depth cueing towards the background colour.
    let f = clamp((-p.z - g.fog.x) / max(g.fog.y - g.fog.x, 1e-3), 0.0, 1.0);
    c = mix(c, g.background.rgb, f * g.fog.z);
    return vec4(c, 1.0);
}

fn depth_of(p: vec3<f32>) -> f32 {
    let clip = g.proj * vec4(p, 1.0);
    return clip.z / clip.w;
}

// ---------------------------------------------------------------- spheres

struct SphereIn {
    @location(0) center: vec3<f32>,
    @location(1) radius: f32,
    @location(2) color: vec4<f32>,
    @location(3) id: u32,
};

struct SphereVary {
    @builtin(position) clip: vec4<f32>,
    @location(0) view_pos: vec3<f32>,
    @location(1) @interpolate(flat) center: vec3<f32>,
    @location(2) @interpolate(flat) radius: f32,
    @location(3) @interpolate(flat) color: vec3<f32>,
    @location(4) @interpolate(flat) id: u32,
};

@vertex
fn vs_sphere(@builtin(vertex_index) vi: u32, inst: SphereIn) -> SphereVary {
    let c = (model_view() * vec4(inst.center, 1.0)).xyz;
    let to_cam = normalize(-c);
    var up = vec3(0.0, 1.0, 0.0);
    if abs(dot(up, to_cam)) > 0.99 {
        up = vec3(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(up, to_cam));
    let up2 = cross(to_cam, right);
    let q = corner(vi);
    // A quad of half-size r placed on the sphere's front, facing the camera,
    // always covers the silhouette under perspective.
    let p = c + to_cam * inst.radius + (right * q.x + up2 * q.y) * inst.radius * 1.02;

    var o: SphereVary;
    o.clip = g.proj * vec4(p, 1.0);
    o.view_pos = p;
    o.center = c;
    o.radius = inst.radius;
    o.color = inst.color.rgb;
    o.id = inst.id;
    return o;
}

@fragment
fn fs_sphere(v: SphereVary) -> FragOut {
    let dir = normalize(v.view_pos);
    let b = dot(dir, v.center);
    let c = dot(v.center, v.center) - v.radius * v.radius;
    let disc = b * b - c;
    if disc < 0.0 {
        discard;
    }
    let t = b - sqrt(disc);
    let hit = dir * t;
    let n = normalize(hit - v.center);

    var o: FragOut;
    o.color = shade(v.color, n, hit);
    o.pick = pick_id(v.id);
    o.depth = depth_of(hit);
    return o;
}

// -------------------------------------------------------------- cylinders

struct CylIn {
    @location(0) p0: vec3<f32>,
    @location(1) radius: f32,
    @location(2) p1: vec3<f32>,
    @location(3) color0: vec4<f32>,
    @location(4) color1: vec4<f32>,
    @location(5) id0: u32,
    @location(6) id1: u32,
};

struct CylVary {
    @builtin(position) clip: vec4<f32>,
    @location(0) view_pos: vec3<f32>,
    @location(1) @interpolate(flat) p0: vec3<f32>,
    @location(2) @interpolate(flat) p1: vec3<f32>,
    @location(3) @interpolate(flat) radius: f32,
    @location(4) @interpolate(flat) color0: vec3<f32>,
    @location(5) @interpolate(flat) color1: vec3<f32>,
    @location(6) @interpolate(flat) id0: u32,
    @location(7) @interpolate(flat) id1: u32,
};

@vertex
fn vs_cylinder(@builtin(vertex_index) vi: u32, inst: CylIn) -> CylVary {
    let a = (model_view() * vec4(inst.p0, 1.0)).xyz;
    let b = (model_view() * vec4(inst.p1, 1.0)).xyz;
    let ax = normalize(b - a);
    let to_cam = normalize(-(a + b) * 0.5);
    var side = cross(ax, to_cam);
    if length(side) < 1e-4 {
        side = cross(ax, vec3(0.0, 1.0, 0.0));
    }
    side = normalize(side);
    let r = inst.radius;
    let q = corner(vi);
    var along = a - ax * r;
    if q.y > 0.0 {
        along = b + ax * r;
    }
    // Widened for perspective and nudged towards the camera so the quad sits
    // in front of the cylinder surface.
    let p = along + side * q.x * r * 1.5 + to_cam * r;

    var o: CylVary;
    o.clip = g.proj * vec4(p, 1.0);
    o.view_pos = p;
    o.p0 = a;
    o.p1 = b;
    o.radius = r;
    o.color0 = inst.color0.rgb;
    o.color1 = inst.color1.rgb;
    o.id0 = inst.id0;
    o.id1 = inst.id1;
    return o;
}

@fragment
fn fs_cylinder(v: CylVary) -> FragOut {
    let dir = normalize(v.view_pos);
    let axis = v.p1 - v.p0;
    let len = length(axis);
    let ax = axis / len;
    let oc = -v.p0;
    let d = dir - ax * dot(dir, ax);
    let o = oc - ax * dot(oc, ax);
    let qa = dot(d, d);
    let qb = 2.0 * dot(d, o);
    let qc = dot(o, o) - v.radius * v.radius;
    let disc = qb * qb - 4.0 * qa * qc;
    if disc < 0.0 || qa < 1e-8 {
        discard;
    }
    let t = (-qb - sqrt(disc)) / (2.0 * qa);
    let hit = dir * t;
    let s = dot(hit - v.p0, ax);
    if s < 0.0 || s > len {
        discard;
    }
    let n = normalize((hit - v.p0) - ax * s);
    let second = s > len * 0.5;

    var out: FragOut;
    out.color = shade(select(v.color0, v.color1, second), n, hit);
    out.pick = pick_id(select(v.id0, v.id1, second));
    out.depth = depth_of(hit);
    return out;
}

// ------------------------------------------------------------------ meshes

struct MeshIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) id: u32,
};

struct MeshVary {
    @builtin(position) clip: vec4<f32>,
    @location(0) view_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) @interpolate(flat) id: u32,
};

@vertex
fn vs_mesh(v: MeshIn) -> MeshVary {
    let p = (model_view() * vec4(v.position, 1.0)).xyz;
    var o: MeshVary;
    o.clip = g.proj * vec4(p, 1.0);
    o.view_pos = p;
    // Rigid transforms only, so the model-view matrix also transforms normals.
    o.normal = (model_view() * vec4(v.normal, 0.0)).xyz;
    o.color = v.color.rgb;
    o.id = v.id;
    return o;
}

@fragment
fn fs_mesh(v: MeshVary) -> FragOut {
    var n = normalize(v.normal);
    // Two-sided lighting.
    if dot(n, -v.view_pos) < 0.0 {
        n = -n;
    }
    var out: FragOut;
    out.color = shade(v.color, n, v.view_pos);
    out.pick = pick_id(v.id);
    out.depth = v.clip.z;
    return out;
}
