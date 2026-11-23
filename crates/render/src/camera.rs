use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Orbit camera: rotates around `target` at `distance`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Camera {
    pub target: Vec3,
    pub distance: f32,
    pub rotation: Quat,
    /// Vertical field of view, radians.
    pub fov_y: f32,
    /// Radius of the scene's bounding sphere; drives clip planes and fog.
    pub scene_radius: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 50.0,
            rotation: Quat::IDENTITY,
            fov_y: 35f32.to_radians(),
            scene_radius: 20.0,
        }
    }
}

impl Camera {
    /// Frame a bounding sphere so it fills the view.
    pub fn fit(&mut self, center: Vec3, radius: f32) {
        self.target = center;
        self.scene_radius = radius;
        self.distance = radius / (self.fov_y * 0.5).tan() * 0.95;
    }

    pub fn eye(&self) -> Vec3 {
        self.target + self.rotation * Vec3::new(0.0, 0.0, self.distance)
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, self.rotation * Vec3::Y)
    }

    pub fn projection(&self, aspect: f32) -> Mat4 {
        let (near, far) = self.clip_planes();
        Mat4::perspective_rh(self.fov_y, aspect.max(1e-3), near, far)
    }

    /// Near/far planes that tightly enclose the scene (better depth precision).
    pub fn clip_planes(&self) -> (f32, f32) {
        let margin = self.scene_radius * 1.2 + 2.0;
        let near = (self.distance - margin).max(self.distance * 0.01).max(0.05);
        let far = self.distance + margin;
        (near, far)
    }

    /// Rotate the view by a mouse drag, in radians around the screen axes.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.rotation = (self.rotation
            * Quat::from_rotation_y(-dx)
            * Quat::from_rotation_x(-dy))
        .normalize();
    }

    /// Translate the target so the scene follows the cursor.
    /// `dx`/`dy` are in pixels, `viewport_height` in pixels.
    pub fn pan(&mut self, dx: f32, dy: f32, viewport_height: f32) {
        let units_per_px = 2.0 * self.distance * (self.fov_y * 0.5).tan() / viewport_height.max(1.0);
        let right = self.rotation * Vec3::X;
        let up = self.rotation * Vec3::Y;
        self.target += (-right * dx + up * dy) * units_per_px;
    }

    /// Multiply the distance by `factor` (< 1 zooms in).
    pub fn zoom(&mut self, factor: f32) {
        let min = 2.0;
        let max = self.scene_radius * 20.0 + 50.0;
        self.distance = (self.distance * factor).clamp(min, max);
    }
}
