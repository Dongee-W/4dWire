"""4Dwire multi-view wire art models, rendering, training and exports.

Run ``python wire4d.py --help`` for commands. The model and geometry helpers
come from the original multive_wire_art notebook in archive/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import imageio
import matplotlib
import matplotlib.cm
import numpy as np
import pydiffvg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image

ROOT = Path(__file__).resolve().parent


def prepare_output_dirs():
    for directory in (
        'outputs', 'outputs/frames/False', 'outputs/frames/True',
        'outputs/frames_dreamwire', 'fill_steps', 'render_360', 'sds_steps/frames',
    ):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)


def get_premium_colors_from_t(t_values, cmap_name='magma', device='cuda'):
    """
    Samples colors from a colormap using specific 't' values (0.0 to 1.0).
    t_values: Tensor of shape [N]
    """
    # 1. Get Colormap
    cmap = matplotlib.cm.get_cmap(cmap_name)
    
    # 2. Sample using the provided t_values (Arc Length)
    # Convert tensor to numpy for matplotlib
    t_numpy = t_values.detach().cpu().numpy()
    
    # 3. Get RGBA
    colors_numpy = cmap(t_numpy)
    
    # 4. Convert back to Tensor
    return torch.from_numpy(colors_numpy).to(device).float()

def load_alpha_mask(image_path, size=512, device='cuda'):
    # 1. Load image and ensure it's in RGBA mode
    img = Image.open(image_path).convert('RGBA')
    
    # 2. Resize to match your canvas
    img = img.resize((size, size), resample=Image.LANCZOS)
    
    # 3. Convert to tensor [4, H, W]
    img_tensor = T.ToTensor()(img).to(device)
    
    # 4. Extract Alpha channel (index 3)
    # Alpha is usually 0.0 for transparent and 1.0 for opaque
    mask = img_tensor[3, :, :]
    
    return mask

class DiffCamera:
    """
    Differentiable Orthographic Camera.
    """
    def __init__(self, eye, target, up, ortho_scale=2.0, aspect=1.0, near=-100.0, far=100.0, device='cuda'):
        """
        Args:
            eye: Position of the camera (tensor or list).
            target: Point the camera is looking at.
            up: Up vector.
            ortho_scale: The vertical size of the view volume in world units. 
                         If 2.0, the camera sees Y from -1.0 to +1.0.
            aspect: Width / Height ratio.
        """
        self.device = device
        self.eye = torch.as_tensor(eye, dtype=torch.float32, device=device)
        self.target = torch.as_tensor(target, dtype=torch.float32, device=device)
        self.up = torch.as_tensor(up, dtype=torch.float32, device=device)
        self.ortho_scale = ortho_scale
        self.aspect = aspect
        self.near = near
        self.far = far

    def get_view_matrix(self):
        # 1. Forward (Z) axis: Vector from Eye to Target
        # In standard GL, camera looks down -Z. 
        # Here we construct the basis vectors (Right, Up, Forward).
        z_axis = F.normalize(self.eye - self.target, dim=0) 
        x_axis = F.normalize(torch.cross(self.up, z_axis), dim=0)
        y_axis = torch.cross(z_axis, x_axis)

        # 2. Construct View Matrix (4x4)
        view_mat = torch.eye(4, device=self.device)
        view_mat[0, :3] = x_axis
        view_mat[1, :3] = y_axis
        view_mat[2, :3] = z_axis
        
        # Translation: Dot product with basis vectors
        view_mat[0, 3] = -torch.dot(x_axis, self.eye)
        view_mat[1, 3] = -torch.dot(y_axis, self.eye)
        view_mat[2, 3] = -torch.dot(z_axis, self.eye)
        
        return view_mat

    def get_projection_matrix(self):
        """
        Returns an Orthographic Projection Matrix.
        Maps view volume to [-1, 1] cube.
        """
        # Top = scale / 2, Right = (scale * aspect) / 2
        top = self.ortho_scale / 2.0
        right = top * self.aspect
        
        proj_mat = torch.eye(4, device=self.device)
        
        # Scaling X and Y
        proj_mat[0, 0] = 1.0 / right
        proj_mat[1, 1] = 1.0 / top
        
        # Scaling Z (map near..far to -1..1)
        proj_mat[2, 2] = -2.0 / (self.far - self.near)
        proj_mat[2, 3] = -(self.far + self.near) / (self.far - self.near)
        
        # Ortho has NO perspective divide (w remains 1)
        proj_mat[3, 3] = 1.0
        
        return proj_mat

    def get_full_matrix(self):
        return self.get_projection_matrix() @ self.get_view_matrix()

def save_gif(frames, path, fps=10):
    processed_frames = []
    
    for frame in frames:
        if torch.is_tensor(frame):
            img = frame.detach().cpu().numpy()
        else:
            img = frame.copy()

        if img.shape[0] <= 4 and img.shape[0] < img.shape[1]:
            img = np.transpose(img, (1, 2, 0))

        # 1. ALPHA BLENDING (Stay in Linear Space for the math)
        if img.shape[-1] == 4:
            alpha = img[:, :, 3:4]
            rgb = img[:, :, :3] # Keep this linear (0.1 stays 0.1)
            bg_color = np.array([1.0, 1.0, 1.0], dtype=img.dtype)
            img = rgb * alpha + bg_color * (1 - alpha)
        
        # 2. GAMMA CORRECTION (Linear -> sRGB for display)
        # We apply this to the WHOLE image after blending
        # This turns 0.1 into ~0.35, preserving your grayish look
        img = np.power(img, 1/2.2) 
        
        # 3. Scale to 0-255 uint8
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        processed_frames.append(img)

    imageio.mimsave(path, processed_frames, fps=fps, loop=0)

def render_360_view(bspline, num_frames=45, radius=1.8):
    """
    Rotates the camera around the Y-axis and renders the 3D curve
    with perspective-aware stroke widths.
    """
    if not os.path.exists("render_360"):
        os.makedirs("render_360")

    frames = []
    with torch.no_grad():
        for i in range(num_frames):
            t = i / num_frames
            
            # 1. Horizontal rotation (Around Z-axis)
            # 2 * pi for a full circle
            azimuth = 2 * math.pi * t
            
            # 2. Vertical oscillation (Tilt / Around X-axis)
            # Use a sine wave to go from eye-level -> top view -> eye-level
            # We add a small offset (0.2) so we don't look from perfectly underneath
            elevation = math.pi/100 * math.sin(2 * math.pi * t) + 0.1
            
            # Spherical to Cartesian conversion
            eye_x = radius * math.cos(elevation) * math.cos(azimuth)
            eye_y = radius * math.cos(elevation) * math.sin(azimuth)
            eye_z = radius * math.sin(elevation)
            
            temp_cam = DiffCamera(eye=[eye_x, eye_y, eye_z], 
                                  target=[0.0, 0.0, 0.0], 
                                  up=[0.0, 0.0, 1.0], 
                                  device=device)
            
            scene_args = bspline.get_serialized_scene(temp_cam, 512, 512, inference=True)
            img = pydiffvg.RenderFunction.apply(512, 512, 2, 2, 0, None, *scene_args)
            frames.append(img.cpu())
            # Save frame
            pydiffvg.imwrite(img.cpu(), f"sds_steps/frames/_frame_{i:03d}.png")
            print(f"Rendering frame {i+1}/{num_frames}", end='\r')
    save_gif(frames, "sds_steps/optimization_360.gif", fps=10)

class UnitaryBSpline3D(nn.Module):
    def __init__(self, num_kp, kp_init, device, imsize=512):
        super().__init__()
        self.num_kp = num_kp
        self.device = device
        self.imsize = imsize
        
        # Quintic Matrix (Keep as buffer)
        self.register_buffer('M_B', torch.tensor([
            [1/120, 26/120, 66/120, 26/120, 1/120, 0],
            [0, 10/120, 60/120, 40/120, 10/120, 0],
            [0, 0, 40/120, 60/120, 20/120, 0],
            [0, 0, 20/120, 60/120, 40/120, 0],
            [0, 0, 10/120, 40/120, 60/120, 10/120],
            [0, 1/120, 26/120, 66/120, 26/120, 1/120]
        ], dtype=torch.float32, device=device))

        # 1. PARAMETERS (Optimized in 3D Space)
        # kp_init should be [num_kp, 3]
        self.points_3d = nn.Parameter(kp_init.clone().detach().to(device))
        self.unit_widths = nn.Parameter(torch.full((num_kp, 1), 0.35, device=device))
        
        # Jerk Loss Matrix
        self.register_buffer('G', self.get_gram_matrix_jerk(num_kp, device))

    @torch.no_grad()
    def set_data(self, new_points, new_widths):
        """
        Safely updates the geometry and recalculates all dependent matrices (G).
        Use this instead of manually assigning self.points_3d.
        """
        device = self.device
        
        # 1. Sanity Checks
        # Ensure we are working with flat lists of points [N, 3]
        if new_points.dim() == 1: new_points = new_points.unsqueeze(0) # Handle single point case
        if new_widths.dim() == 1: new_widths = new_widths.unsqueeze(1)
        
        if new_points.shape[0] != new_widths.shape[0]:
            raise ValueError(f"Size Mismatch: {new_points.shape[0]} points vs {new_widths.shape[0]} widths")

        # 2. Update Scalar State
        self.num_kp = new_points.shape[0]

        # 3. Update Parameters (Wrap in nn.Parameter)
        # .contiguous() is crucial for memory layout after slicing/cat operations
        self.points_3d = nn.Parameter(new_points.contiguous().to(device))
        self.unit_widths = nn.Parameter(new_widths.contiguous().to(device))

        # 4. Rebuild the Jerk Matrix (G)
        # The matrix size is (N-3)xN, so it MUST be rebuilt when N changes.
        # We use register_buffer to ensure it moves with .to(device) and saves in state_dict
        new_G = self.get_gram_matrix_jerk(self.num_kp, device)
        self.register_buffer('G', new_G)

    def project_points_camera(self, points_3d, camera):
        """
        Projects world points to screen using an arbitrary DiffCamera.
        """
        # 1. Get Full Matrix (MVP)
        mvp = camera.get_full_matrix()  # [4, 4]
        
        # 2. Homogenous Coordinates
        ones = torch.ones((points_3d.shape[0], 1), device=self.device)
        points_hom = torch.cat([points_3d, ones], dim=1)
        
        # 3. Apply Transformation
        # Points are row vectors [N, 4], so multiply by MVP.T
        points_clip = points_hom @ mvp.T
        
        # 4. Orthogonal Projection
        # In Ortho, w is usually 1.0 (or just scale factor). 
        # We take X and Y directly (NDC space [-1, 1])
        ndc_x = points_clip[:, 0]
        ndc_y = points_clip[:, 1]
        
        # 5. Map to Screen [0, imsize]
        screen_x = (ndc_x + 1.0) / 2.0 * self.imsize
        screen_y = (1.0 - ndc_y) / 2.0 * self.imsize # Flip Y
        
        return torch.stack([screen_x, screen_y], dim=1)

    def project_points_orthogonal_legacy(self, points_3d, view_index):
        """Legacy axis-swapping logic for view indices 0, 1, 2."""
        if view_index == 0:   
            x_raw, y_raw = points_3d[:, 1], points_3d[:, 2]
        elif view_index == 1: 
            x_raw, y_raw = -points_3d[:, 0], points_3d[:, 2]
        else:                 
            x_raw, y_raw = -points_3d[:, 0], -points_3d[:, 1]

        screen_x = (x_raw + 1.0) / 2.0 * self.imsize
        screen_y = (1.0 - (y_raw + 1.0) / 2.0) * self.imsize
        return torch.stack([screen_x, screen_y], dim=1)

    def to_bezier(self, keypoints):
        """Vectorized B-Spline to Bezier conversion."""
        k = torch.cat([keypoints[0:1].repeat(2,1), keypoints, keypoints[-1:].repeat(2,1)], dim=0)
        segments = k.unfold(0, 6, 1) 
        bezier_segments = torch.matmul(segments, self.M_B.T)
        
        main_body = bezier_segments[:-1, :, :3]
        last_seg = bezier_segments[-1:, :, :]
        
        return torch.cat([
            main_body.transpose(1, 2).reshape(-1, keypoints.shape[-1]), 
            last_seg.transpose(1, 2).reshape(-1, keypoints.shape[-1])
        ], dim=0)

    
    def forward(self, view_input, view_w, view_h, w_min=0.0, w_max=5.0):
        """
        view_input: Can be an integer (0,1,2) OR a DiffCamera object.
        """
        # 1. Projection Strategy
        if isinstance(view_input, int):
            # Use Legacy fixed views
            proj_2d = self.project_points_orthogonal_legacy(self.points_3d, view_input)
        elif isinstance(view_input, DiffCamera):
            # Use Arbitrary Camera
            proj_2d = self.project_points_camera(self.points_3d, view_input)
            w_min=0.2
        else:
            raise ValueError("view_input must be int or DiffCamera")
        
        # 2. Geometry: Convert B-Spline Keypoints to Bezier Control Points
        bezier_pts_2d = self.to_bezier(proj_2d)
        
        # 3. Width Processing
        bezier_base_widths = self.to_bezier(self.unit_widths).squeeze(-1)
        k = 6.0
        final_widths = w_min + (w_max - w_min) * torch.sigmoid(k * (bezier_base_widths - 0.5))

        return bezier_pts_2d, final_widths

    def get_serialized_scene(self, view_index, view_w, view_h, stroke_color=None):
        """
        Serialized scene using Orthogonal view_index.
        """
        # 1. Generate Geometry
        bezier_pts, final_widths = self.forward(view_index, view_w, view_h)
        
        # 2. Path Setup
        # num_segments for a quintic B-spline is num_kp - 1
        num_segments = self.num_kp - 1
        # Each quintic segment in pydiffvg setup needs internal control points
        # For this specific Bezier conversion, we provide 3 points per segment
        num_control_points = torch.full((num_segments,), 2, dtype=torch.int32, device=self.device)

        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=bezier_pts.contiguous(), 
            stroke_width=final_widths, 
            is_closed=False
        )

        # 3. Style Setup
        s_color = stroke_color if stroke_color is not None else torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0], device=self.device),
            fill_color=None,
            stroke_color=s_color
        )
        
        return pydiffvg.RenderFunction.serialize_scene(view_w, view_h, [path], [path_group])
    
    def get_points(self, num_samples=5000):
        device = self.device
        
        # --- SAFEGUARD 1: Check for degenerate curves ---
        # Padding adds 4 points (2 start, 2 end). Unfold needs 6.
        # Total points must be >= 6. So self.num_kp must be >= 2.
        if self.num_kp < 2:
            # Return a single degenerate point to prevent crash
            return self.points_3d[0:1].repeat(num_samples, 1), self.unit_widths[0:1].repeat(num_samples, 1)

        # 1. Padding
        p = torch.cat([self.points_3d[0:1].repeat(2,1), self.points_3d, self.points_3d[-1:].repeat(2,1)], dim=0)
        w = torch.cat([self.unit_widths[0:1].repeat(2,1), self.unit_widths, self.unit_widths[-1:].repeat(2,1)], dim=0)
        
        # 2. Bezier Conversion
        # x.unfold(0, 6, 1) on shape [N, 3] returns [Segments, 3, 6]
        windows_p = p.unfold(0, 6, 1) 
        windows_w = w.unfold(0, 6, 1)

        # Matmul transforms the last dimension (size 6)
        # [Segments, 3, 6] @ [6, 6] -> [Segments, 3, 6]
        seg_p = torch.matmul(windows_p, self.M_B.T)
        seg_w = torch.matmul(windows_w, self.M_B.T)

        # Transpose to get [Segments, 6, 3] so we can index the segment (dim 0) and point (dim 1)
        seg_p = seg_p.transpose(1, 2)
        seg_w = seg_w.transpose(1, 2)
        
        num_segments = seg_p.shape[0]

        # 3. Sampling
        t = torch.linspace(0, 1, num_samples, device=device)
        t_scaled = t * num_segments # range [0, num_segments]
        
        # --- SAFEGUARD 2: Index Clamping ---
        # Prevent float precision from pushing index to num_segments (out of bounds)
        idx = torch.clamp(t_scaled.long(), 0, num_segments - 1)
        
        u = t_scaled - idx.float() # Local t within segment

        # Bernstein Basis
        u_inv = 1.0 - u
        basis = torch.stack([
            u_inv**5, 
            5 * u * u_inv**4, 
            10 * u**2 * u_inv**3, 
            10 * u**3 * u_inv**2, 
            5 * u**4 * u_inv, 
            u**5
        ], dim=-1) # [num_samples, 6]

        # 4. Final Contraction
        p_targets = seg_p[idx] # [num_samples, 6, 3]
        w_targets = seg_w[idx] # [num_samples, 6, 1]
        
        # Broadcasted Multiply and Sum
        # [N, 6, 1] * [N, 6, 3] -> Sum over dim 1 -> [N, 3]
        points_3d = (basis.unsqueeze(-1) * p_targets).sum(dim=1)
        widths_3d = (basis.unsqueeze(-1) * w_targets).sum(dim=1)
        
        return points_3d, widths_3d
    
    def get_points_natural(self, num_samples=5000):
        device = self.device
        
        # 1. Padding & Bezier Conversion (Same as your logic)
        p = torch.cat([self.points_3d[0:1].repeat(2,1), self.points_3d, self.points_3d[-1:].repeat(2,1)], dim=0)
        w = torch.cat([self.unit_widths[0:1].repeat(2,1), self.unit_widths, self.unit_widths[-1:].repeat(2,1)], dim=0)
        
        windows_p = p.unfold(0, 6, 1) 
        windows_w = w.unfold(0, 6, 1)
        seg_p = torch.matmul(windows_p, self.M_B.T).transpose(1, 2)
        seg_w = torch.matmul(windows_w, self.M_B.T).transpose(1, 2)
        num_segments = seg_p.shape[0]

        # 3. Create Reference for Density Calculation
        ref_samples = 2000 
        t_ref = torch.linspace(0, 1, ref_samples, device=device)
        t_ref_scaled = t_ref * (num_segments - 1e-5)
        idx_ref = t_ref_scaled.long()
        u_ref = t_ref_scaled - idx_ref.float()
        
        u_inv_ref = 1.0 - u_ref
        basis_ref = torch.stack([
            u_inv_ref**5, 5 * u_ref * u_inv_ref**4, 10 * u_ref**2 * u_inv_ref**3, 
            10 * u_ref**3 * u_inv_ref**2, 5 * u_ref**4 * u_inv_ref, u_ref**5
        ], dim=-1)

        # Use idx_ref (FIXED HERE)
        p_targets_ref = seg_p[idx_ref] 
        points_ref = (basis_ref.unsqueeze(-1) * p_targets_ref).sum(dim=1)
        
        # A. Calculate Physical Distance (1999 segments)
        dists = torch.norm(torch.diff(points_ref, dim=0), dim=1)

        # B. Calculate Curvature (1998 angles)
        v1 = points_ref[1:-1] - points_ref[:-2]
        v2 = points_ref[2:] - points_ref[1:-1]
        cos_theta = (v1 * v2).sum(dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-8)
        curvature = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

        # C. CORRECTED PADDING
        # curvature has 1998 elements. dists has 1999.
        # We pad curvature by repeating the last value once to reach 1999.
        curvature_padded = torch.cat([curvature, curvature[-1:]], dim=0)

        # Now both are size 1999. This will work!
        human_density = dists * (1.0 + curvature_padded * 10.0)
        
        cum_density = torch.cat([torch.zeros(1, device=device), torch.cumsum(human_density, dim=0)])
        cum_density /= (cum_density[-1] + 1e-8)

        # E. Remap Time
        t_linear = torch.linspace(0, 1, num_samples, device=device)
        t_linear = 3 * t_linear**2 - 2 * t_linear**3 

        # Add .detach() to all three tensors to safely convert to numpy
        t_warped = torch.from_numpy(np.interp(
            t_linear.detach().cpu().numpy(), 
            cum_density.detach().cpu().numpy(), 
            t_ref.detach().cpu().numpy()
        )).to(device).float()

        # Final Pass
        t_scaled = t_warped * (num_segments - 1e-5)
        idx = t_scaled.long()
        u = t_scaled - idx.float()
        u_inv = 1.0 - u
        basis = torch.stack([
            u_inv**5, 5 * u * u_inv**4, 10 * u**2 * u_inv**3, 
            10 * u**3 * u_inv**2, 5 * u**4 * u_inv, u_inv**0 * u**5
        ], dim=-1)

        points_3d = (basis.unsqueeze(-1) * seg_p[idx]).sum(dim=1)
        widths_3d = (basis.unsqueeze(-1) * seg_w[idx]).sum(dim=1)
        
        return points_3d, widths_3d
    

    @torch.no_grad()
    def insert_keypoints(self, indices, thinning_factor=2.5):
        """Stable 3D insertion."""
        new_pts = (self.points_3d[indices] + self.points_3d[indices + 1]) / 2.0
        avg_widths = (self.unit_widths[indices] + self.unit_widths[indices + 1]) / 2.0
        new_widths = avg_widths * thinning_factor
        
        # Build lists (Standard B-Spline midpoint logic)
        updated_points = []
        updated_widths = []
        for i in range(self.num_kp):
            updated_points.append(self.points_3d[i:i+1])
            updated_widths.append(self.unit_widths[i:i+1])
            if i in indices:
                updated_points.append(new_pts[indices == i])
                updated_widths.append(new_widths[indices == i])
                
        self.num_kp = len(updated_points)
        self.points_3d = nn.Parameter(torch.cat(updated_points, dim=0))
        self.unit_widths = nn.Parameter(torch.cat(updated_widths, dim=0))
        self.register_buffer('G', self.get_gram_matrix_jerk(self.num_kp, self.device))

    def get_jerk_loss(self):
        """Smoothness penalty in 3D world space."""
        # Penalizes high-frequency wiggles in X, Y, and Z
        s_pos = torch.trace(self.points_3d.T @ self.G @ self.points_3d)
        s_width = torch.trace(self.unit_widths.T @ self.G @ self.unit_widths)
        return s_pos, s_width

    @staticmethod
    def get_gram_matrix_jerk(num_pts, device):
        D = torch.zeros((num_pts - 3, num_pts), device=device)
        for i in range(num_pts - 3):
            D[i, i:i+4] = torch.tensor([-1.0, 3.0, -3.0, 1.0], device=device)
        return D.T @ D
    
    '''
    Function for visualization
    '''
    def get_serialized_scene_gradient(self, view_index, view_w, view_h, num_samples=3000, w_min=0.2, w_max=5.0, cmap_name='magma'):
        """
        Creates a gradient visualization with CORRECT DEPTH ORDERING and ARC-LENGTH COLORING.
        """
        device = self.device
        
        # 1. DENSE SAMPLING
        points_3d, raw_widths = self.get_points(num_samples=num_samples)

        # 2. ARC LENGTH CALCULATION (3D)
        diffs_3d = points_3d[1:] - points_3d[:-1]
        segment_lengths = torch.norm(diffs_3d, dim=1)
        
        # Cumulative sum
        cumulative_dist = torch.cumsum(torch.cat([torch.tensor([0.0], device=device), segment_lengths]), dim=0)
        total_length = cumulative_dist[-1] + 1e-8
        t_geometric = cumulative_dist / total_length 

        # Calculate midpoint t for every segment
        t_mid = (t_geometric[:-1] + t_geometric[1:]) / 2.0
        
        # --- FIX: USE t_mid TO GENERATE COLORS ---
        # OLD: colors = get_premium_colors(num_samples - 1, ...) -> IGNORES GEOMETRY
        # NEW: Pass the geometric t values we just calculated
        colors = get_premium_colors_from_t(t_mid, cmap_name=cmap_name, device=device)
        # -----------------------------------------

        # 3. WIDTH TRANSFORM
        k = 6.0
        final_widths = w_min + (w_max - w_min) * torch.sigmoid(k * (raw_widths - 0.5))
        
        # 4. PROJECTION
        if isinstance(view_index, int):
            # Legacy Fixed Views
            if view_index == 0:   z_vals = points_3d[:, 2] 
            elif view_index == 1: z_vals = points_3d[:, 1]
            else:                 z_vals = points_3d[:, 2]
            proj_2d = self.project_points_orthogonal_legacy(points_3d, view_index)
        else:
            # Arbitrary Camera
            mvp = view_index.get_full_matrix()
            ones = torch.ones((points_3d.shape[0], 1), device=device)
            points_hom = torch.cat([points_3d, ones], dim=1)
            points_clip = points_hom @ mvp.T 
            z_vals = points_clip[:, 3] 
            
            ndc_x = points_clip[:, 0] / points_clip[:, 3]
            ndc_y = points_clip[:, 1] / points_clip[:, 3]
            screen_x = (ndc_x + 1.0) / 2.0 * self.imsize
            screen_y = (1.0 - ndc_y) / 2.0 * self.imsize
            proj_2d = torch.stack([screen_x, screen_y], dim=1)

        # 5. DEPTH SORTING
        seg_z = (z_vals[:-1] + z_vals[1:]) / 2.0
        sorted_indices = torch.argsort(seg_z, descending=False) 

        # 6. BUILD SHAPES
        shapes = []
        shape_groups = []
        
        for i in range(num_samples - 1):
            idx = sorted_indices[i]
            
            avg_width = (final_widths[idx] + final_widths[idx+1]) / 2.0
            
            path = pydiffvg.Path(
                num_control_points=torch.tensor([0], dtype=torch.int32, device=device),
                points=proj_2d[idx:idx+2].contiguous(),
                stroke_width=avg_width, 
                is_closed=False
            )
            shapes.append(path)
            
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([i], device=device),
                fill_color=None,
                # colors[idx] correctly grabs the intrinsic color of this segment
                stroke_color=colors[idx] 
            )
            shape_groups.append(path_group)
            
        return pydiffvg.RenderFunction.serialize_scene(view_w, view_h, shapes, shape_groups)

def initialize_sphere_tsp(num_kp, radius=0.6, device='cuda'):
    """
    Generates a uniform distribution of points inside a sphere 
    and orders them via Greedy TSP to form a continuous B-spline path.
    """
    # 1. Uniform Spherical Volume Sampling
    # Using the cube root of rand ensures uniform density throughout the volume
    num_samples = num_kp
    phi = torch.acos(1 - 2 * torch.rand(num_samples, device=device))
    theta = 2 * np.pi * torch.rand(num_samples, device=device)
    r = radius * (torch.rand(num_samples, device=device)**(1/3)) 
    
    pts = torch.stack([
        r * torch.sin(phi) * torch.cos(theta),
        r * torch.sin(phi) * torch.sin(theta),
        r * torch.cos(phi)
    ], dim=-1).cpu().numpy()

    # 2. Greedy TSP Ordering
    unvisited = list(range(len(pts)))
    curr = unvisited.pop(0)
    tour = [curr]
    
    while unvisited:
        # Find the point closest to the current point
        diff = pts[unvisited] - pts[curr]
        dists = np.sum(diff**2, axis=1) # L2 squared for speed
        nearest_idx = np.argmin(dists)
        
        curr = unvisited.pop(nearest_idx)
        tour.append(curr)
        
    # 3. Convert back to Tensor
    ordered_pts = torch.tensor(pts[tour], device=device, dtype=torch.float32)
    
    return ordered_pts.requires_grad_(True)

class StableDiffusion(nn.Module):
    def __init__(self, device, model_id=None):
        super().__init__()
        self.device = device
        
        # The public SD 1.5 mirror is the default for fresh installs. Set
        # FOURDWIRE_MODEL_ID to reproduce runs using a cached legacy model.
        model_id = model_id or os.environ.get(
            "FOURDWIRE_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

    # ... get_text_embeds remains the same ...

    def get_sds_loss(self, latents, text_embeddings, guidance_scale=100, ratio=1.0):
        """
        latents: [Batch, 4, 64, 64]
        text_embeddings: [2*Batch, 77, 768]
        ratio: current_iteration / total_iterations (0.0 to 1.0)
        """
        latents = latents.to(torch.float16)
        batch_size = latents.shape[0]

        # --- NOISE ANNEALING LOGIC ---
        # Official DreamWire/VectorFusion strategy: 
        # Start with range [0.02, 0.98], end with range [0.02, 0.50]
        # High ratio (end of training) = lower max_step (less noise)
        min_step_percent = 0.02
        max_step_percent = max(0.50, 0.98 - 0.48 * ratio) 
        
        min_step = int(self.num_train_timesteps * min_step_percent)
        max_step = int(self.num_train_timesteps * max_step_percent)

        # 1. Sample Random Timesteps within the annealed range
        t = torch.randint(
            min_step, 
            max_step, 
            (batch_size,), 
            dtype=torch.long, 
            device=self.device
        )

        # 2. Add Noise
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # 3. Predict Noise (UNet)
        with torch.no_grad():
            latent_model_input = torch.cat([latents_noisy] * 2)
            t_input = torch.cat([t] * 2)

            with torch.cuda.amp.autocast():
                noise_pred = self.unet(
                    latent_model_input, 
                    t_input, 
                    encoder_hidden_states=text_embeddings
                ).sample

        # 4. CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 5. Official SDS Gradient Weighting
        # w(t) = (1 - alpha_t)
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        # 6. Backprop
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target.float(), reduction="sum")
        
        return loss
    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts=None):
        """
        Input: list of strings, e.g., ["Newton", "Einstein", "Turing"]
        Output: [2 * Batch, 77, 768] (Stacked Negative + Positive)
        """
        batch_size = len(prompts)
        
        # Handle negative prompts
        if negative_prompts is None:
            negative_prompts = [""] * batch_size
        elif isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * batch_size

        # 1. Tokenize
        # Positive
        pos_input = self.tokenizer(
            prompts, 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors='pt'
        )
        # Negative
        neg_input = self.tokenizer(
            negative_prompts, 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors='pt'
        )

        # 2. Encode
        with torch.cuda.amp.autocast():
            pos_embeds = self.text_encoder(pos_input.input_ids.to(self.device))[0]
            neg_embeds = self.text_encoder(neg_input.input_ids.to(self.device))[0]
        
        # 3. Cat for CFG: [Neg_1, Neg_2, Neg_3, Pos_1, Pos_2, Pos_3]
        # Note: Standard SD pipeline usually interleaves them or stacks Neg then Pos.
        # We stack [Neg, Pos] along dim 0.
        return torch.cat([neg_embeds, pos_embeds])
    
    def generate_sketch(self, prompt):
        """
        Generates a white-background line drawing.
        """
        # Engineer the prompt to guarantee clean lines
        full_prompt = (
            f"minimalist continuous line drawing of {prompt}, "
            "black ink on white paper, in the middle of the white paper, no shading, high contrast, vector style"
        )
        
        negative_prompt = (
            "shading, colors, complex, realistic, photo, texture, "
            "filled, gradient, grey, messy, multiple lines"
        )

        with torch.autocast(self.device):
            image = self.pipe(
                full_prompt, 
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=768, 
                width=768
            ).images[0]
            
        return image

    def image_to_points(self, pil_image, num_points=1000):
        """
        Converts the generated sketch into a 3D point cloud (N, 3).
        """
        # 1. Convert to OpenCV format (Grayscale)
        img_np = np.array(pil_image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 2. Extract Edges (Canny)
        # Invert so lines are white (255) and bg is black (0) for extraction
        edges = cv2.Canny(gray, 100, 200)
        
        # 3. Get Coordinates of the lines
        y_idxs, x_idxs = np.where(edges > 0)
        
        if len(x_idxs) == 0:
            print("Warning: No edges found! Returning random sphere.")
            return torch.randn(num_points, 3)

        # 4. Normalize to [-1, 1]
        h, w = edges.shape
        x_norm = (x_idxs / w) * 2 - 1
        y_norm = (y_idxs / h) * 2 - 1
        y_norm = -y_norm  # Flip Y (Image coords are top-down, 3D is bottom-up)

        # 5. Sampling & Sorting
        # We have typically >5000 pixels, we only need num_points (e.g., 200)
        
        # Simple Strategy: Sort by X to minimize jumping lines
        # (A TSP solver would be better, but this is fast and works 80% of the time)
        raw_points = np.stack([x_norm, y_norm], axis=1)
        
        # Sort by X coordinate
        sort_idx = np.argsort(raw_points[:, 0])
        raw_points = raw_points[sort_idx]
        
        # Subsample evenly to get exactly num_points
        indices = np.linspace(0, len(raw_points)-1, num_points).astype(int)
        sampled_2d = raw_points[indices]
        
        # 6. Add Z-dimension (Depth)
        # Initialize flat (z=0) or with slight noise to let optimizer decide depth
        z_coords = np.random.uniform(-0.05, 0.05, (num_points, 1))
        
        points_3d = np.hstack([sampled_2d, z_coords])
        
        return torch.tensor(points_3d, dtype=torch.float32, device=self.device)

class DreamWire(nn.Module):
    def __init__(self, num_paths, device, imsize=512, control_points_per_seg=3):
        super().__init__()
        self.device = device
        self.num_paths = num_paths
        self.num_segments = 4
        self.imsize = imsize
        
        # 1 + 3*segments = 16 points for a cubic Bezier chain
        self.num_control_points_per_path = 1 + 3 * self.num_segments
        self.control_points_per_seg = control_points_per_seg
        # --- Random Walk Initialization ---
        self.init_all_paths()
        
        self.unit_widths = nn.Parameter(torch.ones(self.num_paths, device=device) * 3.0)
        self.stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)

    def init_all_paths(self):
        """Initializes 3D points using the official local random walk logic."""
        all_paths = []
        for i in range(self.num_paths):
            points = []
            # 1. Pick initial anchor p0
            p0 = torch.rand(3, device=self.device) * 0.6 + 0.2
            points.append(p0)

            curr_p = p0
            radius = 0.05
            
            # --- FIX STARTS HERE ---
            for j in range(self.num_segments):
                # For Cubic Bezier, we need to add 3 points per segment:
                # (Internal_1, Internal_2, Endpoint)
                for k in range(3): # <--- HARDCODE THIS TO 3 FOR CUBIC
                    next_p = curr_p + radius * (torch.rand(3, device=self.device) - 0.5)
                    points.append(next_p)
                    curr_p = next_p
            # --- FIX ENDS HERE ---
            
            # Stack and convert
            # Result: 1 + (4 * 3) = 13 Points. Correct.
            path_tensor = torch.stack(points) * 2.0 - 1.0
            all_paths.append(path_tensor)
        
        if hasattr(self, "_3D_points"):
            self._3D_points.data = torch.stack(all_paths)
        else:
            self._3D_points = nn.Parameter(torch.stack(all_paths))
    # --- OFFICIAL MST LOGIC (Consolidated from painter_params.py) ---

    def min_key(self, key, visited):
        min_val = float('inf')
        min_idx = -1
        for i in range(len(key)):
            if key[i] < min_val and not visited[i]:
                min_val = key[i]
                min_idx = i
        return min_idx

    def prim_algorithm(self, dist_matrix):
        n = len(dist_matrix)
        visited = [False] * n
        parent = [-1] * n
        key = [float('inf')] * n
        key[0] = 0
        for _ in range(n):
            u = self.min_key(key, visited)
            if u == -1: break
            visited[u] = True
            for v in range(n):
                if dist_matrix[u][v] > 0 and not visited[v] and dist_matrix[u][v] < key[v]:
                    parent[v] = u
                    key[v] = dist_matrix[u][v]
        return parent

    def get_prim_loss(self):
        """
        Authentic MST loss calculation based on endpoints.
        Includes 1e-6 epsilon to prevent gradient explosion (AssertionError).
        """
        # Get start (0) and end (-1) points for each path: [num_paths, 2, 3]
        pts = torch.stack([self._3D_points[:, 0, :], self._3D_points[:, -1, :]], dim=1)
        num = pts.shape[0]
        
        # Vectorized endpoint distance matrix for speed and stability
        # We need the min distance between any of the 4 endpoint combinations for path i and j
        dist_matrix = torch.zeros(num, num, device=self.device)
        
        for i in range(num):
            # start-start, start-end, end-start, end-end
            d00 = torch.norm(pts[i, 0] - pts[:, 0], p=2, dim=-1)
            d01 = torch.norm(pts[i, 0] - pts[:, 1], p=2, dim=-1)
            d10 = torch.norm(pts[i, 1] - pts[:, 0], p=2, dim=-1)
            d11 = torch.norm(pts[i, 1] - pts[:, 1], p=2, dim=-1)
            
            # min distance between path i and all other paths
            mins = torch.min(torch.stack([d00, d01, d10, d11]), dim=0).values
            dist_matrix[i] = mins + 1e-6 # Stability epsilon

        # Compute MST on the CPU (standard for Prim's)
        tree = self.prim_algorithm(dist_matrix.detach().cpu().numpy())
        
        loss_mst = 0
        for i, parent_node in enumerate(tree):
            if parent_node != -1:
                loss_mst += dist_matrix[i, parent_node]
        return loss_mst

    # --- RENDERING ENGINE ---
    @torch.no_grad()
    def reinitialize_paths(self, threshold=0.05):
        """Re-initializes specific paths using the same official logic if they collapse."""
        diffs = self._3D_points[:, 1:, :] - self._3D_points[:, :-1, :]
        lengths = torch.norm(diffs, dim=-1).sum(dim=-1)
        to_reinit = (lengths < threshold).nonzero(as_tuple=True)[0]
        
        radius = 0.05
        for idx in to_reinit:
            points = []
            p0 = torch.rand(3, device=self.device) * 0.6 + 0.2
            points.append(p0)
            curr_p = p0
            for j in range(self.num_segments):
                for k in range(self.control_points_per_seg - 1):
                    next_p = curr_p + radius * (torch.rand(3, device=self.device) - 0.5)
                    points.append(next_p)
                    curr_p = next_p
            self._3D_points.data[idx] = torch.stack(points) * 2.0 - 1.0

    def get_serialized_scene(self, view_index, view_w, view_h):
        # 1. Extract World Coordinates [NumPaths, NumPoints]
        # We access dimensions carefully: [Batch, Points, XYZ]
        w_x = self._3D_points[:, :, 0]
        w_y = self._3D_points[:, :, 1]
        w_z = self._3D_points[:, :, 2]

        # 2. Apply Orthogonal Logic (Matching your BSpline class)
        if view_index == 0:   
            # Front (+X vantage): Screen X = Y, Screen Y = Z
            x_raw = w_y
            y_raw = w_z
            
        elif view_index == 1: 
            # Side (+Y vantage): Screen X = -X, Screen Y = Z
            x_raw = -w_x
            y_raw = w_z
            
        else:                 
            # Top (+Z vantage): Screen X = -X, Screen Y = -Y
            x_raw = -w_x
            y_raw = -w_y

        # 3. Master Mapping to Screen Space
        # X: -1 (Left) -> 0, +1 (Right) -> imsize
        screen_x = (x_raw + 1.0) / 2.0 * self.imsize
        
        # Y: +1 (Top) -> 0, -1 (Bottom) -> imsize (Standard Screen Flip)
        screen_y = (1.0 - (y_raw + 1.0) / 2.0) * self.imsize

        # Combine: [NumPaths, NumPoints, 2]
        points_2d = torch.stack([screen_x, screen_y], dim=2)

        # 4. Construct pydiffvg Scene
        shapes = []
        shape_groups = []
        
        # Cubic Bezier Setup (2 internal points per segment)
        num_ctrl_pts_tensor = torch.full((self.num_segments,), 2, dtype=torch.int32, device=self.device)
        
        for i in range(self.num_paths):
            pts = points_2d[i]
            
            # OPTIONAL SAFETY: Skip collapsed paths to avoid 'Length = Inf' crash
            if (pts.max(dim=0).values - pts.min(dim=0).values).sum() < 0.1:
                continue

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts_tensor,
                points=pts.contiguous(),
                stroke_width=self.unit_widths[i],
                is_closed=False
            )
            shapes.append(path)
            
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes)-1], device=self.device),
                fill_color=None,
                stroke_color=self.stroke_color
            )
            shape_groups.append(path_group)
            
        return pydiffvg.RenderFunction.serialize_scene(view_w, view_h, shapes, shape_groups)

    def points_restrict(self):
        """Clamps 3D points within the canonical volume."""
        with torch.no_grad():
            self._3D_points.data.clamp_(-0.9, 0.9)

@torch.no_grad()
def simplify_invisible_segments(model, width_threshold=0.1, min_run_length=5):
    """
    Replaces long sequences of invisible points with a single connection.
    
    Args:
        min_run_length (int): Only collapse invisible segments if they contain 
                              this many consecutive points or more. 
                              (e.g., 5 means keep runs of 4, simplify runs of 5+)
    """
    print(f"Simplifying invisible segments (Threshold: {width_threshold}, Min Run: {min_run_length})...")
    
    # 1. Get Data
    points = model.points_3d.data
    widths = model.unit_widths.data.squeeze()
    device = points.device
    N = len(points)
    
    # 2. Identify Invisible Points
    is_invisible = widths < width_threshold
    
    # 3. Iterate and Build New Indices
    keep_indices = []
    i = 0
    
    while i < N:
        if not is_invisible[i]:
            # CASE 1: Visible Point -> Always keep
            keep_indices.append(i)
            i += 1
        else:
            # CASE 2: Invisible Point -> Start of a possible run
            start_run = i
            
            # Find the end of this invisible run
            while i < N and is_invisible[i]:
                i += 1
            
            # 'i' is now the first VISIBLE point (or N)
            # 'end_run' is the last INVISIBLE point
            end_run = i - 1 
            
            current_run_len = end_run - start_run + 1
            
            # DECISION: Check against your threshold
            if current_run_len < min_run_length:
                # Run is too short (e.g., length 3 < threshold 5)
                # Keep ALL points in this run
                for k in range(start_run, end_run + 1):
                    keep_indices.append(k)
            else:
                # Run is long enough (e.g., length 10 >= threshold 5)
                # SIMPLIFY: Delete the middle, keep endpoints
                
                # A. Keep the Start (Connects to previous visible)
                keep_indices.append(start_run)
                
                # B. Keep the End (Connects to next visible)
                # (Check ensures we don't duplicate if run_len=1, though threshold usually >1)
                if end_run > start_run:
                    keep_indices.append(end_run)
                
                # We essentially skipped indices [start_run+1 ... end_run-1]

    # 4. Commit Changes
    indices_tensor = torch.tensor(keep_indices, device=device, dtype=torch.long)
    
    new_points = points[indices_tensor]
    new_widths = widths[indices_tensor]
    
    # Safety Check
    if len(new_points) < 4:
        print("Warning: Simplification would result in too few points. Aborting.")
        return model

    print(f"Simplification Complete: {N} -> {len(new_points)} points")
    print(f"Removed {N - len(new_points)} points from long invisible chains.")

    # Update Model
    model.num_kp = len(new_points)
    model.points_3d = torch.nn.Parameter(new_points)
    # Ensure width shape is correct (N, 1) or (N) depending on your model
    if model.unit_widths.dim() > 1:
         model.unit_widths = torch.nn.Parameter(new_widths.unsqueeze(1))
    else:
         model.unit_widths = torch.nn.Parameter(new_widths)
    
    return model

def render_dreamwire_illusion_tour(dreamwire_model, filepath):
    """
    Renders the illusion tour for the DreamWire (Multi-Path) model.
    Includes a SAFETY CHECK to prevent 'Length = inf' crashes.
    """
    imsize = dreamwire_model.imsize
    device = dreamwire_model.device
    frames = []

    # --- 1. BUILD THE TIMELINE ---
    t_vals = []
    frames_move = 40
    frames_pause = 15
    
    # Phase 1: Front (0.0) -> Side (0.33)
    t_vals.extend([0.0] * frames_pause) 
    t_vals.extend(np.linspace(0.0, 0.33, frames_move).tolist())
    
    # Phase 2: Side (0.33) -> Top (0.66)
    t_vals.extend([0.33] * frames_pause)
    t_vals.extend(np.linspace(0.33, 0.66, frames_move).tolist())
    t_vals.extend([0.66] * frames_pause)

    total_frames = len(t_vals)
    
    # Output Directory Check
    if not os.path.exists("outputs/frames"):
        os.makedirs("outputs/frames")
        
    # Pre-fetch width and color to avoid repeated lookup
    current_widths = dreamwire_model.unit_widths
    stroke_color = dreamwire_model.stroke_color

    with torch.no_grad():
        for i, t in enumerate(t_vals):
            # --- ROTATION LOGIC ---
            if t <= 0.33:
                local_t = t / 0.33
                angle = (math.pi / 2) * local_t
                R = torch.tensor([
                    [ math.cos(angle), math.sin(angle), 0],
                    [-math.sin(angle), math.cos(angle), 0],
                    [ 0,               0,               1]
                ], device=device).float()
            elif t <= 0.66:
                local_t = (t - 0.33) / 0.33
                angle = (math.pi / 2) * local_t
                R_side = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], device=device).float()
                R_tilt = torch.tensor([
                    [math.cos(angle), 0, math.sin(angle)],
                    [0,               1, 0],
                    [-math.sin(angle), 0, math.cos(angle)]
                ], device=device).float()
                R = R_tilt @ R_side
            else:
                 # Default to last view if t goes over
                 R = R_tilt @ R_side

            # --- MAPPING ---
            points_3d = dreamwire_model._3D_points # [NumPaths, NumPoints, 3]
            rotated_points = points_3d @ R.T
            
            x_raw = rotated_points[:, :, 1]
            y_raw = rotated_points[:, :, 2]
            
            screen_x = (x_raw + 1.0) / 2.0 * imsize
            screen_y = (1.0 - (y_raw + 1.0) / 2.0) * imsize
            
            points_2d = torch.stack([screen_x, screen_y], dim=2)

            # --- SCENE CONSTRUCTION ---
            shapes = []
            shape_groups = []
            
            num_segments = dreamwire_model.num_segments
            num_ctrl_pts_tensor = torch.full((num_segments,), 2, dtype=torch.int32, device=device)
            
            for p_idx in range(dreamwire_model.num_paths):
                # ----------------------------------------------------
                # SAFETY CHECK: Is this path a single point?
                # ----------------------------------------------------
                pts = points_2d[p_idx] # [NumPoints, 2]
                x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
                y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
                
                is_off_screen = (x_max < 0 or x_min > imsize or 
                                 y_max < 0 or y_min > imsize)
                
                # Check for degenerate "dots"
                span = (x_max - x_min) + (y_max - y_min)
                
                if is_off_screen or span < 0.1:
                    continue
                # Calculate bounding box or length
                # Fast check: Max - Min < epsilon?
                span = pts.max(dim=0).values - pts.min(dim=0).values
                if span.sum() < 0.1: # Less than 0.1 pixel span = Invisible Dot
                    continue # SKIP THIS PATH

                path = pydiffvg.Path(
                    num_control_points=num_ctrl_pts_tensor,
                    points=pts.contiguous(),
                    stroke_width=current_widths[p_idx],
                    is_closed=False
                )
                shapes.append(path)
                
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes)-1], device=device),
                    fill_color=None,
                    stroke_color=stroke_color
                )
                shape_groups.append(path_group)

            # --- RENDER ---
            # If ALL paths were skipped, create a dummy transparent pixel to prevent crash
            if len(shapes) == 0:
                 dummy_path = pydiffvg.Path(
                    num_control_points=torch.tensor([0], dtype=torch.int32, device=device),
                    points=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device),
                    stroke_width=torch.tensor(0.0, device=device),
                    is_closed=False
                 )
                 shapes.append(dummy_path)
                 shape_groups.append(pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([0], device=device),
                    fill_color=None,
                    stroke_color=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
                 ))

            scene_args = pydiffvg.RenderFunction.serialize_scene(imsize, imsize, shapes, shape_groups)
            img = pydiffvg.RenderFunction.apply(imsize, imsize, 2, 2, 0, None, *scene_args)
            
            # Save
            filename = f"outputs/frames_dreamwire/_frame_{i:03d}.png"
            pydiffvg.imwrite(img.cpu(), filename)
            frames.append(img.cpu())
            print(f"Rendering frame {i+1}/{total_frames} (t={t:.2f})", end='\r')

    print(f"\nSaving GIF to outputs/{filepath}.gif...")
    save_gif(frames, f"outputs/{filepath}.gif", fps=15)

def render_orthogonal_illusion_tour(bspline, filepath, show_thin_line=False):
    """
    Renders the illusion tour with:
    1. Front -> Side (Rotation around Z)
    2. Side -> Top (Tilt around Screen-X / Data-Index-1)
    3. Top -> Front (Return)
    Includes pauses at each key view.
    """
    imsize = bspline.imsize
    device = bspline.device
    frames = []

    # --- 1. BUILD THE TIMELINE ---
    # We create a specific list of 't' values to handle smooth motion + freezes
    t_vals = []
    frames_move = 40
    frames_pause = 15
    
    # Phase 1: Front (0.0) -> Side (0.33)
    t_vals.extend([0.0] * frames_pause) # FREEZE SIDE
    t_vals.extend(np.linspace(0.0, 0.33, frames_move).tolist())
    
    # Phase 2: Side (0.33) -> Top (0.66)
    t_vals.extend([0.33] * frames_pause) # FREEZE TOP
    t_vals.extend(np.linspace(0.33, 0.66, frames_move).tolist())
    t_vals.extend([0.66] * frames_pause) # FREEZE TOP


    total_frames = len(t_vals)

    with torch.no_grad():
        for i, t in enumerate(t_vals):
            
            # --- ROTATION LOGIC ---
            
            # PHASE 1: Front -> Side
            # Rotate +90 deg around Z to move World-Y (Index 1) to Screen-X
            if t <= 0.33:
                local_t = t / 0.33
                angle = (math.pi / 2) * local_t
                
                # Standard Z-Rotation that results in:
                # Row 1 (Index 1) receiving -X (which matches Side View training)
                R = torch.tensor([
                    [ math.cos(angle), math.sin(angle), 0],
                    [-math.sin(angle), math.cos(angle), 0],
                    [ 0,               0,               1]
                ], device=device).float()
                
            # PHASE 2: Side -> Top
            # Tilt 90 deg "forward" around the Horizontal Axis (Index 1)
            elif t <= 0.66:
                local_t = (t - 0.33) / 0.33
                angle = (math.pi / 2) * local_t
                
                # 1. Start at Side View (The result of Phase 1)
                R_side = torch.tensor([
                    [0, 1, 0], 
                    [-1, 0, 0], 
                    [0, 0, 1]
                ], device=device).float()
                
                # 2. Tilt Matrix (Around Index 1)
                # This matrix keeps the middle row (Index 1) fixed.
                # It swaps Index 0 (-Y) and Index 2 (Z).
                R_tilt = torch.tensor([
                    [math.cos(angle), 0, math.sin(angle)],
                    [0,               1, 0],
                    [-math.sin(angle), 0, math.cos(angle)]
                ], device=device).float()
                
                R = R_tilt @ R_side


            # --- MAPPING ---
            rotated_points = bspline.points_3d @ R.T
            
            # Extract Horizontal (Index 1) and Vertical (Index 2)
            x_raw = rotated_points[:, 1]
            y_raw = rotated_points[:, 2]
            
            # Map to Screen Space
            screen_x = (x_raw + 1.0) / 2.0 * imsize *0.95
            screen_y = (1.0 - (y_raw + 1.0) / 2.0) * imsize *0.95
            
            points_2d = torch.stack([screen_x, screen_y], dim=1)

            # --- RENDER ---
            scene_args = get_serialized_scene_from_points(bspline, points_2d, show_thin_line=show_thin_line)
            img = pydiffvg.RenderFunction.apply(imsize, imsize, 2, 2, 0, None, *scene_args)
            pydiffvg.imwrite(img.cpu(), f"outputs/frames/{show_thin_line}/_frame_{i:03d}.png")
            frames.append(img.cpu())
            print(f"Rendering frame {i+1}/{total_frames} (t={t:.2f})", end='\r')

    save_gif(frames, f"outputs/{filepath}.gif", fps=15)

def render_orthogonal_360(bspline, num_frames=60):
    frames = []
    with torch.no_grad():
        for i in range(num_frames):
            # Calculate rotation angle
            theta = (2 * math.pi * i) / num_frames
            
            # Create a manual rotation matrix (Rotating around Y-axis)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            R = torch.tensor([
                [cos_t,  0, sin_t],
                [0,      1, 0],
                [-sin_t, 0, cos_t]
            ], device=bspline.device)

            # 1. Rotate the 3D points manually
            # points_3d shape is [N, 3] -> multiply by R
            rotated_points = bspline.points_3d @ R.T
            
            # 2. Project using your Axis Swapping (Front View slice)
            # This ensures we always see the "flat" version even as it turns
            pts_norm = (rotated_points + 1.0) / 2.0 * bspline.imsize
            points_2d = pts_norm[:, [1, 2]] # Y, Z slice
            
            # 3. Render
            scene_args = get_serialized_scene_from_points(bspline, points_2d)
            img = pydiffvg.RenderFunction.apply(512, 512, 2, 2, 0, None, *scene_args)
            frames.append(img.cpu())
            
    save_gif(frames, "outputs/correct_orthogonal_rotation.gif")

def get_serialized_scene_from_points(bspline, points_2d, stroke_color=None, show_thin_line=False):
    """
    Takes 2D projected points (already in screen space [0, imsize])
    and converts them to a serialized scene for pydiffvg.
    """
    # 1. Convert B-Spline Keypoints to Bezier Control Points
    # This is crucial: pydiffvg renders Beziers, not raw B-splines.
    bezier_pts = bspline.to_bezier(points_2d)
    
    # 2. Process Widths
    # We use the optimized unit_widths, processed through the same Bezier logic
    bezier_base_widths = bspline.to_bezier(bspline.unit_widths).squeeze(-1)
    
    # Apply the same Sigmoid mapping used in training for consistency
    k = 6.0
    w_min, w_max = 0.0, 5.0
    if show_thin_line:
        w_min = 0.15
    final_widths = w_min + (w_max - w_min) * torch.sigmoid(k * (bezier_base_widths - 0.5))

    # 3. Path Setup
    num_segments = bspline.num_kp - 1
    # For quintic B-splines, pydiffvg usually expects 2 internal control points per segment
    num_control_points = torch.full((num_segments,), 2, dtype=torch.int32, device=bspline.device)

    path = pydiffvg.Path(
        num_control_points=num_control_points,
        points=bezier_pts.contiguous(), 
        stroke_width=final_widths, 
        is_closed=False
    )

    # 4. Color/Style Setup
    s_color = stroke_color if stroke_color is not None else torch.tensor([0.0, 0.0, 0.0, 1.0], device=bspline.device)
    
    path_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0], device=bspline.device),
        fill_color=None,
        stroke_color=s_color
    )
    
    return pydiffvg.RenderFunction.serialize_scene(
        bspline.imsize, bspline.imsize, [path], [path_group]
    )

def relax_dead_geometry(model, optimizer_point, optimizer_width, 
                        dead_threshold=0.01, 
                        reset_width_value=0.35):
    
    device = model.points_3d.device
    points = model.points_3d.data
    widths = model.unit_widths.data
    
    # 1. Identify Dead Points
    # Flatten mask to 1D
    is_dead = (widths < dead_threshold).view(-1)
    dead_indices = torch.nonzero(is_dead).squeeze()
    
    num_dead = dead_indices.numel()
    if num_dead == 0:
        return

    print(f"Typo-correction: Smoothing {num_dead} dead points...")

    # 2. Laplacian Smoothing (The "Ironing" Step)
    # We move P[i] to 0.5 * (P[i-1] + P[i+1])
    
    # We clone the points first so we don't read modified values during the loop
    # (or we can just do it in place if we don't care about perfect simulatenous updates)
    new_positions = points.clone()
    
    num_total = points.shape[0]
    
    for idx in dead_indices:
        # Handle boundary conditions for a closed or open loop
        # Assuming open loop for now (clamp indices)
        prev_idx = max(0, idx - 1)
        next_idx = min(num_total - 1, idx + 1)
        
        # Calculate Midpoint
        neighbor_avg = (points[prev_idx] + points[next_idx]) / 2.0
        
        # Assign new position
        new_positions[idx] = neighbor_avg

    # Apply changes
    model.points_3d.data = new_positions
    
    # 3. Reset Widths (Gently)
    # Give it a small visible width so gradients can flow again, 
    # but not so big that it pops visually.
    model.unit_widths.data[is_dead] = reset_width_value
    
    # 4. Reset Optimizer State (Crucial!)
    # Kill the momentum that was driving these points to zero
    def reset_opt_state(optimizer, param, mask):
        if param not in optimizer.state: return
        state = optimizer.state[param]
        if 'exp_avg' in state: state['exp_avg'][mask] = 0.0
        if 'exp_avg_sq' in state: state['exp_avg_sq'][mask] = 0.0

    reset_opt_state(optimizer_point, model.points_3d, is_dead)
    reset_opt_state(optimizer_width, model.unit_widths, is_dead)

@torch.no_grad()
def redistribute_capacity(model, optimizer_point, optimizer_width, 
                          width_threshold=0.1, 
                          min_consecutive_len=10, 
                          v_reset_factor=0.1):
    
    device = model.points_3d.device
    
    # 1. Safety Check
    if model.points_3d.grad is None:
        return

    # Handle NaNs in gradients
    grads = model.points_3d.grad
    grads = torch.nan_to_num(grads, nan=0.0)
    old_grads = grads.norm(dim=1)

    # ==========================================
    # PHASE 1: PRUNING (Run-Length Logic)
    # ==========================================
    
    widths = model.unit_widths.data.view(-1) 
    is_dead = widths < width_threshold
    
    change_points = is_dead[1:] != is_dead[:-1]
    block_ids = torch.cat([torch.tensor([0], device=device), change_points.long()]).cumsum(dim=0)
    block_sizes = torch.bincount(block_ids)
    point_block_sizes = block_sizes[block_ids]
    
    to_prune = is_dead & (point_block_sizes >= min_consecutive_len)
    
    to_prune[0] = False
    to_prune[-1] = False
    
    survivor_mask = ~to_prune
    num_freed_points = to_prune.sum().item()
    
    if survivor_mask.sum() < 4:
        print("Skipping prune: Too few survivors.")
        return

    if num_freed_points > 0:
        print(f"Pruning {num_freed_points} points...")

    # Slice Data
    surviving_points = model.points_3d.data[survivor_mask]
    surviving_widths = model.unit_widths.data[survivor_mask]
    surviving_grads = old_grads[survivor_mask]

    # Slice Optimizer States
    old_p_state = optimizer_point.state.get(model.points_3d, {})
    old_w_state = optimizer_width.state.get(model.unit_widths, {})

    def slice_state(state, mask):
        if not state: return {}
        return {
            'step': state['step'], 
            'exp_avg': state['exp_avg'][mask],
            'exp_avg_sq': state['exp_avg_sq'][mask]
        }

    new_p_state_dict = slice_state(old_p_state, survivor_mask)
    new_w_state_dict = slice_state(old_w_state, survivor_mask)

    # ==========================================
    # PHASE 2: RE-INVESTMENT
    # ==========================================
    
    new_seg_vecs = surviving_points[1:] - surviving_points[:-1]
    new_seg_lens = torch.norm(new_seg_vecs, dim=1)
    
    segment_stress = torch.max(surviving_grads[:-1], surviving_grads[1:])
    segment_stress[new_seg_lens < 1e-4] = -1.0
    
    num_candidates = (segment_stress > 0).sum().item()
    k_budget = min(num_freed_points, num_candidates)
    
    if k_budget > 0:
        _, insertion_indices = torch.topk(segment_stress, k_budget)
        insertion_indices = insertion_indices.sort()[0]
    else:
        insertion_indices = torch.tensor([], device=device, dtype=torch.long)

    # Insertion with Jitter
    def insert_with_jitter(param_data, state_dict, indices, is_width=False):
        if len(indices) == 0: return param_data, state_dict
        
        N = param_data.shape[0]
        new_data_list = []
        
        has_state = len(state_dict) > 0
        new_m_list = []
        new_v_list = []
        
        if has_state:
            old_m = state_dict['exp_avg']
            old_v = state_dict['exp_avg_sq']

        for i in range(N):
            new_data_list.append(param_data[i:i+1])
            if has_state:
                new_m_list.append(old_m[i:i+1])
                new_v_list.append(old_v[i:i+1])
            
            if i in indices:
                next_i = min(i + 1, N - 1)
                
                mid_val = (param_data[i] + param_data[next_i]) / 2.0
                
                if not is_width:
                    noise = torch.randn_like(mid_val) * 1e-4
                    mid_val = mid_val + noise
                
                new_data_list.append(mid_val.unsqueeze(0))
                
                if has_state:
                    # CRITICAL FIX: Reset momentum for new points
                    mid_m = torch.zeros_like(old_m[i]) 
                    mid_v = (old_v[i] + old_v[next_i]) / 2.0 * v_reset_factor
                    
                    new_m_list.append(mid_m.unsqueeze(0))
                    new_v_list.append(mid_v.unsqueeze(0))
        
        final_data = torch.cat(new_data_list)
        final_state = {}
        if has_state:
            final_state = {
                'step': state_dict['step'], 
                'exp_avg': torch.cat(new_m_list),
                'exp_avg_sq': torch.cat(new_v_list)
            }
        return final_data, final_state

    final_p_data, final_p_state = insert_with_jitter(surviving_points, new_p_state_dict, insertion_indices, is_width=False)
    final_w_data, final_w_state = insert_with_jitter(surviving_widths, new_w_state_dict, insertion_indices, is_width=True)

    # ==========================================
    # PHASE 3: FINAL COMMIT (MODIFIED)
    # ==========================================
    
    if torch.isnan(final_p_data).any():
        final_p_data = torch.nan_to_num(final_p_data, nan=0.5)
    
    final_p_data += torch.randn_like(final_p_data) * 1e-6

    # 1. Capture Old Parameters (for optimizer cleanup)
    old_p_param = model.points_3d
    old_w_param = model.unit_widths
    
    # 2. UPDATE MODEL (Using the safe setter!)
    # This automatically updates num_kp and recalculates Matrix G
    model.set_data(final_p_data, final_w_data)
    
    # 3. Update Optimizers
    # Remove old params from internal state
    if old_p_param in optimizer_point.state: del optimizer_point.state[old_p_param]
    if old_w_param in optimizer_width.state: del optimizer_width.state[old_w_param]
    
    # Inject new state (if it exists)
    if final_p_state: optimizer_point.state[model.points_3d] = final_p_state
    if final_w_state: optimizer_width.state[model.unit_widths] = final_w_state
    
    # Point optimizer to the NEW parameters created by set_data
    optimizer_point.param_groups[0]['params'] = [model.points_3d]
    optimizer_width.param_groups[0]['params'] = [model.unit_widths]
    
    print(f"Redistribution Complete: {len(old_p_param)} -> {model.num_kp} points")

def get_gradient_driven_indices(model, k_budget, min_segment_length=0.01):
    """
    Selects the top 'k_budget' segments with the highest gradient magnitude.
    Includes a safety check to avoid splitting segments that are already too tiny.
    """
    if k_budget <= 0 or model.points_3d.grad is None:
        return torch.tensor([], dtype=torch.long, device=model.device)

    # 1. Calculate Gradient Magnitudes per Point
    # [N, 3] -> [N]
    grads = model.points_3d.grad.norm(dim=1)
    
    # 2. Map Point Gradients to Segments
    # A segment's "stress" is defined by the max gradient of its two endpoints
    # [N-1] segments
    segment_stress = torch.max(grads[:-1], grads[1:])
    
    # 3. SAFETY: Penalize tiny segments
    # We do NOT want to add points to a segment that is already microscopic.
    # It causes numerical explosions.
    points = model.points_3d.detach()
    segment_lengths = torch.norm(points[1:] - points[:-1], dim=1)
    
    # Zero out the stress score for tiny segments so they aren't picked
    # (Setting to -1 ensures they are at the bottom of the top-k list)
    segment_stress[segment_lengths < min_segment_length] = -1.0
    
    # 4. Select Top K
    # If we have fewer valid segments than k_budget, take what we can
    num_valid = (segment_stress > 0).sum()
    k = min(k_budget, num_valid)
    
    if k == 0:
        return torch.tensor([], dtype=torch.long, device=model.device)

    # indices are the segment indices (0 to N-2)
    _, indices = torch.topk(segment_stress, k)
    
    # Sort indices because insertion usually requires sorted order
    return indices.sort()[0]

def render_batch(bspline, view_w=512, view_h=512, device="cuda"):
    rendered_views = []

    # We iterate 3 times for the 3 orthogonal planes (Front, Side, Top)
    for view_idx in range(3):
        # We pass view_idx instead of a camera object
        scene_args = bspline.get_serialized_scene(view_idx, view_w, view_h)
        
        # Render the scene
        img = pydiffvg.RenderFunction.apply(view_w, view_h, 2, 2, 0, None, *scene_args)
        
        # Composite onto WHITE background
        alpha = img[:, :, 3:4]
        rgb = img[:, :, :3] 
        bg_color = torch.tensor([1.0, 1.0, 1.0], device=img.device)
        
        # alpha * foreground + (1-alpha) * background
        final_view = rgb * alpha + bg_color * (1 - alpha)
        
        # Permute to [C, H, W]
        rendered_views.append(final_view.permute(2, 0, 1))

    # Stack into a batch [3, 3, 512, 512]
    images_batch = torch.stack(rendered_views, dim=0)
    return images_batch

def save_debug_tile(images_batch, epoch):
    """
    Tiles the batch of images (e.g., 4 or 6 views) into a single grid and saves it.
    Input shape: [N, 3, H, W]
    """
    # Create a grid: if N=4, it makes a 2x2 grid. If N=6, a 2x3 grid.
    grid = torchvision.utils.make_grid(images_batch, nrow=2, padding=2, normalize=False)
    
    # Convert from [3, H_grid, W_grid] to [H_grid, W_grid, 3] for saving
    grid_hwc = grid.permute(1, 2, 0).cpu().detach()
    
    # Save using pydiffvg or torchvision
    pydiffvg.imwrite(grid_hwc, f"outputs/debug_tile_{epoch:04d}.png", gamma=1.0)
    
    return

def export_bspline_to_json(bspline_model, filepath="shadow_wire.json", num_samples=250, w_min=0.2, w_max=5.0):
    """
    Evaluates the B-spline into dense points and saves to JSON for Blender.
    """
    bspline_model.eval()
    with torch.no_grad():
        # 1. Get the dense evaluated points (1:1 shape match)
        pts_3d, base_widths = bspline_model.get_points(num_samples=num_samples)
        
        # 2. Apply the exact width transformation from your forward() method
        k = 6.0
        final_widths = w_min + (w_max - w_min) * torch.sigmoid(k * (base_widths - 0.5))
        
        # 3. Convert to standard Python lists
        # .detach().cpu().tolist() safely moves data from GPU to standard Python floats
        data = {
            "points": pts_3d.detach().cpu().tolist(),
            "widths": final_widths.squeeze(-1).detach().cpu().tolist()
        }
        
        # 4. Save to JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
    print(f"Successfully exported {num_samples} curve points to {filepath}")

def load_list_checkpoint(model, checkpoint_path):
    print(f"Loading from {checkpoint_path}...")
    
    # 1. Load the raw list
    # Expected format: [Tensor(13,3), Tensor(13,3), ... 64 times]
    raw_data = torch.load(checkpoint_path, map_location=model.device)
    
    # 2. Convert List to Stacked Tensor
    if isinstance(raw_data, list):
        # Stack into [64, 13, 3]
        new_points = torch.stack(raw_data).to(model.device)
    elif isinstance(raw_data, torch.Tensor):
        # Already a tensor
        new_points = raw_data.to(model.device)
    else:
        raise ValueError(f"Unknown data format: {type(raw_data)}")
        
    # 3. Validate Dimensions
    # Shape should be [NumPaths, NumPoints, 3]
    num_paths, num_points, dim = new_points.shape
    
    if num_points != 13:
        print(f"⚠️ WARNING: Expected 13 points (Cubic Bezier), found {num_points}.")
        
    # 4. Infer Segments (Crucial for Rendering)
    # Formula: Points = 1 + (3 * segments) -> Segments = (Points - 1) / 3
    # For 13 points: (13 - 1) / 3 = 4 Segments
    calculated_segments = (num_points - 1) // 3
    
    print(f"detected: {num_paths} paths, {calculated_segments} segments per path.")

    # 5. UPDATE THE MODEL
    with torch.no_grad():
        model.num_paths = num_paths
        model.num_segments = calculated_segments
        
        # Update Geometry
        scale_factor = 256.0

        mapping = new_points / scale_factor * torch.tensor([-1.0, 1.0, -1.0], device=model.device)
        model._3D_points = nn.Parameter(mapping)
        
        # Update Widths (Resize to match new num_paths)
        # We reset widths to a default value (e.g., 2.0) or keep old mean
        model.unit_widths = nn.Parameter(torch.full((num_paths,), 3.0, device=model.device))
        
        # Reset color just in case
        model.stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0], device=model.device)

    print("✅ Model successfully updated from checklist.")

def render_dreamwire_iso(dreamwire_model, filepath="outputs/dreamwire_overview.png", ortho_scale=2.0):
    """
    Renders the DreamWire model with 'Confetti' coloring (Random PER SEGMENT).
    """
    imsize = dreamwire_model.imsize
    device = dreamwire_model.device
    
    # 1. SETUP CAMERA & PROJECTION (Same as before)
    cam = DiffCamera(
        eye=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0],
        ortho_scale=ortho_scale, aspect=1.0, device=device
    )
    
    points_3d = dreamwire_model._3D_points 
    num_paths, num_points_per_path, _ = points_3d.shape
    points_flat = points_3d.reshape(-1, 3)
    
    mvp = cam.get_full_matrix()
    ones = torch.ones((points_flat.shape[0], 1), device=device)
    points_hom = torch.cat([points_flat, ones], dim=1)
    points_clip = points_hom @ mvp.T
    
    ndc_x = points_clip[:, 0] / points_clip[:, 3]
    ndc_y = points_clip[:, 1] / points_clip[:, 3]
    screen_x = (ndc_x + 1.0) / 2.0 * imsize
    screen_y = (1.0 - ndc_y) / 2.0 * imsize 
    
    points_2d = torch.stack([screen_x, screen_y], dim=1).reshape(num_paths, num_points_per_path, 2)

    # 2. BUILD SCENE (Segmented)
    shapes = []
    shape_groups = []
    
    current_widths = dreamwire_model.unit_widths
    if current_widths.numel() == 1:
        current_widths = current_widths.repeat(num_paths)
    
    # Calculate number of segments (Cubic Bezier logic: (N-1)/3)
    num_segments_per_path = (num_points_per_path - 1) // 3
    
    # Pre-define control points count for a SINGLE segment (2 control points)
    # Each mini-path we create will only have 1 segment.
    single_seg_ctrl = torch.tensor([2], dtype=torch.int32, device=device)

    # --- OUTER LOOP: PATHS ---
    for p_idx in range(num_paths):
        full_path_pts = points_2d[p_idx] 
        
        # Global Visibility Check (Optional optimization)
        x_min, x_max = full_path_pts[:, 0].min(), full_path_pts[:, 0].max()
        y_min, y_max = full_path_pts[:, 1].min(), full_path_pts[:, 1].max()
        if x_max < 0 or x_min > imsize or y_max < 0 or y_min > imsize:
            continue

        # --- INNER LOOP: SEGMENTS ---
        for s_idx in range(num_segments_per_path):
            
            # 1. Slice Points for this segment
            # Cubic Bezier: Start=3*i, End=3*i+4 (inclusive of next anchor)
            start_idx = 3 * s_idx
            end_idx = start_idx + 4
            segment_pts = full_path_pts[start_idx : end_idx] # Shape [4, 2]
            
            # 2. Check for degenerate "dots" within this specific segment
            s_min, s_max = segment_pts.min(0)[0], segment_pts.max(0)[0]
            if (s_max - s_min).sum() < 0.1:
                continue

            # 3. Create Path for ONE segment
            path = pydiffvg.Path(
                num_control_points=single_seg_ctrl, # Just 1 segment
                points=segment_pts.contiguous(),
                stroke_width=current_widths[p_idx],
                is_closed=False
            )
            shapes.append(path)
            
            # 4. Random Color PER SEGMENT
            num_categories = 10
            rand_idx = torch.randint(0, num_categories, (1,), device=device)
            t_discrete = (rand_idx.float() + 0.5) / num_categories
            
            # Use 'tab10' for high contrast distinct segments
            color_tensor = get_premium_colors_from_t(t_discrete, cmap_name='cool', device=device)
            stroke_color = color_tensor[0]
            
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes)-1], device=device),
                fill_color=None,
                stroke_color=stroke_color
            )
            shape_groups.append(path_group)

    # --- FALLBACK ---
    if len(shapes) == 0:
         print("Warning: Scene is empty.")
         dummy_path = pydiffvg.Path(
            num_control_points=torch.tensor([0], dtype=torch.int32, device=device),
            points=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device),
            stroke_width=torch.tensor(0.0, device=device),
            is_closed=False
         )
         shapes.append(dummy_path)
         shape_groups.append(pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0], device=device),
            fill_color=None,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
         ))

    # 3. RENDER
    scene_args = pydiffvg.RenderFunction.serialize_scene(imsize, imsize, shapes, shape_groups)
    img = pydiffvg.RenderFunction.apply(imsize, imsize, 2, 2, 0, None, *scene_args)


    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
        
    pydiffvg.imwrite(img.cpu(), filepath, gamma=1.0)
    print(f"Saved Segmented ISO render to {filepath}")
    return img

def export_dreamwire_to_json(dreamwire_model, filepath="dream_wire.json", samples_per_seg=30):
    """
    Evaluates multiple Cubic Bezier paths from DreamWire into dense points 
    and saves to JSON for Blender.
    """
    dreamwire_model.eval()
    all_paths_data = []
    
    with torch.no_grad():
        points_3d = dreamwire_model._3D_points  # [num_paths, 13, 3]
        widths = dreamwire_model.unit_widths    # [num_paths]
        
        num_paths = points_3d.shape[0]
        num_segments = dreamwire_model.num_segments
        
        # 1. Pre-calculate the Cubic Bezier basis functions
        # B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
        t = torch.linspace(0, 1, samples_per_seg, device=dreamwire_model.device).view(-1, 1)
        u_inv = 1.0 - t
        basis = torch.cat([
            u_inv**3, 
            3 * t * (u_inv**2), 
            3 * (t**2) * u_inv, 
            t**3
        ], dim=1) # Shape: [samples, 4]
        
        # 2. Evaluate every path
        for p in range(num_paths):
            path_pts = points_3d[p]
            path_width = widths[p].item()
            
            dense_path = []
            for s in range(num_segments):
                # Grab the 4 control points for this specific segment
                # Segment 0: 0,1,2,3 | Segment 1: 3,4,5,6 | etc.
                idx_start = s * 3
                seg_ctrl = path_pts[idx_start : idx_start + 4]
                
                # Matrix multiply the basis by the control points to get 3D coordinates
                seg_dense = torch.matmul(basis, seg_ctrl)
                
                # Prevent duplicate points where segments connect
                if s < num_segments - 1:
                    seg_dense = seg_dense[:-1]
                    
                dense_path.append(seg_dense)
            
            # Combine the segments for this path
            dense_path_tensor = torch.cat(dense_path, dim=0)
            
            # DreamWire uses a constant width per path, so we just copy it for every point
            point_widths = [path_width] * dense_path_tensor.shape[0]
            
            all_paths_data.append({
                "points": dense_path_tensor.cpu().tolist(),
                "widths": point_widths
            })
            
    # 3. Save to JSON
    with open(filepath, 'w') as f:
        json.dump(all_paths_data, f, indent=4)
        
    print(f"Successfully exported {num_paths} DreamWire paths to {filepath}")


def render_sample_dreamwire(size=128, device='cpu', output=None):
    """Render three orthogonal views from the bundled DreamWire sample."""
    prepare_output_dirs()
    pydiffvg.set_use_gpu(device == 'cuda')
    model = DreamWire(num_paths=39, device=device, imsize=size)
    load_list_checkpoint(model, ROOT / 'inputs' / 'points_final.pt')
    with torch.no_grad():
        views = render_batch(model, size, size, device=device)
        grid = torchvision.utils.make_grid(views, nrow=3, padding=4, pad_value=1.0)
    destination = Path(output) if output else ROOT / 'outputs' / 'demo_dreamwire.png'
    destination.parent.mkdir(parents=True, exist_ok=True)
    pydiffvg.imwrite(grid.permute(1, 2, 0).cpu(), str(destination), gamma=1.0)
    return destination


def render_sample_bspline(size=128, device='cpu', output=None):
    """Render three orthogonal views from the bundled B-spline checkpoint."""
    prepare_output_dirs()
    pydiffvg.set_use_gpu(device == 'cuda')
    state = torch.load(ROOT / 'checkpoints' / 'model_1200.pth', map_location=device, weights_only=True)
    points = state['points_3d'].to(device)
    model = UnitaryBSpline3D(num_kp=len(points), kp_init=points, device=device, imsize=size)
    model.load_state_dict(state)
    with torch.no_grad():
        views = render_batch(model, size, size, device=device)
        grid = torchvision.utils.make_grid(views, nrow=3, padding=4, pad_value=1.0)
    destination = Path(output) if output else ROOT / 'outputs' / 'demo_bspline.png'
    destination.parent.mkdir(parents=True, exist_ok=True)
    pydiffvg.imwrite(grid.permute(1, 2, 0).cpu(), str(destination), gamma=1.0)
    return destination


def train_bspline(steps=1801, prompt_set='food', seed=365, schedule_steps=1801):
    """Train the original single-wire B-spline workflow on a CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError('B-spline training requires CUDA for Stable Diffusion guidance')
    os.chdir(ROOT)
    prepare_output_dirs()
    if steps < 1 or schedule_steps < 1:
        raise ValueError('steps and schedule_steps must be positive')
    device = 'cuda'
    pydiffvg.set_use_gpu(True)
    guidance = StableDiffusion(device)
    torch.manual_seed(seed)
    num_kp = 210
    keypoints = initialize_sphere_tsp(num_kp, radius=0.8, device=device)
    bspline = UnitaryBSpline3D(num_kp, keypoints.detach(), device=device)
    save_debug_tile(render_batch(bspline, device=device), 0)
    opt_bspline = torch.optim.Adam([bspline.points_3d], lr=0.004)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_bspline, milestones=[50, 100], gamma=0.7)
    opt_fixed = torch.optim.Adam([bspline.unit_widths], lr=0.006)
    prompt_sets = json.loads((ROOT / 'inputs' / 'prompt.json').read_text())
    if prompt_set not in prompt_sets:
        raise ValueError(f'Unknown prompt set: {prompt_set}. Choose from {sorted(prompt_sets)}')
    text_embeds_batch = guidance.get_text_embeds(prompt_sets[prompt_set])
    for epoch in range(steps):
        opt_bspline.zero_grad()
        opt_fixed.zero_grad()
        batch_render = render_batch(bspline, device=device)
        latents = guidance.vae.encode(2.0 * batch_render - 1.0).latent_dist.sample() * 0.18215
        sd_loss = guidance.get_sds_loss(latents, text_embeds_batch, ratio=epoch / schedule_steps)
        s_pos, s_width = bspline.get_jerk_loss()
        loss = sd_loss + 0.1 * s_pos + 0.1 * s_width
        loss.backward()
        opt_bspline.step()
        scheduler.step()
        opt_fixed.step()
        if epoch == 800:
            redistribute_capacity(bspline, opt_bspline, opt_fixed)
        if epoch in (300, 500):
            relax_dead_geometry(bspline, opt_bspline, opt_fixed)
        if epoch in (800, 1000, 1200, 1400):
            torch.save(bspline.state_dict(), ROOT / 'outputs' / f'model_{epoch}.pth')
        if epoch % 50 == 0:
            print(f'Epoch {epoch} | SDS: {sd_loss.item():.4f} | Smooth: {s_pos.item():.4f} | Stroke: {s_width.item():.4f}')
            save_debug_tile(batch_render, epoch)
    torch.save(bspline.state_dict(), ROOT / 'outputs' / 'model_final.pth')
    torch.save(keypoints, ROOT / 'outputs' / 'initialization.pt')
    return bspline


def train_dreamwire(steps=801, seed=365, prompts=None):
    """Train the original multi-path DreamWire workflow on a CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError('DreamWire training requires CUDA for Stable Diffusion guidance')
    os.chdir(ROOT)
    prepare_output_dirs()
    torch.manual_seed(seed)
    device = 'cuda'
    pydiffvg.set_use_gpu(True)
    dreamwire = DreamWire(num_paths=64, device=device)
    guidance = StableDiffusion(device)
    prompts = prompts or [
        'a icon of an large ancient tree on a white background',
        'a sketch of a beautiful flower on a white background',
        'a simple sketch of a cartoon sun on a white background',
    ]
    if len(prompts) != 3:
        raise ValueError('DreamWire requires exactly three view prompts')
    text_embeds_batch = guidance.get_text_embeds(prompts)
    current_lr = 0.002
    optimizer = torch.optim.Adam([{'params': dreamwire._3D_points, 'lr': current_lr}])
    for epoch in range(steps):
        optimizer.zero_grad()
        batch_render = render_batch(dreamwire, 512, 512, device=device).clamp(0.0, 1.0)
        latents = guidance.vae.encode(2.0 * batch_render - 1.0).latent_dist.sample() * 0.18215
        sds_loss = guidance.get_sds_loss(latents, text_embeds_batch, ratio=epoch / steps)
        mst_length = dreamwire.get_prim_loss()
        total_loss = sds_loss + 0.1 * mst_length
        if torch.isfinite(total_loss):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(dreamwire.parameters(), 1.0)
            optimizer.step()
        if epoch % 50 == 0 and epoch > 0:
            dreamwire.points_restrict()
            dreamwire.reinitialize_paths(threshold=0.02)
            current_lr *= 0.95
            optimizer = torch.optim.Adam([{'params': dreamwire._3D_points, 'lr': current_lr}])
            print(f'Epoch {epoch} | SDS: {sds_loss.item():.4f} | MST: {mst_length.item():.4f} | LR: {current_lr:.5f}')
            save_debug_tile(batch_render, epoch)
    torch.save(dreamwire.state_dict(), ROOT / 'outputs' / 'dreamwire_model.pth')
    return dreamwire


def main(argv=None):
    parser = argparse.ArgumentParser(description='4Dwire multi-view wire art')
    subcommands = parser.add_subparsers(dest='command', required=True)
    for command in ('demo-dreamwire', 'demo-bspline'):
        p = subcommands.add_parser(command)
        p.add_argument('--size', type=int, default=128)
        p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
        p.add_argument('--output', type=Path)
    p = subcommands.add_parser('train-bspline')
    p.add_argument('--steps', type=int, default=1801)
    p.add_argument('--prompt-set', default='food')
    p.add_argument('--seed', type=int, default=365)
    p.add_argument('--schedule-steps', type=int, default=1801, help='Noise schedule length from the original notebook')
    p = subcommands.add_parser('train-dreamwire')
    p.add_argument('--steps', type=int, default=801)
    p.add_argument('--seed', type=int, default=365)
    args = parser.parse_args(argv)
    if args.command == 'demo-dreamwire':
        print(render_sample_dreamwire(args.size, args.device, args.output))
    elif args.command == 'demo-bspline':
        print(render_sample_bspline(args.size, args.device, args.output))
    elif args.command == 'train-bspline':
        train_bspline(args.steps, args.prompt_set, args.seed, args.schedule_steps)
    elif args.command == 'train-dreamwire':
        train_dreamwire(args.steps, args.seed)


if __name__ == '__main__':
    main()
