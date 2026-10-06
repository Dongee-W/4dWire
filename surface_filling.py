"""Surface filling wire from 3d_surface_filling_v6.ipynb.

Runs as a standalone companion to the multi-view 4Dwire experiment. The
procedural examples need no external geometry; OBJ inputs use the same axis
permutation (x, y, z) -> (z, x, y) as the source notebook.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw


def load_ordered_sfc(path: str | Path, simplify_epsilon: float = 0.0) -> np.ndarray:
    """Read one unbranched OBJ polyline and return its ordered 3D points."""
    vertices, adjacency = [], {}
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == 'v':
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == 'l':
                indices = [int(value.split('/')[0]) - 1 for value in fields[1:]]
                for a, b in zip(indices, indices[1:]):
                    adjacency.setdefault(a, set()).add(b)
                    adjacency.setdefault(b, set()).add(a)
    if not adjacency or any(len(neighbors) > 2 for neighbors in adjacency.values()):
        raise ValueError(f'{path}: expected one nonbranching OBJ line')
    ends = [node for node, neighbors in adjacency.items() if len(neighbors) == 1]
    if len(ends) not in (0, 2):
        raise ValueError(f'{path}: expected an open line or a closed loop')
    start = min(ends) if ends else min(adjacency)
    order, previous, current = [], None, start
    while current is not None:
        order.append(current)
        candidates = adjacency[current] - ({previous} if previous is not None else set())
        next_node = next((node for node in sorted(candidates) if node != start), None)
        previous, current = current, next_node
    if len(order) != len(adjacency):
        raise ValueError(f'{path}: OBJ line has disconnected components')
    points = np.asarray(vertices, dtype=np.float32)[order][:, [2, 0, 1]]
    if simplify_epsilon > 0:
        # Keep endpoints and remove points close to the previous retained point.
        keep = [0]
        for index in range(1, len(points) - 1):
            if np.linalg.norm(points[index] - points[keep[-1]]) >= simplify_epsilon:
                keep.append(index)
        keep.append(len(points) - 1)
        points = points[keep]
    return points


def resample_curve(points: np.ndarray, count: int) -> np.ndarray:
    """Sample an ordered curve uniformly in arc length for spline control points."""
    if count < 4 or len(points) < 4:
        raise ValueError('At least four input and target points are required')
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, distances > 1e-8]
    points = points[keep]
    if len(points) < 4:
        raise ValueError('Curve has fewer than four distinct points')
    cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if cumulative[-1] <= 0:
        raise ValueError('Curve has zero length')
    sample = np.linspace(0, cumulative[-1], count)
    return np.stack([np.interp(sample, cumulative, points[:, axis]) for axis in range(3)], axis=1).astype(np.float32)


def procedural_example(name: str, samples: int = 1800):
    """Return (ordered wire points, triangular mesh) in Z-up coordinates."""
    import trimesh
    if name == 'sphere':
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.8)
        t = np.linspace(0.001, 0.999, samples)
        phi, theta = np.pi * t, 38 * np.pi * t
        curve = np.stack([0.81 * np.sin(phi) * np.cos(theta),
                          0.81 * np.sin(phi) * np.sin(theta), 0.81 * np.cos(phi)], axis=1)
    elif name == 'torus':
        major, minor = 0.58, 0.23
        rings, sides = 48, 16
        u, v = np.meshgrid(np.arange(rings) * 2 * np.pi / rings,
                           np.arange(sides) * 2 * np.pi / sides, indexing='ij')
        xyz = np.stack([(major + minor * np.cos(v)) * np.cos(u),
                        (major + minor * np.cos(v)) * np.sin(u),
                        minor * np.sin(v)], axis=-1)
        faces = []
        for i in range(rings):
            for j in range(sides):
                a = i * sides + j
                b = ((i + 1) % rings) * sides + j
                c = i * sides + (j + 1) % sides
                d = ((i + 1) % rings) * sides + (j + 1) % sides
                faces.extend([(a, b, c), (b, d, c)])
        mesh = trimesh.Trimesh(vertices=xyz.reshape(-1, 3), faces=faces, process=False)
        t = np.linspace(0, 1, samples)
        u, v = 2 * np.pi * 9 * t, 2 * np.pi * t
        curve = np.stack([(major + minor * np.cos(v)) * np.cos(u),
                          (major + minor * np.cos(v)) * np.sin(u), minor * np.sin(v)], axis=1)
    else:
        raise ValueError(f'Unknown example: {name}')
    return curve.astype(np.float32), mesh


def load_mesh(path: str | Path, max_edge: float = 0.12):
    import trimesh
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f'{path}: expected a triangular mesh OBJ')
    if max_edge > 0:
        vertices, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.vertices = np.asarray(mesh.vertices)[:, [2, 0, 1]]
    return mesh


class Camera:
    """Perspective camera with the source notebook's Z-up convention."""
    def __init__(self, eye, device, fov=55.0):
        self.eye = torch.as_tensor(eye, device=device, dtype=torch.float32)
        self.fov = fov

    def project(self, points, size):
        forward = F.normalize(-self.eye, dim=0)
        up = torch.tensor([0., 0., 1.], device=points.device)
        right = F.normalize(torch.linalg.cross(forward, up), dim=0)
        up = torch.linalg.cross(right, forward)
        relative = points - self.eye
        depth = relative @ forward
        scale = size / (2 * math.tan(math.radians(self.fov) / 2))
        xy = torch.stack([size / 2 + scale * (relative @ right) / depth.clamp_min(1e-4),
                          size / 2 - scale * (relative @ up) / depth.clamp_min(1e-4)], dim=1)
        return xy, depth


def critical_cameras(device):
    return [Camera(eye, device) for eye in ((-3, 0, .3), (0, -3, .3), (.1, 0, 3), (0, 3, .3), (3, 0, .3))]


def silhouette(mesh, camera, size, device):
    """Rasterize projected mesh triangles for the silhouette target."""
    vertices = torch.as_tensor(np.asarray(mesh.vertices), device=device, dtype=torch.float32)
    xy, depth = camera.project(vertices, size)
    xy = xy.detach().cpu().numpy()
    positive = (depth > 0).cpu().numpy()
    canvas = Image.new('L', (size, size))
    draw = ImageDraw.Draw(canvas)
    for face in np.asarray(mesh.faces):
        if positive[face].all():
            draw.polygon([tuple(point) for point in xy[face]], fill=255)
    return torch.from_numpy(np.asarray(canvas).copy()).to(device=device, dtype=torch.float32)[None, None] / 255


def vertex_depth(mesh_vertices, camera, size):
    """A splatted vertex Z-buffer for the notebook's wire occlusion rule."""
    xy, depth = camera.project(mesh_vertices, size)
    pixels = xy.round().long()
    valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < size) &
             (pixels[:, 1] >= 0) & (pixels[:, 1] < size) & (depth > 0))
    flat = pixels[valid, 1] * size + pixels[valid, 0]
    buffer = torch.full((size * size,), 1e4, device=mesh_vertices.device)
    buffer.scatter_reduce_(0, flat, depth[valid], reduce='amin', include_self=True)
    image = buffer.view(1, 1, size, size)
    return -F.max_pool2d(-image, kernel_size=7, stride=1, padding=3)


class SurfaceFillingCurve(nn.Module):
    """Differentiable cubic wire with learnable 3D positions and widths."""
    def __init__(self, points, device):
        super().__init__()
        self.points_3d = nn.Parameter(torch.as_tensor(points, device=device, dtype=torch.float32).clone())
        self.unit_widths = nn.Parameter(torch.full((len(points),), .3, device=device))

    def sample(self, samples_per_segment=2):
        points = self.points_3d
        widths = self.unit_widths[:, None]
        def cubic(values):
            padded = torch.cat([values[:1], values, values[-1:]], dim=0)
            a, b, c, d = (padded[index:index + len(values) - 1] for index in range(4))
            t = torch.arange(samples_per_segment, device=values.device, dtype=values.dtype) / samples_per_segment
            t = t[None, :, None]
            sampled = .5 * ((2 * b[:, None]) + (-a + c)[:, None] * t +
                            (2*a - 5*b + 4*c - d)[:, None] * t*t +
                            (-a + 3*b - 3*c + d)[:, None] * t*t*t)
            return torch.cat([sampled.reshape(-1, values.shape[1]), values[-1:]], dim=0)
        return cubic(points), cubic(widths).squeeze(1)

    def render(self, camera, mesh_depth, size, samples_per_segment=2):
        import pydiffvg
        points, widths = self.sample(samples_per_segment)
        screen, depth = camera.project(points, size)
        grid = torch.stack([screen[:, 0] / (size - 1) * 2 - 1,
                            screen[:, 1] / (size - 1) * 2 - 1], dim=-1)[None, None]
        nearest_depth = F.grid_sample(mesh_depth, grid, align_corners=True, padding_mode='border').flatten()
        visibility = torch.sigmoid((nearest_depth - depth + .05) * 5)
        stroke = (.2 + 8.3 * torch.sigmoid(4 * (widths * 5 / depth.clamp_min(.1) - .5))) * visibility
        path = pydiffvg.Path(num_control_points=torch.zeros(len(screen) - 1, dtype=torch.int32),
                             points=screen, stroke_width=stroke, is_closed=False)
        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=None,
                                    stroke_color=torch.tensor([.1, .1, .1, 1.], device=points.device))
        scene = pydiffvg.RenderFunction.serialize_scene(size, size, [path], [group])
        return pydiffvg.RenderFunction.apply(size, size, 2, 2, 0, None, *scene)


def multiresolution_mse(rendered, target, levels=4):
    loss, weight = 0., 1.
    for level in range(levels):
        loss = loss + weight * F.mse_loss(rendered, target)
        if level < levels - 1 and min(rendered.shape[-2:]) >= 2:
            rendered = F.avg_pool2d(rendered, 2)
            target = F.avg_pool2d(target, 2)
            weight *= 1.5
    return loss


def save_image(image, path):
    image = image.detach().cpu().clamp(0, 1)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3] * image[..., 3:] + (1 - image[..., 3:])
    array = (image.numpy() * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


class ClipDirection:
    """Optional directional CLIP objective from the v6 notebook."""
    def __init__(self, args, device):
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained=args.clip_pretrained, device=str(device), precision='fp32', weights_only=False)
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.device = device
        if args.target_prompt:
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            tokens = tokenizer([args.source_prompt, args.target_prompt]).to(device)
            with torch.no_grad():
                features = F.normalize(model.encode_text(tokens).float(), dim=-1)
                direction = features[1] - features[0]
        else:
            if not args.style_source_image or not args.style_target_image:
                raise ValueError('Style mode requires both --style-source-image and --style-target-image')
            with torch.no_grad():
                images = torch.stack([
                    preprocess(Image.open(path).convert('RGB'))
                    for path in (args.style_source_image, args.style_target_image)
                ]).to(device)
                features = F.normalize(model.encode_image(images).float(), dim=-1)
                direction = features[1] - features[0]
        self.direction = F.normalize(direction, dim=0)
        self.grid = args.clip_grid

    def patches(self, image):
        rgb = image[..., :3] * image[..., 3:] + (1 - image[..., 3:])
        chunks = []
        height, width = rgb.shape[:2]
        for row in range(self.grid):
            for col in range(self.grid):
                y0, y1 = height * row // self.grid, height * (row + 1) // self.grid
                x0, x1 = width * col // self.grid, width * (col + 1) // self.grid
                patch = rgb[y0:y1, x0:x1].permute(2, 0, 1)[None]
                chunks.append(F.interpolate(patch, size=(224, 224), mode='bicubic', align_corners=False))
        images = torch.cat(chunks).clamp(0, 1)
        mean = images.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
        std = images.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
        return (images - mean) / std

    def loss(self, current, source):
        current_features = F.normalize(self.model.encode_image(self.patches(current)).float(), dim=-1)
        with torch.no_grad():
            source_features = F.normalize(self.model.encode_image(self.patches(source)).float(), dim=-1)
        change = current_features - source_features
        valid = change.norm(dim=-1) > 1e-7
        if not valid.any():
            return current_features.sum() * 0
        return 1 - F.cosine_similarity(F.normalize(change[valid], dim=-1),
                                       self.direction[None], dim=-1).mean()


def run(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable in this environment')
    device = torch.device(args.device)
    import pydiffvg
    pydiffvg.set_use_gpu(device.type == 'cuda')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.curve_obj or args.mesh_obj:
        if not args.curve_obj or not args.mesh_obj:
            raise ValueError('--curve-obj and --mesh-obj must be supplied together')
        raw_points = load_ordered_sfc(args.curve_obj, args.simplify_epsilon)
        mesh = load_mesh(args.mesh_obj, args.max_edge)
    else:
        raw_points, mesh = procedural_example(args.example)
    points = resample_curve(raw_points, args.points)
    model = SurfaceFillingCurve(points, device)
    mesh_vertices = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cameras = critical_cameras(device)
    optimizer = torch.optim.Adam([{'params': model.points_3d, 'lr': args.point_lr},
                                  {'params': model.unit_widths, 'lr': args.width_lr}])
    clip = ClipDirection(args, device) if args.target_prompt or args.style_target_image else None
    initial_model = SurfaceFillingCurve(points, device) if clip else None
    for step in range(args.steps):
        optimizer.zero_grad()
        base = cameras[step % len(cameras)]
        if args.perturb_degrees:
            noise = torch.randn(3, device=device) * math.radians(args.perturb_degrees) * .15
            camera = Camera((base.eye + noise).tolist(), device)
        else:
            camera = base
        target = silhouette(mesh, camera, args.size, device)
        with torch.no_grad():
            depth = vertex_depth(mesh_vertices, camera, args.size)
        image = model.render(camera, depth, args.size, args.samples_per_segment)
        area = image[..., 3][None, None]
        coverage = multiresolution_mse(area, target * .6)
        # Keep the wire near the mesh and avoid sharp control-point changes.
        nearest = torch.cdist(model.points_3d[None], mesh_vertices[None]).min(dim=-1).values
        surface_loss = nearest.square().mean()
        smoothness = torch.diff(model.points_3d, n=2, dim=0).square().mean()
        loss = .5 * coverage + args.surface_weight * surface_loss + args.smooth_weight * smoothness
        if clip:
            with torch.no_grad():
                source_image = initial_model.render(camera, depth, args.size, args.samples_per_segment)
            loss = loss + args.clip_weight * clip.loss(image, source_image)
        loss.backward()
        optimizer.step()
        if step % args.save_every == 0 or step == args.steps - 1:
            save_image(image, output / f'step_{step:04d}.png')
            print(f'step {step:4d} loss={loss.item():.5f} silhouette={coverage.item():.5f} surface={surface_loss.item():.5f}')
    with torch.no_grad():
        for index, camera in enumerate(cameras):
            target = silhouette(mesh, camera, args.size, device)
            depth = vertex_depth(mesh_vertices, camera, args.size)
            image = model.render(camera, depth, args.size, args.samples_per_segment)
            save_image(image, output / f'view_{index}.png')
            save_image(target[0, 0], output / f'target_{index}.png')
        torch.save({'points_3d': model.points_3d.detach().cpu(),
                    'unit_widths': model.unit_widths.detach().cpu()}, output / 'curve.pt')
    print(f'Saved five views and curve.pt to {output}')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', choices=('sphere', 'torus'), default='sphere')
    parser.add_argument('--curve-obj', type=Path, help='Ordered OBJ line from the surface filling workflow')
    parser.add_argument('--mesh-obj', type=Path, help='Mesh OBJ corresponding to --curve-obj')
    parser.add_argument('--output', type=Path, default=Path('outputs/surface_filling'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--points', type=int, default=1100)
    parser.add_argument('--samples-per-segment', type=int, default=2)
    parser.add_argument('--steps', type=int, default=251, help='The v6 geometry loop uses 251 updates')
    parser.add_argument('--seed', type=int, default=365)
    parser.add_argument('--point-lr', type=float, default=.004)
    parser.add_argument('--width-lr', type=float, default=.3)
    parser.add_argument('--surface-weight', type=float, default=.02)
    parser.add_argument('--smooth-weight', type=float, default=.002)
    parser.add_argument('--perturb-degrees', type=float, default=20.)
    parser.add_argument('--simplify-epsilon', type=float, default=.01)
    parser.add_argument('--max-edge', type=float, default=.12)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--target-prompt', help='Enable the v6 directional text CLIP objective')
    parser.add_argument('--source-prompt', default='A continuous 3D wire drawing')
    parser.add_argument('--style-target-image', type=Path, help='Enable the v6 directional image style objective')
    parser.add_argument('--style-source-image', type=Path)
    parser.add_argument('--clip-pretrained', help='Local CLIPAG checkpoint or open_clip pretrained tag')
    parser.add_argument('--clip-weight', type=float, default=1.0)
    parser.add_argument('--clip-grid', type=int, default=4)
    args = parser.parse_args(argv)
    if args.steps < 0 or args.save_every < 1 or args.size < 16 or args.samples_per_segment < 1:
        parser.error('steps must be nonnegative; save-every, size, and samples-per-segment must be positive')
    if (args.target_prompt or args.style_target_image) and not args.clip_pretrained:
        parser.error('--clip-pretrained is required for CLIP guidance')
    if args.target_prompt and args.style_target_image:
        parser.error('Choose either --target-prompt or --style-target-image')
    if args.clip_grid < 1 or args.clip_grid > args.size:
        parser.error('--clip-grid must be between 1 and image size')
    run(args)


if __name__ == '__main__':
    main()
