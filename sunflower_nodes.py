import numpy as np

from dataclasses import dataclass
import torch
import torch.nn.functional as F
import math
import matplotlib
import matplotlib.pyplot as plt
import tempfile
from PIL import Image

import nodes
import time


@dataclass
class Rectilinear:
    """Point(s) on a xy plane, 1 'z' unit from origin, 0 'x' (or 'y') is center of any image"""

    x: float | np.ndarray
    y: float | np.ndarray


@dataclass
class AngularCoordinate:
    # inclination => polar
    "The azimuth is on the xy plane, and the elevation is away from the z"
    elevation: float | np.ndarray  # ~latitude in radians, 0 is straight ahead
    azimuth: float | np.ndarray  # ~longitude radians, 0 is straight ahead


@dataclass
class ImageCoordinate:
    x: float | np.ndarray  # -1 to 1
    y: float | np.ndarray  # -1 to 1


def rectilinear_to_angular_projection(
    rect: Rectilinear, coord: AngularCoordinate
) -> AngularCoordinate:
    """

    Maps point(s) from an equirectangular (latitude, longitude) to a spherical angular coordinates.

    Parameters:
        rect (Rectilinear): Rectilinear point(s) on a plane
        coord (AngularCoordinate): direction of center of projection in sphere

    Returns:
        AngularCoordinate: direction of point(s), in AngularCoordinate(s)
    """
    # y_coords = y_coords + 0.5
    X, Y = rect.x, rect.y
    Z = 1
    phi0, theta0 = coord.elevation, coord.azimuth

    theta = np.arctan2(X, Z)
    p = np.sqrt(X**2 + Y**2)
    c = np.arctan(p)

    # fixes the div-by-zero issue
    Y_p = Y.copy()
    Y_p[p != 0] /= p[p != 0]

    assert np.all(Y_p[p == 0] == 0), "Found case where p == 0, and Y_p /= 0"

    phi = np.arcsin(np.cos(c) * np.sin(phi0) + (Y_p * np.sin(c) * np.cos(phi0)))
    theta = theta0 + np.arctan2(
        X * np.sin(c), p * np.cos(phi0) * np.cos(c) - Y * np.sin(phi0) * np.sin(c)
    )

    u = theta
    v = phi

    map_x, map_y = u.astype(np.float32), v.astype(np.float32)

    return AngularCoordinate(azimuth=map_x, elevation=map_y)


def angular_to_image_projection(
    angle: AngularCoordinate, width: int, height: int, border: int | None
) -> ImageCoordinate:
    width = np.deg2rad(width) / 2
    height = np.deg2rad(height) / 2

    map_x2 = angle.azimuth / width
    map_y2 = angle.elevation / height

    if border:
        map_x2[map_x2 < -1.0] = -1
        map_x2[map_x2 > 1.0] = -1
        map_y2[map_y2 < -1.0] = -1
        map_y2[map_y2 > 1.0] = -1

    return ImageCoordinate(map_x2, map_y2)


class ResizeDown:
    """
    Resize an image downwards
    """

    scale_modes = ["bilinear", "nearest", "bicubic", "area"]

    def __init__(self):
        self.counter = 0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "resize": (
                    "FLOAT",
                    {
                        "default": 8,
                        "max": 32,  # Maximum value
                        "min": 1,  # Minimum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                        "lazy": True,  # Will only be evaluated if check_lazy_status requires it
                    },
                ),
                "mode": (s.scale_modes, {"default": s.scale_modes[3]}),
                "modulo": (
                    "INT",
                    {
                        "default": 1,
                        "max": 16,  # Maximum value
                        "min": 1,  # Minimum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                        "lazy": True,  # Will only be evaluated if check_lazy_status requires it
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_down"
    CATEGORY = "image/transform"

    def resize_down(self, image, resize, mode, modulo):

        image = image.permute(0, 3, 1, 2)

        H, W = image.shape[2:4]
        H = int(H / resize)
        W = int(W / resize)
        if W % modulo != 0:
            W = W - (W % modulo)
        if H % modulo != 0:
            H = H - (H % modulo)
        # use F.adaptive_avg_pool2d to reduce to something we can bilinear over
        image = F.interpolate(image, size=(H, W), mode=mode)

        image = image.permute(0, 2, 3, 1)

        return {"ui": {"size": [H, W]}, "result": (image,)}


class MaskChannelSelect:
    """
    Select a channel from an image
    """

    def __init__(self):
        self.counter = 0

    channel_types = [
        "Red",
        "Green",
        "Blue",
        "Alpha",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (s.channel_types, {"default": s.channel_types[0]}),
            },
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "select"
    CATEGORY = "image/transform"

    def select(self, image, channel):
        assert len(image.shape) == 4
        assert channel in self.channel_types
        height, width = image.shape[1:3]
        match channel:
            case "Red":
                mask = image[:, :, :, 0]
            case "Green":
                mask = image[:, :, :, 1]
            case "Blue":
                mask = image[:, :, :, 2]
            case "Alpha":
                mask = image[:, :, :, 3]
        return (mask,)


class ImageChannelSelect:
    """
    Select a channel from a 3D feed
    """

    def __init__(self):
        self.counter = 0

    channel_types = [
        "RGB",
        "Left",
        "Right",
        "Top",
        "Bottom",
        "Left:Right",
        "Top:Bottom",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (s.channel_types, {"default": s.channel_types[0]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "select"
    CATEGORY = "image/transform"

    def select(self, image, channel):
        assert len(image.shape) == 4
        assert channel in self.channel_types
        height, width = image.shape[1:3]
        match channel:
            case "RGB":
                image = image[:, :, :, 0:3]
            case "Left":
                image = image[:, :, 0 : width // 2, :]
            case "Right":
                image = image[:, :, width // 2 :, :]
            case "Top":
                image = image[:, 0 : height // 2, :, :]
            case "Bottom":
                image = image[:, height // 2 :, :, :]
            case "Left:Right":
                images_left, images_right = torch.split(image, width // 2, dim=2)
                image = torch.concat([images_left, images_right], axis=0)
            case "Top:Bottom":
                images_top, images_bottom = torch.split(image, height // 2, dim=1)
                image = torch.concat([images_top, images_bottom], axis=0)
        return (image,)


class EquirectangularToRectilinear:
    """Equirectangular into Rectilinear projection"""

    def __init__(self):
        pass

    width_types = [180, 360]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "input_horizontal_fov": (s.width_types, {"default": s.width_types[1]}),
                "longitude": (
                    "INT",
                    {
                        "default": 0,
                        "max": 180,  # Maximum value
                        "min": -180,  # Minimum value
                        "step": 5,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                        "lazy": True,  # Will only be evaluated if check_lazy_status requires it
                    },
                ),
                "latitude": (
                    "INT",
                    {
                        "default": 0,
                        "max": 180,  # Maximum value
                        "min": -180,  # Minimum value
                        "step": 5,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                        "lazy": True,  # Will only be evaluated if check_lazy_status requires it
                    },
                ),
                "output_diagonal_fov": (
                    "INT",
                    {
                        "default": 60,
                        "max": 150,  # Maximum value
                        "min": 30,  # Minimum value
                        "step": 5,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                        "lazy": True,  # Will only be evaluated if check_lazy_status requires it
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,  # Minimum value
                        "max": nodes.MAX_RESOLUTION,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                        "lazy": True,  # Will only be evaluated if check_lazy_status requires it
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,  # Minimum value
                        "max": nodes.MAX_RESOLUTION,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                        "lazy": True,  # Will only be evaluated if check_lazy_status requires it
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "project"
    CATEGORY = "image/transform"

    def project(
        self,
        image,
        input_horizontal_fov,
        longitude,
        latitude,
        output_diagonal_fov,
        width,
        height,
    ):
        assert image.shape[0] == 1  # batch 1 for now

        fov_rad = np.deg2rad(output_diagonal_fov)

        output_diagonal = np.sqrt(width**2 + height**2)

        x_max = np.tan(fov_rad / 2) * width / output_diagonal
        y_max = np.tan(fov_rad / 2) * height / output_diagonal

        x_coords = np.linspace(-x_max, x_max, width)
        y_coords = np.linspace(-y_max, y_max, height)

        X, Y = np.meshgrid(x_coords, y_coords)

        coord = rectilinear_to_angular_projection(
            Rectilinear(X, Y),
            AngularCoordinate(
                azimuth=np.deg2rad(longitude), elevation=np.deg2rad(latitude)
            ),
        )

        rectilinear = angular_to_image_projection(
            coord, width=input_horizontal_fov, height=180, border=-1
        )

        grid = torch.stack(
            (
                torch.from_numpy(rectilinear.x),
                torch.from_numpy(rectilinear.y),
            ),
            dim=-1,
        ).unsqueeze(0)

        # TODO: Add wrapping for 360 views
        rectilinear_image_tensor = F.grid_sample(
            image.permute(0, 3, 1, 2),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        image_out = rectilinear_image_tensor.permute(0, 2, 3, 1)

        return (image_out,)


def projection_xyz(x, y, d, near, far, rot):
    """We project from 0,0,0 into the z-axis

    We use the OpenGL coordinate system, so x is right, y is up, z is forward

    rot is in radians"""
    assert x.shape == y.shape
    assert x.shape == d.shape
    A = far
    B = near

    z = far * near / (near + d * (far - near))
    x *= z
    y *= z

    return (
        x,
        y * math.cos(rot) - z * math.sin(rot),
        y * math.sin(rot) + z * math.cos(rot),
    )


def fov_to_unit_points(shape, fov):
    """Convert a depth image to a set of points in the unit square"""
    y_nodes, x_nodes = shape
    aspect = x_nodes / y_nodes
    hypotenuse = math.sqrt(1 + aspect**2)
    # length of line between center and edge
    fov_line = math.tan(math.radians(fov) / 2)

    # scale for x_max, y_max
    scale = fov_line / hypotenuse

    x_max = scale
    y_max = scale / aspect

    x_coords = np.linspace(-x_max, x_max, x_nodes)
    y_coords = np.linspace(y_max, -y_max, y_nodes)

    x_range, y_range = np.meshgrid(x_coords, y_coords)

    return x_range, y_range


def compute_points(
    depth_img,  # 2D image
    near,
    far,
    rot_x=0.0,  # degrees
    fov=60,
):
    """Compute the x, y and z point in world space (camera 0,0,0) for the image, based on depth"""

    assert isinstance(depth_img.shape, tuple) and len(depth_img.shape) == 2

    assert isinstance(depth_img, torch.Tensor)

    rot = rot_x

    depth_img = depth_img.numpy()

    x_range, y_range = fov_to_unit_points(depth_img.shape, fov)

    vertexes3 = np.stack(
        projection_xyz(
            x_range,
            y_range,
            depth_img,
            near,
            far,
            np.deg2rad(rot),
        ),
        axis=-1,
    )

    return vertexes3


def compute_normals(points):
    """
    Take a point cloud, and find the normal for each point.
    """
    H, W = points.shape[:2]
    patch = (slice(0, 4), slice(0, 4), slice(None, None))

    # print(H, W)
    # print("points", points[patch])

    start = time.time()

    vH = points[:, 1:, :] - points[:, :-1, :]
    vH = np.concatenate((vH[:, :1, :], vH, vH[:, -1:, :]), axis=1)
    vV = points[1:, :, :] - points[:-1, :, :]
    vV = np.concatenate((vV[:1, :, :], vV, vV[-1:, :, :]), axis=0)

    cTL = np.cross(vH[:, 1:, :], vV[1:, :, :])
    cBR = np.cross(vH[:, :-1, :], vV[:-1, :, :])

    cross = cTL + cBR
    mag = np.linalg.norm(cross, axis=2)
    return cross / mag[:, :, np.newaxis]


def compute_points_cosine(points, normals):
    """
    Take points and normals, and compute the cosine of the angle between the normal and the "camera".

    The result, for depth images, is the range 0..1.
    """
    return np.einsum("ijk,ijk->ij", -points, normals) / np.linalg.norm(points, axis=2)


class DisparityToDepthView:
    """
    Map disparity to XYZ depth points
    """

    def __init__(self):
        self.counter = 0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "disparity": ("MASK",),
                "near": (
                    "FLOAT",
                    {
                        "default": 2,
                        "max": 100,  # Maximum value
                        "min": 0,  # Minimum value
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "far": (
                    "FLOAT",
                    {
                        "default": 5,
                        "max": 1000,  # Maximum value
                        "min": 1,  # Minimum value
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "field_of_view": (
                    "FLOAT",
                    {
                        "default": 65,
                        "max": 150,  # Maximum value
                        "min": 30,  # Minimum value
                        "step": 5,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "tilt": (
                    "FLOAT",
                    {
                        "default": 0,
                        "max": 90,  # Maximum value
                        "min": -90,  # Minimum value
                        "step": 5,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
            },
        }

    #    RETURN_TYPES = ("IMAGE",)

    OUTPUT_NODE = True
    RETURN_TYPES = ("DEPTH_VIEW", "NORMALS", "MASK")

    FUNCTION = "select"
    CATEGORY = "image/transform"

    def select(self, disparity, near, far, field_of_view, tilt):

        points = compute_points(disparity[0, :, :], near, far, tilt, fov=field_of_view)

        normals = compute_normals(points)
        cosines = compute_points_cosine(points, normals)

        points = torch.from_numpy(points).unsqueeze(0)
        normals = torch.from_numpy(normals).unsqueeze(0)
        cosines = torch.from_numpy(cosines).unsqueeze(0)

        return (points, normals, cosines)


def orbit_to_xyz(elevation, azimuth):
    cx = np.cos(math.radians(elevation)) * np.sin(math.radians(azimuth))
    cy = np.sin(math.radians(elevation))
    cz = -np.cos(math.radians(elevation)) * np.cos(math.radians(azimuth))
    return cx, cy, cz


class DepthViewToIsometric:
    """
    Map depth to mesh:
    """

    def __init__(self):
        self.counter = 0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depth_view": ("DEPTH_VIEW",),
                "albedo": ("IMAGE",),
                "elevation": (
                    "INT",
                    {
                        "default": 15,
                        "max": 90,  # Maximum value
                        "min": -90,  # Minimum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "azimuth": (
                    "INT",
                    {
                        "default": 30,
                        "max": 180,  # Maximum value
                        "min": -180,  # Minimum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "camera": ("BOOLEAN", {"default": False}),
                "axis": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "normals": ("NORMALS", {"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "render"
    CATEGORY = "point_cloud/render"

    def render(self, depth_view, albedo, elevation, azimuth, camera, axis, **kwargs):

        #        print("render", depth_view.shape, albedo.shape, kwargs)

        depth_viewx = depth_view.numpy()
        albedox = albedo.numpy()
        normals = kwargs.get("normals", None)

        assert (
            depth_viewx.shape[0:3] == albedox.shape[0:3]
        ), f"Depth and albedo must have the same dimensions, {depth_viewx.shape} != {albedox.shape}"

        if normals is not None:
            assert (
                depth_viewx.shape[0:3] == normals.shape[0:3]
            ), f"Depth and normals must have the same dimensions, {depth_viewx.shape} != {normals.shape}"

        # Create a 4x4x4 grid of points
        x = depth_viewx[0, :, :, 0].flatten()
        y = depth_viewx[0, :, :, 1].flatten()
        z = depth_viewx[0, :, :, 2].flatten()
        colors = albedox.flatten().reshape(-1, 3)

        scale = 512 / math.sqrt(len(colors))

        if normals is not None:
            cz = -np.cos(math.radians(elevation)) * np.cos(math.radians(azimuth))
            cx = np.cos(math.radians(elevation)) * np.sin(math.radians(azimuth))
            cy = np.sin(math.radians(elevation))

            print(
                "cx", cx, "cy", cy, "cz", cz, "elevation", elevation, "azimuth", azimuth
            )

            camera_normal = np.array([cx, cy, cz])

            normals = normals.numpy().flatten().reshape(-1, 3)

            print("normals", normals.shape)

            normals = np.dot(normals, camera_normal)

            mask = normals > 0.0

            x = x[mask]
            y = y[mask]
            z = z[mask]
            colors = colors[mask]

        matplotlib.use("Agg")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the points
        ax.scatter(z, x, y, c=colors, s=scale, marker="o", edgecolors="none", alpha=0.2)
        ax.scatter(
            z, x, y, c=colors, s=scale / 3, marker="o", edgecolors="none", alpha=0.5
        )
        ax.scatter(
            z, x, y, c=colors, s=scale / 9, marker="o", edgecolors="none", alpha=0.9
        )

        if camera:
            ax.scatter(
                [0],
                [0],
                [0],
                color="yellow",
                s=10,
                marker="o",
                edgecolors="black",
            )

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range, z_range)

        # Determine midpoints
        x_mid = np.mean(x_limits)
        y_mid = np.mean(y_limits)
        z_mid = np.mean(z_limits)

        # Set limits around the midpoints with equal range
        ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
        ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
        ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])

        ax.set_xlim(ax.get_xlim()[::-1])

        ax.view_init(
            elev=elevation, azim=azimuth
        )  # Adjust the elevation and azimuthal angles as needed
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_position([0, 0, 1, 1])  # [left, bottom, width, height]

        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        # Add option to disable axis
        if not axis:
            ax.set_axis_off()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.savefig(tmp_file, bbox_inches="tight", pad_inches=0, dpi=300)
            image_pil = Image.open(tmp_file)
            image_np = (
                np.array(image_pil.convert("RGB")).copy().astype(np.float32) / 255
            )

        image = torch.from_numpy(image_np).unsqueeze(0)

        return (image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageChannelSelect": ImageChannelSelect,
    "MaskChannelSelect": MaskChannelSelect,
    "EquirectangularToRectilinear": EquirectangularToRectilinear,
    "ResizeDown": ResizeDown,
    "DisparityToDepthView": DisparityToDepthView,
    "DepthViewToIsometric": DepthViewToIsometric,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageChannelSelect": "Image Channel Select",
    "MaskChannelSelect": "Mask Channel Select",
    "EquirectangularToRectilinear": "Equirectangular to Rectilinear",
    "ResizeDown": "Resize Down",
    "DisparityToDepthView": "Disparity to Depth View",
    "DepthViewToIsometric": "Depth View to Isometric",
}
