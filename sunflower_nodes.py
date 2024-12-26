import numpy as np

from dataclasses import dataclass
import torch
import torch.nn.functional as F

import nodes


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

    OUTPUT_NODE = True
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


class ChannelSelect:
    """
    Select a channel from a 3D feed
    """

    def __init__(self):
        self.counter = 0

    channel_types = ["Left", "Right", "Top", "Bottom", "Left:Right", "Top:Bottom"]

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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ChannelSelect": ChannelSelect,
    "EquirectangularToRectilinear": EquirectangularToRectilinear,
    "ResizeDown": ResizeDown,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ChannelSelect": "Stereo to Mono Channel Select",
    "EquirectangularToRectilinear": "Equirectangular to Rectilinear",
    "ResizeDown": "Resize Down",
}
