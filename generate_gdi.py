#!/usr/bin/env python3
"""Generate Generative Defect Isolation (GDI) samples from VIA annotations.

For each source image containing more than one unique annotated defect type, GDI:

1. isolates each target defect whose annotated area is strictly greater than the
   configured area threshold by inpainting all other annotated defects; and
2. generates one additional ``No_Defect`` image by inpainting all annotated
   defects, but only when that source image produced at least one isolated-defect
   sample.

The implementation uses the LaMa integration provided by Inpaint-Anything.

When visualization output is enabled, the script writes one composite image for
every processed multi-defect source image. The composite is organized by rows:

- first row: original source image, original image with all annotated defects
  outlined in class-specific colors and an external legend. The legend reports
  each class, its annotated area percentage and whether it is above or at/below
  the configured isolation threshold;
- subsequent rows: only classes for which an isolated GDI image was actually
  generated. Each row contains the generated image, the generated image with the
  preserved defect class highlighted and the class name; and
- when at least one isolated-defect sample was generated, a final ``No_Defect``
  row containing only the generated defect-free image.

Classes that fail the area criterion are documented in the first-row legend and
do not receive placeholder rows. The legend and class-information panels sit
outside the EL images, so text never covers image content. The composite is a
visualization only: generated GDI training images are always saved separately at
their native resolution.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Dataset classes
# -----------------------------------------------------------------------------

DEFECT_CLASSES = [
    "Contact_BeltMarks",
    "Contact_Corrosion",
    "Contact_FrontGridInterruption",
    "Contact_NearSolderPad",
    "Crack_Closed",
    "Crack_Isolated",
    "Crack_Resistive",
    "Interconnect_BrightSpot",
    "Interconnect_Disconnected",
    "Interconnect_HighlyResistive",
    "No_Defect",
    "Unknown",
]

ANNOTATED_DEFECT_CLASSES = [c for c in DEFECT_CLASSES if c != "No_Defect"]


# -----------------------------------------------------------------------------
# Visualization style
# -----------------------------------------------------------------------------

# Controls annotation-outline thickness in composite visualizations.
VIZ_BOUNDARY_THICKNESS = 1

VIZ_IMAGE_HEIGHT = 700
VIZ_PANEL_MIN_WIDTH = 560
VIZ_OUTER_MARGIN = 24
VIZ_GRID_GAP = 18
VIZ_CAPTION_HEIGHT = 84
VIZ_CAPTION_FONT_SCALE = 1.05
VIZ_CAPTION_FONT_THICKNESS = 2
VIZ_INFO_TITLE_FONT_SCALE = 1.22
VIZ_INFO_TEXT_FONT_SCALE = 0.98
VIZ_INFO_FONT_THICKNESS = 2
VIZ_SWATCH_SIZE = 34
VIZ_BACKGROUND_COLOR: tuple[int, int, int] = (255, 255, 255)
VIZ_CAPTION_COLOR: tuple[int, int, int] = (20, 20, 20)
VIZ_MUTED_COLOR: tuple[int, int, int] = (110, 110, 110)

# Class colors used consistently across annotation overlays and legends.
VIZ_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "Contact_BeltMarks": (230, 25, 75),
    "Contact_Corrosion": (60, 180, 75),
    "Contact_FrontGridInterruption": (0, 130, 200),
    "Contact_NearSolderPad": (245, 130, 48),
    "Crack_Closed": (145, 30, 180),
    "Crack_Isolated": (70, 240, 240),
    "Crack_Resistive": (240, 50, 230),
    "Interconnect_BrightSpot": (210, 245, 60),
    "Interconnect_Disconnected": (250, 190, 212),
    "Interconnect_HighlyResistive": (0, 128, 128),
    "Unknown": (128, 128, 128),
}


# -----------------------------------------------------------------------------
# Inpaint-Anything / LaMa integration
# -----------------------------------------------------------------------------


def load_lama_inpainter(inpaint_anything_dir: Path) -> Callable[..., np.ndarray]:
    """Load ``inpaint_img_with_lama`` from a local Inpaint-Anything checkout."""
    root = inpaint_anything_dir.expanduser().resolve()
    module_file = root / "lama_inpaint.py"
    if not module_file.exists():
        raise FileNotFoundError(
            f"Could not find {module_file}. Pass the Inpaint-Anything repository "
            "root with --inpaint-anything-dir."
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        from lama_inpaint import inpaint_img_with_lama  # type: ignore
    except Exception as exc:  # pragma: no cover - external dependency
        raise RuntimeError(
            "Failed to import LaMa from Inpaint-Anything. Install the upstream "
            "dependencies in the active environment."
        ) from exc

    return inpaint_img_with_lama


# -----------------------------------------------------------------------------
# Basic image and VIA helpers
# -----------------------------------------------------------------------------


def load_image(path: Path) -> np.ndarray:
    """Load an image as an RGB NumPy array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def save_image(image: np.ndarray, path: Path) -> None:
    """Save an RGB NumPy image, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _json_object(value: Any) -> dict[str, Any]:
    """Parse a VIA JSON cell; empty/NaN values become an empty dictionary."""
    if value is None:
        return {}
    if isinstance(value, float) and math.isnan(value):
        return {}
    if isinstance(value, dict):
        return value

    text = str(value).strip()
    if not text or text == "{}" or text.lower() == "nan":
        return {}

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed


def parse_shapes(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Parse VIA shape and region JSON into a unified list of annotations."""
    shapes: list[dict[str, Any]] = []

    for _, row in rows.iterrows():
        shape = _json_object(row.get("region_shape_attributes"))
        if not shape:
            continue

        region = _json_object(row.get("region_attributes"))
        defect_class = region.get("Defect_Class")
        if not defect_class:
            continue

        item = dict(shape)
        item["Defect_Class"] = str(defect_class)
        shapes.append(item)

    return shapes


def _ellipse_angle_degrees(shape: dict[str, Any]) -> float:
    """Return an ellipse rotation angle in degrees for common VIA schemas."""
    if "angle" in shape:
        return float(shape["angle"])
    if "theta" in shape:
        # VIA commonly stores theta in radians.
        return float(np.degrees(float(shape["theta"])))
    return 0.0


# -----------------------------------------------------------------------------
# Mask creation
# -----------------------------------------------------------------------------


def _draw_shape_on_mask(mask: np.ndarray, shape: dict[str, Any]) -> None:
    """Draw one VIA annotation onto ``mask`` in place."""
    name = shape.get("name")

    if name == "rect":
        x = int(round(float(shape["x"])))
        y = int(round(float(shape["y"])))
        w = int(round(float(shape["width"])))
        h = int(round(float(shape["height"])))
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    elif name == "polygon":
        pts = np.asarray(
            list(zip(shape["all_points_x"], shape["all_points_y"])),
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 255)

    elif name == "circle":
        center = (
            int(round(float(shape["cx"]))),
            int(round(float(shape["cy"]))),
        )
        radius = int(round(float(shape["r"])))
        cv2.circle(mask, center, radius, 255, -1)

    elif name == "ellipse":
        center = (
            int(round(float(shape["cx"]))),
            int(round(float(shape["cy"]))),
        )
        axes = (
            int(round(float(shape["rx"]))),
            int(round(float(shape["ry"]))),
        )
        cv2.ellipse(
            mask,
            center,
            axes,
            _ellipse_angle_degrees(shape),
            0,
            360,
            255,
            -1,
        )

    else:
        raise ValueError(f"Unsupported VIA shape type: {name!r}")


def create_mask_for_defect_type(
    image_shape: tuple[int, ...],
    shapes: Iterable[dict[str, Any]],
    defect_type: str,
) -> np.ndarray:
    """Create a mask covering annotations belonging to one defect class."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for shape in shapes:
        if shape.get("Defect_Class") == defect_type:
            _draw_shape_on_mask(mask, shape)
    return mask


def create_mask_except_defect_type(
    image_shape: tuple[int, ...],
    shapes: Iterable[dict[str, Any]],
    keep_defect_type: str,
) -> np.ndarray:
    """Create a mask covering every annotation except the target defect class."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for shape in shapes:
        if shape.get("Defect_Class") != keep_defect_type:
            _draw_shape_on_mask(mask, shape)
    return mask


def create_all_defects_mask(
    image_shape: tuple[int, ...],
    shapes: Iterable[dict[str, Any]],
) -> np.ndarray:
    """Create the combined mask used to generate a ``No_Defect`` sample."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for shape in shapes:
        _draw_shape_on_mask(mask, shape)
    return mask


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Dilate a binary mask with a square kernel; ``0`` disables dilation."""
    if kernel_size <= 0:
        return mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def area_percentage(mask: np.ndarray) -> float:
    """Return the percentage of image pixels covered by a binary mask."""
    total_pixels = int(mask.shape[0] * mask.shape[1])
    if total_pixels == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(total_pixels) * 100.0


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------


def _resize_to_visualization_height(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Resize a visualization copy to the fixed image-panel height."""
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions: {image.shape}")

    scale = VIZ_IMAGE_HEIGHT / float(height)
    new_width = max(1, int(round(width * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_width, VIZ_IMAGE_HEIGHT), interpolation=interpolation)
    return resized, scale


def _scaled_shape(shape: dict[str, Any], scale: float) -> dict[str, Any]:
    """Scale one VIA annotation for a resized visualization image."""
    out = dict(shape)
    name = shape.get("name")

    if name == "rect":
        for key in ("x", "y", "width", "height"):
            out[key] = float(shape[key]) * scale
    elif name == "polygon":
        out["all_points_x"] = [float(x) * scale for x in shape["all_points_x"]]
        out["all_points_y"] = [float(y) * scale for y in shape["all_points_y"]]
    elif name == "circle":
        for key in ("cx", "cy", "r"):
            out[key] = float(shape[key]) * scale
    elif name == "ellipse":
        for key in ("cx", "cy", "rx", "ry"):
            out[key] = float(shape[key]) * scale
    else:
        raise ValueError(f"Unsupported VIA shape type: {name!r}")

    return out


def _draw_shape_boundary(
    image: np.ndarray,
    shape: dict[str, Any],
    color: tuple[int, int, int],
    thickness: int = VIZ_BOUNDARY_THICKNESS,
) -> None:
    """Draw one annotation outline directly onto an RGB image."""
    name = shape.get("name")

    if name == "rect":
        x = int(round(float(shape["x"])))
        y = int(round(float(shape["y"])))
        w = int(round(float(shape["width"])))
        h = int(round(float(shape["height"])))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    elif name == "polygon":
        pts = np.asarray(
            list(zip(shape["all_points_x"], shape["all_points_y"])),
            dtype=np.int32,
        )
        cv2.polylines(image, [pts], True, color, thickness, lineType=cv2.LINE_AA)

    elif name == "circle":
        center = (
            int(round(float(shape["cx"]))),
            int(round(float(shape["cy"]))),
        )
        radius = int(round(float(shape["r"])))
        cv2.circle(image, center, radius, color, thickness, lineType=cv2.LINE_AA)

    elif name == "ellipse":
        center = (
            int(round(float(shape["cx"]))),
            int(round(float(shape["cy"]))),
        )
        axes = (
            int(round(float(shape["rx"]))),
            int(round(float(shape["ry"]))),
        )
        cv2.ellipse(
            image,
            center,
            axes,
            _ellipse_angle_degrees(shape),
            0,
            360,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    else:
        raise ValueError(f"Unsupported VIA shape type: {name!r}")


def _class_color(defect_type: str) -> tuple[int, int, int]:
    """Return the fixed visualization color for one defect class."""
    return VIZ_CLASS_COLORS.get(defect_type, VIZ_MUTED_COLOR)


def _draw_all_boundaries(
    image: np.ndarray,
    shapes: Iterable[dict[str, Any]],
) -> np.ndarray:
    """Draw all annotations using a different fixed color for each class."""
    out = image.copy()
    for shape in shapes:
        defect_type = str(shape.get("Defect_Class", "Unknown"))
        _draw_shape_boundary(out, shape, color=_class_color(defect_type))
    return out


def _draw_one_class_boundaries(
    image: np.ndarray,
    shapes: Iterable[dict[str, Any]],
    defect_type: str,
) -> np.ndarray:
    """Draw every annotation belonging to one remaining defect class."""
    out = image.copy()
    color = _class_color(defect_type)
    for shape in shapes:
        if shape.get("Defect_Class") == defect_type:
            _draw_shape_boundary(out, shape, color=color)
    return out


def _fit_image_to_panel(image: np.ndarray, panel_width: int) -> np.ndarray:
    """Place a visualization image in a fixed-size white image panel."""
    resized, _ = _resize_to_visualization_height(image)

    if resized.shape[1] > panel_width:
        scale = panel_width / float(resized.shape[1])
        new_height = max(1, int(round(resized.shape[0] * scale)))
        resized = cv2.resize(
            resized,
            (panel_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    panel = np.full(
        (VIZ_IMAGE_HEIGHT, panel_width, 3),
        VIZ_BACKGROUND_COLOR,
        dtype=np.uint8,
    )
    y = max(0, (VIZ_IMAGE_HEIGHT - resized.shape[0]) // 2)
    x = max(0, (panel_width - resized.shape[1]) // 2)
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _make_captioned_panel(
    image: np.ndarray,
    caption: str,
    panel_width: int,
) -> np.ndarray:
    """Create one image tile with a fixed-size caption strip below it."""
    image_panel = _fit_image_to_panel(image, panel_width)
    tile = np.full(
        (VIZ_IMAGE_HEIGHT + VIZ_CAPTION_HEIGHT, panel_width, 3),
        VIZ_BACKGROUND_COLOR,
        dtype=np.uint8,
    )
    tile[:VIZ_IMAGE_HEIGHT] = image_panel

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        caption,
        font,
        VIZ_CAPTION_FONT_SCALE,
        VIZ_CAPTION_FONT_THICKNESS,
    )
    text_x = max(10, (panel_width - text_width) // 2)
    strip_top = VIZ_IMAGE_HEIGHT
    text_y = strip_top + (VIZ_CAPTION_HEIGHT + text_height - baseline) // 2
    cv2.putText(
        tile,
        caption,
        (text_x, text_y),
        font,
        VIZ_CAPTION_FONT_SCALE,
        VIZ_CAPTION_COLOR,
        VIZ_CAPTION_FONT_THICKNESS,
        lineType=cv2.LINE_AA,
    )
    return tile


def _wrap_label(label: str, panel_width: int) -> list[str]:
    """Wrap a class name without changing the fixed visualization font size."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_width = panel_width - 2 * 46 - VIZ_SWATCH_SIZE
    if cv2.getTextSize(
        label,
        font,
        VIZ_INFO_TEXT_FONT_SCALE,
        VIZ_INFO_FONT_THICKNESS,
    )[0][0] <= max_width:
        return [label]

    # Prefer breaks at underscores so canonical class names remain recognizable.
    parts = label.split("_")
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}_{part}"
        width = cv2.getTextSize(
            candidate,
            font,
            VIZ_INFO_TEXT_FONT_SCALE,
            VIZ_INFO_FONT_THICKNESS,
        )[0][0]
        if current and width > max_width:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _make_defect_legend_panel(
    class_results: Sequence[tuple[str, float, np.ndarray | None]],
    area_threshold: float,
    panel_width: int,
) -> np.ndarray:
    """Create the first-row legend with class color, area and threshold status."""
    tile_height = VIZ_IMAGE_HEIGHT + VIZ_CAPTION_HEIGHT
    panel = np.full(
        (tile_height, panel_width, 3),
        VIZ_BACKGROUND_COLOR,
        dtype=np.uint8,
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    left = 46
    y = 82

    cv2.putText(
        panel,
        "Defect legend",
        (left, y),
        font,
        VIZ_INFO_TITLE_FONT_SCALE,
        VIZ_CAPTION_COLOR,
        VIZ_INFO_FONT_THICKNESS,
        lineType=cv2.LINE_AA,
    )
    y += 66

    for defect_type, area_percent, generated_image in class_results:
        color = _class_color(defect_type)
        name_lines = _wrap_label(defect_type, panel_width)
        comparison = ">" if generated_image is not None else "<="
        status_line = (
            f"Area: {area_percent:.2f}% {comparison} {area_threshold:g}% threshold"
        )

        block_height = max(VIZ_SWATCH_SIZE, 44 * len(name_lines)) + 56
        if y + block_height + 14 > tile_height:
            break

        cv2.rectangle(
            panel,
            (left, y + 2),
            (left + VIZ_SWATCH_SIZE, y + 2 + VIZ_SWATCH_SIZE),
            color,
            -1,
        )

        text_x = left + VIZ_SWATCH_SIZE + 18
        text_y = y + 30
        for line in name_lines:
            cv2.putText(
                panel,
                line,
                (text_x, text_y),
                font,
                VIZ_INFO_TEXT_FONT_SCALE,
                VIZ_CAPTION_COLOR,
                VIZ_INFO_FONT_THICKNESS,
                lineType=cv2.LINE_AA,
            )
            text_y += 42

        status_scale = max(0.86, VIZ_INFO_TEXT_FONT_SCALE * 0.90)
        status_thickness = max(1, VIZ_INFO_FONT_THICKNESS - 1)
        max_status_width = panel_width - text_x - 24
        while status_scale > 0.55:
            status_width = cv2.getTextSize(
                status_line,
                font,
                status_scale,
                status_thickness,
            )[0][0]
            if status_width <= max_status_width:
                break
            status_scale -= 0.05

        cv2.putText(
            panel,
            status_line,
            (text_x, text_y + 8),
            font,
            status_scale,
            VIZ_MUTED_COLOR,
            status_thickness,
            lineType=cv2.LINE_AA,
        )
        y += block_height + 14

    return panel


def _make_class_name_panel(class_name: str, panel_width: int) -> np.ndarray:
    """Create a large external panel containing only the defect class name."""
    tile_height = VIZ_IMAGE_HEIGHT + VIZ_CAPTION_HEIGHT
    panel = np.full(
        (tile_height, panel_width, 3),
        VIZ_BACKGROUND_COLOR,
        dtype=np.uint8,
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = _wrap_label(class_name, panel_width)
    line_height = 56
    total_height = line_height * len(lines)
    start_y = max(60, (tile_height - total_height) // 2 + 24)
    for idx, line in enumerate(lines):
        (text_width, text_height), _ = cv2.getTextSize(
            line,
            font,
            VIZ_INFO_TITLE_FONT_SCALE,
            VIZ_INFO_FONT_THICKNESS,
        )
        x = max(20, (panel_width - text_width) // 2)
        y = start_y + idx * line_height
        cv2.putText(
            panel,
            line,
            (x, y),
            font,
            VIZ_INFO_TITLE_FONT_SCALE,
            VIZ_CAPTION_COLOR,
            VIZ_INFO_FONT_THICKNESS,
            lineType=cv2.LINE_AA,
        )
    return panel



def create_gdi_composite_visualization(
    original_image: np.ndarray,
    shapes: list[dict[str, Any]],
    class_results: Sequence[tuple[str, float, np.ndarray | None]],
    area_threshold: float,
    no_defect_image: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """Create one row-structured GDI visualization for a source image.

    Layout (three columns per row):

    Row 1
        Original | Original + class-colored annotations | Legend with area/status

    Subsequent rows
        Generated | Generated + target-class annotations | Class name

    Only classes that actually produced an isolated GDI sample appear after the
    first row. Classes filtered by the area threshold are reported only in the
    legend. An optional final ``No_Defect`` row contains one image cell.

    Returns ``(composite, populated_cell_count)``.
    """
    resized_original, original_scale = _resize_to_visualization_height(original_image)
    panel_width = max(VIZ_PANEL_MIN_WIDTH, resized_original.shape[1])
    scaled_shapes = [_scaled_shape(shape, original_scale) for shape in shapes]

    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    populated_cells = 0

    original_annotated = _draw_all_boundaries(resized_original, scaled_shapes)
    legend = _make_defect_legend_panel(
        class_results=class_results,
        area_threshold=area_threshold,
        panel_width=panel_width,
    )
    rows.append(
        (
            _make_captioned_panel(original_image, "Original", panel_width),
            _make_captioned_panel(
                original_annotated,
                "Original + annotations",
                panel_width,
            ),
            legend,
        )
    )
    populated_cells += 3

    # Only generated isolated-defect samples receive rows. Threshold-filtered
    # classes are still visible in the first-row legend with their area/status.
    for defect_type, _area_percent, generated_image in class_results:
        if generated_image is None:
            continue

        resized_generated, generated_scale = _resize_to_visualization_height(
            generated_image
        )
        generated_shapes = [
            _scaled_shape(shape, generated_scale) for shape in shapes
        ]
        generated_annotated = _draw_one_class_boundaries(
            resized_generated,
            generated_shapes,
            defect_type,
        )
        rows.append(
            (
                _make_captioned_panel(
                    generated_image,
                    "Generated",
                    panel_width,
                ),
                _make_captioned_panel(
                    generated_annotated,
                    "Highlighted defect(s)",
                    panel_width,
                ),
                _make_class_name_panel(defect_type, panel_width),
            )
        )
        populated_cells += 3

    # No_Defect is generated only if this source produced at least one isolated
    # class. It occupies one image cell because there is nothing left to annotate.
    if no_defect_image is not None:
        blank_tile = np.full(
            (VIZ_IMAGE_HEIGHT + VIZ_CAPTION_HEIGHT, panel_width, 3),
            VIZ_BACKGROUND_COLOR,
            dtype=np.uint8,
        )
        rows.append(
            (
                _make_captioned_panel(no_defect_image, "No_Defect", panel_width),
                blank_tile.copy(),
                blank_tile.copy(),
            )
        )
        populated_cells += 1

    tile_height = VIZ_IMAGE_HEIGHT + VIZ_CAPTION_HEIGHT
    columns = 3
    canvas_width = (
        VIZ_OUTER_MARGIN * 2
        + columns * panel_width
        + (columns - 1) * VIZ_GRID_GAP
    )
    canvas_height = (
        VIZ_OUTER_MARGIN * 2
        + len(rows) * tile_height
        + (len(rows) - 1) * VIZ_GRID_GAP
    )
    composite = np.full(
        (canvas_height, canvas_width, 3),
        VIZ_BACKGROUND_COLOR,
        dtype=np.uint8,
    )

    for row_index, row_tiles in enumerate(rows):
        y = VIZ_OUTER_MARGIN + row_index * (tile_height + VIZ_GRID_GAP)
        for col_index, tile in enumerate(row_tiles):
            x = VIZ_OUTER_MARGIN + col_index * (panel_width + VIZ_GRID_GAP)
            composite[y : y + tile_height, x : x + panel_width] = tile

    return composite, populated_cells


# -----------------------------------------------------------------------------
# CSV rows and validation
# -----------------------------------------------------------------------------


def make_label_row(relative_filename: str, label: str) -> dict[str, Any]:
    """Create a one-hot output-label row."""
    row: dict[str, Any] = {"filename": relative_filename}
    for defect_class in DEFECT_CLASSES:
        row[f"Class_{defect_class}"] = int(defect_class == label)
    return row


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate input paths and user-configurable GDI parameters."""
    for path_arg in (
        "csv_file",
        "img_dir",
        "lama_config",
        "lama_ckpt",
        "inpaint_anything_dir",
    ):
        path = Path(getattr(args, path_arg)).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"{path_arg.replace('_', '-')}: {path} does not exist"
            )

    if args.dilate_kernel_size < 0:
        raise ValueError("--dilate-kernel-size must be >= 0")
    if not 0 <= args.area_threshold <= 100:
        raise ValueError("--area-threshold must be between 0 and 100")


# -----------------------------------------------------------------------------
# Main GDI processing
# -----------------------------------------------------------------------------


def process_images(args: argparse.Namespace) -> dict[str, Any]:
    """Run the GDI generation pipeline and return summary statistics."""
    validate_inputs(args)
    started = time.time()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")

    csv_file = Path(args.csv_file).expanduser().resolve()
    image_dir = Path(args.img_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    visualizations_dir = (
        Path(args.visualizations_dir).expanduser().resolve()
        if args.visualizations_dir
        else None
    )
    lama_config = str(Path(args.lama_config).expanduser().resolve())
    lama_ckpt = str(Path(args.lama_ckpt).expanduser().resolve())

    output_dir.mkdir(parents=True, exist_ok=True)
    if visualizations_dir is not None:
        visualizations_dir.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_csv(csv_file)
    required_columns = {"filename", "region_shape_attributes", "region_attributes"}
    missing_columns = required_columns.difference(annotations.columns)
    if missing_columns:
        raise ValueError(
            f"Annotation CSV is missing required columns: {sorted(missing_columns)}"
        )

    inpaint_img_with_lama = load_lama_inpainter(Path(args.inpaint_anything_dir))

    annotations = annotations.copy()
    annotations["filename"] = annotations["filename"].astype(str)
    grouped = {
        name: group for name, group in annotations.groupby("filename", sort=False)
    }
    filenames = list(grouped.keys())

    label_rows: list[dict[str, Any]] = []
    area_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    isolated_per_class: Counter[str] = Counter()

    source_images_found = 0
    source_images_missing = 0
    source_no_defect = 0
    source_single_defect = 0
    source_multi_defect = 0
    ignored_annotation_count = 0
    target_candidates = 0
    targets_below_or_equal_threshold = 0
    generated_isolated = 0
    generated_no_defect = 0
    multi_defect_with_eligible_target = 0
    multi_defect_without_eligible_target = 0
    composite_visualizations = 0
    visualization_panels = 0

    for image_filename in tqdm(filenames, desc="Generating GDI samples"):
        source_path = image_dir / image_filename
        if not source_path.is_file():
            source_images_missing += 1
            tqdm.write(f"WARNING: image not found, skipping: {source_path}")
            continue

        source_images_found += 1
        shapes = parse_shapes(grouped[image_filename])

        recognized_shapes: list[dict[str, Any]] = []
        for shape in shapes:
            defect_class = shape.get("Defect_Class")
            if defect_class in ANNOTATED_DEFECT_CLASSES:
                recognized_shapes.append(shape)
            else:
                ignored_annotation_count += 1
                tqdm.write(
                    f"WARNING: '{image_filename}' contains unrecognized class "
                    f"{defect_class!r}; annotation ignored."
                )
        shapes = recognized_shapes

        defect_types = {
            str(shape["Defect_Class"])
            for shape in shapes
            if shape.get("Defect_Class")
        }

        if len(defect_types) == 0:
            source_no_defect += 1
            continue
        if len(defect_types) == 1:
            source_single_defect += 1
            continue

        # GDI is applied only to source images with more than one unique defect.
        source_multi_defect += 1
        image = load_image(source_path)
        image_stem = Path(image_filename).stem

        # Compute annotated class areas before any dilation.
        area_row: dict[str, Any] = {
            "filename": image_filename,
            "Total_Image_Area": int(image.shape[0] * image.shape[1]),
        }
        defect_area_pct: dict[str, float] = {}
        for defect_class in ANNOTATED_DEFECT_CLASSES:
            mask = create_mask_for_defect_type(image.shape, shapes, defect_class)
            pct = area_percentage(mask)
            defect_area_pct[defect_class] = pct
            area_row[f"Area_{defect_class}"] = pct
        area_rows.append(area_row)

        # Each tuple stores (class, area %, generated image or None).
        class_results_for_visualization: list[tuple[str, float, np.ndarray | None]] = []
        source_generated_isolated = 0

        # Isolate every target defect that is strictly above the area threshold.
        for defect_type in sorted(defect_types):
            target_candidates += 1
            pct = defect_area_pct[defect_type]
            if pct <= args.area_threshold:
                targets_below_or_equal_threshold += 1
                if visualizations_dir is not None:
                    class_results_for_visualization.append((defect_type, pct, None))
                continue

            removal_mask = create_mask_except_defect_type(
                image.shape,
                shapes,
                defect_type,
            )
            removal_mask = dilate_mask(removal_mask, args.dilate_kernel_size)

            generated = inpaint_img_with_lama(
                image,
                removal_mask,
                lama_config,
                lama_ckpt,
                device=device,
            )
            generated = np.asarray(generated, dtype=np.uint8)

            output_name = f"{image_stem}_{defect_type}.jpg"
            relative_path = Path(defect_type) / output_name
            save_image(generated, output_dir / relative_path)

            label_rows.append(make_label_row(relative_path.as_posix(), defect_type))
            manifest_rows.append(
                {
                    "filename": relative_path.as_posix(),
                    "source_filename": image_filename,
                    "label": defect_type,
                    "generation_type": "isolated_defect",
                    "target_area_percent": pct,
                    "source_unique_defects": len(defect_types),
                }
            )

            generated_isolated += 1
            source_generated_isolated += 1
            isolated_per_class[defect_type] += 1

            if visualizations_dir is not None:
                class_results_for_visualization.append((defect_type, pct, generated))

        # Generate one No_Defect sample when at least one isolated class was created.
        no_defect_image: np.ndarray | None = None
        if source_generated_isolated > 0:
            multi_defect_with_eligible_target += 1
            all_defects_mask = create_all_defects_mask(image.shape, shapes)
            all_defects_mask = dilate_mask(all_defects_mask, args.dilate_kernel_size)

            no_defect_image = inpaint_img_with_lama(
                image,
                all_defects_mask,
                lama_config,
                lama_ckpt,
                device=device,
            )
            no_defect_image = np.asarray(no_defect_image, dtype=np.uint8)

            no_defect_name = f"{image_stem}_No_Defect.jpg"
            no_defect_relative = Path("No_Defect") / no_defect_name
            save_image(no_defect_image, output_dir / no_defect_relative)

            label_rows.append(
                make_label_row(no_defect_relative.as_posix(), "No_Defect")
            )
            manifest_rows.append(
                {
                    "filename": no_defect_relative.as_posix(),
                    "source_filename": image_filename,
                    "label": "No_Defect",
                    "generation_type": "all_defects_removed",
                    "target_area_percent": "",
                    "source_unique_defects": len(defect_types),
                }
            )
            generated_no_defect += 1
        else:
            multi_defect_without_eligible_target += 1

        # Write one composite visualization for the processed source image.
        if visualizations_dir is not None:
            composite, panel_count = create_gdi_composite_visualization(
                original_image=image,
                shapes=shapes,
                class_results=class_results_for_visualization,
                area_threshold=args.area_threshold,
                no_defect_image=no_defect_image,
            )
            save_image(
                composite,
                visualizations_dir / f"{image_stem}_gdi_summary.png",
            )
            composite_visualizations += 1
            visualization_panels += panel_count

    # ------------------------------------------------------------------
    # Save CSV outputs
    # ------------------------------------------------------------------
    labels_path = output_dir / "inpainted_single_label_images.csv"
    areas_path = output_dir / "defect_areas.csv"
    manifest_path = output_dir / "gdi_manifest.csv"
    stats_path = output_dir / "generation_stats.json"

    label_columns = ["filename"] + [f"Class_{c}" for c in DEFECT_CLASSES]
    pd.DataFrame(label_rows, columns=label_columns).to_csv(labels_path, index=False)

    area_columns = ["filename", "Total_Image_Area"] + [
        f"Area_{c}" for c in ANNOTATED_DEFECT_CLASSES
    ]
    pd.DataFrame(area_rows, columns=area_columns).to_csv(areas_path, index=False)

    manifest_columns = [
        "filename",
        "source_filename",
        "label",
        "generation_type",
        "target_area_percent",
        "source_unique_defects",
    ]
    pd.DataFrame(manifest_rows, columns=manifest_columns).to_csv(
        manifest_path,
        index=False,
    )

    elapsed = time.time() - started
    total_generated = generated_isolated + generated_no_defect
    avg_seconds_per_multi_source = (
        elapsed / source_multi_defect if source_multi_defect else 0.0
    )

    stats: dict[str, Any] = {
        "device": device,
        "area_threshold_percent": args.area_threshold,
        "area_rule": "strictly_greater_than",
        "dilation_kernel_size": args.dilate_kernel_size,
        "source_images_in_annotation_csv": len(filenames),
        "source_images_found": source_images_found,
        "source_images_missing": source_images_missing,
        "source_images_no_defect": source_no_defect,
        "source_images_single_defect": source_single_defect,
        "source_images_multi_defect": source_multi_defect,
        "multi_defect_sources_with_at_least_one_eligible_target": multi_defect_with_eligible_target,
        "multi_defect_sources_with_no_eligible_targets": multi_defect_without_eligible_target,
        "ignored_unrecognized_annotations": ignored_annotation_count,
        "target_defect_candidates": target_candidates,
        "target_candidates_at_or_below_threshold": targets_below_or_equal_threshold,
        "generated_isolated_defect_images": generated_isolated,
        "generated_no_defect_images": generated_no_defect,
        "generated_total_images": total_generated,
        "generated_isolated_by_class": {
            defect_class: int(isolated_per_class.get(defect_class, 0))
            for defect_class in ANNOTATED_DEFECT_CLASSES
        },
        "composite_visualizations": composite_visualizations,
        "visualization_panels": visualization_panels,
        "elapsed_seconds": elapsed,
        "average_seconds_per_multi_defect_source": avg_seconds_per_multi_source,
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("GDI generation complete")
    print("=" * 72)
    print("Parameters")
    print(f"  Device:                              {device}")
    print(f"  Defect area rule:                    > {args.area_threshold:g}%")
    if args.dilate_kernel_size > 0:
        print(
            f"  Dilation kernel:                     "
            f"{args.dilate_kernel_size} x {args.dilate_kernel_size}"
        )
    else:
        print("  Dilation kernel:                     disabled")

    print("\nSource images")
    print(f"  Listed in annotation CSV:            {len(filenames)}")
    print(f"  Found:                               {source_images_found}")
    print(f"  Missing:                             {source_images_missing}")
    print(f"  No annotated defect:                 {source_no_defect}")
    print(f"  One unique defect:                   {source_single_defect}")
    print(f"  Multi-defect (evaluated by GDI):     {source_multi_defect}")
    print(f"  With >=1 eligible target:            {multi_defect_with_eligible_target}")
    print(f"  With no eligible targets:            {multi_defect_without_eligible_target}")

    print("\nIsolation filtering")
    print(f"  Target-defect candidates:            {target_candidates}")
    print(
        f"  At/below area threshold:             "
        f"{targets_below_or_equal_threshold}"
    )
    print(f"  Above area threshold:                {generated_isolated}")

    print("\nGenerated images")
    print(f"  Isolated-defect images:              {generated_isolated}")
    print(f"  No_Defect images:                    {generated_no_defect}")
    print(
        f"  No_Defect skipped (no eligible):     "
        f"{multi_defect_without_eligible_target}"
    )
    print(f"  Total generated images:              {total_generated}")

    if generated_isolated:
        print("\nIsolated-defect images by class")
        for defect_class in ANNOTATED_DEFECT_CLASSES:
            count = isolated_per_class.get(defect_class, 0)
            if count:
                print(f"  {defect_class:<36} {count:>8}")

    if visualizations_dir is not None:
        print("\nVisualizations")
        print(f"  Composite source summaries:          {composite_visualizations}")
        print(f"  Total panels across composites:      {visualization_panels}")
        print(f"  Directory:                           {visualizations_dir}")

    print("\nRuntime")
    print(f"  Elapsed time:                        {elapsed:.1f} s")
    print(
        f"  Average / multi-defect source:       "
        f"{avg_seconds_per_multi_source:.2f} s"
    )

    print("\nFiles")
    print(f"  Labels CSV:                          {labels_path}")
    print(f"  Manifest CSV:                        {manifest_path}")
    print(f"  Defect areas CSV:                    {areas_path}")
    print(f"  Generation stats:                    {stats_path}")
    print("=" * 72)

    return stats


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate GDI isolated-defect samples from VIA annotations using LaMa "
            "inpainting, plus No_Defect samples only for sources that yield at "
            "least one isolated class."
        )
    )

    parser.add_argument(
        "--csv-file",
        required=True,
        help="Path to the VIA annotation CSV",
    )
    parser.add_argument(
        "--img-dir",
        required=True,
        help="Directory containing source EL images",
    )
    parser.add_argument(
        "--inpaint-anything-dir",
        required=True,
        help="Root of a local geekyutao/Inpaint-Anything checkout",
    )
    parser.add_argument(
        "--lama-config",
        required=True,
        help=(
            "Path to the LaMa prediction config, e.g. "
            "lama/configs/prediction/default.yaml"
        ),
    )
    parser.add_argument(
        "--lama-ckpt",
        required=True,
        help="Path to the LaMa checkpoint directory, e.g. pretrained_models/big-lama",
    )
    parser.add_argument(
        "--output-dir",
        default="./generated_gdi",
        help="Output directory for generated images and CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--visualizations-dir",
        default=None,
        help=(
            "Optional directory for one composite visualization per processed "
            "multi-defect source image."
        ),
    )
    parser.add_argument(
        "--dilate-kernel-size",
        type=int,
        default=15,
        help=(
            "Square dilation kernel applied to inpainting masks. The paper used "
            "15 (15x15). Set 0 to disable dilation. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--area-threshold",
        type=float,
        default=10.0,
        help=(
            "A target defect must cover strictly more than this percentage of "
            "the image to generate an isolated-defect sample. The paper used "
            "10%%. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Compute device; auto-detects CUDA when omitted",
    )

    return parser.parse_args()


def main() -> None:
    process_images(parse_args())


if __name__ == "__main__":
    main()
