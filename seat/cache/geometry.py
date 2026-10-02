"""Apply identical square or aspect-preserving geometry to images and masks."""
from PIL import Image


def resize_image(image: Image.Image, size: int, geometry: str = "square") -> Image.Image:
    if geometry == "square":
        return image.resize((size, size), Image.Resampling.BILINEAR)
    if geometry != "arpad":
        raise ValueError(f"Unknown image geometry: {geometry}")
    scale = size / max(image.size)
    shape = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(shape, Image.Resampling.BILINEAR)
    canvas = Image.new(image.mode, (size, size), 0)
    canvas.paste(resized, ((size - shape[0]) // 2, (size - shape[1]) // 2))
    return canvas


def bbox_features(row: dict, size: int, geometry: str) -> list:
    if geometry == "square":
        return row["bbox_norm"]
    from common import bbox_norm_xyxy
    width, height = row["image_width"], row["image_height"]
    new_w, new_h = max(1, round(width * size / max(width, height))), max(1, round(height * size / max(width, height)))
    dx, dy = (size - new_w) // 2, (size - new_h) // 2
    x1, y1, x2, y2 = row["sam2_bbox_px"]
    return bbox_norm_xyxy([x1 * new_w / width + dx, y1 * new_h / height + dy,
                           x2 * new_w / width + dx, y2 * new_h / height + dy], size, size)
