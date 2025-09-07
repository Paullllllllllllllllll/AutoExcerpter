import base64
import io
import math

import numpy as np
from PIL import Image, ImageFilter, ImageOps

import config


class ImageProcessor:
    """Handles in-memory image optimization for transcription."""

    def __init__(
            self,
            jpeg_quality: int = config.JPEG_QUALITY,
            min_pixels: int = config.MIN_TOTAL_PIXELS_DEFAULT,
            max_pixels: int = config.MAX_TOTAL_PIXELS_DEFAULT,
    ):
        self.jpeg_quality = jpeg_quality
        self.min_total_pixels = min_pixels
        self.max_total_pixels = max_pixels
        self.enable_grayscale = True
        self.enable_border_removal = True
        self.enable_transparency_handling = True
        self.enable_auto_contrast = True

    def process_pil_image(self, img: Image.Image) -> Image.Image:
        processed_img = img.copy()

        if processed_img.mode == "P":
            processed_img = (
                processed_img.convert("RGBA")
                if "transparency" in processed_img.info
                else processed_img.convert("RGB")
            )
        if self.enable_transparency_handling and processed_img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", processed_img.size, (255, 255, 255))
            background.paste(processed_img, mask=processed_img.split()[-1])
            processed_img.close()  # Close intermediate image
            processed_img = background
        elif processed_img.mode not in ("RGB", "L"):
            processed_img = processed_img.convert("RGB")

        if self.enable_border_removal:
            original_before_crop = processed_img
            processed_img = self._remove_borders(processed_img)
            if processed_img != original_before_crop:  # if _remove_borders returned a new (cropped) image
                original_before_crop.close()

        if self.enable_grayscale and processed_img.mode != "L":
            processed_img = ImageOps.grayscale(processed_img)

        if self.enable_auto_contrast:
            cutoff = 1.0 if processed_img.mode == "L" else 0.5
            processed_img = ImageOps.autocontrast(processed_img, cutoff=cutoff)

        processed_img = processed_img.filter(
            ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=3)
        )

        total_pixels = processed_img.width * processed_img.height
        if total_pixels == 0:  # Avoid division by zero for empty images
            return processed_img

        if total_pixels < self.min_total_pixels:
            factor = math.sqrt(self.min_total_pixels / total_pixels)
            new_size = (
                max(1, int(processed_img.width * factor)),
                max(1, int(processed_img.height * factor)),
            )
            resized_img = processed_img.resize(new_size, Image.LANCZOS)
            processed_img.close()
            processed_img = resized_img
        elif total_pixels > self.max_total_pixels:
            factor = math.sqrt(self.max_total_pixels / total_pixels)
            new_size = (
                max(1, int(processed_img.width * factor)),
                max(1, int(processed_img.height * factor)),
            )
            resized_img = processed_img.resize(new_size, Image.LANCZOS)
            processed_img.close()
            processed_img = resized_img
        return processed_img

    def _remove_borders(self, img: Image.Image) -> Image.Image:
        try:
            img_array = np.array(img)
            if img.mode == "L":  # Grayscale
                mask = img_array < 245  # Non-white pixels
            else:  # Color
                # Consider pixels dark if the average channel value is less than 245
                darkness = 255 - np.mean(img_array, axis=2)
                mask = darkness > 10  # Threshold for considering a pixel as content

            rows_with_content = np.any(mask, axis=1)
            cols_with_content = np.any(mask, axis=0)

            if not np.any(rows_with_content) or not np.any(cols_with_content):
                return img  # Blank image or no content detected

            row_indices = np.where(rows_with_content)[0]
            col_indices = np.where(cols_with_content)[0]

            margin = 20
            top = max(0, row_indices[0] - margin)
            bottom = min(img.height, row_indices[-1] + margin + 1)
            left = max(0, col_indices[0] - margin)
            right = min(img.width, col_indices[-1] + margin + 1)

            min_content_size = min(img.width, img.height) * 0.1
            if (bottom - top) < min_content_size or \
                    (right - left) < min_content_size:
                return img  # Crop area too small, return original

            return img.crop((left, top, right, bottom))
        except Exception as e:
            print(
                f"Warning: Border removal failed: {e}. Returning original image.")
            return img  # Return original on any failure

    def encode_pil_to_base64(self, img: Image.Image) -> str:
        buffer = io.BytesIO()
        save_img = img
        try:
            if save_img.mode not in ["RGB", "L"]:
                save_img = img.convert("RGB")

            save_params = {'format': 'JPEG', 'quality': self.jpeg_quality}
            if 'icc_profile' in img.info:
                save_params['icc_profile'] = img.info['icc_profile']

            save_img.save(buffer, **save_params)
            buffer.seek(0)
            image_bytes = buffer.read()
            return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            print(f"Error encoding PIL image to base64: {e}")
            return ""
        finally:
            if save_img != img and hasattr(save_img, 'close'):  # if converted
                save_img.close()
            buffer.close()
