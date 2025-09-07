# modules/image_utils.py

from PIL import Image, ImageOps
from pathlib import Path
from typing import List, Optional, Tuple

from modules.config_loader import ConfigLoader
from modules.logger import setup_logger
from modules.multiprocessing_utils import run_multiprocessing_tasks

logger = setup_logger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp',
                              '.gif', '.webp'}


class ImageProcessor:
    def __init__(self, image_path: Path) -> None:
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.error(f"Unsupported image format: {image_path.suffix}")
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        self.image_path = image_path

        config_loader = ConfigLoader()
        config_loader.load_configs()
        # Full config dict (contains 'image_processing' and 'ocr' sections)
        self.image_config = config_loader.get_image_processing_config()
        # OpenAI API preprocessing settings
        self.img_cfg = self.image_config.get('api_image_processing', {})

    def convert_to_grayscale(self, image: Image.Image) -> Image.Image:
        """
        Convert the image to grayscale if enabled.
        Returns:
            Image.Image: The grayscale image.
        """
        if self.img_cfg.get('grayscale_conversion', True):
            return ImageOps.grayscale(image)
        return image

    @staticmethod
    def resize_for_detail(image: Image.Image, detail: str, img_cfg: dict) -> Image.Image:
        """
        Resize strategy based on desired LLM detail.
        - low: downscale longest side to low_max_side_px.
        - high: fit/pad into high_target_box [width, height].
        - auto: default to 'high' strategy.
        """
        # Normalize flags and defaults
        resize_profile = (img_cfg.get('resize_profile', 'auto') or 'auto').lower()
        if resize_profile == 'none':
            return image
        detail_norm = (detail or 'high').lower()
        if detail_norm not in ('low', 'high', 'auto'):
            detail_norm = 'high'
        if detail_norm == 'low':
            max_side = int(img_cfg.get('low_max_side_px', 512))
            w, h = image.size
            longest = max(w, h)
            if longest <= max_side:
                return image
            scale = max_side / float(longest)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        # high or auto -> box/pad
        box = img_cfg.get('high_target_box', [768, 1536])
        try:
            target_width = int(box[0])
            target_height = int(box[1])
        except Exception:
            target_width, target_height = 768, 1536
        orig_width, orig_height = image.size
        scale_w = target_width / orig_width
        scale_h = target_height / orig_height
        scale = min(scale_w, scale_h)
        new_width = max(1, int(orig_width * scale))
        new_height = max(1, int(orig_height * scale))
        resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        final_img = Image.new("RGB", (target_width, target_height), (255, 255, 255))
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        final_img.paste(resized_img, (paste_x, paste_y))
        return final_img

    def handle_transparency(self, image: Image.Image) -> Image.Image:
        """
        Handle transparency by pasting the image onto a white background.
        Returns:
            Image.Image: The processed image.
        """
        if self.img_cfg.get('handle_transparency', True):
            if image.mode in ('RGBA', 'LA') or (
                    image.mode == 'P' and 'transparency' in image.info):
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                return background
        return image

    def process_image(self, output_path: Path) -> str:
        """
        Process the image and save it to the given output path with compression.
        Returns:
            str: A message indicating the outcome.
        """
        try:
            with Image.open(self.image_path) as img:
                img = self.handle_transparency(img)
                img = self.convert_to_grayscale(img)
                # Choose resizing based on llm_detail and resize_profile
                detail = (self.img_cfg.get('llm_detail', 'high') or 'high')
                img = ImageProcessor.resize_for_detail(img, detail, self.img_cfg)

                # Force output to JPEG with configurable quality (regardless of extension)
                # Create a new path with .jpg extension
                jpg_output_path = output_path.with_suffix('.jpg')
                jpeg_quality = int(self.img_cfg.get('jpeg_quality', 95))
                img.save(
                    jpg_output_path,
                    format='JPEG',
                    quality=jpeg_quality
                )
                logger.debug(
                    f"Saved processed image {jpg_output_path.name} size={img.size} quality={jpeg_quality} detail={detail}"
                )
            return f"Processed and saved: {jpg_output_path.name}"
        except Exception as e:
            logger.error(f"Error processing image {self.image_path.name}: {e}")
            return f"Failed to process {self.image_path.name}: {e}"

    # --- Static Methods for Folder-Level Processing ---

    @staticmethod
    def prepare_image_folder(folder: Path, image_output_dir: Path) -> Tuple[
        Path, Path, Path, Path]:
        """
        Prepares the output directories for processing an image folder.

        Returns a tuple of:
          - parent_folder: The directory for outputs related to this folder.
          - preprocessed_folder: Where preprocessed images will be stored.
          - temp_jsonl_path: File for recording transcription logs.
          - output_txt_path: Final transcription text file.
        """
        parent_folder = image_output_dir / folder.name
        parent_folder.mkdir(parents=True, exist_ok=True)
        preprocessed_folder = parent_folder / "preprocessed_images"
        preprocessed_folder.mkdir(exist_ok=True)
        temp_jsonl_path = parent_folder / f"{folder.name}_transcription.jsonl"
        if not temp_jsonl_path.exists():
            temp_jsonl_path.touch()
        output_txt_path = parent_folder / f"{folder.name}_transcription.txt"
        return parent_folder, preprocessed_folder, temp_jsonl_path, output_txt_path

    @staticmethod
    def process_images_multiprocessing(image_paths: List[Path],
                                       output_paths: List[Path]) -> List[
        Optional[str]]:
        """
        Process images using multiprocessing.

        Parameters:
            image_paths (List[Path]): List of source image paths.
            output_paths (List[Path]): List of destination paths for processed images.

        Returns:
            List[Optional[str]]: Processing results for each image.
        """
        # Load concurrency settings
        cfg_loader = ConfigLoader()
        cfg_loader.load_configs()
        conc = cfg_loader.get_concurrency_config()
        img_conc = (conc.get('concurrency', {})
                         .get('image_processing', {}))
        processes = int(img_conc.get('concurrency_limit', 12))

        args_list = list(zip(image_paths, output_paths))
        results = run_multiprocessing_tasks(
            ImageProcessor._process_image_task,
            args_list,
            processes=processes
        )
        return results

    @staticmethod
    def _process_image_task(img_path: Path, out_path: Path) -> str:
        """
        Helper task to process a single image and save it to out_path.

        Parameters:
            img_path (Path): Path to the input image.
            out_path (Path): Path to save the processed image.

        Returns:
            str: A status message.
        """
        processor = ImageProcessor(img_path)
        return processor.process_image(out_path)

    @staticmethod
    def process_and_save_images(source_folder: Path, preprocessed_folder: Path) -> List[Path]:
        """
        Reads images from source folder, processes them, and saves directly to preprocessed folder.
        Eliminates the need for intermediate raw image storage.

        Parameters:
            source_folder (Path): Path to the source folder containing original images.
            preprocessed_folder (Path): Path to save processed images.

        Returns:
            List[Path]: List of processed image paths.
        """
        # Find all image files in source folder
        image_files: List[Path] = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            image_files.extend(list(source_folder.glob(f'*{ext}')))

        if not image_files:
            return []

        # Create list of output paths
        output_paths = [
            preprocessed_folder / f"{img.stem}_pre_processed.jpg" for img in image_files
        ]

        # Process all images at once using multiprocessing
        ImageProcessor.process_images_multiprocessing(image_files, output_paths)

        # Return all processed image paths that exist
        return [p for p in output_paths if p.exists()]
