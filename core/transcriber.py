import concurrent.futures
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from api.transcribe_api import TranscriptionManager
from api.summary_api import SummaryManager
from api.rate_limiter import RateLimiter
from modules.config_loader import get_config_loader
from modules import app_config as config
from modules.constants import DEFAULT_MODEL
from modules.concurrency_helper import (
    get_api_timeout,
    get_rate_limits,
    get_target_dpi,
    get_transcription_concurrency,
)
from modules.logger import setup_logger
from modules.path_utils import create_safe_directory_name, create_safe_log_filename
from modules.text_cleaner import clean_transcription, get_text_cleaning_config
from modules.types import SummaryResult, TranscriptionResult
from processors.file_manager import (
    append_to_log,
    create_docx_summary,
    finalize_log_file,
    initialize_log_file,
    write_transcription_to_text,
)
from processors.pdf_processor import (
    extract_pdf_pages_to_images,
    get_image_paths_from_folder,
)


logger = setup_logger(__name__)

# Constants for ETA calculation
MIN_SAMPLES_FOR_ETA = 5
RECENT_SAMPLES_FOR_ETA = 10
ETA_BLEND_WEIGHT_OVERALL = 0.7
ETA_BLEND_WEIGHT_RECENT = 0.3

# Constants for page number adjustment
MIN_SEQUENCE_LENGTH_FOR_ANCHOR = 2

# Roman numeral constants
ROMAN_NUMERAL_VALUES = [
	(1000, 'm'), (900, 'cm'), (500, 'd'), (400, 'cd'),
	(100, 'c'), (90, 'xc'), (50, 'l'), (40, 'xl'),
	(10, 'x'), (9, 'ix'), (5, 'v'), (4, 'iv'), (1, 'i')
]


def int_to_roman(num: int) -> str:
	"""Convert integer to lowercase Roman numeral string.
	
	Args:
		num: Positive integer to convert.
		
	Returns:
		Lowercase Roman numeral string.
	"""
	if num <= 0:
		return ""
	result = []
	for value, numeral in ROMAN_NUMERAL_VALUES:
		while num >= value:
			result.append(numeral)
			num -= value
	return "".join(result)


class ItemTranscriber:
	"""Process a single input item (PDF or image folder).

	Attributes:
		input_path: Source path of the item to process.
		input_type: Either "pdf" or "image_folder".
		base_output_dir: Base directory where outputs and working files are written.
		working_dir: Item-specific working directory containing images and logs.
		transcribe_manager: Manages image transcription via LLM API.
		summary_manager: Manages summarization via LLM API (if enabled).
	"""
	def __init__(self, input_path: Path, input_type: str,
	             base_output_dir: Path):
		self.input_path = input_path
		self.input_type = input_type  # "pdf" or "image_folder"
		self.name = self.input_path.stem

		self.base_output_dir = base_output_dir
		self.output_txt_path = self.base_output_dir / f"{self.name}.txt"
		self.output_summary_docx_path = self.base_output_dir / f"{self.name}_summary.docx"

		# Item-specific working directory for logs and temporary images
		# Use safe directory name to avoid Windows MAX_PATH (260 char) limitations
		safe_working_dir_name = create_safe_directory_name(self.name, "_working_files")
		self.working_dir = self.base_output_dir / safe_working_dir_name
		self.working_dir.mkdir(parents=True, exist_ok=True)

		self.images_dir = self.working_dir / "images"  # Temp images for this item
		self.images_dir.mkdir(exist_ok=True)

		# Use safe log filenames to avoid path length issues
		safe_transcription_log_name = create_safe_log_filename(self.name, "transcription")
		safe_summary_log_name = create_safe_log_filename(self.name, "summary")
		self.log_path = self.working_dir / safe_transcription_log_name
		self.summary_log_path = self.working_dir / safe_summary_log_name

		self.total_items_to_transcribe = 0
		self.start_time_processing: Optional[float] = None
		self.transcription_times: List[
			float] = []  # For successful transcriptions

		# Load model configuration from model.yaml
		config_loader = get_config_loader()
		model_cfg = config_loader.get_model_config()
		
		# Get transcription model configuration (centralized in model.yaml)
		transcription_cfg = model_cfg.get("transcription_model", {})
		self.transcription_model = transcription_cfg.get("name", DEFAULT_MODEL)
		self.transcription_provider = transcription_cfg.get("provider", "openai")
		
		# Get summary model configuration (centralized in model.yaml)
		summary_cfg = model_cfg.get("summary_model", {})
		self.summary_model = summary_cfg.get("name", DEFAULT_MODEL)
		summary_provider = summary_cfg.get("provider")
		
		# Use rate limiter with provider-agnostic configuration from concurrency.yaml
		self.transcribe_rate_limiter = RateLimiter(get_rate_limits())
		self.transcribe_manager = TranscriptionManager(
			model_name=self.transcription_model,
			provider=self.transcription_provider,
			rate_limiter=self.transcribe_rate_limiter,
			timeout=get_api_timeout(),
		)

		# Only initialize summary manager if summarization is enabled
		self.summary_manager = None
		if config.SUMMARIZE:
			self.summary_manager = SummaryManager(
				model_name=self.summary_model,
				provider=summary_provider,
			)

		# Image preprocessing is handled within modules.image_utils inside the transcription manager.

	def _get_list_of_images_to_transcribe(self) -> List[Path]:
		if self.input_type == "pdf":
			# Pass provider info for provider-specific image preprocessing
			return extract_pdf_pages_to_images(
				self.input_path,
				self.images_dir,
				provider=self.transcription_provider,
				model_name=self.transcription_model,
			)
		elif self.input_type == "image_folder":
			# For image folders, we process images directly from their source path.
			return get_image_paths_from_folder(self.input_path)
		return []

	def _build_summary_result(
		self,
		original_index: int,
		image_name: str,
		summary_payload: Dict[str, Any],
		page_number: Optional[int],
		model_page_number: Optional[int],
		error_message: Optional[str] = None,
	) -> Dict[str, Any]:
		"""Create a consistent summary result structure for downstream consumers."""
		if isinstance(summary_payload, dict) and page_number is not None:
			summary_payload.setdefault("page", page_number)
		result = {
			"original_input_order_index": original_index,
			"model_page_number": model_page_number,
			"summary": summary_payload,
			"image_filename": image_name,
		}
		if page_number is not None:
			result["page"] = page_number
		elif isinstance(summary_payload, dict) and "page" in summary_payload:
			result["page"] = summary_payload.get("page")
		if error_message:
			result["error"] = error_message
		return result

	def _create_placeholder_summary(
		self,
		page_number: Optional[int],
		page_number_type: str,
		bullet_points: List[str],
		references: Optional[List[str]] = None,
		contains_no_semantic_content: bool = True,
		error_message: Optional[str] = None,
	) -> Dict[str, Any]:
		payload = {
			"page": page_number if page_number is not None else 0,
			"summary": {
				"page_number": {
					"page_number_integer": page_number,
					"page_number_type": page_number_type
				},
				"bullet_points": bullet_points,
				"references": references or [],
				"contains_no_semantic_content": contains_no_semantic_content
			}
		}
		if error_message:
			payload["error"] = error_message
		return payload

	def _transcribe_and_summarize(
			self, image_paths: List[Path]
	) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
		transcription_results: List[Dict[str, Any]] = []
		summary_results: List[Dict[str, Any]] = []
		total_images = len(image_paths)
		processed_count = 0

		logger.info(
			f"Starting transcription{' and summarization' if config.SUMMARIZE else ''} of {total_images} images...")

		image_paths_with_indices = list(enumerate(image_paths))

		if config.SUMMARIZE and self.summary_manager:
			max_workers, _ = get_transcription_concurrency()
			initialize_log_file(
				self.summary_log_path, self.name, str(self.input_path),
				"PDF" if self.input_type == "pdf" else "Image Folder",
				total_images, self.summary_model,
				concurrency_limit=max_workers
			)

		def submit_task(args_tuple):
			original_input_order_index, img_path = args_tuple
			nonlocal processed_count
			try:
				transcription_result_raw = self.transcribe_manager.transcribe_image(
					img_path)
				transcription_result = {
					**transcription_result_raw,
					"original_input_order_index": original_input_order_index
				}
				
				# Clean transcription text (for both TXT output and summarization)
				if "error" not in transcription_result:
					raw_text = transcription_result.get("transcription", "")
					transcription_result["transcription"] = clean_transcription(raw_text)

				if config.SUMMARIZE and self.summary_manager and "error" not in transcription_result:
					page_num_model = transcription_result.get("page")
					transcription_text = transcription_result.get("transcription", "")

					has_valid_model_page_num = isinstance(page_num_model, int)
					page_num_to_use = page_num_model if has_valid_model_page_num else original_input_order_index + 1

					if "[empty page]" in transcription_text or "[no transcription possible]" in transcription_text:
						summary_data = self._create_placeholder_summary(
							page_num_to_use if has_valid_model_page_num else None,
							"arabic" if has_valid_model_page_num else "none",
							["[Empty page or no transcription possible]"],
						)
					else:
						summary_data = self.summary_manager.generate_summary(
							transcription_text,
							page_num_to_use
						)
					summary_error = summary_data.get("error") if isinstance(summary_data, dict) else None
					summary_result = self._build_summary_result(
						original_input_order_index,
						img_path.name,
						summary_data,
						page_num_to_use,
						page_num_model,
						summary_error,
					)
					append_to_log(self.summary_log_path, summary_result)
					summary_results.append(summary_result)
				elif config.SUMMARIZE and self.summary_manager:  # Handles transcription error case
					page_num_model = transcription_result.get("page")
					has_valid_model_page_num = isinstance(page_num_model, int)
					page_num_to_use = page_num_model if has_valid_model_page_num else original_input_order_index + 1
					error_msg = transcription_result.get("error",
					                                     "Unknown error")
					summary_data = self._create_placeholder_summary(
						page_num_to_use if has_valid_model_page_num else None,
						"arabic" if has_valid_model_page_num else "none",
						[f"[Transcription failed: {error_msg}]"],
						error_message=error_msg,
					)
					summary_result = self._build_summary_result(
						original_input_order_index,
						img_path.name,
						summary_data,
						page_num_to_use,
						page_num_model,
					)
					append_to_log(self.summary_log_path, summary_result)
					summary_results.append(summary_result)

				append_to_log(self.log_path, transcription_result)
				processed_count += 1

				if "processing_time" in transcription_result and "error" not in transcription_result:
					self.transcription_times.append(
						transcription_result["processing_time"])

				status = "SUCCESS" if "error" not in transcription_result else f"FAILED ({transcription_result.get('retries', 0)} retries)"
				item_num_str = transcription_result.get("page", "?")

				eta_str = self._calculate_eta(processed_count, total_images)

				logger.debug(
					f"Processed {processed_count}/{total_images} - Item {item_num_str} - Status: {status} - {eta_str}")
				transcription_results.append(transcription_result)

				return transcription_result

			except Exception as e:
				logger.exception(f"Critical error during task for {img_path.name}: {e}")
				# Fallback to original order index + 1 for page numbering on error
				seq_num = original_input_order_index + 1
				error_result = {
					"page": seq_num, "image": img_path.name,
					"transcription": f"[CRITICAL ERROR] Unhandled in task: {e}",
					"error": str(e),
					"original_input_order_index": original_input_order_index
				}
				transcription_results.append(error_result)
				processed_count += 1
				return error_result

		# Load concurrency settings from concurrency.yaml
		max_workers, _ = get_transcription_concurrency()
		max_workers = min(max_workers, len(image_paths))
		if max_workers <= 0:
			max_workers = 1
		
		logger.info(f"Using {max_workers} concurrent workers for transcription")
		with concurrent.futures.ThreadPoolExecutor(
				max_workers=max_workers) as executor:
			list(tqdm(executor.map(submit_task, image_paths_with_indices),
			          total=total_images,
			          desc="Processing images"))

		transcription_results.sort(
			key=lambda x: x.get("original_input_order_index", 0))
		return transcription_results, summary_results

	def _calculate_eta(self, processed_count: int, total_images: int) -> str:
		"""
		Calculate estimated time of arrival for remaining items.
		
		Args:
			processed_count: Number of items processed so far.
			total_images: Total number of images to process.
			
		Returns:
			Formatted ETA string.
		"""
		if processed_count <= MIN_SAMPLES_FOR_ETA or not self.start_time_processing:
			return "ETA: N/A"
		
		elapsed_total = time.time() - self.start_time_processing
		items_per_sec_overall = processed_count / elapsed_total
		
		if items_per_sec_overall <= 0:
			return "ETA: N/A"
		
		# Blend overall and recent rates for more accurate estimates
		blended_rate = self._calculate_blended_processing_rate(
			items_per_sec_overall
		)
		
		if blended_rate <= 0:
			return "ETA: N/A"
		
		remaining_items = total_images - processed_count
		eta_seconds = remaining_items / blended_rate
		return f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}"
	
	def _calculate_blended_processing_rate(self, overall_rate: float) -> float:
		"""Calculate blended processing rate from overall and recent samples."""
		recent_samples = self.transcription_times[-RECENT_SAMPLES_FOR_ETA:]
		if not recent_samples:
			return overall_rate
		
		recent_avg_time = sum(recent_samples) / len(recent_samples)
		recent_rate = 1.0 / recent_avg_time if recent_avg_time > 0 else overall_rate
		
		return (ETA_BLEND_WEIGHT_OVERALL * overall_rate + 
		        ETA_BLEND_WEIGHT_RECENT * recent_rate)

	def _parse_page_number_from_summary(
		self, summary_result: Dict[str, Any]
	) -> Tuple[Optional[int], str, bool]:
		"""
		Extract page number information from a summary result.
		
		Args:
			summary_result: Summary result dictionary.
			
		Returns:
			Tuple of (model_page_number, page_number_type, is_genuinely_unnumbered).
			page_number_type is one of: 'roman', 'arabic', 'none'.
		"""
		summary_container = summary_result.get("summary", {})
		inner_summary = summary_container.get("summary") if isinstance(
			summary_container, dict
		) else None
		
		if isinstance(inner_summary, dict):
			page_number_obj = inner_summary.get('page_number', {})
		else:
			page_number_obj = summary_container.get('page_number', {})
		
		model_page_num = None
		page_number_type = "none"
		is_genuinely_unnumbered = True
		
		if isinstance(page_number_obj, dict):
			# New schema format with nested page_number object
			model_page_num = page_number_obj.get('page_number_integer')
			page_number_type = page_number_obj.get('page_number_type', 'none')
			
			# Derive unnumbered status from page_number_type or null page_number_integer
			is_genuinely_unnumbered = (
				page_number_type == "none" or model_page_num is None
			)
			if is_genuinely_unnumbered:
				page_number_type = "none"
		else:
			# Fallback for old format or direct model_page_number value
			model_page_str = summary_result.get("model_page_number", "")
			try:
				if isinstance(model_page_str, (int, float)):
					model_page_num = int(model_page_str)
					page_number_type = "arabic"
					is_genuinely_unnumbered = False
				elif isinstance(model_page_str, str) and model_page_str.isdigit():
					model_page_num = int(model_page_str)
					page_number_type = "arabic"
					is_genuinely_unnumbered = False
			except ValueError:
				pass  # model_page_num remains None, is_genuinely_unnumbered stays True
		
		return model_page_num, page_number_type, is_genuinely_unnumbered

	def _find_longest_consecutive_sequence(
		self, summaries_with_pages: List[Dict[str, Any]]
	) -> List[Dict[str, Any]]:
		"""
		Find the longest consecutive sequence of page numbers in document order.
		
		Args:
			summaries_with_pages: List of summary wrappers with page number info.
			
		Returns:
			Longest consecutive sequence of summaries.
		"""
		if not summaries_with_pages:
			return []
		
		longest_sequence = []
		current_sequence = []
		
		for item in summaries_with_pages:
			# Start a new sequence or add to current if consecutive
			if not current_sequence:
				current_sequence = [item]
			elif (
				item["model_page_number_int"] == current_sequence[-1]["model_page_number_int"] + 1
				and item["original_input_order_index"] == current_sequence[-1]["original_input_order_index"] + 1
			):
				# Both page number and document position are consecutive
				current_sequence.append(item)
			else:
				# End of sequence, check if it's longer than our longest
				if len(current_sequence) > len(longest_sequence):
					longest_sequence = current_sequence.copy()
				# Start a new sequence with the current item
				current_sequence = [item]
		
		# Check if the last sequence is the longest
		if len(current_sequence) > len(longest_sequence):
			longest_sequence = current_sequence
		
		return longest_sequence

	def _calculate_adjusted_page_number(
		self, 
		original_index: int, 
		anchor_model_page: int, 
		anchor_original_index: int
	) -> int:
		"""
		Calculate adjusted page number based on anchor point.
		
		Args:
			original_index: Original position in document.
			anchor_model_page: Model-detected page number at anchor.
			anchor_original_index: Original position of anchor.
			
		Returns:
			Adjusted page number.
		"""
		offset = original_index - anchor_original_index
		adjusted_page = anchor_model_page + offset
		return max(1, adjusted_page)  # Ensure page number is at least 1

	def _adjust_and_sort_summary_page_numbers(
		self, summary_results: List[Dict[str, Any]]
	) -> List[Dict[str, Any]]:
		"""
		Adjust page numbers based on model-detected sequences and sort results.
		
		Handles both Roman numeral (preface) and Arabic (main text) page numbering
		systems separately, preserving the original document order.
		
		Args:
			summary_results: List of summary results with page information.
			
		Returns:
			Sorted list of summaries with adjusted page numbers.
		"""
		if not summary_results:
			return []
		
		logger.info("Adjusting page numbers based on model-detected page sequences...")
		
		# Parse page numbers from all summaries
		parsed_summaries = []
		for r in summary_results:
			model_page_num, page_type, is_unnumbered = self._parse_page_number_from_summary(r)
			parsed_summaries.append({
				"original_input_order_index": r["original_input_order_index"],
				"model_page_number_int": model_page_num,
				"page_number_type": page_type,
				"data": r,
				"is_genuinely_unnumbered": is_unnumbered
			})

		# Separate Roman and Arabic numbered pages
		roman_pages = [
			s for s in parsed_summaries
			if s["model_page_number_int"] is not None 
			and s["page_number_type"] == "roman"
			and not s["is_genuinely_unnumbered"]
		]
		arabic_pages = [
			s for s in parsed_summaries
			if s["model_page_number_int"] is not None 
			and s["page_number_type"] == "arabic"
			and not s["is_genuinely_unnumbered"]
		]
		
		logger.info(
			f"Found {len(roman_pages)} Roman numeral pages, "
			f"{len(arabic_pages)} Arabic numeral pages"
		)
		
		# Determine anchor points for each numbering system
		roman_anchor_page = 1
		roman_anchor_index = 0
		arabic_anchor_page = 1
		arabic_anchor_index = 0
		
		# Process Roman numeral sequence
		if roman_pages:
			roman_pages.sort(key=lambda x: x["original_input_order_index"])
			detected_roman = [s["model_page_number_int"] for s in roman_pages]
			detected_roman_pos = [s["original_input_order_index"] for s in roman_pages]
			logger.info(f"Roman page numbers (in document order): {detected_roman}")
			logger.info(f"At document positions: {detected_roman_pos}")
			
			longest_roman_seq = self._find_longest_consecutive_sequence(roman_pages)
			if longest_roman_seq and len(longest_roman_seq) > MIN_SEQUENCE_LENGTH_FOR_ANCHOR:
				anchor_item = longest_roman_seq[0]
				roman_anchor_page = anchor_item["model_page_number_int"]
				roman_anchor_index = anchor_item["original_input_order_index"]
				logger.info(f"Roman anchor: page {roman_anchor_page} at index {roman_anchor_index}")
			elif roman_pages:
				anchor_item = roman_pages[0]
				roman_anchor_page = anchor_item["model_page_number_int"]
				roman_anchor_index = anchor_item["original_input_order_index"]
				logger.info(f"Roman fallback anchor: page {roman_anchor_page} at index {roman_anchor_index}")
		
		# Process Arabic numeral sequence
		if arabic_pages:
			arabic_pages.sort(key=lambda x: x["original_input_order_index"])
			detected_arabic = [s["model_page_number_int"] for s in arabic_pages]
			detected_arabic_pos = [s["original_input_order_index"] for s in arabic_pages]
			logger.info(f"Arabic page numbers (in document order): {detected_arabic}")
			logger.info(f"At document positions: {detected_arabic_pos}")
			
			longest_arabic_seq = self._find_longest_consecutive_sequence(arabic_pages)
			if longest_arabic_seq and len(longest_arabic_seq) > MIN_SEQUENCE_LENGTH_FOR_ANCHOR:
				anchor_item = longest_arabic_seq[0]
				arabic_anchor_page = anchor_item["model_page_number_int"]
				arabic_anchor_index = anchor_item["original_input_order_index"]
				logger.info(f"Arabic anchor: page {arabic_anchor_page} at index {arabic_anchor_index}")
			elif arabic_pages:
				anchor_item = arabic_pages[0]
				arabic_anchor_page = anchor_item["model_page_number_int"]
				arabic_anchor_index = anchor_item["original_input_order_index"]
				logger.info(f"Arabic fallback anchor: page {arabic_anchor_page} at index {arabic_anchor_index}")
		
		if not roman_pages and not arabic_pages:
			logger.info("No valid model page numbers detected. Using default numbering starting from 1.")

		# Apply page number adjustment to all summaries
		final_ordered_summaries = []
		for s_wrapper in parsed_summaries:
			original_index = s_wrapper["original_input_order_index"]
			page_type = s_wrapper["page_number_type"]

			# Ensure the 'summary' key exists and is a dictionary in the data part
			if "summary" not in s_wrapper["data"] or not isinstance(
					s_wrapper["data"]["summary"], dict):
				s_wrapper["data"]["summary"] = {}
			summary_data_dict = s_wrapper["data"]["summary"]
			# Identify the target dict that actually holds page_number (nested summary JSON preferred)
			target_summary_json = summary_data_dict.get("summary") if isinstance(summary_data_dict.get("summary"), dict) else summary_data_dict

			# Ensure page_number object structure exists with page_number_type
			if "page_number" not in target_summary_json or not isinstance(
					target_summary_json["page_number"], dict):
				target_summary_json["page_number"] = {
					"page_number_integer": None,
					"page_number_type": "none"
				}

			if s_wrapper["is_genuinely_unnumbered"]:
				target_summary_json["page_number"]["page_number_integer"] = None
				target_summary_json["page_number"]["page_number_type"] = "none"
			elif page_type == "roman" and roman_pages:
				# Use Roman anchor for adjustment
				adjusted_page = self._calculate_adjusted_page_number(
					original_index, roman_anchor_page, roman_anchor_index)
				adjusted_page = max(1, adjusted_page)
				target_summary_json["page_number"]["page_number_integer"] = adjusted_page
				target_summary_json["page_number"]["page_number_type"] = "roman"
			elif page_type == "arabic" and arabic_pages:
				# Use Arabic anchor for adjustment
				adjusted_page = self._calculate_adjusted_page_number(
					original_index, arabic_anchor_page, arabic_anchor_index)
				adjusted_page = max(1, adjusted_page)
				target_summary_json["page_number"]["page_number_integer"] = adjusted_page
				target_summary_json["page_number"]["page_number_type"] = "arabic"
			else:
				# No valid page type or no anchors - default to Arabic with index-based numbering
				adjusted_page = original_index + 1
				# Try to use Arabic anchor if available
				if arabic_pages:
					adjusted_page = self._calculate_adjusted_page_number(
						original_index, arabic_anchor_page, arabic_anchor_index)
				adjusted_page = max(1, adjusted_page)
				target_summary_json["page_number"]["page_number_integer"] = adjusted_page
				target_summary_json["page_number"]["page_number_type"] = page_type if page_type != "none" else "arabic"

			final_ordered_summaries.append(s_wrapper["data"])

		# Sort by original input order to preserve document sequence, regardless of page_number
		final_ordered_summaries.sort(
			key=lambda x: x.get("original_input_order_index", float('inf'))
		)
		return final_ordered_summaries

	def process_item(self) -> None:
		item_type_str = "PDF" if self.input_type == "pdf" else "Image Folder"
		logger.info(
			f"Processing {self.name} ({item_type_str})")
		self.start_time_processing = time.time()

		# Prepare images (extract from PDF or list from folder)
		image_paths_to_process = self._get_list_of_images_to_transcribe()

		if not image_paths_to_process:
			logger.info(
				f"No images found or extracted for {self.name}. Aborting this item.")
			# Clean up empty working directory if it was created for this item
			try:
				if not any(self.working_dir.iterdir()):  # Check if empty
					shutil.rmtree(self.working_dir)
					logger.info(
						f"Removed empty working directory: {self.working_dir}")
			except Exception as e:
				logger.warning(
					f"Could not remove working directory {self.working_dir}: {e}")
			return

		self.total_items_to_transcribe = len(image_paths_to_process)
		logger.info(
			f"Prepared {self.total_items_to_transcribe} images for transcription{' and summarization' if config.SUMMARIZE else ''}.")

		# Initialize log file with a header
		# read target_dpi from modules config for logging
		target_dpi = None
		if self.input_type == "pdf":
			target_dpi = get_target_dpi()
		actual_concurrency, _ = get_transcription_concurrency()
		initialize_log_file(
			self.log_path, self.name, str(self.input_path), item_type_str,
			self.total_items_to_transcribe, self.transcription_model,
			target_dpi, concurrency_limit=actual_concurrency
		)

		# Initialize variables that will be used in finally block
		should_cleanup = False
		transcription_results = []
		summary_results = []

		try:
			# Process all images - transcribe and summarize
			transcription_results, summary_results = self._transcribe_and_summarize(
				image_paths_to_process)

			# Final processing and output
			total_elapsed_time = time.time() - (
					self.start_time_processing or time.time())
			elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed_time))

			# Sort results by page number
			transcription_results.sort(key=lambda x: int(x.get("page", 0)))

			final_success_count = sum(
				1 for r in transcription_results if "error" not in r)
			final_failure_count = len(transcription_results) - final_success_count

			# Save transcription to text file
			logger.info(
				f"Writing final transcription output to: {self.output_txt_path}")
			write_transcription_to_text(
				transcription_results, self.output_txt_path, self.name,
				item_type_str, total_elapsed_time, self.input_path
			)

			# Save summaries to DOCX file if summarization is enabled
			if config.SUMMARIZE and summary_results:
				# Step 1: Adjust page numbers and sort (uses page_number_type for unnumbered detection)
				adjusted_summary_results = self._adjust_and_sort_summary_page_numbers(
					summary_results)

				try:
					# Use the adjusted list for creating summary files
					create_docx_summary(adjusted_summary_results,
					                    self.output_summary_docx_path, self.name)

				except Exception as e:
					logger.error(f"Error creating summary files: {e}")

			logger.info(f"PROCESSING COMPLETE for item: {self.name}")
			logger.info(f"  Total images for this item: {len(transcription_results)}")
			logger.info(f"  Successfully transcribed: {final_success_count}")
			logger.info(f"  Failed items: {final_failure_count}")
			logger.info(f"  Total time for this item: {elapsed_str}")
			if self.transcription_times:  # Based on successful API calls
				avg_api_time = sum(self.transcription_times) / len(
					self.transcription_times)
				logger.info(
					f"  Average API processing time per successful image: {avg_api_time:.2f}s")
			if total_elapsed_time > 0 and final_success_count > 0:
				throughput_iph = (final_success_count / total_elapsed_time) * 3600
				logger.info(
					f"  Overall throughput for this item: {throughput_iph:.1f} successful images/hour")
			logger.info(
				f"  Final transcription output: {self.output_txt_path}")
			if config.SUMMARIZE:
				logger.info(
					f"  Final summary outputs: {self.output_summary_docx_path}")
			logger.info(
				f"  Detailed logs: {self.log_path}{' and ' + str(self.summary_log_path) if config.SUMMARIZE else ''}")

			# Determine if we should cleanup images directory
			# Only delete if we have successful outputs (transcription and/or summary)
			if self.output_txt_path.exists():
				# We have a transcription output
				should_cleanup = True
			if config.SUMMARIZE and self.output_summary_docx_path.exists():
				# We have a summary output
				should_cleanup = True
		
		finally:
			# Always finalize log files by closing JSON arrays, even if errors occurred
			finalize_log_file(self.log_path)
			if config.SUMMARIZE:
				finalize_log_file(self.summary_log_path)

			# Cleanup: Delete images directory after successful processing
			if should_cleanup and self.images_dir.exists():
				try:
					# Delete the images directory and all its contents
					shutil.rmtree(self.images_dir)
					logger.info(f"  Cleaned up images directory: {self.images_dir}")
				except Exception as e:
					logger.warning(f"  Could not delete images directory {self.images_dir}: {e}")
