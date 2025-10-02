import concurrent.futures
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from tqdm import tqdm

from modules import app_config as config
from api.openai_transcribe_api import OpenAITranscriptionManager
from api.openai_api import OpenAISummaryManager
from processors.pdf_processor import extract_pdf_pages_to_images, \
	get_image_paths_from_folder
from processors.file_manager import (
	create_docx_summary, write_transcription_to_text, initialize_log_file,
	append_to_log, finalize_log_file
)
from api.rate_limiter import RateLimiter
from modules.config_loader import ConfigLoader
from modules.logger import setup_logger


logger = setup_logger(__name__)

# Constants for ETA calculation
MIN_SAMPLES_FOR_ETA = 5
RECENT_SAMPLES_FOR_ETA = 10
ETA_BLEND_WEIGHT_OVERALL = 0.7
ETA_BLEND_WEIGHT_RECENT = 0.3

# Constants for page number adjustment
MIN_SEQUENCE_LENGTH_FOR_ANCHOR = 2


class ItemTranscriber:
	"""Process a single input item (PDF or image folder).

	Attributes:
		input_path: Source path of the item to process.
		input_type: Either "pdf" or "image_folder".
		base_output_dir: Base directory where outputs and working files are written.
		working_dir: Item-specific working directory containing images and logs.
		openai_transcribe_manager: Manages image transcription via OpenAI API.
		openai_summary_manager: Manages summarization via OpenAI API (if enabled).
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
		self.working_dir = self.base_output_dir / f"{self.name}_working_files"
		self.working_dir.mkdir(parents=True, exist_ok=True)

		self.images_dir = self.working_dir / "images"  # Temp images for this item
		self.images_dir.mkdir(exist_ok=True)

		self.log_path = self.working_dir / f"{self.name}_transcription_log.json"
		self.summary_log_path = self.working_dir / f"{self.name}_summary_log.json"

		self.total_items_to_transcribe = 0
		self.start_time_processing: Optional[float] = None
		self.transcription_times: List[
			float] = []  # For successful transcriptions

		# Use specific rate limiter configurations
		self.openai_transcribe_rate_limiter = RateLimiter(config.OPENAI_RATE_LIMITS)
		self.openai_transcribe_manager = OpenAITranscriptionManager(
			config.OPENAI_API_KEY,
			config.OPENAI_TRANSCRIPTION_MODEL,
			rate_limiter=self.openai_transcribe_rate_limiter,
			timeout=config.OPENAI_API_TIMEOUT,
		)

		# Only initialize OpenAI manager if summarization is enabled
		self.openai_summary_manager = None
		if config.SUMMARIZE:
			self.openai_summary_manager = OpenAISummaryManager(
				config.OPENAI_API_KEY, config.OPENAI_MODEL)

		# Image preprocessing is handled within modules.image_utils inside the transcription manager.

	def _get_list_of_images_to_transcribe(self) -> List[Path]:
		if self.input_type == "pdf":
			return extract_pdf_pages_to_images(
				self.input_path, self.images_dir)
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
		page_number: int,
		contains_no_page_number: bool,
		bullet_points: List[str],
		references: Optional[List[str]] = None,
		contains_no_semantic_content: bool = True,
		error_message: Optional[str] = None,
	) -> Dict[str, Any]:
		payload = {
			"page": page_number,
			"summary": {
				"page_number": {
					"page_number_integer": page_number,
					"contains_no_page_number": contains_no_page_number
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

		if config.SUMMARIZE and self.openai_summary_manager:
			try:
				cfg_loader = ConfigLoader()
				cfg_loader.load_configs()
				concurrency_cfg = cfg_loader.get_concurrency_config()
				max_workers = concurrency_cfg.get("api_requests", {}).get("transcription", {}).get("concurrency_limit", 4)
			except Exception:
				max_workers = config.CONCURRENT_REQUESTS
			initialize_log_file(
				self.summary_log_path, self.name, str(self.input_path),
				"PDF" if self.input_type == "pdf" else "Image Folder",
				total_images, config.OPENAI_MODEL,
				concurrency_limit=max_workers
			)

		def submit_task(args_tuple):
			original_input_order_index, img_path = args_tuple
			nonlocal processed_count
			try:
				transcription_result_raw = self.openai_transcribe_manager.transcribe_image(
					img_path)
				transcription_result = {
					**transcription_result_raw,
					"original_input_order_index": original_input_order_index
				}

				if config.SUMMARIZE and self.openai_summary_manager and "error" not in transcription_result:
					page_num_model = transcription_result.get("page")
					transcription_text = transcription_result.get(
						"transcription", "")

					has_valid_model_page_num = isinstance(page_num_model, int)
					page_num_to_use = page_num_model if has_valid_model_page_num else original_input_order_index + 1

					if "[empty page]" in transcription_text or "[no transcription possible]" in transcription_text:
						summary_data = self._create_placeholder_summary(
							page_num_to_use,
							not has_valid_model_page_num,
							["[Empty page or no transcription possible]"],
						)
					else:
						summary_data = self.openai_summary_manager.generate_summary(
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
				elif config.SUMMARIZE and self.openai_summary_manager:  # Handles transcription error case
					page_num_model = transcription_result.get("page")
					has_valid_model_page_num = isinstance(page_num_model, int)
					page_num_to_use = page_num_model if has_valid_model_page_num else original_input_order_index + 1
					error_msg = transcription_result.get("error",
					                                     "Unknown error")
					summary_data = self._create_placeholder_summary(
						page_num_to_use,
						not has_valid_model_page_num,
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
		try:
			cfg_loader = ConfigLoader()
			cfg_loader.load_configs()
			concurrency_cfg = cfg_loader.get_concurrency_config()
			max_workers = concurrency_cfg.get("api_requests", {}).get("transcription", {}).get("concurrency_limit", 4)
		except Exception:
			max_workers = config.CONCURRENT_REQUESTS
		
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

	def _calculate_eta(
		self, processed_count: int, total_images: int
	) -> str:
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
		remaining_items = total_images - processed_count
		
		if items_per_sec_overall <= 0:
			return "ETA: N/A"
		
		# Calculate recent average for more accurate short-term estimates
		recent_samples = self.transcription_times[-RECENT_SAMPLES_FOR_ETA:]
		recent_avg_time = sum(recent_samples) / len(recent_samples) if recent_samples else 0
		
		items_per_sec_recent = (
			1.0 / recent_avg_time if recent_avg_time > 0 else items_per_sec_overall
		)
		
		# Blend overall and recent rates for stability
		blended_ips = (
			ETA_BLEND_WEIGHT_OVERALL * items_per_sec_overall + 
			ETA_BLEND_WEIGHT_RECENT * items_per_sec_recent
		)
		
		if blended_ips <= 0:
			return "ETA: N/A"
		
		eta_seconds = remaining_items / blended_ips
		return f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}"

	def _parse_page_number_from_summary(
		self, summary_result: Dict[str, Any]
	) -> Tuple[Optional[int], bool]:
		"""
		Extract page number information from a summary result.
		
		Args:
			summary_result: Summary result dictionary.
			
		Returns:
			Tuple of (model_page_number, is_genuinely_unnumbered).
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
		is_genuinely_unnumbered = False
		
		if isinstance(page_number_obj, dict):
			# New schema format with nested page_number object
			model_page_num = page_number_obj.get('page_number_integer')
			is_genuinely_unnumbered = page_number_obj.get('contains_no_page_number', False)
		else:
			# Fallback for old format or direct model_page_number value
			model_page_str = summary_result.get("model_page_number", "")
			try:
				if isinstance(model_page_str, (int, float)):
					model_page_num = int(model_page_str)
				elif isinstance(model_page_str, str) and model_page_str.isdigit():
					model_page_num = int(model_page_str)
			except ValueError:
				pass  # model_page_num remains None
			
			# Check old schema for unnumbered flag on the container if present
			is_genuinely_unnumbered = bool(
				summary_container.get('contains_no_page_number', False)
			)
		
		return model_page_num, is_genuinely_unnumbered

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
			model_page_num, is_unnumbered = self._parse_page_number_from_summary(r)
			parsed_summaries.append({
				"original_input_order_index": r["original_input_order_index"],
				"model_page_number_int": model_page_num,
				"data": r,
				"is_genuinely_unnumbered": is_unnumbered
			})

		# Filter to valid model pages for anchor calculation
		summaries_with_model_pages = [
			s for s in parsed_summaries
			if s["model_page_number_int"] is not None and not s["is_genuinely_unnumbered"]
		]
		
		logger.info(
			f"Found {len(summaries_with_model_pages)} pages with valid "
			"model-detected page numbers"
		)
		
		# Determine anchor point for page number adjustment
		anchor_model_page = 1
		anchor_original_index = 0

		if summaries_with_model_pages:
			# Sort by original document order
			summaries_with_model_pages.sort(key=lambda x: x["original_input_order_index"])
			
			# Log detected page numbers for debugging
			detected_page_nums = [s["model_page_number_int"] for s in summaries_with_model_pages]
			detected_positions = [s["original_input_order_index"] for s in summaries_with_model_pages]
			logger.info(f"Model-detected page numbers (in document order): {detected_page_nums}")
			logger.info(f"At document positions: {detected_positions}")
			
			# Find longest consecutive sequence
			longest_sequence = self._find_longest_consecutive_sequence(summaries_with_model_pages)
			
			if longest_sequence and len(longest_sequence) > MIN_SEQUENCE_LENGTH_FOR_ANCHOR:
				# Use first item of longest sequence as anchor
				anchor_item = longest_sequence[0]
				anchor_model_page = anchor_item["model_page_number_int"]
				anchor_original_index = anchor_item["original_input_order_index"]
				
				# Log the found sequence
				sequence_page_nums = [s["model_page_number_int"] for s in longest_sequence]
				sequence_indexes = [s["original_input_order_index"] for s in longest_sequence]
				logger.info(f"Longest consecutive sequence (page numbers): {sequence_page_nums}")
				logger.info(f"At document positions: {sequence_indexes}")
				logger.info(
					f"Using anchor: model page {anchor_model_page} "
					f"at image index {anchor_original_index}"
				)
			elif summaries_with_model_pages:
				# Fallback to first page with a number if no sequence > threshold
				anchor_item = summaries_with_model_pages[0]
				anchor_model_page = anchor_item["model_page_number_int"]
				anchor_original_index = anchor_item["original_input_order_index"]
				logger.info(
					f"No consecutive sequence found. Using first detected page "
					f"{anchor_model_page} at image index {anchor_original_index} as anchor"
				)
		else:
			logger.info("No valid model page numbers detected. Using default numbering starting from 1.")

		# Apply page number adjustment to all summaries
		final_ordered_summaries = []
		for s_wrapper in parsed_summaries:
			original_index = s_wrapper["original_input_order_index"]

			# Ensure the 'summary' key exists and is a dictionary in the data part
			if "summary" not in s_wrapper["data"] or not isinstance(
					s_wrapper["data"]["summary"], dict):
				s_wrapper["data"]["summary"] = {}
			summary_data_dict = s_wrapper["data"]["summary"]
			# Identify the target dict that actually holds page_number (nested summary JSON preferred)
			target_summary_json = summary_data_dict.get("summary") if isinstance(summary_data_dict.get("summary"), dict) else summary_data_dict

			# Ensure page_number object structure exists
			if "page_number" not in target_summary_json or not isinstance(
					target_summary_json["page_number"], dict):
				target_summary_json["page_number"] = {
					"page_number_integer": 0,
					"contains_no_page_number": False
				}

			if s_wrapper["is_genuinely_unnumbered"]:
				target_summary_json["page_number"][
					"page_number_integer"] = 0  # Mark as unnumbered
				target_summary_json["page_number"][
					"contains_no_page_number"] = True
			else:
				# Default to original_input_order_index + 1 if no reliable anchor logic could be applied
				adjusted_page = original_index + 1
				if summaries_with_model_pages:  # Only use anchor logic if valid anchors were derived
					adjusted_page = self._calculate_adjusted_page_number(
						original_index, anchor_model_page, anchor_original_index)
				adjusted_page = max(1, adjusted_page)  # Ensure numbered pages start at 1
				target_summary_json["page_number"]["page_number_integer"] = adjusted_page
				target_summary_json["page_number"]["contains_no_page_number"] = False

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
			try:
				cl = ConfigLoader()
				cl.load_configs()
				target_dpi = int(cl.get_image_processing_config().get('api_image_processing', {}).get('target_dpi', 300))
			except Exception:
				target_dpi = None
		try:
			cfg_loader = ConfigLoader()
			cfg_loader.load_configs()
			concurrency_cfg = cfg_loader.get_concurrency_config()
			actual_concurrency = concurrency_cfg.get("api_requests", {}).get("transcription", {}).get("concurrency_limit", 4)
		except Exception:
			actual_concurrency = config.CONCURRENT_REQUESTS
		initialize_log_file(
			self.log_path, self.name, str(self.input_path), item_type_str,
			self.total_items_to_transcribe, config.OPENAI_TRANSCRIPTION_MODEL,
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
				# Step 1: Adjust page numbers and sort (handles 'contains_no_page_number')
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
