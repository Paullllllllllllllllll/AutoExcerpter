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
	append_to_log
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

# Constants for retry logic
RETRY_RATE_LIMIT_PER_2_SECONDS = 1
RETRY_RATE_LIMIT_PER_MINUTE = 60
RETRY_RATE_LIMIT_PER_HOUR = 3000
RETRY_MAX_ATTEMPTS = 3
RETRY_TIMEOUT_MULTIPLIER = 1.5

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
	) -> Tuple[
		List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[int, Path]]]:
		transcription_results: List[Dict[str, Any]] = []
		summary_results: List[Dict[str, Any]] = []
		failed_items_to_retry: List[
			Tuple[int, Path]] = []  # (original_input_order_index, path)
		total_images = len(image_paths)
		processed_count = 0
		# Removed unused stats tracking variables

		logger.info(
			f"Starting transcription{' and summarization' if config.SUMMARIZE else ''} of {total_images} images...")

		image_paths_with_indices = list(enumerate(image_paths))

		if config.SUMMARIZE and self.openai_summary_manager:
			initialize_log_file(
				self.summary_log_path, self.name, str(self.input_path),
				"PDF" if self.input_type == "pdf" else "Image Folder",
				total_images, config.OPENAI_MODEL
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

				eta_str = "ETA: N/A"
				if processed_count > 5 and self.start_time_processing:
					elapsed_total = time.time() - self.start_time_processing
					items_per_sec_overall = processed_count / elapsed_total
					remaining_items = total_images - processed_count
					if items_per_sec_overall > 0:
						recent_avg_time_per_item = sum(
							self.transcription_times[-10:]) / min(
							len(self.transcription_times),
							10) if self.transcription_times else 0
						items_per_sec_recent = 1.0 / recent_avg_time_per_item if recent_avg_time_per_item > 0 else items_per_sec_overall
						blended_ips = 0.7 * items_per_sec_overall + 0.3 * items_per_sec_recent
						eta_seconds = remaining_items / blended_ips if blended_ips > 0 else float(
							'inf')
						if eta_seconds != float('inf'):
							eta_str = f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}"

				logger.info(
					f"Processed {processed_count}/{total_images} - Item {item_num_str} - Status: {status} - {eta_str}")
				transcription_results.append(transcription_result)

				if "error" in transcription_result and transcription_result.get(
						"error_type") != "processing":
					failed_items_to_retry.append(
						(original_input_order_index, img_path))

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

		max_workers = min(config.CONCURRENT_REQUESTS, len(image_paths))
		if max_workers <= 0:
			max_workers = 1
		with concurrent.futures.ThreadPoolExecutor(
				max_workers=max_workers) as executor:
			list(tqdm(executor.map(submit_task, image_paths_with_indices),
			          total=total_images,
			          desc="Processing images"))

		transcription_results.sort(
			key=lambda x: x.get("original_input_order_index", 0))
		return transcription_results, summary_results, failed_items_to_retry

	def _adjust_and_sort_summary_page_numbers(self, summary_results: List[
		Dict[str, Any]]) -> List[Dict[str, Any]]:
		if not summary_results:
			return []
		
		logger.info("Adjusting page numbers based on model-detected page sequences...")
		parsed_summaries = []
		for r in summary_results:
			# Extract model_page_number from the summary data structure
			summary_container = r.get("summary", {})
			# Prefer nested 'summary' JSON if present
			inner_summary = summary_container.get("summary") if isinstance(summary_container, dict) else None
			if isinstance(inner_summary, dict):
				page_number_obj = inner_summary.get('page_number', {})
			else:
				page_number_obj = summary_container.get('page_number', {})
			
			# Get the page number integer and unnumbered flag from the correct location
			model_page_num = None
			is_genuinely_unnumbered = False
			
			if isinstance(page_number_obj, dict):
				# New schema format with nested page_number object
				model_page_num = page_number_obj.get('page_number_integer')
				is_genuinely_unnumbered = page_number_obj.get('contains_no_page_number', False)
			else:
				# Fallback for old format or direct model_page_number value
				model_page_str = r.get("model_page_number", "")
				try:
					if isinstance(model_page_str, (int, float)):
						model_page_num = int(model_page_str)
					elif isinstance(model_page_str, str) and model_page_str.isdigit():
						model_page_num = int(model_page_str)
				except ValueError:
					pass  # model_page_num remains None
				
				# Check old schema for unnumbered flag on the container if present
				is_genuinely_unnumbered = bool(summary_container.get('contains_no_page_number', False))
			
			parsed_summaries.append({
				"original_input_order_index": r["original_input_order_index"],
				"model_page_number_int": model_page_num,
				"data": r,
				"is_genuinely_unnumbered": is_genuinely_unnumbered
			})

		# Pages used for anchor calculation must have a model page number AND not be marked as genuinely unnumbered
		summaries_with_model_pages = [
			s for s in parsed_summaries
			if s["model_page_number_int"] is not None and not s["is_genuinely_unnumbered"]
		]
		
		logger.info(f"Found {len(summaries_with_model_pages)} pages with valid model-detected page numbers")
		
		anchor_model_page = 1
		anchor_original_index = 0

		if summaries_with_model_pages:
			# Sort by model page number to find the longest consistent sequence
			summaries_with_model_pages.sort(key=lambda x: x["model_page_number_int"])
			
			# Debug the detected page numbers
			detected_page_nums = [s["model_page_number_int"] for s in summaries_with_model_pages]
			logger.info(f"Model-detected page numbers: {detected_page_nums}")
			
			longest_sequence = []
			current_sequence = []
			
			for i, item in enumerate(summaries_with_model_pages):
				# Start a new sequence or add to current if consecutive
				if not current_sequence:
					current_sequence = [item]
				elif item["model_page_number_int"] == current_sequence[-1]["model_page_number_int"] + 1:
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
			
			if longest_sequence and len(longest_sequence) > 1:  # Prefer longer sequences
				anchor_item = longest_sequence[0]
				anchor_model_page = anchor_item["model_page_number_int"]
				anchor_original_index = anchor_item["original_input_order_index"]
				
				# Debug the found sequence
				sequence_page_nums = [s["model_page_number_int"] for s in longest_sequence]
				sequence_indexes = [s["original_input_order_index"] for s in longest_sequence]
				logger.info(f"Longest consecutive sequence: {sequence_page_nums}")
				logger.info(f"Corresponding image indexes: {sequence_indexes}")
				logger.info(f"Using anchor: model page {anchor_model_page} at image index {anchor_original_index}")
			elif summaries_with_model_pages:  # Fallback to the first page with a number if no sequence > 1
				anchor_item = summaries_with_model_pages[0]
				anchor_model_page = anchor_item["model_page_number_int"]
				anchor_original_index = anchor_item["original_input_order_index"]
				logger.info(f"No consecutive sequence found. Using first detected page {anchor_model_page} at image index {anchor_original_index} as anchor")
		else:
			logger.info("No valid model page numbers detected. Using default numbering starting from 1.")
		# If no summaries_with_model_pages, default anchors (1, 0) are used.

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
					adjusted_page = anchor_model_page + (original_index - anchor_original_index)
				adjusted_page = max(1, adjusted_page)  # Ensure numbered pages start at 1
				target_summary_json["page_number"]["page_number_integer"] = adjusted_page
				target_summary_json["page_number"]["contains_no_page_number"] = False

			final_ordered_summaries.append(s_wrapper["data"])

		# Sort by original input order to preserve document sequence, regardless of page_number
		final_ordered_summaries.sort(
			key=lambda x: x.get("original_input_order_index", float('inf'))
		)
		return final_ordered_summaries

	def _retry_transcription(self, failed_items: List[Tuple[int, Path]]) -> \
			Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
		"""Retry transcription for failed items with adjusted settings"""
		logger.info(
			f"--- RETRY PHASE for {self.name}: {len(failed_items)} failed items ---")
		# Sort by page number for ordered retries
		failed_items.sort(key=lambda x: int(x[0]))  # x[0] is sequence_num

		# More conservative rate limiter for retries (re-use OpenAI limits)
		retry_rate_limiter = RateLimiter([(1, 2), (60, 60), (3000, 3600)])
		retry_api_manager = OpenAITranscriptionManager(
			config.OPENAI_API_KEY,
			config.OPENAI_TRANSCRIPTION_MODEL,
			rate_limiter=retry_rate_limiter,
			timeout=int(config.OPENAI_API_TIMEOUT * 1.5),
		)

		logger.info(
			f"Retrying {len(failed_items)} image(s) with adjusted settings...")
		retry_transcription_results = []
		retry_summary_results = []

		for original_index, img_path in tqdm(failed_items, desc="Retrying failed items", total=len(failed_items)):
			# Retry transcription
			retry_result = retry_api_manager.transcribe_image(
				img_path, max_retries=3)
			retry_result["original_input_order_index"] = original_index
			retry_transcription_results.append(retry_result)

			# Log retry result
			append_to_log(self.log_path, retry_result)

			# Generate summary for the retry if transcription succeeded and summarization is enabled
			if config.SUMMARIZE and self.openai_summary_manager:
				page_num_raw = retry_result.get("page")
				if isinstance(page_num_raw, int) and page_num_raw > 0:
					page_num = page_num_raw
				else:
					page_num = original_index + 1
				model_page_num = page_num_raw if isinstance(page_num_raw, int) else None
				contains_no_page_number = not isinstance(page_num_raw, int)
				if "error" not in retry_result:
					transcription_text = retry_result.get("transcription", "")
					if "[empty page]" in transcription_text or "[no transcription possible]" in transcription_text:
						summary_data = self._create_placeholder_summary(
							page_num,
							contains_no_page_number,
							["[Empty page or no transcription possible]"],
						)
					else:
						summary_data = self.openai_summary_manager.generate_summary(
							transcription_text, page_num
						)
				else:
					error_msg = retry_result.get("error", "Unknown error")
					summary_data = self._create_placeholder_summary(
						page_num,
						contains_no_page_number,
						[f"[Transcription failed after retry: {error_msg}]"],
						error_message=error_msg,
					)
				summary_error = summary_data.get("error") if isinstance(summary_data, dict) else None
				summary_result = self._build_summary_result(
					original_index,
					img_path.name,
					summary_data,
					page_num,
					model_page_num,
					summary_error,
				)
				retry_summary_results.append(summary_result)
				append_to_log(self.summary_log_path, summary_result)

		logger.info(f"--- RETRY PHASE for {self.name} complete. ---")
		return retry_transcription_results, retry_summary_results

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
		initialize_log_file(
			self.log_path, self.name, str(self.input_path), item_type_str,
			self.total_items_to_transcribe, config.OPENAI_TRANSCRIPTION_MODEL,
			target_dpi
		)

		# Process all images - transcribe and summarize
		transcription_results, summary_results, failed_items = self._transcribe_and_summarize(
			image_paths_to_process)

		# Retry phase for items that failed due to retryable API errors
		if failed_items:
			retry_transcription_results, retry_summary_results = self._retry_transcription(
				failed_items)

			# Update all_results with retry_results
			transcription_dict = {res.get("image"): res for res in
			                      transcription_results}

			for retry_trans_res in retry_transcription_results:
				if retry_trans_res.get("image") in transcription_dict:
					# Find and replace the original failed result
					for i, original_res in enumerate(transcription_results):
						if original_res.get("image") == retry_trans_res.get(
								"image"):
							transcription_results[i] = retry_trans_res
							break
				else:
					transcription_results.append(retry_trans_res)

			# Update summary results if summarization is enabled
			if config.SUMMARIZE and summary_results:
				summary_dict = {
					res.get("original_input_order_index"): res
					for res in summary_results
					if res.get("original_input_order_index") is not None
				}

				for retry_summary_res in retry_summary_results:
					idx = retry_summary_res.get("original_input_order_index")
					if idx in summary_dict:
						# Find and replace the original failed result
						for i, original_res in enumerate(summary_results):
							if original_res.get("original_input_order_index") == idx:
								summary_results[i] = retry_summary_res
								break
					else:
						summary_results.append(retry_summary_res)

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
