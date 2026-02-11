import concurrent.futures
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
from modules.context_resolver import resolve_summary_context, format_context_for_prompt
from modules.types import SummaryResult, TranscriptionResult
from processors.file_manager import (
    append_to_log,
    create_docx_summary,
    create_markdown_summary,
    finalize_log_file,
    initialize_log_file,
    write_transcription_to_text,
)
from processors.pdf_processor import (
    extract_pdf_pages_to_images,
    get_image_paths_from_folder,
)
from core.page_numbering import PageNumberProcessor
from core.resume import load_completed_pages, load_transcription_results_from_log


logger = setup_logger(__name__)

# Constants for ETA calculation
MIN_SAMPLES_FOR_ETA = 5
RECENT_SAMPLES_FOR_ETA = 10
ETA_BLEND_WEIGHT_OVERALL = 0.7
ETA_BLEND_WEIGHT_RECENT = 0.3


class ItemTranscriber:
	"""Process a single input item (PDF or image folder).

	Attributes:
		input_path: Source path of the item to process.
		input_type: Either "pdf" or "image_folder".
		base_output_dir: Base directory where outputs and working files are written.
		working_dir: Item-specific working directory containing images and logs.
		transcribe_manager: Manages image transcription via LLM API.
		summary_manager: Manages summarization via LLM API (if enabled).
		summary_context: Optional context string for guiding summarization focus.
	"""
	def __init__(self, input_path: Path, input_type: str,
	             base_output_dir: Path, summary_context: Optional[str] = None,
	             resume_mode: str = "skip",
	             completed_page_indices: Optional[Set[int]] = None):
		self.input_path = input_path
		self.input_type = input_type  # "pdf" or "image_folder"
		self.name = self.input_path.stem
		self.resume_mode = resume_mode
		self.completed_page_indices = completed_page_indices or set()

		self.base_output_dir = base_output_dir
		self.output_txt_path = self.base_output_dir / f"{self.name}.txt"
		self.output_summary_docx_path = self.base_output_dir / f"{self.name}.docx"
		self.output_summary_md_path = self.base_output_dir / f"{self.name}.md"

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
		self.summary_context = None
		if config.SUMMARIZE:
			# Resolve summary context: CLI/interactive context takes precedence,
			# then file-specific, folder-specific, or general context
			if summary_context:
				self.summary_context = summary_context
				logger.info(f"Using user-provided summary context: {summary_context[:50]}..." if len(summary_context) > 50 else f"Using user-provided summary context: {summary_context}")
			else:
				# Try hierarchical context resolution
				resolved_context, context_path = resolve_summary_context(input_file=input_path)
				if resolved_context:
					self.summary_context = format_context_for_prompt(resolved_context)
					logger.info(f"Resolved summary context from: {context_path}")
			
			self.summary_manager = SummaryManager(
				model_name=self.summary_model,
				provider=summary_provider,
				summary_context=self.summary_context,
			)

		# Page number processor for adjusting and sorting summary page numbers
		self.page_number_processor = PageNumberProcessor()

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
		"""Create a consistent flat summary result structure for downstream consumers.
		
		Merges API result fields with metadata at the same level (no nesting).
		"""
		# Start with metadata fields
		result = {
			"original_input_order_index": original_index,
			"image_filename": image_name,
		}
		
		# Add page number (prefer explicit, then from payload)
		if page_number is not None:
			result["page"] = page_number
		elif isinstance(summary_payload, dict) and "page" in summary_payload:
			result["page"] = summary_payload.get("page")
		
		# Merge content fields from summary_payload at top level
		if isinstance(summary_payload, dict):
			# Core content fields from API
			result["page_information"] = summary_payload.get("page_information")
			result["bullet_points"] = summary_payload.get("bullet_points")
			result["references"] = summary_payload.get("references")
			
			# Metadata fields from API
			if "processing_time" in summary_payload:
				result["processing_time"] = summary_payload["processing_time"]
			if "provider" in summary_payload:
				result["provider"] = summary_payload["provider"]
			if "api_response" in summary_payload:
				result["api_response"] = summary_payload["api_response"]
			if "schema_retries" in summary_payload:
				result["schema_retries"] = summary_payload["schema_retries"]
		
		if error_message:
			result["error"] = error_message
		
		return result

	def _create_placeholder_summary(
		self,
		page_number: Optional[int],
		page_number_type: str,
		bullet_points: Optional[List[str]],
		references: Optional[List[str]] = None,
		page_types: Optional[List[str]] = None,
		error_message: Optional[str] = None,
	) -> Dict[str, Any]:
		"""Create a flat placeholder summary structure."""
		if page_types is None:
			page_types = ["other"]
		
		# Flat structure - all fields at top level
		payload = {
			"page": page_number if page_number is not None else 0,
			"page_information": {
				"page_number_integer": page_number,
				"page_number_type": page_number_type,
				"page_types": page_types,
			},
			"bullet_points": bullet_points,
			"references": references,
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

		# --- Page-level resume: pre-load completed results and filter ---
		image_paths_with_indices = list(enumerate(image_paths))
		skipped_page_count = 0

		if self.completed_page_indices:
			# Load previously completed transcription results from log
			prior_results = load_transcription_results_from_log(self.log_path)
			if prior_results:
				for entry in prior_results:
					idx = entry.get("original_input_order_index")
					if isinstance(idx, int) and idx in self.completed_page_indices:
						transcription_results.append(entry)

			# Filter out already-completed pages
			pending = [
				(idx, path) for idx, path in image_paths_with_indices
				if idx not in self.completed_page_indices
			]
			skipped_page_count = len(image_paths_with_indices) - len(pending)
			image_paths_with_indices = pending

			if skipped_page_count > 0:
				logger.info(
					f"Page-level resume: {skipped_page_count} page(s) already transcribed, "
					f"{len(pending)} page(s) remaining"
				)

		if not image_paths_with_indices:
			logger.info("All pages already transcribed (page-level resume). Skipping transcription.")
			transcription_results.sort(
				key=lambda x: x.get("original_input_order_index", 0))
			return transcription_results, summary_results
		# --- End page-level resume ---

		logger.info(
			f"Starting transcription{' and summarization' if config.SUMMARIZE else ''} of "
			f"{len(image_paths_with_indices)} images"
			f"{f' ({skipped_page_count} skipped via resume)' if skipped_page_count else ''}...")

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
							None,  # null bullet_points for blank pages
							references=None,
							page_types=["blank"],
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
						references=None,
						page_types=["other"],
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
		transcription_results: list[dict[str, Any]] = []
		summary_results: list[dict[str, Any]] = []

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

			# Save summaries to output files if summarization is enabled
			if config.SUMMARIZE and summary_results:
				# Step 1: Adjust page numbers and sort (uses page_number_type for unnumbered detection)
				adjusted_summary_results = self.page_number_processor.adjust_and_sort_page_numbers(
					summary_results)

				try:
					# Generate DOCX if enabled
					if config.OUTPUT_DOCX:
						create_docx_summary(adjusted_summary_results,
						                    self.output_summary_docx_path, self.name)
					
					# Generate Markdown if enabled
					if config.OUTPUT_MARKDOWN:
						create_markdown_summary(adjusted_summary_results,
						                        self.output_summary_md_path, self.name)

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
				summary_outputs = []
				if config.OUTPUT_DOCX:
					summary_outputs.append(str(self.output_summary_docx_path))
				if config.OUTPUT_MARKDOWN:
					summary_outputs.append(str(self.output_summary_md_path))
				if summary_outputs:
					logger.info(f"  Final summary outputs: {', '.join(summary_outputs)}")
			logger.info(
				f"  Detailed logs: {self.log_path}{' and ' + str(self.summary_log_path) if config.SUMMARIZE else ''}")

			# Determine if we should cleanup images directory
			# Only delete if we have successful outputs (transcription and/or summary)
			if self.output_txt_path.exists():
				# We have a transcription output
				should_cleanup = True
			if config.SUMMARIZE:
				# Check if any summary output exists
				if (config.OUTPUT_DOCX and self.output_summary_docx_path.exists()) or \
				   (config.OUTPUT_MARKDOWN and self.output_summary_md_path.exists()):
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
