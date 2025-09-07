import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import re
from modules import app_config as config
import docx
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT


def sanitize_for_xml(text):
	"""Remove or replace characters that are not compatible with XML."""
	if not text:
		return ""
	
	# Remove control characters except for tabs and newlines
	text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
	
	# Replace special XML entities
	text = text.replace('&', '&amp;')
	text = text.replace('<', '&lt;')
	text = text.replace('>', '&gt;')
	text = text.replace('"', '&quot;')
	text = text.replace("'", '&apos;')
	
	return text


def create_docx_summary(summary_results: List[Dict[str, Any]],
                        output_path: Path, document_name: str):
	"""Create a compact, well-formatted DOCX document from summary results"""
	# Filter out pages with no useful content
	filtered_results = filter_empty_pages(summary_results)
	
	if len(filtered_results) < len(summary_results):
		print(f"Filtered out {len(summary_results) - len(filtered_results)} pages with no useful content")
	
	doc = docx.Document()

	style_normal = doc.styles['Normal']
	style_normal.paragraph_format.space_before = docx.shared.Pt(0)
	style_normal.paragraph_format.space_after = docx.shared.Pt(4)

	# Configure heading styles
	for i in range(1, 4):
		heading_style = doc.styles[f'Heading {i}']
		heading_style.paragraph_format.space_before = docx.shared.Pt(
			8) if i > 1 else docx.shared.Pt(12)
		heading_style.paragraph_format.space_after = docx.shared.Pt(4)

	# Add title with compact formatting
	title = doc.add_heading(f"Summary of {sanitize_for_xml(document_name)}", 0)
	title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
	title.paragraph_format.space_after = docx.shared.Pt(8)

	# Compact metadata line
	metadata = f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Pages: {len(filtered_results)}"
	meta_para = doc.add_paragraph(metadata)
	meta_para.paragraph_format.space_after = docx.shared.Pt(8)

	# Add content for each page
	for i, result in enumerate(filtered_results):
		# Handle the nested summary structure
		if "summary" in result:
			if isinstance(result["summary"], dict) and "summary" in result["summary"]:
				# Double-nested structure: result -> summary -> summary
				summary = result["summary"]["summary"]
			else:
				# Single-nested structure: result -> summary
				summary = result["summary"]
		else:
			summary = {}

		# Get the page number from the nested structure
		page_number_obj = summary.get("page_number", {})
		if isinstance(page_number_obj, dict):
			page_num = page_number_obj.get("page_number_integer", "?")
		else:
			# Fallback for old format
			page_num = summary.get("page_number", "?")

		bullet_points = summary.get("bullet_points", [])
		references = summary.get("references", [])

		# Add page header with compact spacing
		page_heading = doc.add_heading(f"Page {page_num}", 1)
		page_heading.paragraph_format.space_before = docx.shared.Pt(
			8) if i > 0 else docx.shared.Pt(4)
		page_heading.paragraph_format.space_after = docx.shared.Pt(4)

		# Add bullet points with compact formatting
		if bullet_points:
			for point in bullet_points:
				para = doc.add_paragraph()
				para.paragraph_format.space_before = docx.shared.Pt(0)
				para.paragraph_format.space_after = docx.shared.Pt(2)
				para.paragraph_format.left_indent = docx.shared.Pt(12)
				bullet = para.add_run("• ")
				bullet.bold = True
				para.add_run(sanitize_for_xml(point))
		else:
			no_points = doc.add_paragraph(
				"No bullet points available for this page.")
			no_points.paragraph_format.space_after = docx.shared.Pt(4)

		# Add references if any with compact formatting
		if references:
			ref_heading = doc.add_heading("References", 2)
			ref_heading.paragraph_format.space_before = docx.shared.Pt(6)
			ref_heading.paragraph_format.space_after = docx.shared.Pt(2)

			for ref in references:
				ref_para = doc.add_paragraph(sanitize_for_xml(ref))
				ref_para.paragraph_format.space_before = docx.shared.Pt(0)
				ref_para.paragraph_format.space_after = docx.shared.Pt(2)
				ref_para.paragraph_format.left_indent = docx.shared.Pt(12)

		# Add a separator instead of page break unless it's the last page
		if result != filtered_results[-1]:
			separator = doc.add_paragraph()
			separator.paragraph_format.space_before = docx.shared.Pt(4)
			separator.paragraph_format.space_after = docx.shared.Pt(4)
			separator.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

	# Save the document
	doc.save(output_path)
	print(f"Compact summary DOCX file saved: {output_path}")


def create_docx_transcription(transcription_results: List[Dict[str, Any]],
                              output_path: Path, document_name: str):
	"""Create a DOCX document from transcription results"""
	doc = docx.Document()

	# Add title
	title = doc.add_heading(f"Transcription of {sanitize_for_xml(document_name)}", 0)
	title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

	# Add processing information
	doc.add_paragraph(
		f"Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	doc.add_paragraph(f"Total pages: {len(transcription_results)}")
	doc.add_paragraph()  # Add space

	# Sort results by page number
	sorted_results = sorted(transcription_results,
	                        key=lambda x: x.get("page", 0))

	# Add content for each page
	for result in sorted_results:
		page_num = result.get("page", "?")
		transcription = result.get("transcription",
		                           "[No transcription available]")

		# Add page header
		doc.add_heading(f"Page {page_num}", 1)

		# Add transcription text
		doc.add_paragraph(sanitize_for_xml(transcription))

		# Add a page break unless it's the last page
		if result != sorted_results[-1]:
			doc.add_page_break()

	# Save the document
	doc.save(output_path)
	print(f"Transcription DOCX file saved: {output_path}")


def write_transcription_to_text(transcription_results: List[Dict[str, Any]],
                                output_path: Path,
                                document_name: str, item_type: str,
                                total_elapsed_time: float,
                                source_path: Path):
	"""Write transcription results to a text file"""
	elapsed_str = datetime.fromtimestamp(total_elapsed_time).strftime(
		"%H:%M:%S")
	final_success_count = sum(
		1 for r in transcription_results if "error" not in r)
	final_failure_count = len(transcription_results) - final_success_count

	try:
		with open(output_path, "w", encoding="utf-8") as f_out:
			f_out.write(f"# Transcription of: {document_name}\n")
			f_out.write(f"# Source Path: {source_path}\n")
			f_out.write(f"# Type: {item_type}\n")
			f_out.write(
				f"# Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			f_out.write(
				f"# Total images processed: {len(transcription_results)}\n")
			f_out.write(f"# Successfully transcribed: {final_success_count}\n")
			f_out.write(f"# Failed items: {final_failure_count}\n")
			f_out.write(
				f"# Total processing time for this item: {elapsed_str}\n\n---\n\n")

			for i, result in enumerate(transcription_results):
				f_out.write(result.get("transcription",
				                       "[ERROR] Transcription data missing"))
				if i < len(transcription_results) - 1:
					f_out.write("\n\n---\n\n")  # Separator between pages/images

		print(f"Transcription text file saved: {output_path}")
		return True
	except Exception as e:
		print(f"Error writing transcription to text file {output_path}: {e}")
		return False


def initialize_log_file(log_path: Path, item_name: str, input_path: str,
                        input_type: str,
                        total_images: int, model_name: str,
                        extraction_dpi=None):
	"""Initialize a JSON log file with header information"""
	try:
		with open(log_path, "w", encoding="utf-8") as f_log:
			log_header = {
				"input_item_name": item_name,
				"input_item_path": input_path,
				"input_type": input_type,
				"processing_start_time": datetime.now().isoformat(),
				"total_images": total_images,
				"configuration": {
					"concurrent_requests": config.CONCURRENT_REQUESTS,
					"api_timeout_seconds": config.API_TIMEOUT,
					"model_name": model_name,
					"extraction_dpi": extraction_dpi if extraction_dpi else "N/A",
					"openai_flex_processing": config.OPENAI_USE_FLEX if model_name == config.OPENAI_MODEL else "N/A",
				},
			}
			json.dump(log_header, f_log)
			f_log.write("\n")  # Newline for subsequent JSON entries
		return True
	except Exception as e:
		print(f"Warning: Failed to initialize log file {log_path}: {e}")
		return False


def append_to_log(log_path: Path, entry: Dict):
	"""Append a JSON entry to a log file"""
	try:
		with open(log_path, "a", encoding="utf-8") as f_log:
			f_log.write(json.dumps(entry) + "\n")
		return True
	except Exception as e:
		print(f"Warning: Failed to write to log file {log_path}: {e}")
		return False


def verify_page_numbering(summary_results: List[Dict[str, Any]]) -> List[
	Dict[str, Any]]:
	"""Verify and correct page numbering in the summary results"""
	if not summary_results:
		return summary_results

	print("\nVerifying page numbering consistency...")
	
	# Sort results by the original page extraction sequence
	sorted_results = sorted(summary_results, key=lambda x: x.get("page", 0))

	# Extract page numbers and check for unnumbered pages
	detected_page_numbers = []
	unnumbered_indices = []
	
	for i, r in enumerate(sorted_results):
		page_number_obj = r.get("summary", {}).get("page_number", {})
		if isinstance(page_number_obj, dict):
			contains_no_page_number = page_number_obj.get("contains_no_page_number", False)
			if contains_no_page_number:
				unnumbered_indices.append(i)
				detected_page_numbers.append(0)  # Placeholder for unnumbered pages
			else:
				page_num = page_number_obj.get("page_number_integer", 0)
				detected_page_numbers.append(page_num)
		else:
			# Fallback for old format
			page_num = r.get("summary", {}).get("page_number", 0)
			detected_page_numbers.append(page_num)

	# Ignore unnumbered pages for consistency check
	non_zero_pages = [p for i, p in enumerate(detected_page_numbers) if i not in unnumbered_indices and p != 0]
	
	# Look for empty (0) or duplicated page numbers
	has_zeros = any(p == 0 for i, p in enumerate(detected_page_numbers) if i not in unnumbered_indices)
	has_duplicates = len(non_zero_pages) != len(set(non_zero_pages))

	# Find the longest consecutive sequence of page numbers
	longest_sequence = []
	if len(non_zero_pages) > 1:
		unique_sorted_pages = sorted(set(non_zero_pages))
		current_sequence = []
		
		for num in unique_sorted_pages:
			if not current_sequence or num == current_sequence[-1] + 1:
				current_sequence.append(num)
			else:
				if len(current_sequence) > len(longest_sequence):
					longest_sequence = current_sequence.copy()
				current_sequence = [num]
				
		# Check the last sequence
		if len(current_sequence) > len(longest_sequence):
			longest_sequence = current_sequence

	# Check if numbering sequence is broken (has large gaps or inconsistent)
	is_sequential = False
	if longest_sequence:
		# Consider the sequence valid if it covers at least 50% of the non-zero pages
		is_sequential = len(longest_sequence) >= len(non_zero_pages) * 0.5
	
	print(f"Detected page numbers: {detected_page_numbers}")
	print(f"Longest consecutive sequence: {longest_sequence}")
	print(f"Has zeros: {has_zeros}, Has duplicates: {has_duplicates}, Is sequential: {is_sequential}")

	# Only intervene if we have clear evidence of page numbering problems
	if has_zeros or has_duplicates or not is_sequential:
		print("Page numbering issues detected. Validating against longest sequence...")
		
		# If we have a valid longest sequence, use it as anchor for adjustments
		if longest_sequence and len(longest_sequence) >= 2:
			print(f"Using longest consecutive sequence {longest_sequence} as reference")
			
			# Find where this sequence appears in our document
			sequence_start_val = longest_sequence[0]
			sequence_positions = []
			
			for i, page_num in enumerate(detected_page_numbers):
				if page_num == sequence_start_val and i not in unnumbered_indices:
					# Check if this is the start of the sequence
					is_sequence_start = True
					for j, seq_val in enumerate(longest_sequence):
						if i+j >= len(detected_page_numbers) or detected_page_numbers[i+j] != seq_val:
							is_sequence_start = False
							break
					if is_sequence_start:
						sequence_positions.append(i)
			
			if sequence_positions:
				first_occurrence = sequence_positions[0]
				print(f"Sequence starts at position {first_occurrence} in document")
				
				# Use this sequence as anchor to adjust all page numbers
				for i, result in enumerate(sorted_results):
					# Skip genuinely unnumbered pages
					page_number_obj = result.get("summary", {}).get("page_number", {})
					if isinstance(page_number_obj, dict) and page_number_obj.get("contains_no_page_number", False):
						continue
						
					# Calculate expected page number based on position relative to sequence
					expected_page = sequence_start_val + (i - first_occurrence)
					if expected_page < 1:
						expected_page = 1  # Ensure we don't have page numbers less than 1
						
					# Update page number in the object
					summary_dict = result.get("summary", {})
					if "page_number" not in summary_dict or not isinstance(summary_dict["page_number"], dict):
						summary_dict["page_number"] = {}
						
					summary_dict["page_number"]["page_number_integer"] = expected_page
					summary_dict["page_number"]["contains_no_page_number"] = False
					
					# Also update the top-level page number if present
					if "page" in result:
						result["page"] = expected_page
			else:
				print("Could not locate sequence in document. Using extraction order as fallback.")
				_use_sequential_numbering(sorted_results, unnumbered_indices)
		else:
			print("No reliable sequence found. Using extraction order as fallback.")
			_use_sequential_numbering(sorted_results, unnumbered_indices)
	else:
		print("Page numbering appears consistent. No corrections needed.")

	return sorted_results

def _use_sequential_numbering(sorted_results, unnumbered_indices):
	"""Helper function to use sequential numbering based on extraction order"""
	print("Using sequential numbering based on extraction order")
	page_num = 1
	
	for i, result in enumerate(sorted_results):
		# Skip genuinely unnumbered pages
		if i in unnumbered_indices:
			continue
			
		# Get the summary object
		summary_dict = result.get("summary", {})
		
		# Update page number in the new schema format
		if "page_number" not in summary_dict or not isinstance(summary_dict["page_number"], dict):
			summary_dict["page_number"] = {}
			
		summary_dict["page_number"]["page_number_integer"] = page_num
		summary_dict["page_number"]["contains_no_page_number"] = False
		
		# Also update the top-level page number
		if "page" in result:
			result["page"] = page_num
			
		page_num += 1
	
	print(f"Corrected page numbers: {[r['summary']['page_number']['page_number_integer'] for r in sorted_results if not r.get('summary', {}).get('page_number', {}).get('contains_no_page_number', False)]}")


def filter_empty_pages(summary_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Filter out pages with no useful content"""
	filtered_results = []
	
	for result in summary_results:
		# Extract summary data from the nested structure
		summary_data = {}
		if "summary" in result:
			if isinstance(result["summary"], dict) and "summary" in result["summary"]:
				summary_data = result["summary"]["summary"]
			else:
				summary_data = result["summary"]
		
		# Check for empty or non-useful pages
		should_include = True
		
		# Condition 1: Check for page with no page number
		page_number_obj = summary_data.get("page_number", {})
		if isinstance(page_number_obj, dict):
			page_num = page_number_obj.get("page_number_integer", 0)
			if page_num == 0:
				should_include = False
		elif isinstance(page_number_obj, int) and page_number_obj == 0:
			should_include = False
			
		# Condition 2: Check for empty bullet points
		bullet_points = summary_data.get("bullet_points", [])
		if not bullet_points:
			should_include = False
			
		# Condition 3: Check for bullet points indicating empty pages
		if bullet_points and len(bullet_points) == 1:
			content = bullet_points[0].strip().lower()
			if any(marker in content for marker in ["[empty page", "no transcription possible", "empty page", "error"]):
				should_include = False
				
		# Condition 4: Check for pages explicitly marked as having no semantic content
		if summary_data.get("contains_no_semantic_content", False):
			should_include = False
			
		# Include only meaningful pages
		if should_include:
			filtered_results.append(result)
			
	return filtered_results