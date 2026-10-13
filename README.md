# AutoExcerpter v2.3.1

AutoExcerpter transcribes PDFs and folders of page images with vision-language
models, summarizes each page and builds a consolidated bibliography whose entries
are enriched from OpenAlex. It runs on OpenAI, Anthropic, Google and OpenRouter
models and on OpenAI-compatible endpoints that you name in the settings. The
`run` command serves scripts and agents; the `ui` command walks you through a run
in the terminal. `run --batch` submits the pages to the providers' batch APIs
instead, and `batch check` collects them. `render` writes the files again from
the store without model calls, and `migrate` imports the outputs of version 1.0.
Please report bugs on the
[GitHub issue tracker](https://github.com/Paullllllllllllllllll/AutoExcerpter/issues).

## Overview

An item is one PDF or one folder of page images (PNG, JPEG, TIFF, BMP, GIF or
WebP). In the transcription stage, every page is rendered or loaded in memory,
sized to the model's image limits and sent to the transcription model, which
returns the page text with Markdown structure, LaTeX math, footnotes, figure
descriptions and the printed page number. AutoExcerpter cleans the text and writes
the pages in scan order. In the summary stage (on by default), the summary model
condenses each content page into bullet points and lists the references it cites.
AutoExcerpter then corrects page numbers across the document, merges duplicate
citations, looks them up in OpenAlex and renders the summary as Markdown and Word
files.

Every page result goes into the output folder's database `autoexcerpter.sqlite`
the moment it arrives, together with every provider response as received. The
files a user reads are views rendered from that store: for an input `X.pdf` or an
image folder `X/`, a run writes `X_transcription.md`, `X_summary.md` and
`X_summary.docx` ([Output Contract](#output-contract)).

## Installation

AutoExcerpter needs Python 3.13 or later and [uv](https://docs.astral.sh/uv/).
Install it from a clone, since the settings files are read from the repository's
`config/` folder.

```bash
git clone https://github.com/Paullllllllllllllllll/AutoExcerpter.git
cd AutoExcerpter
uv sync                      # development: uv sync --all-extras
cp config/settings.example.yaml config/settings.yaml
```

On Windows, copy the file with `copy config\settings.example.yaml
config\settings.yaml`. Then set the API key of each provider you use as an
environment variable. The settings name the variables (`api_keys:`); the defaults
are `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` and
`OPENROUTER_API_KEY`. An OpenAlex key in `OPENALEX_API_KEY` is optional.

```bash
export OPENAI_API_KEY="sk-..."        # PowerShell: $env:OPENAI_API_KEY = "sk-..."
```

## Quick Start

```bash
uv run autoexcerpter run --input paper.pdf           # outputs beside the input
uv run autoexcerpter run --input scans --all --output excerpts
uv run autoexcerpter run --input scans --select 1-3  # or a name: --select braudel
uv run autoexcerpter run --input scans --all --dry-run --json
uv run autoexcerpter ui                              # guided run
uv run autoexcerpter run --input scans --all --output excerpts --batch
uv run autoexcerpter batch check --input excerpts    # collect finished batches
uv run autoexcerpter batch check --input excerpts --wait  # ... until all are in
uv run autoexcerpter render --input excerpts         # rewrite the files, no calls
uv run autoexcerpter migrate --input excerpts        # import version 1.0 outputs
```

A folder input is searched recursively; item numbers count in discovery order,
PDFs first. Relative paths resolve against the current directory. With
`uv --directory <clone> run autoexcerpter ...` the current directory is the
clone, so pass absolute paths.

## Command Reference

The tables list every option of `autoexcerpter run --help` with its help wording
and default. They mirror the option table (`autoexcerpter/spec.py`), and
`tests/cli/test_readme_options.py` checks them against it. Options without a
stated default are off or unset until given. The last column says whether the
settings `defaults:` block can set the option; its key there is the flag without
the leading dashes, with underscores for hyphens (`--transcription-format` becomes
`transcription_format`). `autoexcerpter ui` takes only `--settings FILE`;
`batch check` and `batch cancel` are described under
[Batch Processing](#batch-processing), `render` and `migrate` under
[Store](#store).

### Input

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input PATH` | PDF file, image folder, or folder of PDFs and image folders | | yes |
| `--select EXPR` | select items by number (1,3), range (1-5) or name search | | yes |
| `--all` | process every discovered item | | yes |

`--input` is required. A single discovered item needs neither `--select` nor
`--all`; several items without either are a usage error. A name search matches
part of a file or folder name, ignoring case.

### Output

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--output DIR` | output folder | beside the input | yes |
| `--beside-input` | write outputs beside each input | | no |
| `--transcription-format {md,txt}` | transcription file format | md | yes |
| `--formats LIST` | comma-separated output formats; from summary-md, summary-docx | summary-md,summary-docx | yes |

The transcription file is always written. `--beside-input` overrides an `output`
settings default.

### Summary

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--summarize`, `--no-summarize` | summarize after transcribing | on | yes |
| `--context TEXT\|PATH\|auto\|none` | summary context: topics as text or a path to a text file (overrides sidecars), auto (the item's file or folder sidecar) or none (no context) | auto | yes |
| `--fallback-context TEXT\|PATH\|none` | summary context for items without a sidecar under --context auto: topics as text, a path to a text file, or none | none | yes |

Sidecars are text files with one topic per line: `X_summary_context.txt` beside
the input, else `F_summary_context.txt` beside the folder `F` that holds the
input. A value that names an existing file is read as a file, any other value as
context text.

### Citations

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--openalex`, `--no-openalex` | enrich citations with OpenAlex metadata | on | yes |

### Model

The unprefixed flag sets both phases; the `--transcription-*` and `--summary-*`
flags set one phase and win over it. The `defaults:` block can set every model
option, for example as `model` or `summary_verbosity`.

| Setting | Values | Both phases | Transcription | Summary |
| --- | --- | --- | --- | --- |
| model provider | `{openai,anthropic,google,openrouter,custom}` | `--provider`: inferred from the model name | `--transcription-provider`: inferred from the model name | `--summary-provider`: inferred from the model name |
| model name; a provider: prefix (anthropic:NAME) selects the provider | `NAME` | `--model`: per phase | `--transcription-model`: gpt-5.6-luna | `--summary-model`: gpt-5.6-luna |
| named custom endpoint from the settings (implies provider custom) | `NAME` | `--endpoint`: none | `--transcription-endpoint`: none | `--summary-endpoint`: none |
| reasoning effort | `{none,minimal,low,medium,high,xhigh,max}` | `--reasoning-effort`: per phase | `--transcription-reasoning-effort`: high | `--summary-reasoning-effort`: high |
| output verbosity | `{low,medium,high}` | `--verbosity`: per phase | `--transcription-verbosity`: high | `--summary-verbosity`: low |
| maximum output tokens | `N` | `--max-output-tokens`: the model's output limit | `--transcription-max-output-tokens`: the model's output limit | `--summary-max-output-tokens`: the model's output limit |
| sampling temperature, 0.0 to 2.0 | `T` | `--temperature`: per phase | `--transcription-temperature`: 1.0 | `--summary-temperature`: 1.0 |
| nucleus sampling, 0.0 to 1.0; sent only when given and supported | `P` | `--top-p`: not sent | `--transcription-top-p`: not sent | `--summary-top-p`: not sent |
| OpenAI service tier; only for provider openai | `{auto,default,flex,priority}` | `--service-tier`: per phase | `--transcription-service-tier`: flex | `--summary-service-tier`: flex |
| schema (API-enforced), json (prompted, validated) or text (plain) | `{schema,json,text}` | `--response-format`: schema when the model supports it, else json | `--transcription-response-format`: schema when the model supports it, else json | `--summary-response-format`: schema when the model supports it, else json |

### Images

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--dpi native\|N` | render resolution: native or a DPI value | the provider's recommended value | yes |
| `--image-detail {low,high,auto,original}` | image detail sent to the model | the provider's recommended value | yes |
| `--image-format {jpeg,png}` | page image format | the provider's recommended value | yes |
| `--jpeg-quality N` | JPEG quality, 1 to 100 | the provider's recommended value | yes |
| `--grayscale`, `--no-grayscale` | convert page images to grayscale | the provider's recommended value | yes |

### Run Control

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--force` | reprocess items whose outputs exist (resume is the default) | | no |
| `--retranscribe` | transcribe again instead of reusing stored transcriptions | | no |
| `--concurrency N` | parallel page requests | 16 | yes |
| `--batch`, `--no-batch` | submit the pages and summaries to the providers' batch APIs | off | yes |
| `--dry-run` | plan without API calls or writes | | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

### Exit Codes and JSON Summary

| Code | Meaning |
| --- | --- |
| 0 | every item completed or, under `--batch`, was submitted, or nothing was left to do |
| 1 | an item failed, or no item could be processed (no PDF or image folder found, `--select` matched nothing, a refused or unresolved batch submission, an unexpected error) |
| 2 | usage or configuration error (invalid flags or settings, several items without `--all` or `--select`, two items with one output name in one folder, a stored document in the output folder that belongs to another input of the same name, summaries on without a summary format, an unset API key variable, a database the store refuses, `--batch` on a provider without a batch API) |
| 130 | interrupted (Ctrl+C) |

Progress, warnings and errors go to stderr. With `--json`, stdout carries exactly
one JSON line on every exit, errors and interrupts included. Each line holds
`dry_run`, which is true whenever `--dry-run` was given. A run summary adds:

| Field | Content |
| --- | --- |
| `items_total` | items processed in this run |
| `items_complete` | items that finished with every page and output |
| `items_failed` | items that finished incomplete |
| `items_skipped` | items skipped because they were already complete |
| `items_submitted` | items whose units wait in batches for `batch check` (`--batch`); they count neither complete nor failed |
| `outputs` | absolute paths of the files written, item by item, then the databases |
| `usage` | tokens per role (`transcription`, `summary`) and their sum under `total`, each with `input_tokens`, `output_tokens`, `total_tokens`, `cached_tokens` (input read from the provider's prompt cache) and `reasoning_tokens` |

A dry run that plans successfully prints the plan instead: `to_process` lists each
item's `name`, resume `state` and `completed_pages`; `skipped` lists the complete
items; `spec` maps every field of the effective run specification, such as
`transcription_model.model`, to its `value` and `source` (`flag`, `settings` or
`default`). The plan has no `usage` field, since a dry run makes no calls.

## Guided UI

`autoexcerpter ui` first asks for the task: a new run, checking or cancelling
batches (`batch check`, `batch cancel`), rendering the files of stored documents
again (`render`) or importing version 1.0 outputs (`migrate`). A task chosen
there opens at its first step, and Back from it returns to the task question.
The batch check task asks for the folder, the batch IDs and what to do with an
unresolved submission, and makes one pass (`--wait` is a command-line option);
the batch cancel task asks what to cancel. The render
task asks for the input, the formats, whether to
replace files changed outside the tool and whether to dry-run; the migrate task
asks for the input and whether to dry-run, and its review shows what a dry run
would import. The new run asks one step at a time: input (with the PDFs, image
folders and pages found, and the item selection), output location, summary on or
off, formats, summary context, transcription model, summary model, page images
("Recommended" or "Customize") and run options, which offer `--batch` when every
phase's provider has a batch API. Every list ends with Back; a text
or path prompt takes `<` alone as Back. Settings defaults are preselected, and
answers act like flags over them. The review screen marks values taken from
settings defaults with `(s)`, previews the plan and prints the equivalent command:

```text
Review                                            (s) = from settings defaults
  input                 /data/docs/doc.pdf  (1 PDF, 3 pages)
  transcription model   openai / gpt-5.6-terra (s) | reasoning high | tier priority (s)
  run                   resume | OpenAlex on | concurrency 16

Plan: 1 new, 0 partial, 0 complete
  doc: new

Equivalent command:
  autoexcerpter run --input /data/docs/doc.pdf --model gpt-5.6-terra
    --service-tier priority
? Start:
 > Run
   Edit a step
   Copy command and quit
   Quit
   Back
```

The command spells out every value that differs from the code default, settings
defaults included, so it reproduces the run on another machine. On Windows it is
quoted for PowerShell (single quotes), elsewhere for a POSIX shell. Run shows
progress bars and ends with a summary table. Edit a step returns to one step;
the answers of steps it turns off come back when they apply again. Copy command
and quit prints the command on stdout and copies it to the clipboard where one is
available. Quit exits. When items are already complete, the review offers to reprocess them
(`--force`). Run is withheld while the plan fails or a needed API key variable is
unset; the review names the variable. Under `--batch` the plan fails, and Run is
withheld, when a provider or the extension refuses batch submission. The UI needs
an interactive terminal on
stdin and stdout; anything else exits with code 2.

## Settings File

AutoExcerpter reads `config/settings.yaml` (gitignored). When it is missing, the
tool reads the tracked `config/settings.example.yaml` and logs one informational
line that asks you to copy it. `--settings FILE` reads another file, which must
exist. Every key is optional and keeps its code value when missing; unknown keys
are an error that names the file and the key. The example file, with its
`defaults:` block shown uncommented:

```yaml
api_keys: {openai: OPENAI_API_KEY, anthropic: ANTHROPIC_API_KEY, google: GOOGLE_API_KEY, openrouter: OPENROUTER_API_KEY}
endpoints: {}                       # named endpoints, see Providers
timeouts: {request: 900, connect: 10, write: 30, pool: 30}   # seconds
rate_limits: [[10, 1], [600, 60], [600, 3600]]   # [max_requests, window_seconds]
retry:
  max_attempts: 8                   # every call, the first included
  backoff_base: 0.5
  backoff_cap: 120
  backoff_multipliers: {rate_limit: 2.0, timeout: 1.5, connection: 1.5, server_error: 2.0, other: 2.0, validation: 1.5}
  jitter: {min: 0.5, max: 1.0}
  schema_retries:
    transcription:
      validation_failure: {enabled: true, max_attempts: 3, backoff_base: 0.5, backoff_multiplier: 1.5}
      no_transcribable_text: {enabled: false, max_attempts: 0, backoff_base: 0.1, backoff_multiplier: 1.5}
      transcription_not_possible: {enabled: true, max_attempts: 3, backoff_base: 0.1, backoff_multiplier: 1.5}
    summary:
      validation_failure: {enabled: true, max_attempts: 3, backoff_base: 0.5, backoff_multiplier: 1.5}
state_dir: ''                       # '' means ~/.autoexcerpter
openalex: {email: '', api_key_env: OPENALEX_API_KEY, max_requests: 300}
images: {original_max_side_px: null}   # null keeps the model's default
defaults:
  model: gpt-5.6-terra
  reasoning_effort: high
  transcription_verbosity: high
  summary_verbosity: low
  service_tier: flex
  transcription_format: md
  formats: [summary-md, summary-docx]
  concurrency: 16
  batch: false
  fallback_context: path/to/summary_topics.txt
```

`api_keys` names the variable that holds each provider's key, never the key.
`timeouts.request` is the read timeout; 900 s suits the OpenAI flex tier, which
may queue requests. Anthropic and Google take `request` as their only timeout.
The rate limits hold per provider. A failed API call waits `backoff_base *
multiplier^n` plus jitter, at most `backoff_cap` seconds, a server's
`Retry-After` included. `schema_retries` covers retries driven by the model
output: invalid JSON (`validation_failure`) or a flag the model sets on a page.
`state_dir` and `openalex` are described under
[Citations and OpenAlex](#citations-and-openalex), `images` under
[Providers and Response Formats](#providers-and-response-formats).

The `defaults:` block holds run defaults keyed by option names. Unknown keys,
options that only the command line can set and invalid values are errors.
Precedence is flag (or wizard answer), then settings default, then code default.
`--dry-run --json` prints the effective spec with the source of each value, and
the store records it with every run, so a run stays reproducible after the
defaults change. Tuning values such as the citation merge
thresholds, image byte caps and text cleaning switches are code constants.

## Providers and Response Formats

Each phase's provider comes from `--provider` (or a phase flag), else a
`provider:` prefix on the model name (`anthropic:claude-sonnet-5`), else a
`vendor/model` name (OpenRouter), else the name itself (`claude` is Anthropic,
`gemini` is Google, `gpt`, `chatgpt`, `o1`, `o3`, `o4` and `text-` are OpenAI),
else OpenAI. A prefix that contradicts a provider given at the same or a higher
precedence level is an error. The key comes from the variable `api_keys:` names.
Custom endpoints are named in the settings and selected with `--endpoint NAME`,
which implies provider `custom`:

```yaml
endpoints:
  local:
    base_url: https://your-endpoint.example.com/v1/
    api_key_env: YOUR_ENDPOINT_API_KEY
    supports_vision: true    # the model accepts images (default true)
    supports_schema: false   # the API enforces JSON schemas (default false)
```

`--response-format schema` has the API enforce the JSON schema, `json` asks for
JSON in the prompt and validates it with retries, and `text` asks for plain text.
Unset, a phase uses `schema` when the model supports structured output and `json`
otherwise; a custom endpoint uses `schema` when it declares `supports_schema`. A
requested `schema` on a model without structured output falls back to `json`
with a warning. The summary always asks for the summary schema; with `text` it
keeps a plain answer that fails validation as one bullet point.

Each phase sends one system prompt to every provider
(`autoexcerpter/resources/prompts/`). The prompt is a short frame: it points to
the JSON schema it embeds, states the purpose of the output and, for the summary,
describes the input, the summary context and the language of the bullets. The
rules themselves sit in the field descriptions of the transcription and summary
schemas (`resources/schemas/`). The transcription schema asks for the verbatim
page text as Markdown (null for a page without text) and the two flags
`no_transcribable_text` and `transcription_not_possible`; unreadable words become
`[illegible]`, and the uncertain part of a partly legible word becomes `[?]`. The
summary has a prompt variant for plain-text transcriptions, and the transcription
in the `text` format uses a plain-text prompt without schema. For Anthropic
models the system prompt carries a cache breakpoint, and an item's first page
runs alone before the others start, so later pages read the prompt and schema
from the prompt cache; `cached_tokens` counts these reads.

Reasoning effort reaches OpenAI reasoning models as `reasoning.effort`. Claude
models with adaptive thinking receive an effort level, other Claude reasoning
models a thinking limit of 2,048 (`low`) to 16,384 (`xhigh`) tokens; Gemini 3
models receive a thinking level, earlier Gemini models a limit of 512 to 16,384
tokens. OpenRouter and custom endpoints receive no reasoning parameter. Verbosity
reaches the OpenAI models that support it. Temperature and top-p are left out
while reasoning is active and for models that reject them. Without
`--max-output-tokens`, OpenAI, Anthropic and Google models get their family's
output limit. The service tier is sent with OpenAI calls only; a tier given for
another provider by flag or settings default is ignored with a warning on stderr.

The family rules in `autoexcerpter/llm/capabilities.py` feed the shared resolver
in `autoexcerpter/common/capabilities.py`. Rules match the lowercased model name
in order and the first match wins, so specific names (`gpt-5.6-terra`) precede
general ones (`gpt-5`). A rule sets image input, structured output, reasoning,
verbosity, sampling parameters, context and output limits and image size caps.
Any model name is accepted. One that no rule matches gets a conservative profile
(no image input, no structured output, no reasoning parameters) and a warning.

The image options default to the preset of the transcription model's family.
OpenRouter models use the family their name points to (Claude, Gemini, else
OpenAI); custom endpoints use the custom preset.

| Family | `--dpi` | `--image-detail` | Format | JPEG quality | Grayscale | Base64 cap |
| --- | --- | --- | --- | --- | --- | --- |
| openai | native | original | jpeg | 95 | yes | 50 MB |
| anthropic | native | auto | jpeg | 95 | yes | 10 MB |
| google | 300 | high | jpeg | 95 | yes | 20 MB |
| custom | 150 | high | jpeg | 85 | yes | none |

`native` renders each PDF page at the density of its scan; pages without a
page-spanning scan render at 300 DPI. Payloads are sized to the model's documented
limits. At `original` detail, gpt-6-luna pages are limited to 3,000 px on the long
side, since larger scans led it to invent text; `images.original_max_side_px` in the
settings sets this limit for every OpenAI model. `--image-detail` is the request
detail for OpenAI models and selects the local resize profile for the others (the
media resolution for Google). An oversized PNG falls back to JPEG with a warning; a
JPEG over the cap fails the page.

## Output Contract

For an input `X.pdf` or an image folder `X/`, written beside the input or into
`--output`:

```text
X_transcription.md       (X_transcription.txt with --transcription-format txt)
X_summary.md             summary-md
X_summary.docx           summary-docx
autoexcerpter.sqlite     the store; one database per output folder, always written
```

The transcription starts with a metadata header and holds the pages in scan
order. A printed page number follows its page as `<page_number>12</page_number>`.
A page without one is preceded by `<page_break pdf="N"/>` (`image="N"` for image
folders), N being its 1-based position. A page-number tag glued to a word is a
note marker and becomes a footnote reference (`things.[^10]`).

The summary files list the document structure, one section per content page
headed by its printed page number (or `[No printed number; PDF p. N]`, with an
inferred number in brackets, `Page [6]`) and the consolidated references with
their pages. Page numbers are checked per section (front matter, body,
appendix) against runs of consecutive detected numbers
(`rendering/page_numbering.py`): a misread number inside a run is corrected, while
an excerpt that skips pages, say from 12 to 30, keeps both runs. The Markdown file
keeps LaTeX math as `$...$` and `$$...$$`; the
Word file converts it to native Word equations (`rendering/equations.py`) and
links references matched in OpenAlex.

The files are rendered from the store once an item has every unit stored, each
replaced atomically and recorded with its hash. A document renders to the same
bytes until its units change, on any machine: the transcription header and the
summaries' metadata line show the time the document last changed, in UTC, and the
Word file has fixed archive timestamps. Text files end with exactly one newline.

When an item's longest path would exceed the Windows path limit,
`output/paths.py` shortens the item name and appends an 8-character hash of the
full name, the same way for every file of the item.

## Store

`autoexcerpter.sqlite` is the single source of truth of every run in its output
folder (`common/store.py` with AutoExcerpter's tables in `output/store.py`). It
records each run with its tool version, command and effective spec; the prompt,
schema and context texts it used; one document per item with its input path and
hash and the image settings its pages were produced with; one transcription unit
and one summary unit per page with status, content, model, provider, token usage
and error; every provider attempt with the response body as received over HTTP;
the page summaries, the consolidated citations, and the rendered files with their
hashes. Each unit commits in its own short transaction as its page finishes, so
an interrupted run loses at most the pages in flight. A run leases each document
it works on, so two runs on one folder never write the same document; a lease
that is not renewed expires after 10 minutes, and a write that starts after the
expiry fails.

Outside code reads the store through five views, which stay stable across
versions while the tables behind them may change:

| View | One row per | Columns |
| --- | --- | --- |
| `page_text` | page transcription | `document`, `kind`, `page_index`, `image_name`, `order_index`, `text`, `status`, `model`, `usage`, `outcome`, `error`, `provider`, `run_id` |
| `page_summaries` | page summary | `document`, `order_index`, `page`, `image_name`, `status`, `stale`, `outcome`, `page_information`, `bullet_points`, `page_references`, `model`, `usage`, `error`, `provider`, `run_id` |
| `document_citations` | consolidated citation | `document`, `position`, `text`, `pages`, `partial`, `doi`, `url`, `metadata` |
| `document_status` | document | `document`, `kind`, `input_path`, `status`, `units_total`, `units_done`, `units_failed`, `units_pending`, `last_run_id`, `updated_at` |
| `page_attempts` | provider attempt | `document`, `kind`, `order_index`, `response_id`, `run_id`, `provider`, `model`, `attempt`, `submission_id`, `custom_id`, `raw`, `raw_kind`, `usage`, `error`, `http_status`, `created_at`, `role` |

Open the database read-only and close it after the query, so that a running item
never waits on the reader:

```python
import sqlite3
from pathlib import Path

database = Path("excerpts") / "autoexcerpter.sqlite"
connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
try:
    rows = connection.execute(
        "SELECT order_index, status, text FROM page_text WHERE document = ? "
        "ORDER BY order_index",
        ("doc",),
    ).fetchall()
finally:
    connection.close()
```

A reader that holds a transaction open delays the run's writes, which wait and
warn once a minute. A wait shorter than the 10-minute lease fails no page; after
a longer one the item stops, and its stored pages and provider responses stay
for the next run. A database with a newer schema, an
unreadable file and a database written before the store stop the run with exit
code 2 and are left as they are.

The store's batch tables come from `common/store.py`: `submissions` holds one row
per submitted batch part with its provider batch ID and status, and
`submission_requests` maps each custom ID to its unit. AutoExcerpter adds
`batch_jobs` (schema version 2), one row per document with batch work in flight,
naming the `run --batch` whose stored spec governs it and the summary context it
used. Opening a version 1 database for writing adds the table in its opening
transaction; AutoExcerpter 2.1 then refuses the database as newer (exit 2).

### Render

`autoexcerpter render --input PATH` writes the files of every stored document
below `PATH` (a folder or one `autoexcerpter.sqlite`) again, without model calls.
Without `--formats` each document gets the formats it was rendered in before;
`--formats` takes `transcription-md`, `transcription-txt`, `summary-md` and
`summary-docx`. A file that matches neither its recorded hash nor the new
rendering was changed outside the tool and is left as it is unless `--force`; so
is a file that changes while the render waits for the database.
`--dry-run` reports what would be written. Exit codes: 0 when every file holds
its rendering or was left as edited, 1 when a file could not be replaced or
another run holds a document, 2 for a usage error or a refused database; a
database written before the store names the migrate command.

### Migrate

`autoexcerpter migrate --input PATH` imports what version 1.0 left at or below
`PATH`: the folder databases, the working logs `transcription.jsonl` and
`summary.jsonl` in each `X_autoexcerpter/` folder, and the rendered files. It
snapshots each database beside itself (`autoexcerpter.pre-migrate-<time>.sqlite`)
before its first write and rebuilds a version 1.0 database into the store in one
transaction, verified by its row counts; the pages of a document's `pages_total`
that its rows lack are planned as pending. Version 1.0 wrote times without a
time zone; they are read as the local time of the machine that runs migrate and
stored in UTC. The last log record of each page becomes its unit; pages a log
lacks are planned as pending, so a partial item resumes where it stopped. Each
item is written under its lease in one transaction, which first checks that no
run changed the planned units since migrate read them. Citations are
consolidated from the stored summaries with OpenAlex metadata from the cache in
`state_dir` (`--settings FILE` names the settings file); no request is made. An
existing file that holds its rendering from the store is registered with its
hash, so a later `render` replaces it in place; a file that differs or cannot be
read is registered without one, so `render` keeps it unless `--force`. Old files
are only read, and a second run imports nothing.

Each document is then verified from the store. A unit count or failure count
that differs from the logs fails the import (exit 1), as does a re-rendered body
that differs from the existing file or a file that cannot be read. Text files are
compared without the header, the `<page_break .../>` markers and a missing final
newline; the Word summary by the text of its paragraphs, table cells and
equations without the metadata line. The report counts the documents per resume
state before and after the import. Exit code 2 means a database the store
refuses: newer, unreadable or of an unknown layout. With `--json`, stdout
carries one summary line with `databases`, `converted`, `documents`, `units`,
`citations`, `outputs`, `mismatches`, `warnings` and `snapshots`.

## Batch Processing

Without `--batch`, every page is a direct request. A folder run processes the
selected items one after another, with up to `--concurrency` pages of an item
(default 16) in flight at once. The rate limits apply per provider across the run.
A failed item is counted, and the run continues.

`run --batch` sends the same requests to the batch APIs of OpenAI, Anthropic and
Google instead, which cost about half as much and finish within a day. Both
phases need such a provider: OpenRouter and custom endpoints exit with code 2 at
planning, dry runs included, as does an unset key variable. Each item is planned
and classified as in a direct run. Its pages to transcribe are rendered and
submitted as one transcription batch, and the summaries of pages already
transcribed as one summary batch; a page that fails to render fails at once, and
the summary of a failed or blank page is its placeholder, written without a call.
The run ends once the batches are accepted (`items_submitted`). The store records
each submission before its upload and its provider batch ID after the create, so
an interrupted run never loses a batch.

`batch check` continues the work. For every database below `--input` it stores the
results of each finished batch, submits the summaries of the pages those results
transcribed and writes the files of every document left with nothing to submit or
wait for. Without `--wait` a check makes one pass, so a run that summarizes needs
two checks after its batches finish: the first collects the transcriptions and
submits the summaries, the second collects the summaries and writes the files.
With `--wait` the check repeats every `--poll-minutes` until no batch work it can
advance is pending: no batch still running, no summaries just submitted and no
document held by another run. No check starts after `--wait-hours`, but a running
one finishes. The default of 98 hours covers two rounds of the longest provider
window: OpenAI and Anthropic expire a batch after 24 hours and Google after 48. The
limit is soft: a check fails an expired batch, and the wait then ends on its own;
the limit bounds a wait whose checks cannot read the batch statuses. The
later steps follow the `run --batch` that submitted the document: its stored spec
sets the summary model, the formats and OpenAlex, and the summary context it read
is kept with the document. Prompts, schemas and machine settings come from the
installed version and the current settings file.

```bash
autoexcerpter run --input scans --all --output excerpts --batch
autoexcerpter batch check --input excerpts     # transcriptions in, summaries out
autoexcerpter batch check --input excerpts     # summaries in, files written
autoexcerpter batch check --input excerpts --wait   # or: both, as batches finish
```

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input DIR` | folder searched for batch submissions, subfolders included | the current directory | no |
| `--batch-ids IDS` | comma-separated batch IDs to check; default every batch | | no |
| `--adopt BATCH_ID` | record this provider batch for the unresolved submission it belongs to, then finalize it | | no |
| `--abandon SUBMISSION_ID` | give up an unresolved submission (its upload is deleted) or a submitted one whose results are lost: its pages return to pending and its documents wait for a run | | no |
| `--wait` | check again every --poll-minutes until no batch work is pending, or until --wait-hours have passed | | no |
| `--poll-minutes N` | minutes between two checks with --wait | 15 | no |
| `--wait-hours N` | hours after which --wait starts no further check; a running check finishes | 98 | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

A result is read as a direct answer is: a transcription is validated against its
schema and cleaned, a summary validated and given its placeholder when it fails.
A failed request, a missing result, a content-filter stop and a refusal fail the
unit with an error placeholder. Every result line is stored in `page_attempts`
with its `submission_id` and `custom_id`, and the batch's files are deleted from
the provider once the results are stored. Inside a batch there are no validation
or empty-answer retries and no first-page warm-up of the Anthropic prompt cache;
the cache breakpoint stays in the system prompt. A unit that failed in a batch is
retried by the next `run`, with or without `--batch`, as any failed page is;
AutoExcerpter has no separate repair command. A direct run on a document with no
batch in flight takes its batch work over.

A submission whose create got no confirmed answer is unresolved, and its pages
stay blocked. The next check lists the provider's batches of its time window and
adopts the one that names it; otherwise it lists the candidates, and `--adopt
BATCH_ID` or `--abandon SUBMISSION_ID` settles it. A finished batch whose results
cannot be downloaded fails the check and stays submitted, so a later check can
retry; when its results are lost for good, `--abandon` gives it up too, and a run
submits its pages again. Exit codes of `batch check`: 0
when nothing failed, 1 when a submission or a document failed or a submission is
unresolved, 2 for a usage error, a missing folder, a database the store refuses,
or an `--adopt` or `--abandon` that names nothing to act on. Its JSON line holds
`command` (`check_batches`), the submissions `finalized`, `pending`, `failed` and
`unresolved`, the summary batches `submitted`, the documents `complete`, the
`outputs` written and the `usage` of the stored results per role, plus `adopted`
or `abandoned` when the option is given.

With `--wait` a failure does not end the wait: the other batches are still
collected, and a later check may still settle an unresolved submission. When the
wait ends, its counts are read from the databases for every submission and
document a check saw; a failure stays one unless the database shows it recovered.
The check then also exits 1 when the deadline passed with work pending, or when a
document needs a run: its pages are pending with no batch in flight, and the end
line names the `autoexcerpter run ... --batch` command that submits them. The JSON
line adds `polls` (checks made), `waited_seconds`, `stopped` (`settled`,
`timeout`, `interrupted` or `error`), `needs_run` and `rerun` (those commands).
Between checks stderr shows one line, such as `Waiting for batches: 2 pending;
next check at 14:45 (check 3; 0.50 of 98 h).`

`batch cancel` cancels batches at the provider:

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--batch-ids IDS` | comma-separated batch IDs to cancel | | no |
| `--all` | cancel every unfinished batch of every provider | | no |
| `--input DIR` | cancel the open batch submissions recorded in the databases below this folder | | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

Exactly one of the first three is required. With `--input`, each cancelled
submission is marked cancelled, its pages return to pending and its documents
wait for a run; a batch that has already ended is left for `batch check`. Exit
codes: 0 when every batch was cancelled, 1 when a cancellation or a provider
listing failed, 2 for a usage error, a missing folder or a refused database. Its
JSON line holds `command` (`cancel_batches`), `cancelled` and `failed`.

The shared backends split a submission into parts by the provider's limits:
OpenAI 50,000 requests and 150 MB per input file, Anthropic 100,000 requests and
224 MB, Google 50,000 requests sent inline up to 20 MB, else as an uploaded file
up to 2 GB. Each part is one submission. Page images keep the size caps of a
direct run, and an item's page images are held in memory until its transcription
batch is built.

## Resume

Resume is the default. Each item is classified from the store alone
(`output/store.py`) before the run; a dry run shows the states.

| State | Meaning | Action |
| --- | --- | --- |
| `none` | no stored units | process from scratch |
| `partial` | pages are pending or failed | transcribe them, then summarize them |
| `transcription_only` | every page is transcribed, some summaries are missing, failed or stale | summarize those pages from the stored text |
| `complete` | every unit is stored | skip (`items_skipped`); missing files and files rendered before a later run worked the item are rendered again without model calls or OpenAlex requests (a document without stored citations gets them from the OpenAlex cache); a file that cannot be written counts the item failed |
| `legacy` | output files or version 1.0 working logs exist without stored units | skip and list; `migrate` imports them, `--force` processes the item again |
| `blocked` | another live run holds the document's lease, or an open or unresolved batch holds its units | skip and list |
| `refused` | pages remain to transcribe, but the stored pages used other image settings | count failed; name the changed keys |

A failure never counts as complete, so a rerun retries failed pages. A page
stopped by a content filter or refused by the model fails after one call
(`error_type` `content_filter` or `refusal`) and is retried the same way. When a
page's transcription changes, its summary is marked stale and made again. An
input whose content hash changed since its units were stored is processed from
scratch. When a stored document belongs to another input of the same name that
still exists, as with two `doc.pdf` files sharing one `--output` folder, the plan
stops with exit 2 before any work; give one of them another output folder.

`--force` processes every selected item from scratch; the earlier provider
responses stay in the store. `--retranscribe` keeps resume but transcribes the
stored pages again and summarizes them again. The image settings are compared key
by key with the record stored with the document. Switching the transcription
model on resume is allowed: the model and the settings derived from it (detail
resolution, image cap, the defaults of its model family) are not compared, the
run records a note, and the
transcription file names both models. Each unit names the model that produced
it.

## Citations and OpenAlex

Citations from all pages are merged by a normalized key, then near-duplicates with
the same first author and year are merged conservatively; different years or
volumes never merge. An in-text stub (author and year only) joins its full
reference when exactly one matches. Pages are consolidated per numbering system
and never mixed in one range: `pp. 5-7; PDF p. 12`, `pp. x-xii`, `Image 7`, and
`p. [6]` for an inferred number.

Each unique citation without a cached result is looked up by DOI when the text
holds one, else by title filtered to the cited year plus or minus one, else by a
free-text search. The DOIs of a document are queried 50 per request. Four
lookups run at once; requests start at least 0.1 s apart (at most ten per
second), and results apply in citation order. A candidate links only when its
title overlaps the citation and its year or an author surname confirms the match.
A lookup ends in one of four outcomes:

| Outcome | Cached in `state_dir` | Effect |
| --- | --- | --- |
| found | `openalex_cache.json` | metadata, DOI and link attached |
| not found | `openalex_misses.json` | not looked up again |
| transient failure | no | looked up again in a later run |
| budget exhausted | no | a 429 with a long `retryAfter`; no further requests in this run |

The cache is written every 25 lookups and when enrichment ends, an interruption
included; cached results are served even when the request budget is exhausted.
The `openalex:` settings hold the contact `email`, the variable with an optional
API key (`api_key_env`; the key is redacted from logs) and `max_requests`, the
citations looked up per document. `--no-openalex` turns lookups off.

## Architecture

```text
autoexcerpter/
  __main__.py   entry point: dispatch run, batch, render, migrate or ui
  spec.py       RunSpec dataclasses and the option table (OPTIONS)
  settings.py   typed settings: machine values and the defaults block
  run.py        core: discover, plan, run
  events.py     run events: item started, page done, waiting, warning, item finished
  ext.py        extension seam with pass-through defaults
  providers.py  provider names and inference from model names
  migrate.py    import of version 1.0 databases, working logs and files
  cli/          argparse from the option table, stderr reporter, --json, exit codes
  ui/           questionary wizard with the run, batch, render and migrate tasks
  batch/        run --batch items, request bodies, submit, check, finalize, cancel
  common/       vendored modules shared with the sibling tools, the store and the
                batch backends among them
  llm/          client factory, model calls with attempt capture, capability table
  imaging/      page payload sources and image presets
  output/       the store's tables, views and resume, the renderers, item paths
  pipeline/     item processing, pages, units, context
  rendering/    transcription, Markdown, DOCX, equations, citations/
  resources/    prompts and the transcription and summary schemas
```

The core imports nothing from `cli/` or `ui/`. It never prints, prompts or exits
and reports through the run events, which the CLI writes to stderr and the UI
shows as progress bars. `common/` holds tool-agnostic modules (option tables,
wizard framework, settings loader, structured calls, retry, rate limiting, HTTP
timeouts, images, the run store and its SQLite writer, attempt capture, the batch
backends of OpenAI, Anthropic and Google, the poll loop of `batch check --wait`,
agent contract) kept byte-identical
across the sibling tools: `common/MANIFEST.json` pins each file by its SHA-256, and
`tests/common/test_manifest.py` fails when a copy drifts. `ext.py` holds the
extension hooks (extra arguments, settings blocks and wizard steps, a startup
hook, batch consent, the key variables of a call, an attempt context around every
model and batch call); its defaults pass everything through.

## Development

```bash
uv sync --all-extras
uv run pytest                      # hermetic suite
uv run ruff check .
uv run ruff format --check .
uv run mypy autoexcerpter tests    # strict
markdownlint-cli2 README.md
```

The suite removes provider keys from the environment, points the home folder at
a temporary directory, blocks non-loopback connections and fails any test that
writes outside its temporary directory. Model calls go to a scripted fake chat
model behind the real client, batch calls to a scripted fake batch service behind
the real SDK clients. Golden files hold the help texts, a recorded wizard
session, the outputs per provider and response format, and DOCX, Markdown and
SQLite read-backs. After an intended change, rewrite them with
`AE_UPDATE_GOLDEN=1` (PowerShell: `$env:AE_UPDATE_GOLDEN = "1"`) and review the
diff.

## Troubleshooting

**API key not found.** A run whose model key variable is unset stops before any
work, names the variable and exits with code 2; the UI withholds Run and names
the variable on the review screen.

**Image settings changed.** A resumed item with pages left to transcribe fails
when its image settings differ from those stored with it. Rerun with the earlier
settings, or pass `--force` to start the item over.

**Pages failed with `content_filter` or `refusal`.** A provider's content filter
can stop pages dense with long verbatim passages. Rerun the item with another
model; a model switch resumes, so only the failed pages are sent again.

**"written before the store".** A run or `render` on a folder whose
`autoexcerpter.sqlite` comes from version 1.0 exits with code 2. Run
`autoexcerpter migrate --input <folder>` once; the old database stays as a
snapshot beside the new one.

**Items listed as legacy.** Output files or working logs of version 1.0 without
stored units are skipped. `migrate` imports them; `--force` processes them
again.

**Rate limit errors (429) or timeouts.** Lower `--concurrency`, adjust
`rate_limits` to your provider account, or raise `timeouts.request`.

**"No exact capability profile" warning.** The model name matches no rule, so the
conservative profile applies. A model on an OpenAI-compatible server can run as a
named endpoint whose settings declare its image and schema support.

**No sign of the example settings.** The console shows warnings and errors only,
so a run on `config/settings.example.yaml` does not mention it. Copy the example
to `config/settings.yaml`.

## Versioning

This project follows semantic versioning (`MAJOR.MINOR.PATCH`). The version in
`pyproject.toml` is the single source of truth; it is mirrored in the title
heading above and tagged in git as `vX.Y.Z`. The repository history begins with a
single baseline commit at v1.0.0. Every release is described in
[CHANGELOG.md](CHANGELOG.md).
