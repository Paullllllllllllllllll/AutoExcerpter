# AutoExcerpter v1.1.10

<img src="assets/logo.svg" alt="AutoExcerpter logo" width="64" align="right">

AutoExcerpter transcribes PDFs and folders of page images with vision-language
models, summarizes each page and builds a consolidated bibliography whose entries
are enriched from OpenAlex. It runs on OpenAI, Anthropic, Google and OpenRouter
models and on OpenAI-compatible endpoints that you name in the settings. The
`run` command serves scripts and agents; the `ui` command walks you through a run
in the terminal. `run --batch` submits the pages to the providers' batch APIs
instead, and `batch check` collects them. `render` writes the files again from
the store without model calls, and `drop` deletes stored documents so that the
next run processes them from scratch. Please report bugs on the
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
Install it from a clone, since the settings file is read from the repository's
`config/` folder.

```bash
git clone https://github.com/Paullllllllllllllllll/AutoExcerpter.git
cd AutoExcerpter
uv sync                      # development: uv sync --all-extras
mkdir config
cp autoexcerpter/resources/settings.example.yaml config/settings.yaml
```

On Windows, copy the file with `copy
autoexcerpter\resources\settings.example.yaml config\settings.yaml`. Without a
copy the tool runs on the packaged example ([Settings File](#settings-file)).
Then set the API key of each provider you use as an environment variable. The
settings name the variables (`api_keys:`); the defaults are `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` and `OPENROUTER_API_KEY`. An OpenAlex key
in `OPENALEX_API_KEY` is optional.

```bash
export OPENAI_API_KEY="sk-..."        # PowerShell: $env:OPENAI_API_KEY = "sk-..."
```

## Quick Start

```bash
uv run autoexcerpter run --input paper.pdf           # outputs beside the input
uv run autoexcerpter run --input scans --all --output excerpts
uv run autoexcerpter run --input scans --select 1-3  # or a name: --select braudel
uv run autoexcerpter run --input scans --files a.pdf,b.pdf --formats transcription-txt,summary-md
uv run autoexcerpter run --input paper.pdf --pages 7,2,15-17  # only these pages
uv run autoexcerpter run --input scans --all --dry-run --json
uv run autoexcerpter ui                              # guided run
uv run autoexcerpter run --input scans --all --output excerpts --batch
uv run autoexcerpter batch check --input excerpts    # collect finished batches
uv run autoexcerpter batch check --input excerpts --wait  # ... until all are in
uv run autoexcerpter render --input excerpts         # rewrite the files, no calls
uv run autoexcerpter drop --input excerpts --select braudel --dry-run  # preview a drop
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
the leading dashes, with underscores for hyphens (`--fallback-context` becomes
`fallback_context`). `autoexcerpter ui` takes only `--settings FILE`;
`batch check` and `batch cancel` are described under
[Batch Processing](#batch-processing), `render` and `drop` under
[Store](#store).

### Input

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input PATH` | PDF file, image folder, or folder of PDFs and image folders | | yes |
| `--select EXPR` | select items by number (1,3), range (1-5) or name search | | yes |
| `--files LIST` | comma-separated names of items inside the input folder | | no |
| `--all` | process every discovered item | | yes |
| `--pages RANGE` | pages or images: 5, first:5, last:5, 3-7, 3-, -7 or 1,3,5-8 | all | no |

`--input` is required. A single discovered item needs none of `--all`, `--select`
and `--files`; several items without one of them are a usage error (exit 2), and
the three exclude each other. A name search matches part of a file or folder
name, ignoring case. `--files` takes exact PDF or image folder names
(`a.pdf,scans_1`) and fails with exit 2 when one names no discovered item, as
does a `--select` that matches nothing.

`--pages` selects pages of every selected item (1-based, in source order): a bare
number `N` is the first N pages (as `first:N`), `last:N` the last N, and a list
of numbers and ranges such as `7,2,15-17`, `4-4` or the open `99-` the pages it
names. Only the selected pages are transcribed and summarized; the others stay
pending, so the document stays partial. The transcription and the summaries hold
every stored page in source order, and the summary covers only those pages. A
later run with another `--pages` adds its pages, and a run without `--pages`
works only the pages still missing and completes the document. A range that
selects no page of an item, such as `99-` on a 30-page PDF, skips the item with
a warning that names the range, without a model call; a malformed range
(`3-x`) is a usage error (exit 2) before any call. [Resume](#resume) describes
how `--force` and `--retranscribe` treat the pages outside `--pages`.

### Output

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--output DIR` | output folder | beside the input | yes |
| `--beside-input` | write outputs beside each input | | no |
| `--formats LIST` | comma-separated output formats, at most one transcription format (transcription-md when none is given); from transcription-md, transcription-txt, summary-md, summary-docx | transcription-md,summary-md,summary-docx | yes |

The transcription file is always written: a list without a transcription format
writes `transcription-md`, and two transcription formats are a usage error.
`--formats transcription-txt` writes only the plain-text transcription when
summaries are off (`--no-summarize`); with summaries on, the list needs a summary
format. `--beside-input` overrides an `output` settings default.

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
| model name; a provider: prefix (anthropic:NAME) selects the provider | `NAME` | `--model`: per phase | `--transcription-model`: gpt-6-luna | `--summary-model`: gpt-6-luna |
| named custom endpoint from the settings (implies provider custom) | `NAME` | `--endpoint`: none | `--transcription-endpoint`: none | `--summary-endpoint`: none |
| reasoning effort | `{none,minimal,low,medium,high,xhigh,max}` | `--reasoning-effort`: per phase | `--transcription-reasoning-effort`: xhigh | `--summary-reasoning-effort`: xhigh |
| output verbosity | `{low,medium,high}` | `--verbosity`: per phase | `--transcription-verbosity`: high | `--summary-verbosity`: low |
| maximum output tokens | `N` | `--max-output-tokens`: the model's output limit | `--transcription-max-output-tokens`: the model's output limit | `--summary-max-output-tokens`: the model's output limit |
| sampling temperature, 0.0 to 2.0 | `T` | `--temperature`: per phase | `--transcription-temperature`: 1.0 | `--summary-temperature`: 1.0 |
| nucleus sampling, 0.0 to 1.0; sent only when given and supported | `P` | `--top-p`: not sent | `--transcription-top-p`: not sent | `--summary-top-p`: not sent |
| OpenAI service tier; only for provider openai | `{auto,default,flex,priority}` | `--service-tier`: per phase | `--transcription-service-tier`: default | `--summary-service-tier`: default |
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
| `--force` | reprocess items whose outputs exist (resume is the default): every stored page of the item is reset, also outside --pages | | no |
| `--retranscribe` | transcribe a partial, transcription-only or refused item again instead of reusing stored transcriptions; a complete item is skipped (--force redoes it); with --pages only those of the selected pages, under the stored image settings | | no |
| `--concurrency N` | parallel requests | 16 | yes |
| `--batch`, `--no-batch` | submit the pages and summaries to the providers' batch APIs | off | yes |
| `--dry-run` | plan without API calls or writes | | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

### Exit Codes and JSON Summary

| Code | Meaning |
| --- | --- |
| 0 | every item completed or, under `--batch`, was submitted, or nothing was left to do |
| 1 | an item failed (a refused or unresolved batch submission and an unexpected error included), or another run leases an item (the final message names each blocked item with its reason and a lease with its expiry; an item waiting for a batch is named but does not change the exit code) |
| 2 | usage or configuration error (invalid flags or settings, an input without a PDF or image folder, several items without `--all`, `--select` or `--files`, a `--select` or `--files` that matches nothing, two items with one output name in one folder, a stored document in the output folder that belongs to another input of the same name, summaries on without a summary format, an unset API key variable in a real run, a database the store refuses, `--batch` on a provider without a batch API) |
| 130 | interrupted (Ctrl+C, or Ctrl+Break on Windows) |

An interrupted `run` releases its leases, so a rerun right away resumes with the
pages still to do. A run that is killed instead keeps its leases until they
expire ([Store](#store)); until then a rerun reports its items as blocked and
exits 1. A lease taken while a run plans or processes its items also counts,
including for a complete item whose files would be rendered again.

Progress, warnings and errors go to stderr. With `--json`, stdout carries exactly
one JSON line on every exit, errors and interrupts included. Each line holds
`dry_run`, which is true whenever `--dry-run` was given. A run summary adds:

| Field | Content |
| --- | --- |
| `items_total` | items processed in this run |
| `items_complete` | items that finished with every page and output; an item whose selected pages all succeeded while pages outside `--pages` remain pending or failed counts neither complete nor failed |
| `items_failed` | items that finished incomplete |
| `items_skipped` | items not processed: complete, unstored and blocked ones, and those of which `--pages` selects no page |
| `items_blocked` | the blocked items among `items_skipped`: those another run leases, which exit 1, and those whose units wait in a batch, which do not |
| `items_submitted` | items whose units wait in batches for `batch check` (`--batch`); they count neither complete nor failed |
| `outputs` | absolute paths of the files written: those of complete items rendered again, then those of the processed items, item by item, then the databases |
| `usage` | tokens per role (`transcription`, `summary`) and their sum under `total`, each with `input_tokens`, `output_tokens`, `total_tokens`, `cached_tokens` (input read from the provider's prompt cache) and `reasoning_tokens` |

A dry run that plans successfully prints the same fields, with `items_total` the
items it would process, `items_skipped` the items it would leave alone and a
zero `usage`, and adds the plan: `to_process` lists each item's `name`, resume
`state` and `completed_pages`; `skipped` names the items left alone (complete,
unstored and blocked); `unstored`, `blocked` and `refused` list those items with
their `name` and `reason`; `spec` maps every field of the effective run
specification, such as `transcription_model.model`, to its `value` and `source`
(`flag`, `settings`, `default` or `resolved`, described under
[Settings File](#settings-file)). A dry run needs no API key: each unset key
variable the run would need is a warning on stderr, and the dry run exits 0.

When the environment variable `CHRONO_HEARTBEAT_FILE` names a file, `run` and
`batch check --wait` also write their progress there as one JSON object for a job
launcher: `units_done` of `units_total` `unit_label`, `eta_seconds` and
`updated_at` (ISO 8601, UTC). A run counts the current item's pages and names the
item's position (`pages processed, item 3/12`); its ETA, from the page rate, is
given only during the last item and is null otherwise. A wait counts the
submissions finished of those it has seen (`batches`) before each sleep. The file
is replaced at most every 15 seconds, and a write that fails is skipped without
affecting the run.

## Guided UI

`autoexcerpter ui` first asks for the task: a new run, checking or cancelling
batches (`batch check`, `batch cancel`), rendering the files of stored documents
again (`render`) or dropping stored documents (`drop`). A task chosen there
opens at its first step, and Back from it returns to the task question. The
batch check task asks for the folder, the batch IDs and what to do with an
unresolved submission, and makes one pass (`--wait` is a command-line option);
the batch cancel task asks what to cancel. The render task asks for the input,
the formats, whether to replace files changed outside the tool and whether to
dry-run. The drop task asks for the store, the documents (all, by number or by
name) and whether to dry-run; a number or name choice is passed by name
(`--files`), so the drop deletes the documents its review showed, while "all"
passes `--all`. The review shows what the drop would delete and keep.

The new run asks one step at a time: input (with the PDFs, image folders and
pages found, and the item selection), output location and transcription format,
summary on or off with the summary formats, transcription model, summary model,
summary context, page images ("Recommended" or "Customize"; Anthropic and Google
models are not offered the OpenAI-only `original` detail), run options and pages
(all, or a range in the `--pages` syntax). The run options start with a
`Submission:` question, synchronous or Batch API, asked when every phase's
provider has a batch API and the extension allows batch submission; the parallel
requests are asked only for a synchronous run. Every list ends with Back; a text
or path prompt takes `<` alone as Back. A typed path may be quoted: one pair of
surrounding quotes is removed. A folder without a PDF or image folder asks for
the path again. Settings defaults are preselected, and answers act like flags
over them. The review screen marks values taken from settings defaults with
`(s)`, names on its `access` line how each phase's model calls authenticate (the
provider's API key variable), previews the plan and prints the equivalent
command:

```text
Review                                            (s) = from settings defaults
  input                 /data/docs/doc.pdf  (1 PDF, 3 pages)
  transcription model   openai / gpt-5.6-terra (s) | reasoning xhigh | tier priority (s)
  access                API key OPENAI_API_KEY
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
available. Quit exits. When items are already complete, the review offers to
reprocess them (`--force`). Run is withheld while the plan fails or, unless the
run is a dry run, a needed API key variable is unset; the review names each
missing variable and the phase that needs it. Under `--batch` the plan fails, and
Run is withheld, when a provider or the extension refuses batch submission. The
UI needs an interactive terminal on stdin and stdout; anything else exits with
code 2.

## Settings File

AutoExcerpter reads `config/settings.yaml` in the repository (gitignored). When it
is missing, the tool reads the packaged example
`autoexcerpter/resources/settings.example.yaml` and logs one informational line
that asks you to copy it; when neither exists, the code values apply.
`--settings FILE` reads another file, which must exist. Every key is optional
and keeps its code value when missing; unknown keys are an error that names the
file and the key. The values of the example file, written compactly and with
its `defaults:` block uncommented:

```yaml
api_keys: {openai: OPENAI_API_KEY, anthropic: ANTHROPIC_API_KEY, google: GOOGLE_API_KEY, openrouter: OPENROUTER_API_KEY}
endpoints: {}                       # named endpoints, see Providers
timeouts: {request: 900, connect: 10, write: 30, pool: 30}   # seconds
rate_limits: [[120, 1], [15000, 60], [15000, 3600]]   # [max_requests, window_seconds]
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
concurrency: {delay_between_tasks: 0}   # seconds between two request starts
state_dir: ''                       # '' means ~/.autoexcerpter
logs_dir: ''                        # '' writes no log file
openalex: {email: '', api_key_env: OPENALEX_API_KEY, max_requests: 300}
images: {original_max_side_px: null}   # null keeps the model's default
prompt_caching:
  enabled: true
  anthropic: {ttl: 5m}              # 5m or 1h
  openai: {prompt_cache_retention: null}   # or 24h
  openrouter: {anthropic_cache_control: true}
defaults:
  model: gpt-6-luna
  reasoning_effort: xhigh
  transcription_verbosity: high
  summary_verbosity: low
  service_tier: default
  formats: [transcription-md, summary-md, summary-docx]
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
`concurrency.delay_between_tasks` spaces the starts of an item's page requests
by that many seconds; `--concurrency` still bounds the requests in flight.
`logs_dir` names a log folder, which receives `application.log`, or a file
ending in `.log`; the log holds the informational lines as well, while the
console shows warnings and errors only. `state_dir` and `openalex` are described
under [Citations and OpenAlex](#citations-and-openalex), `images` and
`prompt_caching` under
[Providers and Response Formats](#providers-and-response-formats).

The `defaults:` block holds run defaults keyed by option names. Unknown keys,
options that only the command line can set and invalid values are errors.
Precedence is flag (or wizard answer), then settings default, then code default.
`--dry-run --json` prints the effective spec with the source of each value, and
the store records it with every run, so a run stays reproducible after the
defaults change. In the dry run's spec an unset output limit, response format
and image option shows the value the requests use, with source `resolved` (the
preset of new documents; a resumed document keeps the image settings it was
started with), and a provider other than OpenAI shows no service tier. Tuning
values such as the citation merge thresholds, image byte caps and text cleaning
switches are code constants.

## Providers and Response Formats

Each phase's provider comes from `--provider` (or a phase flag), else a
`provider:` prefix on the model name (`anthropic:claude-sonnet-5`), else the
provider rules of the shared capability table, so that a model runs on the
provider its capabilities are resolved for: Google's `models/` prefix, `gemini`
and `gemma` are Google, a `vendor/model` name is OpenRouter, `claude` is
Anthropic, `gpt`, `chatgpt`, `o1`, `o3`, `o4`, `text-` and `whisper` are OpenAI,
and bare `llama`, `mistral`, `mixtral`, `qwen` and `deepseek` names are
OpenRouter; any other name runs on OpenAI. A prefix that contradicts a provider
given at the same or a higher precedence level is an error. The key comes from
the variable `api_keys:` names. Custom endpoints are named in the settings and
selected with `--endpoint NAME`, which implies provider `custom`:

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
in the `text` format uses a plain-text prompt without schema. Prompt caching is
on by default (`prompt_caching.enabled`). For Anthropic models, and for Claude
models through OpenRouter unless `openrouter.anthropic_cache_control` is false,
the system prompt carries a cache breakpoint with `anthropic.ttl` (5 minutes or
1 hour), and an item's first page runs alone before the others start, so later
pages read the prompt and schema from the prompt cache; `cached_tokens` counts
these reads. OpenAI caches long prompts without a breakpoint; a set
`openai.prompt_cache_retention` (`24h`) also sends the retention parameter the
model documents, which for GPT-5.6 and later models is a 30-minute
`prompt_cache_options` TTL.

Reasoning effort reaches OpenAI reasoning models as `reasoning.effort`. Claude
models with adaptive thinking receive adaptive thinking and an effort level,
other Claude reasoning models a thinking limit of 1,024 (`minimal`, `low`) to
32,768 (`max`) tokens below the output limit; Gemini 3 models and Gemma 4
receive a thinking level, Gemini 2 models a limit of 10 to 50 percent of the
output limit (at most 24,576 tokens, 32,768 on Gemini 2.5 Pro), and the Gemini
image models none. A schema stays enforced on Gemini with thinking. An effort a
model does not document becomes the next level it does, or its highest, with
one warning per run: `none` and `minimal` become `low` on gpt-6-astra and
gpt-6.1-sol, `max` becomes `xhigh` on gpt-5.5, `none` becomes `minimal` on
gpt-5, the pro models take `medium` to `xhigh`, and Gemini `minimal` becomes
`low` on Gemini 3.1 Pro and 3.7/3.8 Flash. o1, o1-mini and the chat-latest
models receive no effort. OpenRouter reasoning models receive OpenRouter's
`reasoning` object: the effort, or for Claude and Gemini models with a known
output limit a token budget of a share of it. Custom endpoints receive no
reasoning parameter.
Verbosity reaches the GPT-5 and GPT-6 reasoning models only, not the pro,
codex or chat-latest models. Temperature and top-p are left out while reasoning
is active and for models that reject them. Without `--max-output-tokens`,
OpenAI, Anthropic and Google models get their family's output limit, and an
OpenRouter model whose output limit the capability facts document gets that
limit, which also caps a larger set one. The service tier is sent with OpenAI
calls only; a tier given for another provider by flag or settings default is
ignored with a warning on stderr. `flex` goes only to the models OpenAI lists for
Flex processing (GPT-6, GPT-5.6, 5.5, 5.4, 5.2, 5.1, 5 and their mini, nano and
pro variants where listed, o3, o4-mini); other models are sent no tier, with one
warning per run.

The family rules in `autoexcerpter/common/capability_table.py`, shared with
ChronoTranscriber and ChronoMiner, feed the resolver in
`autoexcerpter/common/capabilities.py`. The table is a base rule set with the
records of `autoexcerpter/common/capability_facts.py` applied: each record adds,
drops or re-matches a rule, or sets one field to the value the vendor's
documentation states (read 13.10.2026), and names its source. Rules match the lowercased
model name in order and the first match wins, so specific names
(`gpt-5.6-terra`) precede general ones (`gpt-5`); a `provider:` or Google
`models/` prefix is removed first. A rule sets image input, structured output,
reasoning and the documented effort levels, verbosity, sampling parameters,
the Flex tier, context and output limits and image size caps. Any model name is
accepted. One that no rule matches, or that the vendor does not document
(`gpt-5.3`, `glm-flash-latest`), gets a conservative profile (no image input,
no structured output, no reasoning parameters) and a warning.

The image options default to the preset of the transcription model's family.
OpenRouter models use the family their name points to (Claude, Gemini or
OpenAI); other OpenRouter models (Qwen, Llama, Mistral) and custom endpoints use
the custom preset, and an OpenRouter model that takes an image detail is sent
the resolved one.

| Family | `--dpi` | `--image-detail` | Format | JPEG quality | Grayscale | Base64 cap |
| --- | --- | --- | --- | --- | --- | --- |
| openai | native | original | jpeg | 95 | no | 50 MB |
| anthropic | native | auto | jpeg | 95 | no | 10 MB |
| google | 300 | high | jpeg | 95 | no | 20 MB |
| custom | 150 | high | jpeg | 95 | no | none |

`native` renders each PDF page at the density of its scan; pages without a
page-spanning scan render at 300 DPI. Payloads are sized to the model's
documented limits. At `original` detail, gpt-6-luna pages are limited to 3,000 px
on the long side, since larger scans led it to invent text;
`images.original_max_side_px` in the settings sets this limit for every OpenAI
model. `--image-detail` is the request detail for OpenAI models and selects the
local resize profile for the others (the media resolution for Google). An
oversized PNG falls back to JPEG with a warning; a JPEG over the cap fails the
page.

## Output Contract

For an input `X.pdf` or an image folder `X/`, written beside the input or into
`--output`:

```text
X_transcription.md       transcription-md (X_transcription.txt: transcription-txt)
X_summary.md             summary-md
X_summary.docx           summary-docx
autoexcerpter.sqlite     the store; one database per output folder, always written
```

The transcription starts with a metadata header and holds the pages in scan
order. Every page opens with `<page_break pdf="N"/>` (`image="N"` for image
folders), N being its 1-based position, so a page is the block from its marker to
the next. Its printed page number is the `<page_number>12</page_number>` tag
inside that block, left where the page prints it, in the header or the footer. A
page-number tag glued to a word is a note marker and becomes a footnote reference
(`things.[^10]`).

The summary files list the document structure, one section per content page
headed by its printed page number (or `[No printed number; PDF p. N]`, with an
inferred number in brackets, `Page [6]`) and the consolidated references with
their pages. Page numbers are checked per section (front matter, body,
appendix) against runs of consecutive detected numbers
(`rendering/page_numbering.py`): a misread number inside a run is corrected,
while an excerpt that skips pages, say from 12 to 30, keeps both runs. A short
run inside another numbering, such as section numbers between folios, counts as
misread, and a two-page spread that the neighboring numbers do not support is
read as a single page. The
Markdown file keeps LaTeX math as `$...$` and `$$...$$`; the Word file converts
it to native Word equations (`rendering/equations.py`) and links references
matched in OpenAlex.

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

Outside code reads the store through five views, its stable read contract; the
tables behind them may change:

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
warn once a minute. A wait that ends before the lease expires fails no page;
after a longer one the item stops, and its stored pages and provider responses
stay for the next run. A database with a newer schema, an unreadable file and a
database that is not an AutoExcerpter store of this version stop the run with
exit code 2 and are left as they are.

The store's batch tables come from `common/store.py`: `submissions` holds one row
per submitted batch part with its provider batch ID and status, and
`submission_requests` maps each custom ID to its unit. AutoExcerpter adds
`batch_jobs`, one row per document with batch work in flight, naming the
`run --batch` whose stored spec governs it and the summary context it used.

### Render

`autoexcerpter render [--input PATH]` writes the files of every stored document
below `PATH` (a folder or one `autoexcerpter.sqlite`; default the current
directory) again, without model calls. Without `--formats` each document gets
the formats it was rendered in before; `--formats` takes `transcription-md`,
`transcription-txt`, `summary-md` and `summary-docx`. A file that matches
neither its recorded hash nor the new rendering was changed outside the tool and
is left as it is unless `--force`; so is a file that changes while the render
waits for the database. `--dry-run` reports what would be written.
`--settings FILE` names the settings file, whose `logs_dir` receives the log.
Exit codes: 0 when every file holds its rendering or was left as edited, 1 when
a file could not be replaced or another run holds a document, 2 for a usage
error, a missing or invalid settings file or a refused database. With `--json`,
stdout carries one summary line with `dry_run`, `command` (`render`), the counts
`databases`, `documents`, `written`, `unchanged` (files that already hold their
rendering), `kept` (left as edited), `stale` (could not be replaced) and
`skipped` (documents another run holds), and the `outputs` written.

### Drop

`autoexcerpter drop --input PATH (--select EXPR | --files LIST | --all)` deletes
stored documents from the store of one output folder (`PATH` is the folder or its
`autoexcerpter.sqlite`): each document with its units, provider responses, page
summaries, citations, notes, batch job and rendered-file records, in one
transaction, so the next run processes it from scratch. `--select` numbers the
stored documents from 1 in name order or searches their names; `--files` names
documents by name or by the file name of their input, and an entry that names
two documents is refused. A rendered file is deleted while it still matches a
hash recorded for it; a file changed outside the tool, missing or recorded for a
document that stays is kept and named. A run then skips the item as unstored
until a kept file is moved away, or processes it again with `--force`, which
replaces the file. A document that another live run leases, or whose units a
submitting, submitted or unresolved batch submission holds, is refused and stays.
The drop is recorded as a run with the command `drop`. `--dry-run` reports what
would be dropped and changes nothing.

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input PATH` | output folder holding autoexcerpter.sqlite, or the database itself | | no |
| `--select EXPR` | select stored documents by number (1,3), range (1-5) or name search | | no |
| `--files LIST` | comma-separated document names or input file names | | no |
| `--all` | drop every stored document | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |
| `--dry-run` | report what would be dropped; change nothing | | no |
| `--json` | print one JSON summary line on stdout | | no |

Exit codes: 0 when every selected document was dropped (or would be, in a dry
run), 1 when a document was refused, 2 for a usage error, no store at `PATH`, a
selection that is malformed or matches nothing, or a database the store refuses.
With `--json`, stdout carries one summary line with `database`, `selected`,
`dropped` (names), `refused` (`doc_id`, `name`, `reason`), `deleted_files`,
`kept_files` (`path`, `reason`) and `warnings`.

## Batch Processing

Without `--batch`, every page is a direct request. A folder run processes the
selected items one after another, with up to `--concurrency` pages of an item
(default 16) in flight at once, their starts spaced by
`concurrency.delay_between_tasks`. The rate limits apply per provider across the
run. A failed item is counted, and the run continues.

`run --batch` sends the same requests to the batch APIs of OpenAI, Anthropic and
Google instead, which cost about half as much and finish within a day. Both
phases need such a provider: OpenRouter and custom endpoints exit with code 2 at
planning, dry runs included. An unset key variable exits with code 2 as well,
except in a dry run, which warns. Each item is planned and classified as in a
direct run. Its pages to transcribe are rendered and submitted as one
transcription batch, and the summaries of pages already transcribed as one
summary batch; a page that fails to render fails at once, and the summary of a
failed or blank page is its placeholder, written without a call. The run ends
once the batches are accepted (`items_submitted`). The store records each
submission before its upload and its provider batch ID after the create, so an
interrupted run never loses a batch.

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
window: OpenAI and Anthropic expire a batch after 24 hours and Google after 48.
The limit is soft: a check fails an expired batch, and the wait then ends on its
own; the limit bounds a wait whose checks cannot read the batch statuses. The
later steps follow the `run --batch` that submitted the document: its stored
spec sets the summary model, the formats and OpenAlex, and the summary context it
read is kept with the document. `--formats` replaces the stored formats for the files
this check writes; a list without a transcription format keeps the stored one.
Prompts, schemas and machine settings come from the installed version and the
current settings file.

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
| `--abandon SUBMISSION_ID` | give up an unresolved submission or a submitted one whose results are lost: its upload is deleted, its pages return to pending and its documents wait for a run | | no |
| `--wait` | check again every --poll-minutes until no batch work is pending, or until --wait-hours have passed | | no |
| `--poll-minutes N` | minutes between two checks with --wait | 15 | no |
| `--wait-hours N` | hours after which --wait starts no further check; a running check finishes | 98 | no |
| `--formats LIST` | comma-separated formats of the finished documents, replacing those of the submitting run (whose transcription format stays when none is given); from transcription-md, transcription-txt, summary-md, summary-docx | the formats of the submitting run | no |
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
retry; when its results are lost for good, `--abandon` gives it up too (the batch
is not cancelled at the provider), and a run submits its pages again.

Exit codes of `batch check`: 0 when nothing failed, 1 when a submission or a
document failed or a submission is unresolved, 2 for a usage error, a missing
folder, a database the store refuses, or an `--adopt` or `--abandon` that names
nothing to act on. Its JSON line holds `command` (`check_batches`), the
submissions `finalized`, `pending`, `failed` and `unresolved`, the summary
batches `submitted`, the documents `complete`, the `outputs` written and the
`usage` of the stored results per role, plus `adopted` or `abandoned` when the
option is given. A document whose selected pages are done while pages outside the
submitting run's `--pages` remain pending counts as left partial by `--pages` in
the report lines, not as `complete`.

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
| `complete` | every unit is stored | skip (`items_skipped`); missing files, files rendered before a later run worked the item and files the store holds no record of are rendered again without model calls or OpenAlex requests (a document without stored citations gets them from the OpenAlex cache); a file that cannot be written counts the item failed |
| `unstored` | output files exist without stored units, as after a `drop` that kept a file | skip and list; `--force` processes the item again |
| `blocked` | another live run holds the document's lease, or an open or unresolved batch holds its units | skip and list; a held lease makes the run exit 1 |
| `refused` | pages remain to transcribe, but the stored pages used other image settings | count failed; name the changed keys |

A failure never counts as complete, so a rerun retries failed pages. A page
stopped by a content filter or refused by the model fails after one call
(`error_type` `content_filter` or `refusal`) and is retried the same way. When a
page's transcription changes, its summary is marked stale and made again. An
input whose content hash changed since its units were stored is processed from
scratch. When a stored document belongs to another input of the same name that
still exists, as with two `doc.pdf` files sharing one `--output` folder, the plan
stops with exit 2 before any work; give one of them another output folder.

A run with `--pages` works only the selected pages and leaves the document
`partial`; a later run picks up the rest (see [Input](#input)).

`--force` processes every selected item from scratch, resetting every stored
page, also outside `--pages`; the earlier provider responses stay in the store.
`--retranscribe` keeps resume but transcribes the stored pages again and
summarizes them again. It acts on documents that are `partial`,
`transcription_only` or `refused`; a `complete` document is skipped, and
`--force` redoes it. With `--pages` it acts on the selected pages only, and the
other stored pages and their summaries stay as they are. It then also keeps the
document's stored image settings, so a document never mixes two, and refuses an
item whose stored settings it cannot keep; `--force`, or `--retranscribe`
without `--pages`, changes them. The image settings are compared key by key with
the record stored with the document. Switching the transcription model on resume
is allowed: the model and the settings derived from it (detail resolution, image
cap, the defaults of its model family) are not compared, the run records a note,
and the transcription file names both models. Each unit names the model that
produced it.

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
  __main__.py   entry point: dispatch run, batch, render, drop or ui
  spec.py       RunSpec dataclasses and the option table (OPTIONS)
  settings.py   typed settings: machine values and the defaults block
  run.py        core: discover, plan, run
  events.py     run events: item started, page done, waiting, warning, item finished
  ext.py        extension seam with pass-through defaults
  providers.py  provider names and inference from model names
  cli/          argparse from the option table, stderr reporter, --json, exit codes
  ui/           questionary wizard with the run, batch, render and drop tasks
  batch/        run --batch items, request bodies, submit, check, finalize, cancel
  common/       vendored modules shared with the sibling tools, the store and the
                batch backends among them
  llm/          client factory, model calls with attempt capture, capabilities
  imaging/      page payload sources and the resolved image settings
  output/       the store's tables, views and resume, the renderers, item paths, drop
  pipeline/     item processing, pages, units, context
  rendering/    transcription, Markdown, DOCX, equations, citations/
  resources/    prompts, the transcription and summary schemas, the settings example
```

The core imports nothing from `cli/` or `ui/`. It never prints, prompts or exits
and reports through the run events, which the CLI writes to stderr and the UI
shows as progress bars.

`common/` holds the tool-agnostic modules that ChronoTranscriber and ChronoMiner
vendor too: option tables and the shared option rows, the wizard framework, the
settings loader and shared settings blocks, structured calls, retry, rate
limiting, HTTP timeouts, page images and image presets, the model capability
table, the provider request builders, the run store and its SQLite writer,
attempt capture, the batch backends of OpenAI, Anthropic and Google, the poll
loop of `batch check --wait` and the agent contract. The copies are
byte-identical across the tools: `common/MANIFEST.json` pins each file by its
SHA-256, and `tests/common/test_manifest.py` fails when a copy drifts. Each tool
passes its own values to the shared presets and request options:
AutoExcerpter's image presets are `IMAGE_PRESETS` in `spec.py`. The option rows
every tool declares (model, images, output location, run control, batch check
and cancel, render) come from `common/standard_options.py` with AutoExcerpter's
defaults and help texts, and `settings.py` reads the shared settings blocks
through `common/settings_schema.py`; `retry` and `openalex` are its own blocks.

`ext.py` holds the extension hooks: extra arguments and the argv words the
wizard appends to the commands that take them, settings blocks, wizard steps, a
startup hook, batch consent, the key variables of a call and how it
authenticates, client options, and an attempt context around every model and
batch call. Its defaults pass everything through.

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
work, names the variable and exits with code 2; a dry run only warns. The UI
withholds Run and names the variable on the review screen.

**Image settings changed.** A resumed item with pages left to transcribe fails
when its image settings differ from those stored with it. Rerun with the earlier
values, or pass `--force` to start the item over.

**Pages failed with `content_filter` or `refusal`.** A provider's content filter
can stop pages dense with long verbatim passages. Rerun the item with another
model; a model switch resumes, so only the failed pages are sent again.

**"is not an AutoExcerpter store".** A run or `render` on a folder whose
`autoexcerpter.sqlite` lacks the store schema of this version exits with code 2
and leaves the file as it is. Move the file away, or write into another output
folder.

**Items listed as unstored.** Output files without stored units, as a `drop`
leaves a file changed outside the tool, are skipped. Move the files away, or
pass `--force` to process the items again.

**Rate limit errors (429) or timeouts.** Lower `--concurrency`, adjust
`rate_limits` to your provider account, or raise `timeouts.request`.

**"No exact capability profile" warning.** The model name matches no rule, so the
conservative profile applies. A model on an OpenAI-compatible server can run as a
named endpoint whose settings declare its image and schema support.

**No sign of the example settings.** The console shows warnings and errors only,
so a run on the packaged example does not mention it. Copy
`autoexcerpter/resources/settings.example.yaml` to `config/settings.yaml`.

## Versioning

Each release is described in [CHANGELOG.md](CHANGELOG.md).
