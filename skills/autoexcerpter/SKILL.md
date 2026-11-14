---
name: autoexcerpter
description: >-
  AutoExcerpter: use when the user wants a paper, book or scan transcribed and
  summarized page by page (an excerpt), resumes, batch-checks or inspects an
  AutoExcerpter run, or reads its outputs (_transcription.md, _summary.md,
  autoexcerpter.sqlite).
---

# AutoExcerpter

The tool runs from a clone of the repository set up as in the README, Installation;
the context files it names live in that clone. Pass absolute paths, since `uv
--directory` moves the working directory into the clone:

```bash
uv --directory <clone> run autoexcerpter <run | batch check | batch cancel | render | drop> ...
```

`<subcommand> --help` is authoritative for flags, defaults and exit codes;
`README.md` for behavior, the `--json` line and the store.

## Running a job

1. **Check access.** Done when the key variables of the transcription and summary
   models' providers are set (README, Providers and Response Formats) and you know
   whether the job runs directly or as a batch.
2. **Dry run.** Add `--dry-run --json` to the intended command. Done when it exits
   0, `to_process` lists exactly the intended items with the expected resume
   states, and every value you set shows `"source": "flag"` in `spec`. An unset API
   key is only a stderr warning there. A folder input gives its own PDFs and the
   image folders one level down; a deeper tree needs `--recursive`, and the dry run
   warns how many items it left in subfolders. `--select N` counts in the natural
   order of the paths below the input, and `--files` takes those paths.
3. **Sample** anything beyond one short item: run one item, or the first pages of
   one with `--pages`, read `usage.total.total_tokens` from its `--json` line and
   project the job; the full run resumes the sample. Done when the user has seen
   the projection for a job beyond a few items.
4. **Run.** A job that finishes within the turn runs directly with `--json`; a
   longer one runs as a background process whose `--json` line you read when it
   ends. Before any batch work, read [batch.md](batch.md).
5. **Verify.** Done when the exit code is 0, `items_failed` and `items_blocked` are
   0 and every path in `outputs` exists. Items under `items_submitted`, and items
   blocked by a batch, exit 0 too: the job is done only once `batch check` has
   collected them. Exit 1 means an item failed or another run still leases one
   (a killed run holds its lease for 10 minutes): rerun the same command, which
   resumes, then work through Resume and failures. Exit 3 means the provider
   stopped serving the run: nothing failed, the unsent pages stay pending, and the
   same command resumes them once the provider serves again.

## Conventions

Unless the user says otherwise:

- Write the files beside each input: omit `--output`, or pass `--beside-input` when
  the settings set an output default. The dry run's `spec` shows the effective
  output directory (null means beside the input).
- Omit model, image and context flags, so the settings file decides them. A task
  that edits the settings file restores it afterwards.
- Keep summaries and OpenAlex enrichment on.
- A folder of many short items takes `--open-items N` (8, for example): the run
  then keeps up to N items open under the one `--concurrency` limit, where
  otherwise a two-page item uses two request slots and leaves the rest idle. It
  has no effect with `--batch` or on a single item.

`--pages N` selects the first N pages; a single page is `N-N`. A later range adds
pages, and a run without `--pages` completes the document. Pass a context file as
an absolute path: a `--context` value that names no existing file is sent as topic
text. Items without a sidecar get the settings' `fallback_context`, if one is set.

## Resume and failures

A rerun resumes by default and redoes only what is missing, failed or stale; a
`complete` document only gets missing files rendered again.

- A document stored under other image settings is `refused` on resume (counted
  failed, the changed keys named): pass the stored values, or `--force`.
- `--retranscribe` skips a complete document; `--force` redoes it.
- `drop` deletes stored documents and their files so the next run starts them
  fresh; `render` writes the files again from the store without model calls.
- An item `unstored` has output files but no stored units, as after a `drop`
  that kept an edited file: runs skip it; move the file away, or `--force`.
- Pages failed with `error_type` `content_filter` or `refusal`: rerun the item with
  a transcription model of another provider (`--transcription-model`). The model
  switch resumes, so only the failed pages are sent.

## Store and outputs

`autoexcerpter.sqlite` in each output folder is the single source of truth;
`X_transcription.md` and `X_summary.md` are renderings of it. Remove a document
with `drop`, keeping the database. Read the store read-only through its views
(README, Store) and close the connection after each query: an open transaction
makes the run's writes wait, and a wait past the lease stops the item.

When OpenAlex answers that the request budget is exhausted, the run's remaining
citations stay unenriched, and a rerun skips the complete document: report those
citations (README, Citations and OpenAlex).
