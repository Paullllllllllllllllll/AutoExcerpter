# AutoExcerpter batch mode

A batch job submits with `run ... --batch`; a model of a provider without a batch
API (OpenRouter, a custom endpoint) in either phase exits 2. Size a batch job from
a sample before submitting it.

The first `batch check` finalizes the transcription batch and submits the summary
batch; the second writes the files. `--wait` repeats the check until nothing is
pending. Its `--input` is the output folder that the submitting run's `spec` names
(beside the input, the folder holding it). The job is two steps, `run --input <in>
--all --batch`, then `batch check --input <out> --wait`. The check follows the stored spec of the
submitting run, so it takes no model flags.

- A document submitted with `--pages` is reported as left partial, not complete;
  that is the expected outcome.
- `batch check --wait` exits 1 when its deadline passes with work pending
  (`stopped`: `timeout`) or a document needs a run (`needs_run`). On a timeout,
  check again later; for the documents counted in `needs_run`, run the commands in
  `rerun`. Failed batch pages are retried by rerunning the same `run` command,
  with or without `--batch`.
- An unresolved submission blocks its pages, and the next check lists candidate
  batches: settle it with `--adopt BATCH_ID` or `--abandon SUBMISSION_ID`. A
  finished batch whose results cannot be downloaded fails every check; abandon it
  only with the user's consent, since its tokens are spent.
