"""Batch transcription and summaries on the shared backends of three providers.

:mod:`~autoexcerpter.batch.item` submits one item's pages for ``run
--batch``, the request bodies built by :mod:`~autoexcerpter.batch.bodies`
and submitted as store submissions by :mod:`~autoexcerpter.batch.submit`.
``batch check`` (:mod:`~autoexcerpter.batch.check`) finalizes a finished
batch into the store (:mod:`~autoexcerpter.batch.finalize`), submits the
summaries of the pages it finalized (:mod:`~autoexcerpter.batch.work`) and
renders a document once nothing is left, following the run each document's
batch job names (:mod:`~autoexcerpter.batch.jobs`).
:mod:`~autoexcerpter.batch.cancel` cancels and lists batches, and the shared
:mod:`~autoexcerpter.common.batch_reconcile` settles unresolved submissions.
Every API call goes through :mod:`~autoexcerpter.batch.clients`.
"""
