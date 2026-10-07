"""The run subcommand: options to job, parsing, exit codes and stderr reports."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import autoexcerpter.pipeline.managers as managers_module
import autoexcerpter.run as core
from autoexcerpter import __main__ as entry
from autoexcerpter.cli import command
from autoexcerpter.cli.reporter import PROGRESS_STEPS, StderrReporter
from autoexcerpter.common.usage import Usage
from autoexcerpter.events import ItemFinished, ItemStarted, PageDone, Waiting
from autoexcerpter.pipeline.job import job_from_spec
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.settings import Endpoint, OpenAlexSettings
from tests.conftest import make_settings, make_spec

MakePdf = Callable[..., Path]


def _json_lines(stdout: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]


def _run(*argv: str) -> int:
    return entry.main(["run", *argv])


class TestJobFromSpec:
    def test_unprefixed_options_set_both_phases(self) -> None:
        spec = make_spec(
            model="gpt-5", reasoning_effort="low", verbosity="high", temperature=0.5
        )
        job = job_from_spec(spec, make_settings())

        for phase in (job.transcription, job.summary):
            assert phase.name == "gpt-5"
            assert phase.options["reasoning"] == {"effort": "low"}
            assert phase.options["text"] == {"verbosity": "high"}
            assert phase.options["temperature"] == 0.5

    def test_prefixed_options_override_one_phase(self) -> None:
        spec = make_spec(
            reasoning_effort="low",
            transcription_reasoning_effort="none",
            summary_reasoning_effort="xhigh",
            max_output_tokens=8000,
            transcription_max_output_tokens=12000,
        )
        job = job_from_spec(spec, make_settings())

        assert job.transcription.options["reasoning"] == {"effort": "none"}
        assert job.summary.options["reasoning"] == {"effort": "xhigh"}
        assert job.transcription.options["max_output_tokens"] == 12000
        assert job.summary.options["max_output_tokens"] == 8000

    def test_temperature_zero_does_not_fall_through(self) -> None:
        spec = make_spec(temperature=1.0, transcription_temperature=0.0)
        job = job_from_spec(spec, make_settings())

        assert job.transcription.options["temperature"] == 0.0
        assert job.summary.options["temperature"] == 1.0

    def test_top_p_is_sent_only_when_given(self) -> None:
        assert (
            "top_p" not in job_from_spec(make_spec(), make_settings()).summary.options
        )
        job = job_from_spec(make_spec(summary_top_p=0.9), make_settings())
        assert job.summary.options["top_p"] == 0.9
        assert "top_p" not in job.transcription.options

    def test_service_tier_per_phase(self) -> None:
        spec = make_spec(service_tier="priority", summary_service_tier="flex")
        job = job_from_spec(spec, make_settings())

        assert job.transcription.service_tier == "priority"
        assert job.summary.service_tier == "flex"

    def test_openai_transcription_sends_the_image_detail(self) -> None:
        job = job_from_spec(make_spec(), make_settings())
        assert job.transcription.options["image_size"] == "original"
        assert "image_size" not in job.summary.options

        detailed = job_from_spec(make_spec(image_detail="high"), make_settings())
        assert detailed.transcription.options["image_size"] == "high"

        other = job_from_spec(make_spec(provider="anthropic"), make_settings())
        assert "image_size" not in other.transcription.options

    def test_run_switches(self) -> None:
        spec = make_spec(
            summarize=False,
            formats="summary-md",
            keep_working_files=True,
            openalex=False,
            concurrency=3,
        )
        job = job_from_spec(spec, make_settings())

        assert job.summarize is False
        assert job.output_markdown is True
        assert job.output_docx is False
        assert job.cleanup is False
        assert job.openalex_enabled is False
        assert job.openalex is None
        assert job.concurrency == 3
        assert job.timeout == 900

    def test_key_variables_come_from_the_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALT_OPENALEX_KEY", "test-openalex-key")
        settings = make_settings(
            api_keys={"openai": "ALT_OPENAI_KEY", "anthropic": "ANTHROPIC_API_KEY"},
            openalex=OpenAlexSettings(
                email="me@example.org", api_key_env="ALT_OPENALEX_KEY"
            ),
        )
        job = job_from_spec(make_spec(summary_provider="anthropic"), settings)

        assert job.transcription.api_key_env == "ALT_OPENAI_KEY"
        assert job.summary.api_key_env == "ANTHROPIC_API_KEY"
        assert job.openalex is not None
        assert job.openalex.email == "me@example.org"
        assert job.openalex.api_key == "test-openalex-key"

    @pytest.mark.parametrize(
        ("fmt", "expected"),
        [("schema", "schema"), ("json", "json"), ("text", "text"), (None, "schema")],
    )
    def test_named_endpoint_maps_the_response_format(
        self, fmt: str | None, expected: str
    ) -> None:
        endpoint = Endpoint(
            "local",
            "https://local.invalid/v1/",
            "LOCAL_KEY",
            supports_vision=False,
            supports_schema=True,
        )
        settings = make_settings(endpoints={"local": endpoint})
        flags: dict[str, Any] = {"endpoint": "local", "model": "org/model"}
        if fmt is not None:
            flags["response_format"] = fmt
        job = job_from_spec(make_spec(**flags), settings)

        phase = job.transcription
        assert phase.provider == "custom"
        assert phase.endpoint == "local"
        assert phase.base_url == "https://local.invalid/v1/"
        assert phase.api_key_env == "LOCAL_KEY"
        assert phase.supports_vision is False
        assert phase.response_format == expected

    def test_endpoint_without_schema_support_defaults_to_json(self) -> None:
        endpoint = Endpoint("local", "https://local.invalid/v1/", "LOCAL_KEY")
        settings = make_settings(endpoints={"local": endpoint})
        job = job_from_spec(make_spec(endpoint="local", model="m"), settings)

        assert job.summary.response_format == "json"

    def test_registry_models_keep_an_unset_format_for_the_capabilities(self) -> None:
        job = job_from_spec(make_spec(), make_settings())

        assert job.transcription.response_format is None
        assert job.transcription.endpoint is None
        assert job.transcription.supports_vision is None


class TestResponseFormats:
    @pytest.mark.parametrize("fmt", ["json", "text"])
    def test_every_format_is_accepted_on_a_registry_provider(
        self, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str], fmt: str
    ) -> None:
        pdf = make_pdf("doc.pdf")
        code = _run(
            "--input", str(pdf), "--response-format", fmt, "--dry-run", "--json"
        )

        assert code == 0
        assert _json_lines(capsys.readouterr().out)[-1]["to_process"]


class TestParser:
    def test_all_with_select_is_a_usage_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run("--all", "--select", "1-3") == 2
        err = capsys.readouterr().err
        assert "usage: autoexcerpter run" in err
        assert "not allowed with argument" in err

    def test_invalid_service_tier_is_a_usage_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run("--service-tier", "bogus") == 2
        assert "'bogus' is not one of" in capsys.readouterr().err

    def test_unknown_command_is_a_usage_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert entry.main(["transcribe"]) == 2
        captured = capsys.readouterr()
        assert "invalid choice" in captured.err
        assert captured.out == ""


def _transcribe(payload: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "image": payload.image_name,
        "sequence_number": payload.number,
        "transcription": f"Text of {payload.image_name}.",
        "processing_time": 0.01,
        "provider": "openai",
    }
    if "fail" in str(payload.source_file):
        result["error"] = "refused"
    return result


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch, mock_image_processing: Any) -> None:
    """Replace the transcription manager with an offline stand-in."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    manager = MagicMock()
    manager.transcribe_payload = AsyncMock(side_effect=_transcribe)
    monkeypatch.setattr(
        managers_module, "TranscriptionManager", MagicMock(return_value=manager)
    )


def _complete(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    pdf = make_pdf("doc.pdf")
    return ["--input", str(pdf), "--output", str(tmp_path / "out"), "--no-summarize"]


def _item_failed(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    pdf = make_pdf("fail.pdf")
    return ["--input", str(pdf), "--output", str(tmp_path / "out"), "--no-summarize"]


def _dry_run(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    return ["--input", str(make_pdf("doc.pdf")), "--dry-run"]


def _no_items(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    return ["--input", str(tmp_path / "missing")]


def _select_nothing(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    (tmp_path / "in").mkdir()
    make_pdf("in/alpha.pdf")
    return ["--input", str(tmp_path / "in"), "--select", "gamma"]


def _several_items(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    (tmp_path / "in").mkdir()
    make_pdf("in/alpha.pdf")
    make_pdf("in/beta.pdf")
    return ["--input", str(tmp_path / "in")]


def _duplicate(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    for sub in ("a", "b"):
        (tmp_path / "in" / sub).mkdir(parents=True)
        make_pdf(f"in/{sub}/book.pdf")
    return ["--input", str(tmp_path / "in"), "--output", str(tmp_path), "--all"]


def _missing_settings(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    return ["--input", str(tmp_path), "--settings", str(tmp_path / "nope.yaml")]


def _no_input(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    return ["--no-summarize"]


def _unknown_endpoint(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    return ["--input", str(tmp_path), "--endpoint", "missing"]


def _parse_error(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    return ["--input", str(tmp_path), "--concurrency", "zero"]


@pytest.mark.usefixtures("offline")
@pytest.mark.parametrize(
    ("setup", "code", "total"),
    [
        pytest.param(_complete, 0, 1, id="complete"),
        pytest.param(_item_failed, 1, 1, id="item_failed"),
        pytest.param(_dry_run, 0, None, id="dry_run"),
        pytest.param(_no_items, 1, 0, id="no_items"),
        pytest.param(_select_nothing, 1, 0, id="select_nothing"),
        pytest.param(_several_items, 2, 0, id="several_items"),
        pytest.param(_duplicate, 2, 2, id="duplicate_output"),
        pytest.param(_missing_settings, 2, 0, id="missing_settings"),
        pytest.param(_no_input, 2, 0, id="no_input"),
        pytest.param(_unknown_endpoint, 2, 0, id="unknown_endpoint"),
        pytest.param(_parse_error, 2, 0, id="parse_error"),
    ],
)
def test_every_exit_writes_one_json_line(
    tmp_path: Path,
    make_pdf: MakePdf,
    capsys: pytest.CaptureFixture[str],
    setup: Callable[[Path, MakePdf], list[str]],
    code: int,
    total: int | None,
) -> None:
    argv = setup(tmp_path, make_pdf)

    assert _run(*argv, "--json") == code

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1, captured.out
    payload = json.loads(lines[0])
    assert payload["dry_run"] is ("--dry-run" in argv)
    if total is not None:
        assert payload["items_total"] == total
    if code:
        assert captured.err.strip()


@pytest.mark.parametrize("beside", [False, True], ids=["output_dir", "beside_input"])
def test_a_missing_key_exits_2_before_any_work(
    tmp_path: Path,
    make_pdf: MakePdf,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    beside: bool,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "in").mkdir()
    pdf = make_pdf("in/doc.pdf")
    out = tmp_path / "out"
    output = [] if beside else ["--output", str(out)]

    code = _run("--input", str(pdf), *output, "--json")

    captured = capsys.readouterr()
    assert code == 2
    assert "OPENAI_API_KEY" in captured.err
    (payload,) = _json_lines(captured.out)
    assert payload["items_total"] == 1
    assert payload["items_complete"] == payload["items_failed"] == 0
    assert not out.exists()
    assert sorted(path.name for path in pdf.parent.iterdir()) == ["doc.pdf"]


@pytest.mark.parametrize(
    ("failure", "code"), [(KeyboardInterrupt(), 130), (RuntimeError("boom"), 1)]
)
def test_interrupt_and_crash_write_one_json_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
    code: int,
) -> None:
    def _raise(_path: Path) -> Any:
        raise failure

    monkeypatch.setattr(core, "scan_input_path", _raise)

    assert _run("--input", str(tmp_path), "--json") == code
    (payload,) = _json_lines(capsys.readouterr().out)
    zero = dict.fromkeys(
        (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
            "reasoning_tokens",
        ),
        0,
    )
    assert payload == {
        "dry_run": False,
        "items_total": 0,
        "items_complete": 0,
        "items_failed": 0,
        "items_skipped": 0,
        "outputs": [],
        "usage": {"total": zero},
    }


def test_interrupt_during_the_run_exits_130(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def interrupted(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(core, "run", interrupted)

    assert _run("--input", str(make_pdf("doc.pdf")), "--json") == 130
    captured = capsys.readouterr()
    assert len(_json_lines(captured.out)) == 1
    assert "[ERROR] Interrupted" in captured.err


def test_without_json_stdout_stays_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run("--input", str(tmp_path / "missing")) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[ERROR] No items found to process" in captured.err


def test_ignored_select_parts_are_warned_on_stderr(
    tmp_path: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "in").mkdir()
    make_pdf("in/alpha.pdf")

    assert _run("--input", str(tmp_path / "in"), "--select", "1,9", "--dry-run") == 0
    assert "[WARNING] Ignored out-of-range/invalid --select part(s): 9" in (
        capsys.readouterr().err
    )


class TestDryRun:
    def test_dry_run_creates_no_output_tree(
        self, tmp_path: Path, make_pdf: MakePdf
    ) -> None:
        pdf = make_pdf("Doc.pdf")
        out_dir = tmp_path / "outtree" / "sub"

        assert _run("--input", str(pdf), "--output", str(out_dir), "--dry-run") == 0
        assert not out_dir.exists()

    def test_plan_lists_the_items(
        self, tmp_path: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "in").mkdir()
        make_pdf("in/Alpha.pdf")
        out = tmp_path / "out"
        out.mkdir()
        make_pdf("in/Beta.pdf")
        ItemPaths.for_item("Beta", out).transcription.write_text("x", encoding="utf-8")

        code = _run(
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(out),
            "--no-summarize",
            "--json",
            "--dry-run",
            "--all",
        )

        assert code == 0
        captured = capsys.readouterr()
        (payload,) = _json_lines(captured.out)
        spec = payload.pop("spec")
        assert payload == {
            "dry_run": True,
            "to_process": [{"name": "Alpha", "state": "none", "completed_pages": 0}],
            "skipped": ["Beta"],
        }
        assert spec["summary.enabled"] == {"value": False, "source": "flag"}
        assert spec["output.directory"] == {"value": str(out), "source": "flag"}
        assert spec["run.concurrency"]["source"] == "default"
        assert "DRY-RUN: Alpha -> none (0 completed page(s))" in captured.err
        assert "DRY-RUN: Beta -> already complete (skip)" in captured.err


def _report(**kwargs: Any) -> dict[str, Any]:
    base = {
        "pages_total": 2,
        "pages_attempted": 2,
        "pages_ok": 2,
        "pages_failed": 0,
        "pages_deferred": 0,
        "summary_failures": 0,
        "elapsed_s": 1.0,
        "avg_api_s": 0.5,
        "outputs": [],
    }
    base.update(kwargs)
    return base


def _result(*items: tuple[str, bool, dict[str, Any] | None], outputs: int = 0) -> Any:
    entries = tuple(
        core.ItemResult(
            name,
            success,
            ItemPaths.for_item(name, Path("out")),
            tuple(f"/out/{name}{i}.txt" for i in range(outputs)),
            report,
        )
        for name, success, report in items
    )
    return core.RunResult(entries, skipped=("Done",), seconds=12.5)


class TestStderrReporter:
    def test_page_progress_is_reported_in_steps(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reporter = StderrReporter()
        total = PROGRESS_STEPS * 2
        for done in range(1, total + 1):
            reporter.page_done(PageDone("doc", done - 1, "ok", done, total, 3))

        lines = capsys.readouterr().err.splitlines()
        assert len(lines) == PROGRESS_STEPS
        assert (
            lines[-1] == f"  doc: {total}/{total} page(s) done (3 from an earlier run)"
        )
        assert lines[0] == f"  doc: 2/{total} page(s) done (3 from an earlier run)"

    def test_failed_and_deferred_pages_are_always_reported(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reporter = StderrReporter()
        reporter.page_done(PageDone("doc", 4, "failed", 1, 100))
        reporter.page_done(PageDone("doc", 5, "deferred", 2, 100))

        assert capsys.readouterr().err.splitlines() == [
            "  doc: page 5 failed (1/100)",
            "  doc: page 6 deferred (2/100)",
        ]

    def test_item_lines(self, capsys: pytest.CaptureFixture[str]) -> None:
        reporter = StderrReporter()
        reporter.item_started(ItemStarted("doc", "pdf", 1, 2))
        reporter.item_finished(
            ItemFinished("doc", 1, 2, True, (), _report(outputs=["/out/doc.txt"]))
        )
        reporter.item_finished(ItemFinished("gone", 2, 2, False))

        assert capsys.readouterr().err.splitlines() == [
            "[1/2] doc (PDF)",
            "  doc: 2/2 page(s) ok, 0 failed, 0 deferred; 1.0s; avg API 0.5s",
            "    outputs: /out/doc.txt",
        ]

    def test_warning_and_waiting(self, capsys: pytest.CaptureFixture[str]) -> None:
        reporter = StderrReporter()
        reporter.warning("model changed")
        reporter.waiting(Waiting("rate limit", 12.0))

        assert capsys.readouterr().err.splitlines() == [
            "[WARNING] model changed",
            "  waiting 12s: rate limit",
        ]

    def test_completion_overview_caps_outputs(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        StderrReporter().completion(
            _result(
                ("A", True, _report(pages_ok=2, pages_failed=1)),
                ("B", False, _report(pages_ok=1, pages_deferred=3, summary_failures=2)),
                ("C", False, None),
                outputs=9,
            )
        )

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Run complete:" in captured.err
        assert "  items: 1 processed, 2 failed, 1 skipped (of 3)" in captured.err
        assert (
            "  pages: 3 ok, 1 failed, 3 deferred, 2 summary failure(s)" in captured.err
        )
        assert "  run time: 12.5s" in captured.err
        assert "  outputs (27):" in captured.err
        assert "    ... and 7 more" in captured.err

    def test_incomplete_items_are_named_with_page_counts(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        StderrReporter().incomplete(
            _result(
                ("Book_A", False, _report(pages_failed=2, pages_deferred=1)),
                ("Book_B", False, None),
                ("Book_C", True, _report()),
            )
        )

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "[WARN] 2 item(s) finished INCOMPLETE" in captured.err
        assert "Book_A (2 failed, 1 deferred page(s))" in captured.err
        assert "Book_B (one or more pages failed or were deferred)" in captured.err
        assert "Book_C" not in captured.err
        assert "Re-running resumes" in captured.err
        assert "placeholders" in captured.err

    def test_no_incomplete_warning_when_all_complete(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        StderrReporter().incomplete(_result(("A", True, _report())))
        assert capsys.readouterr().err == ""


def test_plan_fields_match_the_dry_run_contract(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    run_plan = core.plan(make_spec(make_pdf("doc.pdf"), dry_run=True), make_settings())
    fields = command.plan_fields(run_plan)
    assert fields.pop("spec") == run_plan.effective_spec()
    assert fields == {
        "to_process": [{"name": "doc", "state": "none", "completed_pages": 0}],
        "skipped": [],
    }


def test_run_summary_reports_tokens_per_role_and_in_total() -> None:
    usage = {
        "transcription": Usage(input_tokens=10, output_tokens=2, total_tokens=12),
        "summary": Usage(input_tokens=5, output_tokens=1, total_tokens=6),
    }

    fields = command.summary_fields(total=1, complete=1, usage=usage)

    assert fields["usage"]["transcription"]["total_tokens"] == 12
    assert fields["usage"]["summary"]["input_tokens"] == 5
    assert fields["usage"]["total"] == Usage(15, 3, 18).as_dict()
