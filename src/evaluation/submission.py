"""Package a task's predictions into the shared task's submission format: a single
{task_1,task_2}.jsonl at the root of a zip, written to submissions/{phase}/ with a
run-scoped filename ("{run_label}_{team_name}_{training_setting}.zip") so dev-phase
and eval-phase zips -- and different variants of the same task -- never collide on
the same filename. (Before this existed, every variant's zip was named identically
inside its own outputs/ subdirectory, which caused real upload mix-ups: it's easy to
grab the wrong same-named zip from a file browser.)

Verifies the zip's contents and integrity immediately after writing it, so a bad
submission is caught here rather than discovered on CodaBench.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path


def write_submission_zip(
    pred_rows: list[dict],
    task_file_stem: str,
    team_name: str,
    training_setting: str,
    output_dir: str | Path,
    run_label: str | None = None,
) -> Path:
    """task_file_stem is "task_1" or "task_2" -- the arcname inside the zip is
    "{task_file_stem}.jsonl", matching what the shared task's submission harness
    expects. run_label (e.g. "task1_boosted") is prepended to the zip filename so
    different runs writing into the same output_dir never overwrite each other; the
    jsonl content is written directly into the zip (zipfile.writestr), never staged
    as a loose file on disk, since output_dir is now a directory SHARED across every
    variant of a task rather than a per-run directory.
    """
    if not team_name:
        raise ValueError("submission.team_name must be set in the config to create a submission zip")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arcname = f"{task_file_stem}.jsonl"
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in pred_rows) + "\n"

    zip_name = f"{run_label}_{team_name}_{training_setting}.zip" if run_label else f"{team_name}_{training_setting}.zip"
    zip_path = output_dir / zip_name
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(arcname, content)
    print(f"Submission ZIP created: {zip_path}")

    with zipfile.ZipFile(zip_path, mode="r") as zip_file:
        file_names = zip_file.namelist()
        print(f"ZIP contents: {file_names}")
        assert file_names == [arcname], f"ZIP must contain only {arcname} at its root."
        corrupted_file = zip_file.testzip()
        assert corrupted_file is None, f"Corrupted ZIP member: {corrupted_file}"

    print("Submission is ready.")
    return zip_path
