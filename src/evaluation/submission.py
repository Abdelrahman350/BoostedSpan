"""Package a task's dev_in predictions into the shared task's submission format:
a single {task_1,task_2}.jsonl at the root of a zip named "{team_name}_{training_setting}.zip".

Verifies the zip's contents and integrity immediately after writing it, so a bad
submission is caught here rather than discovered on CodaBench.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from data.loading import write_jsonl


def write_submission_zip(
    pred_rows: list[dict], task_file_stem: str, team_name: str, training_setting: str, output_dir: str | Path
) -> Path:
    """task_file_stem is "task_1" or "task_2" -- the arcname inside the zip is
    "{task_file_stem}.jsonl", matching what the shared task's submission harness expects.
    """
    if not team_name:
        raise ValueError("submission.team_name must be set in the config to create a submission zip")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arcname = f"{task_file_stem}.jsonl"
    pred_file = output_dir / arcname
    write_jsonl(pred_file, pred_rows)

    zip_path = output_dir / f"{team_name}_{training_setting}.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(pred_file, arcname=arcname)
    print(f"Submission ZIP created: {zip_path}")

    with zipfile.ZipFile(zip_path, mode="r") as zip_file:
        file_names = zip_file.namelist()
        print(f"ZIP contents: {file_names}")
        assert file_names == [arcname], f"ZIP must contain only {arcname} at its root."
        corrupted_file = zip_file.testzip()
        assert corrupted_file is None, f"Corrupted ZIP member: {corrupted_file}"

    print("Submission is ready.")
    return zip_path
