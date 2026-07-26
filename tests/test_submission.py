import json
import zipfile

import pytest

from evaluation.submission import write_submission_zip


def test_write_submission_zip_creates_expected_file(tmp_path):
    pred_rows = [
        {"paragraph_id": "p1", "labels": ["AS", "OT"]},
        {"paragraph_id": "p2", "labels": []},
    ]
    zip_path = write_submission_zip(pred_rows, "task_1", "MyTeam", "closed", tmp_path)

    assert zip_path == tmp_path / "MyTeam_closed.zip"
    assert zip_path.exists()


def test_write_submission_zip_contains_only_expected_jsonl(tmp_path):
    pred_rows = [{"paragraph_id": "p1", "labels": []}]
    zip_path = write_submission_zip(pred_rows, "task_2", "MyTeam", "closed", tmp_path)

    with zipfile.ZipFile(zip_path, mode="r") as zip_file:
        assert zip_file.namelist() == ["task_2.jsonl"]
        assert zip_file.testzip() is None
        content = zip_file.read("task_2.jsonl").decode("utf-8")
        rows = [json.loads(line) for line in content.strip().splitlines()]
        assert rows == pred_rows


def test_write_submission_zip_requires_team_name(tmp_path):
    with pytest.raises(ValueError, match="team_name"):
        write_submission_zip([{"paragraph_id": "p1", "labels": []}], "task_1", "", "closed", tmp_path)


def test_write_submission_zip_naming_uses_team_and_setting(tmp_path):
    pred_rows = [{"paragraph_id": "p1", "labels": ["ST"]}]
    zip_path = write_submission_zip(pred_rows, "task_1", "NU_Analytics", "both", tmp_path)
    assert zip_path.name == "NU_Analytics_both.zip"


def test_write_submission_zip_run_label_prevents_filename_collisions(tmp_path):
    # Two different variants writing into the SAME shared output_dir (submissions/dev/)
    # must not overwrite each other -- run_label is what keeps their filenames apart.
    pred_rows = [{"paragraph_id": "p1", "labels": ["AS"]}]
    zip_a = write_submission_zip(pred_rows, "task_2", "MyTeam", "both", tmp_path, run_label="task2_boosted_crf")
    zip_b = write_submission_zip(pred_rows, "task_2", "MyTeam", "both", tmp_path, run_label="task2_enhanced_track_a")

    assert zip_a != zip_b
    assert zip_a.name == "task2_boosted_crf_MyTeam_both.zip"
    assert zip_b.name == "task2_enhanced_track_a_MyTeam_both.zip"
    assert zip_a.exists() and zip_b.exists()


def test_write_submission_zip_no_loose_jsonl_left_on_disk(tmp_path):
    # write_submission_zip writes the jsonl directly into the zip (zipfile.writestr)
    # rather than staging a loose file in output_dir -- output_dir is now a directory
    # SHARED across every variant of a task, so a loose "task_1.jsonl" would collide.
    write_submission_zip([{"paragraph_id": "p1", "labels": []}], "task_1", "MyTeam", "both", tmp_path, run_label="task1_boosted")
    loose_files = [p.name for p in tmp_path.iterdir() if p.suffix == ".jsonl"]
    assert loose_files == []
