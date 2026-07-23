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
