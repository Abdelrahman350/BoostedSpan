from data.loading import build_shared_split, strat_key


def _make_task1_rows(n: int) -> list[dict]:
    rows = []
    label_cycles = [["AS"], ["AS", "ST"], ["OT"]]
    domains = ["editorial", "debate"]
    for i in range(n):
        rows.append(
            {
                "paragraph_id": f"p{i}",
                "text": f"paragraph number {i}",
                "labels": label_cycles[i % len(label_cycles)],
                "type": domains[i % len(domains)],
            }
        )
    return rows


def _make_task2_rows(task1_rows: list[dict]) -> list[dict]:
    return [
        {
            "paragraph_id": r["paragraph_id"],
            "text": r["text"],
            "type": r["type"],
            "labels": [{"label": l, "start_offset": 0, "end_offset": 5} for l in r["labels"]],
        }
        for r in task1_rows
    ]


def test_split_reproducibility():
    task1_rows = _make_task1_rows(60)
    task2_rows = _make_task2_rows(task1_rows)

    split_a = build_shared_split(task1_rows, task2_rows, random_state=42)
    split_b = build_shared_split(task1_rows, task2_rows, random_state=42)

    ids_a = {r["paragraph_id"] for r in split_a.task1_train}
    ids_b = {r["paragraph_id"] for r in split_b.task1_train}
    assert ids_a == ids_b
    val_ids_a = {r["paragraph_id"] for r in split_a.task1_val}
    val_ids_b = {r["paragraph_id"] for r in split_b.task1_val}
    assert val_ids_a == val_ids_b


def test_task1_task2_paragraph_id_alignment():
    task1_rows = _make_task1_rows(60)
    task2_rows = _make_task2_rows(task1_rows)

    split = build_shared_split(task1_rows, task2_rows, random_state=42)

    task1_train_ids = {r["paragraph_id"] for r in split.task1_train}
    task1_val_ids = {r["paragraph_id"] for r in split.task1_val}
    task2_train_ids = {r["paragraph_id"] for r in split.task2_train}
    task2_val_ids = {r["paragraph_id"] for r in split.task2_val}

    assert task2_train_ids == task1_train_ids
    assert task2_val_ids == task1_val_ids
    assert task1_train_ids.isdisjoint(task1_val_ids)


def test_strat_key_rare_signature_collapses_without_crashing():
    # A handful of rows share common signatures, plus two rows each with their own
    # totally unique label signature (count=1 individually) -- build_shared_split must
    # collapse both into the shared "{domain}|RARE" bucket (count=2 together) before
    # calling train_test_split's stratify, or sklearn raises on a singleton class.
    task1_rows = _make_task1_rows(60)
    task1_rows.append(
        {"paragraph_id": "unique_one", "text": "one of a kind", "labels": ["CO", "ST", "TE"], "type": "editorial"}
    )
    task1_rows.append(
        {"paragraph_id": "unique_two", "text": "also one of a kind", "labels": ["CO", "AN"], "type": "editorial"}
    )
    task2_rows = _make_task2_rows(task1_rows)

    split = build_shared_split(task1_rows, task2_rows, random_state=42)
    all_ids = {r["paragraph_id"] for r in split.task1_train} | {r["paragraph_id"] for r in split.task1_val}
    assert all_ids == {r["paragraph_id"] for r in task1_rows}


def test_strat_key_format():
    row = {"type": "editorial", "labels": ["ST", "AS"]}
    assert strat_key(row) == "editorial|AS,ST"
    row_no_labels = {"type": "debate", "labels": []}
    assert strat_key(row_no_labels) == "debate|NONE"
