import json

import pytest

from scripts.summarize_run import paired_f1


def write_scores(path, value, *, query_prefix="query"):
    rows = [
        {
            "query_id": f"{query_prefix}-{question}",
            "draw": draw,
            "metrics": {"f1": value},
        }
        for question in range(32)
        for draw in range(2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_paired_interval_uses_question_differences(tmp_path):
    reference, candidate = tmp_path / "reference.jsonl", tmp_path / "candidate.jsonl"
    write_scores(reference, 0.125)
    write_scores(candidate, 0.375)
    result = paired_f1(reference, candidate)
    assert result["mean_difference"] == 0.25
    assert result["question_bootstrap_95pct_ci"] == [0.25, 0.25]
    assert result["queries"] == 32 and result["draws_per_query"] == 2


def test_comparison_refuses_repeated_draw_or_changed_questions(tmp_path):
    reference, candidate = tmp_path / "reference.jsonl", tmp_path / "candidate.jsonl"
    write_scores(reference, 0.125)
    write_scores(candidate, 0.375, query_prefix="different")
    with pytest.raises(ValueError, match="query sets differ"):
        paired_f1(reference, candidate)
    write_scores(candidate, 0.375)
    with candidate.open("a") as handle:
        handle.write(candidate.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Duplicate query/draw"):
        paired_f1(reference, candidate)
