from emotional_baseline.inventory import choose_rating_export


def test_choose_rating_export_prefers_completed_ratings():
    exports = [
        {"path": "empty.csv", "rows": 120, "missing_columns": [],
         "missing_valence": 120, "missing_arousal": 120},
        {"path": "complete.csv", "rows": 120, "missing_columns": [],
         "missing_valence": 0, "missing_arousal": 0},
    ]
    assert choose_rating_export(exports)["path"] == "complete.csv"
