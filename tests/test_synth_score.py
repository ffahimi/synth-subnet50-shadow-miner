from synth_shadow.scoring.synth_score import compare_to_miners, top_miner_crps_stats


def test_top_miner_stats_filter_invalid_sentinel_crps():
    scores = [
        {"miner_uid": 1, "crps": -1, "scored_time": "bad"},
        {"miner_uid": 2, "crps": 20.0, "scored_time": "ok"},
        {"miner_uid": 3, "crps": 10.0, "scored_time": "ok"},
        {"miner_uid": 4, "crps": None, "scored_time": "bad"},
        {"miner_uid": 5, "crps": 30.0, "scored_time": "ok"},
    ]

    stats = top_miner_crps_stats(scores, count=2)
    comparison = compare_to_miners(15.0, scores)

    assert stats["count"] == 2
    assert stats["uids"] == [3, 2]
    assert stats["mean"] == 15.0
    assert stats["median"] == 15.0
    assert stats["min"] == 10.0
    assert comparison["miner_count"] == 3
    assert comparison["best_crps"] == 10.0
