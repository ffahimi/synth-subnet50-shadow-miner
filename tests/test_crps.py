import numpy as np

from synth_shadow.scoring.crps import crps_sum_over_interval, score_synth_btc_24h


def test_validator_crps_sums_non_overlapping_bps_increments():
    realized = np.array([100.0, 101.0, 102.0, 103.0])
    predicted = np.array(
        [
            [100.0, 101.0, 102.0, 103.0],
            [100.0, 102.0, 104.0, 106.0],
        ]
    )

    expected = 0.0
    for idx in range(1, realized.shape[0]):
        observation = ((realized[idx] - realized[idx - 1]) / realized[idx - 1]) * 10000.0
        forecast = ((predicted[:, idx] - predicted[:, idx - 1]) / predicted[:, idx - 1]) * 10000.0
        expected += np.mean(np.abs(forecast - observation)) - abs(forecast[1] - forecast[0]) / 4

    assert crps_sum_over_interval(predicted, realized, 1) == expected


def test_validator_crps_uses_non_overlapping_step_points():
    realized = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    predicted = np.array(
        [
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            [100.0, 101.0, 102.0, 106.0, 104.0, 105.0, 112.0],
        ]
    )

    observation_0 = ((realized[3] - realized[0]) / realized[0]) * 10000.0
    forecast_0 = ((predicted[:, 3] - predicted[:, 0]) / predicted[:, 0]) * 10000.0
    observation_1 = ((realized[6] - realized[3]) / realized[3]) * 10000.0
    forecast_1 = ((predicted[:, 6] - predicted[:, 3]) / predicted[:, 3]) * 10000.0
    expected = (
        np.mean(np.abs(forecast_0 - observation_0))
        - abs(forecast_0[1] - forecast_0[0]) / 4
        + np.mean(np.abs(forecast_1 - observation_1))
        - abs(forecast_1[1] - forecast_1[0]) / 4
    )

    assert crps_sum_over_interval(predicted, realized, 3) == expected


def test_score_synth_btc_24h_matches_validator_component_scale():
    realized = np.linspace(100.0, 128.8, 289)
    predicted = np.vstack(
        [
            realized,
            realized + np.linspace(0.0, 12.0, 289),
        ]
    )

    score = score_synth_btc_24h(predicted, realized)

    assert score["components"]["crps_5m"] == crps_sum_over_interval(predicted, realized, 1)
    assert score["components"]["crps_30m"] == crps_sum_over_interval(predicted, realized, 6)
    assert score["components"]["crps_3h"] == crps_sum_over_interval(predicted, realized, 36)
    assert score["components"]["crps_24h"] == crps_sum_over_interval(
        predicted,
        realized,
        288,
        absolute_price=True,
    )
    assert score["raw_crps"] == (
        score["components"]["crps_5m"]
        + score["components"]["crps_30m"]
        + score["components"]["crps_3h"]
        + score["components"]["crps_24h"]
    )
