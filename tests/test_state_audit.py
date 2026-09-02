import json
import pytest
from scripts.audit_states import collect, state_metrics


def test_inclusive_intervals_and_excursion_magnitude():
    result = state_metrics([299, 300, 1800, 1802], 300, 1800)
    assert result['violation_hours'] == 2
    assert result['violation_rate_pct'] == 50
    assert result['mean_outside_distance'] == .75
    assert state_metrics([300, 1800], 300, 1800)['all_hours_in_range']


@pytest.mark.parametrize('values', [[], [float('nan')], [None], [float('inf')]])
def test_invalid_samples_are_not_silently_counted_as_safe(values):
    with pytest.raises(ValueError):
        state_metrics(values, 300, 1800)


def test_control_qualification_is_not_state_qualification(tmp_path):
    row = {'method': 'example', 'scenario': 'baseline', 'seed': 42,
           'profit': 10, 'penalty': 0, 'threshold_qualified': True,
           'plan': [[25, 100, 600, 70]]*24,
           'provenance': {'protocol_hash': 'p', 'source_hash': 's', 'landscape_hash': 'l'},
           'details': {'co2_realized': [1801]*24, 'rh_realized': [70]*24,
                       'soil_water_content_hourly': [.3]*24}}
    (tmp_path/'run.json').write_text(json.dumps(row))
    hourly, states, joint = collect(tmp_path)
    assert len(hourly) == 24 and len(states) == 3
    assert joint[0]['control_qualified']
    assert not joint[0]['joint_qualified']
    (tmp_path/'duplicate.json').write_text(json.dumps(row))
    with pytest.raises(ValueError, match='duplicate'):
        collect(tmp_path)
