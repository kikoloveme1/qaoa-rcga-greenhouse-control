from dataclasses import replace
import numpy as np
import pytest
from paper_protocol.protocol import Protocol
from paper_protocol.runner import run_one, build_landscape, summarize


def test_end_to_end_saves_new_results_and_resume_checks_protocol(tmp_path):
    p=replace(Protocol(),blocks=1,bits_per_variable=1,population=8,elites=1,
              generations=2,candidate_k=4,layers=1,shots=64,qaoa_maxiter=8,
              polish=False,fitness_workers=1,backend='numpy',smoke=True)
    landscape=build_landscape(p,tmp_path)
    result=run_one(p,'qaoa_rcga','baseline',42,tmp_path,landscape)
    assert np.asarray(result['plan']).shape == (24,4)
    assert len(result['details']['co2_realized'])==24
    assert result['fitness']==pytest.approx(result['profit']-result['penalty'])
    assert result['provenance']['result_kind']=='corrected_model_smoke'
    assert run_one(p,'qaoa_rcga','baseline',42,tmp_path,landscape)['fitness']==result['fitness']
    with pytest.raises(ValueError,match='mismatch'):
        run_one(replace(p,jitter=.02),'qaoa_rcga','baseline',42,tmp_path,landscape)
    sac=run_one(p,'sac_ppo','baseline',42,tmp_path,landscape)
    assert sac['initializer']['identity']=='SAC-PPO'
    report=summarize(tmp_path)
    assert report['groups'][0]['n']==1
    assert report['groups'][0]['threshold_qualified']==1


def test_summary_rejects_different_source_versions(tmp_path):
    from paper_protocol.runner import write_json
    for i,source in enumerate(('old','new')):
        write_json(tmp_path/'runs'/f'{i}.json',{'provenance':{
            'protocol_hash':'same','source_hash':source,'landscape_hash':'same'}})
    with pytest.raises(ValueError,match='source'):
        summarize(tmp_path)


def test_named_k50_cannot_silently_use_a_different_k(tmp_path):
    with pytest.raises(ValueError,match='50'):
        run_one(Protocol(),'qaoa_50','baseline',42,tmp_path,{})


def test_summary_rejects_duplicate_runs(tmp_path):
    from paper_protocol.runner import write_json
    row={'method':'rcga','scenario':'baseline','seed':42,
         'provenance':{'protocol_hash':'p','source_hash':'s','landscape_hash':'l'}}
    for name in ('a','b'): write_json(tmp_path/'runs'/f'{name}.json',row)
    with pytest.raises(ValueError,match='duplicate'): summarize(tmp_path)


@pytest.mark.parametrize(
    ('method', 'identity'),
    [
        ('es_policy_search', 'ESPolicySearch'),
        ('sac_ppo', 'SAC-PPO'),
        ('tube_rmpc', 'TubeRMPC'),
    ],
)
def test_runner_dispatches_added_controllers(tmp_path, method, identity):
    p=replace(Protocol(),blocks=1,bits_per_variable=1,population=8,elites=1,
              generations=2,candidate_k=4,layers=1,shots=64,qaoa_maxiter=8,
              polish=False,fitness_workers=1,backend='numpy',smoke=True)
    landscape=build_landscape(p,tmp_path/'landscape')
    result=run_one(p,method,'baseline',42,None,landscape)
    assert result['initializer']['identity']==identity
    assert np.asarray(result['plan']).shape==(24,4)
