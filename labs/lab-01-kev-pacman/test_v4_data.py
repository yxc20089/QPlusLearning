"""CPU-only checks for hard mining; never generate/train a full v4 dataset."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import v4_data as v4
from gameplay_benchmark import BenchmarkEngine
from pacman_lab import ROOT, body
from teacher_validation import fingerprint


IDENTITY = {'checkpoint_name': 'kev-4b-pacman-native-v3', 'base': 'Qwen/Qwen3.5-4B-Base',
    'base_revision': '1001bb4d826a52d1f399e183466143f4da7b741b', 'stage': 'pacman',
    'pacman_fine_tuned': True, 'optimizer_steps': 762, 'checkpoint_sha256': 'a'*64,
    'adapter_sha256': 'b'*64}


class V4DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with BenchmarkEngine() as engine:
            cls.state = engine.request('reset', options={'seed': 400001, 'level': 1, 'action_version': 2})
        cls.options = json.loads((ROOT/'evaluation/teacher-qualification.json').read_text())['options']

    def config(self, folder):
        config = {'version': v4.PREFIX, 'protocol': v4.SPEC, 'teacher_options': self.options,
            'source_sha256': v4.source_hashes(), 'generator_sha256': v4._hash(Path(v4.__file__)),
            'qualification_sha256': 'c'*64, 'probe_checkpoint': IDENTITY, 'episode_sets': {},
            'targets': v4.TARGETS, 'source_mix': v4.SOURCE_MIX, 'harvest': v4.HARVEST,
            'harm_thresholds': v4.HARM, 'minimums': v4.MINIMUMS}
        v4._save(Path(folder)/'collection.json', config)
        return config

    def candidate(self, folder, source='teacher-episode', origin='off_policy'):
        return v4._candidate(folder, split='train', seed=400001, level=1, origin=origin,
            source={'kind': source}, state=self.state, actions=[], hashes=[fingerprint(self.state)])

    def metrics(self, **changes):
        value = {'life_losses':0, 'decisions':100, 'simulation_frames':1000,
            'longest_no_pellet_decisions':10, 'loop_decisions':0, 'level_cleared':True,
            'outcome':'level_cleared', 'pellets_remaining':0, 'longest_loop_streak_decisions':0,
            'multi_tile_actions':0, 'stationary_actions':0, 'avoidable_immediate_deaths':0}
        value.update(changes); return value

    def record(self, number, kind='hard', origin='off_policy', flags=(), family=None, digest=None):
        return {'_meta':{'id':str(number), 'class':kind, 'origin':origin,
            'state_sha256':digest or str(number), 'group_id':family or str(number),
            'cohorts':{flag:True for flag in flags}, 'margin':number}}

    def test_frozen_identity_and_seed_family_partition(self):
        v4.require_v3_identity(IDENTITY)
        for changes in ({'checkpoint_name':'kev-4b-pacman-native-v2'}, {'optimizer_steps':0},
                        {'base_revision':'main'}, {'adapter_sha256':'unknown'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                v4.require_v3_identity({**IDENTITY, **changes})
        v4.check_seed_splits([400001], [460003])
        for train,dev in (([400001], [400001]), ([50011], [460003]), ([91009], [460003]),
                          ([300017], [460003]), ([True], [460003])):
            with self.subTest(train=train), self.assertRaises(ValueError): v4.check_seed_splits(train,dev)
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(folder)
            v4._set_seeds(folder, config, 'offline', [400001], [460003], 'wave-001', {'roots':4})
            v4._set_seeds(folder, config, 'offline', [400001], [460003], 'wave-001', {'roots':4})
            with self.assertRaisesRegex(ValueError, 'append a new wave_id'):
                v4._set_seeds(folder, config, 'offline', [400001], [460003], 'wave-001', {'roots':8})
            v4._set_seeds(folder, config, 'offline', [400009], [460009], 'wave-002', {'roots':8})
            with self.assertRaisesRegex(ValueError, 'change partition'):
                v4.check_seed_splits([460003], [480019], folder)

    def test_counterfactual_preference_is_not_a_hard_negative(self):
        risks = {'left':{'life_lost':False}, 'right':{'life_lost':False}}
        for changes in ({}, {'decisions':107,'simulation_frames':1100},
                        {'decisions':120,'simulation_frames':1059}):
            result = v4.negative_evidence(self.metrics(), self.metrics(**changes), risks, 'left', 'right')
            self.assertFalse(result['harmful'])
        result = v4.negative_evidence(self.metrics(), self.metrics(decisions=108, simulation_frames=1060), risks, 'left', 'right')
        self.assertEqual(result['kind'],'verified_route_cost')
        self.assertFalse(v4.negative_evidence(self.metrics(life_losses=1),
            self.metrics(decisions=200, simulation_frames=2000),risks,'left','right')['harmful'],
            'A slower route that saves a life is not an efficiency negative')
        result = v4.negative_evidence(self.metrics(), self.metrics(life_losses=1), risks, 'left', 'right')
        self.assertEqual(result['kind'],'extra_life_loss_with_same_continuation_controller')
        risks['right']['life_lost'] = True
        self.assertEqual(v4.negative_evidence(self.metrics(), self.metrics(life_losses=1), risks, 'left','right')['kind'],
                         'immediate_fatal_with_safe_teacher')

    def test_expiry_counts_currently_edible_ghosts_crossing_action_time(self):
        state = copy.deepcopy(self.state); state['frightened'] = True
        state['timing']['power']['remaining_frames'] = 3
        ghost = state['ghosts'][0]; ghost.update(mode='outside', frightened=True,
            row=state['player']['row'], column=state['player']['column']-1)
        risk = {'left':{'life_lost':False,'action_frames':8}, 'right':{'life_lost':True,'action_frames':7}}
        flags = v4.cohort_flags(state, risk)
        self.assertTrue(flags['power_expiry']); self.assertTrue(flags['expiry_during_action_with_near_ghost'])
        state['timing']['power']['remaining_frames'] = 120
        flags = v4.cohort_flags(state, risk)
        self.assertTrue(flags['power_expiry']); self.assertFalse(flags['expiry_during_action_with_near_ghost'])

    def test_exact_source_quotas_reserve_rare_hard_cohorts_and_reject_easy_fill(self):
        rows = [self.record(i, origin='off_policy' if i<6 else 'learner', flags=('anticipatory_harm',) if i in (0,6) else ()) for i in range(10)]
        rows += [self.record(20,'informative','off_policy'), self.record(21,'informative','learner')]
        result = v4.select_partition(rows, {'hard':10,'informative':2}, minimums={'anticipatory_harm':2})
        self.assertEqual(len(result),12)
        self.assertEqual(sum(r['_meta']['cohorts'].get('anticipatory_harm',False) for r in result),2)
        with self.assertRaisesRegex(ValueError, 'behavior cohorts'):
            v4.select_partition(rows, {'hard':10,'informative':2}, minimums={'power_expiry':1})
        with self.assertRaisesRegex(ValueError, 'hard/source quotas'):
            v4.select_partition(rows[1:]+[self.record(99,'informative')], {'hard':10,'informative':2})
        self.assertEqual(v4.select_partition(rows, {'hard':0,'informative':0}), [])

    def test_dedup_and_family_cap_do_not_hide_quota_shortages(self):
        rows = [self.record(i, origin='off_policy' if i<6 else 'learner', family='one-family') for i in range(10)]
        with self.assertRaisesRegex(ValueError, 'hard/source quotas'):
            v4.select_partition(rows, {'hard':10,'informative':0}, family_cap=2)
        rows = [self.record(i, origin='off_policy' if i<6 else 'learner', digest='same') for i in range(10)]
        with self.assertRaisesRegex(ValueError, 'hard/source quotas'):
            v4.select_partition(rows, {'hard':10,'informative':0})

    def test_same_state_multiple_provenances_are_resumable_and_pool_counts_once(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(folder)
            offline = self.candidate(folder)
            learner = self.candidate(folder, 'v3-episode', 'learner')
            wave2 = self.candidate(folder, 'teacher-wave-002')
            self.assertEqual(offline,self.candidate(folder)); self.assertEqual(len({offline,learner,wave2}),3)
            for index,identifier in enumerate((offline,learner,wave2)):
                path = Path(folder)/'cases'/identifier; candidate = v4._load(path/'candidate.json')
                probe = {'checkpoint':IDENTITY}; v4._save(path/'probe.json',probe)
                receipt = {'accepted':True,'classification':'hard', 'files':{}, 'cohorts':{},
                    'native_verified':True, 'checkpoint':IDENTITY, 'teacher_choice':'left','learner_choice':'right',
                    'teacher_probability':.1,'learner_probability':.9, 'learner_minus_teacher_log_probability':index,
                    'immediate_counterfactuals':{}, 'negative_evidence':{'harmful':True}}
                v4._save(path/'verification.json',receipt)
            with patch.object(v4,'_validate_binding'):
                pool,duplicates = v4._pool(folder,config)
            self.assertEqual(len(pool['train']),1)
            self.assertEqual(duplicates['exact_state_duplicates'],2)
            self.assertEqual(pool['train'][0]['_meta']['id'],wave2)
            changed = Path(folder)/'cases'/learner/'verification.json'
            receipt = v4._load(changed); receipt['teacher_choice']='right'; v4._save(changed,receipt)
            with patch.object(v4,'_validate_binding'):
                pool,duplicates = v4._pool(folder,config)
            self.assertEqual(pool['train'],[])
            self.assertEqual(duplicates['ambiguous_teacher_label_states'],1)
            self.assertEqual(len(v4._load(Path(folder)/'ambiguous-states.json')['states']),1)

    def test_probe_validates_whole_batch_before_committing_and_can_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            self.config(folder); one = self.candidate(folder); two = self.candidate(folder,'another-teacher-episode')
            info = lambda: copy.deepcopy(IDENTITY)
            valid = v4._one_hot(self.state,'left')
            bad = copy.deepcopy(valid); bad['answers']['move']['probabilities']['right'] = .7
            with self.assertRaises(ValueError):
                v4.probe_candidates(folder,info,lambda requests:[valid,bad],batch_size=2)
            self.assertFalse((Path(folder)/'cases'/one/'probe.json').exists())
            self.assertFalse((Path(folder)/'cases'/two/'probe.json').exists())
            result = v4.probe_candidates(folder,info,lambda requests:[valid for r in requests],batch_size=2)
            self.assertEqual(result['completed'],2)
            callback = Mock(side_effect=AssertionError('Completed probes must be reused'))
            self.assertEqual(v4.probe_candidates(folder,info,callback)['completed'],0)
            path = Path(folder)/'cases'/one/'candidate.json'; value = v4._load(path); value['source']['changed']=True; v4._save(path,value)
            with self.assertRaisesRegex(ValueError,'Cached v3 probe source changed'):
                v4.probe_candidates(folder,info,callback)

    def test_probe_rejects_adapter_change_without_any_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            self.config(folder); identifier = self.candidate(folder)
            info = Mock(side_effect=[IDENTITY,IDENTITY,{**IDENTITY,'adapter_sha256':'d'*64}])
            with self.assertRaisesRegex(RuntimeError,'changed during'):
                v4.probe_candidates(folder,info,lambda requests:[v4._one_hot(self.state,'left')])
            self.assertFalse((Path(folder)/'cases'/identifier/'probe.json').exists())

    def test_completed_files_keeps_rejected_evidence_and_excludes_inflight(self):
        with tempfile.TemporaryDirectory() as folder:
            self.config(folder); identifier = self.candidate(folder); directory = Path(folder)
            trace = directory/'cases'/identifier/'teacher.jsonl'; trace.write_text('complete-proof\n')
            v4._save(trace.with_name('verification.json'),{'accepted':False,'files':{trace.relative_to(directory).as_posix():v4._hash(trace)}})
            (directory/'unfinished.json.tmp').write_text('partial'); (trace.with_name('negative.jsonl.tmp')).write_text('partial')
            files = v4.completed_files(directory)
            self.assertIn('collection.json',files); self.assertIn(trace.relative_to(directory).as_posix(),files)
            self.assertTrue(all(not name.endswith('.tmp') for name in files))
            trace.write_text('changed-proof\n')
            with self.assertRaisesRegex(ValueError,'missing or changed'): v4.completed_files(directory)

    def test_interrupt_saves_completed_receipts_and_cancels_bounded_workers(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            cancelled, pool = Mock(), Mock()
            with patch.object(v4,'ProcessPoolExecutor',return_value=pool), patch.object(v4.multiprocessing,'get_context') as context, \
                    patch.object(v4,'wait',side_effect=KeyboardInterrupt), patch.object(v4,'_backup') as backup:
                context.return_value.Event.return_value = cancelled
                with self.assertRaises(KeyboardInterrupt):
                    v4._run_jobs([{'id':str(i)} for i in range(100)],Mock(),folder,'fixture',workers=2,backup_directory='drive')
                self.assertEqual(pool.submit.call_count,2); cancelled.set.assert_called_once()
                pool.shutdown.assert_called_once_with(wait=True,cancel_futures=True)
                self.assertEqual(backup.call_count,2)
            with patch.object(v4,'_CANCELLED',cancelled):
                cancelled.is_set.return_value = True
                with self.assertRaises(InterruptedError): v4._check_cancel()

    def test_native_prefix_replays_and_modified_ghost_state_fails(self):
        actions,hashes = [],[]
        with BenchmarkEngine() as engine:
            state = engine.request('reset',options={'seed':400001,'level':1,'action_version':2}); hashes.append(fingerprint(state))
            for _ in range(8):
                move = state['player']['heading'] if state['player']['heading'] in state['legal_moves'] else state['legal_moves'][0]
                step = engine.step(move); self.assertIsNone(step['outcome'])
                actions.append(move); state=step['state']; hashes.append(fingerprint(state))
        candidate = {'seed':400001,'level':1,'prefix_actions':actions,'prefix_state_sha256':hashes,'request':body(state)}
        with BenchmarkEngine() as engine: self.assertEqual(v4._replay_prefix(engine,candidate),state)
        changed = copy.deepcopy(candidate); changed['request']['state']['ghosts'][0]['column'] += 1
        with BenchmarkEngine() as engine, self.assertRaisesRegex(ValueError,'exact native state'):
            v4._replay_prefix(engine,changed)

    def test_native_two_turn_verification_rejects_incomplete_recovery(self):
        # Only two actions per branch on a fresh, unreserved seed. This confirms the
        # real engine/teacher/replay path without a full episode or training run.
        with tempfile.TemporaryDirectory() as folder:
            self.config(folder); identifier = self.candidate(folder)
            with BenchmarkEngine() as engine:
                engine.request('reset',options={'seed':400001,'level':1,'action_version':2})
                teacher = engine.request('teacher',options=self.options)['choice']
            other = next(d for d in self.state['legal_moves'] if d != teacher)
            v4.probe_candidates(folder,lambda:IDENTITY,lambda requests:[v4._one_hot(self.state,other)])
            receipt = v4._verification_job({'id':identifier,'directory':folder,'options':self.options,
                'source_sha256':v4.source_hashes(),'caps':{**v4.SPEC,'max_decisions':2}})
            self.assertFalse(receipt['accepted']); self.assertIn('maze incomplete',receipt['failures'])
            self.assertTrue(receipt['native_verified']); self.assertEqual(receipt['teacher_metrics']['decisions'],2)
            self.assertEqual(receipt['negative_metrics']['decisions'],2)
            self.assertEqual(len(receipt['files']),2)
            self.assertTrue(all(name in v4.completed_files(folder) for name in receipt['files']))

    def test_probability_margin_and_native_cohort_edits_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(folder); identifier = self.candidate(folder)
            path = Path(folder)/'cases'/identifier; candidate = v4._load(path/'candidate.json')
            response = {'answers':{'move':{'choice':'right','probabilities':{'left':.1,'right':.9}}}}
            probe = {'candidate_sha256':v4._hash(path/'candidate.json'),'checkpoint':IDENTITY,
                'response':response,'choice':'right','probabilities':{'left':.1,'right':.9}}
            v4._save(path/'probe.json',probe)
            risks = {d:{'life_lost':False,'action_frames':8} for d in self.state['legal_moves']}
            teacher_metrics,negative_metrics = self.metrics(),self.metrics(life_losses=1)
            import math
            receipt = {'candidate_sha256':v4._hash(path/'candidate.json'),'probe_sha256':v4._hash(path/'probe.json'),
                'checkpoint':IDENTITY,'source_sha256':config['source_sha256'],'options':self.options,'protocol':v4.SPEC,
                'teacher_choice':'left','learner_choice':'right','teacher_probability':.1,'learner_probability':.9,
                'learner_minus_teacher_log_probability':math.log(9),'native_verified':True,
                'immediate_counterfactuals':risks,'teacher_metrics':teacher_metrics,'negative_metrics':negative_metrics,
                'negative_evidence':v4.negative_evidence(teacher_metrics,negative_metrics,risks,'left','right'),
                'classification':'hard','cohorts':{**v4.cohort_flags(self.state,risks),'anticipatory_harm':True}}
            args = (candidate,probe,receipt,config,v4._hash(path/'candidate.json'),v4._hash(path/'probe.json'))
            v4._validate_binding(*args)
            changed = copy.deepcopy(receipt); changed['learner_minus_teacher_log_probability']=0
            with self.assertRaisesRegex(ValueError,'probability margin'):
                v4._validate_binding(candidate,probe,changed,*args[3:])
            changed = copy.deepcopy(receipt); changed['cohorts']['power_expiry']=True
            with self.assertRaisesRegex(ValueError,'cohort changed'):
                v4._validate_binding(candidate,probe,changed,*args[3:])
            changed = copy.deepcopy(receipt); changed['negative_evidence']['harmful']=False
            with self.assertRaisesRegex(ValueError,'harm classification'):
                v4._validate_binding(candidate,probe,changed,*args[3:])

    def test_informative_windows_have_one_generation_and_no_repeated_pending_work(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(folder); actions=[]; hashes=[]
            with BenchmarkEngine() as engine:
                state = engine.request('reset',options={'seed':400001,'level':1,'action_version':2})
                hashes.append(fingerprint(state))
                for _ in range(8):
                    move = state['player']['heading'] if state['player']['heading'] in state['legal_moves'] else state['legal_moves'][0]
                    state = engine.step(move)['state']; actions.append(move); hashes.append(fingerprint(state))
            identifier = v4._candidate(folder,split='train',seed=400001,level=1,origin='off_policy',source={'kind':'window-fixture'},
                state=state,actions=actions,hashes=hashes)
            path=Path(folder)/'cases'/identifier; v4._save(path/'probe.json',{'checkpoint':IDENTITY})
            (path/'teacher.jsonl').write_text('')
            receipt = {'accepted':True,'classification':'hard','candidate_sha256':v4._hash(path/'candidate.json'),
                'probe_sha256':v4._hash(path/'probe.json'),'files':{}}
            v4._save(path/'verification.json',receipt)
            def run(jobs,worker,*args,**kwargs):
                for job in jobs: worker(job)
                return {'status':'complete','completed':len(jobs),'errors':[]}
            with patch.object(v4,'_configuration',return_value=config),patch.object(v4,'_run_jobs',side_effect=run):
                first=v4.verify_cases(folder,'fixture'); second=v4.verify_cases(folder,'fixture')
            self.assertEqual(first['new_informative_candidates'],3)
            self.assertEqual(second['new_informative_candidates'],0)
            self.assertEqual(second['pending_probe_count'],3)
            for candidate_path in (Path(folder)/'cases').glob('*/candidate.json'):
                candidate=v4._load(candidate_path)
                if candidate['case_role']=='informative': self.assertEqual(v4._derive_windows(folder,candidate,receipt),[])


if __name__ == '__main__': unittest.main()
