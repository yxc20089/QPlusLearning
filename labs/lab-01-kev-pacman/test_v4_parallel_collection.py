"""Fake-engine concurrency/recovery checks; no GPU or teacher generation."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import gameplay_benchmark as gameplay
from gameplay_benchmark import BenchmarkEngine
from pacman_lab import ROOT
import v4_data as v4
import v4_parallel_collection as parallel

IDENTITY = {'checkpoint_name':'kev-4b-pacman-native-v3','base':'Qwen/Qwen3.5-4B-Base',
    'base_revision':'1001bb4d826a52d1f399e183466143f4da7b741b','stage':'pacman',
    'pacman_fine_tuned':True,'optimizer_steps':762,'checkpoint_sha256':'a'*64,'adapter_sha256':'b'*64}


class FakeEngine:
    """Three deterministic tile entries exercise the real benchmark's loop."""
    lock = threading.Lock(); active = peak = closed = 0; template = None
    def __enter__(self):
        with self.lock:
            type(self).active += 1; type(self).peak = max(type(self).peak,type(self).active)
        return self
    def __exit__(self,*args):
        with self.lock: type(self).active -= 1; type(self).closed += 1
    def request(self,command,**kwargs):
        if command == 'reset':
            self.state = copy.deepcopy(self.template); self.state['pellets_remaining']=3
            self.state['legal_moves']=['left','right']; return copy.deepcopy(self.state)
        if command == 'risks':
            return {d:{'life_lost':False,'pellets':1,'action_frames':8} for d in self.state['legal_moves']}
        raise AssertionError('Unexpected fake engine command: '+command)
    def step(self,move):
        self.state['turn'] += 1; self.state['pellets_remaining'] -= 1; self.state['score'] += 10
        self.state['player']['column'] += -1 if move == 'left' else 1
        self.state['player']['heading']=move
        return {'state':copy.deepcopy(self.state),'outcome':'level_cleared' if self.state['turn']==3 else None,
            'action_frames':8,'events':{'normal_pellets':1,'power_pellets':0,'fruit_eaten':0,'ghosts_eaten':[]}}


class ParallelCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One initial native observation supplies schema fields to the fake;
        # every test game/action after this is simulated by FakeEngine.
        with BenchmarkEngine() as engine:
            FakeEngine.template = engine.request('reset',options={'seed':400001,'level':1,'action_version':2})
    def setUp(self):
        FakeEngine.active=FakeEngine.peak=FakeEngine.closed=0
        self.fixture_spec={**v4.SPEC,'levels':[1]}
        self.configs={}
    def config(self,directory,qualification):
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        if (directory/'collection.json').exists(): return v4._load(directory/'collection.json')
        value={'version':v4.PREFIX,'protocol':self.fixture_spec,'teacher_options':{},'episode_sets':{},
            'source_sha256':v4.source_hashes(),'generator_sha256':v4._hash(Path(v4.__file__))}
        v4._save(directory/'collection.json',value); return value
    def predictor(self,request):
        return {'answers':{'move':{'choice':'left','probabilities':{'left':1.,'right':0.}}}}
    @contextlib.contextmanager
    def fixture(self):
        with patch.object(v4,'_configuration',side_effect=self.config),patch.object(v4,'SPEC',self.fixture_spec), \
             patch.object(gameplay,'BenchmarkEngine',FakeEngine),patch.object(gameplay.time,'perf_counter',return_value=0), \
             contextlib.redirect_stdout(io.StringIO()):
            yield
    def collect(self,folder,predict=None,**kwargs):
        return parallel.collect_learner_parallel(folder,lambda:copy.deepcopy(IDENTITY),predict or self.predictor,'fixture',
            train_seeds=[400001,400009],development_seeds=[460003,460009],roots_per_episode=3,**kwargs)
    def normalized_receipts(self,folder):
        values={}
        for path in Path(folder).glob('episodes/*/attempt.json'):
            value=v4._load(path); value.pop('trace'); values[path.parent.name]=value
        return values
    def candidate_payloads(self,folder):
        return {path.parent.name:v4._load(path) for path in Path(folder).glob('cases/*/candidate.json')}

    def test_parallel_matches_sequential_game_receipts_candidates_and_source_pin(self):
        with tempfile.TemporaryDirectory() as first,tempfile.TemporaryDirectory() as second,self.fixture():
            v4.collect_learner(first,lambda:copy.deepcopy(IDENTITY),self.predictor,'fixture',
                train_seeds=[400001,400009],development_seeds=[460003,460009],roots_per_episode=3)
            result=self.collect(second,workers=4)
            self.assertEqual(result['completed'],4)
            self.assertEqual(self.normalized_receipts(first),self.normalized_receipts(second))
            self.assertEqual(self.candidate_payloads(first),self.candidate_payloads(second))
            self.assertEqual(v4._load(Path(second)/'collection.json')['parallel_learner_helper_sha256'],v4._hash(Path(parallel.__file__)))
            self.assertEqual(FakeEngine.active,0)

    def test_eight_worker_bound_is_real_and_completed_games_are_reused(self):
        barrier=threading.Barrier(8); lock=threading.Lock(); joined=0
        def predict(request):
            nonlocal joined
            if request['state']['turn']==0:
                with lock: joined += 1; ordinal=joined
                if ordinal<=8: barrier.wait(timeout=5)
            return self.predictor(request)
        with tempfile.TemporaryDirectory() as folder,self.fixture():
            result=parallel.collect_learner_parallel(folder,lambda:copy.deepcopy(IDENTITY),predict,'fixture',
                train_seeds=[400001+1009*i for i in range(8)],development_seeds=[460003],roots_per_episode=3,workers=8)
            self.assertEqual(result['completed'],9); self.assertEqual(FakeEngine.peak,8); self.assertEqual(FakeEngine.closed,9)
            never=Mock(side_effect=AssertionError('Completed games must not run again'))
            result=parallel.collect_learner_parallel(folder,lambda:copy.deepcopy(IDENTITY),never,'fixture',
                train_seeds=[400001+1009*i for i in range(8)],development_seeds=[460003],roots_per_episode=3,workers=8)
            self.assertEqual(result['completed'],0); self.assertEqual(result['reused'],9); never.assert_not_called()
            config=v4._load(Path(folder)/'collection.json'); config['parallel_learner_helper_sha256']='changed'; v4._save(Path(folder)/'collection.json',config)
            with self.assertRaisesRegex(ValueError,'pinned helper'):
                parallel.collect_learner_parallel(folder,lambda:copy.deepcopy(IDENTITY),never,'fixture',workers=8)

    def test_identity_change_closes_all_engines_without_committing_mixed_games(self):
        changed=threading.Event()
        def info(): return {**IDENTITY,'adapter_sha256':'d'*64} if changed.is_set() else copy.deepcopy(IDENTITY)
        def predict(request): changed.set(); return self.predictor(request)
        with tempfile.TemporaryDirectory() as folder,self.fixture():
            with self.assertRaisesRegex(RuntimeError,'adapter changed'):
                parallel.collect_learner_parallel(folder,info,predict,'fixture',
                    train_seeds=[400001],development_seeds=[460003],roots_per_episode=3,workers=2)
            self.assertEqual(FakeEngine.active,0); self.assertEqual(len(list(Path(folder).glob('episodes/*/attempt.json'))),0)

    def test_response_identity_and_corrupt_completed_trace_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder,self.fixture():
            def wrong(request): return {**self.predictor(request),'active_checkpoint':{**IDENTITY,'adapter_sha256':'d'*64}}
            with self.assertRaisesRegex(RuntimeError,'response uses a different'):
                self.collect(folder,wrong,workers=2)
            self.assertEqual(FakeEngine.active,0)
        with tempfile.TemporaryDirectory() as folder,self.fixture():
            self.collect(folder,workers=2)
            receipt=next(Path(folder).glob('episodes/*/attempt.json')); value=v4._load(receipt)
            trace=Path(folder)/next(iter(value['files'])); trace.write_text('corrupt\n')
            with self.assertRaisesRegex(ValueError,'trajectory is missing, moved or changed'):
                self.collect(folder,workers=2)

    def test_interrupt_preserves_committed_games_and_closes_workers_before_final_backup(self):
        real_wait=parallel.wait; committed_once=False; backups=[]
        def wait_then_interrupt(*args,**kwargs):
            nonlocal committed_once
            if committed_once: raise KeyboardInterrupt
            result=real_wait(*args,**kwargs); committed_once=True; return result
        def backup(directory,target):
            backups.append({'thread':threading.current_thread().name,'active':FakeEngine.active,
                            'files':v4.completed_files(directory)})
        def predict(request): time.sleep(.005); return self.predictor(request)
        with tempfile.TemporaryDirectory() as folder,self.fixture(),patch.object(parallel,'wait',side_effect=wait_then_interrupt), \
                patch.object(v4,'_backup',side_effect=backup):
            with self.assertRaises(KeyboardInterrupt): self.collect(folder,predict,workers=1,backup_directory='fixture-drive')
            receipts=list(Path(folder).glob('episodes/*/attempt.json'))
            self.assertEqual(len(receipts),1)
            self.assertEqual(FakeEngine.active,0)
            self.assertTrue(all(b['thread']=='MainThread' for b in backups))
            self.assertEqual(backups[-1]['active'],0)
            self.assertIn(receipts[0].relative_to(Path(folder)).as_posix(),backups[-1]['files'])
            self.assertFalse(any(name.endswith('.tmp') for name in backups[-1]['files']))
        for invalid in (0,9,True,2.5):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                parallel.collect_learner_parallel('unused',lambda:IDENTITY,self.predictor,'fixture',workers=invalid)


if __name__=='__main__': unittest.main()
