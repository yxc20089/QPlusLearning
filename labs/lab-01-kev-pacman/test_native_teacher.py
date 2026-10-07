"""Native action geometry, teacher isolation and fail-closed qualification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import hashlib
import zipfile
from unittest.mock import patch

from gameplay_benchmark import BenchmarkEngine, SPEC, diagnostic_record, summarize
from pacman_lab import ROOT, body, destination, engine_script, position, teacher
from teacher_validation import VALIDATION, qualification, require_qualified_teacher, verify_replays
from teacher_data import COLLECTION, PREFIX, generate, prefix_move, validate_partitions, verify_collection
from teacher_validation import fingerprint


class NativeTeacherTests(unittest.TestCase):
    def test_native_consecutive_ghost_eating_pauses_do_not_timeout_a_valid_move(self):
        fixture = """
globalThis.longGhostPause = function() {
 labArcade.reset({seed:7,level:5,action_version:2,skipReady:true});
 pacman.setPos(6*tileSize+midTile.x,8*tileSize+midTile.y);pacman.setDir(DIR_RIGHT);
 energizer.activate();
 ghosts.forEach(function(g) {g.setPos(pacman.pixel.x,pacman.pixel.y);g.mode=GHOST_OUTSIDE;g.scared=true;});
 return labArcade.step('right');
};
"""
        source=engine_script();end=source.rfind('})();');source=source[:end]+fixture+source[end:]
        worker=(ROOT/'games/arcade-worker.js').read_text().split('readline.createInterface')[0]
        with tempfile.TemporaryDirectory() as directory:
            game,runner=Path(directory)/'engine.js',Path(directory)/'fixture.js'
            game.write_text(source);runner.write_text(worker+'process.stdout.write(JSON.stringify(context.longGhostPause()));')
            result=json.loads(subprocess.check_output(['node',str(runner),str(game)],text=True))
        self.assertGreater(result['action_frames'],180)
        self.assertEqual(position(result['state']['player']),(8,7))
        self.assertGreaterEqual(result['state']['score'],1400)
        self.assertIsNone(result['outcome'])

    def test_browser_and_headless_use_identical_v2_boundaries_and_reverse_moves(self):
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'engine.js';script.write_text(engine_script())
            output=json.loads(subprocess.check_output(['node',str(ROOT/'games/test-arcade-browser.cjs'),
                str(script),str(ROOT/'games/arcade-browser.js'),'2'],text=True))
        with BenchmarkEngine() as engine:
            engine.request('reset',options={'seed':7,'action_version':2})
            self.assertEqual(output['firstState'],engine.step('left')['state'])
            self.assertEqual(output['reverseState'],engine.step('right')['state'])

    def test_adjacent_entry_survives_reversals_power_speed_and_respawns(self):
        reversals = powers = losses = 0
        with BenchmarkEngine() as engine:
            for level in (1, 2, 3, 5):
                state = engine.request('reset', options={'seed':7,'level':level,'action_version':2})
                for turn in range(700):
                    move = teacher(state)
                    reversals += move == {'up':'down','down':'up','left':'right','right':'left'}[state['player']['heading']]
                    expected = destination(state['maze'],position(state['player']),move)
                    result = engine.step(move)
                    after = result['state']; powers += result['events']['power_pellets']
                    if not result['outcome']:
                        self.assertEqual(position(after['player']),expected)
                        self.assertGreater(result['action_frames'],0)
                    self.assertIn('pixel_offset',after['player'])
                    self.assertIn('remaining_frames',after['timing']['power'])
                    self.assertEqual(set(after['destination_visits']),set(after['legal_moves']))
                    self.assertLessEqual(len(after['recent_positions']),64)
                    state = after
                    if result['outcome']=='level_cleared': break
                    if result['outcome']=='life_lost':
                        losses += 1
                        continuation = engine.request('continue')
                        if continuation['status']=='game_over':break
                        state = continuation['state']
                        self.assertGreater(state['life_epoch'],0)
        self.assertGreater(reversals,0)
        self.assertGreater(powers,0)
        self.assertGreater(losses,0)

    def test_teacher_and_risks_do_not_alter_future_native_trajectory(self):
        with BenchmarkEngine() as planned, BenchmarkEngine() as control:
            state = planned.request('reset',options={'seed':71009,'level':2,'action_version':2})
            control.request('reset',options={'seed':71009,'level':2,'action_version':2})
            for turn in range(360):
                if turn%31==0:
                    plan = planned.request('teacher',options={'horizon_frames':80,'buffers':[2],'scenario_seeds':[11117]})
                    self.assertEqual({c['action'] for c in plan['candidates']},set(state['legal_moves']))
                    self.assertEqual(planned.request('observe'),state)
                planned.request('risks')
                move = teacher(state)
                actual, expected = planned.step(move), control.step(move)
                self.assertEqual(actual,expected)
                state = actual['state']
                if actual['outcome']=='level_cleared':break
                if actual['outcome']=='life_lost':
                    actual, expected = planned.request('continue'), control.request('continue')
                    self.assertEqual(actual,expected)
                    if actual['status']=='game_over':break
                    state = actual['state']

    def test_every_gate_is_recomputed_and_development_never_authorizes_data(self):
        metrics = {'level_cleared':True,'life_losses':0,'avoidable_immediate_deaths':0,
                   'loop_decisions':0,'longest_loop_streak_decisions':0,'longest_no_pellet_decisions':12,
                   'multi_tile_actions':0,'stationary_actions':0,
                   'nonterminal_multi_tile_actions':0,'nonterminal_stationary_actions':0,
                   'outcome':'level_cleared','pellets_remaining':0}
        report = {'phase':'qualification','algorithm':VALIDATION['teacher'],'options':VALIDATION['teacher_options'],
                  'episodes':[{'level':level,'seed':seed,'metrics':dict(metrics)}
                    for level in VALIDATION['levels'] for seed in VALIDATION['qualification_seeds']]}
        self.assertTrue(qualification(report)['approved_for_training'])
        brief_repeat=copy.deepcopy(report)
        brief_repeat['episodes'][0]['metrics'].update(loop_decisions=1,longest_loop_streak_decisions=1)
        self.assertTrue(qualification(brief_repeat)['approved_for_training'])
        failed_simulation=copy.deepcopy(report);failed_simulation['errors']=[{'level':5,'seed':1,'error':'native action timeout'}]
        self.assertFalse(qualification(failed_simulation)['approved_for_training'])
        for key,value in [('longest_loop_streak_decisions',8),('avoidable_immediate_deaths',1),
                          ('longest_no_pellet_decisions',129),('pellets_remaining',1),
                          ('nonterminal_multi_tile_actions',1),('nonterminal_stationary_actions',1),
                          ('level_cleared',False),('life_losses',5),('outcome','no_progress_watchdog')]:
            changed = copy.deepcopy(report); changed['episodes'][0]['metrics'][key]=value
            self.assertFalse(qualification(changed)['approved_for_training'],key)
        report['phase']='development'
        self.assertFalse(qualification(report)['approved_for_training'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            report['qualification']={'approved_for_training':True}
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'not qualified'):
                require_qualified_teacher(path)
            output=Path(directory)/'labels'
            with self.assertRaisesRegex(ValueError,'not qualified'):
                generate(output,path,counts={'train':1,'development':1},workers=1)
            self.assertFalse(output.exists(), 'Unqualified teachers must not write new labels')
            report.update(phase='qualification',simulation_protocol=SPEC,
                          qualification_spec=VALIDATION,source_sha256={})
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'implementation changed'):
                require_qualified_teacher(path)

    def test_recovery_prefix_reaches_a_native_respawn_using_only_legal_moves(self):
        for level in (1,2,3,5):
            with BenchmarkEngine() as engine:
                state=engine.request('reset',options={'seed':201012,'level':level,'action_version':2})
                for _ in range(COLLECTION['prefix_max_decisions']):
                    move=prefix_move(state,engine.request('risks'))
                    self.assertIn(move,state['legal_moves'])
                    step=engine.step(move);state=step['state']
                    if step['outcome']=='life_lost':break
                    self.assertIsNone(step['outcome'], 'Prefix must reach a real death before completing a maze')
                else:self.fail('Scripted prefix did not produce a native respawn')
                continued=engine.request('continue')
                self.assertEqual(continued['status'],'playing')
                self.assertEqual(continued['state']['life_epoch'],1)
                self.assertEqual(continued['state']['timing']['release']['mode'],'global')

    def test_saved_report_with_real_cycles_roundtrips_through_native_replay(self):
        report=json.loads((ROOT/'evaluation/heuristic-development.json').read_text())
        episode=next(e for e in report['episodes'] if e['metrics']['loop_decisions'])
        with zipfile.ZipFile(ROOT/'evaluation/heuristic-development-replays.zip') as archive:
            content=archive.read(episode['replay_entry'])
        rows=[{**json.loads(line)['diagnostics'],'http_ms':0} for line in content.splitlines()]
        # Recompute the added streak metric without running the controller or
        # changing the archived native actions. JSON roundtrip stringifies periods.
        episode['metrics']=summarize(rows,episode['metrics']['outcome'])
        with tempfile.TemporaryDirectory() as directory:
            replay=Path(directory)/'replay.zip'
            with zipfile.ZipFile(replay,'w') as archive:archive.writestr(episode['replay_entry'],content)
            saved=json.loads(json.dumps({'episodes':[episode],
                'replay_sha256':hashlib.sha256(replay.read_bytes()).hexdigest()}))
            self.assertTrue(verify_replays(saved,replay))

    def test_collection_replays_a_complete_native_game_and_rejects_a_false_controller_boundary(self):
        report=json.loads((ROOT/'evaluation/teacher-development.json').read_text());episode=report['episodes'][0]
        with zipfile.ZipFile(ROOT/'evaluation/teacher-development-replays.zip') as archive:
            records=[json.loads(line) for line in archive.read(episode['replay_entry']).splitlines()]
        rows=[{**r['diagnostics'],'http_ms':0} for r in records]
        metrics=summarize(rows,episode['metrics']['outcome'])
        episode.update(metrics=metrics,full_game_metrics=metrics,collection_mode='normal',
                       prefix_decisions=0,teacher_start_life=0)
        for record in records:record['controller']='qualified_teacher'
        with tempfile.TemporaryDirectory() as directory:
            replay=Path(directory)/'collection.zip'
            def save():
                with zipfile.ZipFile(replay,'w') as archive:
                    archive.writestr(episode['replay_entry'],''.join(json.dumps(r)+'\n' for r in records))
                return json.loads(json.dumps({'episodes':[episode],
                    'replay_sha256':hashlib.sha256(replay.read_bytes()).hexdigest()}))
            self.assertTrue(verify_collection(save(),replay))
            records[0]['controller']='perturbation_prefix'
            with self.assertRaisesRegex(ValueError,'Invalid perturbation/teacher boundary'):
                verify_collection(save(),replay)

    def test_labels_are_bound_to_actual_teacher_steps_and_never_to_scripted_prefix_actions(self):
        with BenchmarkEngine() as engine:
            state=engine.request('reset',options={'seed':201012,'action_version':2})
            move=engine.request('teacher',options={'horizon_frames':16,'buffers':[1],'scenario_seeds':[11117]})['choice']
            risks=engine.request('risks');step=engine.step(move)
        group='level-1-seed-201012';name='episode.jsonl'
        request=body(state);request.pop('model');request['questions']['move'].update(label=move,src='test')
        request['_meta']={'group_id':group,'qualification_sha256':'fixture',
                          'collection_mode':'normal','equally_ranked_actions':[move]}
        manifest={'episodes':[{'level':1,'seed':201012,'split':'train','replay_entry':name}],
                  'counts':{'train':1},'groups':{'train':[group]},'coverage':{'train':{}},
                  'qualification_sha256':'fixture'}
        record={'state_sha256':fingerprint(state),'choice':move,'controller':'qualified_teacher',
                'diagnostics':diagnostic_record(state,{'answers':{'move':{'choice':move,'probabilities':{}}}},step,risks,0)}
        with tempfile.TemporaryDirectory() as directory:
            replay=Path(directory)/'replay.zip'
            def write():
                with zipfile.ZipFile(replay,'w') as archive:archive.writestr(name,json.dumps(record)+'\n')
            def contents():return {f'{PREFIX}-train.jsonl':(json.dumps(request)+'\n').encode()}
            write();self.assertTrue(validate_partitions(manifest,contents(),replay))
            request['questions']['move']['label']=next(d for d in state['legal_moves'] if d!=move)
            with self.assertRaisesRegex(ValueError,'label differs'):
                validate_partitions(manifest,contents(),replay)
            request['questions']['move']['label']=move;record['controller']='perturbation_prefix';write()
            with self.assertRaisesRegex(ValueError,'absent from the teacher-controlled'):
                validate_partitions(manifest,contents(),replay)

    def test_failed_development_collection_does_not_publish_a_training_partition(self):
        with BenchmarkEngine() as engine:
            state=engine.request('reset',options={'seed':201012,'action_version':2})
        metrics={'level_cleared':True,'outcome':'level_cleared','pellets_remaining':0,
                 'life_losses':0,'avoidable_immediate_deaths':0,'loop_decisions':1,
                 'longest_loop_streak_decisions':1,'longest_no_pellet_decisions':12,
                 'multi_tile_actions':0,'stationary_actions':0,
                 'nonterminal_multi_tile_actions':0,'nonterminal_stationary_actions':0}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);receipt=root/'qualification.json';receipt.write_text('{}')
            trace=root/'trace.jsonl'
            request=body(state);move=state['legal_moves'][0]
            trace.write_text(json.dumps({'request':request,'response':{
                'answers':{'move':{'choice':move}},'teacher':{'candidates':[]}}})+'\n')
            good={'seed':201012,'level':1,'metrics':metrics,'trace':str(trace),
                  'collection_mode':'normal','prefix_decisions':0,'teacher_start_life':0,
                  'full_game_metrics':metrics}
            failed=copy.deepcopy(good);failed['seed']=202021
            failed['metrics'].update(loop_decisions=8,longest_loop_streak_decisions=8)
            pool=unittest.mock.MagicMock()
            pool.__enter__.return_value.map.side_effect=[[good],[failed]]
            output=root/'labels'
            with patch('teacher_data.require_qualified_teacher',return_value={
                    'algorithm':VALIDATION['teacher'],'options':VALIDATION['teacher_options']}), \
                 patch('teacher_data.ProcessPoolExecutor',return_value=pool):
                with self.assertRaisesRegex(RuntimeError,'sustained no-pellet cycle'):
                    generate(output,receipt,counts={'train':1,'development':1},workers=1)
            self.assertEqual(list(output.iterdir()),[], 'No partial train/dev set after a failed collection game')


if __name__=='__main__':unittest.main()
