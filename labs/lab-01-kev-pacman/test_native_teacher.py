"""Native action geometry, teacher isolation and fail-closed qualification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import subprocess

from gameplay_benchmark import BenchmarkEngine, SPEC
from pacman_lab import ROOT, destination, engine_script, position, teacher
from teacher_validation import VALIDATION, qualification, require_qualified_teacher
from teacher_data import generate


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
                   'loop_decisions':0,'longest_no_pellet_decisions':12,
                   'multi_tile_actions':0,'stationary_actions':0,
                   'nonterminal_multi_tile_actions':0,'nonterminal_stationary_actions':0,
                   'outcome':'level_cleared','pellets_remaining':0}
        report = {'phase':'qualification','algorithm':VALIDATION['teacher'],'options':VALIDATION['teacher_options'],
                  'episodes':[{'level':level,'seed':seed,'metrics':dict(metrics)}
                    for level in VALIDATION['levels'] for seed in VALIDATION['qualification_seeds']]}
        self.assertTrue(qualification(report)['approved_for_training'])
        failed_simulation=copy.deepcopy(report);failed_simulation['errors']=[{'level':5,'seed':1,'error':'native action timeout'}]
        self.assertFalse(qualification(failed_simulation)['approved_for_training'])
        for key,value in [('loop_decisions',1),('avoidable_immediate_deaths',1),
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


if __name__=='__main__':unittest.main()
