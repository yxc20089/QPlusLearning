"""Instructor prework: demonstrations only after native full-game qualification."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from itertools import zip_longest
from pathlib import Path
import random
import tempfile
import time
import zipfile

from gameplay_benchmark import ROOT, SPEC, BenchmarkEngine, diagnostic_record, strata, summarize
from pacman_lab import body, destination, distances, position
from planner_data import RECIPE
from teacher_validation import VALIDATION, episode_job, episode_quality_failures, fingerprint, require_qualified_teacher, source_hashes

PREFIX = 'pacman-native-v2'
COUNTS = {'train':4096, 'development':256}
COLLECTION = {'modes':['normal','post_respawn'], 'prefix_max_decisions':512,
              'prefix_policy':'Follow a shortest maze route towards the closest outside ghost until the first native life loss. Prefer an immediately fatal legal action when available. This deliberate perturbation is not the teacher and is never labelled.',
              'quality_scope':'Apply every per-game gate to all teacher-controlled suffix decisions, including subsequent native lives. Full-game metrics also retain the intentional prefix death. Never drop failed teacher games.',
              'sampling':'Every fourth teacher-controlled move plus danger, frightened, adjacent power and late-maze states. Seeded within-episode shuffle, then interleave all four levels and both modes.'}


def prefix_move(state, risks):
    """Reach a real native respawn through legal play; do not move actors/reset physics."""
    fatal=[move for move in state['legal_moves'] if risks[move]['life_lost']]
    if fatal:return fatal[0]
    ghosts=[position(g) for g in state['ghosts'] if g['mode']=='outside']
    if not ghosts:raise RuntimeError('Perturbation prefix has no outside ghost target')
    maze=state['maze'];tile=position(state['player'])
    return min(state['legal_moves'],key=lambda move:min(
        distances(maze,destination(maze,tile,move)).get(g,999) for g in ghosts))


def collection_job(algorithm, seed, level, directory, options, mode):
    if mode=='normal':
        episode=episode_job(algorithm,seed,level,directory,options)
        episode.update(collection_mode=mode,prefix_decisions=0,teacher_start_life=0,
                       full_game_metrics=episode['metrics'])
        return episode
    if mode!='post_respawn' or algorithm!='rollout_mpc':raise ValueError('Invalid recovery collection mode')
    started=time.perf_counter();folder=Path(directory)/'recovery';folder.mkdir(parents=True,exist_ok=True)
    trace=folder/f'level-{level}-seed-{seed}.jsonl';rows=[];teacher_rows=[];life=0;active_frames=0
    teacher_started=False;prefix_count=0;stale=0;status='decision_cap'
    with BenchmarkEngine() as engine,trace.open('w') as out:
        state=engine.request('reset',options={'seed':seed,'level':level,'action_version':2})
        for turn in range(SPEC['max_decisions']):
            risks=engine.request('risks')
            if teacher_started:
                plan=engine.request('teacher',options=options);move=plan['choice']
                controller='qualified_teacher'
            else:
                if prefix_count>=COLLECTION['prefix_max_decisions']:
                    raise RuntimeError('Recovery prefix failed to reach a native respawn')
                move=prefix_move(state,risks);plan={'algorithm':'intentional_native_life_loss_prefix'}
                controller='perturbation_prefix';prefix_count+=1
            response={'answers':{'move':{'choice':move,'probabilities':{
                d:float(d==move) for d in state['legal_moves']}}},'teacher':plan}
            step=engine.step(move);row=diagnostic_record(state,response,step,risks,life)
            rows.append(row)
            if teacher_started:teacher_rows.append({**row,'life_index':life-1})
            out.write(json.dumps({'request':body(state),'response':response,'diagnostics':row,
                                  'controller':controller})+'\n')
            active_frames+=step['action_frames'];stale=stale+1 if step['state']['pellets_remaining']==state['pellets_remaining'] else 0
            state=step['state']
            if step['outcome']=='level_cleared':status='level_cleared';break
            if step['outcome']=='life_lost':
                continuation=engine.request('continue')
                if continuation['status']=='game_over':status='game_over';break
                state=continuation['state'];life+=1;stale=0;teacher_started=True
            if teacher_started and stale>=SPEC['max_no_pellet_decisions']:status='no_progress_watchdog';break
            if active_frames>=SPEC['max_simulation_frames']:status='simulation_frame_cap';break
    if not teacher_rows:raise RuntimeError('Recovery collection contains no teacher-controlled continuation')
    return {'seed':seed,'level':level,'collection_mode':mode,'prefix_decisions':prefix_count,
            'teacher_start_life':1,'metrics':summarize(teacher_rows,status),
            'full_game_metrics':summarize(rows,status),'trace':str(trace),'wall_seconds':time.perf_counter()-started}


def verify_collection(manifest, archive_path):
    """Replay all full games, retaining scripted deaths and checking teacher-only gates."""
    if hashlib.sha256(Path(archive_path).read_bytes()).hexdigest()!=manifest['replay_sha256']:
        raise ValueError('Collection replay checksum mismatch')
    with zipfile.ZipFile(archive_path) as archive:
        if set(archive.namelist())!={e['replay_entry'] for e in manifest['episodes']}:
            raise ValueError('Collection replay games differ from the manifest')
        for episode in manifest['episodes']:
            rows=[];teacher_rows=[];life=0;prefix_count=0;teacher_started=False;status=None
            with BenchmarkEngine() as engine:
                state=engine.request('reset',options={'seed':episode['seed'],'level':episode['level'],'action_version':2})
                records=[json.loads(line) for line in archive.read(episode['replay_entry']).splitlines()]
                for index,record in enumerate(records):
                    if fingerprint(state)!=record['state_sha256']:raise ValueError('Collection native state differs')
                    move=record['choice'];risks=engine.request('risks')
                    prefix=record['controller']=='perturbation_prefix'
                    if prefix:
                        if teacher_started or life or episode['collection_mode']!='post_respawn':
                            raise ValueError('Invalid perturbation/teacher boundary')
                        if move!=prefix_move(state,risks):raise ValueError('Perturbation prefix differs from its fixed policy')
                        prefix_count+=1
                    elif record['controller']=='qualified_teacher':
                        if episode['collection_mode']=='post_respawn' and life<1:
                            raise ValueError('Recovery teacher began before a native respawn')
                        teacher_started=True
                    else:raise ValueError('Unknown collection controller')
                    response={'answers':{'move':{'choice':move,'probabilities':{
                        d:float(d==move) for d in state['legal_moves']}}}}
                    step=engine.step(move);row=diagnostic_record(state,response,step,risks,life)
                    if {k:v for k,v in row.items() if k!='http_ms'}!=record['diagnostics']:
                        raise ValueError('Collection native transition differs')
                    rows.append(row)
                    if not prefix:teacher_rows.append({**row,'life_index':life-episode['teacher_start_life']})
                    state=step['state']
                    if step['outcome']=='level_cleared':status='level_cleared'
                    elif step['outcome']=='life_lost':
                        continuation=engine.request('continue')
                        if continuation['status']=='game_over':status='game_over'
                        else:state=continuation['state'];life+=1
                    if status and index!=len(records)-1:raise ValueError('Collection continued past a native terminal state')
            if prefix_count!=episode['prefix_decisions'] or prefix_count>COLLECTION['prefix_max_decisions']:
                raise ValueError('Collection prefix count differs')
            for measured,expected in ((summarize(teacher_rows,status or episode['metrics']['outcome']),episode['metrics']),
                                      (summarize(rows,status or episode['full_game_metrics']['outcome']),episode['full_game_metrics'])):
                if fingerprint({k:v for k,v in measured.items() if k!='mean_http_ms'})!=fingerprint({k:v for k,v in expected.items() if k!='mean_http_ms'}):
                    raise ValueError('Collection metrics differ from native replay')
            failures=episode_quality_failures(episode['metrics'])
            if failures:raise ValueError('Collection teacher failed: '+'; '.join(failures))
    if sum(e['metrics']['life_losses'] for e in manifest['episodes'])>VALIDATION['gates']['maximum_life_losses_total']:
        raise ValueError('Too many teacher-controlled collection life losses')
    return True


def validate_partitions(manifest, contents, replay_path):
    """Bind every input/label to a teacher-controlled step in a verified full game."""
    lookup={};groups={};seen=set()
    with zipfile.ZipFile(replay_path) as archive:
        for episode in manifest['episodes']:
            group=f"level-{episode['level']}-seed-{episode['seed']}";groups[group]=episode['split']
            for line in archive.read(episode['replay_entry']).splitlines():
                record=json.loads(line)
                if record['controller']=='qualified_teacher':lookup[group,record['state_sha256']]=record
    for split,count in manifest['counts'].items():
        records=[json.loads(line) for line in contents[f'{PREFIX}-{split}.jsonl'].splitlines()]
        if len(records)!=count:raise ValueError('Dataset partition count mismatch')
        coverage=manifest['coverage'][split]
        coverage.update(power_expires_within_60_frames=0,labelled_ghost_eating_action=0,labelled_power_pellet_action=0)
        used=set()
        for r in records:
            digest=fingerprint(r['state']);group=r['_meta']['group_id'];used.add(group)
            if digest in seen or groups.get(group)!=split:raise ValueError('Dataset input/episode split overlap')
            seen.add(digest);native=lookup.get((group,digest))
            if native is None:raise ValueError('Label input is absent from the teacher-controlled native replay')
            q=r['questions']['move'];label=q['label']
            if label!=native['choice'] or label not in r['state']['legal_moves']:
                raise ValueError('Dataset label differs from the native teacher action')
            if {k:v for k,v in q.items() if k not in ('label','src')}!=body(r['state'])['questions']['move']:
                raise ValueError('Training/inference question formatting differs')
            if set(r)!={'state','questions','_meta'} or r['_meta']['qualification_sha256']!=manifest['qualification_sha256']:
                raise ValueError('Dataset request/provenance differs')
            coverage['power_expires_within_60_frames']+=r['state']['frightened'] and r['state']['timing']['power']['remaining_frames']<=60
            coverage['labelled_ghost_eating_action']+=bool(native['diagnostics']['events']['ghosts_eaten'])
            coverage['labelled_power_pellet_action']+=bool(native['diagnostics']['events']['power_pellets'])
        if used!=set(manifest['groups'][split]):raise ValueError('Declared dataset groups differ')
    return True


def generate(directory, qualification_path, counts=COUNTS, workers=4):
    receipt = require_qualified_teacher(qualification_path)  # Before writing any labels.
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    sources = source_hashes()
    reserved = set(SPEC['seeds']+VALIDATION['development_seeds']+VALIDATION['qualification_seeds']+
                   VALIDATION.get('retired_qualification_seeds', []))
    manifest = {'dataset':PREFIX, 'counts':dict(counts), 'recipe':RECIPE, 'groups':{}, 'coverage':{},
                'labels':{}, 'files':{}, 'episodes':[], 'qualification_sha256':hashlib.sha256(Path(qualification_path).read_bytes()).hexdigest(),
                'source_sha256':sources, 'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'expected_training_requests':counts['train']+RECIPE['replay'],
                'expected_optimizer_steps':(counts['train']+RECIPE['replay']+7)//8,
                'collection':COLLECTION,
                'split_policy':'Disjoint episode seeds and exact input hashes. Benchmark, teacher development and qualification seeds excluded. Same classic maze, levels 1,2,3,5; no unseen-layout claim.',
                'teacher':{'algorithm':receipt['algorithm'],'options':receipt['options'],
                           'optimality':'Empirically qualified finite-horizon policy portfolio; not globally optimal or universally safe.'}}
    seen, episode_number, partition_bytes, collection_episodes = set(), 0, {}, []
    with tempfile.TemporaryDirectory(prefix='pacman-v2-data-') as scratch, ProcessPoolExecutor(max_workers=workers) as pool:
        for split,count in counts.items():
            selected=[]
            while len(selected)<count:
                jobs=[]
                # Worker count changes parallelism, never level/mode coverage.
                for _ in VALIDATION['levels']:
                    episode_number+=1; seed=200003+episode_number*1009; level=(1,2,3,5)[(episode_number-1)%4]
                    if seed in reserved:raise ValueError('Collection seed overlaps reserved evaluation')
                    mode=COLLECTION['modes'][((episode_number-1)%4+(episode_number-1)//4)%2]
                    jobs.append((receipt['algorithm'],seed,level,scratch,receipt['options'],mode))
                batch_records=[]
                for episode in pool.map(collection_job,*zip(*jobs)):
                    m=episode['metrics']; failures=episode_quality_failures(m)
                    if failures:
                        raise RuntimeError(f"Teacher failed a collection game; no dataset authorized: level {episode['level']} seed {episode['seed']}: {'; '.join(failures)}")
                    collection_episodes.append(episode)
                    replay_entry=f"level-{episode['level']}-seed-{episode['seed']}.jsonl"
                    manifest['episodes'].append({'split':split,'seed':episode['seed'],'level':episode['level'],
                        'collection_mode':episode['collection_mode'],'prefix_decisions':episode['prefix_decisions'],
                        'teacher_start_life':episode['teacher_start_life'],'metrics':m,
                        'full_game_metrics':episode['full_game_metrics'],'replay_entry':replay_entry})
                    records=[json.loads(line) for line in Path(episode['trace']).read_text().splitlines()]
                    records=[record for record in records if record.get('controller','qualified_teacher')=='qualified_teacher']
                    sampled=[record for index,record in enumerate(records)
                        if index%4==0 or any(strata(record['request']['state'])[k] for k in
                            ('danger_within_3_tiles','frightened','late_maze_30_or_fewer','power_pellet_adjacent'))]
                    random.Random(episode['seed']).shuffle(sampled)
                    batch_records.append((episode,sampled))
                # Interleave full episodes so a split's final batch still
                # covers each level rather than taking only its first game.
                for batch in zip_longest(*(records for _,records in batch_records)):
                    for (episode,_),record in zip(batch_records,batch):
                        if record is None:continue
                        request=record['request'];state=request['state'];digest=fingerprint(state)
                        if digest in seen or len(selected)>=count:continue
                        seen.add(digest);request.pop('model',None)
                        move=record['response']['answers']['move']['choice']
                        request['questions']['move'].update(label=move,src='pacman_native_qualified_teacher')
                        request['_meta']={'id':f'{split}-board-{len(selected):04d}',
                            'group_id':f"level-{episode['level']}-seed-{episode['seed']}",
                            'source':'native_pacman_qualified_teacher','variant':'clean',
                            'label_source':'qualified native rollout MPC','teacher':record['response']['teacher'],
                            'collection_mode':episode['collection_mode'],
                            'episode_seed':episode['seed'],'episode_level':episode['level'],'decision':state['turn'],
                            'qualification_sha256':manifest['qualification_sha256']}
                        candidates=record['response']['teacher']['candidates']
                        request['_meta']['equally_ranked_actions']=[c['action'] for c in candidates
                            if c['rank']==candidates[0]['rank']] if candidates else [move]
                        request['_meta']['tie_policy']='Deterministic native legal-order tie break; equally ranked alternatives are retained for review. One hard choice target follows the Kev recipe; no unique-optimum claim.'
                        selected.append(request)
                print(f'{split}: {len(selected)}/{count} from complete native games',flush=True)
            coverage=Counter(); labels=Counter()
            for r in selected:
                for k,v in strata(r['state']).items():coverage[k]+=v
                coverage[f"level_{r['state']['level']}"]+=1
                coverage[r['_meta']['collection_mode']]+=1
                coverage['four_ghosts_outside']+=all(g['mode']=='outside' for g in r['state']['ghosts'])
                coverage['tied_teacher_rank']+=len(r['_meta']['equally_ranked_actions'])>1
                labels[r['questions']['move']['label']]+=1
            manifest['coverage'][split]=dict(coverage); manifest['labels'][split]=dict(labels)
            manifest['groups'][split]=sorted({r['_meta']['group_id'] for r in selected})
            content=''.join(json.dumps(r,separators=(',', ':'))+'\n' for r in selected).encode()
            name=f'{PREFIX}-{split}.jsonl';partition_bytes[name]=content
            manifest['files'][name]=hashlib.sha256(content).hexdigest()
        if (sources!=source_hashes() or manifest['generator_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
            raise RuntimeError('Teacher/generator source changed during collection; reject labels')
        if sum(e['metrics']['life_losses'] for e in manifest['episodes'])>VALIDATION['gates']['maximum_life_losses_total']:
            raise RuntimeError('Too many collection life losses; reject labels')
        manifest['replay_bundle']=f'{PREFIX}-replays.zip'
        with zipfile.ZipFile(directory/manifest['replay_bundle'],'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for episode in collection_episodes:
                records=[json.loads(line) for line in Path(episode['trace']).read_text().splitlines()]
                compact=[{'state_sha256':fingerprint(r['request']['state']),
                          'choice':r['response']['answers']['move']['choice'],
                          'controller':r.get('controller','qualified_teacher'),
                          'diagnostics':{k:v for k,v in r['diagnostics'].items() if k!='http_ms'}} for r in records]
                name=f"level-{episode['level']}-seed-{episode['seed']}.jsonl"
                archive.writestr(name,''.join(json.dumps(r,separators=(',', ':'))+'\n' for r in compact))
        manifest['replay_sha256']=hashlib.sha256((directory/manifest['replay_bundle']).read_bytes()).hexdigest()
        verify_collection(manifest,directory/manifest['replay_bundle'])
        validate_partitions(manifest,partition_bytes,directory/manifest['replay_bundle'])
        manifest['native_collection_replays_verified']=True
    # No partitions are published until every teacher continuation and its full
    # native replay passes, including the held-out development collection.
    for name,content in partition_bytes.items():(directory/name).write_bytes(content)
    (directory/f'{PREFIX}-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(directory/f'{PREFIX}.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in manifest['files']:
            info=zipfile.ZipInfo(name,date_time=(2026,10,5,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,(directory/name).read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    return manifest


def prepare_dataset(directory=ROOT/'data'):
    directory=Path(directory)
    require_qualified_teacher(ROOT/'evaluation/teacher-qualification.json')
    manifest=json.loads((directory/f'{PREFIX}-manifest.json').read_text())
    if manifest['qualification_sha256']!=hashlib.sha256((ROOT/'evaluation/teacher-qualification.json').read_bytes()).hexdigest():
        raise ValueError('Dataset uses another teacher qualification receipt')
    if manifest['source_sha256']!=source_hashes():raise ValueError('Dataset teacher implementation changed')
    if manifest['generator_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Dataset generator changed; regenerate the demonstrations')
    if manifest['counts']!=COUNTS or manifest['recipe']!=RECIPE:
        raise ValueError('Dataset count/recipe differs from the declared task')
    if manifest['collection']!=COLLECTION:raise ValueError('Dataset collection protocol differs')
    if manifest['replay_bundle']!=f'{PREFIX}-replays.zip':raise ValueError('Unexpected collection replay path')
    verify_collection(manifest,directory/manifest['replay_bundle'])
    if set(manifest['files'])!={f'{PREFIX}-{split}.jsonl' for split in COUNTS}:
        raise ValueError('Unexpected declared dataset files')
    with zipfile.ZipFile(directory/f'{PREFIX}.zip') as archive:
        if set(archive.namelist())!=set(manifest['files']):raise ValueError('Unexpected dataset archive contents')
        contents={name:archive.read(name) for name in manifest['files']}
        validated=json.loads(json.dumps(manifest))
        validate_partitions(validated,contents,directory/manifest['replay_bundle'])
        if validated['coverage']!=manifest['coverage']:raise ValueError('Dataset event coverage differs from replay')
        for name,digest in manifest['files'].items():
            content=contents[name]
            if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('Dataset checksum mismatch')
            target=directory/name
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:target.write_bytes(content)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();generate(a.out,a.qualification,workers=a.workers)
