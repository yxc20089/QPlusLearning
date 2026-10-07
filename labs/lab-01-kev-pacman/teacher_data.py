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
import zipfile

from gameplay_benchmark import ROOT, SPEC, strata
from planner_data import RECIPE
from teacher_validation import VALIDATION, episode_job, fingerprint, require_qualified_teacher, source_hashes

PREFIX = 'pacman-native-v2'
COUNTS = {'train':4096, 'development':256}


def generate(directory, qualification_path, counts=COUNTS, workers=4):
    receipt = require_qualified_teacher(qualification_path)  # Before writing any labels.
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    sources = source_hashes()
    reserved = set(SPEC['seeds']+VALIDATION['development_seeds']+VALIDATION['qualification_seeds'])
    manifest = {'dataset':PREFIX, 'counts':dict(counts), 'recipe':RECIPE, 'groups':{}, 'coverage':{},
                'labels':{}, 'files':{}, 'episodes':[], 'qualification_sha256':hashlib.sha256(Path(qualification_path).read_bytes()).hexdigest(),
                'source_sha256':sources, 'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'expected_training_requests':counts['train']+RECIPE['replay'],
                'expected_optimizer_steps':(counts['train']+RECIPE['replay']+7)//8,
                'collection':'Complete qualified-teacher native games from initial boards. Sample every fourth decision plus danger, power and late-maze decisions. Seeded shuffle within each completed episode, then interleave levels, so a split ending mid-batch is not biased to game openings. No random board states. All episodes finish and are checked before sampling; do not filter away failed games.',
                'split_policy':'Disjoint episode seeds and exact input hashes. Benchmark, teacher development and qualification seeds excluded. Same classic maze, levels 1,2,3,5; no unseen-layout claim.',
                'teacher':{'algorithm':receipt['algorithm'],'options':receipt['options'],
                           'optimality':'Empirically qualified finite-horizon policy portfolio; not globally optimal or universally safe.'}}
    seen, episode_number = set(), 0
    with tempfile.TemporaryDirectory(prefix='pacman-v2-data-') as scratch, ProcessPoolExecutor(max_workers=workers) as pool:
        for split,count in counts.items():
            selected=[]
            while len(selected)<count:
                jobs=[]
                for _ in range(workers):
                    episode_number+=1; seed=200003+episode_number*1009; level=(1,2,3,5)[(episode_number-1)%4]
                    if seed in reserved:raise ValueError('Collection seed overlaps reserved evaluation')
                    jobs.append((receipt['algorithm'],seed,level,scratch,receipt['options']))
                batch_records=[]
                for episode in pool.map(episode_job,*zip(*jobs)):
                    m=episode['metrics'];gates=VALIDATION['gates']
                    if (not m['level_cleared'] or m['pellets_remaining'] or m['avoidable_immediate_deaths'] or
                        m['loop_decisions'] or m['longest_no_pellet_decisions']>gates['maximum_no_pellet_decisions_per_game'] or
                        m['nonterminal_multi_tile_actions'] or m['nonterminal_stationary_actions']):
                        raise RuntimeError(f"Teacher failed a collection game; no dataset authorized: level {episode['level']} seed {episode['seed']}")
                    manifest['episodes'].append({'split':split,'seed':episode['seed'],'level':episode['level'],'metrics':m})
                    records=[json.loads(line) for line in Path(episode['trace']).read_text().splitlines()]
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
                            'episode_seed':episode['seed'],'episode_level':episode['level'],'decision':state['turn'],
                            'qualification_sha256':manifest['qualification_sha256']}
                        selected.append(request)
                print(f'{split}: {len(selected)}/{count} from complete native games',flush=True)
            coverage=Counter(); labels=Counter()
            for r in selected:
                for k,v in strata(r['state']).items():coverage[k]+=v
                coverage[f"level_{r['state']['level']}"]+=1
                coverage['four_ghosts_outside']+=all(g['mode']=='outside' for g in r['state']['ghosts'])
                labels[r['questions']['move']['label']]+=1
            manifest['coverage'][split]=dict(coverage); manifest['labels'][split]=dict(labels)
            manifest['groups'][split]=sorted({r['_meta']['group_id'] for r in selected})
            content=''.join(json.dumps(r,separators=(',', ':'))+'\n' for r in selected).encode()
            name=f'{PREFIX}-{split}.jsonl';(directory/name).write_bytes(content)
            manifest['files'][name]=hashlib.sha256(content).hexdigest()
    if sources!=source_hashes():raise RuntimeError('Teacher source changed during collection; reject labels')
    if sum(e['metrics']['life_losses'] for e in manifest['episodes'])>VALIDATION['gates']['maximum_life_losses_total']:
        raise RuntimeError('Too many collection life losses; reject labels')
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
    if set(manifest['files'])!={f'{PREFIX}-{split}.jsonl' for split in COUNTS}:
        raise ValueError('Unexpected declared dataset files')
    with zipfile.ZipFile(directory/f'{PREFIX}.zip') as archive:
        if set(archive.namelist())!=set(manifest['files']):raise ValueError('Unexpected dataset archive contents')
        for name,digest in manifest['files'].items():
            content=archive.read(name)
            if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('Dataset checksum mismatch')
            target=directory/name
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:target.write_bytes(content)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();generate(a.out,a.qualification,workers=a.workers)
