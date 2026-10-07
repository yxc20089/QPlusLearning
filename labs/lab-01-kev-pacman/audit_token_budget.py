"""CPU token audit using the pinned Kev renderer and a local base-tokenizer file."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def audit(kev_source, tokenizer_path, paths, state_limit=4096):
    from tokenizers import Tokenizer
    api=Path(kev_source)/'kev/api.py'
    tokenizer_path=Path(tokenizer_path)
    if (hashlib.sha256(api.read_bytes()).hexdigest()!='a6f53af5354f8ea17915abc78360fdf291a5108ce39a53a2d9135d48c603ae58' or
        hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()!='fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927'):
        raise ValueError('Use the pinned Kev renderer and Qwen3.5-4B tokenizer for this audit')
    tree=ast.parse(api.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('render','option_text')]
    scope={'JSONContent':object}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(api),'exec'),scope)
    render,option_text=scope['render'],scope['option_text']
    tok=Tokenizer.from_file(str(tokenizer_path))
    lengths=[];inputs={}
    for filename in paths:
        path=Path(filename);content=path.read_bytes();inputs[path.name]=hashlib.sha256(content).hexdigest()
        for line in content.splitlines():
            raw=json.loads(line);request=raw.get('request',raw)
            state_tokens=len(tok.encode(render(request['state']),add_special_tokens=False).ids)+1
            question=request['questions']['move']
            branch_tokens=2+len(tok.encode(render(question['instructions']),add_special_tokens=False).ids)
            branch_tokens+=sum(2+len(tok.encode(option_text(k,v),add_special_tokens=False).ids)
                               for k,v in question['criteria'].items())
            lengths.append((state_tokens,state_tokens+branch_tokens))
    if not lengths:raise ValueError('No requests to audit')
    return {'records':len(lengths),'state_limit':state_limit,
            'max_state_tokens':max(s for s,_ in lengths),'max_packed_tokens':max(p for _,p in lengths),
            'over_state_limit':sum(s>state_limit for s,_ in lengths),'files_sha256':inputs,
            'render_source_sha256':hashlib.sha256(api.read_bytes()).hexdigest(),
            'tokenizer_sha256':hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            'method':'Exact Kev render/option_text and local base tokenizer, with explicit state/question/option marker allowances. Labels/search metadata are excluded. CPU length audit; actual CUDA training must also report zero truncation.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kev-source',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('paths',type=Path,nargs='+')
    args=parser.parse_args();report=audit(args.kev_source,args.tokenizer,args.paths)
    report.update(kev_revision='84847f0a883d900f7de5b7a57eaa341ca7f9a6b4',
                  tokenizer_revision='1001bb4d826a52d1f399e183466143f4da7b741b')
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
