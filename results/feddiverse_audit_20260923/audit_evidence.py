"""Read frozen runs and check tensor algebra without training or editing checkpoints."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = 'e0fc5ed5200339ff72c1428bb08115edb57b375d'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8')

def scalars(text):
    result, parents = {}, []
    for line in text.splitlines():
        match = re.match(r'^( *)([\w]+):(?:\s+(.*))?$', line)
        if not match:
            continue
        indent, key, value = len(match[1]), match[2], match[3]
        while parents and parents[-1][0] >= indent:
            parents.pop()
        name = '.'.join([x[1] for x in parents] + [key])
        if value is None:
            parents.append((indent, key))
        else:
            result[name] = value
    return result

keys = ['seed','client_opt.learning_rate','client_opt.batch_size','client_opt.epochs',
        'client_opt.base_optimizer','client_opt.momentum','client_opt.biased_trainer_steps',
        'client_opt.left_right_trainer_steps','client_opt.generalized_cross_entropy_q',
        'client_opt.biased_optimizer','model_options.model_type','model_options.pretrained',
        'server_opt.optimizer','server_opt.learning_rate','server_opt.beta_1','server_opt.rounds',
        'server_opt.weight_clients','server_opt.selection_method','server_opt.client_info',
        'server_opt.paper_faithful_pretrain','server_opt.pretrain_rounds',
        'server_opt.dht_collection_rounds','server_opt.update_static_info_rounds',
        'dataset_options.split_mode','dataset_options.num_clients','dataset_options.input_size']
runs = []
for cfg in sorted((ROOT/'checkpoints/local_backup').glob('*/checkpoints/*/config.yaml')):
    c = scalars(cfg.read_text(encoding='utf-8'))
    info = json.loads((cfg.parent/'client_info.json').read_text())
    ids = sorted(map(int, info))
    assert ids == list(range(int(c['dataset_options.num_clients'])))
    n = np.array([[info[str(i)]['interaction_matrix_'+str(j)] for j in range(4)] for i in ids])
    w = np.loadtxt(cfg.parent/'client_weights.csv', delimiter=',', dtype=int)
    is_fd = c['server_opt.selection_method'] == 'triplets_stochasticmatrix'
    assert w.shape == (201 if is_fd else 200, len(ids))
    assert np.all((w == 0) | (w == 1))
    assert np.all(w[2 if is_fd else 0:].sum(1) == 9)
    assert c['client_opt.learning_rate']=='0.001' and c['client_opt.batch_size']=='28'
    assert c['client_opt.epochs']=='1' and c['client_opt.momentum']=='0.9'
    assert c['server_opt.optimizer']=='FedAvgM' and c['server_opt.beta_1']=='0.95'
    assert c['server_opt.weight_clients']=='same'
    runs.append(dict(config=str(cfg.relative_to(ROOT)), sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(),
                     values={k:c.get(k,'ABSENT') for k in keys}, pooled_counts=n.sum(0).tolist(),
                     client_sizes=n.sum(1).tolist(), selection_rows=len(w)))
assert len(runs)==24
paths=['src/corr.py','src/optimizers/matrix_inference.py','src/optimizers/weighting_strategy.py',
       'src/optimizers/dataloaders.py','src/optimizers/subpopbench.py','src/optimizers/optim_utils.py',
       'src/datasets/cmnist.py','src/datasets/spawrious.py','src/datasets/data_splits.py',
       'src/datasets/data_preparation.py','src/models/mobilenet.py','src/models/model_utils.py',
       'src/flower_client.py','src/flower_manager.py','src/flower_strategy.py','flower_train.py',
       'src/utils.py','conf/client_opt/all_defaults.yaml','conf/server_opt/all_defaults.yaml']
modules=[dict(path=p, differs=bool(git('diff',BASE,'HEAD','--',p).strip())) for p in paths]
(OUT/'config_and_source_audit.json').write_text(json.dumps(dict(upstream_commit=BASE,local_commit=git('rev-parse','HEAD').strip(),runs=runs,modules=modules),ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'official_to_local.diff').write_text(git('diff',BASE,'HEAD','--','src','flower_train.py','conf/client_opt/all_defaults.yaml','conf/server_opt/all_defaults.yaml'),encoding='utf-8')

# All reductions and products in the tensor contract are expressed with einsum.
E=np.einsum
rng=np.random.default_rng(20260923)
def ent(p, axis):
    return -E(axis+', '+axis+' -> '+axis[:-1], p, np.log(p+1e-9))

checks=[]
for K,C in [(24,2),(25,2),(25,4),(100,2)]:
    n=rng.integers(1,100,size=(K,C,2)).astype(float)
    p=n/E('kca->k',n)[:,None,None]
    py=E('kca->kc',p); pa=E('kca->ka',p)
    hy=-E('kc,kc->k',py,np.log(py+1e-9))
    ha=-E('ka,ka->k',pa,np.log(pa+1e-9))
    indep=E('kc,ka->kca',py,pa)
    mi=E('kca,kca->k',p,np.log(p/indep))
    dht=np.stack([2*mi/(hy+ha),1-ha/np.log(2),1-hy/np.log(C)],axis=0)
    assert dht.shape==(3,K)
    pooled=E('kca->ca',n); assert pooled.shape==(C,2)
    client_mean=E('dk->d',dht)/K; assert client_mean.shape==(3,)
    B,F=7,1280
    features=rng.normal(size=(B,F)); weights=rng.normal(size=(C,F))
    logits=E('bf,cf->bc',features,weights); assert logits.shape==(B,C)
    probs=np.exp(logits-logits.max(1,keepdims=True)); probs/=E('bc->b',probs)[:,None]
    y=np.eye(C)[rng.integers(C,size=B)]; a=np.eye(2)[rng.integers(2,size=B)]
    counted=E('bc,ba->ca',y,a); assert E('ca->',counted)==B
    pt=E('bc,bc->b',probs,y); gce=E('b->',(1-pt**0.3)/0.3)/B
    assert np.ndim(gce)==0
    correct=(probs.argmax(1)==y.argmax(1)).astype(float)
    hits=E('b,bc,ba->ca',correct,y,a); assert np.all(hits<=counted)
    theta=rng.normal(size=(9,17)); agg=E('k,kp->p',np.ones(9)/9,theta)
    np.testing.assert_allclose(agg,theta.mean(0))
    checks.append({'K':K,'C':C,'N_shape':list(n.shape),'DHT_shape':list(dht.shape),'passed':True})

eps=np.zeros((3,3,3))
for a,b,c in [(0,1,2),(1,2,0),(2,0,1)]:
    eps[a,b,c]=1;eps[a,c,b]=-1
u=rng.normal(size=3); v=rng.normal(size=3)
np.testing.assert_allclose(E('abc,b,c->a',eps,u,v),np.cross(u,v))

# Use the actual official functions (AST-only extraction avoids unrelated imports).
tree=ast.parse((ROOT/'src/optimizers/weighting_strategy.py').read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['select_3_clients','select_noreplacement']]
ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),'official_selector','exec'),ns)
np.random.seed(20260923)
for K,m in [(24,9),(25,9),(100,12)]:
    for _ in range(20):
        sel=ns['select_noreplacement'](rng.uniform(0.01,1,size=(3,K)),m)
        assert len(sel)==len(set(sel))==m and all(0<=i<K for i in sel)
checks.append({'cross_product':'einsum equals numpy.cross','selector_random_non_degenerate_cases':60,'passed':True})
# Boundary tests, not training. Code's pivot score is identically 1 after max normalization.
pivot_counts=[(90,10),(50,50)]
scores=[]
for counts in pivot_counts:
    w=np.array([sum(counts)/x for x in counts]);w/=w.max();scores.append(w.max())
assert scores==[1.,1.] and np.argmin(scores)==0
assert np.argmin([abs(a-b) for a,b in pivot_counts])==1
checks.append({'pivot_counterexample_counts':pivot_counts,'code_choice':0,'paper_choice':1,'passed':True})

(OUT/'tensor_validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps({'runs':len(runs),'unchanged_modules':[x['path'] for x in modules if not x['differs']], 'dimension_checks':checks},ensure_ascii=False,indent=2))

# Render the relevant original paper pages for visual inspection.
import pypdfium2 as pdfium
paper=next(Path('D:/zotero_file/storage/VZD6UMI6').glob('*.pdf'))
doc=pdfium.PdfDocument(str(paper))
for i in [3,4,5]:
    doc[i].render(scale=1.3).to_pil().save(OUT/f'paper_page_{i+1}.png')
