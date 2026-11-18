# Usage: python3 analysis.py  (run from a folder containing data/*.csv, the eight rater files)
# Reproduces the rater statistics in the paper (applicable criteria only, bootstrap CIs over raters).
import pandas as pd, numpy as np, glob
from scipy import stats
df=pd.concat([pd.read_csv(f) for f in sorted(glob.glob('data/*.csv'))])
app={'p1':['warm','conn','noman','read'],'p2':['hon','warm','conn','read'],'p3':['auto','noman','read'],'p4':['hon','auto','read'],'p5':['warm','auto','read']}
df['tot']=df.apply(lambda r:sum(r['score_'+c] for c in app[r.prompt_id]),axis=1)
df['mx']=df.prompt_id.map(lambda p:2*len(app[p]))
# ICC(2,1) on pct score: targets=20 responses, raters=8
df['pct']=df.tot/df.mx
M=df.pivot(index='response_id',columns='participant_id',values='pct').values
n,k=M.shape
gm=M.mean(); msr=k*((M.mean(1)-gm)**2).sum()/(n-1); msc=n*((M.mean(0)-gm)**2).sum()/(k-1)
sse=((M-M.mean(1,keepdims=True)-M.mean(0,keepdims=True)+gm)**2).sum(); mse=sse/((n-1)*(k-1))
icc21=(msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n); icc2k=(msr-mse)/(msr+(msc-mse)/n)
print('ICC21',round(icc21,2),'ICC2k',round(icc2k,2))
# item-level exact agreement across rater pairs
from itertools import combinations
cr=['hon','warm','conn','auto','noman','read']
ex=[];w1=[]
for r_,g in df.groupby('response_id'):
    for c in app[r_.split('-')[0]]:
        v=g['score_'+c].values
        for a,b in combinations(v,2): ex.append(a==b); w1.append(abs(a-b)<=1)
print('exact',round(np.mean(ex),2),'within1',round(np.mean(w1),2))
# participant-level overall pct
def pct(g): return g.tot.sum()/g.mx.sum()
pp=df.groupby(['participant_id','system']).apply(pct).unstack()
d=pp.Claude-pp.ChatGPT
print('diff mean',d.mean().round(4),'wilcoxon',stats.wilcoxon(d).pvalue, 't',stats.ttest_1samp(d,0).pvalue)
rng=np.random.default_rng(1)
def boot(vals):
    bs=[rng.choice(vals,len(vals)).mean() for _ in range(10000)]
    return np.percentile(bs,[2.5,97.5]).round(2)
# paired by participant, per prompt/criterion differences (Claude-ChatGPT)
for p,c in [('p4','auto'),('p5','auto'),('p1','conn'),('p1','read'),('p5','hon'),('p2','hon'),('p3','read'),('p4','hon')]:
    s=df[df.prompt_id==p].groupby(['participant_id','system'])['score_'+c].mean().unstack()
    dd=(s.Claude-s.ChatGPT).values
    print(p,c,dd.mean().round(2),boot(dd), (dd>0).sum(),(dd<0).sum())
# criterion-level over applicable prompts
for c in cr:
    sub=df[df.prompt_id.map(lambda p:c in app[p])]
    s=sub.groupby(['participant_id','system'])['score_'+c].mean().unstack()
    dd=(s.Claude-s.ChatGPT).values
    print('ALL',c,dd.mean().round(2),boot(dd),(dd>0).sum(),(dd<0).sum())
# per prompt pct
for p in app:
    s=df[df.prompt_id==p].groupby(['participant_id','system']).apply(pct).unstack()
    dd=((s.Claude-s.ChatGPT)*100).values
    print(p,'pct diff',dd.mean().round(1),boot(dd),(dd>0).sum(),(dd<0).sum())
# author's scores by prompt in pct
print('author', 33/34, 28/34)
# run 1 vs 2 within system
print(df.groupby(['prompt_id','system','run']).apply(lambda g:g.tot.mean()/g.mx.iloc[0]).unstack().round(2))
# order effects? response_pos
print(df.groupby('response_pos').pct.mean().round(3)); print(df.groupby('prompt_pos').pct.mean().round(3))
