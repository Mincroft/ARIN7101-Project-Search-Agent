"""Create separate high-resolution PNG figures for the verified W&B training run."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN=Path('runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main')
OUT=Path('figures/qwen35_9b_report')
C={'blue':'#0F4D92','blue2':'#3775BA','teal':'#42949E','violet':'#9A4D8E','green':'#2E8B57','orange':'#D97706','red':'#B64342','gray':'#767676','light':'#CFCECE','dark':'#272727'}
plt.rcParams.update({'font.family':['DejaVu Sans','sans-serif'],'font.size':13,'axes.titlesize':17,'axes.labelsize':14,'axes.linewidth':1.6,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})

def load(prefix, metric_prefix):
 rows=[]
 for p in sorted(RUN.glob(prefix+'*/metrics.json')):
  d=json.loads(p.read_text()); d['update']=int(p.parent.name.rsplit('-',1)[1]); rows.append(d)
 return sorted(rows,key=lambda x:x['update'])
train=load('rollouts-','train_raw/')
evalr=load('eval-','eval/')

def arr(rows,key): return np.array([r[key] for r in rows],float)
def ema(values, span=5):
    """Recursive EMA, alpha=2/(span+1), initialized from the first observation."""
    values=np.asarray(values,float)
    if values.ndim != 1 or len(values) == 0 or span < 1:
        raise ValueError('EMA requires a nonempty 1D sequence and positive span')
    alpha=2.0/(span+1.0)
    out=np.empty_like(values); out[0]=values[0]
    for i in range(1,len(values)): out[i]=alpha*values[i]+(1-alpha)*out[i-1]
    return out
def setup(title,ylabel,xlabel='Optimization update',ylim=None):
 fig,ax=plt.subplots(figsize=(11.5,6.8)); fig.subplots_adjust(left=.11,right=.97,bottom=.14,top=.84)
 ax.set_title(title,loc='left',fontweight='bold',pad=14); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_xlim(-1,81); ax.set_xticks(np.arange(0,81,10)); ax.grid(axis='y',color=C['light'],alpha=.55,lw=.8)
 if ylim: ax.set_ylim(*ylim)
 return fig,ax
def save(fig,name,desc,extra=None):
 OUT.mkdir(parents=True,exist_ok=True)
 fig.savefig(OUT/(name+'.png'),format='png',dpi=1200,bbox_inches='tight',pad_inches=.12)
 plt.close(fig)
# 1: evaluation quality
x=arr(evalr,'update'); f1=arr(evalr,'eval/f1'); f4=arr(evalr,'eval/f_beta'); s1=arr(evalr,'eval/f1_draw_std'); s4=arr(evalr,'eval/f_beta_draw_std')
fig,ax=setup('Development-set quality across training','Score',ylim=(.08,.68)); ax.plot(x,f1,'o-',lw=2.8,ms=5,color=C['blue'],label='F1'); ax.fill_between(x,f1-s1,f1+s1,color=C['blue'],alpha=.13); ax.plot(x,f4,'s-',lw=2.8,ms=4.5,color=C['teal'],label='F4'); ax.fill_between(x,f4-s4,f4+s4,color=C['teal'],alpha=.12)
i=int(np.argmax(f1)); j=int(np.argmax(f4)); ax.scatter([x[i]],[f1[i]],s=115,color=C['blue'],edgecolor='white',lw=1.5,zorder=5); ax.scatter([x[j]],[f4[j]],s=115,color=C['teal'],edgecolor='white',lw=1.5,zorder=5); ax.annotate(f'best F1: update {int(x[i])} ({f1[i]:.3f})',(x[i],f1[i]),xytext=(-55,25),textcoords='offset points',arrowprops={'arrowstyle':'-','color':C['blue']},color=C['blue']); ax.annotate(f'best F4: update {int(x[j])} ({f4[j]:.3f})',(x[j],f4[j]),xytext=(40,-30),textcoords='offset points',arrowprops={'arrowstyle':'-','color':C['teal']},color=C['teal']); ax.legend(ncol=2,loc='upper left'); save(fig,'01_development_quality','F1 and F4 with two-draw standard deviation bands',{'best_f1_update':int(x[i]),'best_f4_update':int(x[j])})
# 2: reward and train quality
x=arr(train,'update'); reward=arr(train,'train_raw/reward'); tf1=arr(train,'train_raw/f1'); tf4=arr(train,'train_raw/f_beta');
fig,axes=plt.subplots(2,1,figsize=(11.5,8.5),sharex=True); fig.subplots_adjust(left=.11,right=.97,bottom=.11,top=.90,hspace=.28); fig.suptitle('Training dynamics',x=.11,ha='left',fontweight='bold',fontsize=18)
axes[0].scatter(x,reward,s=22,color=C['violet'],alpha=.25,label='Per-update reward'); axes[0].plot(x,ema(reward,5),color=C['violet'],lw=3.0,label='EMA (span=5)'); axes[0].set_ylabel('Mean reward'); axes[0].legend(loc='upper left'); axes[0].grid(axis='y',color=C['light'],alpha=.55)
axes[1].plot(x,tf1,'o',ms=3.0,color=C['blue'],alpha=.22); axes[1].plot(x,ema(tf1,5),lw=2.8,color=C['blue'],label='Training F1 (EMA, span=5)'); axes[1].plot(x,tf4,'s',ms=3.0,color=C['teal'],alpha=.22); axes[1].plot(x,ema(tf4,5),lw=2.8,color=C['teal'],label='Training F4 (EMA, span=5)'); axes[1].set_xlabel('Optimization update'); axes[1].set_ylabel('Score'); axes[1].legend(loc='upper left',ncol=2); axes[1].grid(axis='y',color=C['light'],alpha=.55); axes[1].set_xlim(-1,81); axes[1].set_xticks(np.arange(0,81,10)); save(fig,'02_training_dynamics','Training reward and per-rollout quality metrics')
# 3: retrieval
x=arr(train,'update'); fig,axes=plt.subplots(1,2,figsize=(12,5.4)); fig.subplots_adjust(left=.08,right=.97,bottom=.17,top=.84,wspace=.28); fig.suptitle('Retrieval behavior',x=.08,ha='left',fontweight='bold',fontsize=18)
for key,color,label,marker in [('train_raw/candidate_recall',C['blue'],'Candidate recall','o'),('train_raw/precision',C['orange'],'Precision','s'),('train_raw/recall',C['teal'],'Recall','^')]:
    values=arr(train,key); axes[0].plot(x,values,marker=marker,ms=2.8,ls='none',color=color,alpha=.20); axes[0].plot(x,ema(values,5),lw=2.8,color=color,label=label+' (EMA, span=5)'); axes[0].set_ylabel('Retrieval score'); axes[0].set_title('Evidence quality',loc='left'); axes[0].legend(fontsize=11); axes[0].grid(axis='y',color=C['light'],alpha=.55)
for key,color,label,marker in [('train_raw/curated_documents',C['green'],'Curated documents','o'),('train_raw/curate_calls',C['violet'],'Curate calls','s')]:
    values=arr(train,key); axes[1].plot(x,values,marker=marker,ms=2.8,ls='none',color=color,alpha=.20); axes[1].plot(x,ema(values,5),lw=2.8,color=color,label=label+' (EMA, span=5)'); axes[1].set_ylabel('Mean count'); axes[1].set_title('Curation effort',loc='left'); axes[1].legend(fontsize=11); axes[1].grid(axis='y',color=C['light'],alpha=.55)
for ax in axes: ax.set_xlabel('Optimization update'); ax.set_xlim(-1,81); ax.set_xticks(np.arange(0,81,10))
save(fig,'03_retrieval_behavior','Candidate recall, precision, recall, and curation effort')
# 4: agent behavior
fig,axes=plt.subplots(1,2,figsize=(12,5.4)); fig.subplots_adjust(left=.08,right=.97,bottom=.17,top=.84,wspace=.28); fig.suptitle('Agent interaction behavior',x=.08,ha='left',fontweight='bold',fontsize=18)
for key,color,label,marker in [('train_raw/turns',C['blue'],'Turns','o'),('train_raw/bm25_calls',C['teal'],'BM25 calls','s'),('train_raw/read_calls',C['orange'],'Read calls','^')]:
    values=arr(train,key); axes[0].plot(x,values,marker=marker,ms=2.8,ls='none',color=color,alpha=.20); axes[0].plot(x,ema(values,5),lw=2.8,color=color,label=label+' (EMA, span=5)'); axes[0].set_title('Interaction depth',loc='left'); axes[0].set_ylabel('Mean count'); axes[0].legend(fontsize=11); axes[0].grid(axis='y',color=C['light'],alpha=.55)
for key,color,label,marker in [('train_raw/finished',C['green'],'Finished','o'),('train_raw/context_overflow',C['red'],'Context overflow','s'),('train_raw/invalid_curations',C['violet'],'Invalid curation','^')]:
    values=arr(train,key); axes[1].plot(x,values,marker=marker,ms=2.8,ls='none',color=color,alpha=.20); axes[1].plot(x,ema(values,5),lw=2.8,color=color,label=label+' (EMA, span=5)'); axes[1].set_title('Termination and errors',loc='left'); axes[1].set_ylabel('Fraction'); axes[1].legend(fontsize=11); axes[1].grid(axis='y',color=C['light'],alpha=.55)
for ax in axes: ax.set_xlabel('Optimization update'); ax.set_xlim(-1,81); ax.set_xticks(np.arange(0,81,10))
save(fig,'04_agent_behavior','Interaction depth, termination, and curation error rates')
# 6: optimization
ck=load('checkpoint-','optim/'); x=np.array([r['update'] for r in ck]);
fig,axes=plt.subplots(1,2,figsize=(12,5.4)); fig.subplots_adjust(left=.08,right=.97,bottom=.17,top=.84,wspace=.28); fig.suptitle('Optimization stability',x=.08,ha='left',fontweight='bold',fontsize=18)
loss=arr(ck,'optim/loss'); axes[0].plot(x,loss,'o',ms=2.8,color=C['orange'],alpha=.20); axes[0].plot(x,ema(loss,5),lw=2.8,color=C['orange'],label='EMA (span=5)'); axes[0].legend(loc='upper right',fontsize=11); axes[0].axhline(0,color=C['light'],lw=1); axes[0].set_title('Policy loss',loc='left'); axes[0].set_ylabel('Loss'); axes[0].grid(axis='y',color=C['light'],alpha=.55)
grad=arr(ck,'optim/grad_norm'); axes[1].plot(x,grad,'o',ms=2.8,color=C['red'],alpha=.20); axes[1].plot(x,ema(grad,5),lw=2.8,color=C['red'],label='EMA (span=5)'); axes[1].legend(loc='upper right',fontsize=11); axes[1].set_title('Gradient norm',loc='left'); axes[1].set_ylabel('Norm'); axes[1].grid(axis='y',color=C['light'],alpha=.55)
for ax in axes: ax.set_xlabel('Optimization update'); ax.set_xlim(-1,81); ax.set_xticks(np.arange(0,81,10))
save(fig,'06_optimization_stability','Optimization loss and gradient norm across all checkpoints')
print('generated',len(list(OUT.glob('*.png'))),'PNG figures in',OUT)
