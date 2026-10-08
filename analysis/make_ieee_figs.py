import csv, pathlib, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parent.parent
rows=list(csv.DictReader(open(ROOT/'outputs'/'all_experiments_summary.csv')))
OUT=str(ROOT/'paper'/'figures')+'/'
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
 'font.family':'serif','axes.linewidth':0.6,'pdf.fonttype':42})
S=[('ddp','DDP','#0072B2'),('fsdp_zero2','FSDP-Z2','#E69F00'),('fsdp_zero3','FSDP-Z3','#009E73')]
M=['qwen3-0.6b','qwen3-1.7b','qwen3-4b','qwen3-8b','qwen3-14b','qwen3-32b']
ML=['0.6B','1.7B','4B','8B','14B','32B']
def g(m,s,ws='8',ga='1'):
    c=[r for r in rows if r['model']==m and r['strategy']==s and r['world_size']==ws and r['gradient_accumulation_steps']==ga and r['status']=='ok']
    return c[0] if c else None
f=lambda r,k: float(r[k])
def sv(fig,name):
    fig.savefig(OUT+name+'.pdf',bbox_inches='tight'); fig.savefig(OUT+name+'.png',dpi=300,bbox_inches='tight')
# Fig A: decomposition 4B / 8 GPUs
comps=[('fwd_time_ms_mean','Forward','#4C78A8'),('bwd_compute_ms_calibrated','Bwd compute','#F58518'),
 ('recompute_ms_calibrated','Recompute','#EECA3B'),('bwd_exposed_comm_ms','Exposed comm','#E45756'),
 ('opt_time_ms_mean','Optimizer','#54A24B'),('straggler_wait_ms','Straggler','#B279A2')]
fig,ax=plt.subplots(figsize=(3.45,2.2))
for i,(s,l,c) in enumerate(S):
    r=g('qwen3-4b',s); left=0
    for k,n,col in comps:
        v=f(r,k); ax.barh(i,v,left=left,color=col,edgecolor='white',lw=0.4,label=n if i==0 else None); left+=v
    ax.text(left+8,i,f"{f(r,'iter_time_ms_mean'):.0f} ms",va='center',fontsize=7)
ax.set_yticks(range(3)); ax.set_yticklabels([x[1] for x in S]); ax.invert_yaxis()
ax.set_xlim(0,1250); ax.set_xlabel('Attributed time per iteration (ms)')
ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(0.45,1.32),frameon=False,columnspacing=0.8,handlelength=1)
fig.tight_layout(); sv(fig,'ieee_decomp_4b_ws8'); plt.close()
# Fig B: mechanism, 3 panels
fig,axs=plt.subplots(1,3,figsize=(7.16,2.0))
for s,l,c in S:
    xs=[];ov=[];be=[];hr=[]
    for i,m in enumerate(M):
        r=g(m,s)
        if r: xs.append(i);ov.append(f(r,'overlap_percentage_trace'));be.append(f(r,'bandwidth_efficiency'));hr.append(f(r,'headroom_to_80gb_gb'))
    for a,y in zip(axs,(ov,be,hr)): a.plot(xs,y,'-o',color=c,ms=3,lw=1,label=l)
for a,t in zip(axs,('(a) Comm--compute overlap (\\%)','(b) Bus-bandwidth efficiency (\\%)','(c) Memory headroom (GB)')):
    a.set_xticks(range(6)); a.set_xticklabels(ML); a.set_xlabel('Qwen3 model size'); a.grid(alpha=.25,lw=.4)
    a.set_title(t.replace('\\%','%').replace('--','–'),fontsize=8)
axs[0].legend(loc='lower right',frameon=False)
fig.tight_layout(); sv(fig,'ieee_mechanism_ws8'); plt.close()
# Fig C: GA
fig,ax=plt.subplots(figsize=(3.45,2.1))
for s,l,c in S:
    y=[f(g('qwen3-4b',s,'8',str(ga)),'throughput_tokens_per_sec_mean')/1000 for ga in (1,2,4,8)]
    ax.plot(range(4),y,'-o',color=c,ms=3.5,lw=1.1,label=l)
    ax.annotate(f"{(y[-1]/y[0]-1)*100:+.1f}%",(3,y[-1]),xytext=(6,0),textcoords='offset points',fontsize=7,color=c,va='center')
ax.set_xticks(range(4)); ax.set_xticklabels(['1','2','4','8']); ax.set_xlim(-0.2,3.9)
ax.set_xlabel('Gradient-accumulation depth'); ax.set_ylabel('Throughput (ktokens/s)'); ax.grid(alpha=.25,lw=.4)
ax.legend(frameon=False,loc='upper left')
fig.tight_layout(); sv(fig,'ieee_gradaccum_4b_ws8'); plt.close()
print('ok')
# Fig D: percent decomposition across models, 8 GPUs
fig,ax=plt.subplots(figsize=(7.16,2.7)); x=0; ticks=[];labs=[]
for i,m in enumerate(M):
    start=x
    for j,(s_,l,c) in enumerate(S):
        r=g(m,s_)
        if r:
            tot=f(r,'iter_time_ms_mean'); bot=0
            for k,n,col in comps:
                v=100*f(r,k)/tot; ax.bar(x,v,bottom=bot,color=col,edgecolor='white',lw=0.3,width=0.85,label=n if (i==0 and j==0) else None); bot+=v
        else:
            ax.text(x,3,'OOM',rotation=90,ha='center',va='bottom',fontsize=6.5,color='#555555')
        ticks.append(x); labs.append(['DDP','Z2','Z3'][j]); x+=1
    ax.annotate(ML[i],xy=((start+x-1)/2,0),xycoords=('data','axes fraction'),xytext=(0,-24),textcoords='offset points',ha='center',va='top',fontsize=8.5,fontweight='bold')
    ax.plot([start-0.4,x-0.6],[-0.17,-0.17],transform=ax.get_xaxis_transform(),color='k',lw=0.6,clip_on=False)
    x+=1
ax.set_xticks(ticks); ax.set_xticklabels(labs,fontsize=6.5,rotation=0); ax.set_ylim(0,100); ax.set_xlim(-0.8,x-1.2)
ax.set_ylabel('Share of iteration (%)')
ax.legend(ncol=6,loc='lower center',bbox_to_anchor=(0.5,1.01),frameon=False,columnspacing=1.2,handlelength=1.2)
fig.subplots_adjust(bottom=0.27,top=0.88,left=0.075,right=0.995)
sv(fig,'ieee_decomp_pct_ws8'); plt.close()
print('ok')
