import numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import glob
fp=[f for f in fm.findSystemFonts() if 'NotoSansCJK' in f or 'NotoSerifCJK' in f]
fp=sorted(fp,key=lambda f:'Sans' not in f)[0];fm.fontManager.addfont(fp);plt.rcParams['font.family']=fm.FontProperties(fname=fp).get_name()
plt.rcParams['axes.unicode_minus']=False
d=np.load('dig.npz');c=np.load('cmp.npz')
B,O,A,INK,MUT='#2a78d6','#eb6834','#1baf7a','#1a1a19','#6b6a63'
fig,ax=plt.subplots(2,2,figsize=(11,7.6),gridspec_kw=dict(height_ratios=[3,1.25],hspace=.12,wspace=.2),sharex='col')
info={'a':('Al 80 nm / 사파이어  (G = 90 MW/m²K, K = 29.3 W/mK)',7.9,(5,1800)),'b':('Au 100 nm / SiC  (G = 73.6 MW/m²K, K = 349.9 W/mK)',6.1,(4,1800))}
for j,k in enumerate('ab'):
    fit,dat=d['f'+k],d['d'+k];ttl,pk,xl=info[k];a=ax[0,j];r=ax[1,j]
    fit=fit[fit[:,0]>pk+1.5]
    ok=np.ones(len(dat),bool)
    for tk in (10,100,1000): ok&=np.abs(np.log10(dat[:,0]/tk))>0.012
    ok&=(dat[:,3]/dat[:,2]<6);dat=dat[ok]
    from scipy.signal import medfilt
    fit=np.c_[fit[:,0],np.exp(medfilt(np.log(fit[:,1]),9))][5:-5]
    a.fill_between(dat[:,0],dat[:,2],dat[:,3],color='#c9c8c0',lw=0,label='측정 데이터 범위 (논문 Fig. 6 디지타이즈)')
    a.plot(fit[:,0],fit[:,1],color=O,lw=4,alpha=.55,solid_capstyle='round',label='논문의 피팅 곡선 (디지타이즈)')
    for tag,ls,lab,col in (('bw','-','기준해 + 검출기 400 MHz·오실로 500 MHz',B),('ideal',(0,(4,3)),'기준해 (대역폭 무제한)',A)):
        t,R=c[k+tag+'_t']*1e9,c[k+tag+'_R'];t0=pk-t[R.argmax()];m=(t+t0>xl[0])
        a.plot(t[m]+t0,R[m],color=col,lw=1.8,ls=ls,label=lab)
        pf=fit[fit[:,0]>pk+1];r.plot(pf[:,0],pf[:,1]/np.interp(pf[:,0]-t0,t,R),color=col,lw=1.8,ls=ls)
    a.set_xscale('log');a.set_yscale('log');a.set_xlim(*xl);a.set_ylim(3e-3,1.5)
    a.set_title(ttl,fontsize=11,color=INK,loc='left');r.axhline(1,color=MUT,lw=.8)
    r.set_ylim(.85,1.2);r.set_xlabel('시간 (ns, 논문 축 기준)',color=MUT)
    for q in (a,r):
        q.grid(True,which='major',color='#e6e5df',lw=.7);q.tick_params(colors=MUT,labelsize=9)
        for s in q.spines.values():s.set_color('#d0cfc8')
        q.spines['top'].set_visible(False);q.spines['right'].set_visible(False)
ax[0,0].set_ylabel('정규화 ΔT',color=MUT);ax[1,0].set_ylabel('논문 피팅 / 기준해',color=MUT)
ax[0,1].legend(frameon=False,fontsize=9,loc='upper right',labelcolor=INK)
fig.suptitle('Mao 2024 (JAP 135, 095102) Fig. 6  vs  다층 해석 기준해 (refml.py)',fontsize=13,color=INK,x=.125,ha='left',y=.97)
fig.text(.125,.015,'입력: 논문 Table I 물성, 펌프 3.2 ns FWHM, 1/e² 반경 펌프 65/62 µm · 프로브 8.7/3.5 µm, 표면 흡수. 시간축은 피크 위치를 맞춰 정렬. 원 그림: Mao et al., CC BY 4.0',fontsize=8,color=MUT)
fig.savefig('mao_compare.png',dpi=150,bbox_inches='tight',facecolor='white')
