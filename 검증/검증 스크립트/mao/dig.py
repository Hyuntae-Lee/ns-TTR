from PIL import Image;import numpy as np, colorsys
im=np.array(Image.open('mao_fig6.jpg').convert('RGB')).astype(float)
mx=im.max(2);mn=im.min(2);sat=(mx-mn)
col=(sat>90)            # coloured curve pixels
blk=(mx<90)
cal={'a':dict(x=(164,204.5,10),y=(25.5,185.5),xr=(106,632),mask=[(104,250,392,420),(288,15,632,172)]),
     'b':dict(x=(825,201.75,10),y=(21,154.25),xr=(748,1274),mask=[(946,15,1276,200),(746,255,1000,420)])}
out={}
for k,c in cal.items():
    M=np.ones(im.shape[:2],bool)
    for x0,y0,x1,y1 in c['mask']: M[y0:y1,x0:x1]=False
    M[:8]=False;M[424:]=False
    tx=lambda x:c['x'][2]*10**((x-c['x'][0])/c['x'][1]); ty=lambda y:10**(-(y-c['y'][0])/c['y'][1])
    fit=[];dat=[]
    for x in range(c['xr'][0]+3,c['xr'][1]-2):
        yc=np.where(col[:,x]&M[:,x])[0]
        yb=np.where(blk[:,x]&M[:,x])[0]
        if len(yc)>=2: fit.append((tx(x),ty(np.median(yc))))
        if len(yb)>=1: dat.append((tx(x),ty(np.median(yb)),ty(yb.max()),ty(yb.min())))
    out[k]=(np.array(fit),np.array(dat))
    print(k,len(fit),len(dat)); print(np.round(out[k][0][::40],4)); print(np.round(out[k][1][::40],4))
np.savez('dig.npz',fa=out['a'][0],da=out['a'][1],fb=out['b'][0],db=out['b'][1])
