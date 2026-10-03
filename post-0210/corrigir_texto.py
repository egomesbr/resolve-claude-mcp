import cv2, numpy as np, pickle, subprocess, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
V="/root/.claude/uploads/df8ccbb2-8b73-5561-85d5-11f095820b9a/1a052398-Post_0210.mp4"
OUT=sys.argv[1] if len(sys.argv)>1 else "fixed_noaudio.mp4"
cap=cv2.VideoCapture(V); frames=[]
while True:
    ok,f=cap.read()
    if not ok: break
    frames.append(f)
N=len(frames); START,REF,END=240,330,357
ref=frames[REF]
# ---------- 1. phone transform P[i]: ref->frame i (ECC on static lower-screen ROI) ----------
roi_ref=np.array([[160,700],[570,700],[570,1000],[160,1000]],np.float32)
def gray(i): return cv2.GaussianBlur(cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY),(5,5),0)
P={REF:np.array([[1,0,0],[0,1,0]],np.float32)}
def step(src,dst,Ts):
    roi=cv2.transform(roi_ref.reshape(-1,1,2),Ts).reshape(-1,2)
    m=np.zeros(frames[0].shape[:2],np.uint8); cv2.fillPoly(m,[roi.astype(np.int32)],255); m=cv2.erode(m,np.ones((9,9),np.uint8))
    W=np.eye(2,3,dtype=np.float32)
    try: cc,W=cv2.findTransformECC(gray(src),gray(dst),W,cv2.MOTION_AFFINE,(3,200,1e-6),m,5)
    except cv2.error: return Ts
    return (np.vstack([W,[0,0,1]])@np.vstack([Ts,[0,0,1]]))[:2].astype(np.float32)
for i in range(REF+1,END+1): P[i]=step(i-1,i,P[i-1])
for i in range(REF-1,START-1,-1): P[i]=step(i+1,i,P[i+1])
# ---------- 2. elements in ref coords ----------
# glyph bands (x0,y0,x1,y1) for inpainting; box = bubble rect used for per-bubble tracking
E_={
 "label":dict(band=(279,366,392,396), box=(270,360,400,400)),
 "b2":   dict(band=(292,432,545,472), box=(278,412,578,488)),
 "b3a":  dict(band=(195,548,470,606), box=(176,532,556,666)),
 "b3b":  dict(band=(195,604,395,646), box=(176,532,556,666)),
}
b,g,r=[ref[:,:,c].astype(int) for c in range(3)]
lum=(b+g+r)/3
glyph=(((lum>105)&(abs(r-g)<50)&(abs(g-b)<50))|((g>140)&(b>140)&(r<175)&(g-r>45))).astype(np.uint8)*255
refmask=np.zeros_like(glyph)
for k,e in E_.items():
    x0,y0,x1,y1=e["band"]; refmask[y0:y1,x0:x1]=glyph[y0:y1,x0:x1]
refmask=cv2.dilate(refmask,np.ones((5,5),np.uint8))

# ---------- 3. per-bubble position: P-predicted+template match (<=CH), then chained frame-to-frame ----------
CH=345
def sc(i): return float(np.sqrt(abs(np.linalg.det(P[i][:,:2]))))
bub={"label":"label","b2":"b2","b3":"b3a"}
POS={k:{} for k in bub}      # top-left of bubble box in frame i
def gry(i): return cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY)
for k,ek in bub.items():
    x0,y0,x1,y1=E_[ek]["box"]
    prev=(0.0,0.0)
    seq=list(range(REF,START-1,-1))[1:]
    POS[k][REF]=(float(x0),float(y0))
    # backwards + forward to CH: template = ref crop warped by P[i]
    for rng in (list(range(REF,START-1,-1)),list(range(REF,CH+1))):
        prev=(0.0,0.0)
        for i in rng:
            pred=cv2.cvtColor(cv2.warpAffine(ref,P[i],(720,1280)),cv2.COLOR_BGR2GRAY); cur=gry(i)
            p0=cv2.transform(np.array([[[x0,y0]],[[x1,y1]]],np.float32),P[i]).reshape(-1,2)
            tx,ty=int(round(p0[0,0])),int(round(p0[0,1])); tw=int(max(40,(p0[1,0]-p0[0,0])*0.30)); th=int(p0[1,1]-p0[0,1])
            ok=(tx>=0 and ty>=0 and tx+tw<=720 and ty+th<=1280 and th>10)
            d=prev
            if ok:
                R=60; sx0=int(max(0,tx+prev[0]-R)); sy0=int(max(0,ty+prev[1]-R)); sx1=int(min(720,tx+tw+prev[0]+R)); sy1=int(min(1280,ty+th+prev[1]+R))
                win=cur[sy0:sy1,sx0:sx1]; tpl=pred[ty:ty+th,tx:tx+tw]
                if win.shape[0]>=tpl.shape[0] and win.shape[1]>=tpl.shape[1]:
                    res=cv2.matchTemplate(win,tpl,cv2.TM_CCOEFF_NORMED); _,mx,_,loc=cv2.minMaxLoc(res)
                    if mx>0.4: d=(sx0+loc[0]-tx,sy0+loc[1]-ty)
            prev=d
            POS[k][i]=(float(p0[0,0]+d[0]),float(p0[0,1]+d[1]))
    # chained: template = previous frame's actual crop
    for i in range(CH+1,END+1):
        px,py=POS[k][i-1]; s=sc(i-1); tw=int(max(30,(x1-x0)*s*0.22)); th=int((y1-y0)*s)
        pg=gry(i-1); cur=gry(i)
        ix,iy=int(round(px)),int(round(py))
        tpl=pg[max(0,iy):iy+th,max(0,ix):ix+tw]
        if tpl.shape[0]<20 or tpl.shape[1]<20: POS[k][i]=(px+25,py); continue
        Rx0,Rx1,Ry=40,160,50
        sx0=max(0,ix-Rx0); sx1=min(720,ix+tw+Rx1); sy0=max(0,iy-Ry); sy1=min(1280,iy+th+Ry)
        win=cur[sy0:sy1,sx0:sx1]
        if win.shape[0]<tpl.shape[0] or win.shape[1]<tpl.shape[1]: POS[k][i]=(px+25,py); continue
        res=cv2.matchTemplate(win,tpl,cv2.TM_CCOEFF_NORMED); _,mx,_,loc=cv2.minMaxLoc(res)
        POS[k][i]=(float(sx0+loc[0]-max(0,iy and 0)),float(sy0+loc[1])) if mx>0.3 else (px+25,py)
        # loc is crop top-left; crop was taken at (max(0,ix),max(0,iy)) -> bubble tl differs by clip only
        POS[k][i]=(POS[k][i][0]-(max(0,ix)-ix),POS[k][i][1]-(max(0,iy)-iy))
pickle.dump((P,POS),open("PD.pkl","wb"))
for k in bub: print(k,{i:tuple(round(v) for v in POS[k][i]) for i in (240,300,330,345,348,350,352,354,356,357)})

# ---------- 4. render: erase wrong glyphs (inpaint) and draw corrected text ----------
FONT="/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
SS=3                                   # supersampling for the text layers
def font(px): return ImageFont.truetype(FONT,int(round(px*SS)))
# layer geometry: each text layer lives in ref coords relative to its bubble box top-left
def make_layer(ek,draw_fn):
    x0,y0,x1,y1=E_[ek]["box"]; w,h=x1-x0,y1-y0
    im=Image.new("RGBA",(w*SS,h*SS),(0,0,0,0)); d=ImageDraw.Draw(im)
    glow=Image.new("RGBA",im.size,(0,0,0,0)); gd=ImageDraw.Draw(glow)
    draw_fn(d,gd,x0,y0)
    glow=glow.filter(ImageFilter.GaussianBlur(5*SS/2))
    out=Image.alpha_composite(glow,im)
    return np.array(out)
CY=(255,247,161); CYb=(161,247,255)       # cyan (RGB) for 'Isso já'
WH=(236,240,250); GR=(203,206,214)
def fit(text,width):
    px=20.0
    while ImageFont.truetype(FONT,int(px*SS)).getlength(text)<width*SS: px+=0.25
    return px-0.25
FS_B2=fit("já pode ser automático.",246); FS_B3=fit("Isso já pode ser",266)
print("font sizes",FS_B2,FS_B3)
def draw_b2(d,gd,x0,y0):
    f=font(FS_B2); base=(460-y0)*SS
    t="já pode ser automático."
    d.text(((296-x0)*SS,base),t,font=f,fill=GR+(255,),anchor="ls")
def draw_b3(d,gd,x0,y0):
    f=font(FS_B3); bx=(198-x0)*SS; b1=(589-y0)*SS; b2_=(634-y0)*SS
    p1="Isso já"; p2=" pode ser"
    w1=d.textlength(p1,font=f)
    gd.text((bx,b1),p1,font=f,fill=(120,230,255,255),anchor="ls")
    d.text((bx,b1),p1,font=f,fill=(190,245,255,255),anchor="ls")
    d.text((bx+w1,b1),p2,font=f,fill=WH+(255,),anchor="ls")
    d.text((bx,b2_),"automático.",font=f,fill=GR+(255,),anchor="ls")
L={"b2":make_layer("b2",draw_b2),"b3":make_layer("b3a",draw_b3)}
# screen clip polygon in ref coords (glass interior), text must not leave it during slide-out
clip_ref=np.array([[150,120],[592,120],[592,1000],[150,1000]],np.float32)
def Aof(i): return np.eye(2,dtype=np.float32)*sc(i)
def affine_for(k,ek,i):
    A=Aof(i); x0,y0,_,_=E_[ek]["box"]
    t=np.array(POS[k][i])-A@np.array([x0,y0],np.float32)
    return np.hstack([A,t.reshape(2,1)]).astype(np.float32)
def glyph_mask(f):
    b,g,r=[f[:,:,c].astype(np.int16) for c in range(3)]
    lum=(b+g+r)//3
    m=(((lum>95)&(abs(r-g)<55)&(abs(g-b)<55))|((g>130)&(b>130)&(r<175)&(g-r>40)))
    return m.astype(np.uint8)*255
def band_rects(i,k):
    """text-line rectangles of bubble k in frame i (from tracked bubble position and scale)"""
    s=sc(i); out=[]
    for kk,e in E_.items():
        if kk.startswith(k):
            bx0,by0,_,_=E_[k if k in E_ else kk]["box"]; x0,y0,x1,y1=e["band"]
            px,py=POS[{"label":"label","b2":"b2","b3":"b3"}[k]][i]
            out.append((px+(x0-bx0)*s,py+(y0-by0)*s,px+(x1-bx0)*s,py+(y1-by0)*s))
    return out
def process(i):
    f=frames[i].copy(); s=sc(i)
    gm=glyph_mask(f); mask=np.zeros((1280,720),np.uint8); allow=np.zeros((1280,720),np.uint8)
    for k in ("label","b2","b3"):
        m=int(round(5+4*s)); extra=0
        for (x0,y0,x1,y1) in band_rects(i,k):
            xa,ya,xb,yb=int(x0-m),int(y0-m),int(x1+m),int(y1+m)
            xb+=int(10*s)
            if k=="label": xb+=int(40*s)
            xa,ya=max(xa,0),max(ya,0); xb,yb=min(xb,720),min(yb,1280)
            if xb>xa and yb>ya:
                mask[ya:yb,xa:xb]|=gm[ya:yb,xa:xb]
                if k!='label': allow[ya:yb,xa:xb]|=gm[ya:yb,xa:xb]
        M=affine_for(k,{"label":"label","b2":"b2","b3":"b3a"}[k],i)
        # also the warped reference mask (covers tiny, low-contrast text in small frames)
        if s<0.75:
            rm=np.zeros_like(refmask)
            for kk,e in E_.items():
                if kk.startswith(k):
                    x0_,y0_,x1_,y1_=e["band"]; rm[y0_:y1_,x0_:x1_]=refmask[y0_:y1_,x0_:x1_]
            mask|=cv2.warpAffine(rm,M,(720,1280),flags=cv2.INTER_NEAREST)
    # label line sits ~52*s px above b2 and slides right more slowly than the bubbles: use a wide strip
    bx,by=POS["b2"][i]; ly0=int(by-62*s); ly1=int(by-6*s)
    cpoly=cv2.transform(clip_ref.reshape(-1,1,2),P[i]).reshape(-1,2)
    rx=int(cpoly[:,0].min()+ (592-150)*s - 2*s)
    lx=POS['label'][i][0]; xa=max(int(min(bx-30*s,lx-12*s)),0); xb=min(rx,720)
    if ly0>=0 and xb>xa and ly1>ly0: mask[ly0:ly1,xa:xb]|=gm[ly0:ly1,xa:xb]
    mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7) if i>=349 else (5,5)))
    cp=cv2.transform(clip_ref.reshape(-1,1,2),P[i]).reshape(-1,2).astype(np.int32)
    cm=np.zeros_like(mask); cv2.fillPoly(cm,[cp],255); mask&=cm
    # smooth background fill: normalized convolution from unmasked neighbours
    valid=(mask==0).astype(np.float32); ff=f.astype(np.float32)
    sig=max(3.0,5.0*s)
    num=cv2.GaussianBlur(ff*valid[:,:,None],(0,0),sig); den=cv2.GaussianBlur(valid,(0,0),sig)[:,:,None]+1e-4
    fill=num/den
    soft=cv2.GaussianBlur(mask.astype(np.float32)/255,(0,0),0.8)[:,:,None]
    f=(ff*(1-soft)+fill*soft).clip(0,255).astype(np.uint8)
    clipf=cv2.GaussianBlur(cm,(3,3),0).astype(np.float32)/255
    for k,ek in (("b2","b2"),("b3","b3a")):
        if (k=='b2' and i>353) or (k=='b3' and i>356): continue     # original bubble already left the screen
        A=Aof(i)
        Ml=np.hstack([A/SS,np.array(POS[k][i],np.float32).reshape(2,1)]).astype(np.float32)   # layer px -> frame px
        lay=L[k]
        sig2=0.5*SS*0.5/max(s,0.25)
        if s<1.2: lay=cv2.GaussianBlur(lay,(0,0),sig2)
        w=cv2.warpAffine(lay,Ml,(720,1280),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
        a=(w[:,:,3:4].astype(np.float32)/255)*clipf[:,:,None]
        if k=='b3' and i<256: a=a*float(np.clip((i-243)/11.0,0,1))**2
        if i>=350:                       # exit phase: only draw where the original text is still on screen
            al=cv2.dilate(allow,np.ones((15,41),np.uint8)); a=a*(cv2.GaussianBlur(al,(0,0),2).astype(np.float32)/255)[:,:,None]
        rgb=w[:,:,2::-1].astype(np.float32)
        f=(f.astype(np.float32)*(1-a)+rgb*a).clip(0,255).astype(np.uint8)
    return f
if __name__=="__main__":
    prevs=[240,246,250,255,260,300,330,352]
    rows=[]
    for i in prevs:
        o=frames[i]; n=process(i)
        crop=lambda im: cv2.resize(im[250:780,60:700],(320,265))
        rows.append(np.vstack([crop(o),crop(n)]))
    cv2.imwrite("qa.png",np.hstack(rows))
    # full encode
    if len(sys.argv)>2: sys.exit()
    enc=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s","720x1280","-r","30","-i","-","-c:v","libx264","-crf","12","-preset","slow","-pix_fmt","yuv420p",OUT],stdin=subprocess.PIPE)
    for i in range(N):
        out=process(i) if START<=i<=END else frames[i]
        enc.stdin.write(out.tobytes())
    enc.stdin.close(); enc.wait(); print("encoded",OUT)
