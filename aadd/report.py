"""Interactive, self-contained HTML dashboard for a scanned dataset.

    aadd scan DATA_DIR --json scan.json
    aadd report scan.json -o report.html

Per-second probabilities of every file are embedded (one byte per value,
base64) and all statistics are computed in the browser, so the threshold, the
tolerated share of degraded seconds and the set of classes that count can be
changed live. "Subset" = name of the folder that contains the file. No external
assets; light and dark themes follow the OS setting.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import json
from pathlib import Path

import numpy as np

from aadd.constants import NONE_CLASS

PRETTY = {
    "noise": "Background noise",
    "music": "Music",
    "reverb": "Reverberation",
    "clipping": "Clipping",
    "lowpass": "Low-pass / narrow band",
    "highpass": "High-pass",
    "codec": "Codec artefacts",
    "packet_loss": "Packet loss",
    "hum": "Mains hum",
    "clicks": "Clicks",
    "pitch_shift": "Pitch shift",
    "time_stretch": "Time stretch",
    "AddGaussianNoise": "Gaussian noise",
    "PitchShift": "Pitch shift",
    "TimeStretch": "Time stretch",
    "Shift": "Discontinuity (shift)",
    "ClippingDistortion": "Clipping",
    "LowPassFilter": "Low-pass",
    "HighPassFilter": "High-pass",
}

CSS = """
.viz-root{color-scheme:light;--surface-0:#f4f4f2;--surface-1:#fcfcfb;--border:#e4e3df;--grid:#ecebe7;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#77766f;--series-1:#2a78d6;
--seq-0:#f0efec;--seq-1:#cde2fb;--seq-2:#9ec5f4;--seq-3:#6da7ec;--seq-4:#3987e5;--seq-5:#256abf;--seq-6:#184f95;--seq-7:#0d366b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-0:#111110;
--surface-1:#1a1a19;--border:#2c2c2a;--grid:#262624;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#94938b;
--series-1:#3987e5;--seq-0:#262624;--seq-1:#0d366b;--seq-2:#104281;--seq-3:#184f95;--seq-4:#256abf;--seq-5:#3987e5;--seq-6:#6da7ec;--seq-7:#b7d3f6}}
*{box-sizing:border-box}body{margin:0;background:var(--surface-0)}
.viz-root{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;color:var(--text-primary);
background:var(--surface-0);padding:28px 32px 48px;max-width:1280px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0 0 2px}.sub{color:var(--text-secondary);font-size:13px}
.muted{color:var(--text-muted)}.grid{display:grid;gap:16px}.g5{grid-template-columns:repeat(5,1fr)}.g2{grid-template-columns:1fr 1fr}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:16px 18px}
.tile .v{font-size:26px;font-weight:650;letter-spacing:-.01em;font-variant-numeric:tabular-nums}.tile .l{color:var(--text-secondary);font-size:12.5px}
.tile .d{color:var(--text-muted);font-size:12px;margin-top:2px}.tile.hero{border-color:var(--series-1)}
section{margin-top:16px}.head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:10px}
.controls{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:14px 28px;align-items:center;
background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:12px 18px;margin-top:16px}
.ctl{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--text-secondary)}
.ctl b{color:var(--text-primary);font-variant-numeric:tabular-nums;min-width:40px;display:inline-block}
input[type=range]{width:180px;accent-color:var(--series-1)}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--border);
border-radius:999px;padding:2px 10px;font-size:12px;color:var(--text-muted);cursor:pointer;user-select:none;background:var(--surface-1)}
.chip.on{border-color:var(--series-1);color:var(--text-primary)}.chip input{display:none}
.chip .box{width:10px;height:10px;border-radius:3px;border:1.5px solid var(--text-muted)}.chip.on .box{background:var(--series-1);border-color:var(--series-1)}
.tag{display:inline-block;border:1px solid var(--border);border-radius:999px;padding:0 8px;margin:1px 3px 1px 0;font-size:12px;color:var(--text-secondary)}
button{font:inherit;font-size:12.5px;border:1px solid var(--border);background:var(--surface-1);color:var(--text-primary);border-radius:8px;padding:5px 10px;cursor:pointer}
button:hover{border-color:var(--series-1)}
svg text{fill:var(--text-secondary);font-size:12px}svg .lbl{fill:var(--text-primary)}svg .val{fill:var(--text-secondary);font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;color:var(--text-muted);font-weight:500;
border-bottom:1px solid var(--border);padding:6px 8px}td{border-bottom:1px solid var(--grid);padding:6px 8px;vertical-align:middle}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.strip{display:flex;gap:1px;height:14px;min-width:160px}.strip i{flex:1;border-radius:2px}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;
padding:6px 9px;font-size:12px;color:var(--text-primary);box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;z-index:9;max-width:320px}
@media (max-width:900px){.g5{grid-template-columns:repeat(2,1fr)}.g2{grid-template-columns:1fr}}
"""

JS = r"""
const D=JSON.parse(document.getElementById('aadd-data').textContent);
const C=D.classes.length, PR=D.pretty;
for(const f of D.files){const b=atob(f.q);const a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);f.p=a;}
const st={thr:Math.round(D.threshold*100),tol:0,on:D.classes.map(()=>true)};
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct=(x,d=0)=>(100*x).toFixed(d)+'%', hrs=s=>s>=3600?(s/3600).toFixed(1)+' h':(s/60).toFixed(0)+' min';
const seq=v=>v<=0.001?'var(--seq-0)':`var(--seq-${Math.min(7,1+Math.floor(v*7))})`;

function compute(thr,tol,on,light){
  const T=Math.round(thr*255);
  const R={secs:0,deg:0,dur:0,keepDur:0,keepFiles:0,cls:new Float64Array(C),clsFiles:new Float64Array(C),co:new Float64Array(C*C),sub:{},files:[],hist:new Array(10).fill(0)};
  const act=[];
  for(const f of D.files){
    const n=f.n; let deg=0; const fc=new Uint8Array(C);
    const s=R.sub[f.s]||(R.sub[f.s]={files:0,dur:0,keepDur:0,deg:0,secs:0,cls:new Float64Array(C)});
    s.files++; s.dur+=f.d; s.secs+=n; R.dur+=f.d; R.secs+=n;
    for(let i=0;i<n;i++){act.length=0;
      for(let j=0;j<C;j++){if(on[j]&&f.p[i*C+j]>=T){act.push(j);fc[j]=1;if(!light){R.cls[j]++;s.cls[j]++;}}}
      if(act.length){deg++;if(!light)for(const a of act)for(const b of act)R.co[a*C+b]++;}}
    const frac=n?deg/n:0; R.deg+=deg; s.deg+=deg;
    const keep=frac<=tol+1e-9; if(keep){R.keepDur+=f.d;R.keepFiles++;s.keepDur+=f.d;}
    if(!light){R.hist[Math.min(9,Math.floor(frac*10))]++;for(let j=0;j<C;j++)R.clsFiles[j]+=fc[j];R.files.push({f,frac,deg,fc,keep});}
  }
  return R;
}
function tiles(R){
  const fl=D.files.length, d=Math.max(R.dur,1e-9);
  const t=[
    ['hero',hrs(R.keepDur)+' <span class="muted" style="font-size:15px;font-weight:500">of '+hrs(R.dur)+'</span>','clean audio kept',pct(R.keepDur/d,1)+' of the dataset'],
    ['',R.keepFiles.toLocaleString(),'clean files kept',pct(R.keepFiles/Math.max(fl,1),1)+' of '+fl.toLocaleString()],
    ['',(fl-R.keepFiles).toLocaleString(),'files dropped',st.tol>0?'more than '+pct(st.tol)+' of seconds degraded':'any degraded second'],
    ['',pct(R.deg/Math.max(R.secs,1),1),'of audio time degraded',R.deg.toLocaleString()+' of '+R.secs.toLocaleString()+' seconds'],
    ['',D.errors.toLocaleString(),'unreadable files',D.errors?'decode errors':'none']];
  $('tiles').innerHTML=t.map(([c,v,l,x])=>`<div class="card tile ${c}"><div class="v">${v}</div><div class="l">${l}</div><div class="d">${x}</div></div>`).join('');
}
function hbar(el,rows){
  const W=560,rh=26,lw=170,pw=W-lw-56,H=rh*rows.length+8, vmax=Math.max(1e-9,...rows.map(r=>r[1]));
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  rows.forEach(([lab,v,tip,off],i)=>{const y=4+i*rh, w=(v>0&&!off)?Math.max(2,pw*v/vmax):0;
    s+=`<text class="lbl" x="${lw-10}" y="${y+rh/2+4}" text-anchor="end" opacity="${off?0.45:1}">${esc(lab)}</text><rect x="${lw}" y="${y+5}" width="${pw}" height="${rh-10}" rx="4" fill="var(--grid)"/>`;
    if(w)s+=`<rect data-tip="${esc(tip)}" x="${lw}" y="${y+5}" width="${w.toFixed(1)}" height="${rh-10}" rx="4" fill="var(--series-1)"/>`;
    s+=`<text class="val" x="${lw+w+6}" y="${y+rh/2+4}">${off?'not counted':pct(v)}</text>`;});
  el.innerHTML=s+'</svg>';
}
function hist(el,R){
  const W=560,H=220,L=40,B=34,Tp=10,pw=W-L-10,ph=H-B-Tp,bw=pw/10,cm=Math.max(1,...R.hist);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  for(let k=0;k<4;k++){const t=cm*k/3,y=Tp+ph-ph*t/cm;s+=`<line x1="${L}" x2="${W-10}" y1="${y}" y2="${y}" stroke="var(--grid)"/><text x="${L-6}" y="${y+4}" text-anchor="end">${Math.round(t)}</text>`;}
  R.hist.forEach((c,i)=>{const h=ph*c/cm,x=L+i*bw+1;
    if(c)s+=`<rect data-tip="<b>${i*10}–${(i+1)*10}%</b> of the file degraded<br>${c} files" x="${x}" y="${Tp+ph-h}" width="${bw-2}" height="${h}" rx="3" fill="var(--series-1)"/>`;
    s+=`<text x="${x+bw/2-1}" y="${H-B+16}" text-anchor="middle">${i*10}</text>`;});
  const tx=L+pw*st.tol; s+=`<line x1="${tx}" x2="${tx}" y1="${Tp}" y2="${Tp+ph}" stroke="var(--text-primary)" stroke-dasharray="3 3"/><text x="${tx+4}" y="${Tp+10}" class="lbl">tolerance</text>`;
  s+=`<text x="${L+pw/2}" y="${H-4}" text-anchor="middle">share of the file’s seconds with a counted degradation, %</text>`;
  el.innerHTML=s+'</svg>';
}
function curve(el){
  const xs=[],ys=[];for(let t=5;t<=95;t+=5){const R=compute(t/100,st.tol,st.on,true);xs.push(t/100);ys.push(R.keepDur/Math.max(R.dur,1e-9));}
  const W=560,H=220,L=44,B=34,Tp=12,pw=W-L-14,ph=H-B-Tp, X=x=>L+pw*(x-0.05)/0.9, Y=y=>Tp+ph-ph*y;
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  for(const y of [0,.25,.5,.75,1])s+=`<line x1="${L}" x2="${W-14}" y1="${Y(y)}" y2="${Y(y)}" stroke="var(--grid)"/><text x="${L-6}" y="${Y(y)+4}" text-anchor="end">${y*100}%</text>`;
  for(const x of [.1,.3,.5,.7,.9])s+=`<text x="${X(x)}" y="${H-B+16}" text-anchor="middle">${x.toFixed(1)}</text>`;
  s+=`<path d="${xs.map((x,i)=>(i?'L':'M')+X(x).toFixed(1)+' '+Y(ys[i]).toFixed(1)).join(' ')}" fill="none" stroke="var(--series-1)" stroke-width="2"/>`;
  xs.forEach((x,i)=>{s+=`<circle data-tip="threshold ${x.toFixed(2)}<br><b>${pct(ys[i],1)}</b> of the audio kept" cx="${X(x)}" cy="${Y(ys[i])}" r="9" fill="transparent"/>`;});
  const cur=compute(st.thr/100,st.tol,st.on,true), cx=X(Math.min(.95,Math.max(.05,st.thr/100))), cy=Y(cur.keepDur/Math.max(cur.dur,1e-9));
  s+=`<line x1="${cx}" x2="${cx}" y1="${Tp}" y2="${Tp+ph}" stroke="var(--text-primary)" stroke-dasharray="3 3"/><circle cx="${cx}" cy="${cy}" r="5" fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="2"/>`;
  s+=`<text x="${L+pw/2}" y="${H-4}" text-anchor="middle">detection threshold</text>`;
  el.innerHTML=s+'</svg>';
}
function heat(el,R){
  const minSec=Math.max(20,0.005*R.secs); const keep=[...Array(C).keys()].filter(j=>st.on[j]&&R.co[j*C+j]>=minSec).sort((a,b)=>R.co[b*C+b]-R.co[a*C+a]);
  $('heat-note').textContent='row: share of its seconds where the column class is also present · classes with ≥ '+Math.round(minSec)+' s';
  if(keep.length<2){el.innerHTML='<p class="muted">Not enough detections to show co-occurrence at this setting.</p>';return;}
  const W=560,lw=160,n=keep.length,cell=(W-lw-70)/n,H=lw+n*cell+4; let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  keep.forEach((j,c)=>{const x=lw+c*cell+cell/2;s+=`<text x="${x}" y="${lw-8}" transform="rotate(-50 ${x} ${lw-8})">${esc(PR[j])}</text>`;});
  keep.forEach((i,r)=>{const y=lw+r*cell,N=R.co[i*C+i];s+=`<text class="lbl" x="${lw-8}" y="${y+cell/2+4}" text-anchor="end">${esc(PR[i])}</text>`;
    keep.forEach((j,c)=>{const x=lw+c*cell+1;if(i===j){s+=`<rect x="${x}" y="${y+1}" width="${cell-2}" height="${cell-2}" rx="3" fill="none" stroke="var(--border)"/>`;return;}
      const v=R.co[i*C+j]/N;s+=`<rect data-tip="<b>${esc(PR[i])}</b>: ${N} s<br>also <b>${esc(PR[j])}</b>: ${pct(v,1)}" x="${x}" y="${y+1}" width="${cell-2}" height="${cell-2}" rx="3" fill="${seq(v)}"/>`;});});
  el.innerHTML=s+'</svg>';
}
function subs(R){
  const rows=Object.entries(R.sub).sort((a,b)=>b[1].deg/Math.max(b[1].secs,1)-a[1].deg/Math.max(a[1].secs,1));
  $('subs').innerHTML='<tr><th>folder</th><th class=num>files</th><th>clean audio kept</th><th class=num>degraded time</th><th>top degradations (% time)</th></tr>'+rows.map(([k,s])=>{
    const kept=s.keepDur/Math.max(s.dur,1e-9), top=[...Array(C).keys()].filter(j=>s.cls[j]>0).sort((a,b)=>s.cls[b]-s.cls[a]).slice(0,3);
    return `<tr><td>${esc(k)}</td><td class=num>${s.files.toLocaleString()}</td>
    <td><svg viewBox="0 0 100 12" width="100" height="12"><rect width="100" height="12" rx="4" fill="var(--grid)"/><rect width="${kept?Math.max(2,100*kept):0}" height="12" rx="4" fill="var(--series-1)"/></svg> <span class=muted>${pct(kept)} of ${hrs(s.dur)}</span></td>
    <td class=num>${pct(s.deg/Math.max(s.secs,1))}</td><td>${top.map(j=>`<span class=tag>${esc(PR[j])} ${pct(s.cls[j]/Math.max(s.secs,1))}</span>`).join('')||'<span class=muted>—</span>'}</td></tr>`}).join('');
}
function strip(f){
  const n=f.n, step=Math.max(1,Math.ceil(n/60)), T=st.thr/100; let out='';
  for(let k=0;k<n;k+=step){let v=0;const labs=new Set();for(let i=k;i<Math.min(n,k+step);i++)for(let j=0;j<C;j++){if(!st.on[j])continue;const p=f.p[i*C+j]/255;if(p>v)v=p;if(p>=T)labs.add(PR[j]);}
    out+=`<i data-tip="${k}–${Math.min(n,k+step)} s: ${esc([...labs].join(', ')||'clean')}" style="background:${seq(v)}"></i>`;}
  return `<div class=strip>${out}</div>`;
}
function worst(R){
  const flagged=R.files.filter(x=>x.deg>0), nc=x=>x.fc.reduce((s,v)=>s+v,0);
  const top=flagged.sort((a,b)=>b.frac-a.frac||nc(b)-nc(a)||a.f.name.localeCompare(b.f.name)).slice(0,25);
  $('worst-note').textContent=`top ${top.length} of ${flagged.length.toLocaleString()} flagged files · strip = max probability per second`;
  $('worst').innerHTML='<tr><th>file</th><th>folder</th><th class=num>length</th><th class=num>degraded</th><th>timeline</th><th>detected</th></tr>'+top.map(x=>
    `<tr><td title="${esc(x.f.path)}">${esc(x.f.name)}</td><td class=muted>${esc(x.f.s)}</td><td class=num>${x.f.d.toFixed(1)} s</td><td class=num>${pct(x.frac)}</td><td>${strip(x.f)}</td><td>${[...Array(C).keys()].filter(j=>x.fc[j]).map(j=>`<span class=tag>${esc(PR[j])}</span>`).join('')}</td></tr>`).join('');
}
let LAST=null;
function render(){
  $('thr-v').textContent=(st.thr/100).toFixed(2); $('tol-v').textContent=pct(st.tol);
  const R=compute(st.thr/100,st.tol,st.on,false); LAST=R; tiles(R);
  const order=[...Array(C).keys()].sort((a,b)=>(st.on[b]-st.on[a])||R.cls[b]-R.cls[a]);
  hbar($('bar-time'),order.map(j=>[PR[j],R.cls[j]/Math.max(R.secs,1),`<b>${esc(PR[j])}</b><br>${R.cls[j].toLocaleString()} s (${pct(R.cls[j]/Math.max(R.secs,1),1)})<br>in ${R.clsFiles[j]} files`,!st.on[j]]));
  hbar($('bar-files'),order.map(j=>[PR[j],R.clsFiles[j]/Math.max(R.files.length,1),`<b>${esc(PR[j])}</b><br>${R.clsFiles[j]} of ${R.files.length} files`,!st.on[j]]));
  hist($('hist'),R); heat($('heat'),R); subs(R); worst(R); curve($('curve'));
}
$('chips').innerHTML=D.classes.map((c,j)=>`<label class="chip on" data-j="${j}"><input type=checkbox checked><span class=box></span>${esc(PR[j])}</label>`).join('');
$('chips').addEventListener('click',e=>{const ch=e.target.closest('.chip');if(!ch)return;e.preventDefault();const j=+ch.dataset.j;st.on[j]=!st.on[j];ch.classList.toggle('on',st.on[j]);render();});
$('thr').value=st.thr; $('thr').addEventListener('input',e=>{st.thr=+e.target.value;render();});
$('tol').addEventListener('input',e=>{st.tol=+e.target.value/100;render();});
$('reset').addEventListener('click',()=>{st.thr=Math.round(D.threshold*100);st.tol=0;st.on=st.on.map(()=>true);$('thr').value=st.thr;$('tol').value=0;document.querySelectorAll('.chip').forEach(c=>c.classList.add('on'));render();});
$('dl').addEventListener('click',()=>{const rows=['path,folder,duration_s,degraded_fraction,kept'];for(const x of LAST.files)rows.push(`"${x.f.path.replace(/"/g,'""')}","${x.f.s}",${x.f.d.toFixed(2)},${x.frac.toFixed(4)},${x.keep?1:0}`);
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\n')],{type:'text/csv'}));a.download=`aadd_files_thr${(st.thr/100).toFixed(2)}_tol${Math.round(st.tol*100)}.csv`;a.click();});
const tip=$('tip');
document.addEventListener('mousemove',e=>{const t=e.target.closest('[data-tip]');if(!t){tip.style.display='none';return}
  tip.innerHTML=t.dataset.tip;tip.style.display='block';tip.style.left=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=Math.min(e.clientY+14,innerHeight-tip.offsetHeight-8)+'px'});
render();
"""


def _pack(scan: dict) -> dict:
    classes = [c for c in scan["classes"] if c != NONE_CLASS]
    files, errors = [], 0
    for f in scan["files"]:
        fr = f.get("file", {})
        if fr.get("status") == "error":
            errors += 1
            continue
        segs = f["segments"]
        P = (
            np.array([[s.get(f"p_{c}", 0.0) for c in classes] for s in segs], dtype=np.float64)
            if segs
            else np.zeros((0, len(classes)))
        )
        q = np.clip(np.round(P * 255), 0, 255).astype(np.uint8)
        path = f["path"]
        files.append(
            {
                "path": path,
                "name": Path(path).name,
                "s": Path(path).parent.name or ".",
                "d": round(float(f.get("duration_s", fr.get("duration_s", 0.0))), 2),
                "n": len(P),
                "q": base64.b64encode(q.tobytes()).decode(),
            }
        )
    return {
        "classes": classes,
        "pretty": [PRETTY.get(c, c) for c in classes],
        "threshold": float(scan.get("threshold", 0.5)),
        "model": str(scan.get("model", "")),
        "errors": errors,
        "files": files,
    }


def build_report(scan: dict, title: str | None = None, top_files: int = 25) -> str:
    data = _pack(scan)
    title = title or "Audio degradation report"
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    model = Path(data["model"]).name or data["model"]
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    body = f"""
<div class="viz-root">
<h1>{html.escape(title)}</h1>
<div class="sub">model <b>{html.escape(model)}</b> · {len(data["files"]):,} files · generated {now} by <code>aadd report</code></div>
<div class="controls">
 <div class="ctl">detection threshold <input id="thr" type="range" min="5" max="95" step="1"> <b id="thr-v"></b></div>
 <div class="ctl">keep a file if at most <input id="tol" type="range" min="0" max="50" step="1" value="0"> <b id="tol-v"></b> of its seconds are degraded</div>
 <div class="ctl"><button id="reset">reset</button><button id="dl">download file list (CSV)</button></div>
 <div class="ctl" style="flex-basis:100%">count as degradation: <div class="chips" id="chips"></div></div>
</div>
<section class="grid g5" id="tiles"></section>
<section class="grid g2">
 <div class="card"><div class="head"><h2>Clean audio kept vs threshold</h2><span class="muted">at the current tolerance and classes</span></div><div id="curve"></div></div>
 <div class="card"><div class="head"><h2>How degraded is each file?</h2><span class="muted">dashed line = tolerance</span></div><div id="hist"></div></div>
</section>
<section class="grid g2">
 <div class="card"><div class="head"><h2>Share of audio time affected</h2><span class="muted">% of all seconds</span></div><div id="bar-time"></div></div>
 <div class="card"><div class="head"><h2>Files affected</h2><span class="muted">% of files with ≥ 1 detected second</span></div><div id="bar-files"></div></div>
</section>
<section class="grid g2">
 <div class="card"><div class="head"><h2>By folder</h2><span class="muted">sorted by degraded time</span></div><table id="subs"></table></div>
 <div class="card"><div class="head"><h2>Degradations that occur together</h2><span class="muted" id="heat-note"></span></div><div id="heat"></div></div>
</section>
<section class="card"><div class="head"><h2>Most degraded files</h2><span class="muted" id="worst-note"></span></div><table id="worst"></table></section>
</div><div id="tip"></div>"""
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}"
        f'<script id="aadd-data" type="application/json">{payload}</script><script>{JS}</script></body></html>'
    )


def write_report(scan_json: str, out: str, title: str | None = None, top_files: int = 25) -> str:
    scan = json.loads(Path(scan_json).read_text())
    Path(out).write_text(build_report(scan, title=title, top_files=top_files))
    return out
