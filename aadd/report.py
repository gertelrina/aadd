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
button{white-space:nowrap;font:inherit;font-size:12.5px;border:1px solid var(--border);background:var(--surface-1);color:var(--text-primary);border-radius:8px;padding:5px 10px;cursor:pointer}
button:hover{border-color:var(--series-1)}
.seg{display:inline-flex}.seg button{border-radius:0;margin-left:-1px}.seg button:first-child{border-radius:8px 0 0 8px;margin-left:0}.seg button:last-child{border-radius:0 8px 8px 0}
.seg button.on{background:var(--series-1);border-color:var(--series-1);color:#fff}
select{font:inherit;font-size:12.5px;border:1px solid var(--border);background:var(--surface-1);color:var(--text-primary);border-radius:8px;padding:4px 6px}
svg text{fill:var(--text-secondary);font-size:12px}svg .lbl{fill:var(--text-primary)}svg .val{fill:var(--text-secondary);font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;color:var(--text-muted);font-weight:500;
border-bottom:1px solid var(--border);padding:6px 8px}td{border-bottom:1px solid var(--grid);padding:6px 8px;vertical-align:middle}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.strip{display:flex;gap:1px;height:14px;min-width:160px}.strip i{flex:1;border-radius:2px}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;
padding:6px 9px;font-size:12px;color:var(--text-primary);box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;z-index:9;max-width:320px}
#worst tr.row{cursor:pointer}#worst tr.row:hover td{background:var(--grid)}
.play{display:inline-flex;width:22px;height:22px;border-radius:50%;border:1px solid var(--border);align-items:center;justify-content:center;font-size:10px;color:var(--text-secondary)}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.45);display:none;z-index:20;align-items:flex-start;justify-content:center;overflow:auto;padding:40px 16px}
#modal .panel{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;max-width:1100px;width:100%;padding:18px 22px}
#modal audio{width:100%;margin:8px 0 4px}.ivl{display:inline-flex;gap:6px;align-items:center;border:1px solid var(--border);border-radius:8px;padding:3px 9px;margin:3px 6px 3px 0;font-size:12.5px;cursor:pointer}
.ivl:hover{border-color:var(--series-1)}.find{display:flex;gap:8px;align-items:center}.find input{font:inherit;font-size:13px;padding:5px 9px;border:1px solid var(--border);border-radius:8px;background:var(--surface-1);color:var(--text-primary);width:260px}
@media (max-width:900px){.g5{grid-template-columns:repeat(2,1fr)}.g2{grid-template-columns:1fr}}
"""

JS = r"""
const D=window.AADD_DATA;
const C=D.classes.length, PR=D.pretty;
const b64=q=>{const b=atob(q);const a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a};
for(const f of D.files){f.p=b64(f.q);if(f.fq)f.fp=b64(f.fq);}
const st={thr:Math.round(D.threshold*100),tol:0,on:D.classes.map(()=>true),sort:'worst',rows:25};
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct=(x,d=0)=>(100*x).toFixed(d)+'%', hrs=s=>s>=3600?(s/3600).toFixed(1)+' h':(s/60).toFixed(0)+' min';
const qthr=t=>t>=1?256:Math.round(t*255);  // 8-bit probabilities; 1.00 means "never"
const seq=v=>v<=0.001?'var(--seq-0)':`var(--seq-${Math.min(7,1+Math.floor(v*7))})`;

function compute(thr,tol,on,light){
  const T=qthr(thr);
  const R={secs:0,deg:0,dur:0,keepDur:0,keepFiles:0,cls:new Float64Array(C),clsFiles:new Float64Array(C),co:new Float64Array(C*C),sub:{},files:[],hist:new Array(10).fill(0)};
  const act=[];
  for(const f of D.files){
    const n=f.n; let deg=0; const fc=new Uint8Array(C);
    const s=R.sub[f.s]||(R.sub[f.s]={files:0,dur:0,keepDur:0,deg:0,secs:0,cls:new Float64Array(C)});
    s.files++; s.dur+=f.d; s.secs+=n; R.dur+=f.d; R.secs+=n;
    let sev=0;
    for(let i=0;i<n;i++){act.length=0;let mx=0;
      for(let j=0;j<C;j++){if(!on[j])continue;const v=f.p[i*C+j];if(v>mx)mx=v;if(v>=T){act.push(j);fc[j]=1;if(!light){R.cls[j]++;s.cls[j]++;}}}
      sev+=mx;
      if(act.length){deg++;if(!light)for(const a of act)for(const b of act)R.co[a*C+b]++;}}
    const frac=n?deg/n:0; R.deg+=deg; s.deg+=deg;
    const keep=frac<=tol+1e-9; if(keep){R.keepDur+=f.d;R.keepFiles++;s.keepDur+=f.d;}
    if(!light){R.hist[Math.min(9,Math.floor(frac*10))]++;for(let j=0;j<C;j++)R.clsFiles[j]+=fc[j];R.files.push({f,frac,deg,fc,keep,sev:n?sev/n/255:0});}
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
  const xs=[],ys=[];for(let t=0;t<=100;t+=5){const R=compute(t/100,st.tol,st.on,true);xs.push(t/100);ys.push(R.keepDur/Math.max(R.dur,1e-9));}
  const W=560,H=220,L=44,B=34,Tp=12,pw=W-L-14,ph=H-B-Tp, X=x=>L+pw*x, Y=y=>Tp+ph-ph*y;
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  for(const y of [0,.25,.5,.75,1])s+=`<line x1="${L}" x2="${W-14}" y1="${Y(y)}" y2="${Y(y)}" stroke="var(--grid)"/><text x="${L-6}" y="${Y(y)+4}" text-anchor="end">${y*100}%</text>`;
  for(const x of [0,.2,.4,.6,.8,1])s+=`<text x="${X(x)}" y="${H-B+16}" text-anchor="middle">${x.toFixed(1)}</text>`;
  s+=`<path d="${xs.map((x,i)=>(i?'L':'M')+X(x).toFixed(1)+' '+Y(ys[i]).toFixed(1)).join(' ')}" fill="none" stroke="var(--series-1)" stroke-width="2"/>`;
  xs.forEach((x,i)=>{s+=`<circle data-tip="threshold ${x.toFixed(2)}<br><b>${pct(ys[i],1)}</b> of the audio kept" cx="${X(x)}" cy="${Y(ys[i])}" r="9" fill="transparent"/>`;});
  const cur=compute(st.thr/100,st.tol,st.on,true), cx=X(st.thr/100), cy=Y(cur.keepDur/Math.max(cur.dur,1e-9));
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
  for(let k=0;k<n;k+=step){let v=0;const labs=new Set();for(let i=k;i<Math.min(n,k+step);i++)for(let j=0;j<C;j++){if(!st.on[j])continue;const p=f.p[i*C+j]/255;if(p>v)v=p;if(p*255>=qthr(T))labs.add(PR[j]);}
    out+=`<i data-tip="${k}–${Math.min(n,k+step)} s: ${esc([...labs].join(', ')||'clean')}" style="background:${seq(v)}"></i>`;}
  return `<div class=strip>${out}</div>`;
}
function worst(R){
  const nc=x=>x.fc.reduce((s,v)=>s+v,0), best=st.sort==='best';
  const list=R.files.slice().sort(best?(a,b)=>a.frac-b.frac||a.sev-b.sev||a.f.name.localeCompare(b.f.name)
                                       :(a,b)=>b.frac-a.frac||b.sev-a.sev||nc(b)-nc(a)||a.f.name.localeCompare(b.f.name));
  const top=list.slice(0,st.rows);
  $('worst-title').textContent=best?'Cleanest files':'Most degraded files';
  $('worst-note').textContent=`${top.length} of ${list.length.toLocaleString()} files · ranked by share of degraded seconds, then by mean probability · strip = max probability per second`;
  document.querySelectorAll('#rank button').forEach(bt=>bt.classList.toggle('on',bt.dataset.s===st.sort));
  $('worst').innerHTML='<tr><th></th><th>file</th><th>folder</th><th class=num>length</th><th class=num>degraded</th><th class=num>severity</th><th>timeline</th><th>detected</th></tr>'+top.map(x=>
    `<tr class=row data-i="${x.f.i}"><td><span class=play title="open">${x.f.a?'▶':'⋯'}</span></td><td title="${esc(x.f.path)}">${esc(x.f.name)}</td><td class=muted>${esc(x.f.s)}</td><td class=num>${x.f.d.toFixed(1)} s</td><td class=num>${pct(x.frac)}</td><td class=num>${x.sev.toFixed(2)}</td><td>${strip(x.f)}</td><td>${[...Array(C).keys()].filter(j=>x.fc[j]).map(j=>`<span class=tag>${esc(PR[j])}</span>`).join('')||'<span class=muted>clean</span>'}</td></tr>`).join('');
}
// ---------------- file inspector
// ---------------- file inspector (shared by the report and the before/after comparison)
let INSP_N=0;
function inspector(el,f,o){
  // o: {T: threshold, on: [bool per class], C, PR, title}
  const id='i'+(INSP_N++), C_=o.C, PR_=o.PR, T=o.T, QT=qthr(o.T), useF=!!f.fp, hop=useF?f.fh:1, n=useF?f.fn:f.n, P=useF?f.fp:f.p, dur=Math.max(f.d,n*hop*0.999);
  const cls=[...Array(C_).keys()].filter(j=>o.on[j]);
  const W=1040, lw=170, WV=56, MT=60, MH=112, top=f.a?MT+MH+8:8, rh=22, pw=W-lw-10, H=top+rh*cls.length+28;
  const cols=Math.min(n,1500), k=n/cols, X=t=>lw+pw*t/dur;
  let s=`<svg id=${id}-svg viewBox="0 0 ${W} ${H}" width="100%" style="cursor:pointer">`;
  if(f.a)s+=`<g id=${id}-wave></g><text x="${lw-10}" y="${WV/2+4}" text-anchor="end" class="lbl">waveform</text>`+
    `<rect x="${lw}" y="${MT}" width="${pw}" height="${MH}" rx="3" fill="var(--grid)"/><g id=${id}-mel><text x="${lw+pw/2}" y="${MT+MH/2+4}" text-anchor="middle">computing mel spectrogram…</text></g>`+
    `<text x="${lw-10}" y="${MT+MH/2}" text-anchor="end" class="lbl">mel</text><text x="${lw-10}" y="${MT+MH/2+15}" text-anchor="end">spectrogram</text>`+
    `<text x="${lw-4}" y="${MT+9}" text-anchor="end" font-size="10">8k</text><text x="${lw-4}" y="${MT+MH}" text-anchor="end" font-size="10">0 Hz</text>`;
  cls.forEach((j,r)=>{const y=top+r*rh;let any=false;
    for(let c=0;c<cols;c++){let v=0;for(let i=Math.floor(c*k);i<Math.max(Math.floor((c+1)*k),Math.floor(c*k)+1);i++)v=Math.max(v,P[i*C_+j]/255);
      if(v*255>=QT)any=true; if(v>0.02)s+=`<rect x="${(lw+pw*c/cols).toFixed(2)}" y="${y+2}" width="${(pw/cols+0.4).toFixed(2)}" height="${rh-4}" fill="${seq(v)}" data-tip="${esc(PR_[j])} · ${(c*k*hop).toFixed(1)} s · p=${v.toFixed(2)}"/>`;}
    s+=`<text x="${lw-10}" y="${y+rh/2+4}" text-anchor="end" class="${any?'lbl':''}" font-weight="${any?600:400}">${esc(PR_[j])}</text><line x1="${lw}" x2="${W-10}" y1="${y+rh}" y2="${y+rh}" stroke="var(--grid)"/>`;});
  for(let t=0;t<=dur;t+=Math.max(1,Math.ceil(dur/12)))s+=`<text x="${X(t)}" y="${H-8}" text-anchor="middle">${t}s</text>`;
  s+=`<line id=${id}-ph x1="${lw}" x2="${lw}" y1="0" y2="${H-22}" stroke="var(--text-primary)" stroke-width="1.5"/></svg>`;
  const ivs=[];cls.forEach(j=>{const run=[];let a=-1,m=0;for(let i=0;i<=n;i++){const v=i<n?P[i*C_+j]/255:0;if(v*255>=QT){if(a<0)a=i;m=Math.max(m,v)}else if(a>=0){run.push([a*hop,i*hop,m]);a=-1;m=0}}
    const mg=[];for(const r of run){const l=mg[mg.length-1];if(l&&r[0]-l[1]<0.5){l[1]=r[1];l[2]=Math.max(l[2],r[2])}else mg.push([...r])}
    for(const r of mg)ivs.push([j,...r]);});
  ivs.sort((x,y)=>x[1]-y[1]);
  el.innerHTML=(o.title?`<h2 style="margin:6px 0 2px">${o.title}</h2>`:'')+
    (f.a?`<audio id=${id}-au controls preload=auto src="${esc(f.a)}" style="width:100%;margin:6px 0 4px"></audio>`:`<p class=muted>Audio is not available in this report (rebuild with <code>--audio path</code> next to the data or <code>--audio embed</code>).</p>`)+s+
    `<div style="margin-top:8px">${ivs.length?ivs.map(([j,a,b,m])=>`<span class=ivl data-t="${a}"><b>${esc(PR_[j])}</b> ${a.toFixed(1)}–${b.toFixed(1)} s <span class=muted>max ${m.toFixed(2)}</span></span>`).join(''):'<span class=muted>No degradation above the threshold.</span>'}</div>`;
  const au=document.getElementById(id+'-au'), ph=document.getElementById(id+'-ph'), svg=document.getElementById(id+'-svg');
  const seek=t=>{if(au){au.currentTime=Math.max(0,Math.min(dur,t));au.play()}};
  svg.addEventListener('click',e=>{const r=e.currentTarget.getBoundingClientRect();const x=(e.clientX-r.left)*W/r.width;if(x>=lw)seek((x-lw)/pw*dur)});
  el.querySelectorAll('.ivl').forEach(x=>x.onclick=()=>seek(+x.dataset.t));
  let raf=0;const tick=()=>{if(au){const x=X(au.currentTime);ph.setAttribute('x1',x);ph.setAttribute('x2',x)}raf=requestAnimationFrame(tick)};tick();
  if(f.a){
    let ctx; try{ctx=new (window.AudioContext||window.webkitAudioContext)({sampleRate:16000})}catch(e){ctx=new (window.AudioContext||window.webkitAudioContext)()}
    fetch(f.a).then(r=>r.arrayBuffer()).then(b=>ctx.decodeAudioData(b)).then(buf=>{
      const d=buf.getChannelData(0),sr=buf.sampleRate,px=Math.round(pw),step=Math.max(1,Math.floor(d.length/px)),mid=WV/2,amp=WV/2-4;let path='';
      let mx=1e-9;for(let i=0;i<d.length;i+=7)mx=Math.max(mx,Math.abs(d[i]));
      for(let x=0;x<px;x++){let lo=0,hi=0;for(let i=x*step;i<Math.min(d.length,(x+1)*step);i++){const v=d[i];if(v<lo)lo=v;if(v>hi)hi=v}
        path+=`M${(lw+x).toFixed(1)} ${(mid-amp*hi/mx).toFixed(1)}V${(mid-amp*lo/mx).toFixed(1)}`;}
      document.getElementById(id+'-wave').innerHTML=`<path d="${path}" stroke="var(--series-1)" stroke-width="1"/>`;
      document.getElementById(id+'-mel').innerHTML=`<image href="${melImage(d,sr,px)}" x="${lw}" y="${MT}" width="${pw}" height="${MH}" preserveAspectRatio="none"/>`;
      ctx.close&&ctx.close();
    }).catch(()=>{const g=document.getElementById(id+'-mel');if(g)g.innerHTML=`<text x="${lw+pw/2}" y="${MT+MH/2+4}" text-anchor="middle">spectrogram needs embedded audio or a web server (browsers block reading local files)</text>`;});
  }
  return {audio:au, stop(){if(au)au.pause();cancelAnimationFrame(raf);}};
}
function melImage(d,sr,cols){
  const N=512,NM=80,fmax=Math.min(8000,sr/2),hop=Math.max(1,(d.length-N)/Math.max(cols-1,1));
  const win=new Float32Array(N);for(let i=0;i<N;i++)win[i]=0.5-0.5*Math.cos(2*Math.PI*i/N);
  const m2f=m=>700*(10**(m/2595)-1),f2m=f=>2595*Math.log10(1+f/700),mp=[];for(let k=0;k<NM+2;k++)mp.push(m2f(f2m(fmax)*k/(NM+1))*N/sr);
  const re=new Float32Array(N),im=new Float32Array(N),S=new Float32Array(cols*NM);let mxdb=-1e9;
  const rev=new Uint16Array(N);for(let i=0,j=0;i<N;i++){rev[i]=j;let b=N>>1;while(j&b){j^=b;b>>=1}j|=b;}
  for(let c=0;c<cols;c++){const o=Math.floor(c*hop);
    for(let i=0;i<N;i++){re[rev[i]]=(d[o+i]||0)*win[i];im[rev[i]]=0;}
    for(let sz=2;sz<=N;sz<<=1){const h=sz>>1,a=-2*Math.PI/sz;for(let st=0;st<N;st+=sz)for(let k=0;k<h;k++){const wr=Math.cos(a*k),wi=Math.sin(a*k),i=st+k,j=i+h,tr=wr*re[j]-wi*im[j],ti=wr*im[j]+wi*re[j];re[j]=re[i]-tr;im[j]=im[i]-ti;re[i]+=tr;im[i]+=ti;}}
    for(let m=0;m<NM;m++){let e=0;const l=mp[m],ce=mp[m+1],r=mp[m+2];for(let b=Math.max(0,Math.floor(l));b<=Math.min(N/2,Math.ceil(r));b++){const w=b<ce?(b-l)/(ce-l):(r-b)/(r-ce);if(w>0)e+=w*(re[b]*re[b]+im[b]*im[b]);}
      const db=10*Math.log10(e+1e-10);S[c*NM+m]=db;if(db>mxdb)mxdb=db;}}
  const LUT=[[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],[229,80,100],[251,135,97],[254,194,135],[252,253,191]];
  const cv=document.createElement('canvas');cv.width=cols;cv.height=NM;const g=cv.getContext('2d'),img=g.createImageData(cols,NM);
  for(let c=0;c<cols;c++)for(let m=0;m<NM;m++){let v=(S[c*NM+m]-mxdb+80)/80;v=Math.max(0,Math.min(1,v));const x=v*(LUT.length-1),k=Math.min(LUT.length-2,Math.floor(x)),t=x-k;
    const p=((NM-1-m)*cols+c)*4;for(let ch=0;ch<3;ch++)img.data[p+ch]=LUT[k][ch]+(LUT[k+1][ch]-LUT[k][ch])*t;img.data[p+3]=255;}
  g.putImageData(img,0,0);return cv.toDataURL();
}
let OPEN=[];
function openFile(i){
  const f=D.files[i];
  $('m-body').innerHTML=`<div class=head><div><h2>${esc(f.name)}</h2><div class=muted>${esc(f.s)} · ${f.d.toFixed(1)} s · threshold ${(st.thr/100).toFixed(2)} · ${f.fp?'100 ms':'1 s'} resolution</div></div><button id=m-close>close ✕</button></div><div id=m-insp></div>`;
  $('modal').style.display='flex'; $('m-close').onclick=closeFile;
  OPEN=[inspector($('m-insp'),f,{T:st.thr/100,on:st.on,C,PR})];
}
function closeFile(){OPEN.forEach(h=>h.stop());OPEN=[];$('modal').style.display='none';$('m-body').innerHTML='';}
$('modal').addEventListener('click',e=>{if(e.target.id==='modal')closeFile()});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeFile()});
$('worst').addEventListener('click',e=>{const r=e.target.closest('tr.row');if(r)openFile(+r.dataset.i)});
$('find').innerHTML=D.files.map((f,i)=>`<option value="${esc(f.s+'/'+f.name)}">`).join('');
$('find-in').addEventListener('change',e=>{const v=e.target.value;const i=D.files.findIndex(f=>f.s+'/'+f.name===v);if(i>=0)openFile(i);e.target.value=''});
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
$('rank').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;st.sort=b.dataset.s;worst(LAST);});
$('rows').addEventListener('change',e=>{st.rows=+e.target.value;worst(LAST);});
const setAll=v=>{st.on=st.on.map(()=>v);document.querySelectorAll('.chip').forEach(c=>c.classList.toggle('on',v));render();};
$('all-on').addEventListener('click',()=>setAll(true)); $('all-off').addEventListener('click',()=>setAll(false));
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


def _encode_mp3(path: str, level: float = 0.8) -> str | None:
    """Audio file -> 16 kHz mono MP3 (~40 kbit/s) data URI, or None if unreadable."""
    import io

    import soundfile as sf

    from aadd.features import load_audio

    try:
        y, sr = load_audio(path)
        buf = io.BytesIO()
        sf.write(
            buf,
            np.clip(y, -1, 1),
            sr,
            format="MP3",
            subtype="MPEG_LAYER_III",
            compression_level=level,
            bitrate_mode="CONSTANT",
        )
        return "data:audio/mpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def _pack(scan: dict, audio: str = "path", embed_top: int = 40, report_dir: str | None = None) -> dict:
    import os

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
                "i": len(files),
            }
        )
        fr_ = f.get("frames")
        if fr_ and fr_.get("q"):
            files[-1].update(fq=fr_["q"], fh=float(fr_["hop_s"]), fn=int(fr_["n"]))
        if audio == "path" and os.path.exists(path):
            files[-1]["a"] = os.path.relpath(path, report_dir or os.getcwd())
        # same ranking as the "worst first" / "best first" table in the page (quantised values, all classes)
        Q = q.astype(np.float64)
        frac_ = float((Q >= round(float(scan.get("threshold", 0.5)) * 255)).any(1).mean()) if len(Q) else 0.0
        sev_ = float((Q.max(1) / 255).mean()) if len(Q) else 0.0
        ncls_ = int((Q >= round(float(scan.get("threshold", 0.5)) * 255)).any(0).sum()) if len(Q) else 0
        files[-1]["_worst"] = (-frac_, -sev_, -ncls_, files[-1]["name"])
        files[-1]["_best"] = (frac_, sev_, files[-1]["name"])
    if audio == "embed":
        # embed what a reader will click first: the top of both rankings, plus the three worst and
        # two cleanest files of every folder
        worst_ = sorted(files, key=lambda f: f["_worst"])
        best_ = sorted(files, key=lambda f: f["_best"])
        chosen = {id(f): f for f in worst_[:embed_top] + best_[:embed_top]}
        for sub in {f["s"] for f in files}:
            chosen.update(
                {id(f): f for f in [f for f in worst_ if f["s"] == sub][:3] + [f for f in best_ if f["s"] == sub][:2]}
            )
        for f in chosen.values():
            uri = _encode_mp3(f["path"])
            if uri:
                f["a"] = uri
    for f in files:
        f.pop("_worst", None)
        f.pop("_best", None)
    return {
        "classes": classes,
        "pretty": [PRETTY.get(c, c) for c in classes],
        "threshold": float(scan.get("threshold", 0.5)),
        "model": str(scan.get("model", "")),
        "errors": errors,
        "files": files,
    }


def build_report(
    scan: dict,
    title: str | None = None,
    top_files: int = 25,
    audio: str = "path",
    embed_top: int = 40,
    report_dir: str | None = None,
) -> str:
    """``audio``: "path" (relative links, works next to the data), "embed" (MP3 inside the page for the
    ``embed_top`` most degraded files and a few clean ones) or "none"."""
    data = _pack(scan, audio=audio, embed_top=embed_top, report_dir=report_dir)
    title = title or "Audio degradation report"
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    model = Path(data["model"]).name or data["model"]
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    body = f"""
<div class="viz-root">
<h1>{html.escape(title)}</h1>
<div class="sub">model <b>{html.escape(model)}</b> · {len(data["files"]):,} files · generated {now} by <code>aadd report</code></div>
<div class="controls">
 <div class="ctl">detection threshold <input id="thr" type="range" min="0" max="100" step="1"> <b id="thr-v"></b></div>
 <div class="ctl">keep a file if at most <input id="tol" type="range" min="0" max="100" step="1" value="0"> <b id="tol-v"></b> of its seconds are degraded</div>
 <div class="ctl"><button id="reset">reset</button><button id="dl">download file list (CSV)</button></div>
 <div class="ctl" style="flex-basis:100%">count as degradation: <span class="seg"><button id="all-on">select all</button><button id="all-off">clear all</button></span> <div class="chips" id="chips"></div></div>
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
<section class="card"><div class="head"><div><h2 id="worst-title">Most degraded files</h2><span class="muted" id="worst-note"></span></div>
 <div class="find"><span class="seg" id="rank"><button data-s="worst">worst first</button><button data-s="best">best first</button></span>
 <select id="rows"><option>25</option><option>50</option><option>100</option><option>250</option></select><input id="find-in" list="find" placeholder="open any file: type its name…"><datalist id="find"></datalist></div></div>
 <p class="muted" style="margin:-4px 0 8px">Click a row to listen and see its timeline.</p><table id="worst"></table></section>
<div id="modal"><div class="panel" id="m-body"></div></div><div id="tip"></div></div>"""
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}"
        f"<script>window.AADD_DATA={payload};</script><script>{JS}</script></body></html>"
    )


def write_report(
    scan_json: str, out: str, title: str | None = None, top_files: int = 25, audio: str = "path", embed_top: int = 40
) -> str:
    scan = json.loads(Path(scan_json).read_text())
    rep = build_report(
        scan,
        title=title,
        top_files=top_files,
        audio=audio,
        embed_top=embed_top,
        report_dir=str(Path(out).resolve().parent),
    )
    Path(out).write_text(rep)
    return out
