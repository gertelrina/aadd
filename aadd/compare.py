"""Before / after comparison of two scans of the same files (e.g. before and after speech enhancement).

    aadd scan raw/      --json before.json
    aadd scan cleaned/  --json after.json
    aadd compare before.json after.json -o compare.html --audio embed

Files are matched by "<folder>/<file name>". The page shows how every
degradation class changed, per folder, which files improved most and which
gained new problems, and opens both versions of a file side by side
(player, waveform, mel spectrogram, per-class timeline).
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from aadd.report import CSS, JS, _encode_mp3, _pack

CSS_EXTRA = """
.legend{display:inline-flex;gap:14px;font-size:12.5px;color:var(--text-secondary)}.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.delta-up{color:var(--critical,#d03b3b);font-weight:600}.delta-down{color:var(--good,#0ca30c);font-weight:600}
.pair{display:grid;grid-template-columns:1fr;gap:18px}.side{border-top:1px solid var(--border);padding-top:8px}
"""


def _common_js() -> str:
    """Helpers and the file inspector, taken verbatim from the report script."""
    a = JS.index("// ---------------- file inspector")
    b = JS.index("let OPEN=[];")
    helpers = r"""
const b64=q=>{const b=atob(q);const a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a};
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct=(x,d=0)=>(100*x).toFixed(d)+'%';
const qthr=t=>t>=1?256:Math.round(t*255);
const seq=v=>v<=0.001?'var(--seq-0)':`var(--seq-${Math.min(7,1+Math.floor(v*7))})`;
"""
    return helpers + JS[a:b]


JS_COMPARE = r"""
const A=window.AADD_BEFORE, B=window.AADD_AFTER, C=A.classes.length, PR=A.pretty;
for(const D of [A,B])for(const f of D.files){f.p=b64(f.q);if(f.fq)f.fp=b64(f.fq);}
const BI={};B.files.forEach((f,i)=>BI[f.s+'/'+f.name]=i);
const PAIRS=A.files.map((f,i)=>[i,BI[f.s+'/'+f.name]]).filter(p=>p[1]!==undefined);
const st={thr:Math.round(A.threshold*100),on:A.classes.map(()=>true)};
const ROWS=window.AADD_ROWS||15;
const HAS_M=PAIRS.some(([i,k])=>A.files[i].m&&B.files[k].m);
const mos=f=>f.m?f.m[0].toFixed(2):'—';
function fileStats(f,T){const n=f.n,c=new Float64Array(C);let deg=0;for(let i=0;i<n;i++){let any=false;for(let j=0;j<C;j++){if(st.on[j]&&f.p[i*C+j]>=T){c[j]++;any=true}}if(any)deg++}return {n,deg,c}}
function compute(){
  const T=qthr(st.thr/100);const R={secs:0,b:{deg:0,cls:new Float64Array(C),files:0},a:{deg:0,cls:new Float64Array(C),files:0},sub:{},pairs:[],mb:0,ma:0,mn:0};
  for(const [i,k] of PAIRS){const fb=A.files[i],fa=B.files[k],sb=fileStats(fb,T),sa=fileStats(fa,T),n=Math.min(sb.n,sa.n);
    R.secs+=n;R.b.deg+=sb.deg;R.a.deg+=sa.deg;if(sb.deg)R.b.files++;if(sa.deg)R.a.files++;
    const s=R.sub[fb.s]||(R.sub[fb.s]={secs:0,files:0,b:new Float64Array(C),a:new Float64Array(C),bd:0,ad:0,mb:0,ma:0,mn:0});s.secs+=n;s.files++;s.bd+=sb.deg;s.ad+=sa.deg;
    if(fb.m&&fa.m){R.mb+=fb.m[0];R.ma+=fa.m[0];R.mn++;s.mb+=fb.m[0];s.ma+=fa.m[0];s.mn++;}
    let newSec=0;const newCls=[];for(let j=0;j<C;j++){R.b.cls[j]+=sb.c[j];R.a.cls[j]+=sa.c[j];s.b[j]+=sb.c[j];s.a[j]+=sa.c[j];if(sa.c[j]>=1&&sb.c[j]===0){newSec+=sa.c[j];newCls.push(j)}}
    R.pairs.push({i,k,fb,fa,n,bd:sb.deg/Math.max(sb.n,1),ad:sa.deg/Math.max(sa.n,1),newSec,newCls});}
  return R;
}
function tiles(R){
  const imp=R.pairs.filter(p=>p.ad<p.bd-1e-9).length, neu=R.pairs.filter(p=>p.newSec>0).length, N=R.pairs.length;
  const t=[[pct(R.b.deg/Math.max(R.secs,1))+' → '+pct(R.a.deg/Math.max(R.secs,1)),'of audio time degraded',R.secs.toLocaleString()+' seconds compared'],
    [R.b.files.toLocaleString()+' → '+R.a.files.toLocaleString(),'files with any degradation','of '+N.toLocaleString()+' matched files'],
    [imp.toLocaleString(),'files improved',pct(imp/Math.max(N,1))+' have fewer degraded seconds'],
    [neu.toLocaleString(),'files with new problems',pct(neu/Math.max(N,1))+' gained a class absent before']];
  if(HAS_M&&R.mn)t.push([(R.mb/R.mn).toFixed(2)+' → '+(R.ma/R.mn).toFixed(2),'NISQA MOS, mean','1 bad … 5 excellent · '+R.mn.toLocaleString()+' files']);
  $('tiles').style.gridTemplateColumns=`repeat(${t.length},1fr)`;
  $('tiles').innerHTML=t.map(([v,l,d],x)=>`<div class="card tile ${x===0?'hero':''}"><div class="v" style="font-size:22px">${v}</div><div class="l">${l}</div><div class="d">${d}</div></div>`).join('');
}
function bars(R){
  const order=[...Array(C).keys()].filter(j=>st.on[j]).sort((x,y)=>Math.max(R.b.cls[y],R.a.cls[y])-Math.max(R.b.cls[x],R.a.cls[x]));
  const W=1100,lw=180,rh=34,pw=W-lw-140,H=order.length*rh+10,vmax=Math.max(1e-9,...order.map(j=>Math.max(R.b.cls[j],R.a.cls[j])/R.secs));
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  order.forEach((j,r)=>{const y=6+r*rh,vb=R.b.cls[j]/Math.max(R.secs,1),va=R.a.cls[j]/Math.max(R.secs,1),wb=vb?Math.max(2,pw*vb/vmax):0,wa=va?Math.max(2,pw*va/vmax):0,d=va-vb;
    s+=`<text class=lbl x="${lw-10}" y="${y+rh/2+3}" text-anchor="end">${esc(PR[j])}</text>`;
    s+=`<rect data-tip="<b>${esc(PR[j])}</b> before: ${pct(vb,1)}" x="${lw}" y="${y+3}" width="${wb}" height="12" rx="3" fill="var(--series-1)"/>`;
    s+=`<rect data-tip="<b>${esc(PR[j])}</b> after: ${pct(va,1)}" x="${lw}" y="${y+17}" width="${wa}" height="12" rx="3" fill="var(--series-2)"/>`;
    s+=`<text class=val x="${lw+wb+6}" y="${y+13}">${pct(vb,1)}</text><text class=val x="${lw+wa+6}" y="${y+27}">${pct(va,1)}</text>`;
    s+=`<text x="${W-10}" y="${y+rh/2+3}" text-anchor="end" font-weight="600" style="fill:${Math.abs(d)<0.005?'var(--text-muted)':d>0?'#d03b3b':'#0ca30c'}">${d>=0?'+':''}${(100*d).toFixed(1)} pt</text>`;});
  $('bars').innerHTML=s+'</svg>';
}
function subs(R){
  $('subs').innerHTML='<tr><th>folder</th><th class=num>files</th><th class=num>degraded time before</th><th class=num>after</th>'+(HAS_M?'<th class=num>NISQA MOS before → after</th>':'')+'<th>largest changes (% of time)</th></tr>'+Object.entries(R.sub).map(([k,s])=>{
    const ch=[...Array(C).keys()].filter(j=>st.on[j]).map(j=>[j,(s.a[j]-s.b[j])/Math.max(s.secs,1)]).filter(x=>Math.abs(x[1])>=0.01).sort((x,y)=>Math.abs(y[1])-Math.abs(x[1])).slice(0,4);
    return `<tr><td>${esc(k)}</td><td class=num>${s.files}</td><td class=num>${pct(s.bd/Math.max(s.secs,1))}</td><td class=num>${pct(s.ad/Math.max(s.secs,1))}</td>${HAS_M?`<td class=num style="white-space:nowrap">${s.mn?(s.mb/s.mn).toFixed(2)+' → '+(s.ma/s.mn).toFixed(2):'—'}</td>`:''}<td>${ch.map(([j,d])=>`<span class=tag>${esc(PR[j])} ${pct(s.b[j]/s.secs)} → <span class="${d>0?'delta-up':'delta-down'}">${pct(s.a[j]/s.secs)}</span></span>`).join('')||'<span class=muted>no change ≥ 1 pt</span>'}</td></tr>`}).join('');
}
function lists(R){
  const row=p=>`<tr class=row data-i="${p.i}" data-k="${p.k}"><td><span class=play>${p.fb.a&&p.fa.a?'▶':'⋯'}</span></td><td>${esc(p.fb.name)}</td><td class=muted>${esc(p.fb.s)}</td><td class=num style="white-space:nowrap">${pct(p.bd)} → ${pct(p.ad)}</td>${HAS_M?`<td class=num style="white-space:nowrap">${mos(p.fb)} → ${mos(p.fa)}</td>`:''}<td>${p.newCls.map(j=>`<span class=tag>+ ${esc(PR[j])}</span>`).join('')}</td></tr>`;
  const hd='<tr><th></th><th>file</th><th>folder</th><th class=num>degraded time</th>'+(HAS_M?'<th class=num>NISQA MOS</th>':'')+'<th>new classes after</th></tr>';
  const imp=R.pairs.slice().sort((x,y)=>(y.bd-y.ad)-(x.bd-x.ad)||x.fb.name.localeCompare(y.fb.name)).slice(0,ROWS);
  const neu=R.pairs.filter(p=>p.newSec>0).sort((x,y)=>y.newSec-x.newSec||x.fb.name.localeCompare(y.fb.name)).slice(0,ROWS);
  $('improved').innerHTML=hd+imp.map(row).join(''); $('newprob').innerHTML=hd+(neu.map(row).join('')||`<tr><td colspan=${HAS_M?6:5} class=muted>none</td></tr>`);
}
let OPEN=[];
function openPair(i,k){
  const fb=A.files[i],fa=B.files[k],o={T:st.thr/100,on:st.on,C,PR};
  $('m-body').innerHTML=`<div class=head><div><h2>${esc(fb.name)}</h2><div class=muted>${esc(fb.s)} · ${fb.d.toFixed(1)} s · threshold ${(st.thr/100).toFixed(2)}</div></div><button id=m-close>close ✕</button></div><div class=pair><div class=side id=m-b></div><div class=side id=m-a></div></div>`;
  $('modal').style.display='flex';$('m-close').onclick=closePair;
  const mt=f=>f.m?` · NISQA MOS ${f.m[0].toFixed(2)}`:'';
  OPEN=[inspector($('m-b'),fb,{...o,title:'Before'+mt(fb)}),inspector($('m-a'),fa,{...o,title:'After'+mt(fa)})];
  OPEN.forEach((h,x)=>h.audio&&h.audio.addEventListener('play',()=>OPEN[1-x].audio&&OPEN[1-x].audio.pause()));
}
function closePair(){OPEN.forEach(h=>h.stop());OPEN=[];$('modal').style.display='none';$('m-body').innerHTML='';}
$('modal').addEventListener('click',e=>{if(e.target.id==='modal')closePair()});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closePair()});
for(const id of ['improved','newprob'])$(id).addEventListener('click',e=>{const r=e.target.closest('tr.row');if(r)openPair(+r.dataset.i,+r.dataset.k)});
function render(){$('thr-v').textContent=(st.thr/100).toFixed(2);const R=compute();tiles(R);bars(R);subs(R);lists(R);}
$('chips').innerHTML=A.classes.map((c,j)=>`<label class="chip on" data-j="${j}"><span class=box></span>${esc(PR[j])}</label>`).join('');
$('chips').addEventListener('click',e=>{const ch=e.target.closest('.chip');if(!ch)return;const j=+ch.dataset.j;st.on[j]=!st.on[j];ch.classList.toggle('on',st.on[j]);render();});
const setAll=v=>{st.on=st.on.map(()=>v);document.querySelectorAll('.chip').forEach(c=>c.classList.toggle('on',v));render();};
$('all-on').addEventListener('click',()=>setAll(true));$('all-off').addEventListener('click',()=>setAll(false));
$('thr').value=st.thr;$('thr').addEventListener('input',e=>{st.thr=+e.target.value;render();});
const tip=$('tip');
document.addEventListener('mousemove',e=>{const t=e.target.closest('[data-tip]');if(!t){tip.style.display='none';return}
  tip.innerHTML=t.dataset.tip;tip.style.display='block';tip.style.left=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=Math.min(e.clientY+14,innerHeight-tip.offsetHeight-8)+'px'});
render();
"""


def _key(f: dict) -> str:
    return f"{f['s']}/{f['name']}"


def build_compare(
    before: dict,
    after: dict,
    title: str | None = None,
    audio: str = "embed",
    embed_top: int = 25,
    report_dir: str | None = None,
) -> str:
    A = _pack(before, audio="path" if audio == "path" else "none", report_dir=report_dir)
    B = _pack(after, audio="path" if audio == "path" else "none", report_dir=report_dir)
    if audio == "embed":
        # embed both versions of the files a reader will open: biggest improvements, biggest new problems,
        # and a few per folder (default threshold, all classes)
        thr = round(A["threshold"] * 255)
        C = len(A["classes"])
        bi = {_key(f): f for f in B["files"]}
        rows = []
        for fb in A["files"]:
            fa = bi.get(_key(fb))
            if fa is None:
                continue
            qb = np.frombuffer(__import__("base64").b64decode(fb["q"]), dtype=np.uint8).reshape(fb["n"], C)
            qa = np.frombuffer(__import__("base64").b64decode(fa["q"]), dtype=np.uint8).reshape(fa["n"], C)
            db = float((qb >= thr).any(1).mean()) if len(qb) else 0.0
            da = float((qa >= thr).any(1).mean()) if len(qa) else 0.0
            new = float(((qa >= thr).sum(0) * ((qb >= thr).sum(0) == 0)).sum()) if len(qa) and len(qb) else 0.0
            rows.append((fb, fa, db - da, new))
        pick = (
            sorted(rows, key=lambda r: (-r[2], r[0]["name"]))[:embed_top]
            + sorted(rows, key=lambda r: (-r[3], r[0]["name"]))[:embed_top]
        )
        for sub in {r[0]["s"] for r in rows}:
            rs = [r for r in rows if r[0]["s"] == sub]
            pick += (
                sorted(rs, key=lambda r: (-r[2], r[0]["name"]))[:3]
                + sorted(rs, key=lambda r: (-r[3], r[0]["name"]))[:3]
            )
        seen = set()
        for fb, fa, _, _ in pick:
            if _key(fb) in seen:
                continue
            seen.add(_key(fb))
            for f in (fb, fa):
                uri = _encode_mp3(f["path"])
                if uri:
                    f["a"] = uri
    title = title or "Before / after"
    pa = json.dumps(A, separators=(",", ":")).replace("</", "<\\/")
    pb = json.dumps(B, separators=(",", ":")).replace("</", "<\\/")
    body = f"""
<div class="viz-root">
<h1>{html.escape(title)}</h1>
<div class="sub">model <b>{html.escape(Path(A["model"]).name or A["model"])}</b> · {len(A["files"]):,} files before, {len(B["files"]):,} after · generated by <code>aadd compare</code></div>
<div class="controls">
 <div class="ctl">detection threshold <input id="thr" type="range" min="0" max="100" step="1"> <b id="thr-v"></b></div>
 <div class="ctl" style="flex-basis:100%">count as degradation: <span class="seg"><button id="all-on">select all</button><button id="all-off">clear all</button></span> <div class="chips" id="chips"></div></div>
</div>
<section class="grid" style="grid-template-columns:repeat(4,1fr)" id="tiles"></section>
<section class="card"><div class="head"><h2>Every degradation, before and after</h2>
 <span class="legend"><span><i style="background:var(--series-1)"></i>before</span><span><i style="background:var(--series-2)"></i>after</span><span class="muted">% of audio time · right: change in percentage points</span></span></div><div id="bars"></div></section>
<section class="card"><div class="head"><h2>By folder</h2></div><table id="subs"></table></section>
<section class="grid g2">
 <div class="card"><div class="head"><h2>Most improved files</h2><span class="muted">click to compare both versions</span></div><table id="improved"></table></div>
 <div class="card"><div class="head"><h2>Files with new problems</h2><span class="muted">a class absent before appears after</span></div><table id="newprob"></table></div>
</section>
<div id="modal"><div class="panel" id="m-body"></div></div><div id="tip"></div></div>"""
    css = CSS.replace("--series-1:#2a78d6;", "--series-1:#2a78d6;--series-2:#eb6834;").replace(
        "--series-1:#3987e5;", "--series-1:#3987e5;--series-2:#d95926;"
    )
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{css}{CSS_EXTRA}</style></head><body>{body}"
        f"<script>window.AADD_BEFORE={pa};window.AADD_AFTER={pb};window.AADD_ROWS={max(5, embed_top)};</script><script>{_common_js()}{JS_COMPARE}</script></body></html>"
    )


def write_compare(
    before_json: str, after_json: str, out: str, title: str | None = None, audio: str = "embed", embed_top: int = 25
) -> str:
    before = json.loads(Path(before_json).read_text())
    after = json.loads(Path(after_json).read_text())
    Path(out).write_text(
        build_compare(
            before, after, title=title, audio=audio, embed_top=embed_top, report_dir=str(Path(out).resolve().parent)
        )
    )
    return out
