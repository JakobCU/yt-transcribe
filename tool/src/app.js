"use strict";
/* Transkript-Checker — correction + (later) qualitative coding.
   Document model v2: speaker entities + versioned envelope + shared span anchors
   for highlights / comments / code applications. Backward-compatible with v1
   (flat segments[] + speakerColors{}) via migrate-on-load. */

const PALETTE=['#2563eb','#dc2626','#059669','#d97706','#7c3aed','#db2777','#0891b2','#65a30d','#9333ea','#e11d48','#0d9488','#ca8a04'];
const $=id=>document.getElementById(id);
const media=$('player');

/* ---------- document state (v2, all references into the live doc) ---------- */
const SCHEMA_VERSION=2;
let segments=[];           // [{type:'turn',id,time,seconds,speakerId,speaker(mirror),text,verified,edited,words?,provenance?} | {type:'divider',id,label}]
let speakerList=[];        // [{id,label,color,role,aliases[]}]
let highlights=[];         // [{id,anchor,color,note,createdBy,createdAt}]
let comments=[];           // [{id,anchor,body,createdBy,createdAt,resolved}]
let codeSystem=[];         // [{id,name,parentId,color,definition,isCodable}]
let codeApplications=[];   // [{id,codeId,anchor,selectedText,source,confidence,rationale,status,reviewer,createdBy,createdAt}]
let headerText='', transcriptId='', transcriptName='';
let docMeta={};            // {docId,rev,createdAt,updatedAt,media}
let codingCfg={mode:'inductive',codebookName:'',codebookVersion:1};  // induktiv | deduktiv | hybrid
let serverDoc=null, currentUser=null, pendingProjectId=null;  // team-server mode (null = offline/localStorage)
let pendingAudioFile=null;  // file picked for transcription → auto-loaded into the player afterwards

let activeIndex=-1, lastFollow=-1, saveTimer=null;
let followOn=true, autoRewind=true, loopOn=false, editSeek=true;
let lastCaret={i:-1,off:0};
let colorIdx=0;

/* ---------- helpers ---------- */
function clamp(v,a,b){return Math.max(a,Math.min(b,v));}
function pad(n){return String(n).padStart(2,'0');}
function fmt(sec){sec=Math.max(0,Math.floor(sec));const h=Math.floor(sec/3600),m=Math.floor(sec%3600/60),s=sec%60;return (h?h+':':'')+pad(m)+':'+pad(s);}
function toSec(t){const p=t.split(':').map(Number);return p.length===3?p[0]*3600+p[1]*60+p[2]:p[0]*60+p[1];}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function isCheck(t){return /\?\?|Audio prüfen/.test(t);}
function toast(msg){const t=$('toast');t.textContent=msg;t.classList.add('show');clearTimeout(t._t);t._t=setTimeout(()=>t.classList.remove('show'),1800);}
function nowISO(){return new Date().toISOString();}
let _uidn=0;
function uid(p){return (p||'id')+'_'+Date.now().toString(36)+(_uidn++).toString(36)+Math.floor(Math.random()*46656).toString(36);}
function nextColor(){return PALETTE[colorIdx++%PALETTE.length];}
function textHash(s){let h=5381;for(let i=0;i<s.length;i++)h=((h<<5)+h+s.charCodeAt(i))>>>0;return h.toString(36);}

function highlight(text){
  return esc(text).replace(/\[[^\]]*\]/g,m=>{
    const raw=m.replace(/&lt;/g,'<');
    if(/\?\?|Audio prüfen/.test(raw)){
      if(/mehrere Sprecher/.test(raw))return '<span class="mk multi">'+m+'</span>';
      return '<span class="mk check">'+m+'</span>';
    }
    if(/Wiederholung/.test(raw))return '<span class="mk rep">'+m+'</span>';
    return m;
  });
}

/* ---------- speaker entities ---------- */
function speakerById(id){return speakerList.find(s=>s.id===id);}
function speakerByLabel(l){return speakerList.find(s=>s.label===l);}
function ensureSpeaker(label){
  label=(label||'?').trim()||'?';
  let s=speakerByLabel(label);
  if(!s){s={id:uid('spk'),label,color:label==='UNKNOWN'?'#6b7280':nextColor(),role:'',aliases:/^SPEAKER_|^UNKNOWN$/.test(label)?[label]:[]};speakerList.push(s);}
  return s;
}
function speakerLabel(seg){const s=speakerById(seg.speakerId);return s?s.label:(seg.speaker||'?');}
function speakerColor(seg){const s=speakerById(seg.speakerId);return s?s.color:'#cbd5e1';}

/* ---------- parsing ---------- */
const TURN=/^\[(\d{2}:\d{2}:\d{2})\]\s+(.*?):\s?([\s\S]*)$/;
function isDivider(l){const t=l.trim();return /^={3,}/.test(t)||/^TEIL\s/.test(t)||/^ENDE\b/.test(t);}
function parse(raw){
  const lines=raw.split(/\r?\n/);
  const segs=[]; const header=[]; let started=false; let id=0;
  for(const line of lines){
    const m=TURN.exec(line);
    if(m){started=true;const time=m[1];segs.push({type:'turn',id:id++,time,seconds:toSec(time),speaker:m[2].trim(),text:m[3].trim(),verified:false,edited:false});continue;}
    const bareEq=/^={3,}$/.test(line.trim());
    if(isDivider(line)&&!bareEq){started=true;segs.push({type:'divider',id:id++,label:line.replace(/^=+\s*|\s*=+$/g,'').trim()});continue;}
    if(bareEq){if(!started)header.push(line);continue;}
    if(!started){header.push(line);}
    else if(line.trim()){const prev=segs[segs.length-1];if(prev&&prev.type==='turn')prev.text+=' '+line.trim();}
  }
  return {segs,header:header.join('\n').trim()};
}

/* ---------- document construction / migration ---------- */
function blankDoc(){return {schemaVersion:SCHEMA_VERSION,docId:uid('doc'),rev:0,createdAt:nowISO(),updatedAt:nowISO(),name:'',media:{},header:'',speakers:[],segments:[],speakerColors:{},highlights:[],comments:[],codeSystem:[],codeApplications:[],coding:{mode:'inductive',codebookName:'',codebookVersion:1}};}

// Build speaker entities for a set of just-parsed/migrated segments.
function attachSpeakers(segs,colors){
  colors=colors||{};
  const speakers=[]; const byLabel={}; let ci=0;
  for(const s of segs){
    if(s.type!=='turn')continue;
    const label=(s.speaker||'?').trim()||'?';
    let e=byLabel[label];
    if(!e){
      const color=colors[label]||(label==='UNKNOWN'?'#6b7280':PALETTE[ci++%PALETTE.length]);
      e={id:uid('spk'),label,color,role:'',aliases:/^SPEAKER_|^UNKNOWN$/.test(label)?[label]:[]};
      byLabel[label]=e; speakers.push(e);
    }
    s.speakerId=e.id;
    if(s.verified==null)s.verified=false;
    if(s.edited==null)s.edited=false;
  }
  return speakers;
}
function docFromParse(raw,name){
  const {segs,header}=parse(raw);
  const doc=blankDoc(); doc.name=name||''; doc.header=header; doc.segments=segs; doc.speakers=attachSpeakers(segs); return doc;
}
function migrateV1toV2(d,name,parsedHeader){
  const doc=blankDoc();
  doc.name=d.name||name||'';
  doc.segments=(d.segments||[]).map(s=>({...s}));
  doc.speakers=attachSpeakers(doc.segments,d.colors||{});
  doc.header=d.header||parsedHeader||'';
  return doc;
}
function normalizeDoc(d,name,parsedHeader){
  if(d&&d.schemaVersion>=2){
    const doc=blankDoc();
    Object.assign(doc,d);
    doc.speakers=d.speakers||[];
    doc.highlights=d.highlights||[];
    doc.comments=d.comments||[];
    doc.codeSystem=d.codeSystem||d.codes||[];
    doc.codeApplications=d.codeApplications||[];
    if(!doc.header&&parsedHeader)doc.header=parsedHeader;
    if(!doc.speakers.length)doc.speakers=attachSpeakers(doc.segments,doc.speakerColors||{});
    return doc;
  }
  return migrateV1toV2(d||{},name,parsedHeader);
}
function installDoc(doc){
  segments=doc.segments||[];
  speakerList=doc.speakers||[];
  highlights=doc.highlights||[];
  comments=doc.comments||[];
  codeSystem=doc.codeSystem||[];
  codeApplications=doc.codeApplications||[];
  headerText=doc.header||'';
  docMeta={docId:doc.docId,rev:doc.rev||0,createdAt:doc.createdAt,updatedAt:doc.updatedAt,media:doc.media||{}};
  codingCfg=doc.coding||{mode:'inductive',codebookName:'',codebookVersion:1};
  colorIdx=speakerList.length; codeColorIdx=codeSystem.length;
  // defensive: every turn must resolve to a speaker entity; keep mirror in sync
  for(const s of segments){
    if(s.type!=='turn')continue;
    if(!s.speakerId||!speakerById(s.speakerId))s.speakerId=ensureSpeaker(s.speaker||'?').id;
    s.speaker=speakerLabel(s);
  }
  activeIndex=-1; lastFollow=-1;
  render();
}
function currentDoc(){
  const speakerColors={}; speakerList.forEach(s=>speakerColors[s.label]=s.color);
  segments.forEach(s=>{if(s.type==='turn')s.speaker=speakerLabel(s);});
  return {schemaVersion:SCHEMA_VERSION,docId:docMeta.docId,rev:docMeta.rev||0,createdAt:docMeta.createdAt,updatedAt:nowISO(),
    name:transcriptName,media:docMeta.media||{},header:headerText,coding:codingCfg,
    speakers:speakerList,segments,speakerColors,highlights,comments,codeSystem,codeApplications};
}

/* ---------- span anchors (shared by highlights / comments / codes) ----------
   Quote-based with prefix/suffix context + an offset hint as fast path.
   Never trust raw offsets as the source of truth: text is freely edited. */
const ANCHOR_CTX=24;
function makeAnchor(segmentId,start,end){
  const seg=segments.find(s=>s.id===segmentId); if(!seg)return null;
  const text=seg.text||'';
  start=clamp(start,0,text.length); end=clamp(end,start,text.length);
  return {segmentId,quote:text.slice(start,end),
    prefix:text.slice(Math.max(0,start-ANCHOR_CTX),start),
    suffix:text.slice(end,end+ANCHOR_CTX),
    hint:{start,end,textHash:textHash(text)}};
}
function wholeAnchor(segmentId){return {segmentId,whole:true};}
function resolveAnchor(a){
  const seg=segments.find(s=>s.id===a.segmentId);
  if(!seg||seg.type!=='turn')return {ok:false,status:'orphaned',seg:null};
  const text=seg.text||'';
  if(a.whole)return {ok:true,status:'ok',segmentId:a.segmentId,start:0,end:text.length,seg};
  if(a.hint&&a.hint.textHash===textHash(text)&&text.slice(a.hint.start,a.hint.end)===a.quote)
    return {ok:true,status:'ok',segmentId:a.segmentId,start:a.hint.start,end:a.hint.end,seg};
  if(!a.quote)return {ok:false,status:'orphaned',segmentId:a.segmentId,seg};
  const pfx=a.prefix||'',sfx=a.suffix||'';
  let from=0,idx,first=-1,best=-1;
  while((idx=text.indexOf(a.quote,from))!==-1){
    if(first<0)first=idx;
    const pre=text.slice(Math.max(0,idx-pfx.length),idx);
    const suf=text.slice(idx+a.quote.length,idx+a.quote.length+sfx.length);
    if((!pfx||pre===pfx)&&(!sfx||suf===sfx)){best=idx;break;}
    from=idx+1;
  }
  const hit=best>=0?best:first;
  if(hit>=0)return {ok:true,status:'shifted',segmentId:a.segmentId,start:hit,end:hit+a.quote.length,seg};
  return {ok:false,status:'orphaned',segmentId:a.segmentId,seg};
}
function anchorsForSegment(segId){
  const out=[];
  highlights.forEach(h=>{if(h.anchor&&h.anchor.segmentId===segId)out.push(h.anchor);});
  comments.forEach(c=>{if(c.anchor&&c.anchor.segmentId===segId)out.push(c.anchor);});
  codeApplications.forEach(c=>{if(c.anchor&&c.anchor.segmentId===segId)out.push(c.anchor);});
  return out;
}
// After a segment's text changes, re-resolve its anchors and refresh their hints
// so drift never accumulates; orphaned ones are flagged, not dropped.
function rehintAnchorsForSegment(segId){
  for(const a of anchorsForSegment(segId)){
    if(a.whole)continue;
    const r=resolveAnchor(a);
    if(r.ok){
      a.orphaned=false;
      a.quote=r.seg.text.slice(r.start,r.end);
      a.prefix=r.seg.text.slice(Math.max(0,r.start-ANCHOR_CTX),r.start);
      a.suffix=r.seg.text.slice(r.end,r.end+ANCHOR_CTX);
      a.hint={start:r.start,end:r.end,textHash:textHash(r.seg.text)};
    }else{
      a.orphaned=true;
    }
  }
}

/* ---------- render ---------- */
function speakerOptions(selId){
  return speakerList.map(s=>`<option value="${s.id}"${s.id===selId?' selected':''}>${esc(s.label)}${s.role?' · '+esc(s.role):''}</option>`).join('')
    +'<option value="__new">＋ neuer Sprecher…</option>';
}
function render(){
  const root=$('transcript'); root.innerHTML='';
  const frag=document.createDocumentFragment();
  segments.forEach((s,i)=>{
    const el=document.createElement('div');
    if(s.type==='divider'){el.className='seg divider';el.innerHTML=`<span>${esc(s.label)}</span>`;frag.appendChild(el);return;}
    const col=speakerColor(s);
    el.className='seg turn'+(s.verified?' verified':'')+(s.edited?' edited':'')+(isCheck(s.text)&&!s.verified?' review':'');
    el.dataset.i=i; el.dataset.id=s.id; el.style.setProperty('--spk',col);
    el.innerHTML=
      `<button class="ts" data-sec="${s.seconds}" title="Audio hierher springen">⏱ ${s.time}</button>`+
      `<select class="spk">${speakerOptions(s.speakerId)}</select>`+
      `<div class="text" contenteditable="true" spellcheck="false">${renderText(s)}</div>`+
      `<div class="rowtools">`+
        `<button class="v" data-act="verify" title="geprüft (Strg+Enter)">✓</button>`+
        `<button data-act="split" title="An Cursor teilen">₥</button>`+
        `<button data-act="mergeup" title="Mit voriger Zeile verbinden">⤒</button>`+
        `<button data-act="del" title="Zeile löschen">🗑</button>`+
      `</div>`;
    frag.appendChild(el);
  });
  root.appendChild(frag);
  renderInfo(); updateProgress(); applyFilter(); updateCommentCount(); updateCodeCount(); updateSuggestCount();
}
function renderInfo(){
  const box=$('info');
  const legend=speakerList.map(s=>`<span class="lg" data-spk="${s.id}" title="Klick: umbenennen"><span class="dot" style="background:${s.color}"></span>${esc(s.label)}${s.role?` <small>${esc(s.role)}</small>`:''}</span>`).join('');
  box.innerHTML=`<details class="infobox"><summary>Info &amp; Sprecher-Legende (Klick auf einen Sprecher = umbenennen)</summary>
    <div class="legend">${legend}</div>
    ${headerText?`<pre class="hdr">${esc(headerText)}</pre>`:''}</details>`;
}

/* ---------- row updates ---------- */
function rowEl(i){return document.querySelector(`.seg[data-i="${i}"]`);}
function refreshRowClasses(i){const s=segments[i],el=rowEl(i);if(!el)return;
  el.classList.toggle('verified',!!s.verified);
  el.classList.toggle('edited',!!s.edited);
  el.classList.toggle('review',isCheck(s.text)&&!s.verified);
}
function updateProgress(){
  const turns=segments.filter(s=>s.type==='turn');
  const ver=turns.filter(s=>s.verified).length;
  const mk=turns.filter(s=>isCheck(s.text)&&!s.verified).length;
  $('progress').innerHTML=`<b>${ver}</b>/${turns.length} geprüft · <b>${mk}</b> Marker`;
}

/* ---------- active / follow ---------- */
function nextTurnSeconds(i){for(let j=i+1;j<segments.length;j++)if(segments[j].type==='turn')return segments[j].seconds;return media.duration||1e9;}
function turnAt(t){let res=-1;
  for(let i=0;i<segments.length;i++){const s=segments[i];if(s.type==='turn'&&s.seconds<=t)res=i;else if(s.type==='turn'&&s.seconds>t)break;}
  return res;}
function onTime(){
  const t=media.currentTime;
  if(loopOn&&activeIndex>=0){const end=nextTurnSeconds(activeIndex);if(t>=end||t<segments[activeIndex].seconds-0.3)media.currentTime=segments[activeIndex].seconds;}
  const idx=turnAt(t);
  if(idx!==activeIndex){
    if(activeIndex>=0&&rowEl(activeIndex))rowEl(activeIndex).classList.remove('active');
    activeIndex=idx;
    const el=rowEl(idx);
    if(el){el.classList.add('active');
      if(followOn&&idx!==lastFollow){lastFollow=idx;el.scrollIntoView({block:'center',behavior:'smooth'});}}
  }
  $('timePill').textContent=fmt(t)+' / '+fmt(media.duration||0);
}

/* ---------- editing ---------- */
function commit(el){const seg=segments[+el.closest('.seg').dataset.i];const nt=el.textContent.trim();
  if(nt!==seg.text){seg.text=nt;seg.edited=true;rehintAnchorsForSegment(seg.id);}
  const i=+el.closest('.seg').dataset.i;
  el.innerHTML=renderText(seg);refreshRowClasses(i);updateProgress();save();}
function focusText(i,toStart){const el=rowEl(i)?.querySelector('.text');if(!el)return;el.focus();
  const r=document.createRange();r.selectNodeContents(el);r.collapse(!!toStart);
  const sel=getSelection();sel.removeAllRanges();sel.addRange(r);
  if(editSeek&&media.src&&segments[i].type==='turn'){media.currentTime=segments[i].seconds;}}

document.addEventListener('focusin',e=>{const el=e.target;if(el.classList&&el.classList.contains('text')){
  const i=+el.closest('.seg').dataset.i;el.textContent=segments[i].text;}});
document.addEventListener('focusout',e=>{const el=e.target;if(el.classList&&el.classList.contains('text'))commit(el);});
document.addEventListener('selectionchange',()=>{const a=document.activeElement;if(a&&a.classList&&a.classList.contains('text')){const s=getSelection();if(s.rangeCount)lastCaret={i:+a.closest('.seg').dataset.i,off:s.anchorOffset};}});

/* ---------- click handling ---------- */
$('transcript').addEventListener('click',e=>{
  const ts=e.target.closest('.ts');if(ts){media.currentTime=+ts.dataset.sec;if(media.src)media.play();return;}
  const tb=e.target.closest('.rowtools button');if(tb){const i=+tb.closest('.seg').dataset.i;rowAction(tb.dataset.act,i);return;}
});
$('transcript').addEventListener('change',e=>{const sel=e.target.closest('select.spk');if(!sel)return;
  const i=+sel.closest('.seg').dataset.i;let v=sel.value;
  if(v==='__new'){const lbl=prompt('Name des neuen Sprechers:','');if(!lbl||!lbl.trim()){render();return;}v=ensureSpeaker(lbl.trim()).id;}
  segments[i].speakerId=v;segments[i].speaker=speakerLabel(segments[i]);render();save();});

function rowAction(act,i){const s=segments[i];
  if(act==='verify'){s.verified=!s.verified;refreshRowClasses(i);updateProgress();save();return;}
  if(act==='del'){if(confirm('Diese Zeile löschen?')){segments.splice(i,1);render();save();}return;}
  if(act==='mergeup'){let p=i-1;while(p>=0&&segments[p].type!=='turn')p--;if(p<0)return;
    segments[p].text=(segments[p].text+' '+s.text).trim();segments[p].edited=true;rehintAnchorsForSegment(segments[p].id);segments.splice(i,1);render();save();return;}
  if(act==='split'){let off=(lastCaret.i===i)?clamp(lastCaret.off,0,s.text.length):s.text.length;
    const a=s.text.slice(0,off).trim(),b=s.text.slice(off).trim();if(!b){toast('Cursor in die Zeile setzen, wo geteilt werden soll');return;}
    s.text=a;s.edited=true;rehintAnchorsForSegment(s.id);segments.splice(i+1,0,{type:'turn',id:uid('seg'),time:s.time,seconds:s.seconds,speakerId:s.speakerId,speaker:s.speaker,text:b,verified:false,edited:true});render();save();return;}
}

/* legend rename (operates on the speaker entity → updates everywhere) */
$('info').addEventListener('click',e=>{const lg=e.target.closest('.lg');if(!lg)return;
  const ent=speakerById(lg.dataset.spk);if(!ent)return;
  const nn=prompt(`„${ent.label}" umbenennen in (gilt für alle Zeilen):`,ent.label);
  if(!nn||!nn.trim()||nn.trim()===ent.label)return;
  const n=nn.trim(); const old=ent.label;
  const clash=speakerByLabel(n);
  if(clash&&clash!==ent){if(!confirm(`„${n}" existiert schon. Beide Sprecher zusammenführen?`))return;
    segments.forEach(s=>{if(s.type==='turn'&&s.speakerId===ent.id)s.speakerId=clash.id;});
    speakerList=speakerList.filter(s=>s!==ent);
  }else{ent.label=n;if(!ent.aliases.includes(old)&&/^SPEAKER_|^UNKNOWN$/.test(old))ent.aliases.push(old);}
  render();save();toast(`Umbenannt: ${old} → ${n}`);});

/* ---------- transport / shortcuts ---------- */
function togglePlay(){if(!media.src){toast('Erst Audio/Video laden');return;}media.paused?media.play():media.pause();}
function seek(d){if(media.src)media.currentTime=clamp(media.currentTime+d,0,media.duration||0);}
function setRate(d){media.playbackRate=clamp(+(media.playbackRate+d).toFixed(2),0.5,2);$('rateVal').textContent=media.playbackRate.toFixed(1)+'×';}
function curRowIndex(){const a=document.activeElement;if(a&&a.classList.contains('text'))return +a.closest('.seg').dataset.i;return activeIndex;}
function verifyAndNext(){let i=curRowIndex();if(i<0)return;const s=segments[i];if(s&&s.type==='turn'){s.verified=true;refreshRowClasses(i);updateProgress();save();}
  for(let j=i+1;j<segments.length;j++)if(segments[j].type==='turn'&&!segments[j].verified){focusText(j,true);rowEl(j)?.scrollIntoView({block:'center'});return;}}
function jumpMarker(dir){let from=curRowIndex();if(from<0)from=dir>0?-1:segments.length;
  const rng=dir>0?[...Array(segments.length).keys()].filter(j=>j>from):[...Array(segments.length).keys()].filter(j=>j<from).reverse();
  for(const j of rng){const s=segments[j];if(s.type==='turn'&&isCheck(s.text)&&!s.verified){rowEl(j)?.scrollIntoView({block:'center'});focusText(j,true);if(media.src)media.currentTime=s.seconds;return;}}
  toast('Keine weiteren offenen Marker');}
function moveEdit(dir){const i=curRowIndex();if(i<0)return;
  const rng=dir>0?[...Array(segments.length).keys()].filter(j=>j>i):[...Array(segments.length).keys()].filter(j=>j<i).reverse();
  for(const j of rng){if(segments[j].type==='turn'){focusText(j,true);rowEl(j)?.scrollIntoView({block:'center'});return;}}}
function editingField(){const a=document.activeElement;return a&&(a.isContentEditable||['INPUT','TEXTAREA','SELECT'].includes(a.tagName));}

document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){e.preventDefault();togglePlay();return;}
  if(e.ctrlKey&&e.key==='ArrowLeft'){e.preventDefault();seek(-3);return;}
  if(e.ctrlKey&&e.key==='ArrowRight'){e.preventDefault();seek(3);return;}
  if(e.ctrlKey&&e.key==='ArrowUp'){e.preventDefault();setRate(0.1);return;}
  if(e.ctrlKey&&e.key==='ArrowDown'){e.preventDefault();setRate(-0.1);return;}
  if(e.ctrlKey&&e.key==='Enter'){e.preventDefault();verifyAndNext();return;}
  if(e.key==='Enter'&&!e.ctrlKey&&!e.shiftKey&&document.activeElement?.classList.contains('text')){e.preventDefault();moveEdit(1);return;}
  if(e.ctrlKey&&(e.key==='s'||e.key==='S')){e.preventDefault();exportTxt();return;}
  if(e.altKey&&(e.key==='n'||e.key==='N')){e.preventDefault();jumpMarker(1);return;}
  if(e.altKey&&(e.key==='p'||e.key==='P')){e.preventDefault();jumpMarker(-1);return;}
  if(e.key==='Tab'&&document.activeElement?.classList.contains('text')){e.preventDefault();moveEdit(e.shiftKey?-1:1);return;}
  if(!editingField()&&(e.key==='?')){e.preventDefault();toggleHelp();return;}
});

media.addEventListener('timeupdate',onTime);
media.addEventListener('play',()=>$('playBtn').textContent='⏸');
media.addEventListener('pause',()=>{$('playBtn').textContent='▶︎';if(autoRewind&&media.currentTime>1.5)media.currentTime-=1.2;});
media.addEventListener('loadedmetadata',()=>$('timePill').textContent='0:00 / '+fmt(media.duration));

/* ---------- toolbar wiring ---------- */
$('playBtn').onclick=togglePlay;
function toggleBtn(id,setter){$(id).onclick=()=>{const v=$(id).classList.toggle('on');setter(v);};}
toggleBtn('followBtn',v=>followOn=v);
toggleBtn('rewindBtn',v=>autoRewind=v);
toggleBtn('editSeekBtn',v=>editSeek=v);
$('loopBtn').onclick=()=>{loopOn=$('loopBtn').classList.toggle('on');};
$('helpBtn').onclick=toggleHelp;$('helpClose').onclick=toggleHelp;
$('help').onclick=e=>{if(e.target.id==='help')toggleHelp();};
function toggleHelp(){$('help').classList.toggle('show');}

$('filter').onchange=applyFilter;$('search').oninput=applyFilter;
function applyFilter(){const f=$('filter').value,q=$('search').value.trim().toLowerCase();
  document.querySelectorAll('.seg.turn').forEach(el=>{const s=segments[+el.dataset.i];let show=true;
    if(f==='unverified'&&s.verified)show=false;
    if(f==='marker'&&!(isCheck(s.text)&&!s.verified))show=false;
    if(q&&!s.text.toLowerCase().includes(q)&&!speakerLabel(s).toLowerCase().includes(q))show=false;
    el.classList.toggle('hidden',!show);});}

/* ---------- file loading ---------- */
$('loadTxt').onclick=()=>pick('.txt,text/plain',f=>f.text().then(t=>loadTranscript(t,f.name)));
$('loadAud').onclick=()=>pick('audio/*,video/*',loadMedia);
function pick(accept,cb){const i=document.createElement('input');i.type='file';i.accept=accept;i.onchange=()=>{if(i.files[0])cb(i.files[0]);};i.click();}
function loadMedia(f){media.src=URL.createObjectURL(f);media.classList.toggle('audioOnly',!f.type.startsWith('video'));$('noMedia').style.display='none';media.playbackRate=clamp(media.playbackRate,0.5,2);$('rateVal').textContent=media.playbackRate.toFixed(1)+'×';toast('Geladen: '+f.name);}
function loadMediaUrl(url){media.src=url;media.classList.add('audioOnly');$('noMedia').style.display='none';media.playbackRate=clamp(media.playbackRate,0.5,2);$('rateVal').textContent=media.playbackRate.toFixed(1)+'×';}

/* drag & drop */
const drop=$('drop');let dragN=0;
window.addEventListener('dragenter',e=>{e.preventDefault();dragN++;drop.classList.add('show');});
window.addEventListener('dragover',e=>e.preventDefault());
window.addEventListener('dragleave',e=>{if(--dragN<=0)drop.classList.remove('show');});
window.addEventListener('drop',e=>{e.preventDefault();dragN=0;drop.classList.remove('show');
  for(const f of e.dataTransfer.files){if(/\.txt$/i.test(f.name)||f.type==='text/plain')f.text().then(t=>loadTranscript(t,f.name));
    else if(f.type.startsWith('audio')||f.type.startsWith('video'))loadMedia(f);}});

/* ---------- state ---------- */
let _lastRaw='';
function loadTranscript(raw,name){
  const {segs,header}=parse(raw);
  if(!segs.length){toast('Keine [HH:MM:SS] SPEAKER: Zeilen gefunden');return;}
  _lastRaw=raw;transcriptName=name||'transkript';serverDoc=null;  // local load = offline (localStorage)
  transcriptId='tc:'+transcriptName+':'+raw.length;
  let doc=null,restored=false;
  const saved=localStorage.getItem(transcriptId);
  if(saved){try{doc=normalizeDoc(JSON.parse(saved),transcriptName,header);restored=true;}catch(_){doc=null;}}
  if(!doc)doc=docFromParse(raw,transcriptName);
  installDoc(doc);
  const b=$('restoreBanner');
  if(restored){$('restoreMsg').textContent=`Gespeicherter Bearbeitungsstand von „${transcriptName}" geladen.`;b.classList.add('show');}
  else b.classList.remove('show');
  toast('Transkript geladen: '+transcriptName);
}
function save(){clearTimeout(saveTimer);saveTimer=setTimeout(()=>{
  if(serverDoc){serverSaveFlush();return;}  // team mode → server
  try{docMeta.rev=(docMeta.rev||0)+1;localStorage.setItem(transcriptId,JSON.stringify(currentDoc()));}catch(_){}
},400);}
$('resetState').onclick=()=>{if(!confirm('Bearbeitungsstand verwerfen und Original neu laden?'))return;
  localStorage.removeItem(transcriptId);loadEmbeddedOrReload();$('restoreBanner').classList.remove('show');};
function loadEmbeddedOrReload(){if(_lastRaw){installDoc(docFromParse(_lastRaw,transcriptName));save();}}

/* ---------- export ---------- */
function buildTxt(){let out=headerText?headerText+'\n\n':'';
  segments.forEach(s=>{if(s.type==='divider')out+='\n'+s.label+'\n\n';else out+=`[${s.time}] ${speakerLabel(s)}: ${s.text}\n`;});
  return out.replace(/\n{3,}/g,'\n\n').trim()+'\n';}
function download(name,text){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:'text/plain'}));a.download=name;a.click();}
function exportTxt(){download((transcriptName||'transkript').replace(/\.txt$/i,'')+'_geprueft.txt',buildTxt());toast('Export .txt');}
$('exportTxt').onclick=exportTxt;
$('moreExport').onclick=()=>{const c=['Reine Lesefassung (ohne Timestamps) .txt','SRT-Untertitel .srt','JSON-Snapshot (v2, Backup)','Snapshot laden…'];
  const ch=prompt('Export/Backup:\n1) '+c[0]+'\n2) '+c[1]+'\n3) '+c[2]+'\n4) '+c[3]+'\n\nNummer eingeben:');
  if(ch==='1'){let o='';segments.forEach(s=>{if(s.type==='turn')o+=speakerLabel(s)+': '+s.text+'\n\n';});download('lesefassung.txt',o.trim()+'\n');}
  else if(ch==='2'){let o='',n=1;const turns=segments.filter(s=>s.type==='turn');
    turns.forEach((s,k)=>{const end=k+1<turns.length?turns[k+1].seconds:s.seconds+5;o+=n++ +'\n'+srt(s.seconds)+' --> '+srt(end)+'\n'+speakerLabel(s)+': '+s.text+'\n\n';});download('untertitel.srt',o);}
  else if(ch==='3'){download((transcriptName||'transkript')+'.snapshot.json',JSON.stringify(currentDoc(),null,1));}
  else if(ch==='4'){pick('.json,application/json',f=>f.text().then(t=>{try{const d=JSON.parse(t);const doc=normalizeDoc(d,transcriptName,headerText);transcriptName=doc.name||transcriptName;installDoc(doc);save();toast('Snapshot geladen');}catch(_){toast('Ungültige Datei');}}));}
};
function srt(sec){const h=Math.floor(sec/3600),m=Math.floor(sec%3600/60),s=Math.floor(sec%60);return pad(h)+':'+pad(m)+':'+pad(s)+',000';}

/* ---------- annotations: unified span renderer ----------
   Merges markers + highlights + comments (+ codes later) into one HTML string.
   Splits the text at every span boundary so overlapping spans nest correctly.
   Invariant: output textContent === seg.text exactly (we only wrap, never add
   characters) so selection offsets map straight back to seg.text. */
const codeById=id=>codeSystem.find(c=>c.id===id);
function spanLayers(seg){
  const text=seg.text||'';
  const spans=[];
  let m; const re=/\[[^\]]*\]/g;
  while((m=re.exec(text))!==null){
    const raw=m[0]; let cls=null;
    if(/\?\?|Audio prüfen/.test(raw))cls=/mehrere Sprecher/.test(raw)?'mk multi':'mk check';
    else if(/Wiederholung/.test(raw))cls='mk rep';
    if(cls)spans.push({kind:'mk',start:m.index,end:m.index+raw.length,cls});
  }
  for(const h of highlights){if(!h.anchor||h.anchor.segmentId!==seg.id||h.anchor.whole)continue;
    const r=resolveAnchor(h.anchor);if(r.ok)spans.push({kind:'hl',start:r.start,end:r.end,color:h.color,id:h.id});}
  for(const c of comments){if(!c.anchor||c.anchor.segmentId!==seg.id||c.anchor.whole)continue;
    const r=resolveAnchor(c.anchor);if(r.ok)spans.push({kind:'cm',start:r.start,end:r.end,id:c.id});}
  for(const ca of codeApplications){if(ca.status==='rejected'||!ca.anchor||ca.anchor.segmentId!==seg.id||ca.anchor.whole)continue;
    const r=resolveAnchor(ca.anchor);if(r.ok){const code=codeById(ca.codeId);spans.push({kind:'code',start:r.start,end:r.end,color:code?code.color:'#9333ea',id:ca.id,suggested:ca.status==='suggested'});}}
  return {text,spans};
}
function renderText(seg){
  const {text,spans}=spanLayers(seg);
  if(!spans.length)return highlight(text);
  const pts=new Set([0,text.length]);
  spans.forEach(s=>{pts.add(clamp(s.start,0,text.length));pts.add(clamp(s.end,0,text.length));});
  const arr=[...pts].sort((a,b)=>a-b);
  let html='';
  for(let k=0;k<arr.length-1;k++){
    const a=arr[k],b=arr[k+1];if(a>=b)continue;
    const active=spans.filter(s=>s.start<=a&&s.end>=b);
    const slice=esc(text.slice(a,b));
    if(!active.length){html+=slice;continue;}
    const cls=[],data=[];let style='';
    const hl=active.find(s=>s.kind==='hl');if(hl){style+=`background:${hl.color};`;data.push(`data-hl="${hl.id}"`);}
    const code=active.find(s=>s.kind==='code');if(code){cls.push('codespan');if(code.suggested)cls.push('suggest');style+=`--code:${code.color};`;data.push(`data-code="${code.id}"`);}
    const cm=active.find(s=>s.kind==='cm');if(cm){cls.push('cmspan');data.push(`data-cm="${cm.id}"`);}
    const mk=active.find(s=>s.kind==='mk');if(mk)cls.push(...mk.cls.split(' '));
    html+=`<span${cls.length?` class="${cls.join(' ')}"`:''}${style?` style="${style}"`:''}${data.length?' '+data.join(' '):''}>${slice}</span>`;
  }
  return html;
}
function rerenderSegment(segId){
  const seg=segments.find(s=>s.id===segId);
  const el=document.querySelector(`.seg[data-id="${segId}"] .text`);
  if(seg&&el&&document.activeElement!==el)el.innerHTML=renderText(seg);
}

/* ---------- annotations: selection toolbar ---------- */
const HL_COLORS=[{n:'Gelb',c:'#fef08a'},{n:'Grün',c:'#bbf7d0'},{n:'Rosa',c:'#fbcfe8'},{n:'Blau',c:'#bfdbfe'},{n:'Orange',c:'#fed7aa'}];
let pendingSel=null;
const selbar=$('selbar');
selbar.innerHTML=HL_COLORS.map(h=>`<button class="sw" title="Markieren: ${h.n}" data-hlc="${h.c}" style="background:${h.c}"></button>`).join('')
  +`<span class="vsep"></span><button class="act" data-act="code" title="Code zuweisen">🏷 Code</button>`
  +`<button class="act" data-act="comment" title="Kommentar zur Auswahl">💬 Notiz</button>`
  +`<button class="act" data-act="erase" title="Markierung(en) in der Auswahl entfernen">⌫</button>`;
selbar.addEventListener('mousedown',e=>e.preventDefault()); // keep the text selection alive

function closestText(node){return (node&&(node.nodeType===3?node.parentElement:node))?.closest?.('.text')||null;}
function caretCharOffset(container,node,offset){
  if(node.nodeType!==3){ // element node: sum children up to offset
    let acc=0;for(let i=0;i<offset&&i<node.childNodes.length;i++)acc+=node.childNodes[i].textContent.length;
    node=node.childNodes[offset-1]||container;return caretCharOffset(container,container,0)+acc;
  }
  let acc=0;const w=document.createTreeWalker(container,NodeFilter.SHOW_TEXT,null);let n;
  while((n=w.nextNode())){if(n===node)return acc+offset;acc+=n.nodeValue.length;}
  return acc;
}
function currentSelectionInText(){
  const sel=getSelection();if(!sel||sel.rangeCount===0||sel.isCollapsed)return null;
  const r=sel.getRangeAt(0);
  const startEl=closestText(r.startContainer),endEl=closestText(r.endContainer);
  if(!startEl||startEl!==endEl)return null;
  let start=caretCharOffset(startEl,r.startContainer,r.startOffset);
  let end=caretCharOffset(startEl,r.endContainer,r.endOffset);
  if(start>end)[start,end]=[end,start];
  if(start===end)return null;
  const seg=segments[+startEl.closest('.seg').dataset.i];
  return {segId:seg.id,start,end,rect:r.getBoundingClientRect()};
}
function showSelbar(s){pendingSel={segId:s.segId,start:s.start,end:s.end};
  selbar.classList.add('show');
  const bw=selbar.offsetWidth||220,bh=selbar.offsetHeight||34;
  let left=s.rect.left+window.scrollX+s.rect.width/2-bw/2;
  left=clamp(left,8+window.scrollX,window.scrollX+document.documentElement.clientWidth-bw-8);
  let top=s.rect.top+window.scrollY-bh-8;if(top<window.scrollY+4)top=s.rect.bottom+window.scrollY+8;
  selbar.style.left=left+'px';selbar.style.top=top+'px';}
function hideSelbar(){selbar.classList.remove('show');pendingSel=null;}
document.addEventListener('mouseup',()=>{const s=currentSelectionInText();if(s)showSelbar(s);else if(!($('codepick')&&$('codepick').classList.contains('show')))hideSelbar();});
document.addEventListener('mousedown',e=>{const cp=$('codepick');if(!selbar.contains(e.target)&&!(cp&&cp.contains(e.target)))hideSelbar();});
window.addEventListener('scroll',()=>{if(!($('codepick')&&$('codepick').classList.contains('show')))hideSelbar();},true);

selbar.addEventListener('click',e=>{
  const sw=e.target.closest('[data-hlc]');if(sw){addHighlight(sw.dataset.hlc);return;}
  const act=e.target.closest('[data-act]');if(!act)return;
  if(act.dataset.act==='comment')addComment();
  else if(act.dataset.act==='erase')eraseHighlights();
  else if(act.dataset.act==='code')openCodePick();
});
function addHighlight(color){if(!pendingSel)return;const a=makeAnchor(pendingSel.segId,pendingSel.start,pendingSel.end);if(!a)return;
  highlights.push({id:uid('hl'),anchor:a,color,note:'',createdBy:'local',createdAt:nowISO()});
  rerenderSegment(pendingSel.segId);getSelection().removeAllRanges();hideSelbar();save();toast('Markiert');}
function eraseHighlights(){if(!pendingSel)return;const{segId,start,end}=pendingSel;let n=0;
  highlights=highlights.filter(h=>{if(h.anchor.segmentId!==segId)return true;const r=resolveAnchor(h.anchor);if(r.ok&&r.start<end&&r.end>start){n++;return false;}return true;});
  rerenderSegment(segId);getSelection().removeAllRanges();hideSelbar();save();toast(n?`${n} Markierung(en) entfernt`:'Keine Markierung in der Auswahl');}
function addComment(){if(!pendingSel)return;const body=prompt('Kommentar / Notiz:','');if(body==null){hideSelbar();return;}
  const a=makeAnchor(pendingSel.segId,pendingSel.start,pendingSel.end);
  comments.push({id:uid('cm'),anchor:a,body:body.trim(),createdBy:'local',createdAt:nowISO(),resolved:false});
  rerenderSegment(pendingSel.segId);getSelection().removeAllRanges();hideSelbar();updateCommentCount();save();
  if($('cpanel').classList.contains('show'))renderComments();toast('Kommentar hinzugefügt');}

/* ---------- comments panel ---------- */
function updateCommentCount(){const open=comments.filter(c=>!c.resolved).length;$('cCount').textContent=open;}
function flashSegment(segId){const el=document.querySelector(`.seg[data-id="${segId}"]`);if(!el)return;
  el.scrollIntoView({block:'center',behavior:'smooth'});el.classList.remove('flash');void el.offsetWidth;el.classList.add('flash');}
function commentSnippet(c){const r=resolveAnchor(c.anchor);
  if(c.anchor.whole)return {text:'(ganze Zeile)',orphan:false};
  if(r.ok)return {text:'„'+r.seg.text.slice(r.start,r.end)+'"',orphan:false};
  return {text:'„'+(c.anchor.quote||'?')+'" (Stelle nicht mehr gefunden)',orphan:true};}
function renderComments(){const list=$('clist');
  if(!comments.length){list.innerHTML='<div class="cempty">Noch keine Kommentare.<br>Text markieren → 💬 Notiz.</div>';return;}
  const ordered=[...comments].sort((a,b)=>{const sa=segments.findIndex(s=>s.id===a.anchor.segmentId),sb=segments.findIndex(s=>s.id===b.anchor.segmentId);return sa-sb;});
  list.innerHTML=ordered.map(c=>{const sn=commentSnippet(c);const seg=segments.find(s=>s.id===c.anchor.segmentId);
    return `<div class="ccard${c.resolved?' resolved':''}" data-cid="${c.id}">
      <div class="cquote${sn.orphan?' orphan':''}" data-go>${seg?'<b>'+esc(seg.time||'')+'</b> ':''}${esc(sn.text)}</div>
      <div class="cbody">${esc(c.body)||'<i>(leer)</i>'}</div>
      <div class="cmeta"><button data-cact="resolve">${c.resolved?'wieder öffnen':'erledigt'}</button>
        <button data-cact="edit">bearbeiten</button><button data-cact="del">löschen</button></div>
    </div>`;}).join('');
}
$('clist').addEventListener('click',e=>{const card=e.target.closest('.ccard');if(!card)return;const id=card.dataset.cid;const c=comments.find(x=>x.id===id);if(!c)return;
  if(e.target.closest('[data-go]')){flashSegment(c.anchor.segmentId);return;}
  const act=e.target.closest('[data-cact]');if(!act)return;
  if(act.dataset.cact==='resolve'){c.resolved=!c.resolved;}
  else if(act.dataset.cact==='edit'){const b=prompt('Kommentar bearbeiten:',c.body);if(b!=null)c.body=b.trim();}
  else if(act.dataset.cact==='del'){if(!confirm('Kommentar löschen?'))return;comments=comments.filter(x=>x.id!==id);rerenderSegment(c.anchor.segmentId);}
  renderComments();updateCommentCount();save();});
$('commentsBtn').onclick=()=>{const p=$('cpanel');const show=!p.classList.contains('show');$('codepanel').classList.remove('show');p.classList.toggle('show',show);if(show)renderComments();};
$('cpanelClose').onclick=()=>$('cpanel').classList.remove('show');

/* ---------- qualitative coding ----------
   Modes: inductive (codes emerge from the data), deductive (fixed categories
   from a codebook), hybrid (both). The data model is identical across modes;
   the mode tunes the workflow — in deductive mode, codes created ad-hoc are
   marked 'provisional' so the fixed codebook stays clean until promoted. */
const CODE_PALETTE=['#9333ea','#dc2626','#0891b2','#65a30d','#d97706','#db2777','#2563eb','#0d9488','#e11d48','#7c3aed','#ca8a04','#059669'];
let codeColorIdx=0;
let retrievedCodeId=null;
function nextCodeColor(){return CODE_PALETTE[codeColorIdx++%CODE_PALETTE.length];}
function appsForCode(id){return codeApplications.filter(a=>a.codeId===id&&a.status!=='rejected');}
function codeChildren(parentId){return codeSystem.filter(c=>(c.parentId||null)===(parentId||null));}
function createCode({name,color,parentId,definition,id}={}){
  const code={id:id||uid('code'),name:(name||'Neuer Code').trim(),parentId:parentId||null,color:color||nextCodeColor(),
    definition:definition||'',inclusion:'',exclusion:'',examples:[],isCodable:true,
    provisional:codingCfg.mode==='deductive',createdBy:'local',createdAt:nowISO()};
  codeSystem.push(code);return code;
}
function deleteCode(id){const ids=new Set();(function collect(cid){ids.add(cid);codeSystem.filter(c=>c.parentId===cid).forEach(c=>collect(c.id));})(id);
  codeSystem=codeSystem.filter(c=>!ids.has(c.id));codeApplications=codeApplications.filter(a=>!ids.has(a.codeId));}
function applyCodeToSelection(codeId){
  if(!pendingSel)return;const a=makeAnchor(pendingSel.segId,pendingSel.start,pendingSel.end);if(!a)return;
  const segId=pendingSel.segId;
  const dup=codeApplications.find(x=>x.codeId===codeId&&x.status!=='rejected'&&x.anchor.segmentId===segId&&Math.abs((x.anchor.hint?.start??-9)-a.hint.start)<2&&Math.abs((x.anchor.hint?.end??-9)-a.hint.end)<2);
  if(!dup)codeApplications.push({id:uid('ca'),codeId,anchor:a,selectedText:a.quote,source:'human',confidence:null,rationale:'',status:'accepted',reviewer:'local',createdBy:'local',createdAt:nowISO()});
  getSelection().removeAllRanges();hideCodePick();hideSelbar();rerenderSegment(segId);save();renderCodes();
  if(retrievedCodeId===codeId)retrieveCode(codeId);
  toast(dup?'Schon kodiert':'Kodiert: '+(codeById(codeId)?.name||''));
}
function removeApplication(appId){const ca=codeApplications.find(a=>a.id===appId);if(!ca)return;const seg=ca.anchor.segmentId;
  codeApplications=codeApplications.filter(a=>a.id!==appId);rerenderSegment(seg);save();}

/* code picker popover (opened from the selection toolbar) */
const codepick=document.createElement('div');codepick.id='codepick';document.body.appendChild(codepick);
codepick.addEventListener('mousedown',e=>{if(e.target.tagName!=='INPUT')e.preventDefault();});
function openCodePick(){
  if(!pendingSel)return;
  codepick.innerHTML='<input type="text" placeholder="Code suchen oder neu anlegen…"><div class="pklist"></div>';
  const inp=codepick.querySelector('input');
  inp.addEventListener('input',()=>fillCodePick(inp.value));
  inp.addEventListener('keydown',codePickKey);
  fillCodePick('');
  codepick.classList.add('show');
  const bb=selbar.getBoundingClientRect();
  let left=clamp(bb.left,8,document.documentElement.clientWidth-270)+window.scrollX;
  codepick.style.left=left+'px';codepick.style.top=(window.scrollY+bb.bottom+6)+'px';
  inp.focus();
}
function hideCodePick(){codepick.classList.remove('show');}
function fillCodePick(q){
  q=(q||'').trim();const ql=q.toLowerCase();
  const codes=codeSystem.filter(c=>c.isCodable!==false&&(!ql||c.name.toLowerCase().includes(ql)));
  const list=codepick.querySelector('.pklist');
  list.innerHTML=codes.map((c,i)=>`<div class="pkrow${i===0?' sel':''}" data-cid="${c.id}"><span class="cdot" style="background:${c.color}"></span>${esc(c.name)}${c.provisional?' <span class="prov" style="color:var(--review);font-size:11px">(Vorschlag)</span>':''}</div>`).join('')
    +`<div class="pkrow pknew" data-new="1">＋ Neuer Code${q?' „'+esc(q)+'"':' aus Auswahl'}</div>`;
}
function codePickKey(e){
  const rows=[...codepick.querySelectorAll('.pkrow')];let si=rows.findIndex(r=>r.classList.contains('sel'));if(si<0)si=0;
  if(e.key==='ArrowDown'){e.preventDefault();rows[si]?.classList.remove('sel');rows[Math.min(si+1,rows.length-1)]?.classList.add('sel');rows[Math.min(si+1,rows.length-1)]?.scrollIntoView({block:'nearest'});}
  else if(e.key==='ArrowUp'){e.preventDefault();rows[si]?.classList.remove('sel');rows[Math.max(si-1,0)]?.classList.add('sel');}
  else if(e.key==='Enter'){e.preventDefault();(rows.find(r=>r.classList.contains('sel'))||rows[0])?.click();}
  else if(e.key==='Escape'){e.preventDefault();hideCodePick();}
}
codepick.addEventListener('click',e=>{
  const row=e.target.closest('.pkrow');if(!row)return;
  if(row.dataset.new){const q=codepick.querySelector('input').value.trim();const nm=q||prompt('Name des neuen Codes:','');if(!nm||!nm.trim())return;const c=createCode({name:nm.trim()});applyCodeToSelection(c.id);return;}
  applyCodeToSelection(row.dataset.cid);
});

/* codes side panel */
function updateCodeCount(){const el=$('codeCount');if(el)el.textContent=codeSystem.filter(c=>c.isCodable!==false).length;}
function renderCodeNode(c){
  const kids=codeChildren(c.id).filter(k=>!k.provisional);
  let html=`<div class="cnode${retrievedCodeId===c.id?' active':''}" data-code="${c.id}" title="${esc(c.definition||'(keine Definition)')}">`+
    `<span class="cdot" style="background:${c.color}"></span>`+
    `<span class="cname">${esc(c.name)}${c.provisional?'<span class="prov"> Vorschlag</span>':''}</span>`+
    `<span class="ccount">${appsForCode(c.id).length}</span>`+
    `<span class="ctools">`+
      `<button data-cact="retrieve" title="Alle Stellen anzeigen">↳</button>`+
      `<button data-cact="addsub" title="Unter-Code">＋</button>`+
      `<button data-cact="edit" title="Bearbeiten">✎</button>`+
      `<button data-cact="del" title="Löschen">🗑</button>`+
    `</span></div>`;
  if(kids.length)html+=`<div class="csub">${kids.map(renderCodeNode).join('')}</div>`;
  return html;
}
function renderCodes(){
  const ms=$('codeMode');if(ms)ms.value=codingCfg.mode||'inductive';
  refreshLLMUI();
  const tree=$('ctree');if(!tree)return;
  if(!codeSystem.length){tree.innerHTML='<div class="cempty">Noch keine Codes.<br>Text markieren → 🏷 Code, „＋ Code", oder ein Codebook laden.</div>';updateCodeCount();return;}
  const roots=codeChildren(null).filter(c=>!c.provisional);
  const prov=codeSystem.filter(c=>c.provisional);
  let html=roots.map(renderCodeNode).join('');
  if(prov.length)html+=`<div class="cprovhdr">Vorschläge / emergent (${prov.length})</div>`+prov.map(renderCodeNode).join('');
  tree.innerHTML=html;updateCodeCount();
}
function editCode(c){
  const name=prompt('Code-Name:',c.name);if(name==null)return;if(name.trim())c.name=name.trim();
  const def=prompt('Definition (wann anwenden?):',c.definition||'');if(def!=null)c.definition=def.trim();
  const col=prompt('Farbe (Hex, z.B. #9333ea):',c.color);if(col&&/^#?[0-9a-fA-F]{6}$/.test(col.trim())){c.color=col.trim().startsWith('#')?col.trim():'#'+col.trim();}
  if(c.provisional&&confirm('Diesen Vorschlag fest ins Codebook übernehmen?'))c.provisional=false;
  renderCodes();render();save();if(retrievedCodeId===c.id)retrieveCode(c.id);
}
function retrieveCode(id){
  const c=codeById(id);if(!c)return;retrievedCodeId=id;
  document.querySelectorAll('#ctree .cnode').forEach(n=>n.classList.toggle('active',n.dataset.code===id));
  const apps=appsForCode(id);const box=$('cretrieve');
  let html=`<div class="rhdr"><span class="cdot" style="background:${c.color};width:12px;height:12px;border-radius:4px"></span> „${esc(c.name)}" · ${apps.length} Stelle(n)<button class="btn" id="rclose" style="margin-left:auto;padding:2px 8px">✕</button></div>`;
  if(c.definition)html+=`<div style="font-size:11px;color:var(--muted);margin-bottom:6px">${esc(c.definition)}</div>`;
  html+=apps.map(a=>{const r=resolveAnchor(a.anchor);const seg=segments.find(s=>s.id===a.anchor.segmentId);
    const text=r.ok&&seg?seg.text.slice(r.start,r.end):(a.selectedText||'?');const sp=seg?speakerLabel(seg):'';
    return `<div class="ritem" data-app="${a.id}" data-seg="${a.anchor.segmentId}" style="--rcol:${c.color}">`+
      `<button class="rx" data-rx title="Kodierung entfernen">✕</button>`+
      `<span class="rsrc ${a.source==='llm'?'llm':'human'}">${a.source==='llm'?'KI':'manuell'}</span> `+
      `<span class="rmeta">${esc(seg?seg.time:'')} · ${esc(sp)}</span><br>„${esc(text)}"${r.ok?'':' <span style="color:#dc2626">(verschoben)</span>'}</div>`;
  }).join('')||'<div style="color:var(--muted);font-size:12px">Noch keine Kodierungen.</div>';
  box.innerHTML=html;box.classList.add('show');
}
$('ctree').addEventListener('click',e=>{
  const node=e.target.closest('.cnode');if(!node)return;const id=node.dataset.code;const c=codeById(id);if(!c)return;
  const act=e.target.closest('[data-cact]');
  if(act){const a=act.dataset.cact;
    if(a==='del'){if(confirm(`Code „${c.name}" samt Unter-Codes und allen Kodierungen löschen?`)){deleteCode(id);if(retrievedCodeId===id){retrievedCodeId=null;$('cretrieve').classList.remove('show');}renderCodes();render();save();}return;}
    if(a==='addsub'){const nm=prompt('Name des Unter-Codes:','');if(nm&&nm.trim()){createCode({name:nm.trim(),parentId:id});renderCodes();save();}return;}
    if(a==='edit'){editCode(c);return;}
    if(a==='retrieve'){retrieveCode(id);return;}}
  retrieveCode(id);
});
$('cretrieve').addEventListener('click',e=>{
  if(e.target.id==='rclose'){$('cretrieve').classList.remove('show');retrievedCodeId=null;document.querySelectorAll('#ctree .cnode.active').forEach(n=>n.classList.remove('active'));return;}
  if(e.target.id==='accAll'){codeApplications.forEach(a=>{if(a.status==='suggested'){a.status='accepted';a.reviewer='local';}});render();renderCodes();updateSuggestCount();renderReview();save();return;}
  const rev=e.target.closest('[data-rev]');
  if(rev){const it=e.target.closest('.ritem');const app=codeApplications.find(a=>a.id===it.dataset.app);if(!app)return;
    if(rev.dataset.rev==='goto'){flashSegment(app.anchor.segmentId);return;}
    app.status=rev.dataset.rev==='accept'?'accepted':'rejected';app.reviewer='local';
    rerenderSegment(app.anchor.segmentId);renderCodes();updateSuggestCount();renderReview();save();return;}
  const item=e.target.closest('.ritem');if(!item)return;
  if(e.target.closest('[data-rx]')){removeApplication(item.dataset.app);renderCodes();if(retrievedCodeId)retrieveCode(retrievedCodeId);return;}
  flashSegment(item.dataset.seg);
});
$('codesBtn').onclick=()=>{const p=$('codepanel');const show=!p.classList.contains('show');$('cpanel').classList.remove('show');p.classList.toggle('show',show);if(show)renderCodes();};
$('codepanelClose').onclick=()=>$('codepanel').classList.remove('show');
$('codeMode').onchange=()=>{codingCfg.mode=$('codeMode').value;save();toast('Modus: '+$('codeMode').selectedOptions[0].text);};
$('newCodeBtn').onclick=()=>{const nm=prompt('Name des neuen Codes:','');if(!nm||!nm.trim())return;createCode({name:nm.trim()});renderCodes();save();};

/* codebook YAML import / export — the editable "how to code" instruction file */
function codebookToYaml(){
  function node(c){const o={id:c.id,name:c.name,color:c.color};
    if(c.definition)o.definition=c.definition;if(c.inclusion)o.inclusion=c.inclusion;if(c.exclusion)o.exclusion=c.exclusion;
    if(c.examples&&c.examples.length)o.examples=c.examples;
    const kids=codeChildren(c.id);if(kids.length)o.children=kids.map(node);return o;}
  const data={codebook:{name:codingCfg.codebookName||transcriptName||'Codebook',version:codingCfg.codebookVersion||1,mode:codingCfg.mode||'inductive',codes:codeChildren(null).map(node)}};
  return jsyaml.dump(data,{lineWidth:100,noRefs:true});
}
function importCodebook(cb){
  if(cb.mode&&['inductive','deductive','hybrid'].includes(cb.mode))codingCfg.mode=cb.mode;
  if(cb.name)codingCfg.codebookName=cb.name;if(cb.version)codingCfg.codebookVersion=cb.version;
  (function add(nodes,parentId){(nodes||[]).forEach(n=>{
    const id=n.id||uid('code');let c=codeById(id);
    if(!c){c={id,name:n.name||id,parentId:parentId||null,color:n.color||nextCodeColor(),definition:n.definition||'',inclusion:n.inclusion||'',exclusion:n.exclusion||'',examples:n.examples||[],isCodable:true,provisional:false,createdBy:'codebook',createdAt:nowISO()};codeSystem.push(c);}
    else{c.name=n.name||c.name;c.parentId=parentId||null;if(n.color)c.color=n.color;c.definition=n.definition||'';c.inclusion=n.inclusion||'';c.exclusion=n.exclusion||'';c.examples=n.examples||[];c.provisional=false;}
    add(n.children,id);});})(cb.codes,null);
  renderCodes();render();save();
}
$('cbSave').onclick=()=>{if(!codeSystem.length){toast('Noch keine Codes zum Exportieren');return;}download((codingCfg.codebookName||'codebook').replace(/[^\w.-]+/g,'_')+'.yaml',codebookToYaml());toast('Codebook exportiert');};
$('cbLoad').onclick=()=>pick('.yaml,.yml,text/yaml,application/x-yaml,text/plain',f=>f.text().then(t=>{
  let data;try{data=jsyaml.load(t);}catch(err){toast('YAML-Fehler: '+err.message);return;}
  const cb=data&&data.codebook?data.codebook:data;
  if(!cb||!Array.isArray(cb.codes)){toast('Kein gültiges Codebook (codebook.codes fehlt)');return;}
  importCodebook(cb);toast('Codebook geladen: '+(cb.name||(cb.codes.length+' Codes')));
}));

/* ---------- backend / server mode ----------
   When the tool is served by the FastAPI backend, /api/health responds and we
   expose the "Transkribieren" button: upload audio → pipeline job → progress →
   the finished transcript loads straight into the tool. In offline single-file
   mode there is no backend, the probe fails quietly, and the button stays hidden. */
const backend={available:false,diarization:false,fake:false};
const STAGE_LABEL={queued:'In Warteschlange',starting:'Startet…',convert:'Audio vorbereiten…',
  transcribe:'Transkription',diarize:'Sprecher-Erkennung…',merge:'Zusammenführen…',store:'Audio speichern…',code:'Kodiere…',done:'Fertig'};
function fmtDur(s){s=Math.max(0,Math.round(s));const m=Math.floor(s/60),ss=s%60;return m?m+':'+String(ss).padStart(2,'0')+' min':ss+' s';}
function showJobBanner(msg,frac,done){const b=$('jobBanner'),fill=$('jobFill');$('jobMsg').textContent=msg;
  if(frac==null){b.classList.add('indet');fill.style.width='';}
  else{b.classList.remove('indet');fill.style.width=Math.round(clamp(frac,0,1)*100)+'%';}
  b.classList.add('show');$('jobCancel').style.display=done?'':'none';}
function hideJobBanner(){$('jobBanner').classList.remove('show');}
let activeJobFinish=null;  // closes the current job's SSE/polling
$('jobCancel').onclick=()=>{if(activeJobFinish)activeJobFinish();hideJobBanner();};

async function initBackend(){
  try{
    const r=await fetch('/api/health',{cache:'no-store'});if(!r.ok)return;
    const h=await r.json();
    backend.available=!!h.ok;backend.diarization=!!h.diarizationAvailable;backend.fake=!!h.fake;backend.llm=h.llm||{};backend.auth=h.auth||{};
    if(!backend.available)return;
    let me=null;
    try{const mr=await fetch('/api/auth/me',{cache:'no-store'});if(mr.ok)me=await mr.json();}catch(_){}
    if(me)enterServerMode(me); else showAuthGate();
  }catch(_){/* offline mode: no backend */}
}
function openTxModal(){
  const diar=$('txDiar'),note=$('txDiarNote');
  if(!backend.diarization){diar.checked=false;diar.disabled=true;note.textContent='(kein HF-Token am Server — nur Transkription)';}
  else{diar.disabled=false;note.textContent='';}
  if(backend.fake)note.textContent='(Server im Test-Modus — liefert ein Fake-Transkript)';
  $('txModal').classList.add('show');
}
$('transcribeBtn').onclick=openTxModal;
function closeTxModal(){$('txModal').classList.remove('show');}
$('txCancel').onclick=closeTxModal;
$('txModal').addEventListener('click',e=>{if(e.target.id==='txModal')closeTxModal();});
$('txStart').onclick=async()=>{
  const f=$('txFile').files[0];if(!f){toast('Bitte eine Audio-/Videodatei wählen');return;}
  pendingAudioFile=f;  // keep it to auto-load into the player once transcription finishes
  const fd=new FormData();fd.append('audio',f);fd.append('model',$('txModel').value);
  fd.append('language',$('txLang').value.trim());fd.append('diarize',$('txDiar').checked?'true':'false');
  fd.append('device',$('txDevice').value);
  if(pendingProjectId)fd.append('project_id',pendingProjectId);
  closeTxModal();showJobBanner('Lade „'+f.name+'" hoch…',0.02,false);
  let jobId;
  try{const r=await fetch('/api/transcribe',{method:'POST',body:fd});if(!r.ok)throw new Error('Upload fehlgeschlagen ('+r.status+')');jobId=(await r.json()).job_id;}
  catch(err){showJobBanner('Fehler: '+err.message,0,true);return;}
  trackJob(jobId,f.name);
};
function trackJob(jobId,fname,onDone){
  if(activeJobFinish)activeJobFinish();  // close any previous job's stream first
  let es=null,polling=null,finished=false,txStart=0;
  const finish=()=>{finished=true;if(es){es.onmessage=null;es.close();}if(polling)clearInterval(polling);if(activeJobFinish===finish)activeJobFinish=null;};
  activeJobFinish=finish;
  const onUpdate=(job)=>{
    if(finished)return;
    if(job.error){finish();showJobBanner('Fehler: '+job.error,0,true);return;}
    if(job.status==='done'){finish();(onDone||(()=>loadJobResult(jobId,fname)))();return;}
    if(job.stage==='transcribe'&&job.progress>0){
      if(!txStart)txStart=Date.now();
      const el=(Date.now()-txStart)/1000;let eta='';
      if(job.progress>0.03&&el>2)eta=' · noch '+fmtDur(el/job.progress*(1-job.progress));
      showJobBanner('„'+fname+'" — Transkription '+Math.round(job.progress*100)+'%'+eta,job.progress,false);
    }else{
      showJobBanner('„'+fname+'" — '+(STAGE_LABEL[job.stage]||job.stage),null,false);  // indeterminate
    }
  };
  const startPolling=()=>{if(polling)return;polling=setInterval(async()=>{try{const r=await fetch('/api/jobs/'+jobId,{cache:'no-store'});if(r.ok)onUpdate(await r.json());}catch(_){}}, 1200);};
  try{
    es=new EventSource('/api/jobs/'+jobId+'/events');
    es.onmessage=ev=>{try{onUpdate(JSON.parse(ev.data));}catch(_){}};
    es.onerror=()=>{if(finished)return;if(es){es.onmessage=null;es.close();es=null;}startPolling();};
  }catch(_){startPolling();}
}
async function loadJobResult(jobId,fname){
  try{
    const r=await fetch('/api/jobs/'+jobId+'/result');if(!r.ok)throw new Error('Ergebnis nicht abrufbar ('+r.status+')');
    const res=await r.json();
    if(res.document_id){hideJobBanner();pendingProjectId=null;await loadProjects();await openServerDocument(res.document_id);
      if(pendingAudioFile){loadMedia(pendingAudioFile);pendingAudioFile=null;}
      toast('Transkript erstellt'+(res.device?' · '+res.device.toUpperCase():'')+(res.diarized?' · mit Sprechern':''));return;}
    if(!res||!res.text)throw new Error('Leeres Ergebnis');
    showJobBanner('Fertig — lade Transkript…',1,false);
    const name=(res.name||fname||'transkript').replace(/\.[^.]+$/,'')+'.txt';
    loadTranscript(res.text,name);hideJobBanner();
    if(pendingAudioFile){loadMedia(pendingAudioFile);pendingAudioFile=null;}
    toast('Transkript geladen'+(res.device?' · '+res.device.toUpperCase():'')+(res.diarized?' · mit Sprechern':' · ohne Diarization'));
  }catch(err){showJobBanner('Fehler beim Laden: '+err.message,0,true);}
}
initBackend();

/* ---------- LLM-assisted coding (frontend) ----------
   Sends the current codebook + transcript to the backend, which codes each
   segment deductively. Results come back as status='suggested' applications that
   the human reviews (anti-anchoring: the review card leads with the quote +
   rationale, not just the code label). Nothing is auto-accepted. */
function refreshLLMUI(){
  const row=$('cpllm');if(!row)return;
  if(!backend.available){row.style.display='none';return;}  // offline: no LLM backend
  row.style.display='';
  const provs=backend.llm?Object.entries(backend.llm).filter(([,v])=>v&&v.available):[];
  const sel=$('llmProvider'),btn=$('llmCodeBtn'),hint=$('llmHint');
  if(provs.length){
    sel.style.display='';btn.disabled=false;hint.style.display='none';
    btn.title='Transkript automatisch mit dem Codebook kodieren';
    if(sel.dataset.n!==String(provs.length)){
      sel.innerHTML=provs.map(([k,v])=>`<option value="${k}">${esc(v.label||k)}</option>`).join('');
      sel.dataset.n=String(provs.length);
    }
  }else{
    sel.style.display='none';sel.dataset.n='';btn.disabled=true;
    btn.title='Kein LLM konfiguriert';
    hint.style.display='';
    hint.textContent='Kein LLM aktiv — ANTHROPIC_API_KEY in .env setzen (Claude) oder „ollama serve" starten, dann Server neu starten.';
  }
}
function updateSuggestCount(){
  const n=codeApplications.filter(a=>a.status==='suggested').length;
  const el=$('suggestCount');if(el)el.textContent=n;
}
async function startLLMCoding(){
  const turns=segments.filter(s=>s.type==='turn');
  if(!turns.length){toast('Kein Transkript geladen');return;}
  const codable=codeSystem.filter(c=>c.isCodable!==false);
  const mode=codingCfg.mode||'inductive';
  if(mode==='deductive'&&!codable.length){toast('Deduktiv braucht Codes — anlegen, Codebook laden, oder Modus auf induktiv stellen');return;}
  const provider=$('llmProvider').value;
  const how=mode==='inductive'?'induktiv (Codes entstehen am Material)':mode==='hybrid'?`hybrid (${codable.length} Codes + neue)`:`deduktiv (${codable.length} Codes)`;
  if(!confirm(`Transkript (${turns.length} Segmente) ${how} mit „${provider}" kodieren?\nDie Vorschläge prüfst du danach — nichts wird automatisch übernommen.`))return;
  const body={provider,mode,context:1,name:'KI-Kodierung',
    codes:codable.map(c=>({id:c.id,name:c.name,definition:c.definition,inclusion:c.inclusion,exclusion:c.exclusion,examples:c.examples,isCodable:true})),
    segments:turns.map(s=>({id:s.id,speaker:speakerLabel(s),text:s.text}))};
  showJobBanner('Starte KI-Kodierung…',0.02,false);
  let jobId;
  try{
    const r=await fetch('/api/code',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!r.ok)throw new Error((await r.text())||('HTTP '+r.status));
    jobId=(await r.json()).job_id;
  }catch(err){showJobBanner('Fehler: '+err.message,0,true);return;}
  trackJob(jobId,'KI-Kodierung',()=>applyCodingResult(jobId));
}
async function applyCodingResult(jobId){
  try{
    const r=await fetch('/api/jobs/'+jobId+'/result');if(!r.ok)throw new Error('Ergebnis nicht abrufbar');
    const res=await r.json();let added=0;
    (res.suggestions||[]).forEach(s=>{
      const seg=segments.find(x=>x.id===s.segment_id);if(!seg)return;
      let codeId=s.code_id;
      if(!codeId&&s.code_name){  // inductive/hybrid: emergent code — find or create (provisional)
        let c=codeSystem.find(x=>x.name.toLowerCase()===s.code_name.toLowerCase());
        if(!c){c=createCode({name:s.code_name});c.provisional=true;}
        codeId=c.id;
      }
      if(!codeId||!codeById(codeId))return;
      const dup=codeApplications.find(a=>a.codeId===codeId&&a.anchor.segmentId===s.segment_id&&a.status!=='rejected'&&Math.abs((a.anchor.hint?.start??-9)-s.char_start)<2);
      if(dup)return;
      const anchor=makeAnchor(s.segment_id,s.char_start,s.char_end);if(!anchor)return;
      codeApplications.push({id:uid('ca'),codeId,anchor,selectedText:s.quote,source:'llm',
        confidence:s.confidence,rationale:s.rationale||'',status:'suggested',reviewer:null,
        createdBy:'model:'+(res.provider||'llm'),createdAt:nowISO()});
      added++;
    });
    (res.suggested_new_codes||[]).forEach(n=>{
      if(!n.name)return;const exists=codeSystem.find(c=>c.name.toLowerCase()===n.name.toLowerCase());
      if(!exists){const c=createCode({name:n.name});c.provisional=true;c.definition=n.rationale||'';}
    });
    hideJobBanner();render();renderCodes();save();
    const st=res.stats||{};
    toast(`KI-Kodierung: ${added} Vorschläge`+(st.invalid_quotes?` · ${st.invalid_quotes} ohne gültiges Zitat verworfen`:''));
    $('cpanel').classList.remove('show');$('codepanel').classList.add('show');renderReview();
  }catch(err){showJobBanner('Fehler beim Laden: '+err.message,0,true);}
}
function renderReview(){
  retrievedCodeId=null;
  document.querySelectorAll('#ctree .cnode.active').forEach(n=>n.classList.remove('active'));
  const sugg=codeApplications.filter(a=>a.status==='suggested');
  const box=$('cretrieve');
  let html=`<div class="rhdr">✨ KI-Vorschläge · ${sugg.length}<span style="margin-left:auto"></span>`;
  if(sugg.length)html+=`<button class="btn" id="accAll" style="padding:2px 8px">alle annehmen</button>`;
  html+=`<button class="btn" id="rclose" style="padding:2px 8px;margin-left:6px">✕</button></div>`;
  if(!sugg.length)html+='<div style="color:var(--muted);font-size:12px">Keine offenen Vorschläge. „🤖 KI-Kodieren" erzeugt welche.</div>';
  else html+=sugg.map(a=>{
    const code=codeById(a.codeId);const r=resolveAnchor(a.anchor);const seg=segments.find(s=>s.id===a.anchor.segmentId);
    const text=r.ok&&seg?seg.text.slice(r.start,r.end):(a.selectedText||'?');const col=code?code.color:'#9333ea';
    return `<div class="ritem rev" data-app="${a.id}" data-seg="${a.anchor.segmentId}" style="--rcol:${col}">`+
      `<div>“${esc(text)}”</div>`+
      (a.rationale?`<div class="rmeta" style="font-style:italic;margin:3px 0">${esc(a.rationale)}</div>`:'')+
      `<div class="rmeta"><span class="cdot" style="display:inline-block;width:10px;height:10px;border-radius:3px;background:${col};vertical-align:-1px"></span> ${esc(code?code.name:'?')}`+
        `${a.confidence!=null?' · '+Math.round(a.confidence*100)+'%':''} · ${esc(seg?seg.time:'')} ${esc(seg?speakerLabel(seg):'')}</div>`+
      `<div style="margin-top:5px;display:flex;gap:5px">`+
        `<button class="btn" data-rev="accept" style="padding:1px 8px">✓ annehmen</button>`+
        `<button class="btn" data-rev="reject" style="padding:1px 8px">✗ ablehnen</button>`+
        `<button class="btn" data-rev="goto" style="padding:1px 8px">↳ Stelle</button>`+
      `</div></div>`;
  }).join('');
  box.innerHTML=html;box.classList.add('show');
}
$('llmCodeBtn').onclick=startLLMCoding;
$('reviewBtn').onclick=renderReview;

/* ---------- theme (light default + dark, universal) ---------- */
function applyTheme(t){document.documentElement.setAttribute('data-theme',t);try{localStorage.setItem('tc:theme',t);}catch(_){}}
function initTheme(){let t=null;try{t=localStorage.getItem('tc:theme');}catch(_){}
  document.documentElement.setAttribute('data-theme',t||'light');}  // light by default; dark via toggle
$('themeBtn').onclick=()=>applyTheme(document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark');
initTheme();

/* ---------- team server: auth + library + server persistence ---------- */
let authMode='login', projectsCache=[], activeProjectId=null;
let lastTextJson='', lastLayerJson='', lastCbJson='';

function enterServerMode(me){
  currentUser=me;
  $('authGate').classList.remove('show');
  $('libBtn').style.display=''; $('transcribeBtn').style.display='none';
  $('userChip').style.display='inline-flex'; $('userName').textContent=me.name||me.email;
  $('luMail').textContent=me.email;
  refreshLLMUI();
  openLibrary();
}

/* auth gate */
function showAuthGate(){$('authGate').classList.add('show');setAuthMode('login');
  $('authHint').textContent=(backend.auth&&backend.auth.allowedDomains&&backend.auth.allowedDomains.length)?('Registrierung nur für: '+backend.auth.allowedDomains.join(', ')):'';}
function setAuthMode(m){authMode=m;const reg=m==='register';
  $('authNameRow').style.display=reg?'':'none';
  $('authSub').textContent=reg?'Konto anlegen (AIT-Adresse).':'Anmelden, um mit deinem Team zu arbeiten.';
  $('authSubmit').textContent=reg?'Registrieren':'Anmelden';
  $('authHint').style.display=reg?'':'none';$('authErr').textContent='';
  $('authToggle').innerHTML=reg?'Schon ein Konto? <a id="authSwitch">Anmelden</a>':'Noch kein Konto? <a id="authSwitch">Registrieren</a>';
  $('authSwitch').onclick=()=>setAuthMode(reg?'login':'register');}
async function submitAuth(){
  const email=$('authEmail').value.trim(),pass=$('authPass').value,name=$('authName').value.trim();
  $('authErr').textContent='';
  const ep=authMode==='register'?'/api/auth/register':'/api/auth/login';
  const body=authMode==='register'?{email,name,password:pass}:{email,password:pass};
  try{const r=await fetch(ep,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();if(!r.ok)throw new Error(d.detail||'Fehler');enterServerMode(d);}
  catch(err){$('authErr').textContent=typeof err.message==='string'?err.message:'Anmeldung fehlgeschlagen';}
}
$('authSubmit').onclick=submitAuth;
$('authPass').addEventListener('keydown',e=>{if(e.key==='Enter')submitAuth();});
$('logoutBtn').onclick=async()=>{try{await fetch('/api/auth/logout',{method:'POST'});}catch(_){}location.reload();};

/* library */
function openLibrary(){$('cpanel').classList.remove('show');$('codepanel').classList.remove('show');$('libpanel').classList.add('show');loadProjects();}
$('libBtn').onclick=openLibrary;$('userChip').onclick=openLibrary;
$('libClose').onclick=()=>$('libpanel').classList.remove('show');
async function loadProjects(){try{projectsCache=await (await fetch('/api/projects',{cache:'no-store'})).json();}catch(_){projectsCache=[];}renderLibrary();}
function renderLibrary(){
  const body=$('libBody');let html='';
  html+='<div class="lactions"><button class="btn primary" id="newProjBtn">＋ Neues Projekt</button></div>';
  html+='<div class="lsec">Projekte</div>';
  html+=projectsCache.length?projectsCache.map(p=>`<div class="lrow${p.id===activeProjectId?' active':''}" data-proj="${p.id}"><span class="ltitle">${esc(p.name)}</span><span class="lmeta">${p.documents}📄</span><span class="lrole">${esc(p.role)}</span></div>`).join('')
    :'<div class="lempty">Noch keine Projekte — leg eines an.</div>';
  if(activeProjectId){
    const proj=projectsCache.find(p=>p.id===activeProjectId);
    html+=`<div class="lsec">Dokumente · ${esc(proj?proj.name:'')}</div>`;
    html+='<div class="lactions"><button class="btn" id="txInProj">🎙 Audio transkribieren</button><button class="btn" id="importTxt">📄 .txt importieren</button>';
    if(proj&&proj.role==='admin')html+='<button class="btn" id="addMember">＋ Mitglied</button>';
    html+='</div><div id="docList"><div class="lempty">…</div></div>';
  }
  body.innerHTML=html;
  $('newProjBtn').onclick=createProjectFlow;
  body.querySelectorAll('[data-proj]').forEach(el=>el.onclick=()=>{activeProjectId=el.dataset.proj;renderLibrary();});
  if(activeProjectId){$('txInProj').onclick=()=>{pendingProjectId=activeProjectId;openTxModal();};
    $('importTxt').onclick=importTxtFlow;const am=$('addMember');if(am)am.onclick=addMemberFlow;loadDocs();}
}
async function loadDocs(){const pid=activeProjectId;let docs=[];
  try{docs=await (await fetch('/api/projects/'+pid+'/documents',{cache:'no-store'})).json();}catch(_){}
  const el=$('docList');if(!el)return;
  el.innerHTML=docs.length?docs.map(d=>`<div class="lrow${serverDoc&&serverDoc.id===d.id?' active':''}" data-doc="${d.id}"><span class="ltitle">${esc(d.name)}</span><span class="lmeta">${d.segments} Zeilen</span></div>`).join('')
    :'<div class="lempty">Noch keine Dokumente. Transkribiere oder importiere eines.</div>';
  el.querySelectorAll('[data-doc]').forEach(x=>x.onclick=()=>openServerDocument(x.dataset.doc));
}
function createProjectFlow(){const name=prompt('Name des neuen Projekts:','');if(!name||!name.trim())return;
  fetch('/api/projects',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name.trim()})})
    .then(r=>r.json()).then(p=>{activeProjectId=p.id;loadProjects();});}
function addMemberFlow(){const email=prompt('E-Mail des Mitglieds (muss bereits registriert sein):','');if(!email||!email.trim())return;
  const role=confirm('Als Admin hinzufügen?  (Abbrechen = Coder)')?'admin':'coder';
  fetch('/api/projects/'+activeProjectId+'/members',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:email.trim(),role})})
    .then(r=>r.json().then(d=>toast(r.ok?'Mitglied hinzugefügt':(d.detail||'Fehler'))));}
function importTxtFlow(){pick('.txt,text/plain',f=>f.text().then(t=>{
  fetch('/api/projects/'+activeProjectId+'/documents',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:f.name.replace(/\.txt$/i,''),text:t})})
    .then(r=>r.json()).then(d=>{if(d.id){loadProjects();openServerDocument(d.id);}else toast(d.detail||'Import fehlgeschlagen');});}));}

/* open + persist a server document */
async function openServerDocument(docId){
  let doc;
  try{const r=await fetch('/api/documents/'+docId,{cache:'no-store'});if(!r.ok)throw new Error('Laden fehlgeschlagen ('+r.status+')');doc=await r.json();}
  catch(err){toast('Fehler: '+err.message);return;}
  transcriptName=doc.name||'Dokument';
  installDoc(normalizeDoc(doc,doc.name,doc.header||''));
  serverDoc={id:docId,projectId:doc.projectId,rev:doc.rev||0,conflict:false};
  captureServerBaseline();
  if(doc.hasAudio)loadMediaUrl('/api/documents/'+docId+'/audio');
  else{media.removeAttribute('src');media.classList.add('audioOnly');$('noMedia').style.display='';}
  $('libpanel').classList.remove('show');$('restoreBanner').classList.remove('show');
  toast('Geöffnet: '+transcriptName);
}
function captureServerBaseline(){
  lastTextJson=JSON.stringify({speakers:speakerList,segments,header:headerText});
  lastLayerJson=JSON.stringify({codeApplications,highlights,comments});
  lastCbJson=JSON.stringify({codeSystem,coding:codingCfg});
}
async function serverSaveFlush(){
  if(!serverDoc)return;
  const tj=JSON.stringify({speakers:speakerList,segments,header:headerText});
  if(tj!==lastTextJson&&!serverDoc.conflict){
    try{const r=await fetch('/api/documents/'+serverDoc.id+'/text',{method:'PUT',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({speakers:speakerList,segments,header:headerText,rev:serverDoc.rev})});
      if(r.status===409){serverDoc.conflict=true;showConflict();}
      else if(r.ok){lastTextJson=tj;serverDoc.rev=(await r.json()).rev;}}catch(_){}
  }
  const lj=JSON.stringify({codeApplications,highlights,comments});
  if(lj!==lastLayerJson){
    try{const r=await fetch('/api/documents/'+serverDoc.id+'/layer',{method:'PUT',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({codeApplications,highlights,comments})});if(r.ok)lastLayerJson=lj;}catch(_){}
  }
  const cj=JSON.stringify({codeSystem,coding:codingCfg});
  if(cj!==lastCbJson&&serverDoc.projectId){
    try{const r=await fetch('/api/projects/'+serverDoc.projectId+'/codebook',{method:'PUT',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({codeSystem,coding:codingCfg})});if(r.ok)lastCbJson=cj;}catch(_){}
  }
}
function showConflict(){const b=$('restoreBanner');
  $('restoreMsg').textContent='⚠ Transkript wurde von jemand anderem geändert. Deine Text-Änderungen werden nicht gespeichert.';
  $('resetState').textContent='Neu laden';b.classList.add('show');
  $('resetState').onclick=()=>{const id=serverDoc.id;serverDoc=null;b.classList.remove('show');openServerDocument(id);};}

/* ---------- boot ---------- */
(function(){const emb=$('embedded-transcript').textContent.trim();
  if(emb&&!/^@@/.test(emb)){_lastRaw=emb;loadTranscript(emb,'GAIN co-design ws 2 diotima MERGED.txt');}
  else{$('transcript').innerHTML='<p class="hint" style="text-align:center;padding:40px">Kein Transkript eingebettet. Oben „📄 Transkript laden" wählen (eine .txt mit <code>[HH:MM:SS] SPEAKER: Text</code>-Zeilen) und „🔊 Audio/Video laden".</p>';}
})();
