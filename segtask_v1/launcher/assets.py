"""前端静态资源（内联 CSS + JS），零外部依赖。

与 ``monitor/`` / ``visualization/`` 一致的自包含范式：CSS/JS 以字符串常量存放，
由 ``page.py`` 注入单页 HTML。JS 通过 ``/api/*`` 与本地服务交互，按 manifest 载荷
渲染 per-mode 有效参数表单、联动启用、校验、预览 YAML、启动并实时回传日志。
"""

from __future__ import annotations

CSS = r"""
:root{
  --bg:#0f1419; --panel:#1a2230; --panel2:#222c3d; --line:#2c3a50;
  --fg:#e6edf3; --muted:#8b98a9; --accent:#4f9cf9; --accent2:#2d7d46;
  --danger:#d9534f; --warn:#c9a227; --radius:8px;
  --mono:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",
  "PingFang SC","Microsoft YaHei",sans-serif;font-size:14px;line-height:1.5}
header{position:sticky;top:0;z-index:20;background:var(--panel);
  border-bottom:1px solid var(--line);padding:10px 18px;display:flex;
  align-items:center;gap:14px;flex-wrap:wrap}
header h1{font-size:16px;margin:0;font-weight:600}
.badge{background:var(--accent);color:#04101f;border-radius:20px;
  padding:2px 10px;font-weight:700;font-size:12px}
.modes a{color:var(--muted);text-decoration:none;margin-right:10px;font-size:13px}
.modes a.active{color:var(--accent);font-weight:600}
.wrap{display:flex;gap:0;min-height:calc(100vh - 52px)}
.left{flex:1;min-width:0;padding:16px 18px;overflow:auto}
.right{width:42%;max-width:680px;border-left:1px solid var(--line);
  background:#0b0f14;display:flex;flex-direction:column;position:sticky;top:52px;
  height:calc(100vh - 52px)}
.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:14px;
  position:sticky;top:52px;z-index:15;background:var(--bg);padding:8px 0;
  margin-top:-8px}
button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);
  border-radius:var(--radius);padding:7px 14px;cursor:pointer;font-size:13px;
  transition:border-color .12s,background .12s}
button:hover{border-color:var(--accent)}
button.primary{background:var(--accent);color:#04101f;border-color:var(--accent);
  font-weight:600}
button.danger{background:var(--danger);color:#fff;border-color:var(--danger)}
button:disabled{opacity:.45;cursor:not-allowed}
select,input[type=text],input[type=number]{background:#0d141d;color:var(--fg);
  border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:13px;
  width:100%}
input:focus,select:focus{outline:none;border-color:var(--sec,var(--accent));
  box-shadow:0 0 0 2px color-mix(in srgb,var(--sec,var(--accent)) 22%,transparent)}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:var(--radius);
  overflow:hidden}
.seg button{border:none;border-radius:0;margin:0}
.seg button.on{background:var(--accent);color:#04101f;font-weight:600}

/* ---- 模块分组（每个 section 一种主题色，由 data-sec 决定 --sec） ---- */
.group{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
  margin-bottom:12px;overflow:hidden;border-left:4px solid var(--sec,var(--accent))}
.group[data-sec=data]{--sec:#4f9cf9}
.group[data-sec=augment]{--sec:#26c0c0}
.group[data-sec=model]{--sec:#a972f0}
.group[data-sec=loss]{--sec:#e8893b}
.group[data-sec=train]{--sec:#46c476}
.group[data-sec=predict]{--sec:#ef6fa6}
.group[data-sec=vismon]{--sec:#7d92b5}
.group[data-sec=run]{--sec:#d8b738}
.group>.ghead{padding:9px 14px;cursor:pointer;display:flex;justify-content:space-between;
  align-items:center;font-weight:600;user-select:none;
  background:linear-gradient(90deg,
    color-mix(in srgb,var(--sec,var(--accent)) 20%,var(--panel2)),
    var(--panel2) 60%)}
.group>.ghead .htitle{display:flex;align-items:center;gap:8px}
.group>.ghead .gdot{width:9px;height:9px;border-radius:50%;
  background:var(--sec,var(--accent));box-shadow:0 0 7px var(--sec,var(--accent))}
.group>.ghead .count{color:var(--muted);font-weight:400;font-size:12px}
.group>.ghead .arrow{transition:transform .15s;color:var(--muted);font-size:11px}
.group.collapsed>.ghead .arrow{transform:rotate(-90deg)}
.group.collapsed>.gbody{display:none}

/* ---- 表单：自适应多列网格；长字段占整行 ---- */
.gbody{padding:12px 14px 14px;display:grid;gap:11px 16px;
  grid-template-columns:repeat(auto-fill,minmax(230px,1fr))}
.field{display:flex;flex-direction:column;gap:5px;min-width:0}
.field.wide{grid-column:1/-1}
.field.flag{flex-direction:row;align-items:center;gap:9px}
.field.flag .ctrl{order:-1}
.field label{display:flex;align-items:center;gap:6px;color:#cdd9e5;font-size:12.5px;
  word-break:break-word}
.field .ctrl{min-width:0}
.help{display:inline-flex;width:16px;height:16px;border-radius:50%;
  background:#33425a;color:#cfe;font-size:11px;align-items:center;justify-content:center;
  cursor:help;flex:none;position:relative}
.help:hover .tip{display:block}
.tip{display:none;position:absolute;left:20px;top:-4px;z-index:30;width:320px;
  background:#04101f;border:1px solid var(--sec,var(--accent));border-radius:6px;
  padding:8px 10px;color:#dfe;font-size:12px;line-height:1.45;
  box-shadow:0 6px 20px rgba(0,0,0,.5);white-space:pre-wrap}
.cb{width:17px;height:17px;accent-color:var(--sec,var(--accent))}
.row-flag{display:flex;align-items:center;gap:8px}
.muted{color:var(--muted)}
.err{color:var(--danger)}
.ok{color:#5cc97f}
.err-b{border-color:var(--danger)!important}

/* ---- 右侧：终端化日志 ---- */
.rhead{padding:9px 14px;border-bottom:1px solid var(--line);display:flex;
  justify-content:space-between;align-items:center;background:var(--panel)}
.rhead .dot{width:9px;height:9px;border-radius:50%;background:#555;display:inline-block;
  margin-right:6px;vertical-align:middle}
.rhead .dot.run{background:#5cc97f;box-shadow:0 0 6px #5cc97f;
  animation:pulse 1.3s ease-in-out infinite}
.rhead .dot.done{background:var(--accent)}
.rhead .dot.fail{background:var(--danger)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.term{flex:1;display:flex;flex-direction:column;min-height:0;background:#05080c;
  position:relative}
.termbar{display:flex;align-items:center;gap:10px;padding:7px 12px;
  background:#0c1119;border-bottom:1px solid #1a2433}
.termbar .lights{display:flex;gap:6px}
.termbar .lights i{width:11px;height:11px;border-radius:50%;display:inline-block}
.termbar .lights i.r{background:#ff5f57}
.termbar .lights i.y{background:#febc2e}
.termbar .lights i.g{background:#28c840}
.termbar .ttitle{color:#9fb1c6;font-family:var(--mono);font-size:12px;flex:1;
  text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.termbar .ttools{display:flex;gap:6px}
.tbtn{padding:3px 9px;font-size:12px;background:#16202e;border:1px solid #243348}
#logs{flex:1;overflow:auto;padding:10px 14px;font-family:var(--mono);
  font-size:12.5px;line-height:1.6;white-space:pre-wrap;word-break:break-all}
#logs .l{display:block;padding:0 4px;border-radius:3px}
#logs .l:hover{background:rgba(255,255,255,.04)}
#logs .l{color:#c2cedb}
#logs .l.cmd{color:#6fd2ff;font-weight:600}
#logs .l.cmd::before{content:"";color:#5cc97f}
#logs .l.meta{color:#c9a227}
#logs .l.warn{color:#e8c249;background:rgba(232,194,73,.06)}
#logs .l.err{color:#ff8a82;background:rgba(217,83,79,.09)}
#logs .l .num{color:#9ed0a8}
#logs .l.cmd .num{color:inherit}
.tobottom{position:absolute;right:16px;bottom:14px;z-index:5;font-size:12px;
  padding:5px 11px;background:var(--accent);color:#04101f;border-color:var(--accent);
  font-weight:600;box-shadow:0 4px 12px rgba(0,0,0,.5)}
#yamlview{flex:1;overflow:auto;padding:10px 14px;font-family:var(--mono);
  font-size:12.5px;line-height:1.55;white-space:pre;background:#05080c;color:#bcd}
.msgbar{padding:8px 14px;border-top:1px solid var(--line);min-height:20px;font-size:13px}
.tabs{display:flex;border-bottom:1px solid var(--line)}
.tabs button{border:none;border-radius:0;background:transparent;color:var(--muted);
  border-bottom:2px solid transparent}
.tabs button.on{color:var(--accent);border-bottom-color:var(--accent)}
.hidden{display:none!important}
.baserow{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
.note{font-size:12px;color:var(--muted);margin:4px 0 12px}

/* 暗色滚动条 */
.left::-webkit-scrollbar,#logs::-webkit-scrollbar,#yamlview::-webkit-scrollbar{
  width:10px;height:10px}
.left::-webkit-scrollbar-thumb,#logs::-webkit-scrollbar-thumb,
#yamlview::-webkit-scrollbar-thumb{background:#2a3850;border-radius:6px;
  border:2px solid transparent;background-clip:padding-box}
.left::-webkit-scrollbar-thumb:hover,#logs::-webkit-scrollbar-thumb:hover{
  background:#3a4d6e;background-clip:padding-box}
"""


JS = r"""
'use strict';
const MODE = window.__MODE__;          // '2_5d' | '3d'
let PAYLOAD = null;                     // /api/payload
let STATE = {};                         // {section:{field:value}}
let RUNVALS = {};                       // predict run-args
let TASK = 'train';                     // 'train' | 'predict'
let logTimer = null, logCursor = 0;

const $ = (s,r=document)=>r.querySelector(s);
const $$ = (s,r=document)=>Array.from(r.querySelectorAll(s));
function el(tag, attrs={}, ...kids){
  const e=document.createElement(tag);
  for(const k in attrs){
    if(k==='class') e.className=attrs[k];
    else if(k==='html') e.innerHTML=attrs[k];
    else if(k.startsWith('on')) e.addEventListener(k.slice(2), attrs[k]);
    else if(attrs[k]!=null) e.setAttribute(k, attrs[k]);
  }
  for(const c of kids){ if(c!=null) e.append(c.nodeType?c:document.createTextNode(c)); }
  return e;
}

async function api(path, body){
  const opt = body!==undefined
    ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)}
    : {};
  const r = await fetch(path, opt);
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}

// ---- value get/set on STATE ("section.field") ----
function getVal(ref){
  const [s,f]=ref.split('.');
  return (STATE[s]||{})[f];
}
function setVal(ref,v){
  const [s,f]=ref.split('.');
  (STATE[s]=STATE[s]||{})[f]=v;
}

// ---- depends_on evaluation ----
function condMet(c){
  const v=getVal(c.f);
  if('in' in c)        return c.in.includes(v);
  if('contains' in c)  return typeof v==='string' && v.includes(c.contains);
  if('truthy' in c)    return !!v;
  if('nonempty' in c)  return v!=null && String(v).length>0 &&
                               !(Array.isArray(v)&&v.length===0);
  if('len_gt' in c)    return Array.isArray(v) && v.length>c.len_gt;
  return true;
}
function fieldActive(fld){ return (fld.depends_on||[]).every(condMet); }

// ---- control rendering ----
function controlFor(fld){
  const ref=fld.ref, cur=getVal(ref);
  if(fld.control==='bool'){
    const cb=el('input',{type:'checkbox',class:'cb'});
    cb.checked=!!cur;
    cb.addEventListener('change',()=>{ setVal(ref,cb.checked); refresh(); });
    return el('div',{class:'row-flag'},cb);
  }
  if(fld.control==='enum'){
    const sel=el('select');
    for(const opt of fld.enum){
      const o=el('option',{value:opt}, opt===''?'(空)':opt);
      if(opt===cur) o.selected=true;
      sel.append(o);
    }
    sel.disabled=!!fld.readonly;
    sel.addEventListener('change',()=>{ setVal(ref,sel.value); refresh(); });
    return sel;
  }
  if(fld.control==='int'||fld.control==='float'){
    const inp=el('input',{type:'number'});
    inp.step = fld.control==='int' ? '1' : 'any';
    inp.value = cur==null ? '' : cur;
    inp.addEventListener('change',()=>{
      if(inp.value===''){ setVal(ref,null); return; }
      setVal(ref, fld.control==='int'? parseInt(inp.value,10): parseFloat(inp.value));
      refresh();
    });
    return inp;
  }
  if(fld.control==='list'){
    const inp=el('input',{type:'text',placeholder:'JSON，如 [1.0, 1.5] 或 [0,1,2]'});
    inp.value = JSON.stringify(cur);
    inp.addEventListener('change',()=>{
      try{ const v=JSON.parse(inp.value); setVal(ref,v); inp.classList.remove('err-b'); }
      catch(e){ inp.classList.add('err-b'); setMsg('字段 '+ref+' 不是合法 JSON 列表','err'); }
      refresh();
    });
    return inp;
  }
  // str
  const inp=el('input',{type:'text'});
  inp.value = cur==null?'':cur;
  inp.disabled=!!fld.readonly;
  inp.addEventListener('change',()=>{ setVal(ref,inp.value); refresh(); });
  return inp;
}

// 长字段（路径 / 列表 / JSON）占整行；布尔紧凑同行；短字符串/数值/枚举进多列网格
const WIDE_RE=/(_dir|_list|path|resume|pretrain|output|input|checkpoint|weights|bbox)/i;
function fieldClass(fld){
  if(fld.control==='bool') return 'field flag';
  if(fld.control==='list') return 'field wide';
  const name=(fld.ref||'').split('.').pop();
  if(fld.control==='str' && WIDE_RE.test(name)) return 'field wide';
  return 'field';
}
function fieldRow(fld){
  const lbl=el('label',{},
    fld.label,
    fld.tooltip? el('span',{class:'help'},'?', el('span',{class:'tip'},fld.tooltip)) : null,
    fld.readonly? el('span',{class:'muted'},' (锁定)') : null);
  const row=el('div',{class:fieldClass(fld)}, lbl, el('div',{class:'ctrl'}, controlFor(fld)));
  row.__fld=fld;
  return row;
}

// ---- build form ----
function renderForm(){
  const host=$('#form'); host.innerHTML='';
  const showSecs = PAYLOAD.task_sections[TASK];
  for(const g of PAYLOAD.groups){
    if(!showSecs.includes(g.section_tag)) continue;
    addGroup(host, g.section_tag, g.title, g.fields.length,
             g.fields.map(fieldRow));
  }
  // predict: run-args group
  if(TASK==='predict'){
    addGroup(host, 'run', '运行参数 Run (CLI)', PAYLOAD.predict_run_args.length,
             PAYLOAD.predict_run_args.map(runArgRow));
  }
  refresh();
}

function addGroup(host, sec, title, count, rows){
  const body=el('div',{class:'gbody'});
  for(const r of rows) body.append(r);
  const head=el('div',{class:'ghead'},
    el('span',{class:'htitle'},
      el('span',{class:'gdot'}),
      el('span',{},title),
      el('span',{class:'count'},'· '+count)),
    el('span',{class:'arrow'},'▼'));
  const grp=el('div',{class:'group','data-sec':sec}, head, body);
  head.addEventListener('click',()=>grp.classList.toggle('collapsed'));
  host.append(grp);
}

function runArgRow(ra){
  const ref='run.'+ra.name;
  let ctrl;
  if(ra.control==='bool'){
    const cb=el('input',{type:'checkbox',class:'cb'}); cb.checked=!!RUNVALS[ra.name];
    cb.addEventListener('change',()=>RUNVALS[ra.name]=cb.checked);
    ctrl=el('div',{class:'row-flag'},cb);
  }else if(ra.control==='enum'){
    const sel=el('select');
    for(const o of ra.enum){ const op=el('option',{value:o},o);
      if(o===RUNVALS[ra.name])op.selected=true; sel.append(op); }
    sel.addEventListener('change',()=>RUNVALS[ra.name]=sel.value);
    ctrl=sel;
  }else{
    const inp=el('input',{type:'text',placeholder:'留空用默认'});
    inp.value=RUNVALS[ra.name]||'';
    inp.addEventListener('change',()=>RUNVALS[ra.name]=inp.value);
    ctrl=inp;
  }
  const lbl=el('label',{}, ra.name,
    ra.tooltip? el('span',{class:'help'},'?', el('span',{class:'tip'},ra.tooltip)):null);
  const cls = ra.control==='bool' ? 'field flag'
            : ra.control==='enum' ? 'field' : 'field wide';
  return el('div',{class:cls}, lbl, el('div',{class:'ctrl'},ctrl));
}

// re-evaluate depends_on visibility without full rebuild
function refresh(){
  $$('#form .field').forEach(row=>{
    const fld=row.__fld; if(!fld) return;
    row.classList.toggle('hidden', !fieldActive(fld));
  });
}

function setMsg(t, cls=''){ const m=$('#msg'); m.textContent=t; m.className='msgbar '+cls; }

// collect only active+visible values for the run yaml
function collectValues(){
  // start from defaults, overlay STATE; only include sections relevant to task.
  const out={};
  for(const s in STATE){ out[s]={}; for(const f in STATE[s]) out[s][f]=STATE[s][f]; }
  return out;
}

// ---- actions ----
async function doValidate(){
  setMsg('校验中…');
  try{ const r=await api('/api/validate',{values:collectValues()});
    setMsg(r.ok?('[OK] '+r.message):('[X] '+r.message), r.ok?'ok':'err'); return r.ok; }
  catch(e){ setMsg('校验请求失败: '+e.message,'err'); return false; }
}
async function doPreview(){
  showTab('yaml');
  try{ const r=await api('/api/preview',{values:collectValues()});
    $('#yamlview').textContent=r.yaml; }
  catch(e){ $('#yamlview').textContent='预览失败: '+e.message; }
}
async function doLaunch(){
  if(!await doValidate()){ setMsg('配置未通过校验，已阻止启动。','err'); return; }
  showTab('logs'); $('#logs').innerHTML=''; logCursor=0;
  try{
    const r=await api('/api/launch',
      {task:TASK, mode:MODE, values:collectValues(), run_values:RUNVALS});
    if(!r.ok){ setMsg('启动失败: '+r.error,'err'); return; }
    setMsg('已启动: '+r.cmd.join(' ')+'  （YAML: '+r.yaml_path+'）','ok');
    startLogPoll();
  }catch(e){ setMsg('启动请求失败: '+e.message,'err'); }
}
async function doStop(){
  try{ await api('/api/stop',{}); setMsg('已请求停止。','warn'); }
  catch(e){ setMsg('停止失败: '+e.message,'err'); }
}

function setDot(cls){ const d=$('#statusdot'); d.className='dot '+cls; }

// ---- 终端化日志渲染 ----
let stickBottom=true, runStart=0, titleTimer=null;
function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function classifyLine(line){
  if(line.startsWith('$')) return 'cmd';
  if(/\b(error|traceback|exception|fatal|failed)\b/i.test(line)) return 'err';
  if(/\bwarn(ing)?\b/i.test(line)) return 'warn';
  if(line.startsWith('[')) return 'meta';
  return 'info';
}
function appendLine(line){
  const cls=classifyLine(line);
  let html=esc(line);
  if(cls==='info'||cls==='meta') html=html.replace(/(\d+(?:\.\d+)?)/g,'<span class="num">$1</span>');
  $('#logs').append(el('div',{class:'l '+cls, html:html}));
}
function fmtElapsed(ms){
  const s=Math.floor(ms/1000), m=Math.floor(s/60);
  return String(m).padStart(2,'0')+':'+String(s%60).padStart(2,'0');
}
function setTermTitle(state){
  $('#termtitle').textContent=(TASK==='train'?'train':'predict')+' ▸ '+state;
}

function startLogPoll(){
  if(logTimer) clearInterval(logTimer);
  setDot('run'); $('#stopBtn').disabled=false; $('#launchBtn').disabled=true;
  runStart=Date.now(); stickBottom=true; $('#toBottom').classList.add('hidden');
  if(titleTimer) clearInterval(titleTimer);
  titleTimer=setInterval(()=>setTermTitle(fmtElapsed(Date.now()-runStart)),1000);
  setTermTitle('00:00');
  logTimer=setInterval(async()=>{
    try{
      const r=await api('/api/logs?since='+logCursor);
      for(const line of r.lines) appendLine(line);
      logCursor=r.next;
      if(stickBottom) $('#logs').scrollTop=$('#logs').scrollHeight;
      if(!r.running){
        clearInterval(logTimer); logTimer=null;
        if(titleTimer){ clearInterval(titleTimer); titleTimer=null; }
        setDot(r.returncode===0?'done':'fail');
        $('#stopBtn').disabled=true; $('#launchBtn').disabled=false;
        const tail=r.returncode===0?'完成':('退出码 '+r.returncode);
        $('#statustext').textContent=tail;
        setTermTitle((r.returncode===0?'完成 · ':'失败 · ')+fmtElapsed(Date.now()-runStart));
      }else{ $('#statustext').textContent='运行中…'; }
    }catch(e){ /* keep polling */ }
  }, 1000);
}

function showTab(which){
  $('#tabLogs').classList.toggle('on', which==='logs');
  $('#tabYaml').classList.toggle('on', which==='yaml');
  $('#term').classList.toggle('hidden', which!=='logs');
  $('#yamlview').classList.toggle('hidden', which!=='yaml');
}

function setTask(t){
  TASK=t;
  $('#taskTrain').classList.toggle('on', t==='train');
  $('#taskPredict').classList.toggle('on', t==='predict');
  $('#launchBtn').textContent = t==='train'?'开始训练':'开始推理';
  $('#baseRow').classList.toggle('need', t==='predict');
  renderForm();
}

async function loadBaseConfigs(){
  try{
    const r=await api('/api/base_configs?mode='+MODE);
    const sel=$('#baseSel'); sel.innerHTML='';
    sel.append(el('option',{value:''},'（不载入，用默认值）'));
    for(const p of r.configs) sel.append(el('option',{value:p},p));
  }catch(e){ setMsg('基础配置列表加载失败: '+e.message,'err'); }
}
async function applyBase(){
  const p=$('#baseSel').value;
  if(!p){ return; }
  try{
    const r=await api('/api/load_base',{path:p});
    STATE=r.values;
    setMsg('已载入基础配置: '+p,'ok');
    renderForm();
  }catch(e){ setMsg('载入失败: '+e.message,'err'); }
}

async function init(){
  PAYLOAD = await api('/api/payload?mode='+MODE);
  // init STATE from all_defaults
  STATE={};
  for(const s in PAYLOAD.all_defaults){ STATE[s]={};
    for(const f in PAYLOAD.all_defaults[s]) STATE[s][f]=PAYLOAD.all_defaults[s][f]; }
  // force page-mode patch_mode
  setVal('data.patch_mode', PAYLOAD.patch_mode_default);
  // init run-arg defaults
  for(const ra of PAYLOAD.predict_run_args) RUNVALS[ra.name]=ra.default;
  await loadBaseConfigs();
  setTask('train');
  showTab('logs');
  // wire buttons
  $('#validateBtn').onclick=doValidate;
  $('#previewBtn').onclick=doPreview;
  $('#launchBtn').onclick=doLaunch;
  $('#stopBtn').onclick=doStop;
  $('#taskTrain').onclick=()=>setTask('train');
  $('#taskPredict').onclick=()=>setTask('predict');
  $('#applyBase').onclick=applyBase;
  $('#tabLogs').onclick=()=>showTab('logs');
  $('#tabYaml').onclick=()=>showTab('yaml');
  // terminal controls
  $('#copyBtn').onclick=async()=>{
    const txt=$$('#logs .l').map(n=>n.textContent).join('\n');
    try{ await navigator.clipboard.writeText(txt); setMsg('日志已复制到剪贴板。','ok'); }
    catch(e){ setMsg('复制失败（浏览器限制）。','err'); }
  };
  $('#clearBtn').onclick=()=>{ $('#logs').innerHTML=''; };
  $('#toBottom').onclick=()=>{
    const lg=$('#logs'); lg.scrollTop=lg.scrollHeight;
    stickBottom=true; $('#toBottom').classList.add('hidden');
  };
  $('#logs').addEventListener('scroll',()=>{
    const lg=$('#logs');
    stickBottom = lg.scrollHeight-lg.scrollTop-lg.clientHeight < 24;
    $('#toBottom').classList.toggle('hidden', stickBottom);
  });
  // poll existing run (in case of reload)
  try{ const st=await api('/api/status'); if(st.running){ startLogPoll(); } }catch(e){}
}
window.addEventListener('DOMContentLoaded', init);
"""
