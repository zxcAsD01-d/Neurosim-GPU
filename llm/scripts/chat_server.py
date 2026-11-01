#!/usr/bin/env python3
"""Local chat page for Cortex-LM.

This is a separate local web app, not the DSH Web GUI. It loads one checkpoint
once and keeps it in GPU memory; each request streams generated tokens.

Run:
    python llm/scripts/chat_server.py --checkpoint runs/cortex_stage3/final.pt --port 7860
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import torch
from flask import Flask, Response, jsonify, request, stream_with_context

from neurosim_gpu.cortex.checkpoint import load_checkpoint
from neurosim_gpu.cortex.config import ExperimentConfig
from neurosim_gpu.cortex.model import CortexSandwichLM
from neurosim_gpu.cortex.tokenization import build_tokenizer

PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cortex-LM 对话测试</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0b0f14;color:#e9ecef;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}
header{padding:18px 24px;border-bottom:1px solid #222a33;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:18px;margin:0;font-weight:600}
.sub{color:#8b98a5;font-size:12px}
#status{margin-left:auto;font-size:12px;color:#8b98a5}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#51cf66;margin-right:6px}
.wrap{max-width:980px;margin:0 auto;padding:18px 18px 120px}
#log{display:flex;flex-direction:column;gap:12px}
.msg{display:flex;gap:10px;align-items:flex-start}
.msg.user{flex-direction:row-reverse}
.bubble{max-width:78%;padding:10px 13px;border-radius:14px;white-space:pre-wrap;word-break:break-word}
.user .bubble{background:#173a4a;border:1px solid #23576c}
.bot .bubble{background:#151b22;border:1px solid #27313c}
.meta{font-size:11px;color:#7f8c99;margin-top:4px}
.composer{position:fixed;left:0;right:0;bottom:0;background:linear-gradient(transparent,#0b0f14 30%);padding:18px}
.composer-inner{max-width:980px;margin:0 auto;display:flex;gap:10px;background:#111820;border:1px solid #27313c;border-radius:16px;padding:10px}
textarea{flex:1;resize:none;min-height:44px;max-height:140px;background:transparent;border:0;outline:0;color:#e9ecef;font:inherit;padding:8px}
button{border:1px solid #2f6f85;background:#1b5568;color:#e9ecef;border-radius:10px;padding:0 16px;cursor:pointer;font:inherit}
button:disabled{opacity:.5;cursor:default}
button.ghost{background:transparent;border-color:#2a3540;color:#9aa7b4}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
select,input[type=range]{background:#111820;color:#e9ecef;border:1px solid #27313c;border-radius:8px;padding:4px 6px}
.examples{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.chip{border:1px solid #2a3540;color:#9aa7b4;background:#111820;border-radius:999px;padding:4px 10px;font-size:12px;cursor:pointer}
</style>
</head>
<body>
<header>
  <h1>Cortex-LM 对话测试</h1>
  <span class="sub" id="model">loading…</span>
  <span id="status"><span class="dot"></span>local runtime</span>
</header>
<div class="wrap">
  <div class="examples" id="examples"></div>
  <div id="log"></div>
</div>
<div class="composer"><div class="composer-inner">
  <textarea id="input" placeholder="输入消息，Enter 发送，Shift+Enter 换行"></textarea>
  <div style="display:flex;flex-direction:column;gap:6px;justify-content:flex-end">
    <div class="row"><button id="send">发送</button><button class="ghost" id="clear">清空</button></div>
    <div class="row"><select id="temp"><option value="0.4">0.4</option><option value="0.7" selected>0.7</option><option value="0.9">0.9</option><option value="0">greedy</option></select><span class="sub">temp</span></div>
  </div>
</div></div>
<script>
const log=document.getElementById('log'), input=document.getElementById('input'), send=document.getElementById('send');
let messages=[]; let busy=false;
const examples=['你好','你是谁？','今天天气怎么样？','1+1等于几？','给我讲一个短故事。'];
const ex=document.getElementById('examples');
examples.forEach(t=>{const b=document.createElement('button');b.className='chip';b.textContent=t;b.onclick=()=>{input.value=t;input.focus()};ex.appendChild(b)});
function add(role,text,meta){const row=document.createElement('div');row.className='msg '+role;const b=document.createElement('div');b.className='bubble bot';if(role==='user')b.className='bubble';b.textContent=text;row.appendChild(b);if(meta){const m=document.createElement('div');m.className='meta';m.textContent=meta;row.appendChild(m)}log.appendChild(row);return b}
async function loadHealth(){try{const r=await fetch('/api/health');const j=await r.json();document.getElementById('model').textContent=`${j.checkpoint} · ${j.device} · ticks ${j.ticks_per_token}`;}catch(e){}}
loadHealth();
function setBusy(v){busy=v;send.disabled=v;document.getElementById('clear').disabled=v}
async function sendMessage(){
  const text=input.value.trim(); if(!text||busy) return; input.value='';
  messages.push({role:'user',content:text}); add('user',text);
  const bot=add('assistant','', '');
  setBusy(true); const started=performance.now(); let full='';
  try{
    const res=await fetch('/api/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages,temperature:parseFloat(document.getElementById('temp').value),max_new_tokens:64})});
    if(!res.ok||!res.body) throw new Error('HTTP '+res.status);
    const reader=res.body.getReader(); const dec=new TextDecoder(); let buf='';
    while(true){const {value,done}=await reader.read(); if(done)break; buf+=dec.decode(value,{stream:true});
      let idx; while((idx=buf.indexOf('\\n\\n'))>=0){const chunk=buf.slice(0,idx);buf=buf.slice(idx+2); if(!chunk.startsWith('data: '))continue; const data=JSON.parse(chunk.slice(6));
        if(data.type==='token'){full+=data.text;bot.textContent=full;window.scrollTo(0,document.body.scrollHeight)}
        else if(data.type==='done'){const meta=document.createElement('div');meta.className='meta';meta.textContent=`${data.tokens} tok · ${data.seconds.toFixed(2)}s · ${(data.tokens/Math.max(data.seconds,1e-6)).toFixed(1)} tok/s · ${data.device}`;bot.parentElement.appendChild(meta)}
      }
    }
    messages.push({role:'assistant',content:full});
  }catch(e){bot.textContent='[error] '+e.message;}
  setBusy(false); input.focus();
}
send.onclick=sendMessage;
document.getElementById('clear').onclick=()=>{messages=[];log.innerHTML=''};
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage()}});
input.focus();
</script>
</body>
</html>
"""


def build_prompt(messages: list[dict]) -> str:
    parts = []
    for message in messages[-10:]:
        role = "用户" if message.get("role") == "user" else "助手"
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(f"{role}：{content}")
    parts.append("助手：")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/cortex_stage3/final.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ExperimentConfig.from_dict(payload["config"])
    tokenizer = build_tokenizer(config.data, config.backbone)
    model = CortexSandwichLM(config, tokenizer.vocab_size, tokenizer=tokenizer)
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    lock = threading.Lock()
    started = time.time()
    state = {"requests": 0, "tokens": 0}

    app = Flask(__name__)

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "checkpoint": args.checkpoint,
                "device": device,
                "ticks_per_token": int(config.brain.ticks_per_token),
                "uptime_seconds": time.time() - started,
                "requests": state["requests"],
                "tokens": state["tokens"],
            }
        )

    @app.post("/api/chat")
    def chat():
        body = request.get_json(force=True)
        prompt = build_prompt(body.get("messages", []))
        ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device
        )
        with lock, torch.no_grad():
            generated = model.generate(
                ids,
                max_new_tokens=int(body.get("max_new_tokens", 64)),
                temperature=float(body.get("temperature", 0.7)),
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = generated[0, ids.shape[1] :].tolist()
        return jsonify(
            {
                "text": tokenizer.decode(new_ids),
                "tokens": len(new_ids),
                "prompt": prompt,
            }
        )

    @app.post("/api/chat/stream")
    def chat_stream():
        body = request.get_json(force=True)
        messages = body.get("messages", [])
        prompt = build_prompt(messages)
        ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device
        )
        max_new = int(body.get("max_new_tokens", 64))
        temperature = float(body.get("temperature", 0.7))
        top_p = float(body.get("top_p", 0.95))
        repetition_penalty = float(body.get("repetition_penalty", 1.1))

        def generate():
            generated = ids
            generated_ids: list[int] = []
            previous = ""
            begin = time.perf_counter()
            with lock, torch.no_grad():
                state["requests"] += 1
                for _ in range(max_new):
                    output = model(generated)
                    logits = output.logits[:, -1].float()
                    if temperature <= 0:
                        token = int(logits.argmax(dim=-1))
                    else:
                        logits = logits / temperature
                        if repetition_penalty > 1 and generated_ids:
                            seen = torch.tensor(
                                sorted(set(generated_ids)),
                                dtype=torch.long,
                                device=device,
                            )
                            selected = logits[0, seen]
                            logits[0, seen] = torch.where(
                                selected > 0,
                                selected / repetition_penalty,
                                selected * repetition_penalty,
                            )
                        if 0 < top_p < 1:
                            sorted_logits, sorted_indices = torch.sort(
                                logits, descending=True, dim=-1
                            )
                            cumulative = torch.cumsum(
                                torch.softmax(sorted_logits, dim=-1), dim=-1
                            )
                            remove = cumulative > top_p
                            remove[..., 1:] = remove[..., :-1].clone()
                            remove[..., 0] = False
                            logits = logits.masked_fill(
                                remove.scatter(1, sorted_indices, remove), -1e9
                            )
                        probabilities = torch.softmax(logits, dim=-1)
                        token = int(torch.multinomial(probabilities, 1))
                    generated_ids.append(token)
                    generated = torch.cat(
                        [generated, torch.tensor([[token]], device=device)],
                        dim=1,
                    )
                    text = tokenizer.decode(generated_ids)
                    delta = text[len(previous) :]
                    previous = text
                    if delta:
                        yield "data: " + json.dumps(
                            {"type": "token", "text": delta},
                            ensure_ascii=False,
                        ) + "\n\n"
                    if token == tokenizer.eos_token_id:
                        break
            seconds = time.perf_counter() - begin
            state["tokens"] += len(generated_ids)
            yield "data: " + json.dumps(
                {
                    "type": "done",
                    "tokens": len(generated_ids),
                    "seconds": seconds,
                    "device": device,
                },
                ensure_ascii=False,
            ) + "\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    print(
        f"serving {args.checkpoint} on http://{args.host}:{args.port} ({device})",
        flush=True,
    )
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
