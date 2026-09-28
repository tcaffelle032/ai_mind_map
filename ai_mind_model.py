#!/usr/bin/env python3
"""Mapping Meaning: a local, offline 3D vector teaching simulation.

Run: python3 ai_mind_model.py
Optional: python3 ai_mind_model.py --nodes 100 --seed 42
Python 3.9+; standard library only. Open the printed address in a browser.
"""

import argparse
import json
import math
import random
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


MAX_NODES = 10000
COLORS = {"Fruit": "#85a67e", "Animals": "#c37455", "Transport": "#d4aa43",
          "People": "#718fb1", "Characters": "#af86ac", "Random": "#91a5b1"}

# Deliberately invented coordinates, not embeddings from a trained AI model.
WORDS = {
    "Fruit": [("apple", [-6, 5, 3]), ("banana", [-5, 5.5, 3.5]),
              ("pear", [-6.8, 4.5, 2.5]), ("orange", [-5.5, 6.5, 2]),
              ("peach", [-7, 6, 3.8]), ("grape", [-4.5, 5, 4.5])],
    "Animals": [("cat", [-5, -5, -3]), ("dog", [-4, -5.5, -2]),
                ("wolf", [-3.2, -6.2, -3]), ("lion", [-5.5, -7, -4]),
                ("tiger", [-6.5, -6, -4.5]), ("kitten", [-5.8, -4.6, -2.8]),
                ("puppy", [-3.8, -4.8, -1.5])],
    "Transport": [("car", [5, -5, 3]), ("bus", [6, -4.5, 3.5]),
                  ("train", [7, -5.5, 2.5]), ("truck", [5.7, -6, 4]),
                  ("bicycle", [4, -4, 2]), ("plane", [7, -3.8, 5])],
    "People": [("man", [-3, 2, 0]), ("woman", [-3, 4, 0]),
               ("king", [3, 2, 0]), ("queen", [3, 4, 0]),
               ("prince", [2.8, 2, -1.5]), ("princess", [2.8, 4, -1.5])],
}
CHARACTERS = [("Warrior", [9, 1, 3]), ("Barbarian", [10, 0, 2]),
              ("Wizard", [2, 10, 1]), ("Knight", [8, 2, 4]),
              ("Rogue", [4, 2, 10]), ("Ranger", [6, 3, 8])]


def integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a whole number from {minimum:,} to {maximum:,}.")
    return value


def finite_number(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a number.")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return float(value)


class VectorModel:
    """All coordinates, vector arithmetic, scoring, and sampling run in Python."""

    def __init__(self, seed=42, preset="words", random_nodes=0):
        self.seed = seed
        self.reset(preset, random_nodes)

    def append_node(self, label, xyz, group):
        self.nodes.append({"id": len(self.nodes), "label": label,
                           "xyz": list(xyz), "group": group, "color": COLORS[group]})

    def reset(self, preset, random_nodes=0):
        if preset not in ("words", "characters", "empty"):
            raise ValueError("Choose words, characters, or empty.")
        integer(random_nodes, "Node count", 0, MAX_NODES)
        base_count = sum(map(len, WORDS.values())) if preset == "words" else len(CHARACTERS) if preset == "characters" else 0
        if base_count + random_nodes > MAX_NODES:
            raise ValueError(f"The scene can contain at most {MAX_NODES:,} nodes in total.")
        self.preset = preset
        self.rng = random.Random(self.seed)
        self.sample_rng = random.Random(self.seed + 1)
        self.nodes = []
        self.random_count = 0
        if preset == "words":
            for group, entries in WORDS.items():
                for label, xyz in entries:
                    self.append_node(label, xyz, group)
        elif preset == "characters":
            for label, xyz in CHARACTERS:
                self.append_node(label, xyz, "Characters")
        self.add_random(random_nodes)

    @property
    def bounds(self):
        return [0, 10] if self.preset == "characters" else [-10, 10]

    def add_random(self, count):
        integer(count, "Node count", 0, MAX_NODES)
        if len(self.nodes) + count > MAX_NODES:
            raise ValueError(f"This scene has {len(self.nodes):,} nodes. The maximum is {MAX_NODES:,}.")
        low, high = self.bounds
        for _ in range(count):
            self.random_count += 1
            self.append_node(f"node_{self.random_count:04d}",
                             [self.rng.uniform(low, high) for _ in range(3)], "Random")

    def snapshot(self):
        return {"nodes": self.nodes, "preset": self.preset, "bounds": self.bounds,
                "axes": ["Strength", "Magic", "Stealth"] if self.preset == "characters" else ["X", "Y", "Z"],
                "seed": self.seed, "max_nodes": MAX_NODES}

    def resolve(self, text):
        """Average known terms, or add/subtract named vectors. Never use eval."""
        if not isinstance(text, str) or len(text) > 500:
            raise ValueError("Enter up to 500 characters.")
        text = text.strip().lower().replace("−", "-")
        if not text:
            raise ValueError("Enter a word from the scene or click a sphere.")
        lookup = {n["label"].lower(): n for n in self.nodes}
        equation = "+" in text or "-" in text
        if equation:
            if not re.fullmatch(r"[a-z0-9_]+(?:\s*[+-]\s*[a-z0-9_]+)+", text):
                raise ValueError("Use a vector equation such as king - man + woman.")
            tokens = re.findall(r"[a-z0-9_]+|[+-]", text)
            terms, sign = [], 1
            for token in tokens:
                if token in ("+", "-"):
                    sign = 1 if token == "+" else -1
                else:
                    if token not in lookup:
                        raise ValueError(f"'{token}' is not in this scene. Click a sphere to see its name.")
                    terms.append((lookup[token], sign))
        else:
            if not re.fullmatch(r"[a-z0-9_]+(?:[\s,]+[a-z0-9_]+)*", text):
                raise ValueError("Separate words with spaces or commas, or use + and - for an equation.")
            tokens = re.findall(r"[a-z0-9_]+", text)
            unknown = [t for t in tokens if t not in lookup]
            if unknown:
                raise ValueError("Not in this scene: " + ", ".join(dict.fromkeys(unknown)) + ". Use the labels on the spheres.")
            if not tokens:
                raise ValueError("Enter at least one word from the scene.")
            terms = [(lookup[t], 1 / len(tokens)) for t in tokens]
        xyz = [sum(n["xyz"][i] * weight for n, weight in terms) for i in range(3)]
        return xyz, sorted({n["id"] for n, _ in terms}), "Vector equation" if equation else "Average of input vectors"

    def evaluate(self, data):
        if "text" in data:
            xyz, excluded, explanation = self.resolve(data["text"])
        else:
            xyz = data.get("xyz")
            if not isinstance(xyz, list) or len(xyz) != 3:
                raise ValueError("A vector must contain three numbers.")
            xyz = [finite_number(x, "Coordinate", -10000, 10000) for x in xyz]
            excluded = data.get("exclude", [])
            if not isinstance(excluded, list) or len(excluded) > MAX_NODES:
                raise ValueError("Invalid input nodes.")
            for n in excluded:
                integer(n, "Input node", 0, MAX_NODES - 1)
            explanation = "Coordinates selected in the scene"
        metric = data.get("metric", "distance")
        if metric not in ("distance", "cosine"):
            raise ValueError("Choose distance or cosine similarity.")
        temperature = finite_number(data.get("temperature", 0.8), "Temperature", 0.1, 3)
        qnorm = math.sqrt(sum(v * v for v in xyz))
        if metric == "cosine" and qnorm < 1e-12:
            raise ValueError("The zero vector has no direction. Move it away from the origin or use distance.")
        include_random = data.get("include_random", False)
        if not isinstance(include_random, bool):
            raise ValueError("Include random nodes must be true or false.")
        rows = []
        excluded_set = set(excluded)
        for node in self.nodes:
            if node["id"] in excluded_set or (node["group"] == "Random" and not include_random):
                continue
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(xyz, node["xyz"])))
            norm = math.sqrt(sum(v * v for v in node["xyz"]))
            cosine = max(-1, min(1, sum(a * b for a, b in zip(xyz, node["xyz"])) / (qnorm * norm))) if qnorm > 1e-12 and norm > 1e-12 else None
            if metric == "cosine" and cosine is None:
                continue
            score = -distance if metric == "distance" else 6 * cosine
            rows.append({"id": node["id"], "label": node["label"], "xyz": node["xyz"],
                         "color": node["color"], "distance": distance, "cosine": cosine, "score": score})
        if not rows:
            raise ValueError("No candidates available. Add nodes, include random nodes, or use fewer input words.")
        highest = max(r["score"] for r in rows)
        weights = [math.exp((r["score"] - highest) / temperature) for r in rows]
        total = sum(weights)
        for row, weight in zip(rows, weights):
            row["probability"] = weight / total
        rows.sort(key=lambda r: r["probability"], reverse=True)
        return {"xyz": xyz, "exclude": excluded, "explanation": explanation,
                "metric": metric, "temperature": temperature, "candidate_count": len(rows),
                "candidates": rows[:8], "visible_probability": sum(r["probability"] for r in rows[:8])}, rows

    def action(self, data):
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        op = data.get("action")
        if op == "add":
            self.add_random(data.get("count"))
            return self.snapshot()
        if op == "reset":
            self.reset(data.get("preset", "words"))
            return self.snapshot()
        if op in ("query", "sample"):
            result, rows = self.evaluate(data)
            if op == "sample":
                result["chosen"] = self.sample_rng.choices(rows, weights=[r["probability"] for r in rows], k=1)[0]
            return result
        raise ValueError("Unknown action.")


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mapping Meaning · AI vector explorer</title>
<style>
:root{--paper:#f5f4ed;--ink:#153344;--muted:#667878;--line:#d8ded5;--sage:#59775c;--yellow:#e8c463}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{height:90px;display:flex;align-items:center;justify-content:space-between;padding:0 30px;border-bottom:1px solid var(--line);gap:20px}
.eyebrow{text-transform:uppercase;letter-spacing:.16em;font-size:10px;font-weight:700;color:var(--sage)}h1{font-family:Georgia,serif;font-size:30px;letter-spacing:-.7px;margin:1px 0}header p{margin:0;color:var(--muted);font-size:12px}
.badge{border:1px solid #c7d3c4;border-radius:30px;padding:7px 12px;font-size:11px;white-space:nowrap;color:var(--sage)}
main{display:grid;grid-template-columns:290px minmax(350px,1fr) 290px;min-height:calc(100vh - 90px)}
aside{padding:22px 22px 16px;background:#fafaf5}aside.left{border-right:1px solid var(--line)}aside.right{border-left:1px solid var(--line)}
h2{font-family:Georgia,serif;font-weight:normal;font-size:21px;margin:0 0 14px}h3{font-size:12px;margin:0 0 8px;font-weight:650}section+section{border-top:1px solid var(--line);margin-top:21px;padding-top:19px}
label{display:block;font-size:12px;margin-bottom:6px}.muted,small{color:var(--muted);font-size:12px}p{margin:8px 0 12px}.row{display:flex;gap:7px;align-items:center}.row>*{min-width:0}.row input{flex:1;width:100%}
input,select,button{font:inherit;color:inherit}input[type=text],input[type=number],select{border:1px solid #c9d1c8;background:#fff;border-radius:6px;padding:9px 10px;width:100%;outline:none}input:focus,select:focus{border-color:var(--sage);box-shadow:0 0 0 2px #59775c20}
button{border:1px solid #c9d1c8;background:transparent;border-radius:6px;padding:9px 12px;cursor:pointer;white-space:nowrap;font-size:12px}button:hover{background:#e8ece3}button:disabled{opacity:.45;cursor:wait}.primary{background:var(--ink);border-color:var(--ink);color:white}.primary:hover{background:#285064}.full{width:100%}.examples{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px}.examples button{font-size:10px;padding:4px 7px;border-radius:4px}
.check{display:flex;align-items:center;gap:7px;margin:9px 0;font-size:12px}.check input{accent-color:var(--sage)}input[type=range]{accent-color:var(--sage);width:100%;margin:5px 0 10px}.range-label{display:flex;justify-content:space-between;margin-bottom:0}.range-label output{font-variant-numeric:tabular-nums;color:var(--sage)}
.stage{position:relative;min-height:680px;overflow:hidden;background:radial-gradient(ellipse at 45% 40%,#fffefa 0,#eff1e9 95%)}canvas{width:100%;height:100%;display:block;position:absolute;inset:0;touch-action:none;cursor:grab}canvas:active{cursor:grabbing}
.scene-top{position:absolute;top:20px;left:24px;right:24px;pointer-events:none;display:flex;justify-content:space-between;align-items:start;gap:15px}.scene-title{font-size:12px;font-weight:650}.scene-title small{display:block;font-size:11px;font-weight:normal;margin-top:3px}.scene-top button{pointer-events:auto;background:#fafaf1dd;padding:6px 9px;font-size:11px}
.scene-bottom{position:absolute;bottom:20px;left:24px;right:24px;pointer-events:none}.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:11px;margin-bottom:13px;color:#506565}.dot{width:8px;height:8px;display:inline-block;border-radius:50%;margin-right:4px}.hint{font-size:11px;color:var(--muted)}
.tooltip{display:none;position:absolute;pointer-events:none;background:#153344ee;color:#fff;padding:8px 10px;border-radius:6px;font-size:11px;z-index:3;max-width:220px}.tooltip b{font-size:13px}.tooltip span{display:block;opacity:.8}
.query-card{border-left:3px solid var(--yellow);padding-left:12px;margin:14px 0}.vector{font:20px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:-.8px}.formula{font-size:11px;color:var(--muted);min-height:32px}.candidate{border:0;padding:8px 0;display:block;width:100%;text-align:left;border-radius:0}.candidate:hover{background:#eff2e8}.candidate-head{display:flex;justify-content:space-between;gap:6px;font-size:12px}.candidate-head span:last-child{font-variant-numeric:tabular-nums;color:var(--muted)}.bar{background:#e6eae2;height:4px;margin-top:5px;border-radius:4px}.bar i{display:block;height:4px;border-radius:4px}.chosen{color:var(--sage);font-family:Georgia,serif;font-size:24px;min-height:35px;overflow-wrap:anywhere}.status{font-size:12px;color:var(--muted);min-height:36px;margin:6px 0 10px}.error{color:#a14035}.footnote{font-size:11px;color:var(--muted)}details{font-size:11px;color:var(--muted)}summary{cursor:pointer;color:var(--ink);font-weight:600}details p{line-height:1.6;margin-bottom:4px}
@media(min-width:1500px){main{grid-template-columns:310px minmax(350px,1fr) 320px}aside{padding:26px}}
@media(max-width:1100px){main{grid-template-columns:260px minmax(350px,1fr)}aside.right{grid-column:1/-1;border-top:1px solid var(--line);display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px}.right section+section{border-top:0;margin-top:0;padding-top:0}.stage{min-height:720px}}
@media(max-width:680px){header{padding:16px;height:auto}.badge{display:none}h1{font-size:27px}main{display:flex;flex-direction:column}.stage{min-height:470px;order:-1}.left{border-top:1px solid var(--line)}aside.right{display:block}.right section+section{border-top:1px solid var(--line);margin-top:20px;padding-top:20px}}
</style></head><body>
<header><div><div class="eyebrow">An interactive vector lab</div><h1>Mapping Meaning</h1><p>A 3D teaching model of words, coordinates, and word choice.</p></div><span class="badge">Runs locally with Python</span></header>
<main><aside class="left">
<section><h2>Build your space</h2><label for="preset">Lesson</label><div class="row"><select id="preset"><option value="words">Word neighborhoods</option><option value="characters">Character coordinates</option><option value="empty">Empty space</option></select><button id="load">Load</button></div><p class="muted">Loading a lesson replaces the current scene.</p>
<label for="count">Number of random spheres to add</label><div class="row"><input id="count" type="number" value="50" min="1" max="10000" step="1"><button id="add" class="primary">Add nodes</button></div><p class="muted">Random nodes have random coordinates and numbered names.</p></section>
<section><h2>Find a word</h2><label for="query">One word, several words, or an equation</label><input id="query" type="text" value="apple" maxlength="500" placeholder="king - man + woman"><div class="examples" id="examples"></div><button id="map" class="primary full" style="margin-top:10px">Map input vector</button><p class="muted">Multiple words are averaged. Equations add or subtract their vectors.</p></section>
<section><h3>Or move the query vector</h3><div id="sliders"></div><label for="metric">Compare using</label><select id="metric"><option value="distance">Distance · nearby positions</option><option value="cosine">Cosine · similar directions</option></select><label class="check"><input id="randomCandidates" type="checkbox">Include random nodes as candidates</label></section>
<section><label class="check"><input id="labels" type="checkbox" checked>Show word labels</label><label class="check"><input id="vectors" type="checkbox" checked>Show nearby connections</label><label class="check"><input id="rotate" type="checkbox">Slowly rotate the space</label><details><summary>What this model represents</summary><p>Each sphere represents one word vector or character vector. Random spheres are extra mathematical points. These are invented coordinates, not neurons or learned embeddings.</p><p>Real language models use many dimensions and process context with learned network layers. They produce scores for tokens, then select from probabilities. This lab substitutes simple vector matching to make that last idea visible.</p></details></section>
</aside>
<div class="stage" id="stage"><canvas id="scene" aria-label="Interactive 3D vector space. Drag to rotate, scroll to zoom, or select words using the text input." role="img"></canvas><div class="scene-top"><div class="scene-title"><span id="sceneName">Word neighborhoods</span><small id="nodeCount">Loading…</small></div><button id="resetView">Reset view</button></div><div id="tooltip" class="tooltip"></div><div class="scene-bottom"><div id="legend" class="legend"></div><div class="hint">Drag to orbit · Scroll or pinch to zoom · Click a sphere to inspect</div></div></div>
<aside class="right">
<section><div class="eyebrow">01 / Vector</div><h2 style="margin-top:5px">Your query</h2><div class="query-card"><div class="vector" id="coordinates">[ —, —, — ]</div><div class="formula" id="queryNote">Select a word to see its coordinates.</div></div><div id="selection" class="muted">The gold arrow points from the origin to the query.</div></section>
<section><div class="eyebrow">02 / Compare</div><h2 style="margin-top:5px">Possible choices</h2><div class="muted" id="metricNote">Closer points receive higher scores.</div><div id="candidates"></div><p id="coverage" class="footnote"></p></section>
<section><div class="eyebrow">03 / Choose</div><h2 style="margin-top:5px">Watch a choice</h2><label class="range-label" for="temperature">Temperature <output id="tempValue">0.80</output></label><input id="temperature" type="range" min="0.1" max="3" step="0.05" value="0.8"><p class="muted">Lower favors the highest score. Higher spreads the probabilities.</p><button class="primary full" id="sample">Animate a choice</button><div id="status" class="status" role="status" aria-live="polite">Ready to explore.</div><div id="chosen" class="chosen"></div><div class="footnote">Choice probabilities are for this toy simulation. Input words are excluded from the choices.</div></section>
</aside></main>
<script>
'use strict';
const $=id=>document.getElementById(id), canvas=$('scene'), ctx=canvas.getContext('2d');
let state=null,result=null,querySource={text:'apple'},queryVector=[0,0,0],selected=null,hovered=null;
let yaw=-0.58,pitch=0.36,zoom=1,projected=[],animation=null,busy=false,dirty=true,W=0,H=0;
let pointer=null,dragged=false,dragDistance=0,queryRevision=0,timer=null;
const pointers=new Map();let pinchDistance=0;
const coord=v=>'[ '+v.map(n=>n.toFixed(1)).join(', ')+' ]';
function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error)}
function clearResult(){result=null;animation=null;$('candidates').replaceChildren();$('coverage').textContent='';$('chosen').textContent='';dirty=true}
async function api(action){const r=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(action)});const data=await r.json();if(!r.ok)throw Error(data.error||'Request failed.');return data}
function settings(){return{...querySource,metric:$('metric').value,temperature:Number($('temperature').value),include_random:$('randomCandidates').checked}}
function setBusy(on){busy=on;for(const id of ['add','load','map','sample'])$(id).disabled=on}
async function transact(fn){if(busy)return;setBusy(true);try{await fn()}catch(e){status(e.message,true)}finally{setBusy(false)}}
function setSliders(){for(let i=0;i<3;i++){const el=$('axis'+i);el.min=Math.min(state.bounds[0],queryVector[i]);el.max=Math.max(state.bounds[1],queryVector[i]);el.value=queryVector[i];$('axisValue'+i).textContent=queryVector[i].toFixed(1)}$('coordinates').textContent=coord(queryVector)}
function buildScene(data){
 state=data;clearResult();selected=null;hovered=null;queryVector=state.preset==='characters'?[8,2,4]:[0,0,0];
 $('preset').value=state.preset;$('sceneName').textContent={words:'Word neighborhoods',characters:'Strength, Magic & Stealth',empty:'An open vector space'}[state.preset];
 $('nodeCount').textContent=state.nodes.length.toLocaleString()+' spheres · 3 dimensions';
 $('sliders').replaceChildren();state.axes.forEach((name,i)=>{const label=document.createElement('label');label.className='range-label';label.htmlFor='axis'+i;label.textContent=name+' ';const out=document.createElement('output');out.id='axisValue'+i;label.append(out);const slider=document.createElement('input');slider.type='range';slider.id='axis'+i;slider.step='0.1';slider.addEventListener('input',()=>{queryVector[i]=Number(slider.value);querySource={xyz:[...queryVector]};selected=null;setSliders();scheduleQuery()});$('sliders').append(label,slider)});setSliders();
 $('legend').replaceChildren();const groups=[...new Set(state.nodes.map(n=>n.group))];for(const group of groups){const el=document.createElement('span'),dot=document.createElement('i');dot.className='dot';dot.style.background=state.nodes.find(n=>n.group===group).color;el.append(dot,document.createTextNode(group));$('legend').append(el)}
 $('examples').replaceChildren();for(const text of state.preset==='words'?['apple','cat dog','king - man + woman']:state.preset==='characters'?['Warrior','Warrior Barbarian','Wizard']:[]){const b=document.createElement('button');b.textContent=text;b.onclick=()=>{$('query').value=text;mapInput()};$('examples').append(b)}
 dirty=true;
}
function showResult(data){
 result=data;queryVector=data.xyz;setSliders();$('queryNote').textContent=data.explanation;
 $('metricNote').textContent=data.metric==='distance'?'Closer points receive higher scores.':'Similar directions from the origin receive higher scores.';
 $('candidates').replaceChildren();const peak=data.candidates[0]?.probability||1;
 for(const row of data.candidates){const button=document.createElement('button');button.className='candidate';button.title='Inspect '+row.label+' as a new query';const head=document.createElement('div');head.className='candidate-head';const label=document.createElement('span');label.textContent=row.label;const value=document.createElement('span');value.textContent=(row.probability*100).toFixed(1)+'%';head.append(label,value);const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('i');fill.style.width=(100*row.probability/peak)+'%';fill.style.background=row.color;bar.append(fill);button.append(head,bar);button.onclick=()=>selectNode(state.nodes.find(n=>n.id===row.id));$('candidates').append(button)}
 $('coverage').textContent='Showing '+data.candidates.length+' of '+data.candidate_count+' candidates ('+(data.visible_probability*100).toFixed(1)+'% of total probability). Bars scale to the leading choice.';
 $('selection').textContent=selected!==null?'Selected: '+state.nodes.find(n=>n.id===selected).label+'. Gold marks the query vector.':'The gold arrow points from the origin to the query.';
 dirty=true;
}
async function updateQuery(){const rev=++queryRevision;const data=settings();clearResult();try{const next=await api({action:'query',...data});if(rev!==queryRevision)return;showResult(next);status('Ready. Animate a choice to sample these probabilities.')}catch(e){if(rev===queryRevision){$('queryNote').textContent='No result for this input.';status(e.message,true)}}}
function scheduleQuery(){queryRevision++;clearResult();clearTimeout(timer);timer=setTimeout(updateQuery,100);dirty=true}
function mapInput(){if(busy)return;querySource={text:$('query').value.trim()};selected=null;clearTimeout(timer);transact(updateQuery)}
function selectNode(n){if(!n||busy)return;selected=n.id;queryVector=n.xyz.slice();querySource={xyz:n.xyz.slice(),exclude:[n.id]};$('query').value=n.label;setSliders();clearTimeout(timer);updateQuery()}
$('map').onclick=mapInput;$('query').onkeydown=e=>{if(e.key==='Enter')mapInput()};
$('load').onclick=()=>transact(async()=>{queryRevision++;clearTimeout(timer);buildScene(await api({action:'reset',preset:$('preset').value}));resetView();$('randomCandidates').checked=state.preset==='empty';$('query').value=state.preset==='words'?'apple':state.preset==='characters'?'Warrior':'';querySource=state.preset==='empty'?{xyz:[0,0,0]}:{text:$('query').value};if(state.nodes.length)await updateQuery();else{setSliders();$('queryNote').textContent='Add random nodes to begin.';status('Empty scene. Enter a number and choose Add nodes.')}});
$('add').onclick=()=>transact(async()=>{const count=Number($('count').value);if(!Number.isInteger(count)||count<1||count>10000)throw Error('Enter a whole number from 1 to 10,000.');queryRevision++;clearTimeout(timer);const source=querySource,vector=queryVector.slice(),sel=selected;buildScene(await api({action:'add',count}));querySource=source;queryVector=vector;selected=sel;setSliders();await updateQuery();if(result)status('Added '+count.toLocaleString()+' random spheres. '+state.nodes.length.toLocaleString()+' total.');});
for(const id of ['metric','randomCandidates'])$(id).onchange=scheduleQuery;
$('temperature').oninput=()=>{$('tempValue').textContent=Number($('temperature').value).toFixed(2);scheduleQuery()};
for(const id of ['labels','vectors','rotate'])$(id).onchange=()=>{dirty=true};
function resetView(){yaw=-0.58;pitch=0.36;zoom=1;dirty=true}$('resetView').onclick=resetView;
$('sample').onclick=()=>transact(async()=>{clearTimeout(timer);const rev=++queryRevision;const data=await api({action:'sample',...settings()});if(rev!==queryRevision)return;showResult(data);$('chosen').textContent='';animation={start:performance.now(),from:[0,0,0],query:data.xyz,to:data.chosen.xyz,chosen:data.chosen};status('1 / Follow the input vector into the space.');dirty=true});

// Lightweight 3D renderer: perspective projection, depth sorting and shaded spheres.
// Python owns the model; this canvas provides the interactive camera and animation.
function projection(point){const center=state?.preset==='characters'?5:0;let [x,y,z]=point.map(v=>v-center);const a=x*Math.cos(yaw)-y*Math.sin(yaw),b=x*Math.sin(yaw)+y*Math.cos(yaw);const up=z*Math.cos(pitch)-b*Math.sin(pitch),depth=z*Math.sin(pitch)+b*Math.cos(pitch);const perspective=55/(55+depth);const scale=Math.min(W,H)*.029*zoom;return{x:W*.5+a*scale*perspective,y:H*.48-up*scale*perspective,depth,k:scale*perspective,visible:55+depth>2}}
function line(a,b,color,width=1,dash=[]){const p=projection(a),q=projection(b);if(!p.visible||!q.visible)return;ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([])}
function arrow(a,b,color){line(a,b,color,2.8);const p=projection(a),q=projection(b),angle=Math.atan2(q.y-p.y,q.x-p.x);if(!p.visible||!q.visible||Math.hypot(p.x-q.x,p.y-q.y)<2)return;ctx.beginPath();ctx.moveTo(q.x,q.y);ctx.lineTo(q.x-12*Math.cos(angle-.42),q.y-12*Math.sin(angle-.42));ctx.lineTo(q.x-12*Math.cos(angle+.42),q.y-12*Math.sin(angle+.42));ctx.closePath();ctx.fillStyle=color;ctx.fill()}
const sprites=new Map();function sphere(p,r,color){if(!p.visible)return;if(!sprites.has(color)){const c=document.createElement('canvas');c.width=c.height=80;const g=c.getContext('2d'),grad=g.createRadialGradient(28,23,2,40,40,38);grad.addColorStop(0,'#ffffff');grad.addColorStop(.25,color);grad.addColorStop(.78,color);grad.addColorStop(1,'#264049');g.fillStyle=grad;g.beginPath();g.arc(40,40,38,0,Math.PI*2);g.fill();sprites.set(color,c)}ctx.drawImage(sprites.get(color),p.x-r,p.y-r,r*2,r*2)}
function ring(p,r,color){ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.lineWidth=2;ctx.strokeStyle=color;ctx.stroke()}
function grid(){const[low,high]=state.bounds,step=(high-low)/10;for(let v=low;v<=high+.001;v+=step){line([low,v,low],[high,v,low],'#15334412');line([v,low,low],[v,high,low],'#15334412')}const o=[0,0,0];for(let i=0;i<3;i++){const end=[0,0,0],start=[0,0,0];end[i]=high+1;start[i]=low;line(start,end,['#89a484','#bd7c61','#bd9a42'][i],1.25);const p=projection(end);ctx.fillStyle='#52666a';ctx.font='11px -apple-system, sans-serif';ctx.fillText(state.axes[i],p.x+6,p.y-6)}const p=projection(o);sphere(p,2.5,'#153344')}
function draw(now){ctx.clearRect(0,0,W,H);if(!state)return;grid();const top=new Set(result?.candidates.slice(0,5).map(n=>n.id)||[]);if(result&&$('vectors').checked)for(const n of result.candidates.slice(0,5))line(queryVector,n.xyz,'#71857580',1,[3,4]);
 projected=state.nodes.map(n=>({...projection(n.xyz),node:n})).sort((a,b)=>b.depth-a.depth);
 const occupied=[],labelList=[];for(const p of projected){const n=p.node;let r=n.group==='Random'?Math.max(2,Math.min(5,p.k*.18)):Math.max(5,Math.min(13,p.k*.38));p.r=r;const important=n.id===selected||n.id===hovered||top.has(n.id)||n.id===animation?.chosen.id;sphere(p,r,n.color);if(important)ring(p,r+4,n.id===selected?'#b99224':'#59775c');if(p.visible&&($('labels').checked&&n.group!=='Random'||important))labelList.push({p,important})}
 labelList.sort((a,b)=>Number(b.important)-Number(a.important));let shown=0;for(const{p,important}of labelList){if(shown>=45&&!important)continue;const text=p.node.label;ctx.font=(important?'600 ':'')+'12px -apple-system, sans-serif';const width=ctx.measureText(text).width,x=p.x+p.r+6,y=p.y-3;const rect=[x,y-12,x+width,y+3];if(!important&&occupied.some(b=>rect[0]<b[2]+4&&rect[2]>b[0]-4&&rect[1]<b[3]+2&&rect[3]>b[1]-2))continue;occupied.push(rect);ctx.lineWidth=3;ctx.strokeStyle='#f5f4eddd';ctx.strokeText(text,x,y);ctx.fillStyle='#24404d';ctx.fillText(text,x,y);shown++}
 if(result){arrow([0,0,0],queryVector,'#cca22e');const p=projection(queryVector);ring(p,10,'#cca22e');}
 if(animation){const t=(now-animation.start)/1000;let point;if(t<.9){point=animation.query.map(v=>v*Math.min(1,t/.9))}else{const f=Math.min(1,(t-.9)/1.25),ease=f*f*(3-2*f);point=animation.query.map((v,i)=>v+(animation.to[i]-v)*ease);line(animation.query,animation.to,'#c59a25',2.5);if(t<2.15)status('2 / Sample one candidate using the probabilities.');else{status('3 / Chosen with '+(animation.chosen.probability*100).toFixed(1)+'% probability.');$('chosen').textContent=animation.chosen.label;}}
 const p=projection(point);sphere(p,8,'#f4cb51');ring(p,13+2*Math.sin(now*.008),'#cca22e');if(t>3.1){animation=null;dirty=true}}
}
let lastFrame=0;function tick(now){const dt=Math.min((now-lastFrame)/1000,.05);lastFrame=now;if($('rotate').checked&&!pointer){yaw+=dt*.12;dirty=true}if(dirty||animation){dirty=false;draw(now)}requestAnimationFrame(tick)}
new ResizeObserver(()=>{const rect=canvas.getBoundingClientRect();W=rect.width;H=rect.height;const dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);dirty=true}).observe($('stage'));
function local(e){const rect=canvas.getBoundingClientRect();return{x:e.clientX-rect.left,y:e.clientY-rect.top}}
function hit(pos){return projected.slice().reverse().find(p=>p.visible&&Math.hypot(p.x-pos.x,p.y-pos.y)<p.r+5)?.node}
canvas.addEventListener('pointerdown',e=>{pointers.set(e.pointerId,local(e));pointer=local(e);dragged=false;dragDistance=0;canvas.setPointerCapture(e.pointerId);$('tooltip').style.display='none';if(pointers.size===2){const p=[...pointers.values()];pinchDistance=Math.hypot(p[0].x-p[1].x,p[0].y-p[1].y);dragged=true}});
canvas.addEventListener('pointermove',e=>{const pos=local(e);if(pointers.has(e.pointerId)){pointers.set(e.pointerId,pos);if(pointers.size===2){const p=[...pointers.values()],distance=Math.hypot(p[0].x-p[1].x,p[0].y-p[1].y);if(pinchDistance>0)zoom=Math.max(.35,Math.min(3,zoom*distance/pinchDistance));pinchDistance=distance;dragged=true}else if(pointer){const dx=pos.x-pointer.x,dy=pos.y-pointer.y;dragDistance+=Math.hypot(dx,dy);if(dragDistance>4)dragged=true;yaw+=dx*.007;pitch=Math.max(-1.3,Math.min(1.3,pitch+dy*.007))}pointer=pos;dirty=true;return}const n=hit(pos);hovered=n?.id??null;dirty=true;if(n){$('tooltip').replaceChildren();const title=document.createElement('b');title.textContent=n.label;const v=document.createElement('span');v.textContent=coord(n.xyz);$('tooltip').append(title,v);$('tooltip').style.display='block';$('tooltip').style.left=Math.min(pos.x+16,W-205)+'px';$('tooltip').style.top=Math.max(65,pos.y-45)+'px'}else $('tooltip').style.display='none'});
canvas.addEventListener('pointerup',e=>{const wasPinch=pinchDistance>0;pointers.delete(e.pointerId);if(!dragged&&!wasPinch)selectNode(hit(local(e)));pointer=pointers.size?[...pointers.values()][0]:null;if(!pointers.size)pinchDistance=0});
canvas.addEventListener('pointercancel',e=>{pointers.delete(e.pointerId);pointer=null;pinchDistance=0});
canvas.addEventListener('pointerleave',()=>{hovered=null;$('tooltip').style.display='none';dirty=true});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.35,Math.min(3,zoom*Math.exp(-e.deltaY*.001)));dirty=true},{passive:false});
async function init(){try{const response=await fetch('/api/state');if(!response.ok)throw Error('Could not load the scene.');buildScene(await response.json());$('randomCandidates').checked=state.preset==='empty';$('query').value=state.preset==='characters'?'Warrior':state.preset==='words'?'apple':'';querySource=state.preset==='empty'?{xyz:[0,0,0]}:{text:$('query').value};if(state.nodes.length)await updateQuery();else status('Empty scene. Add random spheres to begin.')}catch(e){status('Unable to reach Python: '+e.message,true)}}
requestAnimationFrame(tick);init();
</script></body></html>'''


def make_server(model, port):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send_data(self, value, status=200, html=False):
            content = (value if html else json.dumps(value, allow_nan=False)).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def allowed_request(self):
            actual_port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{actual_port}", f"localhost:{actual_port}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in hosts and (origin is None or origin in {"http://" + h for h in hosts})

        def do_GET(self):
            if not self.allowed_request():
                return self.send_data({"error": "Local requests only."}, 403)
            if self.path == "/":
                return self.send_data(HTML, html=True)
            if self.path == "/api/state":
                with lock:
                    return self.send_data(model.snapshot())
            return self.send_data({"error": "Not found."}, 404)

        def do_POST(self):
            if not self.allowed_request():
                return self.send_data({"error": "Local requests only."}, 403)
            if self.path != "/api/action":
                return self.send_data({"error": "Not found."}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Request is empty or too large.")
                data = json.loads(self.rfile.read(length))
                with lock:
                    response = model.action(data)
                    self.send_data(response)
            except (ValueError, TypeError, OverflowError, RecursionError) as exc:
                self.send_data({"error": str(exc)}, 400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", type=int, default=0, help="Number of random spheres to add at startup (default: 0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable scenes (default: 42)")
    parser.add_argument("--preset", choices=["words", "characters", "empty"], default="words")
    parser.add_argument("--port", type=int, default=0, help="Local port; 0 chooses a free port (default: 0)")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535.")
    try:
        model = VectorModel(args.seed, args.preset, args.nodes)
        server = make_server(model, args.port)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"\nMapping Meaning is running at {url}\n"
          f"Scene: {len(model.nodes):,} spheres. Keep this terminal open.\n"
          "Press Ctrl+C to stop. No internet connection or extra packages needed.\n", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
