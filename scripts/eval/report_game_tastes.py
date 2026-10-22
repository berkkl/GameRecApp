"""Render the complete 100-game benchmark and per-result judgments for review."""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

from .benchmark_game_tastes import DEFAULT_ROOT, read, save


def precision(rows, judgments):
    displayed = [(row["id"], g["app_id"]) for row in rows for g in row["results"]]
    judged = [judgments[key] for key in displayed if key in judgments]
    known = [j for j in judged if j["grade"] is not None]
    top_one = [
        judgments.get((row["id"], row["results"][0]["app_id"]))
        for row in rows
        if row["results"]
    ]
    return {
        "displayed_results": len(displayed),
        "judgment_coverage": len(judged) / len(displayed),
        "evidence_verified_results": len(known),
        "useful_precision_at_10_lower_bound": sum(j["grade"] >= 2 for j in known)
        / len(displayed),
        "strong_precision_at_10_lower_bound": sum(j["grade"] == 3 for j in known)
        / len(displayed),
        "wrong_gameplay_results": sum(j["grade"] == 0 for j in known),
        "adjacent_results": sum(j["grade"] == 1 for j in known),
        "uncertain_results": len(displayed) - len(known),
        "useful_precision_at_1_lower_bound": sum(
            j is not None and j["grade"] is not None and j["grade"] >= 2
            for j in top_one
        )
        / len(rows),
    }


def report(root, allow_partial, publish):
    cases = read(root / "cases.json")
    manifest = read(root / "manifest.json")
    qrels = (
        read(root / "judgments.json")
        if (root / "judgments.json").exists()
        else {"judgments": []}
    )
    judgments = {(j["case_id"], j["app_id"]): j for j in qrels["judgments"]}
    variants = {}
    for name in ("current", "dense", "reranker"):
        path = root / f"offline-{name}.json"
        variants[name] = read(path) if path.exists() else {"cases": [], "summary": {}}
    complete = all(
        len(v["cases"]) == 100
        and precision(v["cases"], judgments)["judgment_coverage"] == 1
        for v in variants.values()
    )
    if not complete and not allow_partial:
        raise ValueError(
            "All 100 cases and all displayed result judgments must be complete"
        )
    summary = {}
    for name, value in variants.items():
        if not value["cases"]:
            continue
        summary[name] = {**value["summary"], **precision(value["cases"], judgments)}
        times = sorted(row["latency_ms"] for row in value["cases"])
        summary[name]["p95_latency_ms"] = times[math.ceil(len(times) * 0.95) - 1]
        summary[name]["target_shortlist_recall_at_40"] = sum(
            row["target_shortlist_rank"] is not None for row in value["cases"]
        ) / len(value["cases"])
        summary[name]["hybrid_candidate_target_recall"] = sum(
            row["target_candidate_rank"] is not None for row in value["cases"]
        ) / len(value["cases"])
        for split in ("development", "heldout"):
            ids = {c["id"] for c in cases if c["split"] == split}
            split_rows = [row for row in value["cases"] if row["id"] in ids]
            summary[name][split] = (
                precision(split_rows, judgments) if split_rows else {}
            )
    by_variant = {
        v: {r["id"]: r for r in value["cases"]} for v, value in variants.items()
    }
    enriched = []
    for case in cases:
        row = dict(case)
        row["variants"] = {}
        for name, rows in by_variant.items():
            if case["id"] not in rows:
                continue
            result = rows[case["id"]]
            row["variants"][name] = {
                key: result[key]
                for key in (
                    "target_rank",
                    "target_group_rank",
                    "target_shortlist_rank",
                    "target_candidate_rank",
                    "latency_ms",
                )
            }
            row["variants"][name]["results"] = [
                {
                    "app_id": game["app_id"],
                    "name": game["name"],
                    "source_url": f"https://store.steampowered.com/app/{game['app_id']}/?l=english",
                    "judgment": judgments.get((case["id"], game["app_id"])),
                }
                for game in result["results"]
            ]
        enriched.append(row)
    payload = {
        "complete": complete,
        "manifest": manifest,
        "summary": summary,
        "cases": enriched,
    }
    diagnostic_path = root / "candidate-depth-diagnostic.json"
    if diagnostic_path.exists():
        diagnostic = read(diagnostic_path)
        diagnostic_rows = diagnostic["cases"]
        payload["candidate_depth_diagnostic"] = {
            "source_depth": diagnostic["requested_source_depth"],
            "cases": len(diagnostic_rows),
            "target_candidate_recall": sum(
                row["target_candidate_rank"] is not None for row in diagnostic_rows
            )
            / len(diagnostic_rows),
            "target_shortlist_recall_at_40": sum(
                row["target_candidate_rank"] is not None
                and row["target_candidate_rank"] <= 40
                for row in diagnostic_rows
            )
            / len(diagnostic_rows),
            "limitation": diagnostic["limitation"],
        }
    save(root / "report.json", payload)
    columns = [
        "name",
        "app_id",
        "type",
        "prompt",
        "requirements",
        "source_url",
        "catalog_review_count",
        "catalog_rating_pct",
        "current_target_rank",
        "dense_target_rank",
        "reranker_target_rank",
        "current_mismatches",
        "current_uncertain",
        "reranker_mismatches",
    ]
    with (root / "games-100.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for case in enriched:
            row = {key: case.get(key) for key in columns if key in case}
            row["requirements"] = "; ".join(case["requirements"])
            for variant in ("current", "dense", "reranker"):
                value = case["variants"].get(variant, {})
                row[f"{variant}_target_rank"] = (
                    value.get("target_rank") or "outside top 10"
                )
            for variant in ("current", "reranker"):
                value = case["variants"].get(variant, {})
                row[f"{variant}_mismatches"] = " | ".join(
                    f"{g['name']}: {g['judgment']['reason']}"
                    for g in value.get("results", [])
                    if g["judgment"]
                    and g["judgment"]["grade"] is not None
                    and g["judgment"]["grade"] <= 1
                )
            row["current_uncertain"] = " | ".join(
                g["name"]
                for g in case["variants"].get("current", {}).get("results", [])
                if not g["judgment"] or g["judgment"]["grade"] is None
            )
            writer.writerow(row)
    public_payload = {
        **payload,
        "manifest": {
            **manifest,
            "serving_bundle": {
                k: v
                for k, v in manifest["serving_bundle"].items()
                if k not in {"catalog", "index"}
            },
        },
    }
    data = (
        json.dumps(public_payload, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    document = TEMPLATE.replace("__DATA__", data)
    (root / "report.html").write_text(document, encoding="utf-8")
    if publish:
        shutil.copyfile(root / "report.html", "public/benchmark.html")
    print(json.dumps({"complete": complete, "summary": summary}, indent=2))


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>100-game gameplay benchmark</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#101615;color:#ecf3ec;line-height:1.5}*{box-sizing:border-box}body{margin:0;padding:32px;max-width:1480px;margin:auto}a{color:#b7e89d}h1{font-size:clamp(28px,5vw,46px);line-height:1.15;margin-bottom:12px}h2{margin-top:0}p{color:#bac8bd}button,input,select{font:inherit;background:#19231f;color:inherit;border:1px solid #405448;border-radius:8px;padding:10px 14px}button{cursor:pointer}.toolbar{display:flex;gap:12px;flex-wrap:wrap;position:sticky;top:0;background:#101615f5;padding:16px 0;z-index:2}input{flex:1;min-width:200px}table{width:100%;border-collapse:collapse;margin:20px 0}th,td{text-align:left;padding:12px;border-bottom:1px solid #344438}th{color:#b7e89d}.tablewrap{overflow-x:auto}.case{border:1px solid #344438;border-radius:14px;margin:16px 0;background:#16201b}.case>summary{padding:20px;cursor:pointer}.heading{font-size:20px;font-weight:650}.meta{font-size:13px;color:#b7c4ba}.body{padding:0 20px 22px}.prompt{border-left:3px solid #b7e89d;padding:12px 16px;background:#1e2b23;color:#eff7ee}.requirements{color:#bccbbb}.result{border-bottom:1px solid #344438;padding:14px 0}.result strong{display:block}.reason{margin:5px 0 0;font-size:14px}.grade{display:inline-block;font-size:12px;border-radius:5px;padding:3px 7px;margin-left:8px}.g3{background:#295b39}.g2{background:#375e46}.g1{background:#6a5226}.g0{background:#6a3030}.gn{background:#38423d}.variants{display:flex;gap:8px;margin:16px 0;flex-wrap:wrap}.active{border-color:#b7e89d;color:#b7e89d}.notice{background:#283623;border:1px solid #536649;border-radius:10px;padding:16px}.hidden{display:none}footer{margin:32px 0;color:#b8c8bd;font-size:13px}.mono{font-family:ui-monospace,monospace;font-size:12px;word-break:break-all}@media(max-width:600px){body{padding:16px}th,td{padding:8px}.case>summary{padding:14px}.body{padding:0 14px 18px}}
</style></head><body>
<a href="/">← Game discovery</a><h1>100 games. Specific tastes. Real searches.</h1>
<p>Every request describes gameplay without naming its expected game. Open a row to see its type, prompt, defining mechanics and an assessment of every displayed recommendation.</p>
<div id="notice" class="notice"></div><div class="tablewrap"><table><thead><tr><th>Search variant</th><th>Expected game in top 10</th><th>Expected game first</th><th>Useful first result</th><th>Useful matches across ten</th><th>Wrong / adjacent</th><th>Mean search time</th></tr></thead><tbody id="metrics"></tbody></table></div>
<p>Useful means the core gameplay and most requested mechanics fit. Wrong and adjacent results fail the gameplay request to different degrees. These are blind <strong>Grok 4.7 judgments from publisher descriptions</strong>, checked for completeness; they are not human-rated ground truth. Unverified results count as zero in the displayed precision lower bound.</p>
<div class="toolbar"><input id="search" type="search" placeholder="Search titles, types or gameplay…" aria-label="Search benchmark"><select id="filter" aria-label="Filter cases"><option value="all">All 100 games</option><option value="missing">Target missing from current top 10</option><option value="mismatch">Current results with gameplay mismatches</option><option value="improved">Reranker improves target rank</option></select><span id="count"></span></div>
<main id="cases"></main><footer id="provenance"></footer>
<script id="benchmark-data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('benchmark-data').textContent);
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels={current:'Current hybrid',dense:'Semantic only',reranker:'Hybrid + BGE reranker'};
const pct=value=>(100*value).toFixed(1)+'%';const rank=value=>value===undefined?'pending':value?'#'+value:'outside top 10';
document.getElementById('notice').textContent=data.complete?'Complete: 100 requests per variant; every displayed result assessed. The catalog and prompts were frozen before evaluation.':'Work in progress: comparisons or judgments are incomplete. Numbers shown here are provisional.';
document.getElementById('metrics').innerHTML=Object.entries(data.summary).map(([v,s])=>`<tr><td>${labels[v]} (${s.cases})</td><td>${pct(s.target_recall_at_10)}</td><td>${pct(s.target_recall_at_1)}</td><td>${pct(s.useful_precision_at_1_lower_bound)}</td><td>${pct(s.useful_precision_at_10_lower_bound)} (${pct(s.judgment_coverage)} assessed)</td><td>${s.wrong_gameplay_results} / ${s.adjacent_results}</td><td>${(s.mean_latency_ms/1000).toFixed(2)}s</td></tr>`).join('');
function results(c,v){const r=c.variants[v];if(!r)return '<p>Comparison pending.</p>';return `<p>Expected target: <strong>${rank(r.target_rank)}</strong>. Including expandable series: ${rank(r.target_group_rank)}. Target in hybrid shortlist: ${r.target_shortlist_rank?'#'+r.target_shortlist_rank:'outside top 40'}.</p>`+r.results.map((g,i)=>{const j=g.judgment;const grade=j?.grade??null;const label=grade===null?'Unverified':['Wrong gameplay','Adjacent','Useful match','Strong match'][grade];return `<div class="result"><strong>${i+1}. <a href="${esc(g.source_url)}" target="_blank" rel="noopener">${esc(g.name)}</a><span class="grade g${grade===null?'n':grade}">${label}</span></strong><p class="reason">${esc(j?.reason??'Judgment pending.')} ${j?'<span class="meta">('+esc(j.confidence)+' confidence)</span>':''}</p></div>`}).join('')}
function render(){const query=document.getElementById('search').value.toLowerCase();const filter=document.getElementById('filter').value;const selected=data.cases.filter(c=>{if(![c.name,c.type,c.prompt].join(' ').toLowerCase().includes(query))return false;const a=c.variants.current;const b=c.variants.reranker;if(filter==='missing')return !a?.target_rank;if(filter==='mismatch')return a?.results.some(g=>g.judgment?.grade!==null&&g.judgment?.grade<=1);if(filter==='improved')return b?.target_rank&&(b.target_rank<(a?.target_rank??99));return true});document.getElementById('count').textContent=selected.length+' games';document.getElementById('cases').innerHTML=selected.map(c=>`<details class="case" data-id="${c.id}"><summary><span class="heading">${esc(c.name)}</span><div class="meta">${esc(c.type)} · Current ${rank(c.variants.current?.target_rank)} · Reranker ${rank(c.variants.reranker?.target_rank)}</div></summary><div class="body"><p class="prompt">${esc(c.prompt)}</p><p class="requirements"><strong>Defining mechanics:</strong> ${esc(c.requirements.join('; '))}.</p><p class="meta"><a href="${esc(c.source_url)}" target="_blank" rel="noopener">Publisher's Steam page</a> · Catalog snapshot: ${c.catalog_review_count.toLocaleString()} reviews, ${c.catalog_rating_pct}% positive. ${esc(c.split)} case.</p><div class="variants">${Object.entries(labels).map(([v,label])=>`<button data-variant="${v}" class="${v==='current'?'active':''}">${label}</button>`).join('')}</div><div class="results">${results(c,'current')}</div></div></details>`).join('')}
document.getElementById('search').addEventListener('input',render);document.getElementById('filter').addEventListener('change',render);document.getElementById('cases').addEventListener('click',e=>{const button=e.target.closest('button[data-variant]');if(!button)return;const detail=button.closest('.case');const c=data.cases.find(c=>c.id===detail.dataset.id);detail.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b===button));detail.querySelector('.results').innerHTML=results(c,button.dataset.variant)});render();
document.getElementById('provenance').innerHTML=`<p>Limitations: this is a curated, fairly specific English benchmark. Some combinations match only one or a few games, so precision at ten also tests whether the app pads results. It does not establish Turkish quality, broad mood search or human satisfaction. Alternative games can fit when the expected title is absent. Only the ten primary cards per search are relevance-graded; series members are tracked separately for target recall. Fresh publisher evidence was used for judging, while retrieval used the frozen catalog. Publisher claims and partial excerpts can be incomplete. Latency is local CPU wall time under concurrent background work, not a GPU production benchmark.</p><p>Embedding: ${esc(data.manifest.serving_bundle.model)}. Revision <span class="mono">${esc(data.manifest.serving_bundle.model_revision)}</span>. Catalog: ${data.manifest.serving_bundle.searchable_games.toLocaleString()} searchable games; immutable checksum <span class="mono">${esc(data.manifest.serving_bundle.catalog_sha256)}</span>.</p>`;
</script></body></html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--publish-local", action="store_true")
    args = parser.parse_args()
    report(args.root, args.allow_partial, args.publish_local)
