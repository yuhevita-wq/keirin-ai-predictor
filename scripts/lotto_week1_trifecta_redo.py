#!/usr/bin/env python3
"""Week-one exploratory trifecta formations. Outcomes never construct candidates."""
import csv,gzip,hashlib,json,os,re,sqlite3,subprocess
from pathlib import Path
from itertools import product,permutations
from collections import defaultdict
OUT=Path("research/formation_cover/week1_trifecta_redo")
OUT.mkdir(parents=True,exist_ok=True)

def formations(entry):
    nums=sorted(e["bike_number"] for e in entry["entries"])
    groups=entry.get("line_formation") or []
    flattened=[x["bike_number"] for g in groups for x in g]
    if len(nums)!=len(set(nums)) or sorted(flattened)!=nums:raise ValueError("invalid_or_missing_line_formation")
    role={};line={}
    for i,g in enumerate(groups):
        for j,x in enumerate(g):
            b=x["bike_number"];role[b]="SOLO" if len(g)==1 else f"P{j+1}";line[b]=i
    sets={k:sorted(b for b in nums if role[b]==k) for k in sorted(set(role.values()))}
    sets["ALL"]=nums
    variants=[]
    for labels in product(sets,repeat=3):
        columns=[sets[x] for x in labels]
        tickets=sorted(set(t for t in product(*columns) if len(set(t))==3))
        if not tickets:continue
        variants.append(dict(template="-".join(labels),formation="-".join("".join(map(str,c)) for c in columns),
                             tickets=tickets,points=len(tickets)))
    return variants,role,line

def group_stats(rows):
    groups=defaultdict(list)
    for r in rows:groups[r["template"]].append(r)
    result=[]
    for key,rs in groups.items():
        stake=sum(r["stake"] for r in rs);paid=sum(r["paid"] for r in rs)
        wins=[r for r in rs if r["hit"]]
        biggest=max((r["paid"] for r in wins),default=0)
        result.append(dict(template=key,races=len(rs),hits=len(wins),stake=stake,paid=paid,
                           profit=paid-stake,return_pct=100*paid/stake if stake else None,
                           largest_payout=biggest,profit_without_largest_hit=paid-biggest-stake))
    return sorted(result,key=lambda x:(-x["profit"],x["template"]))

def main():
    # Keep the same 27 previously unresolved races excluded, as instructed.
    previous=Path("research/formation_cover/week_20230101_20230107/races.csv")
    with previous.open() as f:excluded={r["race_id"] for r in csv.DictReader(f) if r["status"]!="settled"}
    pre=[];bad=[]
    for day in range(1,8):
        ds=f"2023-01-{day:02d}";db=Path("/tmp/trifecta-redo.sqlite")
        subprocess.run(["gpg","--batch","--yes","--quiet","--pinentry-mode","loopback","--passphrase-fd","0","--output",str(db),"--decrypt",f"source/encrypted/netkeirin-{ds}.sqlite.gpg"],input=os.environ["COLLECTION_KEY"].encode(),check=True)
        try:
            c=sqlite3.connect(f"file:{db}?mode=ro",uri=True)
            for rid,date,start,venue in c.execute("select race_id,date,start_time,velodrome from races"):
                if rid in excluded:continue
                raw=c.execute("select payload from source_records where race_id=?",(rid,)).fetchone()
                if not raw:bad.append(dict(race_id=rid,reason="no_source"));continue
                entry=json.loads(raw[0]).get("entry",{})
                try:
                    if not re.fullmatch(r"\d{2}:\d{2}",start or ""):raise ValueError("missing_start_time")
                    fs,role,line=formations(entry)
                    # Full normalized input retained for comparisons; no similarity claims from role alone.
                    context=dict(grade=entry.get("grade"),distance=entry.get("distance"),
                                 line_formation=entry.get("line_formation"),
                                 entries=entry.get("entries"))
                    pre.append(dict(race_id=rid,date=date,start_time=start,venue=venue,context=context,
                                    roles=role,line_index=line,formations=fs))
                except (ValueError,KeyError,TypeError) as e:bad.append(dict(race_id=rid,reason=str(e)))
            c.close()
        finally:db.unlink(missing_ok=True)
    pre.sort(key=lambda r:(r["date"],r["start_time"],r["race_id"]))
    # Materialize candidates before any result or payout reads.
    frozen=json.dumps(pre,ensure_ascii=False,sort_keys=True).encode()
    with gzip.open(OUT/"frozen_prerace_formations.json.gz","wb") as f:f.write(frozen)
    digest=hashlib.sha256(frozen).hexdigest()
    print("FROZEN",len(pre),"races",sum(len(r["formations"]) for r in pre),"formations",digest,flush=True)
    results={}
    for day in range(1,8):
        ds=f"2023-01-{day:02d}";db=Path("/tmp/trifecta-redo.sqlite")
        subprocess.run(["gpg","--batch","--yes","--quiet","--pinentry-mode","loopback","--passphrase-fd","0","--output",str(db),"--decrypt",f"source/encrypted/netkeirin-{ds}.sqlite.gpg"],input=os.environ["COLLECTION_KEY"].encode(),check=True)
        try:
            c=sqlite3.connect(f"file:{db}?mode=ro",uri=True)
            for rid,payload in c.execute("select race_id,payload from source_records"):
                obj=json.loads(payload);p=obj.get("payouts") or {}
                ns=re.findall(r"\d+",str(p.get("sanrentan_combo","")))
                pay=p.get("sanrentan_payout")
                if len(ns)==3 and len(set(ns))==3 and isinstance(pay,(int,float)) and pay>0:
                    results[rid]=(tuple(map(int,ns)),int(pay),p.get("sanrentan_popularity"))
            c.close()
        finally:db.unlink(missing_ok=True)
    rows=[]
    for r in pre:
        if r["race_id"] not in results:bad.append(dict(race_id=r["race_id"],reason="invalid_trifecta_payout"));continue
        winner,payout,pop=results[r["race_id"]]
        roles={int(k):v for k,v in r["roles"].items()}
        line={int(k):v for k,v in r["line_index"].items()}
        es=r["context"]["entries"];lookup={e["bike_number"]:e for e in es}
        win_roles="-".join(roles.get(b,"UNKNOWN") for b in winner)
        # One key is strictly structural; exact-context key preserves all entry inputs.
        struct=json.dumps([r["context"]["grade"],[len(g) for g in r["context"]["line_formation"]],
                           [[e.get("leg_type"),e.get("class")] for g in r["context"]["line_formation"] for x in g for e in [lookup[x["bike_number"]]]]],ensure_ascii=False)
        exact=hashlib.sha256(json.dumps(r["context"],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        for f in r["formations"]:
            hit=winner in f["tickets"];stake=f["points"]*100;paid=payout if hit else 0
            rows.append(dict(race_id=r["race_id"],date=r["date"],start_time=r["start_time"],venue=r["venue"],
                             template=f["template"],formation=f["formation"],points=f["points"],stake=stake,
                             winner="-".join(map(str,winner)),winner_roles=win_roles,
                             winner_lines="-".join(str(line.get(b,"?")) for b in winner),
                             trifecta_payout=payout,popularity=pop,hit=int(hit),paid=paid,profit=paid-stake,
                             realized_return_multiple=paid/stake,structure_key=struct,exact_context_key=exact))
    if not rows:raise RuntimeError("No evaluated formations")
    with gzip.open(OUT/"all_cases.csv.gz","wt",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    stats=group_stats(rows)
    (OUT/"template_summary.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    # Cases are prompts for investigation, not selection rules or proof of mispricing.
    best=sorted((r for r in rows if r["hit"]),key=lambda r:(-r["realized_return_multiple"],r["race_id"],r["template"]))[:50]
    comparisons=[]
    for case in best:
        same=[r for r in rows if r["template"]==case["template"] and r["structure_key"]==case["structure_key"]]
        comparisons.append(dict(case=case,comparison_scope="same role-template, line-size, leg-type and class; state not matched",
                                samples=len(same),low_payout_hits=sorted([r for r in same if r["hit"]],key=lambda r:r["realized_return_multiple"])[:5],
                                misses=[r for r in same if not r["hit"]][:5],
                                hypothesis="Why was the winning role/line placement priced this way? Full losing-ticket odds and comparable state are needed before attributing market error."))
    (OUT/"high_payout_cases_with_controls.json").write_text(json.dumps(comparisons,ensure_ascii=False,indent=2))
    meta=dict(period=["2023-01-01","2023-01-07"],races=len({r["race_id"] for r in rows}),formations=len(rows),
              prior_special_exclusions=len(excluded),other_exclusions=bad,frozen_sha256=digest,
              status="exploratory_in_sample_not_trading_policy",week2_used=False)
    (OUT/"audit.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    report=["# 第1週やり直し：三連単フォーメーションと配当の観察","",
            "期間2023-01-01〜07。発走日時順。全候補を出走前の並びから作成した後、結果と照合。実購入なし。",
            "","## 観察の単位",
            "各列は先頭P1・番手P2・3番手P3・4番手P4等・単騎SOLO・全車ALLから選ぶ。全列組合せを列挙し、同一車の重複を除いて正規フォーメーションへ翻訳。固定点数上限なし。",
            "各テンプレートは独立の仮想方策。重複するテンプレートの収支を足して総収支と呼ばない。",
            "第1週の既知結果を分析する探索であり、新しい未見検証ではない。第2週は使用しない。",
            "","## 読み方",
            "all_cases.csv.gzに全的中・不的中、実車番フォーメーション、点数、費用、配当、差額を保存。",
            "high_payout_cases_with_controls.jsonは配当/総費用が大きい例と同型の低配当・不的中例。上位例だけを成功証拠にしない。",
            "構造比較はライン構成・役割・脚質・級班まで。状態等を含めた完全一致標本とは呼ばない。全出走前入力も別保存。",
            "実現払戻/購入費は事後の回収倍率であり、事前の合成オッズではない。",
            "","## 配当がついた理由について",
            "確定当たり配当だけでは全買い目の価格構造が分からない。役割配置、ライン跨ぎ、人気順位との関連を観察するが、『市場が割安だった』と断定しない。",
            "各候補の収支・標本数・最大1的中を除いた収支を併記。少数の高配当だけによる見かけの優位性を確認する。",
            "","## テンプレート別成績（収支上位20、全件はJSON）",
            "|役割フォーメーション|R数|的中|投資|払戻|収支|最大1的中を除く収支|",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for s in stats[:20]:
        report.append(f"|{s['template']}|{s['races']}|{s['hits']}|{s['stake']}|{s['paid']}|{s['profit']}|{s['profit_without_largest_hit']}|")
    report+=["","## 集計範囲",json.dumps(meta,ensure_ascii=False,indent=2)]
    (OUT/"REPORT.md").write_text("\n".join(report))
    print("DONE",json.dumps(meta,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
