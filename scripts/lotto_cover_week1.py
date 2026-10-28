#!/usr/bin/env python3
"""Frozen week-1 lottery-style cover; no model fitting or outcome-based selection."""
import csv, json, os, re, sqlite3, subprocess, time
from pathlib import Path
from itertools import combinations
from collections import Counter, defaultdict
import requests
from bs4 import BeautifulSoup

OUT=Path("research/formation_cover/week_20230101_20230107")
OUT.mkdir(parents=True,exist_ok=True)
frozen=[]; exclusions=[]
for day in range(1,8):
    ds=f"2023-01-{day:02d}"
    db=Path("/tmp/lotto-week.sqlite")
    subprocess.run(["gpg","--batch","--yes","--quiet","--pinentry-mode","loopback","--passphrase-fd","0","--output",str(db),"--decrypt",f"source/encrypted/netkeirin-{ds}.sqlite.gpg"],input=os.environ["COLLECTION_KEY"].encode(),check=True)
    try:
        c=sqlite3.connect(f"file:{db}?mode=ro",uri=True);c.row_factory=sqlite3.Row
        report=json.loads(Path(f"source/reports/netkeirin-{ds}.json").read_text())
        for r in c.execute("select race_id,date,start_time,velodrome from races"):
            r=dict(r);rid=r["race_id"]
            raw=c.execute("select payload from source_records where race_id=?",(rid,)).fetchone()
            if not raw: exclusions.append(dict(r,reason="missing_source_entry"));continue
            entry=json.loads(raw[0]).get("entry",{})
            es=entry.get("entries",[]);nums=sorted(e["bike_number"] for e in es)
            if not re.fullmatch(r"\d{2}:\d{2}",r["start_time"] or "") or not 5<=len(nums)<=9 or len(set(nums))!=len(nums):
                exclusions.append(dict(r,reason="invalid_prerace_input"));continue
            groups=entry.get("line_formation") or []
            line_sets=[set(e["bike_number"] for e in g) for g in groups]
            # Fixed balanced partition: preserve as many original same-line pairs as possible.
            # Tie-break uses bike numbers only; no scores, odds, finish positions.
            candidates=[]
            for a in combinations(nums,len(nums)//2):
                A=set(a);B=set(nums)-A
                split=sum(len(g&A)*len(g&B) for g in line_sets)
                candidates.append((split,a,tuple(sorted(B))))
            _,a,b=min(candidates)
            tickets=sorted(tuple(sorted(p)) for g in (a,b) for p in combinations(g,2))
            assert all(any(set(p)<=set(t) for p in tickets) for t in combinations(nums,3))
            frozen.append(dict(r,n=len(nums),A=a,B=b,tickets=tickets,stake=100*len(tickets),
                               day_report_status=report.get("status"),input_scope="stored_prerace_entry"))
        c.close()
    finally: db.unlink(missing_ok=True)
frozen.sort(key=lambda r:(r["date"],r["start_time"],r["race_id"]))
(OUT/"frozen_tickets.json").write_text(json.dumps(frozen,ensure_ascii=False,indent=2))
(OUT/"prerace_exclusions.json").write_text(json.dumps(exclusions,ensure_ascii=False,indent=2))
print("FROZEN",len(frozen),"EXCLUDED_INPUT",len(exclusions),flush=True)

rows=[]; stop_reason=None
session=requests.Session()
for r in frozen:
    row={k:r[k] for k in ("date","start_time","race_id","velodrome","n","stake","day_report_status")}
    row.update(tickets=json.dumps(r["tickets"]),status="unresolved",paid="",profit="",hits="",reason="")
    url=f"https://keirin.netkeiba.com/race/result/?race_id={r['race_id']}"
    row["source_url"]=url
    if stop_reason:
        row["reason"]=stop_reason;rows.append(row);continue
    try:
        time.sleep(3)
        response=session.get(url,timeout=35)
        if response.status_code in (401,403,429):
            stop_reason=f"source_access_{response.status_code}"
            raise ValueError(stop_reason)
        response.raise_for_status();response.encoding=response.apparent_encoding
        soup=BeautifulSoup(response.text,"html.parser")
        identity=soup.select_one('input[name="plp-raceid"]')
        if not identity or identity.get("value")!=r["race_id"]:
            stop_reason="source_identity_or_interface_changed";raise ValueError(stop_reason)
        pays={}
        for tr in soup.select("tr.Wide"):
            combo=tr.select_one(".Result");pay=tr.select_one(".Payout")
            if not combo or not pay:continue
            ns=re.findall(r"\d+",combo.get_text())
            if len(ns)!=2:raise ValueError("unparsed_wide_combo")
            key=tuple(sorted(map(int,ns)));amount=int(re.sub(r"[^0-9]","",pay.get_text()))
            if key in pays:raise ValueError("duplicate_wide_payout")
            pays[key]=amount
        if len(pays)!=3:raise ValueError("nonstandard_or_missing_wide_payout")
        top=set(x for p in pays for x in p)
        if len(top)!=3 or set(pays)!=set(combinations(sorted(top),2)):
            raise ValueError("tie_or_inconsistent_wide_payout")
        # Missing finishers / refunds are audited against the visible result table.
        result_tables=[t for t in soup.select("table") if "着順" in t.get_text() and "選手名" in t.get_text()]
        if not result_tables:raise ValueError("missing_result_table")
        result_text=result_tables[0].get_text(" ",strip=True)
        if any(w in result_text for w in ("欠場","欠車","棄権","不成立","返還")):
            raise ValueError("refund_or_special_result_needs_review")
        payout_table=soup.select_one("tr.Wide").find_parent("table")
        if "返還" in payout_table.get_text():raise ValueError("refund_needs_review")
        chosen={tuple(p) for p in r["tickets"]}
        paid=sum(v for p,v in pays.items() if p in chosen)
        hits=len(chosen&set(pays))
        if hits<1:raise ValueError("cover_violation")
        row.update(status="settled",paid=paid,profit=paid-r["stake"],hits=hits,
                   reason="disqualification_present" if "失格" in result_text else "")
    except Exception as exc:
        row["reason"]=str(exc)[:180]
    rows.append(row)
    # Checkpoint only derived public race settlements, never plaintext source DB.
    fields=list(rows[0])
    with (OUT/"races.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    if len(rows)%20==0:print("PROGRESS",len(rows),"/",len(frozen),flush=True)

if rows:
    with (OUT/"races.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
settled=[r for r in rows if r["status"]=="settled"]
stake=sum(r["stake"] for r in settled);paid=sum(r["paid"] for r in settled)
summary={"period":["2023-01-01","2023-01-07"],"frozen_races":len(frozen),"input_exclusions":len(exclusions),
         "settled":len(settled),"unresolved":len(rows)-len(settled),"stake_yen":stake,"paid_yen":paid,
         "profit_yen":paid-stake,"return_pct":100*paid/stake if stake else None,
         "profitable_races":sum(r["profit"]>0 for r in settled),
         "loss_races":sum(r["profit"]<0 for r in settled),
         "hit_count_distribution":dict(Counter(r["hits"] for r in settled)),
         "unresolved_reasons":dict(Counter(r["reason"] for r in rows if r["status"]!="settled")),
         "stop_reason":stop_reason,"not_out_of_sample":"Jan 1 race 1 was previously observed; other races not tuned"}
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
byday=defaultdict(list)
for r in settled:byday[r["date"]].append(r)
lines=["# ロト型競輪・第1週検証","", "対象: 2023-01-01〜2023-01-07。全会場を発走日時順。実購入なし。",
"","## 事前固定したルール",
"出走表の全車を人数が均等になる2群へ分割。同ライン内ペアの分断を最小化し、同点なら車番順。各群内の全ペアをワイド各100円。確率推定・オッズによる選別・結果による変更なし。",
"これは券種間の最適化ではなく、ロト論文の鳩の巣原理を移した最初の基準実験。任意の上位3車集合を被覆。7車なら9点、6車なら6点、5車なら4点、8車なら12点、9車なら16点。",
"","## 集計範囲",
"収支は確定ワイド払戻を検証できたレースだけ。未解決・返還等の要確認レースを勝手に補完しない。未解決があれば週全体の確定収支とは呼ばない。",
"一部取得日の利用可能レースも含むが日別収集状態をCSVに明記。収集日全体の完全性・発走前時点の保存履歴は保証しない。",
"1月1日1Rは既に結果を観測済みであり、未見検証とは区別する。確定オッズは事前判断に使用しない。",
"","## 収支", "", "|日付|照合済みR|投資円|払戻円|収支円|","|---|---:|---:|---:|---:|"]
for day,rr in sorted(byday.items()):
    st=sum(r["stake"] for r in rr);pa=sum(r["paid"] for r in rr)
    lines.append(f"|{day}|{len(rr)}|{st}|{pa}|{pa-st}|")
lines+=["",json.dumps(summary,ensure_ascii=False,indent=2),"","## 何がダメだったか・限界",
"1. 被覆条件には払戻金額が入っていない。最低1点当たっても、外れ券の購入費を回収する条件はない。",
"2. 同ラインを保つ分割は高配当や回収を保証しない。今回のルールは確率に依存しないが、利益の優位性を作れたかは収支で別に評価する。",
"3. 点数は人数とともに増える。人数別・的中数別の回収を下表で確認する。",
"4. この実験では券種間比較も、事前オッズを使った購入見送りも未実装。最適券種を発見したとは言えない。",
"5. 返還・同着等の要確認、取得失敗・部分取得による除外は成績の偏りにつながる。未解決件数を必ず併記する。",
"","|人数|R数|投資円|払戻円|収支円|","|---|---:|---:|---:|---:|"]
for n in sorted(set(r["n"] for r in settled)):
    rr=[r for r in settled if r["n"]==n];st=sum(r["stake"] for r in rr);pa=sum(r["paid"] for r in rr)
    lines.append(f"|{n}|{len(rr)}|{st}|{pa}|{pa-st}|")
(OUT/"REPORT.md").write_text("\n".join(lines))
print("SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)
