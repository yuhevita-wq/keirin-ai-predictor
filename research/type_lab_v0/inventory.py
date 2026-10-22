#!/usr/bin/env python3
"""Inventory ALL tracked source files at an immutable Git commit; no sampling."""
from __future__ import annotations
import collections
import json
import os
from pathlib import Path
import subprocess
import sys

repo = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True,exist_ok=True)
revision = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
data = subprocess.check_output(["git","-C",str(repo),"ls-tree","-r","-z","HEAD"],stderr=subprocess.STDOUT)
entries=[]
for item in data.split(b"\0"):
    if not item: continue
    meta, filename=item.split(b"\t",1)
    mode,kind,sha=meta.decode().split()
    path=filename.decode("utf-8","surrogateescape")
    entries.append({"path":path,"sha":sha,"type":kind})
entries.sort(key=lambda r:r["path"])
(out/"tree_inventory.json").write_text(json.dumps({"repository":"yuhevita-wq/keirin-market-analysis-","revision":revision,"n_files":len(entries),"files":entries},ensure_ascii=False),encoding="utf-8")
top=collections.Counter(x["path"].split("/")[0] for x in entries)
ext=collections.Counter(os.path.splitext(x["path"])[-1].lower() or "(no ext)" for x in entries)
themes=collections.Counter()
samples=collections.defaultdict(list)
for e in entries:
    s=e["path"].lower()
    kinds=[]
    for k in ("odds","result","payout","race","entr","encrypt","archive","manifest","decrypt","secret","password","grade","g3","zip","gz","parquet","sqlite","dataset"):
        if k in s: kinds.append(k)
    for k in kinds:
        themes[k]+=1
        if len(samples[k])<20: samples[k].append(e["path"])
byyear=collections.Counter()
for e in entries:
    p=e["path"]
    for year in range(2021,2027):
        if str(year) in p:
            byyear[str(year)]+=1
            break
summary={"revision":revision,"files":len(entries),"top_prefix":top.most_common(35),"extensions":ext.most_common(35),"themes":dict(themes),"year_name_occurrences":dict(byyear),"samples":dict(samples)}
(out/"inventory_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
print("INVENTORY_SUMMARY_START")
print(json.dumps(summary,ensure_ascii=False))
print("INVENTORY_SUMMARY_END")
