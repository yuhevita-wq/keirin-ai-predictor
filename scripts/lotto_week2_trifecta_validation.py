#!/usr/bin/env python3
"""Apply frozen week-one templates to January 8-14, without reselection."""
import json
from pathlib import Path
import hashlib
out=Path("research/formation_cover/week2_trifecta_validation")
out.mkdir(parents=True,exist_ok=True)
prior_path=Path("research/formation_cover/week1_trifecta_redo/template_summary.json")
prior=json.loads(prior_path.read_text())
assert len(prior)==302
locked={r["template"] for r in prior}
(out/"locked_week1_templates.json").write_text(json.dumps(prior,ensure_ascii=False,indent=2))
src=Path("scripts/lotto_week1_trifecta_redo.py").read_text()
src=src.replace('week1_trifecta_redo','week2_trifecta_validation')
src=src.replace('range(1,8)','range(8,15)')
src=src.replace('["2023-01-01","2023-01-07"]','["2023-01-08","2023-01-14"]')
a=src.index('    # Keep the same 27')
b=src.index('    pre=[];bad=[]',a)
src=src[:a]+'    excluded=set()\n'+src[b:]
src=src.replace('fs,role,line=formations(entry)','fs,role,line=formations(entry)\n                    fs=[f for f in fs if f["template"] in LOCKED]')
src=src.replace('status="exploratory_in_sample_not_trading_policy",week2_used=False','status="heldout_week2_fixed_week1_templates",week2_used=True')
ns={"__name__":"week2_worker","LOCKED":locked}
exec(compile(src,"fixed_week2_worker","exec"),ns)
ns["main"]()
current=json.loads((out/"template_summary.json").read_text())
now={x["template"]:x for x in current}
comparison=[]
for p in prior:
    q=now.get(p["template"],dict(template=p["template"],races=0,hits=0,stake=0,paid=0,profit=0,return_pct=None,largest_payout=0,profit_without_largest_hit=0))
    comparison.append(dict(template=p["template"],week1=p,week2=q,selected_week1_profit=p["profit"]>0,selected_week1_without_max=p["profit_without_largest_hit"]>0))
(out/"week1_vs_week2.json").write_text(json.dumps(comparison,ensure_ascii=False,indent=2))
subsets={}
for label,predicate in (("all302",lambda p:True),("week1_profitable26",lambda p:p["profit"]>0),("week1_without_max_profitable4",lambda p:p["profit_without_largest_hit"]>0)):
    cs=[r for r in comparison if predicate(r["week1"])]
    subsets[label]=dict(templates=len(cs),applicable_week2=sum(r["week2"]["races"]>0 for r in cs),week2_positive=sum(r["week2"]["profit"]>0 for r in cs),week2_negative=sum(r["week2"]["profit"]<0 for r in cs),week2_zero=sum(r["week2"]["profit"]==0 for r in cs))
(out/"validation_summary.json").write_text(json.dumps(subsets,ensure_ascii=False,indent=2))
report=["# 第2週：第1週の302型を固定した三連単検証","",
"対象2023-01-08〜14。各点100円。同じ予定並び・役割集合・フォーメーション生成方法。実購入なし。",
"第1週黒字26型、最大1的中を除いて黒字4型は第2週の結果を読む前に固定。第2週の好成績型への選び直しなし。",
"各型は独立した仮想方策。収支を足すと重複購入になるため、型を合算した週間総収支は出さない。",
"第1週の27除外IDは第2週へ流用しない。入力不備・配当不備はaudit.jsonに記録。",
"特殊結果の返還までの監査は未実施。正の三連単確定配当と一意の3車組があるレースの払戻照合であり、返還が判明すれば精算訂正が必要。",
"", "## 固定集合ごとの黒字・赤字型数",json.dumps(subsets,ensure_ascii=False,indent=2),
"", "## 第1週黒字26型の次週成績（第1週の収支順）",
"|型|第1週収支|第2週R数|的中|投資|払戻|第2週収支|回収率|",
"|---|---:|---:|---:|---:|---:|---:|---:|"]
for x in comparison:
    if not x["selected_week1_profit"]:continue
    p=x["week1"];q=x["week2"]
    roi="適用なし" if q["return_pct"] is None else str(round(q["return_pct"],2))+"%"
    report.append(f"|{x['template']}|{p['profit']}|{q['races']}|{q['hits']}|{q['stake']}|{q['paid']}|{q['profit']}|{roi}|")
report+=["","## 評価","赤字も全件表示。高配当事例の次週での再現性を検証する。第2週を見て選んだ黒字型を新たな市場優位性と扱わない。",
"全302型の比較はweek1_vs_week2.json。第2週に作れなかった型は適用なし。",
"高配当例ファイルは観察用。比較標本が少ない例、状態が一致しない例は価格誤差の証拠とはしない。"]
(out/"REPORT.md").write_text("\n".join(report))
print("VALIDATION",json.dumps(subsets,ensure_ascii=False),flush=True)
