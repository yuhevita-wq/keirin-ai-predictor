"""Diagnose source line HTML after a real entry fetch; never infer lines."""
import json
import re
import requests
from bs4 import BeautifulSoup

ID="202610255301"
url=f"https://keirin.netkeiba.com/race/entry/?race_id={ID}"
r=requests.get(url,headers={"User-Agent":"Mozilla/5.0","Referer":"https://keirin.netkeiba.com/"},timeout=25)
r.encoding=r.apparent_encoding
soup=BeautifulSoup(r.text,"lxml")
selectors=[
 ".DeployYoso",".DeployInBox",".WakuSeparat",".Shaban_Num",
 "[class*='Line']","[class*='line']","[class*='Deploy']","[class*='deploy']",
 ".RaceCard_Simple_Table",".RaceCard_Simple_Table_Static"
]
summary={k:len(soup.select(k)) for k in selectors}
chunks={}
for term in ["DeployYoso","DeployInBox","並び予想","並び","ライン","SENYU","line_group","RaceCard_Simple_Table"]:
  idx=r.text.find(term)
  chunks[term]={"offset":idx,"snippet":re.sub(r"\s+"," ",r.text[max(0,idx-200):idx+480])[:700] if idx!=-1 else ""}
candidates=[]
for el in soup.find_all(class_=re.compile(r"(?i)line|deploy|waku|shaban"))[:50]:
  candidates.append({"tag":el.name,"class":el.get("class"),"text":el.get_text(" ",strip=True)[:100],"fragment":str(el)[:400]})
out={"id":ID,"url":r.url,"status":r.status_code,"bytes":len(r.content),
     "title":soup.title.get_text(" ",strip=True)[:160] if soup.title else None,
     "selectors":summary,"terms":chunks,"candidate_elements":candidates,
     "first_tables":[{"class":t.get("class"),"head":t.get_text(" ",strip=True)[:180]} for t in soup.select("table")[:4]]}
print("LINE_SOURCE_DIAG_BEGIN")
print(json.dumps(out,ensure_ascii=False,indent=2))
print("LINE_SOURCE_DIAG_END")
