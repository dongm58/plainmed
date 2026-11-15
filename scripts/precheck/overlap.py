import sys, re, unicodedata
import pandas as pd
R = sys.argv[1]
def norm(s): return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s))).lower()).strip()
def lines(p): return open(p, encoding="utf-8").read().rstrip("\n").split("\n")
def grams(t, n=8):
    w = t.split(); return {" ".join(w[i:i+n]) for i in range(len(w)-n+1)}
co = {sp: lines(f"{R}/cochrane_repo/data/data-1024/{sp}.source") for sp in ["train","val","test"]}
for csp in ["train","val","test"]:
    m = pd.read_csv(f"{R}/cells_x/{csp}_meta.csv")
    pool = m[m.journal == "cochrane_all"]
    pg = [grams(norm(a)) for a in pool.abs_text]
    index = {}
    for i, g in enumerate(pg):
        for x in g: index.setdefault(x, set()).add(i)
    for sp, srcs in co.items():
        hits = 0
        for s in srcs:
            g = grams(norm(s))
            if not g: continue
            cnt = {}
            for x in g:
                for i in index.get(x, ()): cnt[i] = cnt.get(i, 0) + 1
            if cnt and max(cnt.values()) / len(g) >= 0.5: hits += 1
        print(f"cochrane {sp} ({len(srcs)}) with >=50% 8-gram overlap to a CELLS {csp} cochrane abstract: {hits}")
