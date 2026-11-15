import sys, re, textstat, statistics as st
import pandas as pd
R = sys.argv[1]
def lines(p): return open(p, encoding="utf-8").read().rstrip("\n").split("\n")
s = lines(f"{R}/cochrane_repo/data/data-1024/train.source"); t = lines(f"{R}/cochrane_repo/data/data-1024/train.target")
def mask(x): return re.sub(r"\d+(?:[.,]\d+)*", "NUM", x)
for name, f in [("raw", lambda x: x), ("numbers masked", mask)]:
    print(name, "source", round(st.mean(textstat.flesch_kincaid_grade(f(x)) for x in s), 1), "target", round(st.mean(textstat.flesch_kincaid_grade(f(x)) for x in t), 1),
          "| sentences/abstract src", round(st.mean(textstat.sentence_count(f(x)) for x in s), 1), "tgt", round(st.mean(textstat.sentence_count(f(x)) for x in t), 1))
m = pd.concat([pd.read_csv(f"{R}/cells_x/{sp}_meta.csv").assign(split=sp) for sp in ["train","val","test"]])
n = m[m.journal.str.startswith("nhir_")].copy()
n["k"] = n.abs_text.str.split().str.join(" ").str.lower()
d = n[n.duplicated("k", keep=False)]
print("NIHR rows in duplicate groups", len(d), "groups", d.k.nunique(), "unique abstracts", n.k.nunique())
print("groups where the summaries also match", sum(g.pls_text.str.split().str.join(" ").nunique() == 1 for _, g in d.groupby("k")))
print("split mix inside groups", dict(d.groupby("k").split.apply(lambda x: "+".join(sorted(set(x)))).value_counts()))
print("example dup abstract start:", d.iloc[0].abs_text[:200])
