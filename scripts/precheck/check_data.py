import sys, json, re, unicodedata, collections
import pandas as pd
R = sys.argv[1]
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().lower()
def lines(p): return open(p, encoding="utf-8").read().rstrip("\n").split("\n")
co = {}
for sp in ["train", "val", "test"]:
    b = f"{R}/cochrane_repo/data/data-1024/{sp}"
    s, t, d = lines(b + ".source"), lines(b + ".target"), lines(b + ".doi")
    co[sp] = (s, t, d)
    print("cochrane", sp, len(s), len(t), len(d), "empty src", sum(not x.strip() for x in s), "empty tgt", sum(not x.strip() for x in t),
          "dup src", len(s) - len(set(map(norm, s))), "dup tgt", len(t) - len(set(map(norm, t))), "dup doi", len(d) - len(set(d)))
cs = {sp: set(map(norm, co[sp][0])) for sp in co}
cd = {sp: set(co[sp][2]) for sp in co}
print("src overlap train-val", len(cs["train"] & cs["val"]), "train-test", len(cs["train"] & cs["test"]), "val-test", len(cs["val"] & cs["test"]))
print("doi overlap train-val", len(cd["train"] & cd["val"]), "train-test", len(cd["train"] & cd["test"]), "val-test", len(cd["val"] & cd["test"]))
meta = {}
for sp in ["train", "val", "test"]:
    m = pd.read_csv(f"{R}/cells_x/{sp}_meta.csv")
    src = lines(f"{R}/cells_x/CELLS/{sp}.source")
    meta[sp] = m
    print("cells", sp, "meta rows", len(m), "source lines", len(src), "journals", dict(m.journal.value_counts()))
for sp, m in meta.items():
    src = lines(f"{R}/cells_x/CELLS/{sp}.source")
    if len(src) == len(m):
        same = sum(norm(a) == norm(b) for a, b in zip(src, m.abs_text.astype(str)))
        print("cells", sp, "meta abs_text == source line", same, "/", len(m))
allm = pd.concat([m.assign(split=sp) for sp, m in meta.items()])
nihr = allm[allm.journal.str.lower().str.startswith("nhir_", na=False)]
print("NIHR total", len(nihr), dict(nihr.split.value_counts()))
print("NIHR empty abs", (nihr.abs_text.astype(str).str.strip() == "").sum(), "empty pls", (nihr.pls_text.astype(str).str.strip() == "").sum(), "dup abs", len(nihr) - nihr.abs_text.map(lambda x: norm(str(x))).nunique())
nset = set(nihr.abs_text.map(lambda x: norm(str(x))))
for sp in cs: print("NIHR overlap with cochrane", sp, len(nset & cs[sp]))
ctr = meta["train"]; ctrset = set(ctr.abs_text.map(lambda x: norm(str(x))))
for sp in ["val", "test"]: print("cochrane", sp, "sources found in CELLS train (exact)", len(cs[sp] & ctrset))
coc_in_cells = allm[allm.journal.str.lower().str.contains("cochrane", na=False)]
cset = set(coc_in_cells.abs_text.map(lambda x: norm(str(x))))
for sp in cs: print("cochrane", sp, "exact match to any CELLS cochrane abstract", len(cs[sp] & cset))
print("cells train internal dup pairs", len(ctr) - ctr.apply(lambda r: norm(str(r.abs_text)) + "||" + norm(str(r.pls_text)), axis=1).nunique())
import textstat
def wc(x): return len(x.split())
def dig(x): return 100 * len(re.findall(r"\d", x)) / max(1, wc(x))
rows = []
for name, s, t in [("cochrane_train", *co["train"][:2]), ("cochrane_val", *co["val"][:2]), ("cochrane_test", *co["test"][:2]), ("nihr", list(nihr.abs_text.astype(str)), list(nihr.pls_text.astype(str)))]:
    for side, xs in [("source", s), ("target", t)]:
        w = pd.Series([wc(x) for x in xs])
        rows.append(dict(set=name, side=side, n=len(xs), mean_words=round(w.mean()), median_words=int(w.median()), min=int(w.min()), max=int(w.max()),
                         fkgl=round(pd.Series([textstat.flesch_kincaid_grade(x) for x in xs]).mean(), 1), digits_per_100=round(pd.Series([dig(x) for x in xs]).mean(), 1),
                         has_CI=sum(bool(re.search(r"\bCI\b", x)) for x in xs)))
print(pd.DataFrame(rows).to_string(index=False))
print("targets shorter than source (cochrane train)", round(100 * sum(wc(t) < wc(s) for s, t in zip(*co["train"][:2])) / len(co["train"][0]), 1))
