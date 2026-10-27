import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
main = (ROOT / "main.tex").read_text(encoding="utf-8")

# 1. inputs exist
missing = []
for target in re.findall(r"\\input\{([^}]+)\}", main):
    path = ROOT / (target if target.endswith(".tex") else target + ".tex")
    if not path.exists():
        missing.append(str(path))
print("inputs:", "OK" if not missing else f"MISSING {missing}")

# 2. cites resolve
bib = (ROOT / "references.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for group in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", main):
    for key in group.split(","):
        cited.add(key.strip())
unknown = sorted(cited - keys)
unused = sorted(keys - cited)
print("cite keys:", len(cited), "| unknown:", unknown or "none", "| unused:", unused or "none")

# 3. balanced environments and braces in every tex file
tex_files = [ROOT / "main.tex"] + sorted((ROOT / "tables").glob("*.tex")) + sorted((ROOT / "figures").glob("*.tex"))
problems = []
for path in tex_files:
    text = path.read_text(encoding="utf-8")
    begins = re.findall(r"\\begin\{([^}]+)\}", text)
    ends = re.findall(r"\\end\{([^}]+)\}", text)
    if sorted(begins) != sorted(ends):
        problems.append(f"{path.name}: env mismatch {begins} vs {ends}")
    # strip escaped braces and comments before counting
    body = re.sub(r"\\[{}%$&_#]", "", text)
    body = re.sub(r"(?m)%.*$", "", body)
    if body.count("{") != body.count("}"):
        problems.append(f"{path.name}: brace imbalance {body.count('{')} vs {body.count('}')}")
    if "$" in body:
        n = body.count("$")
        if n % 2:
            problems.append(f"{path.name}: odd number of $ ({n})")
print("structure:", "OK" if not problems else "\n  " + "\n  ".join(problems))

# 4. non-ascii report (LaTeX source should be ascii unless declared)
for path in tex_files:
    bad = [(i + 1, line) for i, line in enumerate(path.read_text(encoding="utf-8").splitlines())
           if any(ord(c) > 127 for c in line)]
    if bad:
        print(f"non-ascii in {path.name}: line {bad[0][0]}: {bad[0][1][:80]}")

sys.exit(1 if (missing or unknown or problems) else 0)
