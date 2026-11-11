"""Build the anonymised code supplement for double-blind review (NeurIPS: one zip, under 100 MB).

  python tools/make_supplement.py            # writes supplement.zip and lists anything that still identifies the author

Copies tracked files except the notebook, the project log and git metadata, replaces identifying strings, and
refuses to finish if any remain. Add your own strings (name, college, handles) to IDENTIFYING before use.
"""
import pathlib, re, subprocess, sys, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
IDENTIFYING = [r"yashwanthsagar2004[-\w]*", r"Yashwanth", r"\bYash\b", r"\bGone\b", r"MVSR", r"Osmania", r"Hyderabad"]
SKIP = ("notebooks/", "PROJECT_LOG.md", ".github/", "tools/make_supplement.py")
pattern = re.compile("|".join(IDENTIFYING), re.IGNORECASE)
files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
left, size = [], 0
with zipfile.ZipFile(ROOT / "supplement.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if f.startswith(SKIP):
            continue
        text = (ROOT / f).read_text(errors="replace")
        clean = pattern.sub("anonymous", text).replace("github.com/anonymous/accountable-teams", "anonymous.4open.science/r/accountable-teams")
        if pattern.search(clean):
            left.append(f)
        z.writestr(f"accountable-teams/{f}", clean)
        size += len(clean)
print(f"supplement.zip: {len(files)} files considered, {size / 1e6:.2f} MB before compression")
if left:
    sys.exit(f"identifying strings remain in: {left}")
