"""Print the weighted completion of the project from status/progress.json."""
import json, pathlib
p = json.loads((pathlib.Path(__file__).resolve().parents[1] / "status/progress.json").read_text())
tot = sum(w["weight"] for w in p["packages"])
done = sum(w["weight"] * w["fraction"] for w in p["packages"])
for w in p["packages"]:
    print(f'{w["id"]:5s} {w["weight"]:3d}  {100*w["fraction"]:5.0f}%  {w["name"]}')
print(f"complete: {100*done/tot:.1f}% of {tot} weight (as of {p['as_of']})")
