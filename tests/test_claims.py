import subprocess, sys, pathlib


def test_every_quoted_number_matches_its_record():
    root = pathlib.Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, "tools/verify_claims.py", "--no-write"], cwd=root, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout
