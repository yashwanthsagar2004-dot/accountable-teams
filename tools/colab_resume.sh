#!/usr/bin/env bash
# Resume the version-2 GPU runs on Colab after a reset. Mount Drive first (from google.colab import drive;
# drive.mount('/content/drive')), then:  !bash /content/accountable-teams/tools/colab_resume.sh
# Every run writes to Drive and resumes from <out>.partial and <out>.cache/, so rerunning this is always safe.
set -u
D=/content/drive/MyDrive/accteams_runs
mkdir -p "$D"
cd /content
[ -d accountable-teams ] || git clone -q -b set-critic https://github.com/yashwanthsagar2004-dot/accountable-teams
cd accountable-teams && git pull -q origin set-critic && git log --oneline -1
pip -q install -e . jumanji > /tmp/pip.log 2>&1
# credit comparison (fixed credit, 10 seeds x 10k updates) alongside the LBF critic run; RWARE after LBF
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 nohup python experiments/credit_compare.py --env lbf --agents 4 --updates 10000 \
  --seeds 10 --out "$D/credit_compare_lbf_v2.json" >> "$D/cc_lbf_v2.log" 2>&1 &
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 nohup python experiments/critic_budget.py --env lbf --agents 4 --window 20 \
  --seeds 3 --out "$D/critic_budget_lbf_w20.json" >> "$D/cb_lbf.log" 2>&1 &
P=$!
nohup bash -c "while kill -0 $P 2>/dev/null; do sleep 30; done; XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 python \
  experiments/critic_budget.py --env rware --agents 4 --window 10 --seeds 3 --out $D/critic_budget_rware_w10.json" \
  >> "$D/cb_rware.log" 2>&1 &
sleep 20; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader; pgrep -af experiments | cut -c1-90
