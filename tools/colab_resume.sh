#!/usr/bin/env bash
# Resume the version-2 GPU runs on Colab after a reset. Mount Drive first (from google.colab import drive;
# drive.mount('/content/drive')), then:  !bash /content/accountable-teams/tools/colab_resume.sh
# Every run writes to Drive and resumes from <out>.partial and <out>.cache/, so rerunning this is always safe.
# v3 outputs (19 Nov 2026): the critic loss changed (review C1) and partial results now carry a fingerprint, so v2 partials are not resumed.
set -u
D=/content/drive/MyDrive/accteams_runs
mkdir -p "$D"
cd /content
[ -d accountable-teams ] || git clone -q -b set-critic https://github.com/yashwanthsagar2004-dot/accountable-teams
cd accountable-teams && git pull -q origin set-critic && git log --oneline -1
pip -q install -e . jumanji > /tmp/pip.log 2>&1
# Installing jumanji can upgrade jaxlib past Colab's CUDA plugin; JAX then fails to start its GPU backend
# ("Unexpected option name ... memory_fraction_policy", 20 Nov 2026). Install the plugin that matches jaxlib.
JV=$(python -c "import jaxlib; print(jaxlib.__version__)")
pip -q install -U "jax[cuda13]==$JV" >> /tmp/pip.log 2>&1
python -c "import jax; d = jax.devices()[0]; assert d.platform == 'gpu', d; print('JAX', jax.__version__, 'on', d)" || { echo "NO GPU: stopping"; exit 1; }
# credit comparison (fixed credit, 10 seeds x 10k updates) alongside the LBF critic run; RWARE after LBF
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 nohup python experiments/credit_compare.py --env lbf --agents 4 --updates 10000 \
  --seeds 10 --out "$D/credit_compare_lbf_v3.json" >> "$D/cc_lbf_v3.log" 2>&1 &
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 nohup python experiments/critic_budget.py --env lbf --agents 4 --window 20 --horizon 20 \
  --seeds 3 --out "$D/critic_budget_lbf_w20_v3.json" >> "$D/cb_lbf.log" 2>&1 &
P=$!
nohup bash -c "while kill -0 $P 2>/dev/null; do sleep 30; done; XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 python \
  experiments/critic_budget.py --env rware --agents 4 --window 10 --seeds 3 --out $D/critic_budget_rware_w10_v3.json" \
  >> "$D/cb_rware.log" 2>&1 &
sleep 20; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader; pgrep -af experiments | cut -c1-90
