"""Page 8 - Rail Naming: status of automatic naming, plus a manual re-run.

Rails are named automatically in the background on the first visit after a
deployment (see utils/auto_naming.py). This page shows that run's status and
lets an admin regenerate names on demand and download the result files.
"""
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import llm as _llm  # noqa: E402
from utils import auto_naming as an  # noqa: E402

st.title("Rail Naming")
st.markdown(
    "Every rail and sub-rail is named by the configured language model from its twelve "
    "highest-confidence films, genre mix and dominant decade. The prompt forbids naming a rail "
    "after a single film or franchise. Naming runs **automatically in the background** the first "
    "time the app starts after a deployment; this page shows its status and lets you re-run it."
)

status = an.read_status()
backend = _llm.llm_backend()
c1, c2, c3 = st.columns(3)
c1.metric("LLM backend", backend if backend != 'huggingface' else 'Hugging Face')
c2.metric("Last run", status.get('state', 'never'))
c3.metric("Progress", f"{status.get('done', 0)}/{status.get('total', 0)}" if status.get('total') else '-')
if status.get('state') == 'running':
    st.info("A naming run is in progress. Refresh this page to update the progress counter.")
if status.get('reason'):
    st.caption(f"Reason: {status['reason']}")
if status.get('failures'):
    st.warning(f"{status['failures']} names failed in the last run. Last error: {status.get('last_error', '')}")

st.markdown("---")
st.subheader("Re-run naming")


def _admin_key() -> str:
    key = os.getenv('ADMIN_KEY', '')
    if not key:
        try:
            key = str(st.secrets.get('ADMIN_KEY', '') or '')
        except Exception:
            key = ''
    return key


_expected = _admin_key()
if _expected:
    _given = st.text_input("Admin key", type="password",
                           help="Set ADMIN_KEY in secrets to protect manual re-runs.")
    if _given != _expected:
        st.caption("Enter the admin key to re-run naming manually.")
        st.stop()
else:
    st.warning("No ADMIN_KEY secret is set, so anyone who can open this page can trigger a re-run.")

if backend == 'none':
    st.error("No LLM backend available: start Ollama locally or set HF_TOKEN in secrets.")
    st.stop()

scope = st.radio("Scope", ["Rails and sub-rails", "Rails only", "Sub-rails only"], horizontal=True)
scope_key = {'Rails and sub-rails': 'all', 'Rails only': 'parent', 'Sub-rails only': 'sub'}[scope]

if st.button("Regenerate names now", type="primary"):
    bar = st.progress(0.0)
    line = st.empty()

    def _progress(done, total, job, name):
        label = f"Rail {job['cid']}" if job['kind'] == 'parent' else f"Rail {job['cid']} / sub {job['sid']}"
        line.markdown(f"**{label}** ({done}/{total}): {job['old']!r} → {name or 'FAILED'!r}")
        bar.progress(done / total)

    results = an.run_batch(scope_key, progress=_progress)
    st.session_state['batch_naming_results'] = results
    failures = sum(1 for r in results if not r['name'])
    line.empty()
    if failures:
        st.warning(f"{failures} of {len(results)} names failed. Last error: {_llm.LAST_LLM_ERROR}")
    else:
        st.success(f"Renamed {len(results)} rails. Other pages now show the new names.")

results = st.session_state.get('batch_naming_results')
if results:
    st.subheader("Old vs new")
    rows = [{'Rail': (f"{r['cid']}" if r['kind'] == 'parent' else f"{r['cid']} / {r['sid']}"),
             'Type': r['kind'], 'Films': int(len(r['idx'])), 'Old name': r['old'],
             'New name': r['name'] or f"FAILED: {r.get('error', '')}",
             'New description': r['description']} for r in results]
    st.dataframe(rows, use_container_width=True, hide_index=True)

if (an.ART_DIR / 'cluster_names.json').exists():
    st.markdown("---")
    st.subheader("Persist the current names")
    st.caption(
        "Hosted deployments have an ephemeral filesystem, so generated names last until the next "
        "redeploy. Commit these two files to `Final Documents/Dashboard/results/artifacts/` to make "
        "them permanent (and skip the automatic run on future startups)."
    )
    d1, d2 = st.columns(2)
    d1.download_button("Download cluster_names.json",
                       data=(an.ART_DIR / 'cluster_names.json').read_text(),
                       file_name='cluster_names.json', mime='application/json')
    d2.download_button("Download sub_cluster_names.json",
                       data=(an.ART_DIR / 'sub_cluster_names.json').read_text(),
                       file_name='sub_cluster_names.json', mime='application/json')
