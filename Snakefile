import os
import sys
import tempfile
from pathlib import Path

from ploomber_engine import execute_notebook


# Make `src/` and the vendored EMU_data_collection submodule importable to
# every rule — both Snakemake `script:` directives (which run in this process,
# so sys.path is what matters) and the Jupyter kernels spawned by
# ploomber_engine (which inherit PYTHONPATH).
_EXTRA_PATHS = [
    "src",
    "vendor/EMU_data_collection/src/python",
]
for p in _EXTRA_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    _EXTRA_PATHS + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
)


def run_notebook(input_path, output_path, parameters):
    """Execute a notebook (either .ipynb or jupytext .py-percent) with ploomber_engine.

    Parameters are injected into the cell tagged `parameters`.
    """
    import jupytext

    input_path = Path(input_path)
    if input_path.suffix == ".py":
        nb = jupytext.read(input_path)
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as f:
            tmp_path = Path(f.name)
        jupytext.write(nb, tmp_path)
    else:
        tmp_path = None

    try:
        actual_input = tmp_path if tmp_path is not None else input_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        return execute_notebook(str(actual_input), str(output_path), parameters=parameters)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


configfile: "config.yaml"

# Local overrides (e.g. data roots on this machine). Gitignored.
if Path("config.local.yaml").exists():

    configfile: "config.local.yaml"


SUBJECTS = list(config["subjects"].keys())
BANDS = config["frequency_bands"]


def subj(wc):
    return config["subjects"][wc.subject]


def raw_session_dir(wc):
    s = subj(wc)
    return f"{config['neuralynx_raw_root']}/{wc.subject}/{wc.subject}_{s['block']}"


def ecog_hilbert_h5(wc):
    s = subj(wc)
    root = config["neuralynx_preproc_root"]
    return (
        f"{root}/{wc.subject}/{wc.subject}_{s['block']}/ECoG_preproc/"
        f"{wc.subject}_{s['block']}_100hz_nocar_Hilb.h5"
    )


rule all:
    input:
        expand(
            "results/{subject}/epochs/{band}-epo.fif",
            subject=SUBJECTS,
            band=BANDS,
        ),


rule extract_photodiode:
    output:
        parquet="results/{subject}/photodiode.parquet",
        meta="results/{subject}/photodiode_meta.json",
    params:
        raw_dir=raw_session_dir,
        session_subdir=lambda wc: subj(wc)["session_subdir"],
        channel=lambda wc: subj(wc)["photodiode_channel"],
        ncs_suffix=lambda wc: subj(wc)["ncs_suffix"],
    script:
        "scripts/extract_photodiode.py"


rule detect_photodiode_edges:
    input:
        parquet=rules.extract_photodiode.output.parquet,
        meta=rules.extract_photodiode.output.meta,
    output:
        edges="results/{subject}/photodiode_edges.parquet",
    params:
        expt_start_time=lambda wc: subj(wc)["expt_start_time"],
        expt_end_time=lambda wc: subj(wc)["expt_end_time"],
        detector=lambda wc: subj(wc)["photodiode_detector"],
    script:
        "scripts/detect_photodiode_edges.py"


rule align_behavior:
    input:
        behavior="data/{subject}/behavior/data.csv",
        edges=rules.detect_photodiode_edges.output.edges,
        notebook="notebooks/align_behavior.py",
    output:
        trials="results/{subject}/trials.parquet",
        notebook="results/{subject}/notebooks/align_behavior.ipynb",
    run:
        run_notebook(
            input.notebook,
            output.notebook,
            parameters=dict(
                behavior_path=input.behavior,
                edges_path=input.edges,
                trials_out=output.trials,
            ),
        )


rule epoch_ecog:
    input:
        ecog_h5=ecog_hilbert_h5,
        trials=rules.align_behavior.output.trials,
    output:
        epochs=expand("results/{{subject}}/epochs/{band}-epo.fif", band=BANDS),
    params:
        bands=BANDS,
        final_hz=config["final_hz"],
        buffer_before=config["buffer_before"],
        buffer_after=config["buffer_after"],
    script:
        "scripts/epoch_ecog.py"
