"""Benchmark an xdflow pipeline with MOABB, next to the same model in plain scikit-learn.

MOABB (Mother of All BCI Benchmarks) evaluates scikit-learn estimators on public EEG datasets with standard
protocols. `xdflow.to_sklearn` turns an xdflow pipeline into such an estimator, so xdflow pipelines can be
compared with published MOABB baselines under exactly the same evaluation.

Here: left vs right hand motor imagery on the PhysioNet dataset, within-session 5-fold evaluation, CSP + LDA
written once in scikit-learn and once as an xdflow pipeline (which crops the window by time value).

Run from the repository root (MOABB downloads the recordings on the first run):

    uv pip install moabb
    uv run --extra mne python examples/moabb_evaluation.py
"""

import mne
from mne.decoding import CSP
from moabb.datasets import PhysionetMI
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

import xdflow
from xdflow.transforms import CropTimeTransform, CSPTransform

SUBJECTS = [1, 2, 3, 4, 5]
FMIN, FMAX, TMIN, TMAX = 8.0, 30.0, 0.0, 3.0  # Hz band and seconds after the cue, applied by the paradigm


def main() -> None:
    mne.set_log_level("ERROR")
    dataset = PhysionetMI(imagined=True, executed=False)
    dataset.subject_list = SUBJECTS
    paradigm = LeftRightImagery(fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)

    # Channel names and time points, so xdflow steps can select by label.
    epochs, _, _ = paradigm.get_data(dataset, subjects=SUBJECTS[:1], return_epochs=True)
    coords = {"channel": epochs.ch_names, "time": epochs.times}

    xdflow_pipeline = xdflow.make_pipeline(
        CropTimeTransform(0.5, TMAX),  # by time value: the first 0.5 s after the cue are dropped
        CSPTransform(n_components=6),
        xdflow.SKLearnPredictor(LinearDiscriminantAnalysis, sample_dim="trial", target_coord="label"),
    )
    sklearn_pipeline = make_pipeline(CSP(n_components=6, log=True), LinearDiscriminantAnalysis())

    evaluation = WithinSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        n_jobs=1,
        random_state=0,  # same folds for both
    )
    results = evaluation.process(
        {
            "CSP+LDA (scikit-learn)": sklearn_pipeline,
            "CSP+LDA, 0.5-3 s (xdflow)": xdflow.to_sklearn(xdflow_pipeline, ("trial", "channel", "time"), coords),
        }
    )
    print(results.pivot_table(index="subject", columns="pipeline", values="score").round(3).to_string())
    print()
    print(results.groupby("pipeline")["score"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
