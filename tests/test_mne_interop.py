import pickle

import numpy as np
import pandas as pd
import pytest
import xarray as xr

mne = pytest.importorskip("mne")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from xdflow import DataContainer, KFoldValidator, Pipeline, from_mne_epochs  # noqa: E402
from xdflow.transforms.csp import CSPTransform  # noqa: E402
from xdflow.transforms.sklearn_transform import SKLearnPredictor  # noqa: E402

N_TRIALS, N_CHANNELS, N_TIMES, SFREQ = 40, 6, 100, 100.0


def _epochs(seed: int = 0):
    """Two classes; class 'right' has extra variance on the first two channels (CSP-separable)."""
    rng = np.random.default_rng(seed)
    codes = np.array([1, 2] * (N_TRIALS // 2))
    values = rng.normal(size=(N_TRIALS, N_CHANNELS, N_TIMES))
    values[codes == 2, :2] *= 3.0
    info = mne.create_info([f"EEG{i}" for i in range(N_CHANNELS)], SFREQ, ch_types="eeg")
    events = np.column_stack([np.arange(N_TRIALS) * N_TIMES, np.zeros(N_TRIALS, int), codes])
    metadata = pd.DataFrame({"block": np.repeat([1, 2], N_TRIALS // 2)})
    return mne.EpochsArray(
        values, info, events=events, event_id={"left": 1, "right": 2}, metadata=metadata, verbose=False
    )


def test_from_mne_epochs_keeps_labels_metadata_and_extra_coords():
    epochs = _epochs()
    data = from_mne_epochs(epochs, subject=7).data

    assert data.dims == ("trial", "channel", "time")
    np.testing.assert_array_equal(data.values, epochs.get_data())
    assert list(data["channel"].values) == epochs.ch_names
    np.testing.assert_array_equal(data["label"].values[:4], ["left", "right", "left", "right"])
    np.testing.assert_array_equal(data["block"].values, epochs.metadata["block"].values)
    assert set(np.unique(data["subject"].values)) == {7}
    assert data.attrs["sfreq"] == SFREQ
    with pytest.raises(ValueError, match="clashes"):
        from_mne_epochs(epochs, block=1)


def test_csp_transform_is_refit_per_fold_and_follows_channel_labels():
    data = from_mne_epochs(_epochs())
    pipeline = Pipeline(
        "csp_lda",
        [
            ("csp", CSPTransform(n_components=2)),
            ("lda", SKLearnPredictor(LinearDiscriminantAnalysis, sample_dim="trial", target_coord="label")),
        ],
    )
    cv = KFoldValidator(n_splits=4, scoring="accuracy", use_stateful_fit_cache=False, verbose=False)
    cv.set_pipeline(pipeline)
    assert cv.cross_validate(data) > 0.8

    csp = CSPTransform(n_components=2).fit(data)
    reordered = DataContainer(data.data.isel(channel=list(reversed(range(N_CHANNELS)))))
    np.testing.assert_allclose(csp.transform(reordered).data.values, csp.transform(data).data.values)


def test_existing_sklearn_pipeline_runs_on_nd_data_inside_xdflow_validation():
    data = from_mne_epochs(_epochs())
    existing = make_pipeline(mne.decoding.CSP(n_components=2, log=True), LinearDiscriminantAnalysis())
    predictor = SKLearnPredictor.from_estimator(existing, sample_dim="trial", target_coord="label")

    cv = KFoldValidator(n_splits=4, scoring="accuracy", use_stateful_fit_cache=False, verbose=False)
    cv.set_pipeline(Pipeline("wrapped", [("model", predictor)]))
    assert cv.cross_validate(data) > 0.8

    predictor.set_params(csp__n_components=3)
    clone = pickle.loads(pickle.dumps(predictor.clone()))
    assert clone.estimator.named_steps["csp"].n_components == 3
    assert existing.named_steps["csp"].n_components == 2  # the template is never mutated
    assert not hasattr(existing.named_steps["lineardiscriminantanalysis"], "coef_")  # nor fitted
    xr.testing.assert_equal(data.data, from_mne_epochs(_epochs()).data)
