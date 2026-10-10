# Notebooks

Colab-ready versions of the examples. Each notebook installs xdflow in its first code cell. The committed outputs come
from a local run, so you can read the results without running anything.

| Notebook | What it shows | |
|---|---|---|
| [evaluate.ipynb](evaluate.ipynb) | Your data and scikit-learn model, cross-validated by subject in one call; auditing a shuffled split; preprocessing as a `@step` function; back into scikit-learn. Downloads 61 MB on the first run. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/general-sense/xdflow/blob/main/examples/notebooks/evaluate.ipynb) |
| [quickstart.ipynb](quickstart.ipynb) | Coordinate targets, stratified CV, stateful refits and aligned predictions on synthetic data. Base install only; runs in seconds. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/general-sense/xdflow/blob/main/examples/notebooks/quickstart.ipynb) |
| [eeg_motor_imagery.ipynb](eeg_motor_imagery.ipynb) | Within- vs. cross-subject decoding of public EEG with CSP+LDA, side by side with plain scikit-learn, and the inflation from fitting CSP before CV. Downloads 38.6 MB from PhysioNet on the first run. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/general-sense/xdflow/blob/main/examples/notebooks/eeg_motor_imagery.ipynb) |
| [har_activity.ipynb](har_activity.ipynb) | Activity recognition from phone sensors (30 people): subject-grouped vs. shuffled-window validation, per-subject accuracy, and the same analysis in plain scikit-learn. Downloads 61 MB on the first run. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/general-sense/xdflow/blob/main/examples/notebooks/har_activity.ipynb) |

The same code is available as scripts (`examples/quickstart.py`, `examples/eeg_motor_imagery.py`) and as written
[tutorials](https://xdflow.readthedocs.io/en/latest/tutorials/).

To re-execute a notebook from a repository checkout and save its outputs (the `%pip` cell is tagged `skip-execution`,
so the local checkout is used instead of the GitHub install):

```bash
uv run --all-extras --with mne --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.record_timing=False examples/notebooks/eeg_motor_imagery.ipynb
```
