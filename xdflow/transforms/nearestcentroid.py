import numpy as np
from numpy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from xdflow.transforms.sklearn_transform import SKLearnTransformer


class NearestCentroidTransform(BaseEstimator, TransformerMixin):
    """
    Fisher-style discriminant transform under spherical within-class covariance (Σ = I).
    This yields the same subspace as LDA's .transform when shrinkage α=1 (nearest-centroid case).

    Parameters
    ----------
    n_components : int or None (<= C-1)
        Number of components to keep. If None, uses C-1.

    use_priors : bool
        If True, weight the overall mean by class priors (n_k / n). If False, unweighted.

    Attributes (after fit)
    ----------------------
    classes_ : (C,)
    means_   : (C, p)
    priors_  : (C,)
    scalings_ : (p, r)   # projection matrix
    explained_variance_ratio_ : (r,)
    mean_ : (p,)         # overall mean used to center before projecting
    """

    def __init__(self, n_components=None, use_priors=True):
        self.n_components = n_components
        self.use_priors = use_priors

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64, ensure_2d=True)
        n, p = X.shape
        classes = unique_labels(y)
        C = classes.size
        if C < 2:
            raise ValueError("Need at least two classes.")

        # class means and counts
        means = []
        counts = []
        for c in classes:
            Xi = X[y == c]
            counts.append(len(Xi))
            means.append(Xi.mean(axis=0))
        counts = np.asarray(counts, dtype=float)
        priors = counts / counts.sum()
        means = np.vstack(means)  # (C, p)

        # overall mean
        if self.use_priors:
            mu_bar = np.average(means, axis=0, weights=priors)
        else:
            mu_bar = means.mean(axis=0)

        # B in shape (p, C): columns are sqrt(n_k/n)*(mu_k - mu_bar)
        B = (means - mu_bar) * np.sqrt((counts / n)[:, None])  # (C, p)
        B = B.T  # (p, C)

        # Small CxC eigendecomposition
        K = B.T @ B  # (C, C)
        evals, V = eigh(K)  # ascending
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        V = V[:, order]

        r_max = max(0, C - 1)
        r = r_max if self.n_components is None else min(int(self.n_components), r_max)
        if r < 0:
            raise ValueError("n_components must be >= 0 or None.")

        if r > 0:
            W = B @ V[:, :r]  # (p, r)
            # column-normalize for stability
            norms = np.linalg.norm(W, axis=0)
            norms[norms == 0] = 1.0
            W = W / norms
            pos = np.clip(evals[:r], 0.0, None)
            expl = pos / (pos.sum() + 1e-12)
        else:
            W = np.zeros((p, 0), dtype=X.dtype)
            expl = np.zeros((0,), dtype=X.dtype)

        # store
        self.classes_ = classes
        self.means_ = means
        self.priors_ = priors
        self.mean_ = mu_bar
        self.scalings_ = W
        self.explained_variance_ratio_ = expl
        return self

    def transform(self, X):
        check_is_fitted(self, attributes=["scalings_", "mean_"])
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        if self.scalings_.shape[1] == 0:
            return np.zeros((X.shape[0], 0), dtype=X.dtype)
        return (X - self.mean_) @ self.scalings_

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            raise ValueError("NearestCentroidTransform.fit_transform requires y.")
        return self.fit(X, y).transform(X)


class NearestCentroid(SKLearnTransformer):
    """
    Transform wrapper for NearestCentroidTransform that can be used in pipelines.

    Fisher-style discriminant transform under spherical within-class covariance (Σ = I).
    This yields the same subspace as LDA's .transform when shrinkage α=1 (nearest-centroid case).

    Parameters
    ----------
    sample_dim : str
        The name of the dimension that corresponds to samples.
    target_coord : str
        The name of the coordinate containing the target variable for supervised fitting.
    n_components : int or None
        Number of components to keep. If None, uses C-1.
    use_priors : bool
        If True, weight the overall mean by class priors (n_k / n). If False, unweighted.
    output_dim_name : str
        The name for the new dimension created by the transformer.
    sel : dict, optional
        Selection dictionary passed to parent.
    drop_sel : dict, optional
        Drop selection dictionary passed to parent.
    """

    def __init__(
        self,
        sample_dim: str,
        target_coord: str,
        n_components=None,
        use_priors=True,
        output_dim_name: str = "component",
        sel: dict | None = None,
        drop_sel: dict | None = None,
        sample_weight_coord: str | None = "sample_weight",
        **kwargs,
    ):
        super().__init__(
            estimator_cls=NearestCentroidTransform,
            sample_dim=sample_dim,
            target_coord=target_coord,
            output_dim_name=output_dim_name,
            sel=sel,
            drop_sel=drop_sel,
            n_components=n_components,
            use_priors=use_priors,
            sample_weight_coord=sample_weight_coord,
            **kwargs,
        )
