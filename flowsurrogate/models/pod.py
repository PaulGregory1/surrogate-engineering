"""Reduced-order model baseline: POD (a.k.a. PCA) + regression on the design vector.

This is what an engineering team would build before reaching for deep learning, and
on a 7-parameter shape family it is a genuinely strong baseline -- reporting a
neural surrogate without it would be misleading.

Its structural limitation is the point of the comparison: the regressor takes the
*Fourier coefficients* as input, so it is welded to one parametrisation.  Change the
CAD template and the model is worthless, whereas the image/graph-based surrogates
only need the new geometry rasterised.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class PODRegressor:
    """POD of the flow fields + (kernel) ridge regression of the POD coefficients."""

    def __init__(self, n_modes: int = 64, kernel: str = "rbf", alpha: float = 1e-3, gamma: float | None = None):
        self.n_modes = n_modes
        self.kernel = kernel
        self.alpha = alpha
        self.gamma = gamma
        self.pca: PCA | None = None
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def design_vector(params: np.ndarray, re: np.ndarray) -> np.ndarray:
        return np.concatenate([params, re[:, None]], axis=1)

    def _make_regressor(self):
        if self.kernel == "linear":
            return Ridge(alpha=self.alpha)
        return KernelRidge(kernel="rbf", alpha=self.alpha, gamma=self.gamma)

    # --------------------------------------------------------------------- api
    def fit(self, params: np.ndarray, re: np.ndarray, fields: np.ndarray, cd: np.ndarray) -> "PODRegressor":
        n = fields.shape[0]
        flat = fields.reshape(n, -1)
        self.pca = PCA(n_components=min(self.n_modes, n - 1), svd_solver="randomized", random_state=0)
        coeffs = self.pca.fit_transform(flat)
        self.coeff_scale = coeffs.std(axis=0) + 1e-8

        x = self.scaler.fit_transform(self.design_vector(params, re))
        self.reg_fields = self._make_regressor().fit(x, coeffs / self.coeff_scale)
        self.reg_cd = self._make_regressor().fit(x, cd)
        return self

    def predict(self, params: np.ndarray, re: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        x = self.scaler.transform(self.design_vector(params, re))
        coeffs = self.reg_fields.predict(x) * self.coeff_scale
        fields = self.pca.inverse_transform(coeffs).reshape(-1, *shape)
        return fields, self.reg_cd.predict(x)

    @property
    def n_parameters(self) -> int:
        """Stored floats -- the fair analogue of a network's parameter count."""
        assert self.pca is not None
        return int(self.pca.components_.size + self.pca.mean_.size)
