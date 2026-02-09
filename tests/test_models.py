"""
Tests for model building: baseline models, classical model factories,
ensemble methods, and calibration.

Covers:
- Baseline models: HomeAlwaysModel, OddsImpliedModel predictions
- Classical model factories: return properly configured sklearn-compatible models
- All models implement fit/predict/predict_proba interface
- Ensemble: VotingEnsemble and StackingEnsemble with mock base models
- Calibration: CalibratedModel wrapping works
- Uses small synthetic datasets (20-50 rows)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Synthetic data fixtures
# ============================================================================


@pytest.fixture()
def small_feature_df() -> pd.DataFrame:
    """Small synthetic feature DataFrame for model testing."""
    np.random.seed(42)
    n = 30
    return pd.DataFrame({
        "home_team": np.random.choice(
            ["Melbourne Storm", "Penrith Panthers", "Sydney Roosters",
             "Canterbury Bulldogs", "Cronulla Sharks"],
            size=n,
        ),
        "away_team": np.random.choice(
            ["Brisbane Broncos", "Parramatta Eels", "Manly Sea Eagles",
             "Newcastle Knights", "Canberra Raiders"],
            size=n,
        ),
        "home_ladder_pos": np.random.randint(1, 17, n),
        "away_ladder_pos": np.random.randint(1, 17, n),
        "home_odds": np.round(np.random.uniform(1.3, 3.0, n), 2),
        "away_odds": np.round(np.random.uniform(1.3, 3.0, n), 2),
        "season": [2023] * 15 + [2024] * 15,
        "round": list(range(1, 16)) + list(range(1, 16)),
        "home_elo": np.random.normal(1500, 80, n),
        "away_elo": np.random.normal(1500, 80, n),
        "home_win_rate_5": np.random.uniform(0.2, 0.8, n),
        "away_win_rate_5": np.random.uniform(0.2, 0.8, n),
    })


@pytest.fixture()
def small_target(small_feature_df) -> np.ndarray:
    """Binary target aligned with small_feature_df."""
    np.random.seed(42)
    return np.random.randint(0, 2, size=len(small_feature_df))


@pytest.fixture()
def numeric_features(small_feature_df) -> pd.DataFrame:
    """Numeric-only features (no string columns) for sklearn models."""
    return small_feature_df[
        ["home_elo", "away_elo", "home_win_rate_5", "away_win_rate_5",
         "home_ladder_pos", "away_ladder_pos"]
    ].astype(float)


# ============================================================================
# HomeAlwaysModel tests
# ============================================================================


class TestHomeAlwaysModel:
    """Tests for modelling.baseline_models.HomeAlwaysModel."""

    def test_predict_always_returns_ones(self, small_feature_df):
        from modelling.baseline_models import HomeAlwaysModel

        model = HomeAlwaysModel()
        model.fit(small_feature_df)
        preds = model.predict(small_feature_df)
        assert (preds == 1).all()
        assert len(preds) == len(small_feature_df)

    def test_predict_proba_shape(self, small_feature_df):
        from modelling.baseline_models import HomeAlwaysModel

        model = HomeAlwaysModel()
        model.fit(small_feature_df)
        proba = model.predict_proba(small_feature_df)
        assert proba.shape == (len(small_feature_df), 2)

    def test_predict_proba_values(self, small_feature_df):
        from modelling.baseline_models import HomeAlwaysModel

        model = HomeAlwaysModel()
        model.fit(small_feature_df)
        proba = model.predict_proba(small_feature_df)
        # Home win probability should be 0.58
        assert abs(proba[0, 1] - 0.58) < 0.01
        assert abs(proba[0, 0] - 0.42) < 0.01

    def test_predict_proba_sums_to_one(self, small_feature_df):
        from modelling.baseline_models import HomeAlwaysModel

        model = HomeAlwaysModel()
        proba = model.predict_proba(small_feature_df)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_fit_returns_self(self, small_feature_df):
        from modelling.baseline_models import HomeAlwaysModel

        model = HomeAlwaysModel()
        result = model.fit(small_feature_df)
        assert result is model

    def test_get_params(self):
        from modelling.baseline_models import HomeAlwaysModel

        model = HomeAlwaysModel()
        assert model.get_params() == {}


# ============================================================================
# OddsImpliedModel tests
# ============================================================================


class TestOddsImpliedModel:
    """Tests for modelling.baseline_models.OddsImpliedModel."""

    def test_predict_proba_normalises_odds(self, small_feature_df):
        from modelling.baseline_models import OddsImpliedModel

        model = OddsImpliedModel()
        model.fit(small_feature_df)
        proba = model.predict_proba(small_feature_df)

        assert proba.shape == (len(small_feature_df), 2)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_predict_proba_known_values(self):
        from modelling.baseline_models import OddsImpliedModel

        df = pd.DataFrame({
            "home_odds": [2.0, 1.5],
            "away_odds": [2.0, 3.0],
        })
        model = OddsImpliedModel()
        model.fit(df)
        proba = model.predict_proba(df)

        # When both odds are 2.0: implied = 0.5/0.5, normalised = 0.5/0.5
        assert abs(proba[0, 1] - 0.5) < 0.01

        # home=1.5, away=3.0: raw = 0.667/0.333, normalised = 0.667/0.333
        assert proba[1, 1] > 0.6

    def test_predict_favourite(self):
        from modelling.baseline_models import OddsImpliedModel

        df = pd.DataFrame({
            "home_odds": [1.5],  # Home favourite
            "away_odds": [3.0],
        })
        model = OddsImpliedModel()
        model.fit(df)
        pred = model.predict(df)
        assert pred[0] == 1  # Home predicted to win

    def test_predict_underdog(self):
        from modelling.baseline_models import OddsImpliedModel

        df = pd.DataFrame({
            "home_odds": [3.0],  # Home underdog
            "away_odds": [1.5],
        })
        model = OddsImpliedModel()
        model.fit(df)
        pred = model.predict(df)
        assert pred[0] == 0  # Away predicted to win

    def test_missing_columns_raises(self):
        from modelling.baseline_models import OddsImpliedModel

        model = OddsImpliedModel()
        df = pd.DataFrame({"home_odds": [1.5]})  # Missing away_odds
        with pytest.raises(ValueError, match="missing required columns"):
            model.fit(df)

    def test_non_dataframe_raises(self):
        from modelling.baseline_models import OddsImpliedModel

        model = OddsImpliedModel()
        with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
            model.fit(np.array([[1.5, 2.0]]))


# ============================================================================
# LadderModel tests
# ============================================================================


class TestLadderModel:
    """Tests for modelling.baseline_models.LadderModel."""

    def test_predict_higher_ranked_team(self):
        from modelling.baseline_models import LadderModel

        df = pd.DataFrame({
            "home_ladder_pos": [1],   # Better position
            "away_ladder_pos": [10],
        })
        model = LadderModel()
        model.fit(df)
        pred = model.predict(df)
        assert pred[0] == 1  # Home should win (ranked higher)

    def test_predict_lower_ranked_home(self):
        from modelling.baseline_models import LadderModel

        df = pd.DataFrame({
            "home_ladder_pos": [16],  # Much lower ranked
            "away_ladder_pos": [1],
        })
        model = LadderModel()
        model.fit(df)
        pred = model.predict(df)
        assert pred[0] == 0  # Away should win

    def test_predict_proba_shape_and_range(self, small_feature_df):
        from modelling.baseline_models import LadderModel

        model = LadderModel()
        model.fit(small_feature_df)
        proba = model.predict_proba(small_feature_df)
        assert proba.shape == (len(small_feature_df), 2)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_equal_positions_home_advantage(self):
        from modelling.baseline_models import LadderModel

        df = pd.DataFrame({
            "home_ladder_pos": [5],
            "away_ladder_pos": [5],
        })
        model = LadderModel()
        model.fit(df)
        proba = model.predict_proba(df)
        # When positions are equal, home prob should be exactly 0.5
        assert abs(proba[0, 1] - 0.5) < 0.01


# ============================================================================
# EloModel tests
# ============================================================================


class TestEloModel:
    """Tests for modelling.baseline_models.EloModel."""

    def test_fit_and_predict(self, small_feature_df, small_target):
        from modelling.baseline_models import EloModel

        model = EloModel(k_factor=25, home_advantage=50)
        model.fit(small_feature_df, small_target)
        assert model.is_fitted_

        preds = model.predict(small_feature_df)
        assert len(preds) == len(small_feature_df)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, small_feature_df, small_target):
        from modelling.baseline_models import EloModel

        model = EloModel()
        model.fit(small_feature_df, small_target)
        proba = model.predict_proba(small_feature_df)
        assert proba.shape == (len(small_feature_df), 2)

    def test_predict_proba_sums_to_one(self, small_feature_df, small_target):
        from modelling.baseline_models import EloModel

        model = EloModel()
        model.fit(small_feature_df, small_target)
        proba = model.predict_proba(small_feature_df)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_unfitted_predict_raises(self, small_feature_df):
        from modelling.baseline_models import EloModel

        model = EloModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(small_feature_df)

    def test_get_ratings_after_fit(self, small_feature_df, small_target):
        from modelling.baseline_models import EloModel

        model = EloModel()
        model.fit(small_feature_df, small_target)
        ratings = model.get_ratings()
        assert len(ratings) > 0
        assert all(isinstance(v, float) for v in ratings.values())

    def test_get_params_and_set_params(self):
        from modelling.baseline_models import EloModel

        model = EloModel(k_factor=30)
        params = model.get_params()
        assert params["k_factor"] == 30

        model.set_params(k_factor=50)
        assert model.k_factor == 50


# ============================================================================
# Classical model factories tests
# ============================================================================


class TestClassicalModels:
    """Tests for modelling.classical_models factory functions."""

    def test_build_logistic_regression_returns_pipeline(self):
        from modelling.classical_models import build_logistic_regression

        model = build_logistic_regression()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_build_logistic_regression_with_custom_params(self):
        from modelling.classical_models import build_logistic_regression

        model = build_logistic_regression({"C": 0.5, "max_iter": 500})
        # The classifier is the second step in the pipeline
        clf = model.named_steps["clf"]
        assert clf.C == 0.5
        assert clf.max_iter == 500

    def test_build_random_forest_returns_classifier(self):
        from modelling.classical_models import build_random_forest

        model = build_random_forest()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_build_random_forest_custom_params(self):
        from modelling.classical_models import build_random_forest

        model = build_random_forest({"n_estimators": 10, "max_depth": 3})
        assert model.n_estimators == 10
        assert model.max_depth == 3

    def test_get_model_logistic_regression(self):
        from modelling.classical_models import get_model

        model = get_model("logistic_regression")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    def test_get_model_aliases(self):
        from modelling.classical_models import get_model

        model_lr = get_model("logreg")
        assert hasattr(model_lr, "fit")

        model_rf = get_model("rf")
        assert hasattr(model_rf, "fit")

    def test_get_model_unknown_raises(self):
        from modelling.classical_models import get_model

        with pytest.raises(ValueError, match="Unknown model name"):
            get_model("nonexistent_model")

    def test_get_model_case_insensitive(self):
        from modelling.classical_models import get_model

        model = get_model("LOGISTIC_REGRESSION")
        assert hasattr(model, "fit")

    def test_list_models_returns_list(self):
        from modelling.classical_models import list_models

        models = list_models()
        assert isinstance(models, list)
        assert "logistic_regression" in models
        assert "random_forest" in models

    def test_logistic_regression_fit_predict(self, numeric_features, small_target):
        from modelling.classical_models import build_logistic_regression

        model = build_logistic_regression()
        model.fit(numeric_features, small_target)
        preds = model.predict(numeric_features)
        assert len(preds) == len(numeric_features)
        assert set(preds).issubset({0, 1})

    def test_random_forest_fit_predict(self, numeric_features, small_target):
        from modelling.classical_models import build_random_forest

        model = build_random_forest({"n_estimators": 10, "max_depth": 3})
        model.fit(numeric_features, small_target)
        preds = model.predict(numeric_features)
        assert len(preds) == len(numeric_features)

    def test_all_models_have_sklearn_interface(self):
        """All models from the factory should have fit/predict/predict_proba."""
        from modelling.classical_models import build_logistic_regression, build_random_forest

        for builder in [build_logistic_regression, build_random_forest]:
            model = builder()
            assert hasattr(model, "fit"), f"{type(model).__name__} missing fit"
            assert hasattr(model, "predict"), f"{type(model).__name__} missing predict"
            assert hasattr(model, "predict_proba"), f"{type(model).__name__} missing predict_proba"


# ============================================================================
# Ensemble tests
# ============================================================================


class _MockModel:
    """Simple mock model for ensemble testing."""

    def __init__(self, pred_proba_home: float = 0.6):
        self._p = pred_proba_home
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.ones(n, dtype=int) if self._p >= 0.5 else np.zeros(n, dtype=int)

    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full((n, 2), [1 - self._p, self._p])


class TestVotingEnsemble:
    """Tests for modelling.ensemble.VotingEnsemble."""

    def test_soft_voting_averages_probabilities(self, numeric_features, small_target):
        from modelling.ensemble import VotingEnsemble

        m1 = _MockModel(pred_proba_home=0.7)
        m2 = _MockModel(pred_proba_home=0.5)
        ensemble = VotingEnsemble(base_models=[m1, m2], voting="soft")
        ensemble.fit(numeric_features, small_target)

        proba = ensemble.predict_proba(numeric_features)
        # Average of 0.7 and 0.5 = 0.6
        assert abs(proba[0, 1] - 0.6) < 0.01

    def test_soft_voting_with_weights(self, numeric_features, small_target):
        from modelling.ensemble import VotingEnsemble

        m1 = _MockModel(pred_proba_home=0.8)
        m2 = _MockModel(pred_proba_home=0.4)
        ensemble = VotingEnsemble(
            base_models=[m1, m2],
            voting="soft",
            weights=[3.0, 1.0],
        )
        ensemble.fit(numeric_features, small_target)
        proba = ensemble.predict_proba(numeric_features)
        # Weighted: (3*0.8 + 1*0.4) / 4 = 2.8/4 = 0.7
        assert abs(proba[0, 1] - 0.7) < 0.01

    def test_hard_voting_majority(self, numeric_features, small_target):
        from modelling.ensemble import VotingEnsemble

        m1 = _MockModel(pred_proba_home=0.7)  # Predicts 1
        m2 = _MockModel(pred_proba_home=0.7)  # Predicts 1
        m3 = _MockModel(pred_proba_home=0.3)  # Predicts 0
        ensemble = VotingEnsemble(
            base_models=[m1, m2, m3], voting="hard"
        )
        ensemble.fit(numeric_features, small_target)
        preds = ensemble.predict(numeric_features)
        # 2 out of 3 predict 1
        assert (preds == 1).all()

    def test_voting_ensemble_predict_proba_shape(self, numeric_features, small_target):
        from modelling.ensemble import VotingEnsemble

        m1 = _MockModel(0.6)
        m2 = _MockModel(0.5)
        ensemble = VotingEnsemble(base_models=[m1, m2])
        ensemble.fit(numeric_features, small_target)
        proba = ensemble.predict_proba(numeric_features)
        assert proba.shape == (len(numeric_features), 2)

    def test_voting_ensemble_unfitted_raises(self, numeric_features):
        from modelling.ensemble import VotingEnsemble

        m1 = _MockModel(0.6)
        ensemble = VotingEnsemble(base_models=[m1])
        with pytest.raises(RuntimeError, match="not been fitted"):
            ensemble.predict_proba(numeric_features)

    def test_voting_invalid_mode_raises(self):
        from modelling.ensemble import VotingEnsemble

        with pytest.raises(ValueError, match="voting must be"):
            VotingEnsemble(base_models=[_MockModel()], voting="invalid")

    def test_voting_ensemble_fits_base_models(self, numeric_features, small_target):
        from modelling.ensemble import VotingEnsemble

        m1 = _MockModel(0.6)
        m2 = _MockModel(0.5)
        ensemble = VotingEnsemble(base_models=[m1, m2], refit=True)
        ensemble.fit(numeric_features, small_target)
        assert m1.fitted
        assert m2.fitted

    def test_voting_ensemble_no_refit(self, numeric_features, small_target):
        from modelling.ensemble import VotingEnsemble

        m1 = _MockModel(0.6)
        ensemble = VotingEnsemble(base_models=[m1], refit=False)
        ensemble.fit(numeric_features, small_target)
        assert not m1.fitted  # Should not have been fitted


class TestStackingEnsemble:
    """Tests for modelling.ensemble.StackingEnsemble."""

    def test_stacking_fit_and_predict(self, numeric_features, small_target):
        from modelling.ensemble import StackingEnsemble

        m1 = _MockModel(0.6)
        m2 = _MockModel(0.5)
        stack = StackingEnsemble(base_models=[m1, m2], n_folds=2)
        stack.fit(numeric_features, small_target)
        assert stack.is_fitted_

        preds = stack.predict(numeric_features)
        assert len(preds) == len(numeric_features)
        assert set(preds).issubset({0, 1})

    def test_stacking_predict_proba_shape(self, numeric_features, small_target):
        from modelling.ensemble import StackingEnsemble

        m1 = _MockModel(0.6)
        m2 = _MockModel(0.5)
        stack = StackingEnsemble(base_models=[m1, m2], n_folds=2)
        stack.fit(numeric_features, small_target)
        proba = stack.predict_proba(numeric_features)
        assert proba.shape == (len(numeric_features), 2)

    def test_stacking_unfitted_raises(self, numeric_features):
        from modelling.ensemble import StackingEnsemble

        m1 = _MockModel(0.6)
        stack = StackingEnsemble(base_models=[m1])
        with pytest.raises(RuntimeError, match="not been fitted"):
            stack.predict_proba(numeric_features)

    def test_stacking_base_models_refitted_on_full_data(
        self, numeric_features, small_target
    ):
        from modelling.ensemble import StackingEnsemble

        m1 = _MockModel(0.6)
        m2 = _MockModel(0.5)
        stack = StackingEnsemble(base_models=[m1, m2], n_folds=2)
        stack.fit(numeric_features, small_target)
        # After fit, base models should be fitted on full data
        assert m1.fitted
        assert m2.fitted


# ============================================================================
# Calibration tests
# ============================================================================


class TestCalibration:
    """Tests for modelling.calibration."""

    def test_calibrated_model_platt(self, numeric_features, small_target):
        from modelling.calibration import CalibratedModel

        base = _MockModel(0.6)
        cal = CalibratedModel(base, method="platt")
        cal.fit(numeric_features, small_target)
        assert cal.is_fitted_

        proba = cal.predict_proba(numeric_features)
        assert proba.shape == (len(numeric_features), 2)
        # Calibrated probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_calibrated_model_isotonic(self, numeric_features, small_target):
        from modelling.calibration import CalibratedModel

        base = _MockModel(0.6)
        cal = CalibratedModel(base, method="isotonic")
        cal.fit(numeric_features, small_target)
        proba = cal.predict_proba(numeric_features)
        assert proba.shape == (len(numeric_features), 2)

    def test_calibrated_model_predict_binary(self, numeric_features, small_target):
        from modelling.calibration import CalibratedModel

        base = _MockModel(0.6)
        cal = CalibratedModel(base, method="platt")
        cal.fit(numeric_features, small_target)
        preds = cal.predict(numeric_features)
        assert set(preds).issubset({0, 1})

    def test_calibrated_model_unfitted_raises(self, numeric_features):
        from modelling.calibration import CalibratedModel

        cal = CalibratedModel(_MockModel(0.6))
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.predict_proba(numeric_features)

    def test_calibrated_model_invalid_method_raises(self):
        from modelling.calibration import CalibratedModel

        with pytest.raises(ValueError, match="method must be"):
            CalibratedModel(_MockModel(0.6), method="invalid")

    def test_calibrated_model_with_explicit_cal_set(
        self, numeric_features, small_target
    ):
        from modelling.calibration import CalibratedModel

        base = _MockModel(0.6)
        n = len(numeric_features)
        split = int(n * 0.7)

        cal = CalibratedModel(base, method="platt")
        cal.fit(
            X_train=numeric_features.iloc[:split],
            y_train=small_target[:split],
            X_cal=numeric_features.iloc[split:],
            y_cal=small_target[split:],
        )
        assert cal.is_fitted_

    def test_compute_ece_perfect_calibration(self):
        from modelling.calibration import compute_ece

        # Perfect calibration: predicted prob matches actual outcomes
        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
        y_prob = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        ece = compute_ece(y_true, y_prob, n_bins=5)
        assert ece < 0.1  # Should be very low

    def test_compute_ece_range(self):
        from modelling.calibration import compute_ece

        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        ece = compute_ece(y_true, y_prob)
        assert 0.0 <= ece <= 1.0

    def test_calibrate_platt_returns_logistic(self, numeric_features, small_target):
        from modelling.calibration import calibrate_platt

        base = _MockModel(0.6)
        base.fit(numeric_features, small_target)
        calibrator = calibrate_platt(base, numeric_features, small_target)
        assert hasattr(calibrator, "predict_proba")


# ============================================================================
# All models implement the full interface
# ============================================================================


class TestModelInterface:
    """Verify all model types implement fit/predict/predict_proba."""

    MODEL_CLASSES = [
        ("HomeAlwaysModel", "modelling.baseline_models"),
        ("OddsImpliedModel", "modelling.baseline_models"),
        ("LadderModel", "modelling.baseline_models"),
        ("EloModel", "modelling.baseline_models"),
    ]

    @pytest.mark.parametrize("class_name,module_path", MODEL_CLASSES)
    def test_model_has_full_interface(self, class_name, module_path):
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        model = cls()
        assert hasattr(model, "fit"), f"{class_name} missing fit"
        assert hasattr(model, "predict"), f"{class_name} missing predict"
        assert hasattr(model, "predict_proba"), f"{class_name} missing predict_proba"
        assert hasattr(model, "get_params"), f"{class_name} missing get_params"
        assert hasattr(model, "set_params"), f"{class_name} missing set_params"

    def test_voting_ensemble_has_interface(self):
        from modelling.ensemble import VotingEnsemble

        ens = VotingEnsemble(base_models=[_MockModel()])
        assert hasattr(ens, "fit")
        assert hasattr(ens, "predict")
        assert hasattr(ens, "predict_proba")

    def test_stacking_ensemble_has_interface(self):
        from modelling.ensemble import StackingEnsemble

        ens = StackingEnsemble(base_models=[_MockModel()])
        assert hasattr(ens, "fit")
        assert hasattr(ens, "predict")
        assert hasattr(ens, "predict_proba")

    def test_calibrated_model_has_interface(self):
        from modelling.calibration import CalibratedModel

        cal = CalibratedModel(_MockModel())
        assert hasattr(cal, "fit")
        assert hasattr(cal, "predict")
        assert hasattr(cal, "predict_proba")
