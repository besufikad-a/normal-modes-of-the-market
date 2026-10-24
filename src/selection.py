"""Walk-forward supervised prediction of which candidate modes revert.

For episode ``j`` a model is trained on all (episode, mode) rows of
episodes ``< j`` — whose labels are realized by the start of episode
``j`` — and used to score episode ``j``'s candidates. Models are
re-trained at every episode on an expanding window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str, seed: int = 0):
    """Model factory. All models are deliberately lightly tuned."""
    if name == "logit":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000),
        )
    if name == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=150,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=seed,
        )
    if name == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=25,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
            ),
        )
    raise ValueError(f"Unknown model {name!r}.")


def walk_forward_predict(
    panel: pd.DataFrame,
    features: list[str],
    model: str = "gbm",
    min_train_episodes: int = 24,
    target: str = "y",
    retrain_every: int = 1,
    seed: int = 0,
) -> pd.Series:
    """Out-of-sample predicted probability of reversion for every row.

    Rows in the first ``min_train_episodes`` episodes get NaN.
    """
    pred = pd.Series(np.nan, index=panel.index, name=f"p_{model}")
    episodes = np.sort(panel["episode"].unique())
    fitted = None
    for n, j in enumerate(episodes):
        if j < min_train_episodes:
            continue
        test = panel["episode"] == j
        if fitted is None or n % retrain_every == 0:
            train = panel[(panel["episode"] < j) & panel[target].notna()]
            fitted = make_model(model, seed).fit(train[features], train[target])
        pred[test] = fitted.predict_proba(panel.loc[test, features])[:, 1]
    return pred


def oos_auc(panel: pd.DataFrame, score: str, target: str = "y") -> float:
    df = panel[[score, target]].dropna()
    return float(roc_auc_score(df[target], df[score]))


def block_bootstrap_auc(
    panel: pd.DataFrame, score: str, n_boot: int = 1000, seed: int = 0,
    target: str = "y",
) -> tuple[float, float]:
    """95% CI for AUC, resampling whole episodes (cross-sectional clusters)."""
    df = panel[["episode", score, target]].dropna()
    groups = {e: g for e, g in df.groupby("episode")}
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        draw = pd.concat([groups[k] for k in rng.choice(keys, len(keys))])
        if draw[target].nunique() == 2:
            stats.append(roc_auc_score(draw[target], draw[score]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def select_top_fraction(
    panel: pd.DataFrame, score: str, frac: float = 1 / 3, pool: pd.Series | None = None
) -> pd.Series:
    """Binary weights: top ``frac`` of each episode's pool by ``score``."""
    w = pd.Series(0.0, index=panel.index)
    mask = panel[score].notna()
    if pool is not None:
        mask &= pool
    for _, g in panel[mask].groupby("episode"):
        k = max(1, int(round(frac * len(g))))
        w[g[score].nlargest(k).index] = 1.0
    return w
