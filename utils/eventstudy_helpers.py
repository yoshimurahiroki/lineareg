"""Helpers for cell-based Event Study estimators (CS, DR-DID).

Shared logic for constructing (g,t) cells, handling anticipation/base periods,
and aggregating results by event-time tau.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence
else:
    Sequence = tuple

from lineareg.utils.helpers import time_to_pos, event_tau, prev_time

def pret_for_cohort(g, times_sorted, anticipation: int = 0):
    """Find the last *strictly untreated* period for cohort g given anticipation.

    We treat `anticipation` as a number of time periods (positions) rather than
    numeric time arithmetic. A period t is strictly pre-treatment iff
    pos(t) + anticipation < pos(g) (with pos(g)=len(times) if g is after-sample).
    """
    if int(anticipation) < 0:
        raise ValueError("anticipation must be >= 0.")
    t2pos = time_to_pos(times_sorted)
    g_pos = int(t2pos.get(g, len(times_sorted)))
    last_pre_pos = g_pos - int(anticipation) - 1
    if last_pre_pos < 0:
        return None
    return times_sorted[last_pre_pos]


@dataclass
class ESCellSpec:
    """Specification for Event Study (g,t) cell construction."""
    id_name: str
    t_name: str
    cohort_name: str
    y_name: str
    event_time_name: str | None = None
    control_group: str = "notyet"
    center_at: int = -1
    covariate_names: Sequence[str] | None = None
    cov_method: str = "none"
    anticipation: int = 0
    base_period: str = "varying"

    def control_mask(self, df: pd.DataFrame, g, t, pret, times_sorted) -> np.ndarray:
        cg_normalized = self.control_group.lower().replace("treated", "")
        cohort_arr = df[self.cohort_name].to_numpy()
        if cg_normalized == "never":
            return cohort_arr == 0
        if cg_normalized == "notyet":
            if int(self.anticipation) < 0:
                raise ValueError("anticipation must be >= 0.")
            t2pos = time_to_pos(times_sorted)
            t_pos = t2pos.get(t, -1)
            if t_pos < 0:
                raise ValueError(
                    "Time t is not present in times_sorted; cannot form not-yet-treated controls.",
                )
            # Not-yet-treated controls at time t are units that are never treated
            # (cohort==0) or treated strictly after t+anticipation. This cutoff
            # must depend on the evaluation time t, not on the cohort-specific
            # pre-period (pret).
            cutoff_pos = t_pos + int(self.anticipation)
            cutoff_vals = [tv for tv in times_sorted if t2pos[tv] > cutoff_pos]
            if len(cutoff_vals) == 0:
                cutoff = float('inf')
            else:
                cutoff = min(cutoff_vals)
            return (cohort_arr == 0) | (cohort_arr >= cutoff)
        raise ValueError(
            "control_group must be 'never'/'nevertreated' or 'notyet'/'notyettreated'",
        )


def build_cells(
    df: pd.DataFrame, spec: ESCellSpec,
) -> tuple[pd.DataFrame, list[tuple], dict]:
    """Construct (g,t) cells for long-difference estimation."""
    # DID compatibility checks: require panel data (no repeated cross-sections)
    id_name = spec.id_name
    t_name = spec.t_name
    # Strict: require non-missing identifiers and timing/cohort values.
    for col in (id_name, t_name, spec.cohort_name):
        if df[col].isna().any():
            raise ValueError(f"Missing values detected in required column '{col}'.")
    if df.duplicated([id_name, t_name]).any():
        raise ValueError("DID requires unique (id,time) rows. Duplicates detected.")
    t_per_id = df.groupby(id_name)[t_name].nunique()
    if t_per_id.max() <= 1:
        raise ValueError("Repeated cross-section (RCS) is not supported. Panel data required.")

    # Unbalanced panel is allowed; behavior is explicit.
    df_aug = df.copy()
    times = np.sort(df_aug[spec.t_name].unique())
    cohorts = np.sort(df_aug[spec.cohort_name].unique())
    pret_map = {}
    for g in cohorts:
        if g > 0:
            pret = pret_for_cohort(g, times, spec.anticipation)
            if pret is not None:
                pret_map[g] = pret
    cell_keys = []
    for g in cohorts:
        if g <= 0 or g not in pret_map:
            continue
        pret_g = pret_map[g]
        t2pos = time_to_pos(times)
        g_pos = t2pos.get(g, len(times))
        for t in times:
            t_pos = t2pos.get(t, -1)
            if t_pos < 0:
                continue

            # Drop periods that are before g but within the anticipation window
            # (i.e., not strictly untreated). This prevents contaminated placebo
            # leads from entering the (g,t) cell list.
            if int(t_pos) < int(g_pos) and (int(t_pos) + int(spec.anticipation)) >= int(g_pos):
                continue

            if spec.base_period == "varying" and t_pos < g_pos:
                base_t = prev_time(t, times)
                if base_t is None:
                    continue
            else:
                base_t = pret_g
            cell_keys.append((g, t, base_t))
    return df_aug, cell_keys, {"times": times, "pret_map": pret_map}


def aggregate_tau(att_gt: pd.DataFrame, base_tau: int) -> tuple[pd.DataFrame, dict]:
    """Aggregate (g,t) ATTs into event-time ATTs."""
    base_tau_int = int(base_tau)
    tau_values = {int(t) for t in att_gt["tau"].unique()}
    tau_values.add(base_tau_int)
    taus = sorted(tau_values)
    att_tau = pd.DataFrame({"tau": taus, "att": 0.0})
    rows_by_tau = {tau: att_gt[att_gt["tau"] == tau].index.to_numpy() for tau in taus}
    return att_tau, rows_by_tau
