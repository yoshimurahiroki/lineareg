import numpy as np
import pandas as pd

from lineareg.utils.eventstudy_helpers import ESCellSpec, build_cells, pret_for_cohort


def test_pret_for_cohort_uses_positions_and_anticipation() -> None:
    times = np.array([0, 1, 2, 3, 4, 5])
    # With anticipation=2, strictly pre requires pos(t)+2 < pos(g=5)=5 => t<=2
    assert pret_for_cohort(5, times, anticipation=2) == 2


def test_build_cells_drops_anticipation_leads() -> None:
    ids = np.repeat([1, 2], 6)
    times = np.tile(np.arange(6), 2)
    cohorts = np.where(ids == 1, 5, 0)
    y = np.zeros_like(times, dtype=float)

    df = pd.DataFrame({"id": ids, "t": times, "g": cohorts, "y": y})

    spec = ESCellSpec(
        id_name="id",
        t_name="t",
        cohort_name="g",
        y_name="y",
        anticipation=2,
        base_period="varying",
    )

    _df_aug, cell_keys, meta = build_cells(df, spec)
    assert 5 in meta["pret_map"]
    assert meta["pret_map"][5] == 2

    # t=3,4 are within anticipation window for g=5 and must be absent.
    assert (5, 3, 2) not in cell_keys
    assert (5, 4, 3) not in cell_keys

    # A true pre-period t=2 should remain, with varying base_t=1.
    assert (5, 2, 1) in cell_keys

    # A post-period t=5 should remain, with base_t=pret_g=2.
    assert (5, 5, 2) in cell_keys
