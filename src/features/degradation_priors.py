import pandas as pd


TARGET_COL = "fuel_adjusted_degradation_delta"


def get_prior_feature_columns() -> list[str]:
    return [
        "driver_compound_mean_deg",
        "team_compound_mean_deg",
        "team_stint_mean_deg",
        "compound_mean_deg",
        "driver_mean_deg",
        "team_mean_deg",
    ]


def _safe_group_mean(
    train_df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    new_col: str,
) -> pd.DataFrame:
    return (
        train_df.groupby(group_cols)[target_col]
        .mean()
        .reset_index()
        .rename(columns={target_col: new_col})
    )


def add_degradation_priors(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Adds historical degradation prior features.

    Important:
    - Priors are computed from train_df only.
    - Then they are merged into target_df.
    - This avoids holdout leakage.

    Features:
    - driver_compound_mean_deg
    - team_compound_mean_deg
    - team_stint_mean_deg
    - compound_mean_deg
    - driver_mean_deg
    - team_mean_deg
    """

    result = target_df.copy()
    global_mean = train_df[target_col].mean()

    prior_specs = [
        (
            ["Driver", "Compound"],
            "driver_compound_mean_deg",
        ),
        (
            ["Team", "Compound"],
            "team_compound_mean_deg",
        ),
        (
            ["Team", "Stint"],
            "team_stint_mean_deg",
        ),
        (
            ["Compound"],
            "compound_mean_deg",
        ),
        (
            ["Driver"],
            "driver_mean_deg",
        ),
        (
            ["Team"],
            "team_mean_deg",
        ),
    ]

    for group_cols, new_col in prior_specs:
        prior_df = _safe_group_mean(
            train_df=train_df,
            group_cols=group_cols,
            target_col=target_col,
            new_col=new_col,
        )

        result = result.merge(
            prior_df,
            on=group_cols,
            how="left",
        )

        result[new_col] = result[new_col].fillna(global_mean)

    return result