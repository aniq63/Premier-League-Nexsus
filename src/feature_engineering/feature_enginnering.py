import pandas as pd
from src.utils.logger import logging

class RowTracker:
    def __init__(self):
        self.log = []
        self._current = None

    def before(self, step_name: str, df: pd.DataFrame):
        self._current = {"step": step_name, "before": len(df)}

    def after(self, df: pd.DataFrame, note: str = ""):
        assert self._current is not None, "Call .before() first"
        entry = {
            **self._current,
            "after": len(df),
            "dropped": self._current["before"] - len(df),
            "note": note,
        }
        self.log.append(entry)

        dropped = entry["dropped"]
        flag = " ROWS DROPPED" if dropped > 0 else ""

        logging.info(f"[{entry['step']}] {entry['before']} -> {entry['after']} "
                     f"(dropped {dropped}){flag}")

        self._current = None

    def print_audit(self):
        logging.info("\n" + "=" * 60)
        logging.info("PIPELINE ROW AUDIT")
        logging.info("=" * 60)
        for e in self.log:
            logging.info(f"{e['step']}: {e['before']} -> {e['after']} "
                         f"(lost {e['dropped']})")
            if e["note"]:
                logging.info(f"  |_ {e['note']}")
        logging.info("=" * 60)


# ============================================================
# Feature Engineering Class
# ============================================================

class FeatureEngineering:

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        self.df = df.copy()
        self.tracker = RowTracker()

        logging.info(f"[INIT] Data shape: {self.df.shape}")

    # ============================================================
    # STEP 1 — Basic Features
    # ============================================================
    def basic_features(self):
        try:
            self.tracker.before("feature_engineering", self.df)

            df = self.df
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)

            df["result"] = df.apply(
                lambda r: "Win" if r["home_goals"] > r["away_goals"]
                else "Draw" if r["home_goals"] == r["away_goals"]
                else "Lose",
                axis=1
            )

            df["home_goals_conceded"] = df["away_goals"]
            df["away_goals_conceded"] = df["home_goals"]

            logging.info("Result distribution:")
            logging.info(df["result"].value_counts())

            self.df = df
            self.tracker.after(self.df, "no rows should be dropped here")
            return self

        except Exception as e:
            raise RuntimeError(f"Basic feature step failed: {e}")

    # ============================================================
    # STEP 2 — Rolling Features (timeline approach)
    # ============================================================
    def rolling_features(self):
        try:
            self.tracker.before("rolling_averages", self.df)

            df = self.df

            STAT_COLS = [
                "goals", "goals_conceded",
                "xg", "ppda", "deep_completions",
            ]

            # ---- Build timeline ----
            home = df[[
                "date", "home_team",
                "home_goals", "home_goals_conceded",
                "home_xg", "home_ppda", "home_deep_completions",
            ]].rename(columns={
                "home_team": "team",
                "home_goals": "goals",
                "home_goals_conceded": "goals_conceded",
                "home_xg": "xg",
                "home_ppda": "ppda",
                "home_deep_completions": "deep_completions",
            })
            home["venue"] = "home"

            away = df[[
                "date", "away_team",
                "away_goals", "away_goals_conceded",
                "away_xg", "away_ppda", "away_deep_completions",
            ]].rename(columns={
                "away_team": "team",
                "away_goals": "goals",
                "away_goals_conceded": "goals_conceded",
                "away_xg": "xg",
                "away_ppda": "ppda",
                "away_deep_completions": "deep_completions",
            })
            away["venue"] = "away"

            tl = pd.concat([home, away], ignore_index=True)
            tl.sort_values(["team", "date"], inplace=True)

            # ---- Rolling ----
            grp = tl.groupby("team")
            for col in STAT_COLS:
                tl[f"{col}_avg_last5"] = (
                    grp[col]
                    .rolling(5, closed="left")
                    .mean()
                    .reset_index(0, drop=True)
                )

            avg_cols = [f"{c}_avg_last5" for c in STAT_COLS]
            before = len(tl)
            tl.dropna(subset=avg_cols, inplace=True)
            logging.info(f"Timeline: {before} -> {len(tl)} (rolling drop)")

            # ---- Merge ----
            keep = ["date", "team"] + avg_cols

            home_avgs = tl[tl["venue"] == "home"][keep].rename(
                columns={"team": "home_team",
                         **{c: f"home_{c}" for c in avg_cols}}
            )

            away_avgs = tl[tl["venue"] == "away"][keep].rename(
                columns={"team": "away_team",
                         **{c: f"away_{c}" for c in avg_cols}}
            )

            df = df.merge(home_avgs, on=["date", "home_team"], how="left")
            df = df.merge(away_avgs, on=["date", "away_team"], how="left")

            before = len(df)
            df.dropna(inplace=True)
            logging.info(f"Matches dropped (rolling window): {before - len(df)}")

            self.df = df
            self.tracker.after(self.df, "dropped early matches (<5 games)")
            return self

        except Exception as e:
            raise RuntimeError(f"Rolling feature step failed: {e}")

    # ============================================================
    # STEP 3 — Points last 5
    # ============================================================
    def points_last5(self):
        try:
            self.tracker.before("points_last5", self.df)

            df = self.df

            home = df[["date", "home_team", "home_points"]].rename(
                columns={"home_team": "team", "home_points": "points"}
            )
            away = df[["date", "away_team", "away_points"]].rename(
                columns={"away_team": "team", "away_points": "points"}
            )

            tm = pd.concat([home, away], ignore_index=True)
            tm.sort_values(["team", "date"], inplace=True)

            tm["pts_last5"] = (
                tm.groupby("team")["points"]
                .rolling(5, closed="left")
                .sum()
                .reset_index(0, drop=True)
            )

            tm.dropna(inplace=True)

            df = df.merge(
                tm[["date", "team", "pts_last5"]],
                left_on=["date", "home_team"],
                right_on=["date", "team"],
                how="left"
            ).rename(columns={"pts_last5": "home_points_last5"}).drop(columns="team")

            df = df.merge(
                tm[["date", "team", "pts_last5"]],
                left_on=["date", "away_team"],
                right_on=["date", "team"],
                how="left"
            ).rename(columns={"pts_last5": "away_points_last5"}).drop(columns="team")

            before = len(df)
            df.dropna(inplace=True)
            logging.info(f"Points drop: {before - len(df)}")

            self.df = df
            self.tracker.after(self.df)
            return self

        except Exception as e:
            raise RuntimeError(f"Points feature step failed: {e}")

    # ============================================================
    # FINAL RUN
    # ============================================================
    def run(self):
        logging.info("\n========== PIPELINE START ==========")

        result = (
            self.basic_features()
                .rolling_features()
                .points_last5()
                .df
        )

        logging.info("\n========== PIPELINE COMPLETE ==========")
        self.tracker.print_audit()

        return result