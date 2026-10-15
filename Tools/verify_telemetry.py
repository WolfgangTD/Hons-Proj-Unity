"""
Sanity-check a telemetry capture before spending training time on it.

    python Tools/verify_telemetry.py                  # newest session
    python Tools/verify_telemetry.py Telemetry/session_20260901_101500_1234

Three things get checked, in increasing order of what they tell you:

  STRUCTURE   the files parse, the columns line up, steps and episodes join
  HYGIENE     no cheat label leaked into the feature stream
  SIGNAL      the cheats actually move the features they are supposed to move

The third is the one that matters. A capture can be perfectly well-formed and
still be useless, if honest and cheating play look identical in it.
"""

import sys
from pathlib import Path

import pandas as pd

KEY = ["session", "episode_id", "agent_id"]

# Anything matching these must never appear in steps.csv, or the detector can read
# the answer instead of learning behaviour.
LABEL_MARKERS = ("label_", "gt_", "wallhack", "aim_snap", "aimsnap", "trigger_bot", "triggerbot")

STEP_COLUMNS = 30
EPISODE_COLUMNS = 25
ACTIVATION_COLUMNS = 8

# activations.csv cheat name -> the episodes.csv counter it must reconcile against
GT_COUNTER = {
    "wallhack": "gt_wallhack_reveals",
    "aimsnap": "gt_aimsnap_adjustments",
    "triggerbot": "gt_triggerbot_shots",
}


def find_session(argv):
    if len(argv) > 1:
        return Path(argv[1])

    root = Path("Telemetry")
    if not root.is_dir():
        sys.exit("No Telemetry/ directory. Run a capture first, or pass a session path.")

    sessions = sorted(root.glob("session_*"), key=lambda p: p.stat().st_mtime)
    if not sessions:
        sys.exit("No session_* directories under Telemetry/.")

    return sessions[-1]


def head(title):
    print(f"\n{title}\n{'-' * len(title)}")


def check_structure(session):
    head(f"STRUCTURE  {session}")

    steps_path, episodes_path = session / "steps.csv", session / "episodes.csv"
    for p in (steps_path, episodes_path):
        if not p.is_file():
            sys.exit(f"missing {p}")

    steps = pd.read_csv(steps_path)
    episodes = pd.read_csv(episodes_path)

    ok = True

    print(f"steps.csv     {len(steps):>9,} rows x {len(steps.columns)} cols"
          f"   ({steps_path.stat().st_size / 1e6:.1f} MB)")
    print(f"episodes.csv  {len(episodes):>9,} rows x {len(episodes.columns)} cols")

    if len(steps.columns) != STEP_COLUMNS:
        print(f"  FAIL steps.csv should have {STEP_COLUMNS} columns")
        ok = False
    if len(episodes.columns) != EPISODE_COLUMNS:
        print(f"  FAIL episodes.csv should have {EPISODE_COLUMNS} columns")
        ok = False

    # A repeated header is the classic symptom of buffered-stream header detection:
    # it survives parsing as a string row rather than raising.
    repeats = (steps["session"] == "session").sum() + (episodes["session"] == "session").sum()
    if repeats:
        print(f"  FAIL header line repeated {repeats}x inside the data")
        ok = False

    numeric = steps.select_dtypes(include="number")
    if len(numeric.columns) < STEP_COLUMNS - 3:
        stringy = [c for c in steps.columns if c not in numeric.columns]
        print(f"  FAIL columns did not parse as numeric: {stringy}")
        ok = False

    nulls = steps.isna().sum()
    if nulls.any():
        print(f"  FAIL nulls present: {dict(nulls[nulls > 0])}")
        ok = False

    # A reused episode id silently fuses two different rounds into one training sequence,
    # and fans out every downstream join. It must be caught here or not at all.
    dup = episodes[episodes.duplicated(KEY, keep=False)]
    if not dup.empty:
        ids = sorted(dup["episode_id"].unique().tolist())
        print(f"  FAIL {len(dup)} episode rows reuse an id: {ids[:10]}")
        ok = False

    # Within one episode a decision index must run 0..n-1 exactly once. A restart mid-episode
    # means two rounds landed under the same id.
    restarts = []
    for keys, grp in steps.groupby(KEY, sort=False):
        d = grp["decision"].to_numpy()
        if len(d) != d[-1] + 1 or (d[1:] <= d[:-1]).any():
            restarts.append((keys[1], keys[2], len(d), int(d.max())))

    if restarts:
        print(f"  FAIL {len(restarts)} episodes have a restarting decision counter "
              f"(episode, agent, rows, max): {restarts[:5]}")
        ok = False

    # Every step must belong to a completed episode, or it has no label to train against.
    merged = steps.merge(episodes[KEY], on=KEY, how="left", indicator=True)
    orphans = (merged["_merge"] == "left_only").sum()
    orphan_eps = steps.loc[merged["_merge"] == "left_only", "episode_id"].nunique()

    print(f"\nepisodes      {len(episodes):>9,} labelled")
    print(f"orphan steps  {orphans:>9,}  ({orphan_eps} unfinished episodes)")

    # One round per arena is still in flight when the capture stops, so that many
    # unfinished episodes is expected rather than a fault.
    n_arenas = steps["arena_id"].nunique()
    if orphan_eps > n_arenas:
        print(f"  WARN {orphan_eps} unfinished episodes for {n_arenas} arenas - expected at most one each")

    print(f"arenas        {sorted(steps['arena_id'].unique().tolist())}")
    print(f"agents        {sorted(steps['agent_id'].unique().tolist())}")
    print(f"decisions/ep  median {episodes['decisions'].median():.0f}"
          f"   range {episodes['decisions'].min()}-{episodes['decisions'].max()}")

    return steps, episodes, ok


def check_hygiene(steps):
    head("HYGIENE")

    leaked = [c for c in steps.columns if any(m in c.lower() for m in LABEL_MARKERS)]
    if leaked:
        print(f"  FAIL label columns leaked into steps.csv: {leaked}")
        return False

    print("no label column in steps.csv  OK")
    return True


def describe(label, series, unit=""):
    if series.empty:
        print(f"  {label:<28} (no rows)")
        return
    print(f"  {label:<28} n={len(series):>7,}   median {series.median():7.3f}{unit}"
          f"   mean {series.mean():7.3f}{unit}")


def check_signal(steps, episodes):
    head("SIGNAL")

    labels = episodes[KEY + ["label_type", "label_cheating",
                             "label_wallhack", "label_aim_snap", "label_trigger_bot"]]
    df = steps.merge(labels, on=KEY, how="inner")

    if df.empty:
        print("  no joined rows - nothing to compare")
        return False

    counts = episodes["label_type"].value_counts()
    print("episodes by label:")
    for name, n in counts.items():
        print(f"  {name:<12} {n:>6,}  ({n / len(episodes):5.1%})")

    present = set(counts.index)
    honest = df[df["label_type"] == "none"]

    if "none" not in present:
        print("\n  no honest episodes captured - cannot compare")
        return False
    if present == {"none"}:
        print("\n  WARN only honest episodes captured. Raise Default Cheat Rate on the")
        print("       ArenaManager to 1 and re-run, or this proves nothing.")
        return False

    # Trigger-bot: fires the instant a target enters the cone, so dwell time collapses.
    print("\ntrigger-bot  -> cone_dwell_s on rows where a shot was taken on target")
    fired = df[(df["shot"] == 1) & (df["target_in_cone"] == 1)]
    describe("no trigger-bot", fired.loc[fired["label_trigger_bot"] == 0, "cone_dwell_s"], "s")
    describe("trigger-bot",    fired.loc[fired["label_trigger_bot"] > 0, "cone_dwell_s"], "s")

    # Aim-snap: closes the angle to the target in a single decision.
    print("\naim-snap     -> |aim_error_deg| with line of sight")
    los = df[df["has_los"] == 1]
    describe("no aim-snap", los.loc[los["label_aim_snap"] == 0, "aim_error_deg"].abs(), "deg")
    describe("aim-snap",    los.loc[los["label_aim_snap"] > 0, "aim_error_deg"].abs(), "deg")

    # Wallhack: tracks a target it has no honest way of seeing.
    print("\nwallhack     -> |aim_error_deg| with NO line of sight")
    blind = df[df["has_los"] == 0]
    describe("no wallhack", blind.loc[blind["label_wallhack"] == 0, "aim_error_deg"].abs(), "deg")
    describe("wallhack",    blind.loc[blind["label_wallhack"] > 0, "aim_error_deg"].abs(), "deg")
    
    print("\nepisode outcomes by label:")
    summary = episodes.groupby("label_type").agg(
        episodes=("episode_id", "count"),
        accuracy=("accuracy", "mean"),
        win_rate=("outcome", lambda s: (s == "win").mean()),
        reward=("cumulative_reward", "mean"),
    )
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    # The counters say whether a cheat ever actually fired, which is the difference
    # between "configured to cheat" and "cheated".
    cheats = episodes[episodes["label_cheating"] == 1]
    if not cheats.empty:
        print("\nground-truth cheat activations (episodes where the cheat never fired "
              "are unlearnable):")
        for col, sev, kind in (("gt_wallhack_reveals", "label_wallhack", "wallhack"),
                               ("gt_aimsnap_adjustments", "label_aim_snap", "aimsnap"),
                               ("gt_triggerbot_shots", "label_trigger_bot", "triggerbot")):
            sub = cheats[cheats[sev] > 0]
            if sub.empty:
                continue
            dead = (sub[col] == 0).mean()
            print(f"  {kind:<12} median {sub[col].median():6.1f} activations"
                  f"   {dead:5.1%} of episodes never fired")

    return True


def check_activations(session, steps, episodes):
    """The sidecar is only useful if it lines up with the two files it annotates.

    Two things are checked, and the second is the one that proves it: every activation
    must sit on a decision that exists in steps.csv, and the per-episode activation
    counts must reconcile exactly with the gt_* counters, which are incremented by the
    same code path but written independently.
    """
    path = session / "activations.csv"
    if not path.is_file():
        head("ACTIVATIONS")
        print("  no activations.csv - capture predates the sidecar, or logActivations is off")
        return True

    head("ACTIVATIONS")
    acts = pd.read_csv(path)
    ok = True

    print(f"activations.csv {len(acts):>9,} rows x {len(acts.columns)} cols")

    if len(acts.columns) != ACTIVATION_COLUMNS:
        print(f"  FAIL expected {ACTIVATION_COLUMNS} columns")
        ok = False

    if (acts["session"] == "session").any():
        print("  FAIL header line repeated inside the data")
        ok = False

    unknown = set(acts["cheat"].unique()) - set(GT_COUNTER)
    if unknown:
        print(f"  FAIL unknown cheat names: {sorted(unknown)}")
        ok = False

    print(f"by cheat        {acts['cheat'].value_counts().to_dict()}")

    # Every activation must land on a tick that actually produced a step row.
    step_keys = steps[KEY + ["decision"]].drop_duplicates()
    joined = acts.merge(step_keys, on=KEY + ["decision"], how="left", indicator=True)
    orphans = int((joined["_merge"] == "left_only").sum())
    print(f"orphan rows     {orphans:>9,}  (activations with no matching step)")
    if orphans:
        print("  FAIL decision indices do not align with steps.csv")
        ok = False

    # The reconciliation. Counted per episode per cheat, against the independent counter.
    counts = acts.groupby(KEY + ["cheat"]).size().unstack("cheat", fill_value=0)
    mismatches = []
    for cheat, col in GT_COUNTER.items():
        if col not in episodes.columns:
            continue
        got = counts[cheat] if cheat in counts.columns else None
        merged = episodes.set_index(KEY)[[col]].copy()
        merged["rows"] = got if got is not None else 0
        merged["rows"] = merged["rows"].fillna(0).astype(int)
        bad = merged[merged[col] != merged["rows"]]
        if not bad.empty:
            mismatches.append((cheat, len(bad), bad.head(3).to_dict("index")))

    if mismatches:
        for cheat, n, sample in mismatches:
            print(f"  FAIL {cheat}: {n} episodes where row count != counter; e.g. {sample}")
        ok = False
    else:
        print("counts reconcile with gt_* counters  OK")

    if "cheat" in acts.columns and not acts.empty:
        per_ep = acts.groupby(KEY).size()
        print(f"rows per episode median {per_ep.median():.0f}  max {per_ep.max()}")

    return ok


def main():
    session = find_session(sys.argv)
    steps, episodes, structure_ok = check_structure(session)
    hygiene_ok = check_hygiene(steps)
    activations_ok = check_activations(session, steps, episodes)
    signal_ok = check_signal(steps, episodes)

    head("VERDICT")
    for name, ok in (("structure", structure_ok), ("hygiene", hygiene_ok),
                     ("activations", activations_ok), ("signal", signal_ok)):
        print(f"  {name:<12} {'ok' if ok else 'CHECK ABOVE'}")

    return 0 if (structure_ok and hygiene_ok and activations_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
