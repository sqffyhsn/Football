"""Verify which timezone football-data.co.uk `Time` values are in, from the data itself.

Evidence used (all well-known kick-off conventions):
  - D1 Bundesliga Saturday afternoon slot is 15:30 German time (14:30 UK time).
  - SP1/I1 evening slots are 21:00 / 20:45 local (20:00 / 19:45 UK time).
  - E0 Saturday afternoon slot is 15:00 UK time.
  - El Clasico, Real Madrid v Barcelona, 26 Oct 2024 kicked off 21:00 CEST = 20:00 BST.
If the German/Spanish/Italian times appear one hour EARLIER than local, the file uses
UK time (Europe/London). Writes reports/timezone_check.md; update config.yaml by hand.
"""
import pandas as pd

from fh.config import load_config, path
from fh.data.clean import normalise, read_raw_csv
from fh.data.download import raw_path


def modal_times(df: pd.DataFrame, div: str, weekday: int, top: int = 3) -> list[tuple[str, int]]:
    d = df[(df["div"] == div) & (df["date"].dt.weekday == weekday) & df["time"].notna()]
    return list(d["time"].value_counts().head(top).items())


def main():
    cfg = load_config()
    frames = []
    for season in cfg["seasons"]["history"][-3:]:
        for div in ("E0", "D1", "SP1", "I1"):
            p = raw_path(cfg, season, div)
            if p.exists():
                frames.append(normalise(read_raw_csv(p), season=season, div=div))
    if not frames:
        raise SystemExit("No raw data for recent seasons; run scripts/download_data.py first.")
    df = pd.concat(frames, ignore_index=True)

    lines = ["# Kick-off timezone check", ""]
    sat = 5
    for div, local_expect in (("E0", "15:00 (UK)"), ("D1", "15:30 local / 14:30 UK"),
                              ("SP1", "21:00 local / 20:00 UK (evening)"), ("I1", "20:45 local / 19:45 UK (evening)")):
        lines.append(f"- {div} Saturday modal kick-offs: {modal_times(df, div, sat)} — expected {local_expect}")
    clasico = df[(df["div"] == "SP1") & (df["date"] == "2024-10-26") & (df["home"] == "Real Madrid")]
    if len(clasico):
        lines.append(f"- Real Madrid v Barcelona 26/10/2024 listed at {clasico['time'].iloc[0]} "
                     "(actual 21:00 CEST = 20:00 BST)")
    d1 = dict(modal_times(df, "D1", sat, 5))
    uk = d1.get("14:30", 0) > d1.get("15:30", 0)
    lines += ["", f"**Conclusion:** times look like {'UK time (Europe/London)' if uk else 'LOCAL time — NOT UK'}.",
              "Set `data.kickoff_timezone` and `kickoff_timezone_verified: true` in config.yaml accordingly."]
    out = path("reports/timezone_check.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
