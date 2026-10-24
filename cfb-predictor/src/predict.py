"""CFB Top-25 game predictor.

Pulls data from the CollegeFootballData (CFBD) REST API, builds a blended
rating + matchup model, and writes docs/data/predictions.json for the website.

Usage:
    export CFBD_API_KEY=your_key
    python src/predict.py                 # latest AP week of the current season
    python src/predict.py --year 2025 --week 8
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
from collections import defaultdict

import requests

API = "https://api.collegefootballdata.com"

# ---- Model settings (tune these after you backtest) ------------------------
HFA = 2.5            # home-field advantage in points (0 on neutral sites)
W_SP = 0.7           # weight on SP+ rating margin
W_PPA = 0.3          # weight on pass/rush matchup (PPA) margin
PLAYS_PER_TEAM = 68  # converts PPA/play into points
MARGIN_SD = 13.5     # std dev of FBS game margins, for win probability
ELO_PER_POINT = 25.0 # fallback only: Elo points per point of spread
USE_PRIOR_YEAR_SP = True  # use last year's SP+ when a team lacks a current one
# ---------------------------------------------------------------------------


def api_get(path, **params):
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        sys.exit("Set the CFBD_API_KEY environment variable (free key: collegefootballdata.com/key).")
    r = requests.get(
        API + path,
        headers={"Authorization": f"Bearer {key}"},
        params=params,
        timeout=90,
    )
    r.raise_for_status()
    return r.json()


def default_season():
    today = dt.date.today()
    return today.year if today.month >= 6 else today.year - 1


def get_ap(year, week=None):
    """Return (week, {team: rank}) for the AP poll (latest week if week is None)."""
    best = None
    for wk in api_get("/rankings", year=year, seasonType="regular"):
        for poll in wk.get("polls", []):
            if poll.get("poll") != "AP Top 25":
                continue
            if week is not None and wk["week"] != week:
                continue
            if best is None or wk["week"] > best[0]:
                best = (wk["week"], {r["school"]: r["rank"] for r in poll["ranks"]})
    return best


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def g(d, *keys, default=None):
    """Safe nested get."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d


# ---- Halves / overtime -----------------------------------------------------
def halves_table(games):
    """Per-team first-half / second-half / overtime splits from line scores."""
    acc = defaultdict(lambda: dict(n=0, h1f=0, h1a=0, h2f=0, h2a=0, ot=0, otw=0))
    for gm in games:
        if not gm.get("completed"):
            continue
        hl, al = gm.get("homeLineScores"), gm.get("awayLineScores")
        if not hl or not al or len(hl) < 4 or len(al) < 4:
            continue
        sides = ((gm["homeTeam"], hl, al, gm.get("homePoints"), gm.get("awayPoints")),
                 (gm["awayTeam"], al, hl, gm.get("awayPoints"), gm.get("homePoints")))
        for team, mine, theirs, pf, pa in sides:
            a = acc[team]
            a["n"] += 1
            a["h1f"] += mine[0] + mine[1]
            a["h1a"] += theirs[0] + theirs[1]
            a["h2f"] += mine[2] + mine[3]
            a["h2a"] += theirs[2] + theirs[3]
            if len(mine) > 4:
                a["ot"] += 1
                if pf is not None and pa is not None and pf > pa:
                    a["otw"] += 1
    out = {}
    for team, a in acc.items():
        n = a["n"]
        h1, h2 = (a["h1f"] - a["h1a"]) / n, (a["h2f"] - a["h2a"]) / n
        out[team] = dict(
            games=n,
            h1_for=round(a["h1f"] / n, 1), h1_against=round(a["h1a"] / n, 1), h1_net=round(h1, 1),
            h2_for=round(a["h2f"] / n, 1), h2_against=round(a["h2a"] / n, 1), h2_net=round(h2, 1),
            ot_games=a["ot"], ot_wins=a["otw"],
            better_half="1st" if h1 > h2 else "2nd" if h2 > h1 else "even",
        )
    return out


# ---- Team profiles ---------------------------------------------------------
def build_profiles(year):
    sp = {r["team"]: r.get("rating") for r in api_get("/ratings/sp", year=year)}
    if USE_PRIOR_YEAR_SP:
        for r in api_get("/ratings/sp", year=year - 1):
            sp.setdefault(r["team"], r.get("rating"))
    elo = {r["team"]: r.get("elo") for r in api_get("/ratings/elo", year=year)}

    adv = {r["team"]: r for r in api_get("/stats/season/advanced", year=year, excludeGarbageTime=True)}

    att = defaultdict(dict)
    for r in api_get("/stats/season", year=year):
        if r.get("statName") in ("passAttempts", "rushingAttempts"):
            att[r["team"]][r["statName"]] = r.get("statValue")

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    lg = dict(
        o_pass=mean(g(a, "offense", "passingPlays", "ppa") for a in adv.values()),
        o_rush=mean(g(a, "offense", "rushingPlays", "ppa") for a in adv.values()),
        d_pass=mean(g(a, "defense", "passingPlays", "ppa") for a in adv.values()),
        d_rush=mean(g(a, "defense", "rushingPlays", "ppa") for a in adv.values()),
    )

    profiles = {}
    for team in set(sp) | set(elo) | set(adv):
        a = adv.get(team, {})
        pa, ra = att[team].get("passAttempts"), att[team].get("rushingAttempts")
        pass_rate = pa / (pa + ra) if pa and ra else 0.5
        profiles[team] = dict(
            sp=sp.get(team), elo=elo.get(team), pass_rate=pass_rate,
            o_pass=g(a, "offense", "passingPlays", "ppa"),
            o_rush=g(a, "offense", "rushingPlays", "ppa"),
            d_pass=g(a, "defense", "passingPlays", "ppa"),
            d_rush=g(a, "defense", "rushingPlays", "ppa"),
        )
    return profiles, lg


def matchup(o, d, lg):
    """Expected PPA/play for offense `o` against defense `d`, plus pass/rush edges."""
    if any(v is None for v in (o.get("o_pass"), o.get("o_rush"), d.get("d_pass"), d.get("d_rush"))):
        return None
    pass_edge = o["o_pass"] + d["d_pass"] - lg["o_pass"] - lg["d_pass"]
    rush_edge = o["o_rush"] + d["d_rush"] - lg["o_rush"] - lg["d_rush"]
    exp = o["pass_rate"] * (lg["o_pass"] + pass_edge) + (1 - o["pass_rate"]) * (lg["o_rush"] + rush_edge)
    return dict(exp=exp, pass_edge=pass_edge, rush_edge=rush_edge)


def predict_game(home, away, neutral, prof, lg):
    ph, pa = prof.get(home), prof.get(away)
    if not ph or not pa:
        return None
    hfa = 0.0 if neutral else HFA
    sp_margin = None
    if ph.get("sp") is not None and pa.get("sp") is not None:
        sp_margin = ph["sp"] - pa["sp"]
    elif ph.get("elo") and pa.get("elo"):
        sp_margin = (ph["elo"] - pa["elo"]) / ELO_PER_POINT

    mh = matchup(ph, pa, lg)   # home offense vs away defense
    ma = matchup(pa, ph, lg)   # away offense vs home defense
    ppa_margin = (mh["exp"] - ma["exp"]) * PLAYS_PER_TEAM if mh and ma else None

    if sp_margin is None and ppa_margin is None:
        return None
    if sp_margin is not None and ppa_margin is not None:
        base = W_SP * sp_margin + W_PPA * ppa_margin
    else:
        base = sp_margin if sp_margin is not None else ppa_margin
    margin = base + hfa  # positive = home favored
    return dict(margin=margin, sp_margin=sp_margin, ppa_margin=ppa_margin,
                home_matchup=mh, away_matchup=ma, win_prob_home=norm_cdf(margin / MARGIN_SD))


def reason(pick_is_home, res, home, away):
    pick, other = (home, away) if pick_is_home else (away, home)
    m = res["home_matchup"] if pick_is_home else res["away_matchup"]
    parts = []
    if res["sp_margin"] is not None:
        edge = res["sp_margin"] if pick_is_home else -res["sp_margin"]
        if edge > 0:
            parts.append(f"{pick} +{edge:.1f} on power rating")
    if m:
        if abs(m["rush_edge"]) >= abs(m["pass_edge"]):
            kind, v = f"run game vs. {other}'s run defense", m["rush_edge"]
        else:
            kind, v = f"pass game vs. {other}'s pass defense", m["pass_edge"]
        if v > 0.03:
            parts.append(f"{pick}'s {kind} is an advantage ({v:+.2f} PPA/play)")
    return "; ".join(parts) or "Close call: ratings and matchups roughly even"


# ---- Main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--out", default="docs/data/predictions.json")
    args = ap.parse_args()

    year = args.year or default_season()
    poll = get_ap(year, args.week)
    if not poll:
        sys.exit(f"No AP poll found for {year}" + (f" week {args.week}" if args.week else ""))
    week, ranks = poll

    games = api_get("/games", year=year, seasonType="regular", classification="fbs")
    halves = halves_table(games)
    profiles, lg = build_profiles(year)

    slate = [gm for gm in games
             if gm.get("week") == week and (gm["homeTeam"] in ranks or gm["awayTeam"] in ranks)]
    slate.sort(key=lambda gm: min(ranks.get(gm["homeTeam"], 99), ranks.get(gm["awayTeam"], 99)))

    out_games, teams_used = [], set()
    for gm in slate:
        home, away = gm["homeTeam"], gm["awayTeam"]
        res = predict_game(home, away, gm.get("neutralSite"), profiles, lg)
        if not res:
            continue
        pick_home = res["margin"] >= 0
        wp = res["win_prob_home"] if pick_home else 1 - res["win_prob_home"]
        entry = dict(
            id=gm.get("id"), start=gm.get("startDate"), neutral=bool(gm.get("neutralSite")),
            home=home, away=away, home_rank=ranks.get(home), away_rank=ranks.get(away),
            pick=home if pick_home else away,
            win_prob=round(wp, 3), margin=round(abs(res["margin"]), 1),
            reason=reason(pick_home, res, home, away),
            completed=bool(gm.get("completed")),
            home_points=gm.get("homePoints"), away_points=gm.get("awayPoints"),
        )
        if entry["completed"] and entry["home_points"] is not None and entry["away_points"] is not None:
            entry["correct"] = (entry["home_points"] > entry["away_points"]) == pick_home
        out_games.append(entry)
        teams_used.update((home, away))

    record = [e["correct"] for e in out_games if "correct" in e]
    payload = dict(
        generated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        year=year, week=week,
        model=dict(hfa=HFA, w_sp=W_SP, w_ppa=W_PPA, margin_sd=MARGIN_SD),
        week_record=dict(correct=sum(record), total=len(record)),
        games=out_games,
        teams={t: halves[t] for t in teams_used if t in halves},
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(out_games)} games for {year} week {week} -> {args.out}")


if __name__ == "__main__":
    main()
