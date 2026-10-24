"""Offline smoke test with fake API data: python tests/test_smoke.py"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import predict


def adv(op, orr, dp, dr):
    return dict(offense=dict(passingPlays=dict(ppa=op), rushingPlays=dict(ppa=orr)),
                defense=dict(passingPlays=dict(ppa=dp), rushingPlays=dict(ppa=dr)))


def fake(path, **p):
    if path == "/rankings":
        return [dict(week=5, polls=[dict(poll="AP Top 25", ranks=[
            dict(rank=1, school="Alpha"), dict(rank=2, school="Beta")])])]
    if path == "/games":
        return [
            dict(id=1, week=4, homeTeam="Alpha", awayTeam="Gamma", completed=True, homePoints=30, awayPoints=20,
                 homeLineScores=[7, 10, 7, 6], awayLineScores=[3, 7, 7, 3]),
            dict(id=2, week=4, homeTeam="Beta", awayTeam="Gamma", completed=True, homePoints=27, awayPoints=24,
                 homeLineScores=[7, 3, 7, 7, 3], awayLineScores=[7, 7, 7, 3, 0]),
            dict(id=3, week=5, homeTeam="Alpha", awayTeam="Beta", completed=False, neutralSite=False,
                 startDate="2026-10-31T19:00:00Z"),
        ]
    if path == "/ratings/sp":
        return [dict(team="Alpha", rating=25), dict(team="Beta", rating=22), dict(team="Gamma", rating=5)]
    if path == "/ratings/elo":
        return [dict(team="Alpha", elo=2000), dict(team="Beta", elo=1950), dict(team="Gamma", elo=1500)]
    if path == "/stats/season/advanced":
        return [dict(team="Alpha", **adv(.3, .1, .0, .1)), dict(team="Beta", **adv(.1, .2, .2, -.1)),
                dict(team="Gamma", **adv(0, 0, .1, .1))]
    if path == "/stats/season":
        return [dict(team=t, statName=s, statValue=v) for t in ("Alpha", "Beta", "Gamma")
                for s, v in (("passAttempts", 300), ("rushingAttempts", 300))]
    raise AssertionError(path)


predict.api_get = fake
out = os.path.join(tempfile.mkdtemp(), "p.json")
sys.argv = ["predict", "--year", "2026", "--out", out]
predict.main()
d = json.load(open(out))
assert d["week"] == 5 and len(d["games"]) == 1, d
print(json.dumps(d["games"][0], indent=1))
print(d["teams"]["Beta"])
print("OK")
