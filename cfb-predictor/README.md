# Top 25 Game Predictor

Free, automated predictions for every game involving an AP Top 25 team.
Data from [CollegeFootballData.com](https://collegefootballdata.com); site hosted on GitHub Pages.

## What it does
- Finds the latest AP Top 25 and every game that week involving a ranked team
- Predicts a winner, margin, and win probability from:
  - SP+ power rating (already adjusted for strength of schedule); last year's rating fills in for teams missing one; Elo is the fallback
  - Pass/rush matchup: each offense's pass and rush PPA vs. the opposing defense's, weighted by how often the offense passes
  - Home-field advantage (none at neutral sites)
- Shows first-half, second-half, and overtime tendencies for each team (from quarter line scores)
- Grades finished games so you can track accuracy

## Setup (about 10 minutes)
1. Get a free API key: https://collegefootballdata.com/key
2. Create a GitHub repo and push these files.
3. Repo **Settings > Secrets and variables > Actions > New repository secret**: name `CFBD_API_KEY`, value = your key.
4. Repo **Settings > Pages**: Source = *Deploy from a branch*, Branch = `main`, folder = `/docs`.
5. **Actions > Update predictions > Run workflow**. Your site appears at `https://<you>.github.io/<repo>/`.

It then refreshes itself every Tuesday and Saturday.

## Run locally
```
pip install -r requirements.txt
export CFBD_API_KEY=your_key
python src/predict.py            # or: --year 2025 --week 8
python -m http.server -d docs    # open http://localhost:8000
python tests/test_smoke.py       # offline test, no key needed
```

## Tuning
Model weights are constants at the top of `src/predict.py` (`HFA`, `W_SP`, `W_PPA`, ...).
Next steps: backtest on past seasons to fit those weights, add a calibrated logistic model, add rest/travel and injury adjustments.
