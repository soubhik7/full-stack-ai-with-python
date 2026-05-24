# Chapter 03 — Training Data

This directory holds CSV datasets used by the ML notebooks.
Most files are small and committed to git.
**One file is large and must be downloaded separately** — see below.

---

## Files in `training_data/`

| File | Size | In git? | Used by |
|------|------|---------|---------|
| `ChurnData.csv` | 35 KB | ✅ yes | Logistic regression notebook |
| `Cust_Segmentation.csv` | 33 KB | ✅ yes | Clustering notebook |
| `FuelConsumption.csv` | 71 KB | ✅ yes | Regression notebooks |
| `cars_clus.csv` | 20 KB | ✅ yes | Hierarchical clustering |
| `cell_samples.csv` | 20 KB | ✅ yes | SVM cancer notebook |
| `china_gdp.csv` | 1.2 KB | ✅ yes | Nonlinear regression |
| `drug200.csv` | 5.9 KB | ✅ yes | Decision tree notebook |
| `teleCust1000t.csv` | 31 KB | ✅ yes | KNN notebook |
| `weather-stations*.csv` | 127 KB | ✅ yes | DBSCAN notebook |
| `links.csv` | 709 KB | ✅ yes | Recommendation notebooks |
| `movies.csv` | 1.6 MB | ✅ yes | Recommendation notebooks |
| `tags.csv` | 20 MB | ✅ yes | Recommendation notebooks |
| **`ratings.csv`** | **591 MB** | ❌ **gitignored** | **Collaborative filtering notebook** |

> `ratings.csv` exceeds GitHub's 100 MB limit and is excluded from git.
> Download it with the script below before running the recommendation notebooks.

---

## Downloading `ratings.csv`

**One command** from the repo root:

```bash
python 03_machine_learning/data/download_data.py
```

Or from inside the `data/` folder:

```bash
cd 03_machine_learning/data
python download_data.py
```

This downloads the **MovieLens 20M** dataset (~190 MB compressed, ~620 MB extracted)
from [GroupLens](https://grouplens.org/datasets/movielens/20m/),
extracts `ratings.csv` (and the other MovieLens files) into `training_data/`,
then deletes the zip automatically.

### Other useful flags

```bash
# See what's present / missing without downloading
python download_data.py --list

# Force re-download even if file already exists
python download_data.py --force

# Download a specific named dataset
python download_data.py --dataset movielens
```

---

## About MovieLens 20M

| Property | Value |
|----------|-------|
| Source | GroupLens Research, University of Minnesota |
| URL | https://grouplens.org/datasets/movielens/20m/ |
| Ratings | ~20 million |
| Movies | ~27,000 |
| Users | ~138,000 |
| Period | Jan 1995 – Mar 2015 |
| License | [GroupLens terms](https://files.grouplens.org/datasets/movielens/ml-20m-README.html) |

---

## Why is `ratings.csv` gitignored?

GitHub enforces a **100 MB hard limit** per file. At 591 MB, `ratings.csv`
cannot be pushed. Adding it to `.gitignore` is the cleanest solution for a
learning repo — the download script makes it trivially easy to get back.
