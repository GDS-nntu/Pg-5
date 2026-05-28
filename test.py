import time
from github_api import fetch_user_data, load_token
from data_cleaning import clean_user_data
from data_processing import make_ref_stats_synthetic
from analytics import compute_gds
from ml_model import predict as ml_predict

token = load_token()
username = "GDS-nntu"

weights = {
    "indices":  {"activity": 0.30, "quality": 0.30, "social": 0.20, "tech": 0.20},
    "activity": {"commits": 0.35, "prs": 0.30, "issues": 0.20, "pr_reviews": 0.15},
    "quality":  {"stars": 0.35, "forks": 0.25, "repo_count": 0.15, "closed_issues": 0.15, "total_issues": 0.10},
    "social":   {"followers": 0.60, "contributed_to": 0.40},
    "tech":     {"diversity": 0.50, "relevance": 0.50},
}

# 1. API запрос
t = time.time()
raw = fetch_user_data(username, token)
print(f"API запрос:        {time.time()-t:.3f} с")

# 2. Очистка
t = time.time()
data = clean_user_data(raw)
print(f"Очистка:           {time.time()-t:.6f} с")

# 3. Эталонная статистика
t = time.time()
ref = make_ref_stats_synthetic()
print(f"Эталон:            {time.time()-t:.6f} с")
from data_processing import normalize_metric

t = time.time()
for key in ref:
    normalize_metric(data.get(key, 0), ref[key]["mean"], ref[key]["std"])
print(f"Нормализация:      {time.time()-t:.6f} с")
# 4. GDS
t = time.time()
scores = compute_gds(data, ref, weights)
print(f"GDS:               {time.time()-t:.6f} с")

# 5. ML
t = time.time()
ml = ml_predict(data)
print(f"ML предсказание:   {time.time()-t:.3f} с")

print(f"\nРезультат: GDS={scores['gds']}, ML={ml['level']}")