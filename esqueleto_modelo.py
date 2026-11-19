import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV = sys.argv[1] if len(sys.argv) > 1 else "fps_modelo.csv"
N_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # 0 usa todas as linhas (mais lento)

# ---------- ajustes para ir testando ----------
USE_FONTE = False                       # usar a coluna fonte como feature
PESO = "none"                           # peso de cada configuração no treino: "none", "sqrt" ou "n"
DIVISOES = ["por GPU", "aleatória", "por CPU", "por jogo"]     # também: "por CPU", "por jogo"
MODELOS = ["média global", "só o jogo", "Ridge", "Random Forest", "Gradient Boosting"]
MIN_N = 6                               # "medições confiáveis": configurações com pelo menos MIN_N medições

df = pd.read_csv(CSV)
if N_MAX and len(df) > N_MAX:
    df = df.sample(N_MAX, random_state=0)
df = df.reset_index(drop=True)
y = df["logFPS"].values
w = {"none": np.ones(len(df)), "sqrt": np.sqrt(df["n"]), "n": df["n"].astype(float)}[PESO]

ids = ["CpuName", "GpuName", "GameName", "fonte"]
cat = [c for c in ["GameName", "GpuArchitecture", "GpuMemoryType"] + (["fonte"] if USE_FONTE else []) if c in df]
num = [c for c in df.select_dtypes("number") if c not in ("n", "logFPS")]
X = df[num + cat]
print(f"linhas: {len(df):,} | features numéricas: {len(num)} | categóricas: {cat} | peso: {PESO}")

lin = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                         ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
tree = ColumnTransformer([("n", SimpleImputer(strategy="median"), num),
                          ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
so_jogo = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), ["GameName"])])

modelos = {
    "média global": make_pipeline(DummyRegressor()),
    "só o jogo": make_pipeline(so_jogo, LinearRegression()),
    "Ridge": make_pipeline(lin, Ridge(alpha=10)),
    "Random Forest": make_pipeline(tree, RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                                               n_jobs=-1, random_state=0)),
    "Gradient Boosting": make_pipeline(tree, HistGradientBoostingRegressor(random_state=0)),
}

cortes = {
    "por GPU": (GroupKFold(5), df["GpuName"]),
    "por CPU": (GroupKFold(5), df["CpuName"]),
    "por jogo": (GroupKFold(5), df["GameName"]),
    "aleatória": (KFold(5, shuffle=True, random_state=0), None),
}


def metricas(p, mask):
    if mask.sum() < 2:
        return np.nan, np.nan
    return r2_score(y[mask], p[mask]), 100 * (np.exp(mean_absolute_error(y[mask], p[mask])) - 1)


todas = np.ones(len(df), bool)
conf = (df["n"] >= MIN_N).values
print(f"configurações com n >= {MIN_N}: {conf.sum():,}\n")
print(f"{'divisão':10s} {'modelo':18s} {'R²':>6s} {'erro%':>6s} | {'R² n>=' + str(MIN_N):>8s} {'erro%':>6s} | {'s':>5s}")

preds = df[ids + ["n"]].copy()
preds["y"] = y
for div in DIVISOES:
    cv, groups = cortes[div]
    for nome in MODELOS:
        t0 = time.time()
        p = np.zeros(len(df))
        for tr, te in cv.split(X, y, groups):
            m = clone(modelos[nome])
            m.fit(X.iloc[tr], y[tr], **{f"{m.steps[-1][0]}__sample_weight": w[tr]})
            p[te] = m.predict(X.iloc[te])
        r2a, ea = metricas(p, todas)
        r2c, ec = metricas(p, conf)
        print(f"{div:10s} {nome:18s} {r2a:6.3f} {ea:6.1f} | {r2c:8.3f} {ec:6.1f} | {time.time() - t0:5.0f}")
        preds[f"{div}|{nome}"] = p

preds.to_csv("predicoes_oof.csv", index=False)
print("\npredições fora da amostra salvas em predicoes_oof.csv (para analisar os erros)")