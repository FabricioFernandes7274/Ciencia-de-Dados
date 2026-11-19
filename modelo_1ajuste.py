import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV = sys.argv[1] if len(sys.argv) > 1 else "fps_modelo.csv"
N_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # 0 usa todas as linhas (mais lento)

# ---------- ajustes ----------
PESO = "none"                                      # "none", "sqrt" ou "n"
MIN_N = 7                                          # configurações "confiáveis": pelo menos MIN_N medições
ETAPAS = ["ablacao", "log", "ajuste", "erros"]     # tire as que não quiser rodar
GRADE = [dict(learning_rate=lr, max_leaf_nodes=nl) for lr in (0.05, 0.1) for nl in (15, 31, 63)]

df = pd.read_csv(CSV)
if N_MAX and len(df) > N_MAX:
    df = df.sample(N_MAX, random_state=0)
df = df.reset_index(drop=True)
y = df["logFPS"].values
w = {"none": np.ones(len(df)), "sqrt": np.sqrt(df["n"]), "n": df["n"].astype(float)}[PESO]
conf = (df["n"] >= MIN_N).values

# grupos de features
ctx = ["GameName", "GameResolution", "GameSetting"]
gpu = [c for c in df if c.startswith("Gpu") and c != "GpuName"]
cpu = [c for c in df if c.startswith("Cpu") and c != "CpuName"]
numericas = set(df.select_dtypes("number"))
todas = ctx + gpu + cpu
X = df[todas]
print(f"linhas: {len(df):,} | configurações com n >= {MIN_N}: {conf.sum():,} | peso: {PESO}")
print(f"features: contexto {len(ctx)}, GPU {len(gpu)}, CPU {len(cpu)}")

cortes = {
    "por GPU": (GroupKFold(5), df["GpuName"]),
    "por CPU": (GroupKFold(5), df["CpuName"]),
    "aleatória": (KFold(5, shuffle=True, random_state=0), None),
}


def separa(cols):
    return [c for c in cols if c in numericas], [c for c in cols if c not in numericas]


def prep(num, cat, escala=False):
    t = []
    if num:
        imp = SimpleImputer(strategy="median")
        t.append(("n", make_pipeline(imp, StandardScaler()) if escala else imp, num))
    if cat:
        t.append(("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(t)


def gb(cols, **kw):
    return make_pipeline(prep(*separa(cols)), HistGradientBoostingRegressor(random_state=0, **kw))


def ridge(cols):
    return make_pipeline(prep(*separa(cols), escala=True), Ridge(alpha=10))


def oof(modelo, dados, corte):
    cv, groups = corte
    p = np.zeros(len(df))
    for tr, te in cv.split(dados, y, groups):
        m = clone(modelo)
        m.fit(dados.iloc[tr], y[tr], **{f"{m.steps[-1][0]}__sample_weight": w[tr]})
        p[te] = m.predict(dados.iloc[te])
    return p


def metricas(p, mask):
    return r2_score(y[mask], p[mask]), 100 * (np.exp(mean_absolute_error(y[mask], p[mask])) - 1)


def linha(nome, p, t0=None):
    a, b = metricas(p, np.ones(len(df), bool)), metricas(p, conf)
    seg = f"{time.time() - t0:5.0f}s" if t0 else ""
    print(f"{nome:42s} R² {a[0]:6.3f} erro {a[1]:5.1f}% | n>={MIN_N}: R² {b[0]:6.3f} erro {b[1]:5.1f}% {seg}")
    return a[0]


# ---------- 1. Ablação por grupo de features ----------
if "ablacao" in ETAPAS:
    print("\n== 1. ABLAÇÃO (Gradient Boosting, divisão por GPU) ==")
    grupos = {"só contexto (jogo, resolução, config.)": ctx, "contexto + CPU": ctx + cpu,
              "contexto + GPU": ctx + gpu, "contexto + GPU + CPU": todas}
    for nome, cols in grupos.items():
        t0 = time.time()
        linha(nome, oof(gb(cols), df[cols], cortes["por GPU"]), t0)

# ---------- 2. Log nas features (só muda o Ridge; árvores não mudam) ----------
if "log" in ETAPAS:
    print("\n== 2. LOG NAS FEATURES (Ridge, divisão por GPU) ==")
    specs = [c for c in gpu + cpu if c in numericas]
    skew = df[specs].skew()
    cand = [c for c in specs if df[c].min() >= 0 and skew[c] > 1]
    print("colunas com log1p:", cand)
    X_log = X.copy()
    X_log[cand] = np.log1p(X_log[cand])
    linha("Ridge", oof(ridge(todas), X, cortes["por GPU"]))
    linha("Ridge com log nas features", oof(ridge(todas), X_log, cortes["por GPU"]))

# ---------- 3. Ajuste do Gradient Boosting ----------
MELHOR = {}
if "ajuste" in ETAPAS:
    print("\n== 3. AJUSTE DO GRADIENT BOOSTING (divisão por GPU) ==")
    base = dict(max_iter=500, early_stopping=True, n_iter_no_change=20)
    linha("padrão do scikit-learn", oof(gb(todas), X, cortes["por GPU"]))
    melhor_r2 = -np.inf
    for kw in GRADE:
        t0 = time.time()
        r2 = linha(str(kw), oof(gb(todas, **base, **kw), X, cortes["por GPU"]), t0)
        if r2 > melhor_r2:
            melhor_r2, MELHOR = r2, {**base, **kw}
    print("\nmelhor:", MELHOR)
    print("(o ajuste usou a divisão por GPU; a confirmação abaixo usa outras divisões)")
    for div in ("por CPU", "aleatória"):
        linha(f"{div}: padrão", oof(gb(todas), X, cortes[div]))
        linha(f"{div}: ajustado", oof(gb(todas, **MELHOR), X, cortes[div]))

# ---------- 4. Análise dos erros ----------
if "erros" in ETAPAS:
    print("\n== 4. ONDE O MODELO ERRA (Gradient Boosting, por GPU) ==")
    p = oof(gb(todas, **MELHOR), X, cortes["por GPU"])
    d = df[["CpuName", "GpuName", "GameName", "GameResolution", "GameSetting", "n"]].copy()
    d["GpuArchitecture"] = df["GpuArchitecture"] if "GpuArchitecture" in df else "?"
    d["faixa_n"] = pd.cut(df["n"], [0, 1, 2, 5, 20, np.inf], labels=["1", "2", "3-5", "6-20", ">20"])
    d["e"] = p - y
    d["ae"] = d["e"].abs()

    def tabela(col, min_linhas=1, top=None):
        g = d.groupby(col, observed=True).agg(linhas=("e", "size"), vies=("e", "mean"), ae=("ae", "mean"))
        g = g[g["linhas"] >= min_linhas]
        g = pd.DataFrame({"linhas": g["linhas"], "viés%": 100 * (np.exp(g["vies"]) - 1),
                          "erro%": 100 * (np.exp(g["ae"]) - 1)}).sort_values("erro%", ascending=False)
        print(f"\n-- por {col} (viés + = o modelo prevê FPS a mais) --")
        print(g.head(top).round(1).to_string())

    tabela("GameName")
    tabela("faixa_n")
    tabela("GameSetting")
    tabela("GameResolution")
    tabela("GpuArchitecture", min_linhas=50)
    tabela("GpuName", min_linhas=100, top=10)
    tabela("CpuName", min_linhas=100, top=10)
    d.assign(y=y, previsto=p).to_csv("erros_por_linha.csv", index=False)
    print("\nsalvo: erros_por_linha.csv")