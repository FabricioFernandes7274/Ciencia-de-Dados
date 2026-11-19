import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_validate
from sklearn.preprocessing import OneHotEncoder

CSV = sys.argv[1] if len(sys.argv) > 1 else "fps_full.csv.gz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figs_full"
N_SAMPLE = 50_000  # amostra para as partes pesadas (correlação e modelo)
os.makedirs(OUT, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=130)
    plt.close()


df = pd.read_csv(CSV)
bad = (df["FPS"] <= 0).sum()
df = df[df["FPS"] > 0].copy()
df["logFPS"] = np.log(df["FPS"])
y = df["logFPS"].values

has_src = "Dataset" in df and df["Dataset"].nunique() > 1
cfg = ["CpuName", "GpuName", "GameName", "GameResolution", "GameSetting"]
factors = [c for c in cfg + ["Dataset"] if c in df]
specs = [c for c in df if c.startswith(("Cpu", "Gpu")) and c not in ("CpuName", "GpuName")]

# ---------- 1. Visão geral ----------
print("== VISÃO GERAL ==")
print("shape:", df.shape, "| linhas com FPS <= 0 removidas:", bad)
print(df.dtypes.astype(str).value_counts().to_string())
nun = df.nunique().sort_values()
print("colunas constantes ou vazias:", nun[nun <= 1].index.tolist())

# ---------- 2. Fontes ----------
if has_src:
    print("\n== FONTES (coluna Dataset) ==")
    print(df["Dataset"].value_counts().to_string())
    print(df.groupby("Dataset")["FPS"].describe().round(1).to_string())
    print(df.groupby("Dataset")[["CpuName", "GpuName", "GameName"]].nunique().to_string())
    print("resolução por fonte:")
    print(pd.crosstab(df["Dataset"], df["GameResolution"]).to_string())

# ---------- 3. Estrutura e repetição ----------
print("\n== ESTRUTURA ==")
n = {c: df[c].nunique() for c in cfg}
combos = int(np.prod(list(n.values())))
g = df.groupby(cfg)["logFPS"].agg(["size", "std"])
print(n)
print(f"combinações possíveis: {combos:,} | observadas: {len(g):,} ({len(g) / combos:.1%})")
rep = g[g["size"] > 1]
print(f"configurações com mais de uma medição: {len(rep):,} | linhas nelas: {rep['size'].sum() / len(df):.1%}")
print(f"ruído típico dentro da mesma configuração (desvio no log ≈ %): {rep['std'].median():.1%}")
print("medições por configuração:", g["size"].describe().round(1).to_dict())
rep["std"].hist(bins=60, figsize=(6, 3.5)); plt.title("Desvio do log(FPS) dentro da mesma configuração")
save("01_ruido")

print("\nlinhas por GPU:", df["GpuName"].value_counts().describe().round(0).to_dict())
print("linhas por CPU:", df["CpuName"].value_counts().describe().round(0).to_dict())
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
df["GpuName"].value_counts().hist(bins=40, ax=ax[0]); ax[0].set_title("Linhas por GPU")
df["CpuName"].value_counts().hist(bins=40, ax=ax[1]); ax[1].set_title("Linhas por CPU")
save("02_repeticao_hardware")

# ---------- 4. Ausentes ----------
print("\n== AUSENTES ==")
miss = df.isna().mean().sort_values(ascending=False)
print(miss[miss > 0].round(3).to_string())
miss[miss > 0].head(20).sort_values().plot.barh(figsize=(7, 5), title="Fração de ausentes (20 maiores)")
save("03_ausentes")

# ---------- 5. Alvo ----------
print("\n== ALVO (FPS) ==")
print(df["FPS"].describe().round(1).to_string())
print(f"assimetria: {df['FPS'].skew():.2f} | log: {df['logFPS'].skew():.2f}")
mult10 = (df["FPS"] % 10 == 0)
print(f"FPS múltiplo de 10: {mult10.mean():.1%}")
if has_src:
    print("por fonte:", mult10.groupby(df["Dataset"]).mean().round(3).to_dict())
q1, q3 = df["FPS"].quantile([.25, .75])
lim = q3 + 1.5 * (q3 - q1)
print(f"acima do limite IQR ({lim:.0f}): {(df['FPS'] > lim).sum():,}")
print(df.loc[df["FPS"] > lim, "GameName"].value_counts().head(5).to_string())
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
df["FPS"].clip(upper=df["FPS"].quantile(.999)).hist(bins=80, ax=ax[0]); ax[0].set_title("FPS (corte no percentil 99,9)")
df["logFPS"].hist(bins=80, ax=ax[1]); ax[1].set_title("log(FPS)")
save("04_alvo")
if has_src:
    for s_, d in df.groupby("Dataset"):
        d["FPS"].clip(upper=300).hist(bins=100, alpha=.5, label=str(s_), figsize=(8, 3.5))
    plt.legend(); plt.title("FPS por fonte (corte em 300)")
    save("05_fps_por_fonte")

# ---------- 6. O que explica o FPS ----------
print("\n== FATORES (R² do log FPS pelas médias do grupo) ==")
tot = ((y - y.mean()) ** 2).sum()
for c in factors:
    m = df.groupby(c)["logFPS"].transform("mean")
    print(f"{c}: {1 - ((y - m) ** 2).sum() / tot:.3f}")
Xf = OneHotEncoder(handle_unknown="ignore").fit_transform(df[factors].astype(str))
print(f"modelo aditivo com todos os fatores: {LinearRegression().fit(Xf, y).score(Xf, y):.3f}")
print("(sem grade balanceada, os R² individuais não precisam somar o total)")

top = df["GameName"].value_counts().head(25).index
df[df["GameName"].isin(top)].boxplot("FPS", by="GameName", figsize=(11, 4.5), rot=70)
plt.suptitle(""); plt.title("FPS por jogo (25 mais frequentes)"); save("06_jogos")

# ---------- 7. Correlação das specs com o FPS (amostra) ----------
print("\n== CORRELAÇÃO (Spearman, amostra) COM log(FPS) ==")
s = df.sample(min(N_SAMPLE, len(df)), random_state=0)
num = [c for c in specs if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
corr = s[num + ["logFPS"]].corr(method="spearman")
print(corr["logFPS"].drop("logFPS").sort_values(ascending=False).round(2).to_string())
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
fig.colorbar(im); save("07_correlacao")

# ---------- 8. Comparação entre fontes na mesma configuração ----------
if has_src:
    print("\n== MESMA CONFIGURAÇÃO EM DUAS FONTES ==")
    a, b = df["Dataset"].value_counts().index[:2]
    t = df.groupby(cfg + ["Dataset"])["logFPS"].mean().unstack("Dataset")[[a, b]].dropna()
    print(f"fontes: {a} x {b} | configurações em comum: {len(t):,}")
    if len(t):
        print(f"razão média de FPS ({a} / {b}): {np.exp((t[a] - t[b]).mean()):.2f} | correlação no log: {t[a].corr(t[b]):.3f}")

# ---------- 9. Risco de vazamento (teste rápido, amostra) ----------
print(f"\n== VAZAMENTO (amostra de {len(s):,} linhas) ==")
feats = [c for c in specs if nun[c] > 1]
cats = [c for c in ["GameName", "GameSetting", "GameResolution", "Dataset"] if c in df]
Xm = pd.get_dummies(s[feats + cats].astype({c: str for c in cats}), dtype=float)
ys = s["logFPS"]
model = HistGradientBoostingRegressor(random_state=0)
splits = {"aleatória": (KFold(5, shuffle=True, random_state=0), None)}
for name, col in [("por GPU", "GpuName"), ("por CPU", "CpuName"), ("por jogo", "GameName")]:
    splits[name] = (GroupKFold(5), s[col])
for name, (cv, groups) in splits.items():
    r = cross_validate(model, Xm, ys, cv=cv, groups=groups, scoring=("r2", "neg_mean_absolute_error"))
    mae = -r["test_neg_mean_absolute_error"].mean()
    print(f"{name:10s} R² = {r['test_r2'].mean():.3f} (±{r['test_r2'].std():.3f}) | erro médio ≈ {100 * (np.exp(mae) - 1):.1f}%")

if has_src:
    for src in s["Dataset"].value_counts().index[:2]:
        te = (s["Dataset"] == src).values
        keep = Xm.columns[Xm[~te].notna().any()]  # coluna toda vazia no treino quebra o modelo
        model.fit(Xm.loc[~te, keep], ys[~te])
        p = model.predict(Xm.loc[te, keep])
print(f"\nfiguras salvas em {OUT}/")