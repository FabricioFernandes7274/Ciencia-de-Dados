import sys

import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "fps_full.csv.gz"
DST = sys.argv[2] if len(sys.argv) > 2 else "fps_modelo.csv"

df = pd.read_csv(SRC)
df = df[df["FPS"] > 0].copy()            # FPS <= 0 é inválido
df["logFPS"] = np.log(df["FPS"])
print("linhas após remover FPS <= 0:", len(df))

# colunas descartadas: muito vazias, redundantes ou sem valor para o modelo
DROP = ["GpuNumberOfExecutionUnits", "GpuNumberOfComputeUnits",  # 97% e 84% vazias
        "CpuNumberOfTransistors", "CpuDieSize",                  # ~50% vazias
        "CpuMultiplier"]                                         # redundante com CpuFrequency
specs = [c for c in df
         if c.startswith(("Cpu", "Gpu")) and c not in ["CpuName", "GpuName"] + DROP
         and not c.startswith("GpuBus")]

# uma linha por configuração: a média do log(FPS) reduz o ruído das medições repetidas
cfg = ["CpuName", "GpuName", "GameName", "GameResolution", "GameSetting", "Dataset"]
g = df.groupby(cfg, as_index=False, dropna=False).agg(
    logFPS=("logFPS", "mean"),
    n=("logFPS", "size"),
    **{c: (c, "first") for c in specs},
)
print(f"configurações agregadas: {len(g):,} (a partir de {len(df):,} medições)")

# configuração gráfica em ordem; versões de API viram número
g["GameSetting"] = g["GameSetting"].map({"low": 0, "med": 1, "high": 2, "max": 3})
print("níveis de configuração sem mapa:", g["GameSetting"].isna().sum())
for c in ["GpuDirectX", "GpuVulkan"]:
    g[c] = pd.to_numeric(g[c].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")

g = g.rename(columns={"Dataset": "fonte"})
ids = ["CpuName", "GpuName", "GameName", "fonte"]   # ids servem para dividir por grupo, não são features
g = g[ids + ["GameResolution", "GameSetting", "n", "logFPS"] + specs]

print("\ncolunas:", len(g.columns), "| texto:", g.select_dtypes(exclude="number").columns.tolist())
miss = g.isna().mean().sort_values(ascending=False)
print("ausentes após a limpeza:\n", miss[miss > 0].round(3).to_string())
print("\nmedições por configuração:", g["n"].describe().round(1).to_dict())
print("\nconfigurações por fonte:", g["fonte"].value_counts().to_dict())
print("jogos em cada fonte:", g.groupby("fonte")["GameName"].nunique().to_dict())

g.to_csv(DST, index=False)
print(f"\nsalvo em {DST}: {g.shape}")