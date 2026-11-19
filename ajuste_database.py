import pandas as pd

df = pd.read_csv("fps_full.csv.gz")
df = df[df["FPS"] > 0]
print(df["GameSetting"].value_counts())
print(pd.crosstab(df["GameName"], df["Dataset"]))
print("FPS == 10:", (df["FPS"] == 10).mean())
print(df["GpuArchitecture"].value_counts().head(10))
cfg = ["CpuName", "GpuName", "GameName", "GameResolution", "GameSetting", "Dataset"]
print("linhas após agregar:", df.groupby(cfg).ngroups)