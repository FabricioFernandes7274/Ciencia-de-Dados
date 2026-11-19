import openml

ds = openml.datasets.get_dataset(42737)
df = ds.get_data()[0]
print(df.shape)
df.to_csv("fps_full.csv.gz", index=False)