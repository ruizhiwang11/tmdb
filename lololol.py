import pandas as pd

df = pd.read_csv("label.csv")

print(df["isThriller"].value_counts())