import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/20191122_final_train.csv')


df_train, df_test = train_test_split(df, test_size=10000, shuffle=True)

df_train.to_csv('data/train.csv')
df_test.to_csv('data/test.csv')

