import csv
import random

import pandas as pd

df = pd.read_csv("/save_data/Parallel/sks_rd10.csv")

jobs = ['j01', 'j02', 'j03', 'j04', 'j05']
job_list = []
#for index, df2 in df.iterrows():
    #job_list.append(random.choice(jobs))

#df['item'] = job_list
df['d_time'] = df['r_time'] + 336

df.to_csv("sks_rd20.csv", index=False)