import csv
import random

import pandas as pd

df = pd.read_csv("/save_data/Parallel/sks_mac_st10.csv")
job_list = []
f_list = []
due_list = []
jobs = ['j01', 'j02', 'j03', 'j04', 'j05']
#for _ in range(10):
    #job_list.append(random.choice(jobs))
    #a = random.randint(10, 80)
    #f_list.append(a)
    #due_list.append(a+1)

#df['job_type'] = job_list
#df['job_id'] = job_list
#df['finish_time'] = f_list
df['due_date'] = df['finish_time'] +1
df.to_csv("sks_mac_st20.csv", index=False)