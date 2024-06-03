import pickle
import pandas as pd

# 피클 파일을 열어서 데이터 로드
with open('DFJSP-Qtime/src/master_db/pickleDBData/sks_train_1_db_data.pkl', 'rb') as file:
    data = pickle.load(file)

# 데이터가 pandas DataFrame 형식인 경우
if isinstance(data, pd.DataFrame):
    data.to_csv('data.csv', index=False)
else:
    # DataFrame이 아닌 경우, DataFrame으로 변환 필요
    # 예: data가 리스트의 리스트인 경우
    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)
a = 1
print("CSV 파일로 저장되었습니다.")