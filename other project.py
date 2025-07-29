####################데이터 전처리 코드 - mActivity#########################
######## https://dacon.io/competitions/official/236468/codeshare/12355?page=1&dtype=recent 
# 데이터 원-핫 인코딩
""" 활동 코드를 원-핫 인코딩하여 각 활동 유형별 칼럼 생성"""
df_in_mActivity = pd.merge(df_in_mActivity, pd.get_dummies(df_in_mActivity, columns=["m_activity"],prefix="m_activity",dtype=int),
                        how='left',
                        on=['subject_id','timestamp'])
#MET값 매핑
"""
각 활동 코드에 해당하는 MET(Metabolic Equivalent of Task) 값 할당
MET는 신체 활동의 에너지 소비량을 측정하는 단위

활동 코드별 MET 값:
    0: 1.3 MET (가벼운 좌식 활동)
    1: 8.0 MET (격렬한 활동)
    3: 1.2 MET (매우 가벼운 활동)
    4: 3.0 MET (중간 강도 활동)
    7: 3.5 MET (중간 강도 활동)
    8: 10.0 MET (매우 격렬한 활동)
"""
dict_met_value = {0: 1.3, 1: 8.0, 3: 1.2, 4: 3.0, 7: 3.5, 8: 10.0}
for activity, met in dict_met_value.items():
  df_in_mActivity.loc[df_in_mActivity["m_activity"].isin([activity]),"m_activity_met"]=met

df_in_mActivity.head()

# 데이터 집계 함수 정의
def fn_love_aespa(
  df_input: pd.DataFrame,
  str_value_col: str
  str_agg_func:str="mean",
  str_freq:str="30min",)
-> pd.DataFrame:
df_input_copy = df_input.copy()
df_input_copy["timestamp"]=pd.to_datetime(df_input_copy["timestamp"])

