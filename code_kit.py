### 텍스트 데이터 전처리
# 소재지에서 동 정보 추출
df_soil['동'] = df_soil['소재지'].str.extract(r'(\S+동)')
df_soil['동'] = df_soil['동'].str.replace(r'[\(\)]', '', regex=True).str.strip()

# 동 이름 앞의 숫자 제거
df_soil['동'] = df_soil['동'].str.replace(r'^\d+', '', regex=True).str.strip()

# NaN, '동', 'B동' 등을 '기타'로 대체
df_soil['동'] = df_soil['동'].fillna('기타')  # NaN을 '기타'로 대체
df_soil['동'] = df_soil['동'].replace(['동', 'B동', 'A동', 'C동'], '기타') 

# 동 별로 그룹화 해서 동의 빈도수 계산하기
dong_counts = df_soil.groupby('동')['동'].transform('count')

# 각 동별로 그룹화하여 무게 합계 계산
dong_weight_stats = df_food.groupby('동')['무게'].agg(['sum', 'mean', 'count']).round(2)

# outer join 병합
merged_dong_data = pd.merge(
    dong_weight_stats.reset_index(), 
    address_dong_counts, 
    on='동', 
    how='outer'
).fillna(0)

# 종류별 개수 집계
purpose_counts = df_river['종류'].value_counts()

# 행 값 변경
df_river['점용목적'] = df_river['점용목적'].replace(
    '스케이트장',
    '기타'
)

# 특정 칼럼값의 다른 칼럼값 확인
df_river[df_river['점용목적'] == '공업용수']['부과대상 소재지주소']

# 각 칼럼값에 새로운 칼럼값 갱신
final_merged_data.loc[final_merged_data['동'] == '이동', '하천 점용 개수'] = 3
final_merged_data.loc[final_merged_data['동'] == '안산동', '하천 점용 개수'] = 1
final_merged_data.loc[final_merged_data['동'] == '반월동', '하천 점용 개수'] = 1

# 각 칼럼값에 새로운 값 할당하기
merged_data.loc[merged_data['동'] == '초지동', '하수처리장'] = 2
merged_data.loc[merged_data['동'] == '대부동', '하수처리장'] = 1

# 모든 읍면동의 오염도가 0인 오염도 관련 칼럼 삭제
pollution_avg_by_dong_filtered = pollution_avg_by_dong.loc[:, (pollution_avg_by_dong != 0).any(axis=0)]

# 표준화
X = df[numeric_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 월별 집계
wongok_monthly = wongok_data.groupby(wongok_data['수거일자'].dt.to_period('M'))['무게'].sum().reset_index()
wongok_monthly['날짜'] = wongok_monthly['수거일자'].dt.to_timestamp()

