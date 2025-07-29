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

#####################################################################################################################

#########결측치 보간
import missingno as msno
msno.matrix(data)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(data["Value"])
plt.show()

# 데이터 길이, 결측치 길이
cnt_value = len(data["Value"])
cnt_missval = data["Value"].isna().sum()
print(cnt_value, cnt_missval)

from scipy import interpolate

# 6가지 보간법으로 함수를 생성
f_prev = interpolate.interp1d(lst_val, data.loc[lst_val,"Value"].values, kind='previous')
f_near = interpolate.interp1d(lst_val, data.loc[lst_val,"Value"].values, kind='nearest')
f_quad = interpolate.interp1d(lst_val, data.loc[lst_val,"Value"].values, kind='quadratic')

f_zero = interpolate.interp1d(lst_val, data.loc[lst_val,"Value"].values, kind='zero')

# 결측치 보간
y_prev = f_prev(lst_nan)
y_near = f_near(lst_nan)
y_quad = f_quad(lst_nan)

y_zero = f_zero(lst_nan)
y_slin = f_slin(lst_nan)
y_cubi = f_cubi(lst_nan)

# 보간 확인
ict_kind = {'previous':y_prev,'nearest':y_near,'quadratic':y_quad,
             'zero':y_zero,'slinear':y_slin,'cubic':y_cubi}
df_missval = pd.DataFrame(dict_kind, index=lst_nan)

print(df_missval.shape)
df_missval.head(3)

# 보간 시각적 확인
plt.figure(figsize=(4,2))
plt.plot(data["Value"])
plt.show()

plt.figure(figsize=(8,6))
plt.subplot(3,2,1)
plt.plot(data["Value"], c='r')
plt.plot(lst_nan, y_prev, '--', c='green', label='previous')
plt.legend()

plt.subplot(3,2,3)
plt.plot(data["Value"], c='r')
plt.plot(lst_nan, y_near, '--', c='green', label='nearest')
plt.legend()

plt.subplot(3,2,5)
plt.plot(data["Value"], c='r')
plt.plot(lst_nan, y_quad, '--', c='green', label='quadratic')
plt.legend()

plt.subplot(3,2,2)
plt.plot(data["Value"], c='r')
plt.plot(lst_nan, y_zero, '--', c='green', label='zero')
plt.legend()

plt.subplot(3,2,4)
plt.plot(data["Value"], c='r')
plt.plot(lst_nan, y_slin, '--', c='green', label='slinear')
plt.legend()

plt.subplot(3,2,6)
plt.plot(data["Value"], c='r')
plt.plot(lst_nan, y_cubi, '--', c='green', label='cubic')
plt.legend()

plt.show()


f_slin = interpolate.interp1d(lst_val, data.loc[lst_val,"Value"].values, kind='slinear')
f_cubi = interpolate.interp1d(lst_val, data.loc[lst_val,"Value"].values, kind='cubic')

# 앙상블
# interpolation 차이가 큰 결측구간은 제외함 :

col_tmp = [l for l in df_missval.columns if l not in ["quadratic","cubic"]]

for idx in df_missval.loc[:18707, col_tmp].mean(axis=1).index:
    df_missval.loc[idx, "mean"] = df_missval.loc[idx, col_tmp].mean()

for idx in df_missval.loc[18707:, df_missval.columns].mean(axis=1).index:
    df_missval.loc[idx, "mean"] = df_missval.loc[idx, df_missval.columns].mean()

df_missval.info()
print(df_missval.shape)
df_missval.head(3)

#평균 보간 시각화
plt.figure(figsize=(9,3))
plt.subplot(1,2,1)
plt.plot(data["Value"])

plt.subplot(1,2,2)
plt.plot(data["Value"], c='r')
plt.plot(lst_nan, df_missval['mean'].values, '--', c='orange', label='mean')
plt.legend()

plt.show()

for idx in df_missval.index:
    val = df_missval.loc[idx, "mean"]
    data.loc[idx, "Value(rev)"] = val

data.info()
print(data.shape)

plt.figure(figsize=(9,3))
plt.subplot(1,2,1)
plt.plot(data["Value"], label='Value')
plt.legend()

plt.subplot(1,2,2)
plt.plot(data["Value(rev)"], '--', c='orange', label='Value(rev)')
plt.legend()
plt.show()

###### 이상값 처리를 위한 박스플롯 확인
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(6, 4))
plt.boxplot(data['Value'].dropna())
plt.title('Value 박스플롯')
plt.ylabel('Value')
plt.show()

# 2. 그룹별 박스플롯
plt.figure(figsize=(10, 6))
data.boxplot(column='Value', by='Category')
plt.title('Category별 Value 분포')
plt.suptitle('')
plt.show()

# 3. Seaborn 박스플롯
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Category', y='Value')
plt.title('Category별 Value 분포')
plt.show()

####################
### z-score 은 표준편차 값, 일정 이상또는 이하의 표준편차는 이상치 간주
# 이상치 제거
import numpy as np

def outliers_z_score(ys):
    threshold=3
    mean_y=np.mean(ys)
    study_y = np.std(ys)
    z_scores=[(y-mean_y)/stdev_y for y in ys]
    return np.where(np.abs(z_scores)>threshold)

# 완전 중복 제거
def remove_duplicates(data):
    cleaned_data = data.drop_duplicates()
    return cleaned_data

# 특정 칼럼 기준 중복 제거
def remove_duplicates_subset(data, columns):
    cleaned_data = data.drop_duplicates(subset=columns)
    return cleaned_data

########## 표준화 ###############
# Z-score 표준화
z_score = (data-mean)/std_dev

print("z-score", z-score)

# z-score outlier +-3 시각화
outliers = np.where(np.abs(z_scores)>3)
print("outliers", data[outliers])

plt.figure(figsize=(10,6))
plt.plot(data,'bo-',label='original data')
plt.plot(outliers[0],data[outliers],'ro',label='outliers')
plt.axhline(mean, color='g', linestyle='--',label='Mean')
plt.xlabel('index')
plt.ylabel('Value')
plt.show()

# MinMax-Sclaer
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train, y_train)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns = X_train.columns)
X_train_scaled_df.describe()

# Robust Sclaer (IQR 사용)
from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X)
transformer.transformer(X)

###### 인코딩 #############
#원핫 인코딩
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
endcoded_data = encoder.fit_transform(categories.reshape(-1,1))

# 라벨 인코딩
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_data = encoder.fit_transform(data)

# 순서 인코딩 (순서가 중요한 범주형 데이터 에서 처리) 
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(categories = [categories] if categories else 'auto')
encoded_data = encoder.fit_transform(data.reshape(-1,1))

# 시간 특성 추출
df['year'] = df[date_column].dt.year
df['month'] = df[date_column].dt.month
df['day']=df[date_column].dt.day
df['hour'] = df[date_column].dt.hour
df['minute'] = df[date_column].dt.minute

#########차원축소###########
# PCA
from sklearn.decomposition import PCA

pca = PCA(n_componenets= 3)
X_transformed = pca.fit_transform(X)

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=3)
X_transformed = lad.fit_transform(X_train, y_train)

# t-SNE
from sklearn.manifold import TSNE

tsne = TSNE(n_components=3, perplexity =30,random_state=42)
X_embedded = tsne.fit_transform(X)

#####pivot table 생성########
pivot_table = pd.pivot_table(df, index = '행기준칼럼',
                            columns = '열기준칼럼',
                            values='집계할값칼럼',
                            aggfunc='mean',
                            fill_value=0)

#교차표 생성
cosstab = pd.crosstab(df['범주'],df['범주'])

##### melt 연산 코드#####
melted_df = pd.melt(df,
                    id_var=['고정컬럼1', '고정컬럼2'],
                    value_vars=['변환컬럼1', '변환컬럼2'],
                    var_name = '변수명',
                    value_name = '값')

#전체 칼럼 멜트
melted_all = pd.melt(df, id_vars=['ID'])

#스택 연산 - 칼럼을 행 인덱스로 변환
stacked = df.stack()

#언스택 연산 - 행 인덱스를 칼럼으로 변환
unstacked = df.unstack()

#특정 레벨 지원
stack_level = df.stack(level=0)
unstacked_level = df.unstack(level=-1)

####그룹별 집계####
grouped_mean = df.groupby('그룹칼럼')['값을 볼 칼럼'].mean()

#여러 통계량 동시 계산
grouped_agg = df.groupby(['그룹1,'그룹2'])['값을 볼 칼럼'].sum()

# 사용자 정의 함수 적용
custom_func = df.groupby('그룹칼럼')['값을 볼 칼럼'].apply(lambda x:x.max() - x.min())

################### 상관관계 계산 #########################
#  상관계수 행렬 계산
correlation_matrix = df.corr()
print(correlation_matrix)

#상관관계 히트맵 생성
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
plt.title('상관관계 히트맵')
plt.show()

################## 정규성 검정 ##########################
#shaprio-wilk 검정: 표본이 정규분포를 따르는지 검정(소표본에 적합)
shaprio_stat, shapiro_p = stats.shapiro(data)
print(f"Shapiro-Wilk: 통계량={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")

#kolmogorov-smirnov 검정: 표본분포와 이론적 분포 차이를 검정
ks_stat, ks_p = stats.kstest(data, 'norm', args=(data,mean(), data.std()))
print(f"Kolmogorov-Smirnov: 통계량={ks_stat:.4f}, p-value={ks_p:.4f}")

# anderson-darling 검정: 정규성 검정
ad_stat, ad_critical, ad_significance = stats.anderson(data, dist='norm')
print(f"Anderson-Darling: 통계량={ad_stat:.4f}")


print(f"임계값 (유의수준): {dict(zip(ad_significance, ad_critical))}")
