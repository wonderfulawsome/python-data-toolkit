#### ✅ Train
info_categories = ["customer", "credit", "sales", "billing", "balance", "channel", "marketing", "performance"]

train_dfs = {}
for prefix in info_categories:
    df_list = [globals()[f"{prefix}_train_{month}"] for month in months]
    train_dfs[f"{prefix}_train_df"] = pd.concat(df_list, axis=0)
    gc.collect()
    print(f"{prefix}_train_df is created with shape: {train_dfs[f'{prefix}_train_df'].shape}")

######## 변수 분리
test_dfs = {}
for prefix in info_categories:
    df_list = [globals()[f"{prefix}_test_{month}"] for month in months]
    test_dfs[f"{prefix}_test_df"] = pd.concat(df_list, axis=0)
    gc.collect()
    print(f"{prefix}_test_df is created with shape: {test_dfs[f'{prefix}_test_df'].shape}")

######## 데이터 병합
import gc

temp = train_dfs
train_df = temp["customer_train_df"].merge(temp["credit_train_df"], on=["기준년월", "ID"], how="left")
print("Step1 저장 완료: train_step1, shape:", train_df.shape)
del temp["customer_train_df"], temp["credit_train_df"]
gc.collect()

merge_list = [
    ("sales_train_df",    "Step2"),
    ("billing_train_df",  "Step3"),
    ("balance_train_df",  "Step4"),
    ("channel_train_df",  "Step5"),
    ("marketing_train_df","Step6"),
    ("performance_train_df", "최종")
]

for df_name, step in merge_list:
    train_df = train_df.merge(temp[df_name], on=["기준년월", "ID"], how="left")
    print(f"{step} 저장 완료: train_{step}, shape:", train_df.shape)
    del temp[df_name]
    gc.collect()

######### 데이터 인코딩
feature_cols = [col for col in train_df.columns if col not  in['ID','Segment']]
X=train_df[feature_cols].copy()
y= train_df['Segment'].copy()

from sklearn.preprocessing import LabelEncoder

le_target = LabelEncoder
y_encoded = le_target.fit_transform(y)

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
X_test = test_df.copy()

encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le
    unseen = set(X_test[col]) - set(le.clasees_)
    if unseen:
        le.classes_ = np.append(le.classes_, list(unseen))
    X_test[col] = le.transform(X_test[col])

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

# 주문시간에서 월정보 추출
##날짜 변환
orders['Order_purchase_timestamp'] = pd.to_datetime(orders['Order_purchase_timestamp'])

#월 추출
orders['month'] = orders['Order_purchase_timestamp'].dt.to_period('M')

#월별 매출 집계
montly_df = pd.merge(orders, order_item, on = 'order_id')

# 매출 시각화
plt.figure(figsize=(10,6))

################################ 데이터 전처리
train_df = pd.read_csv('data_train.csv')
test_df = pd.read_csv('data_test.csv')

missing_train_data = [col for col in train_df.columns if train_df[col].isna().any()]
missing_test_data  = [col for col in test_df.columns  if test_df[col].isna().any()]

from scipy import interpolate
import pandas as pd

def ensemble_interpolation(df, missing_cols):
    df = df.copy().reset_index(drop=True)

    for col in missing_cols:
        series = df[col]
        total_len = len(series)

        lst_nan = series[series.isna()].index.tolist()
        lst_val = series[series.notna()].index.tolist()

        # ① 비결측값이 하나도 없으면 건너뛰기
        if len(lst_val) == 0:
            continue

        # ② 사용할 보간 기법 선택
        methods = ['previous', 'nearest', 'zero', 'slinear']
        if len(lst_val) >= 3:
            methods += ['quadratic', 'cubic']

        # ③ 함수 만들기
        funcs = {}
        for kind in methods:
            funcs[kind] = interpolate.interp1d(
                lst_val, series[lst_val],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True
            )

        # ④ 예측값 생성
        df_interp = pd.DataFrame(
            {kind: func(lst_nan) for kind, func in funcs.items()},
            index=lst_nan
        )

        # ⑤ mean 컬럼 생성 (앞/뒤 구간 구분 없이 사용 가능)
        df_interp['mean'] = df_interp.mean(axis=1)

        # ⑥ 반영
        df.loc[df_interp.index, col] = df_interp['mean']

    return df

# 보간 함수 호출
train_df = ensemble_interpolation(train_df, missing_train_data)
test_df  = ensemble_interpolation(test_df, missing_test_data)

#####################################################################################################################
#####################################################################################################################

# 데이터에서 텍스트 추출
df_soil['동'] = df_soil['소재지'].str.extract(r'(\S+동)')

#  텍스트에서 괄호 없애기 
df_soil['동'] = df_soil['동'].str.replace(r'[\(\)]', '', regex=True).str.strip()

# 텍스트 데이터에서 NaN, '동', 'B동' 등을 '기타'로 대체
df_soil['동'] = df_soil['동'].fillna('기타')  # NaN을 '기타'로 대체
df_soil['동'] = df_soil['동'].replace(['동', 'B동', 'A동', 'C동'], '기타') 


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

#사용자 정의 함수 적용
custom_func = df.groupby('그룹칼럼')['값을 볼 칼럼'].apply(lambda x:x.max() - x.min())
