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
#####################################################################################################################
