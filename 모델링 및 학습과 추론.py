### 모델링 및 학습
import xgboost as xgb

try:
    model = xgb.XGBClassifier(tree_method = 'gpu_hist', gpu_id = 0, random_state=42)
    model.fit(X, y)
except Exception:
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X,y)

###### 추론
X_test.drop(columns=['ID'], inplace=True)
y_test_pred = model.predict(X_test)
y_test_pred_labels = le_target.inverse_transform(y_test_pred) # 인코딩 변수 역인코딩
test_data = test_df.copy()
test_data['pred_label']=y_test_pred_labels

##########################################################################################################################

# ✅ 필수 라이브러리 임포트
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import sqlite3
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import os

### Optuna study 저장 설정
study_name = 'xgb_label_v1'
db_dir = '/optuna'
os.makedirs(db_dir, exist_ok=True)
storage_name = f'sqlite:///{db_dir}/{study_name}.db'

#타겟 분리 및 데이터 분할
target_df = train_df['가격(백만원)']
train_df = train_df.drop(columns=['가격(백만원)'])
train_df, test_df, target_df, test_target_df = train_test_split(train_df, target_df, test_size = 0.2)

# 모델별 objective 함수정의(XGBoost)
def objectiveLR_xgb(trial: Trial, train_df, target_df, test_df, test_target_df):
  params = {
    'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
    'max_depth': trial.suggest_int('max_depth', 3, 30),
    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
  }
  model = xgb.XGBRegressor(objective='reg:squarederror', **params)
  model.fit(train_df, target_df)
  predictions = model.predict(test_df)
  rmse = np.sqrt(mean_squared_error(test_target_df, predictions))
  return rmse

# ✅ 모델별 objective 함수 정의 (Random Forest)
def objectiveLR_rf(trial: Trial, train_df, target_df, test_df, test_target_df):
  from sklearn.ensemble import RandomForestRegressor
  params = {
    'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
    'max_depth': trial.suggest_int('max_depth', 3, 30),
    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    'max_features': trial.suggest_float('max_features', 0.1, 1.0)
  }
  model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
  model.fit(train_df, target_df)
  predictions = model.predict(test_df)
  rmse = np.sqrt(mean_squared_error(test_target_df, predictions))
  return rmse

# ✅ 모델별 objective 함수 정의 (CatBoost)
def objectiveLR_catboost(trial: Trial, train_df, target_df, test_df, test_target_df):
  from catboost import CatBoostRegressor
  params = {
    'iterations': trial.suggest_int('iterations', 100, 1000),
    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
    'depth': trial.suggest_int('depth', 4, 15),
    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
    'border_count': trial.suggest_int('border_count', 32, 255),
    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0)
  }
  model = CatBoostRegressor(**params, verbose=False, random_state=42)
  model.fit(train_df, target_df)
  predictions = model.predict(test_df)
  rmse = np.sqrt(mean_squared_error(test_target_df, predictions))
  return rmse

# optuna 모델최적화 함수 정의
def optimize_model(model_name, objective_func, train_df, target_df, test_df, test_target_df):
    study_name = f'{model_name}_v1'
    storage_name = f'sqlite:///optuna/{study_name}.db'
    try:
        study = optuna.create_study(
            storage = storage_name,
            direction='minimize',
            sampler = TPESampler(multivariate=True, n_startup_trials=50, seed=42)
        )
        print(f'{model_name}: create new study')
    except:
        study = optuna.load_study(study_name = study_name, storage = storage_name)
        print(f'{model_name}:load existing study')
    
    study.optimize(lambda trial:objective_func(trial, train_df, target_df, test_df, test_target_df), n_trials=50)
    return study

#최적화된 모델로 예측 수행 함수
def get_optimized_predictions(model_name, study, train_df, target_df, test_df):
    if model_name == 'xgb':
        model = xgb.XGBRegressor(objective='reg:squarederror', **study.best_params)
    elif model_name == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
    else:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(**study.best_params, verbose=False, random_state=42)
        
    model.fit(train_df, target_df)
    return model.predict(test_df)

#모델 학습 및 최적화 
print("모델 최적화 시작")
study_xgb = optimize_model('xgb',objectiveLR_xgb, train_df, target_df, test_df, test_target_df)
study_rf = optimize_model('rf', objectiveLR_rf, train_df, target_df, test_df, test_target_df)
study_catboost = optimize_model('catboost', objectiveLR_catboost, train_df, target_df, test_df, test_target_df)

# 예측수행
pred_xgb = get_optimized_predictions('xgb', study_xgb, train_df, target_df, test_df)
pred_rf = get_optimized_predictions('rf', study_rf, train_df, target_df, test_df)
pred_catboost = get_optimized_predictions('catboost', study_catboost, train_df, target_df, test_df)

# 앙상블 예측
ensemble_mean = (pred_xgb + pred_rf + pred_catboost) / 3
weights = [
    1 / study_xgb.best_value,
    1 / study_rf.best_value,
    1 / study_catboost.best_value
]
weights = np.array(weights) / sum(weights)
ensemble_weighted = weights[0] * pred_xgb + weights[1] * pred_rf + weights[2] * pred_catboost

#모델 평가 결과 출력
from sklearn.metrics import r2_score
print('모델 성능 비교')
result = {
    'Model': ['XGBoost', 'RandomForest', 'CatBoost', 'Ensemble(Mean)', 'Ensemble(Weighted)'],
    'RMSE' : [
        np.sqrt(mean_squared_error(test_target_df, pred_xgb)),
        np.sqrt(mean_squared_error(test_target_df, pred_rf)),
        np.sqrt(mean_squared_error(test_target_df, pred_catboost)),
        np.sqrt(mean_squared_error(test_target_df, ensemble_mean)),
        np.sqrt(mean_squared_error(test_target_df, ensemble_weighted))
  ],
  'R2' : [
      r2_score(test_target_df, pred_xgb),
      r2_score(test_target_df, pred_rf),
      r2_score(test_target_df, pred_catboost),
      r2_score(test_target_df, ensemble_mean),
      r2_score(test_target_df, ensemble_weighted)
  ]
}
results_df = pd.DataFrame(result)
print(results_df.round(4))

#결과 시각화
import seaborn as sns
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
models = results_df['Model'].tolist()
rmse_values = results_df['RMSE'].values
r2_values = results_df['R2'].values
bars = ax1.bar(models, rmse_values, color='skyblue', alpha=0.7)
ax1.set_title('RMSE by Model')
ax1.set_ylabel('RMSE')
ax1.grid(True, alpha=0.3)

for bar in bars:
  ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom')
ax2.plot(models, r2_values, marker='o', linewidth=2, color='orange')
ax2.fill_between(models, r2_values, alpha=0.2, color='orange')
ax2.set_title('R² Score by Model')
ax2.set_ylabel('R² Score')
ax2.grid(True, alpha=0.3)

for i, r2 in enumerate(r2_values):
  ax2.text(i, r2, f'{r2:.4f}', ha='center', va='bottom')
plt.suptitle('Model Performance Comparison')
plt.tight_layout()
plt.show()

####### 단일 모델로 테스트 데이터 추론
pred_submit_rf = get_optimized_predictions('rf', study_rf, train_df, target_df, submit_test_df)

####### 앙상블 가중치로 테스트 데이터 추론
# 각 모델로 submit_test_df 예측
pred_submit_xgb = get_optimized_predictions('xgb', study_xgb, train_df, target_df, submit_test_df)
pred_submit_rf = get_optimized_predictions('rf', study_rf, train_df, target_df, submit_test_df)
pred_submit_catboost = get_optimized_predictions('catboost', study_catboost, train_df, target_df, submit_test_df)

# 가중치 계산
weights = [
    1 / study_xgb.best_value,  # XGBoost의 RMSE 역수
    1 / study_rf.best_value,   # RandomForest의 RMSE 역수
    1 / study_catboost.best_value  # CatBoost의 RMSE 역수
]
weights = np.array(weights) / sum(weights)  # 가중치 정규화

# 가중치 기반 앙상블
ensemble_weighted_submit = (
    weights[0] * pred_submit_xgb +
    weights[1] * pred_submit_rf +
    weights[2] * pred_submit_catboost
)

# 최종 예측 결과
y_pred = ensemble_weighted_submit
###############################################################################################################################