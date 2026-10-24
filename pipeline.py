import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET='final_score'
DEFAULT_FEATURES=['study_hours','attendance_pct','previous_score','sleep_hours','assignments_completed','extracurricular']

def make_demo_data(n_rows=700, random_state=42):
    """Generate synthetic demo records; they are not real student data."""
    rng=np.random.default_rng(random_state)
    study=rng.uniform(0,8,n_rows); attendance=rng.uniform(45,100,n_rows); previous=rng.uniform(25,100,n_rows)
    sleep=rng.uniform(4,10,n_rows); assignments=rng.integers(0,11,n_rows); extra=rng.choice(['Yes','No'],n_rows,p=[.42,.58])
    score=20+study*3.2+attendance*.22+previous*.25+sleep*.65+assignments*1.1+(extra=='Yes')*rng.uniform(0,3,n_rows)+rng.normal(0,8,n_rows)
    frame=pd.DataFrame({'study_hours':study.round(1),'attendance_pct':attendance.round(1),'previous_score':previous.round(1),'sleep_hours':sleep.round(1),'assignments_completed':assignments,'extracurricular':extra,TARGET:np.clip(score,0,100).round(1)})
    for col in ['study_hours','sleep_hours']:
        frame.loc[rng.choice(frame.index,size=max(1,n_rows//100),replace=False),col]=np.nan
    return frame

def split_xy(frame,target,feature_cols):
    if target not in frame.columns: raise ValueError(f"Target column '{target}' not found.")
    if not feature_cols: raise ValueError('Select at least one feature.')
    if target in feature_cols: raise ValueError('Target cannot also be a feature.')
    missing=[c for c in feature_cols if c not in frame.columns]
    if missing: raise ValueError(f'Missing feature columns: {missing}')
    clean=frame[feature_cols+[target]].copy(); clean[target]=pd.to_numeric(clean[target],errors='coerce'); clean=clean.dropna(subset=[target])
    if clean.empty: raise ValueError('No valid numeric target values found.')
    return clean,clean[feature_cols].copy(),clean[target].astype(float)

def build_pipeline(X):
    numeric=X.select_dtypes(include=['number']).columns.tolist(); categorical=[c for c in X.columns if c not in numeric]; transformers=[]
    if numeric: transformers.append(('numeric',Pipeline([('imputer',SimpleImputer(strategy='median'))]),numeric))
    if categorical: transformers.append(('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),categorical))
    return Pipeline([('preprocess',ColumnTransformer(transformers=transformers,remainder='drop')),('model',RandomForestRegressor(n_estimators=250,min_samples_leaf=2,random_state=42,n_jobs=-1))])

def evaluate_model(y_true,y_pred):
    return {'mae':float(mean_absolute_error(y_true,y_pred)),'rmse':float(np.sqrt(mean_squared_error(y_true,y_pred))),'r2':float(r2_score(y_true,y_pred))}
