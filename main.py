from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import DEFAULT_FEATURES, TARGET, build_pipeline, evaluate_model, make_demo_data, split_xy

st.set_page_config(page_title='Student Insights Pro', page_icon='🎓', layout='wide')
st.markdown('''<style>
.block-container{padding-top:1.5rem;max-width:1400px}
[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 15% 0%,#152b4d 0,transparent 36%),radial-gradient(circle at 85% 5%,#251344 0,transparent 32%),#080d18}
[data-testid="stSidebar"]{background:#0d1424;border-right:1px solid #243653}
.hero{padding:1.7rem;border-radius:22px;border:1px solid #365a91;background:linear-gradient(115deg,#142c55,#221449);margin-bottom:1.2rem}
.hero h1{margin:0;font-size:clamp(2rem,4vw,3.2rem);letter-spacing:-.04em}.hero p{color:#bdcce8;margin:.5rem 0 0}
div[data-testid="stMetric"]{background:#111c30;border:1px solid #243653;padding:12px 16px;border-radius:14px}
</style>''', unsafe_allow_html=True)
if 'dataset' not in st.session_state: st.session_state.dataset = make_demo_data()
if 'model' not in st.session_state: st.session_state.model = None
if 'evaluation' not in st.session_state: st.session_state.evaluation = None
st.markdown('<div class="hero"><h1>🎓 Student Insights <span style="color:#62d5ff">Pro</span></h1><p>Data exploration · Machine learning · Model evaluation · Predictions</p></div>', unsafe_allow_html=True)
with st.sidebar:
    st.header('⚙️ Workspace')
    upload = st.file_uploader('Upload training CSV', type=['csv'])
    if upload:
        try:
            st.session_state.dataset = pd.read_csv(upload); st.success('CSV loaded')
        except Exception as e: st.error(f'Could not read CSV: {e}')
    df = st.session_state.dataset.copy()
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    target = st.selectbox('Prediction target', numeric, index=numeric.index(TARGET) if TARGET in numeric else 0) if numeric else None
    choices = [c for c in df.columns if c != target]
    defaults = [c for c in DEFAULT_FEATURES if c in choices] or choices[:6]
    features = st.multiselect('Features', choices, default=defaults)
    test_pct = st.slider('Test set (%)', 10, 40, 20, 5)
    seed = st.number_input('Random seed', 0, 99999, 42)
    train = st.button('🚀 Train / evaluate model', type='primary', use_container_width=True)
df = st.session_state.dataset.copy()
a,b,c,d = st.columns(4)
a.metric('Records', f'{len(df):,}'); b.metric('Columns', len(df.columns)); c.metric('Missing cells', int(df.isna().sum().sum())); d.metric('Duplicate rows', int(df.duplicated().sum()))
tabs = st.tabs(['📊 Explore data','🧠 Train & evaluate','🔮 Predict','📦 Export'])
with tabs[0]:
    l,r=st.columns([1.2,1]); l.markdown('#### Dataset preview'); l.dataframe(df.head(100),use_container_width=True,height=320)
    quality=pd.DataFrame({'column':df.columns,'dtype':[str(x) for x in df.dtypes],'missing':[int(x) for x in df.isna().sum()],'unique':[int(df[x].nunique()) for x in df.columns]})
    r.markdown('#### Data quality'); r.dataframe(quality,use_container_width=True,height=320)
    if target and pd.api.types.is_numeric_dtype(df[target]):
        fig=px.histogram(df,x=target,nbins=25,marginal='box',title=f'Distribution of {target}'); fig.update_layout(template='plotly_dark'); st.plotly_chart(fig,use_container_width=True)
    nums=df.select_dtypes(include=np.number)
    if len(nums.columns)>1:
        fig=px.imshow(nums.corr(),text_auto='.2f',aspect='auto',color_continuous_scale='Tealrose',title='Numeric feature correlation'); fig.update_layout(template='plotly_dark'); st.plotly_chart(fig,use_container_width=True)
with tabs[1]:
    st.write('Pipeline: numeric/categorical imputation, one-hot encoding, Random Forest regression, held-out evaluation.')
    if train:
        try:
            if not target or not features: raise ValueError('Choose a numeric target and at least one feature.')
            _,X,y=split_xy(df,target,features)
            from sklearn.model_selection import train_test_split
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_pct/100,random_state=int(seed))
            model=build_pipeline(Xtr); model.fit(Xtr,ytr); pred=model.predict(Xte); metrics=evaluate_model(yte,pred)
            st.session_state.model=model; st.session_state.evaluation={**metrics,'target':target,'features':features,'X_test':Xte,'y_test':yte,'preds':pred,'n_train':len(Xtr),'n_test':len(Xte)}
            st.success('Model trained successfully!')
        except Exception as e: st.error(f'Training failed: {e}')
    ev=st.session_state.evaluation
    if ev:
        x,y,z,w=st.columns(4); x.metric('MAE',f"{ev['mae']:.2f}"); y.metric('RMSE',f"{ev['rmse']:.2f}"); z.metric('R²',f"{ev['r2']:.3f}"); w.metric('Test rows',ev['n_test'])
        chart=pd.DataFrame({'Actual':ev['y_test'],'Predicted':ev['preds']})
        fig=px.scatter(chart,x='Actual',y='Predicted',trendline='ols',title='Actual vs predicted'); fig.update_layout(template='plotly_dark'); st.plotly_chart(fig,use_container_width=True)
        chart['Residual']=chart.Actual-chart.Predicted
        fig=px.histogram(chart,x='Residual',nbins=25,title='Residual distribution'); fig.update_layout(template='plotly_dark'); st.plotly_chart(fig,use_container_width=True)
        with st.expander('Show evaluation rows'): st.dataframe(chart,use_container_width=True)
    else: st.info('Choose target/features, then click Train / evaluate model.')
with tabs[2]:
    ev=st.session_state.evaluation; model=st.session_state.model
    if model is None or ev is None: st.info('Train a model first in the previous tab.')
    else:
        values={}; cols=st.columns(2)
        for i,col in enumerate(ev['features']):
            series=df[col]
            with cols[i%2]:
                if pd.api.types.is_numeric_dtype(series):
                    med=float(series.median()) if series.notna().any() else 0.0; lo=float(series.min()) if series.notna().any() else med-1; hi=float(series.max()) if series.notna().any() else med+1
                    if lo==hi: hi=lo+1
                    values[col]=st.number_input(col,value=min(max(med,lo),hi),min_value=lo,max_value=hi)
                else:
                    opts=series.dropna().astype(str).unique().tolist(); values[col]=st.selectbox(col,opts or ['Unknown'])
        if st.button('✨ Predict target',type='primary'):
            prediction=float(model.predict(pd.DataFrame([values],columns=ev['features']))[0]); st.success(f"Predicted **{ev['target']}**: **{prediction:.2f}**")
            st.caption('Prediction is an estimate, not a guarantee or a decision about a student.')
with tabs[3]:
    st.download_button('⬇️ Download current dataset CSV',df.to_csv(index=False).encode(), 'student_insights_dataset.csv','text/csv',use_container_width=True)
    ev=st.session_state.evaluation
    if ev:
        export=pd.DataFrame({'actual':ev['y_test'],'predicted':ev['preds']}); export['residual']=export.actual-export.predicted
        st.download_button('⬇️ Download evaluation predictions',export.to_csv(index=False).encode(),'evaluation_predictions.csv','text/csv',use_container_width=True)
st.markdown('---')
st.caption('Built with Python · Streamlit · Pandas · scikit-learn · Plotly | Demo data is synthetic; validate models before real-world use.')
