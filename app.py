
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Universal Bank - Trainable Dashboard", layout="wide")
st.title("Universal Bank — Marketing Dashboard (Train in-app, 5-fold CV)")

tabs = st.tabs(["Overview (Insights)", "Train & Evaluate Models", "Predict (Upload & Score)", "Data & Notes"])

# --- helpers ---
def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error("Error reading CSV: " + str(e))
        return None

def preprocess_for_model(df):
    # Drop ID/Zip-like columns if present and ensure target exists
    df = df.copy()
    drop_cols = [c for c in df.columns if c.strip().lower() in ['id','zip','zip code','zipcode']]
    df = df.drop(columns=drop_cols, errors='ignore')
    target_candidates = [c for c in df.columns if c.strip().lower().replace(' ','') in ['personalloan','personal_loan']]
    if not target_candidates:
        raise ValueError("Could not find 'Personal Loan' target column. Ensure CSV includes 'Personal Loan' column.")
    target = target_candidates[0]
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y

def train_all_models(X, y, random_state=42):
    # scale continuous features
    cont = [c for c in X.columns if c.lower() in ['age','experience','income','ccavg','mortgage']]
    scaler = StandardScaler()
    Xs = X.copy()
    if cont:
        Xs[cont] = scaler.fit_transform(X[cont])
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.30, stratify=y, random_state=random_state)
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=random_state)
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:,1] if hasattr(model,'predict_proba') else y_test_pred
        results[name] = {
            'model': model,
            'cv_mean': float(np.round(cv_scores.mean(),4)),
            'cv_std': float(np.round(cv_scores.std(),4)),
            'train_acc': float(np.round(accuracy_score(y_train, y_train_pred),4)),
            'test_acc': float(np.round(accuracy_score(y_test, y_test_pred),4)),
            'precision': float(np.round(precision_score(y_test, y_test_pred, zero_division=0),4)),
            'recall': float(np.round(recall_score(y_test, y_test_pred, zero_division=0),4)),
            'f1': float(np.round(f1_score(y_test, y_test_pred, zero_division=0),4)),
            'auc': float(np.round(roc_auc_score(y_test, y_test_proba),4)),
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba,
            'feature_names': X.columns.tolist()
        }
    return results, scaler, X_train, X_test, y_train, y_test

def metrics_df(results):
    rows = []
    for k,v in results.items():
        rows.append({
            'Algorithm': k,
            'CV Acc (mean)': v['cv_mean'],
            'CV Acc (std)': v['cv_std'],
            'Train Acc': v['train_acc'],
            'Test Acc': v['test_acc'],
            'Precision': v['precision'],
            'Recall': v['recall'],
            'F1': v['f1'],
            'AUC': v['auc']
        })
    return pd.DataFrame(rows).set_index('Algorithm')

def plot_roc_overlay(results):
    fig = go.Figure()
    for name,v in results.items():
        fpr, tpr, _ = roc_curve(v['y_test'], v['y_test_proba'])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={v['auc']})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), showlegend=False))
    fig.update_layout(title='ROC Curves (overlay)', xaxis_title='FPR', yaxis_title='TPR', width=800, height=500)
    return fig

def plot_confusion(cm, title="Confusion Matrix"):
    fig = go.Figure(data=go.Heatmap(z=cm, x=['Pred 0','Pred 1'], y=['True 0','True 1'], colorscale='Blues', showscale=True, text=cm, texttemplate='%{text}'))
    fig.update_layout(title=title, width=420, height=360)
    return fig

def feature_importance_plot(model, feature_names, title="Feature importances"):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        inds = np.argsort(imp)[::-1]
        fig = go.Figure([go.Bar(x=[feature_names[i] for i in inds], y=imp[inds])])
        fig.update_layout(title=title, xaxis_title='Feature', yaxis_title='Importance', width=800, height=400)
        return fig
    return None

# ---------- Overview tab ----------
with tabs[0]:
    st.header("Overview — 5 combined business & statistical insights")
    uploaded = st.file_uploader("Upload dataset for overview (optional)", type=['csv'], key='overview_file')
    if uploaded:
        try:
            df_over = load_csv(uploaded)
            st.success("Loaded dataset for overview.")
        except Exception as e:
            st.error("Error loading file: " + str(e))
            st.stop()
    else:
        st.info("No dataset uploaded — using a generated sample. For best results upload your UniversalBank.csv.")
        # generate sample similar to Universal Bank
        rng = np.random.default_rng(1)
        n = 1000
        df_over = pd.DataFrame({
            'Age': rng.integers(22,65,n),
            'Experience': rng.integers(0,40,n),
            'Income': rng.integers(10,220,n),
            'Family': rng.integers(1,4,n),
            'CCAvg': np.round(rng.random(n)*10,2),
            'Education': rng.choice([1,2,3], n, p=[0.6,0.3,0.1]),
            'Mortgage': rng.integers(0,500,n),
            'Securities Account': rng.choice([0,1], n, p=[0.92,0.08]),
            'CD Account': rng.choice([0,1], n, p=[0.96,0.04]),
            'Online': rng.choice([0,1], n, p=[0.7,0.3]),
            'CreditCard': rng.choice([0,1], n, p=[0.7,0.3])
        })
        logits = (df_over['Income']*0.02 + df_over['CCAvg']*0.6 + (df_over['Education']-1)*1.0) - 5
        probs = 1/(1+np.exp(-logits))
        df_over['Personal Loan'] = (rng.random(n) < probs).astype(int)

    st.subheader("Sample rows")
    st.dataframe(df_over.head(5))

    # Insight 1: Income bin conversion rates (actionable)
    st.subheader("1) Income bin — observed conversion rates (actionable)")
    df_over['Income_bin'] = pd.cut(df_over['Income'], bins=[0,25,50,75,100,150,1000], labels=['0-25','25-50','50-75','75-100','100-150','150+'])
    bin_rates = df_over.groupby('Income_bin')['Personal Loan'].mean().reset_index()
    fig1 = px.bar(bin_rates, x='Income_bin', y='Personal Loan', text='Personal Loan', labels={'Personal Loan':'Acceptance rate','Income_bin':'Income bin'}, title='Acceptance rate by Income bin')
    st.plotly_chart(fig1, use_container_width=True)

    # Insight 2: Education x Family heatmap
    st.subheader("2) Education x Family — heatmap of acceptance rates (segmentation)")
    pivot = pd.pivot_table(df_over, values='Personal Loan', index='Education', columns='Family', aggfunc='mean').fillna(0)
    heat = px.imshow(pivot.values, x=pivot.columns.astype(str), y=["Edu_"+str(i) for i in pivot.index], text_auto='.3f', color_continuous_scale='RdYlGn', labels=dict(x='Family', y='Education'))
    st.plotly_chart(heat, use_container_width=True)

    # Insight 3: Income distribution by loan acceptance (statistical + business)
    st.subheader("3) Income distribution by loan acceptance (violin)")
    fig2 = go.Figure()
    fig2.add_trace(go.Violin(x=df_over['Personal Loan'].astype(str), y=df_over['Income'], box_visible=True, meanline_visible=True))
    fig2.update_layout(title='Income distribution - No vs Yes', xaxis_title='Personal Loan', yaxis_title='Income ($000)')
    st.plotly_chart(fig2, use_container_width=True)

    # Insight 4: Feature importance (Quick RF fit on current dataset)
    st.subheader("4) Quick RandomForest feature importance (on this dataset)")
    try:
        X_tmp, y_tmp = preprocess_for_model(df_over)
        rf_tmp = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tmp, y_tmp)
        imp = rf_tmp.feature_importances_
        feats = X_tmp.columns.tolist()
        imp_df = pd.DataFrame({'feature':feats, 'importance':imp}).sort_values('importance', ascending=False)
        fig3 = px.bar(imp_df, x='feature', y='importance', title='Feature importances (RF on current dataset)')
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.write("Could not compute quick importance:", e)

    # Insight 5: Age vs CCAvg scatter (color by conversion)
    st.subheader("5) Age vs CCAvg scatter (color = Personal Loan) — demographic targeting")
    fig4 = px.scatter(df_over, x='Age', y='CCAvg', color=df_over['Personal Loan'].astype(str), size='Income', hover_data=['Income','Education','Family'], title='Age vs CCAvg (color=PersonalLoan)')
    st.plotly_chart(fig4, use_container_width=True)

# ---------- Train & Evaluate tab ----------
with tabs[1]:
    st.header("Train models (Decision Tree, Random Forest, Gradient Boosting) and evaluate")
    uploaded = st.file_uploader("Upload training CSV (must include 'Personal Loan' column)", type=['csv'], key='train_upload')
    if uploaded is None:
        st.info("Upload your UniversalBank.csv to train models (5-fold CV).")
    else:
        df_train = load_csv(uploaded)
        st.write("Dataset shape:", df_train.shape)
        if st.button("Run training with 5-fold CV"):
            try:
                X, y = preprocess_for_model(df_train)
            except Exception as e:
                st.error("Preprocessing error: " + str(e))
                st.stop()
            with st.spinner("Training models (this may take a minute)..."):
                results, scaler, X_tr, X_te, y_tr, y_te = train_all_models(X, y)
            st.success("Training complete. Results below.")

            # show metrics table
            st.subheader("Performance table (CV 5-fold, train/test metrics)")
            mdf = metrics_df(results)
            st.dataframe(mdf)

            # ROC overlay
            st.subheader("ROC Curves overlay")
            st.plotly_chart(plot_roc_overlay(results), use_container_width=True)

            # Confusion matrices (test) and training confusion matrices
            st.subheader("Confusion Matrices (Test datasets)")
            cols = st.columns(3)
            for i, (name, res) in enumerate(results.items()):
                cols[i].plotly_chart(plot_confusion(confusion_matrix(res['y_test'], res['y_test_pred']), title=f"{name} - Test CM"))
            st.subheader("Confusion Matrices (Training datasets)")
            for name, res in results.items():
                st.plotly_chart(plot_confusion(confusion_matrix(y_tr, res['model'].predict(X_tr)), title=f"{name} - Train CM"))

            # Feature importances
            st.subheader("Feature importances per model")
            for name, res in results.items():
                fig_imp = feature_importance_plot(res['model'], res['feature_names'], title=f"{name} - Feature importances")
                if fig_imp is not None:
                    st.plotly_chart(fig_imp)

            # Save models and scaler to session for prediction tab
            st.session_state['trained_models'] = {k: v['model'] for k,v in results.items()}
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = results[next(iter(results))]['feature_names']

# ---------- Predict tab ----------
with tabs[2]:
    st.header("Upload new customer data and predict Personal Loan (downloadable)")
    uploaded = st.file_uploader("Upload new customers CSV (no Personal Loan column necessary)", type=['csv'], key='pred_upload')
    model_choice = st.selectbox("Choose model for prediction", ['Decision Tree','Random Forest','Gradient Boosting'])
    if uploaded is not None:
        newdf = load_csv(uploaded)
        if 'trained_models' not in st.session_state:
            st.warning("No trained models in session. Please train models on 'Train & Evaluate Models' tab first. You can still run predictions by training quickly here by pressing 'Quick train' below.")
            if st.button("Quick train on uploaded data and predict (5-fold CV)"):
                try:
                    X_all, y_all = preprocess_for_model(newdf.assign(**{'Personal Loan':0}))  # attempt, will fail gracefully
                except Exception:
                    st.error("Quick train requires a full training dataset with 'Personal Loan' column. Please upload a training dataset in Train & Evaluate tab.")
        else:
            models = st.session_state['trained_models']
            feature_names = st.session_state['feature_names']

            # ensure required features present
            missing = [c for c in feature_names if c not in newdf.columns]
            if missing:
                st.warning(f"Uploaded data is missing features expected by model: {missing}. Filling missing columns with zeros.")
                for c in missing:
                    newdf[c] = 0
            Xnew = newdf[feature_names].copy()
            # scale numeric features if scaler in session
            scaler = st.session_state.get('scaler', None)
            cont = [c for c in feature_names if c.lower() in ['age','experience','income','ccavg','mortgage']]
            if cont and scaler is not None:
                Xnew[cont] = scaler.transform(Xnew[cont])
            model = st.session_state['trained_models'][model_choice]
            preds = model.predict(Xnew)
            proba = model.predict_proba(Xnew)[:,1] if hasattr(model,'predict_proba') else np.zeros(len(preds))
            newdf['Predicted_PersonalLoan'] = preds.astype(int)
            newdf['Prediction_Prob'] = np.round(proba,4)
            st.write("Sample predictions:")
            st.dataframe(newdf.head(20))
            # download
            towrite = BytesIO()
            newdf.to_csv(towrite, index=False)
            towrite.seek(0)
            st.download_button("Download predictions CSV", data=towrite, file_name="predictions.csv", mime='text/csv')

# ---------- Data & Notes ----------
with tabs[3]:
    st.header("Data dictionary and deployment notes")
    dd = pd.DataFrame({
        'Field':['ID','Personal Loan','Age','Experience','Income','Zip code','Family','CCAvg','Education','Mortgage','Securities','CDAccount','Online','CreditCard'],
        'Description':[
            'unique identifier',
            'did the customer accept the personal loan offered (1=Yes, 0=No)',
            'customer age',
            'years of professional experience',
            'annual income ($000)',
            'home address zip code',
            'family size',
            'avg credit card spend per month ($000)',
            'education level (1 undergrad,2 grad,3 advanced)',
            'value of house mortgage ($000)',
            'has securities account (1/0)',
            'has CD account (1/0)',
            'uses online banking (1/0)',
            'has credit card from bank (1/0)'
        ]
    })
    st.dataframe(dd)
    st.markdown("**Deployment notes**: This app trains models inside the session using scikit-learn and avoids pickle compatibility issues. Include requirements.txt in your repo root when deploying to Streamlit Cloud.")
