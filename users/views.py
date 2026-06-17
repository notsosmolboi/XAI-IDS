from xml.parsers.expat import model

from django.shortcuts import render, redirect
from django.contrib import messages
from .models import UserRegistrationModel
import pandas as pd
import numpy as np
import os
import pickle
import base64
import io
import warnings
warnings.filterwarnings('ignore')

from django.conf import settings

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score, f1_score)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

MEDIA_ROOT        = settings.MEDIA_ROOT
MODEL_PATH        = os.path.join(MEDIA_ROOT, 'rf_model.pkl')
SCALER_PATH       = os.path.join(MEDIA_ROOT, 'scaler.pkl')
FEATURE_PATH      = os.path.join(MEDIA_ROOT, 'feature_cols.pkl')
LABEL_PATH        = os.path.join(MEDIA_ROOT, 'label_encoder.pkl')
ACCURACY_PATH     = os.path.join(MEDIA_ROOT, 'accuracy_results.pkl')
DATASET_NAME_PATH = os.path.join(MEDIA_ROOT, 'dataset_name.txt')
DEFAULT_DATASET   = 'Balanced_IDS_Data.csv'

IMPORTANT_FEATURES = [
    "FlowDuration","TotalFwdPackets","TotalLengthofFwdPackets",
    "FlowIATMean","FwdPacketLengthMean","FlowPackets_per_s",
    "FwdHeaderLength","BwdHeaderLength","AvgFwdSegmentSize","AvgBwdSegmentSize",
]

ATTACK_REASONS = {
    'DDOS': {
        'icon': 'fa-water', 'color': '#ff6b6b',
        'reason': 'Distributed Denial of Service (DDoS) attack detected. The network traffic shows characteristics of a flood attack designed to overwhelm the target server.',
        'indicators': [
            'Extremely high packet rate (FlowPackets_per_s) — server being flooded',
            'Very short flow duration with massive packet count',
            'Minimal backward traffic — server unable to respond',
            'Abnormally high TotalFwdPackets in very short time window',
        ]
    },
    'BRUTE_FORCE': {
        'icon': 'fa-key', 'color': '#ffa500',
        'reason': 'Brute Force attack detected. Repeated automated login attempts observed targeting authentication systems.',
        'indicators': [
            'High packet frequency consistent with automated login attempts',
            'Repeated connection patterns with similar packet sizes',
            'Symmetric forward/backward headers — typical of login protocols',
            'Elevated FlowPackets_per_s matching automated attack tools',
        ]
    },
    'BOT': {
        'icon': 'fa-robot', 'color': '#9b59b6',
        'reason': 'Bot / Command-and-Control (C&C) traffic detected. The connection shows signs of automated bot behaviour communicating with a remote server.',
        'indicators': [
            'Very long flow duration — persistent C&C connection',
            'Low packet rate — periodic bot check-in behaviour',
            'Consistent average segment sizes — automated pattern',
            'Long FlowIATMean — bot timer between communications',
        ]
    },
    'NORMAL': {
        'icon': 'fa-shield-halved', 'color': '#64ffda',
        'reason': 'Traffic is normal. No malicious patterns detected in this network flow.',
        'indicators': [
            'Flow duration within expected range for normal traffic',
            'Packet rate consistent with regular browsing or data transfer',
            'Balanced forward and backward segment sizes',
            'No anomalous patterns detected by the model',
        ]
    },
}


def get_dataset_name():
    try:
        with open(DATASET_NAME_PATH, 'r') as f:
            return f.read().strip()
    except Exception:
        return DEFAULT_DATASET


def set_dataset_name(name):
    with open(DATASET_NAME_PATH, 'w') as f:
        f.write(name)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


def UserRegisterActions(request):
    if request.method == 'POST':
        name = request.POST.get('txtName')
        email = request.POST.get('txtEmail')
        mobile = request.POST.get('txtMobile')
        password = request.POST.get('txtPassword')
        if UserRegistrationModel.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered!')
            return render(request, 'UserRegister.html')
        UserRegistrationModel.objects.create(
            name=name, email=email, mobile=mobile, password=password, status='pending')
        messages.success(request, 'Registration successful! Please wait for admin approval.')
        return render(request, 'UserRegister.html')
    return render(request, 'UserRegister.html')


def UserLoginCheck(request):
    if request.method == 'POST':
        name = request.POST.get('loginid')
        password = request.POST.get('pswd')
        try:
            user = UserRegistrationModel.objects.get(name=name, password=password)
            if user.status == 'activated':
                request.session['user_id'] = user.id
                request.session['user_name'] = user.name
                request.session.set_expiry(86400)
                return redirect('UserHome')
            else:
                messages.error(request, 'Your account is not yet activated by admin.')
        except UserRegistrationModel.DoesNotExist:
            messages.error(request, 'Invalid name or password.')
    return render(request, 'UserLogin.html')


def UserHome(request):
    if 'user_id' not in request.session:
        return redirect('UserLogin')
    return render(request, 'users/UserHome.html', {'user_name': request.session.get('user_name', '')})


def index(request):
    return render(request, 'index.html')

def UserLogout(request):
    request.session.flush()
    return redirect('index')


def ViewDataset(request):
    if 'user_id' not in request.session:
        return redirect('UserLogin')
    csv_path = os.path.join(MEDIA_ROOT, DEFAULT_DATASET)
    data_html = ''
    shape_info = ''
    label_counts = ''
    dataset_name = get_dataset_name()
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        df =df.dropna(subset=['Label'])
        df = df.reset_index(drop=True)
        shape_info = f"Dataset: <strong>{dataset_name}</strong> &mdash; {df.shape[0]} rows &times; {df.shape[1]} columns (showing first 200 rows)"
        if 'Label' in df.columns:
            label_counts = df['Label'].value_counts().to_dict()
        data_html = df.to_html(classes='table table-striped table-hover table-sm', index=False, border=0, max_rows=100)
    except Exception as e:
        data_html = f'<p class="text-danger">Error loading dataset: {e}</p>'
    return render(request, 'users/ViewDataset.html', {
        'data_html': data_html, 'shape_info': shape_info,
        'label_counts': label_counts, 'dataset_name': dataset_name,
        'user_name': request.session.get('user_name', ''),
    })


def upload_data_view(request):
    if 'user_id' not in request.session:
        return redirect('UserLogin')
    dataset_name = get_dataset_name()
    if request.method == 'POST' and request.FILES.get('datafile'):
        uploaded_file = request.FILES['datafile']
        original_name = uploaded_file.name
        file_path = os.path.join(MEDIA_ROOT, DEFAULT_DATASET)
        with open(file_path, 'wb+') as dest:
            for chunk in uploaded_file.chunks():
                dest.write(chunk)
        set_dataset_name(original_name)
        dataset_name = original_name
        messages.success(request, f'Dataset "{original_name}" uploaded successfully!')
    return render(request, 'users/UploadData.html', {
        'user_name': request.session.get('user_name', ''),
        'dataset_name': dataset_name,
    })


def training(request):
    if 'user_id' not in request.session:
        return redirect('UserLogin')

    dataset_name = get_dataset_name()
    context = {'user_name': request.session.get('user_name', ''), 'dataset_name': dataset_name}

    if request.method == 'POST':
        try:
            csv_path = os.path.join(MEDIA_ROOT, DEFAULT_DATASET)
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()

            if 'Label' not in df.columns:
                raise ValueError("No 'Label' column found in dataset.")

            available_features = [f for f in IMPORTANT_FEATURES if f in df.columns]
            if not available_features:
                raise ValueError("None of the important features found in dataset.")

            X = df[available_features].apply(pd.to_numeric, errors='coerce')
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            feature_cols = available_features

            le = LabelEncoder()
            y = le.fit_transform(df['Label'].astype(str))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

            if SMOTE_AVAILABLE:
                try:
                    sm = SMOTE(random_state=42)
                    X_train, y_train = sm.fit_resample(X_train, y_train)
                except Exception:
                   pass

            rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                max_iter=300, random_state=42, early_stopping=True,
                validation_fraction=0.1, learning_rate_init=0.001,
                batch_size=512, tol=0.001, n_iter_no_change=15)

            estimators = [('rf', rf), ('mlp', mlp)]
            models_list = [(rf, 'Random Forest'), (mlp, 'MLP')]

            if XGBOOST_AVAILABLE:
                xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                    eval_metric='mlogloss', random_state=42, use_label_encoder=False)
                estimators.append(('xgb', xgb))
                models_list.append((xgb, 'XGBoost'))

            voting = VotingClassifier(estimators=estimators, voting='soft')
            models_list.append((voting, 'Hybrid Voting'))

            all_metrics = []
            for m, mname in models_list:
                m.fit(X_train, y_train)
                yp = m.predict(X_test)
                all_metrics.append({
                    'Algorithm': mname,
                    'Accuracy':  round(accuracy_score(y_test, yp) * 100, 2),
                    'Precision': round(precision_score(y_test, yp, average='weighted', zero_division=0) * 100, 2),
                    'Recall':    round(recall_score(y_test, yp, average='weighted', zero_division=0) * 100, 2),
                    'F1Score':  round(f1_score(y_test, yp, average='weighted', zero_division=0) * 100, 2),
                })

            model = voting
            y_pred = model.predict(X_test)
            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            report = classification_report(y_test, y_pred,
                                           target_names=[str(c) for c in le.classes_], zero_division=0)

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=le.classes_, yticklabels=le.classes_)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix – Hybrid Voting')
            cm_img = fig_to_base64(fig)

            comparison_img = None
            try:
                mdf = pd.DataFrame(all_metrics)
                fig_c, ax_c = plt.subplots(figsize=(10, 6))
                x = np.arange(len(mdf)); w = 0.2
                ax_c.bar(x-w*1.5, mdf['Accuracy'],  w, label='Accuracy',  color='steelblue')
                ax_c.bar(x-w*0.5, mdf['Precision'], w, label='Precision', color='green')
                ax_c.bar(x+w*0.5, mdf['Recall'],    w, label='Recall',    color='orange')
                ax_c.bar(x+w*1.5, mdf['F1Score'],  w, label='F1 Score',  color='red')
                ax_c.set_xticks(x); ax_c.set_xticklabels(mdf['Algorithm'])
                ax_c.set_ylabel('Score (%)'); ax_c.set_title('Algorithm Comparison')
                ax_c.legend(); ax_c.set_ylim(0, 115)
                for i, row in mdf.iterrows():
                    ax_c.text(i-w*1.5, row['Accuracy']+1, f"{row['Accuracy']:.1f}", ha='center', fontsize=7)
                plt.tight_layout()
                comparison_img = fig_to_base64(fig_c)
            except Exception:
                comparison_img = None

            fi_img = None
            try:
                importances = rf.feature_importances_
                indices = np.argsort(importances)[::-1]
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.bar(range(len(indices)), importances[indices], color='steelblue')
                ax2.set_xticks(range(len(indices)))
                ax2.set_xticklabels([feature_cols[i] for i in indices], rotation=45, ha='right', fontsize=9)
                ax2.set_title('Feature Importances (Random Forest)')
                ax2.set_ylabel('Importance Score')
                plt.tight_layout()
                fi_img = fig_to_base64(fig2)
            except Exception as fi_err:
                fi_img = None; context['fi_error'] = str(fi_err)

            shap_img = None
            if SHAP_AVAILABLE:
                try:
                    explainer = shap.KernelExplainer(mlp.predict_proba, shap.sample(X_train, 50))
                    shap_sample = X_test[:50]
                    shap_values = explainer.shap_values(shap_sample)
                    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
                    shap.summary_plot(sv, shap_sample, feature_names=feature_cols, show=False, max_display=10)
                    shap_img = fig_to_base64(plt.gcf())
                    plt.close('all')
                except Exception as se:
                    shap_img = None; context['shap_error'] = str(se)

            lime_img = None
            if LIME_AVAILABLE:
                try:
                    lime_exp = lime.lime_tabular.LimeTabularExplainer(
                        X_train, feature_names=feature_cols,
                        class_names=[str(c) for c in le.classes_], discretize_continuous=True)
                    exp = lime_exp.explain_instance(X_test[0], mlp.predict_proba, num_features=10)
                    fig4, ax4 = plt.subplots(figsize=(10, 6))
                    lv = exp.as_list()
                    ax4.barh([x[0] for x in lv], [x[1] for x in lv],
                             color=['#e74c3c' if x[1] < 0 else '#2ecc71' for x in lv])
                    ax4.set_title('LIME Explanation – Sample Instance (MLP)')
                    ax4.set_xlabel('Feature Weight')
                    ax4.axvline(0, color='black', linewidth=0.8)
                    plt.tight_layout()
                    lime_img = fig_to_base64(fig4)
                except Exception as le_err:
                    lime_img = None; context['lime_error'] = str(le_err)

            with open(MODEL_PATH, 'wb')   as f: pickle.dump(model, f)
            with open(SCALER_PATH, 'wb')  as f: pickle.dump(scaler, f)
            with open(FEATURE_PATH, 'wb') as f: pickle.dump(feature_cols, f)
            with open(LABEL_PATH, 'wb')   as f: pickle.dump(le, f)

            results = {
                'accuracy': round(acc*100, 2), 'precision': round(prec*100, 2),
                'recall': round(rec*100, 2), 'f1': round(f1*100, 2),
                'report': report, 'cm_img': cm_img, 'fi_img': fi_img,
                'shap_img': shap_img, 'lime_img': lime_img,
                'comparison_img': comparison_img, 'all_metrics': all_metrics,
                'n_train': len(X_train), 'n_test': len(X_test),
                'n_classes': len(le.classes_), 'classes': list(le.classes_),
                'dataset_name': dataset_name,
            }
            with open(ACCURACY_PATH, 'wb') as f: pickle.dump(results, f)
            context.update(results)
            context['success'] = True

        except Exception as e:
            import traceback
            context['error'] = str(e)
            context['traceback'] = traceback.format_exc()

    return render(request, 'users/Training.html', context)


def prediction(request):
    if 'user_id' not in request.session:
        return redirect('UserLogin')

    context = {'user_name': request.session.get('user_name', '')}

    model_loaded = False
    try:
        with open(MODEL_PATH, 'rb')   as f: model = pickle.load(f)
        with open(SCALER_PATH, 'rb')  as f: scaler = pickle.load(f)
        with open(FEATURE_PATH, 'rb') as f: feature_cols = pickle.load(f)
        with open(LABEL_PATH, 'rb')   as f: le = pickle.load(f)
        model_loaded = True
        context['feature_cols'] = feature_cols
        context['important_features'] = feature_cols
        context['classes'] = list(le.classes_)
    except Exception:
        context['model_error'] = 'Model not trained yet. Please run training first.'

    if request.method == 'POST' and model_loaded:
        try:
            input_vals = []
            for fc in feature_cols:
                val = request.POST.get(fc, '0')
                input_vals.append(float(val))

            input_arr    = np.array(input_vals).reshape(1, -1)
            input_scaled = scaler.transform(input_arr)
            pred_class_idx = model.predict(input_scaled)[0]
            pred_proba     = model.predict_proba(input_scaled)[0]
            pred_label     = le.inverse_transform([pred_class_idx])[0]
            confidence     = round(pred_proba.max() * 100, 2)

            # Attack reason and explanation
            label_upper = pred_label.upper()
            attack_info = ATTACK_REASONS.get(label_upper, {
                'icon': 'fa-triangle-exclamation', 'color': '#ff6b6b',
                'reason': f'Suspicious traffic detected: {pred_label}',
                'indicators': ['Unusual network pattern detected by the model.'],
            })

            input_dict = dict(zip(feature_cols, input_vals))
            feature_contributions = []
            if label_upper == 'DDOS':
                if input_dict.get('FlowPackets_per_s', 0) > 5000:
                    feature_contributions.append(f"FlowPackets_per_s = {input_dict['FlowPackets_per_s']:.1f} → extremely high (flood attack)")
                if input_dict.get('TotalFwdPackets', 0) > 1000:
                    feature_contributions.append(f"TotalFwdPackets = {int(input_dict['TotalFwdPackets'])} → massive packet count")
                if input_dict.get('FlowDuration', 0) < 10000:
                    feature_contributions.append(f"FlowDuration = {int(input_dict['FlowDuration'])} µs → very short burst")
            elif label_upper == 'BRUTE_FORCE':
                if input_dict.get('FlowPackets_per_s', 0) > 200:
                    feature_contributions.append(f"FlowPackets_per_s = {input_dict['FlowPackets_per_s']:.1f} → high (repeated attempts)")
                if input_dict.get('TotalFwdPackets', 0) > 30:
                    feature_contributions.append(f"TotalFwdPackets = {int(input_dict['TotalFwdPackets'])} → repeated login tries")
                if input_dict.get('FlowIATMean', 0) < 5000:
                    feature_contributions.append(f"FlowIATMean = {input_dict['FlowIATMean']:.1f} → rapid repeated connections")
            elif label_upper == 'BOT':
                if input_dict.get('FlowDuration', 0) > 500000:
                    feature_contributions.append(f"FlowDuration = {int(input_dict['FlowDuration'])} µs → persistent C&C connection")
                if input_dict.get('FlowPackets_per_s', 0) < 50:
                    feature_contributions.append(f"FlowPackets_per_s = {input_dict['FlowPackets_per_s']:.1f} → low (periodic bot check-in)")
                if input_dict.get('FlowIATMean', 0) > 20000:
                    feature_contributions.append(f"FlowIATMean = {input_dict['FlowIATMean']:.1f} → long timer (automated bot)")

            context['attack_info'] = attack_info
            context['feature_contributions'] = feature_contributions
            context['is_attack'] = label_upper not in ('NORMAL', 'BENIGN')
        

            # LIME
            lime_pred_img = None
            if LIME_AVAILABLE:
                try:
                    csv_path = os.path.join(MEDIA_ROOT, DEFAULT_DATASET)
                    df_ref   = pd.read_csv(csv_path)
                    df_ref.columns = df_ref.columns.str.strip()
                    df_ref.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df_ref.fillna(0, inplace=True)
                    X_ref        = df_ref[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
                    X_ref_scaled = scaler.transform(X_ref[:500])
                
                    predict_fn = model.predict_proba
                    if hasattr(model, 'named_estimators_'):
                        for name, est in model.named_estimators_.items():
                            if isinstance(est, MLPClassifier):
                                predict_fn = lambda x: est.predict_proba(x)
                                break

                    lime_exp = lime.lime_tabular.LimeTabularExplainer(
                        X_ref_scaled, feature_names=feature_cols,
                        class_names=[str(c) for c in le.classes_], discretize_continuous=True)
                    exp = lime_exp.explain_instance(input_scaled[0], predict_fn, num_features=10)
                    fig5, ax5 = plt.subplots(figsize=(9, 5))
                    lv = exp.as_list()
                    ax5.barh([x[0] for x in lv], [x[1] for x in lv],
                             color=['#e74c3c' if x[1] < 0 else '#2ecc71' for x in lv])
                    ax5.set_title(f'LIME – Why predicted: {pred_label}')
                    ax5.set_xlabel('Feature Weight')
                    ax5.axvline(0, color='black', linewidth=0.8)
                    plt.tight_layout()
                    lime_pred_img = fig_to_base64(fig5)
                except Exception as lime_ex:
                    lime_pred_img = None
                    context['lime_pred_error'] = str(lime_ex)

            # SHAP
            shap_pred_img = None
            if SHAP_AVAILABLE:
                try:
                    # Replace with:
                    rf_est = None
                    if hasattr(model, 'named_estimators_'):
                        for name, est in model.named_estimators_.items():
                            if isinstance(est, RandomForestClassifier):
                                rf_est = est
                                break
                    if rf_est:
                        explainer = shap.TreeExplainer(rf_est)
                        sv = explainer.shap_values(input_scaled, check_additivity=False)
                       # To this:
                        sv_inst = sv[pred_class_idx] if isinstance(sv, list) else sv[:, :]
                        plt.figure(figsize=(9, 5))
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=sv_inst[0],
                                base_values=(
                                    explainer.expected_value[pred_class_idx]
                                    if isinstance(explainer.expected_value, (list, np.ndarray))
                                    else explainer.expected_value),
                                data=input_scaled[0],
                                feature_names=feature_cols),
                            show=False, max_display=10)
                        shap_pred_img = fig_to_base64(plt.gcf())
                        plt.close('all')
                except Exception as shap_ex:
                    shap_pred_img = None
                    context['shap_pred_error'] = str(shap_ex)

            context.update({
                'prediction':    pred_label,
                'confidence':    confidence,
                'proba':         list(zip(le.classes_, [round(p*100, 2) for p in pred_proba])),
                'lime_pred_img': lime_pred_img,
                'shap_pred_img': shap_pred_img,
                'input_vals':    dict(zip(feature_cols, input_vals)),
            })

        except Exception as e:
            context['pred_error'] = str(e)

    try:
        with open(ACCURACY_PATH, 'rb') as f:
            acc_results = pickle.load(f)
        context['acc_results'] = acc_results
    except Exception:
        pass


    return render(request, 'users/Prediction.html', context)
def auto_prediction(request):
    if 'user_id' not in request.session:
        return redirect('UserLogin')

    context = {'user_name': request.session.get('user_name', '')}

    model_loaded = False
    try:
        with open(MODEL_PATH, 'rb')   as f: model = pickle.load(f)
        with open(SCALER_PATH, 'rb')  as f: scaler = pickle.load(f)
        with open(FEATURE_PATH, 'rb') as f: feature_cols = pickle.load(f)
        with open(LABEL_PATH, 'rb')   as f: le = pickle.load(f)
        model_loaded = True
    except Exception:
        context['model_error'] = 'Model not trained yet. Please run training first.'

    if request.method == 'POST' and model_loaded:
        try:
            uploaded_file = request.FILES.get('trafficfile')
            if not uploaded_file:
                context['error'] = 'Please upload a CSV file.'
                return render(request, 'users/AutoPrediction.html', context)

            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()

            missing = [f for f in feature_cols if f not in df.columns]
            if missing:
                context['error'] = f'Missing columns in CSV: {missing}'
                return render(request, 'users/AutoPrediction.html', context)

            X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            X_scaled = scaler.transform(X)

            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            pred_labels = le.inverse_transform(predictions)
            confidences = [round(p.max() * 100, 2) for p in probabilities]

            results = []
            attack_counts = {}

            for i, (label, conf) in enumerate(zip(pred_labels, confidences)):
                label_upper = label.upper()
                attack_info = ATTACK_REASONS.get(label_upper, {
                    'icon': 'fa-triangle-exclamation',
                    'color': '#ff6b6b',
                })
                results.append({
                    'row': i + 1,
                    'label': label,
                    'confidence': conf,
                    'is_attack': label_upper not in ('NORMAL', 'BENIGN'),
                    'color': attack_info.get('color', '#ff6b6b'),
                    'icon': attack_info.get('icon', 'fa-triangle-exclamation'),
                })
                attack_counts[label_upper] = attack_counts.get(label_upper, 0) + 1

            total = len(results)
            attack_total = sum(v for k, v in attack_counts.items() if k != 'NORMAL')

            context.update({
                'results': results,
                'total': total,
                'attack_counts': attack_counts,
                'attack_total': attack_total,
                'normal_total': attack_counts.get('NORMAL', 0),
                'filename': uploaded_file.name,
            })

        except Exception as e:
            context['error'] = str(e)

    return render(request, 'users/AutoPrediction.html', context)

def network_simulation(request):
    if 'user_id' not in request.session:
        return redirect('UserLogin')
    return render(request, 'users/NetworkSimulation.html', {
        'user_name': request.session.get('user_name', ''),
    })
