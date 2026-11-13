"""Churn explorer (Gradio): describe a customer → lifetime curve, chance to stay, expected months and revenue at risk.

The model is rebuilt at start-up exactly as in the notebook (Cox proportional hazards, stratified by contract).
Data: TELCO_CSV, else ./data/WA_Fn-UseC_-Telco-Customer-Churn.csv, else IBM's public copy is downloaded to ./data.
No customer data is stored in this repository.

    pip install -r requirements.txt
    python app.py

Optional: CHURN_DEMO=1 runs an example customer on page load (used for the README screenshot).
"""
import os
import urllib.request
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

HERE = Path(__file__).resolve().parent
URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
PENALIZER, HORIZON = 0.01, 24
ADDONS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']


def load_data():
    path = Path(os.environ.get('TELCO_CSV', HERE / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return pd.read_csv(path)


def encode(df):
    """Same encoding as the notebook (TotalCharges left out: it leaks tenure)."""
    yes = lambda s: (s == 'Yes').astype(int)
    out = pd.DataFrame(index=df.index)
    out['male'] = (df['gender'] == 'Male').astype(int)
    out['senior'] = df['SeniorCitizen'].astype(int)
    for col in ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'PaperlessBilling'] + ADDONS:
        out[col] = yes(df[col])
    out['fiber_optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    out['no_internet'] = (df['InternetService'] == 'No').astype(int)
    out['contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    out['pay_electronic_check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    out['pay_mailed_check'] = (df['PaymentMethod'] == 'Mailed check').astype(int)
    out['pay_credit_card'] = (df['PaymentMethod'] == 'Credit card (automatic)').astype(int)
    out['monthly_charges_10usd'] = df['MonthlyCharges'] / 10
    return out


raw = load_data()
raw = raw[raw['tenure'] > 0]
X = encode(raw).assign(tenure=raw['tenure'], churn=(raw['Churn'] == 'Yes').astype(int))
cph = CoxPHFitter(penalizer=PENALIZER).fit(X, duration_col='tenure', event_col='churn', strata=['contract'])
INPUTS = [c for c in X.columns if c not in ('tenure', 'churn')]
TYPICAL = X[INPUTS].median()
GRID = list(range(0, 85))
CONTRACTS = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}


def curve(row):
    s = cph.predict_survival_function(pd.DataFrame([row])[INPUTS], times=GRID).iloc[:, 0].to_numpy()
    s[0] = 1.0
    return s


def explore(contract, internet, payment, tech, security, backup, paperless, partner, monthly, months_so_far):
    row = TYPICAL.copy()
    row.update({'contract': CONTRACTS[contract], 'fiber_optic': int(internet == 'Fiber optic'), 'no_internet': int(internet == 'No'),
                'pay_electronic_check': int(payment == 'Electronic check'), 'pay_mailed_check': int(payment == 'Mailed check'),
                'pay_credit_card': int(payment == 'Credit card (automatic)'), 'TechSupport': int(tech), 'OnlineSecurity': int(security),
                'OnlineBackup': int(backup), 'PaperlessBilling': int(paperless), 'Partner': int(partner),
                'monthly_charges_10usd': monthly / 10})
    t0 = int(months_so_far)
    s = curve(row)
    cond = s[t0:t0 + HORIZON + 1] / max(s[t0], 1e-9)
    typical = curve({**TYPICAL.to_dict(), 'contract': CONTRACTS[contract]})
    cond_typ = typical[t0:t0 + HORIZON + 1] / max(typical[t0], 1e-9)
    months_12 = np.trapz(cond[:13])
    text = (f'### Expected to stay {np.trapz(cond):.1f} of the next {HORIZON} months\n'
            f'Revenue at risk in the next 12 months: **${monthly * (12 - months_12):,.0f}** '
            f'({12 - months_12:.1f} expected months lost × ${monthly:.0f}).\n\n'
            f'A typical {contract.lower()} customer at the same point would stay {np.trapz(cond_typ):.1f} months.')
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(range(len(cond)), cond, color='red', label='this customer')
    ax.plot(range(len(cond_typ)), cond_typ, color='grey', linestyle='--', label=f'typical {contract.lower()} customer')
    for h in (6, 12):
        ax.axvline(h, color='lightgrey', linestyle=':')
    ax.set(xlabel='months from now', ylabel='P(still a customer)', ylim=(0, 1.02), title=f'After {t0} months as a customer')
    ax.legend(loc='lower left')
    fig.tight_layout()
    return {f'still a customer in {h} months': float(cond[h]) for h in (6, 12, 24)}, text, fig


with gr.Blocks(title='Churn explorer') as demo:
    gr.Markdown('# Customer Churn Explorer\nDescribe a customer → how long they are likely to stay, and what that is worth.')
    with gr.Row():
        with gr.Column(scale=2):
            contract = gr.Dropdown(list(CONTRACTS), value='Month-to-month', label='Contract')
            internet = gr.Dropdown(['DSL', 'Fiber optic', 'No'], value='Fiber optic', label='Internet')
            payment = gr.Dropdown(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                                  value='Electronic check', label='Payment method')
            with gr.Row():
                tech, security = gr.Checkbox(label='Tech support'), gr.Checkbox(label='Online security')
            with gr.Row():
                backup, paperless, partner = gr.Checkbox(label='Online backup'), gr.Checkbox(value=True, label='Paperless billing'), gr.Checkbox(label='Has partner')
            monthly = gr.Slider(18, 120, value=90, step=0.5, label='Monthly charge ($)')
            months = gr.Slider(1, 60, value=3, step=1, label='Months as a customer so far')
            btn = gr.Button('Explore', variant='primary')
        with gr.Column(scale=3):
            label = gr.Label(label='Chance to stay')
            text = gr.Markdown()
            plot = gr.Plot(show_label=False)
    inputs = [contract, internet, payment, tech, security, backup, paperless, partner, monthly, months]
    btn.click(explore, inputs=inputs, outputs=[label, text, plot])
    if os.environ.get('CHURN_DEMO') == '1':
        cached = explore('Month-to-month', 'Fiber optic', 'Electronic check', False, False, False, True, False, 90, 3)
        demo.load(lambda: cached, outputs=[label, text, plot])

if __name__ == '__main__':
    demo.launch()
