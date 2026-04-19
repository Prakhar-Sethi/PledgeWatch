"""
Generate docs/data.js from actual pipeline output.
Run: python3 docs/generate_data.py
"""
import pandas as pd, json, pickle, numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

master = pd.read_parquet('data/processed/master_dataset.parquet')
tone_df = pd.read_parquet('data/processed/tone_features.parquet')
shap_csv = pd.read_csv('data/processed/shap_importance.csv')

with open('data/processed/models/M2_pledge_price_fin.pkl', 'rb') as f:
    m2_pkg = pickle.load(f)
m2 = m2_pkg['model']
m2_feats = m2_pkg['features']

df = master.copy()
X_all = df[m2_feats].fillna(-1)
df['risk_score'] = m2.predict_proba(X_all)[:, 1]

latest = df.sort_values('quarter').groupby('nse_symbol').last().reset_index()

price_hist = {}
for sym, grp in df.sort_values('quarter').groupby('nse_symbol'):
    prices = grp['close_price'].tail(13).tolist()
    prices = [float(p) for p in prices if p and not (p != p)]
    if len(prices) >= 4:
        base = prices[0]
        if base and base > 0:
            price_hist[sym] = [round(p / base * 100, 1) for p in prices]

tone_best = (tone_df[tone_df['overall_distress_score'] > 0]
             .sort_values('overall_distress_score', ascending=False)
             .groupby('nse_symbol').first().reset_index())
tone_best = tone_best.rename(columns={c: 'tone_' + c for c in tone_best.columns if c != 'nse_symbol'})
merged = latest.merge(tone_best, on='nse_symbol', how='left')

import shap
explainer = shap.TreeExplainer(m2)
X_latest = latest[m2_feats].fillna(-1)
shap_vals = explainer.shap_values(X_latest)

company_shap = {}
for i, sym in enumerate(latest['nse_symbol']):
    sv = shap_vals[i]
    pairs = sorted(zip(m2_feats, sv), key=lambda x: abs(x[1]), reverse=True)[:6]
    company_shap[sym] = [[f, round(float(v), 4)] for f, v in pairs]

risk_scores = latest['risk_score'].values
p75 = float(np.percentile(risk_scores, 75))
p40 = float(np.percentile(risk_scores, 40))

REAL_CASES = {'DHFL', 'ZEEL', 'YESBANK', 'JETAIRWAYS', 'ILFSENGG', 'RELCAPITAL'}

def safe_float(v, default=0.0):
    try:
        r = float(v)
        return default if (r != r) else r
    except:
        return default

def make_ticker(row, tier):
    sym = str(row['nse_symbol'])
    name = str(row.get('company_name') or sym)
    q = str(row.get('quarter', ''))
    q_display = (q[:4] + ' ' + q[4:]) if len(q) == 6 else q
    real = sym in REAL_CASES
    risk = round(safe_float(row['risk_score']), 3)

    feats = {
        'pledge_pct_4q_max':    round(safe_float(row.get('pledge_pct_4q_max')) / 100, 3),
        'price_volatility_60d': round(safe_float(row.get('price_volatility_60d')) / 100, 3),
        'price_vs_52w_high':    round(safe_float(row.get('price_vs_52w_high')) / 100, 3),
        'pledge_acceleration':  round(safe_float(row.get('pledge_acceleration')), 3),
        'price_return_6m':      round(safe_float(row.get('price_return_6m')) / 100, 3),
        'market_cap_log':       round(safe_float(row.get('market_cap_log')), 3),
    }

    entry = {
        't': sym, 'n': name, 'sector': 'equities',
        'risk': risk, 'tier': tier, 'real': real,
        'features': feats,
        'shap': company_shap.get(sym, []),
        'price': price_hist.get(sym, [100]),
        'quarter': q_display,
        'note': f"Model score {risk:.2f} · {q_display}",
    }

    td = row.get('tone_overall_distress_score')
    if pd.notna(td) and td and td > 0:
        kp_raw = row.get('tone_key_phrases', '[]')
        try:
            kp = json.loads(kp_raw) if isinstance(kp_raw, str) else list(kp_raw or [])
        except:
            kp = []
        tq = str(row.get('tone_quarter', ''))
        tq_display = (tq[:4] + ' ' + tq[4:]) if len(tq) == 6 else tq
        entry['tone'] = {
            'call': f"Earnings call · {tq_display} · DeepSeek V3 analysis",
            'scores': {
                'evasiveness':    safe_float(row.get('tone_evasiveness_score')),
                'reassurance':    safe_float(row.get('tone_reassurance_score')),
                'liquidity_stress': safe_float(row.get('tone_liquidity_stress_mentions')),
                'confidence':     safe_float(row.get('tone_confidence_score')),
                'vagueness':      safe_float(row.get('tone_guidance_vagueness_score')),
                'distress':       safe_float(row.get('tone_overall_distress_score')),
            },
            'quotes': [
                {'tag': 'key phrase', 'role': 'CEO', 'text': p,
                 'flag': 'Flagged by DeepSeek V3 earnings-call tone analysis'}
                for p in kp[:3] if p and len(p) > 10
            ],
        }
    return entry

# Select tickers
all_tickers = []
seen = set()

# High: top 50 by risk, pick those with price history
for _, row in merged.nlargest(60, 'risk_score').iterrows():
    sym = row['nse_symbol']
    if sym in seen or sym not in price_hist: continue
    seen.add(sym)
    all_tickers.append(make_ticker(row, 'high'))
    if len([t for t in all_tickers if t['tier'] == 'high']) >= 10: break

# Med: p40-p75 range
med_pool = merged[(merged['risk_score'] >= p40) & (merged['risk_score'] < p75)]
for _, row in med_pool.sample(frac=1, random_state=42).iterrows():
    sym = row['nse_symbol']
    if sym in seen or sym not in price_hist: continue
    seen.add(sym)
    all_tickers.append(make_ticker(row, 'med'))
    if len([t for t in all_tickers if t['tier'] == 'med']) >= 6: break

# Low: bottom 25%
low_pool = merged[merged['risk_score'] < p40]
for _, row in low_pool.sample(frac=1, random_state=42).iterrows():
    sym = row['nse_symbol']
    if sym in seen or sym not in price_hist: continue
    seen.add(sym)
    all_tickers.append(make_ticker(row, 'low'))
    if len([t for t in all_tickers if t['tier'] == 'low']) >= 5: break

# Global SHAP from CSV
shap_rows = []
for _, row in shap_csv.iterrows():
    shap_rows.append([str(row['feature']), round(float(row['mean_abs_shap']), 7)])

# Write JS
out = f"""// Auto-generated by docs/generate_data.py from pipeline output
// DO NOT EDIT MANUALLY — re-run generate_data.py to update

window.PW_TICKERS = {json.dumps(all_tickers, indent=2)};

window.PW_GLOBAL_SHAP = {json.dumps(shap_rows, indent=2)};
"""

with open('docs/data.js', 'w') as f:
    f.write(out)

print(f"Written docs/data.js: {len(all_tickers)} tickers "
      f"({sum(1 for t in all_tickers if t['tier']=='high')} high, "
      f"{sum(1 for t in all_tickers if t['tier']=='med')} med, "
      f"{sum(1 for t in all_tickers if t['tier']=='low')} low)")
print(f"Tickers with tone: {sum(1 for t in all_tickers if 'tone' in t)}")
print("Tickers:", [t['t'] for t in all_tickers])
