import pandas as pd
import numpy as np
import plotly.graph_objects as go

print("=" * 62)
print("  STORY BEAT 17: BEYOND AUC -- PRECISION, RECALL, F1")
print("=" * 62)

def precision_recall_f1(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = ((y_pred==1)&(y_true==1)).sum()
    fp = ((y_pred==1)&(y_true==0)).sum()
    fn = ((y_pred==0)&(y_true==1)).sum()
    tn = ((y_pred==0)&(y_true==0)).sum()
    precision = tp / max(tp+fp, 1)
    recall    = tp / max(tp+fn, 1)
    f1        = 2*precision*recall / max(precision+recall, 0.001)
    accuracy  = (tp+tn) / max(len(y_true), 1)
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=round(precision,4), recall=round(recall,4),
                f1=round(f1,4), accuracy=round(accuracy,4))

# Full model predictions
full_preds = full_model.predict(sc2.transform(X_te))
fm = precision_recall_f1(y_te, full_preds)

# Early warning model
ew_preds = ew_model.predict(sc_d3.transform(X_d3_te))
em = precision_recall_f1(y_d3_te, ew_preds)

# Upgrade model
up_preds = up_model.predict(sc_up.transform(X_up_te))
um = precision_recall_f1(y_up_te, up_preds)

print('\n=== CLASSIFICATION PERFORMANCE -- ALL 3 MODELS ===\n')
print(f'{"Metric":<20} {"Full Model":>12} {"Early Warning":>14} {"Upgrade Model":>14}')
print('-' * 62)
for key in ['accuracy','precision','recall','f1']:
    print(f'  {key:<18} {fm[key]:>12.4f} {em[key]:>14.4f} {um[key]:>14.4f}')
print(f'\n  {"AUC":<18} {full_auc:>12.4f} {ew_auc:>14.4f} {up_auc:>14.4f}')

print(f'\n  Confusion Matrix (Full Model):')
print(f'                  Predicted No   Predicted Yes')
print(f'  Actual No     {fm["tn"]:>10d}    {fm["fp"]:>10d}')
print(f'  Actual Yes    {fm["fn"]:>10d}    {fm["tp"]:>10d}')

# Visual confusion matrix
cm_data = [[fm['tn'], fm['fp']],
           [fm['fn'], fm['tp']]]
fig = go.Figure(go.Heatmap(
    z=cm_data,
    x=['Predicted: Not Successful','Predicted: Successful'],
    y=['Actual: Not Successful','Actual: Successful'],
    colorscale='Blues', text=cm_data, texttemplate='%{text}',
    showscale=True
))
fig.update_layout(
    title=f'Confusion Matrix -- Full Model (AUC={full_auc:.4f}, F1={fm["f1"]:.4f})',
    height=400)
fig.show()

# Precision-Recall tradeoff across thresholds
thresholds_pr = np.linspace(0.1, 0.9, 17)
pr_rows = []
for t in thresholds_pr:
    pred_t = (ew_proba >= t).astype(int)
    m = precision_recall_f1(y_d3_te, pred_t)
    pr_rows.append({'threshold': round(t,2), **m})
pr_df = pd.DataFrame(pr_rows)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pr_df['threshold'], y=pr_df['precision'],
    mode='lines+markers', name='Precision', line=dict(color='#00CC96', width=2)))
fig.add_trace(go.Scatter(
    x=pr_df['threshold'], y=pr_df['recall'],
    mode='lines+markers', name='Recall', line=dict(color='#EF553B', width=2)))
fig.add_trace(go.Scatter(
    x=pr_df['threshold'], y=pr_df['f1'],
    mode='lines+markers', name='F1', line=dict(color='#636EFA', width=2)))
fig.add_vline(x=0.3, line_dash='dash', line_color='orange',
    annotation_text='Used threshold (0.3)')
fig.update_layout(
    title='Early Warning -- Precision vs Recall Tradeoff by Threshold',
    xaxis_title='Decision Threshold',
    yaxis_title='Score', height=420)
fig.show()

print('\n  Threshold analysis for Early Warning:')
print(f'  {"Threshold":>10} {"Precision":>10} {"Recall":>10} {"F1":>8} {"TP":>6} {"FP":>6}')
print('  ' + '-'*54)
for _, r in pr_df[pr_df['threshold'].isin([0.2,0.3,0.4,0.5,0.6])].iterrows():
    print(f'  {r["threshold"]:>10.1f} {r["precision"]:>10.4f} '
          f'{r["recall"]:>10.4f} {r["f1"]:>8.4f} '
          f'{r["tp"]:>6} {r["fp"]:>6}')

print('\n  Summary:')
print(f'  Full Model    -- Accuracy={fm["accuracy"]:.4f}  Precision={fm["precision"]:.4f}  Recall={fm["recall"]:.4f}  F1={fm["f1"]:.4f}')
print(f'  Early Warning -- Accuracy={em["accuracy"]:.4f}  Precision={em["precision"]:.4f}  Recall={em["recall"]:.4f}  F1={em["f1"]:.4f}')
print(f'  Upgrade Model -- Accuracy={um["accuracy"]:.4f}  Precision={um["precision"]:.4f}  Recall={um["recall"]:.4f}  F1={um["f1"]:.4f}')