import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("=" * 62)
print("  STORY BEAT 16: HOW FAST DO WINNERS WIN?")
print("  Time-to-first-value. Network effect. Survival curve.")
print("=" * 62)

# -- 16A: Time to First Value
first_value = (
    df[df['is_production_event']==1]
    .groupby('person_id')['timestamp'].min()
    .reset_index().rename(columns={'timestamp':'first_value_ts'})
)
user_df2 = user_df2.merge(first_value, on='person_id', how='left')
user_df2['hours_to_first_value'] = (
    (user_df2['first_value_ts'] - user_df2['first_seen'])
    .dt.total_seconds() / 3600
)

def ttv_bucket(h):
    if pd.isna(h):  return '5_Never'
    elif h <= 1:    return '1_Under 1h'
    elif h <= 24:   return '2_Under 24h'
    elif h <= 72:   return '3_Under 72h'
    else:           return '4_After 72h'

user_df2['ttv_bucket'] = user_df2['hours_to_first_value'].apply(ttv_bucket)
ttv_analysis = user_df2.groupby('ttv_bucket').agg(
    users      = ('person_id',    'count'),
    successful = ('is_successful','sum')
).reset_index()
ttv_analysis['success_rate'] = (ttv_analysis['successful'] / ttv_analysis['users'] * 100).round(1)
ttv_analysis = ttv_analysis.sort_values('ttv_bucket')

print('\n=== TIME TO FIRST VALUE ===')
print(ttv_analysis.to_string(index=False))

fig = go.Figure(go.Bar(
    x=ttv_analysis['ttv_bucket'],
    y=ttv_analysis['success_rate'],
    marker_color=['#00CC96','#19D3F3','#FFA15A','#EF553B','#636EFA'],
    text=[f"{r}%  (n={n})" for r,n in zip(ttv_analysis['success_rate'], ttv_analysis['users'])],
    textposition='outside'
))
fig.update_layout(
    title='Time to First Value -- Faster Code Run = Higher Success',
    xaxis_title='Time to First Production Event',
    yaxis_title='Success Rate (%)', yaxis_range=[0,120], height=450)
fig.show()

# -- 16B: Network Effect
share_analysis = user_df2.groupby('ever_shared').agg(
    users         = ('person_id',            'count'),
    successful    = ('is_successful',         'sum'),
    upgrade_cands = ('is_upgrade_candidate', 'sum')
).reset_index()
share_analysis['success_rate'] = (share_analysis['successful'] / share_analysis['users'] * 100).round(1)
share_analysis['upgrade_rate'] = (share_analysis['upgrade_cands'] / share_analysis['users'] * 100).round(1)
share_analysis['label'] = share_analysis['ever_shared'].map({0:'Never Shared', 1:'Shared Canvas'})

print('\n=== NETWORK EFFECT: Canvas Share vs Success ===')
print(share_analysis[['label','users','success_rate','upgrade_rate']].to_string(index=False))

fig = go.Figure()
fig.add_trace(go.Bar(
    name='Success Rate %', x=share_analysis['label'], y=share_analysis['success_rate'],
    marker_color=['#EF553B','#00CC96'],
    text=share_analysis['success_rate'].astype(str)+'%', textposition='outside'))
fig.add_trace(go.Bar(
    name='Upgrade Rate %', x=share_analysis['label'], y=share_analysis['upgrade_rate'],
    marker_color=['#FFA15A','#636EFA'],
    text=share_analysis['upgrade_rate'].astype(str)+'%', textposition='outside'))
fig.update_layout(
    barmode='group',
    title='Network Effect -- Users Who Share Are More Successful',
    yaxis_title='Rate (%)', yaxis_range=[0,120], height=430)
fig.show()

# -- 16C: Survival Curve
user_first_week = df.groupby('person_id')['week'].min().reset_index().rename(columns={'week':'first_week'})
df_surv = df.merge(user_first_week, on='person_id', how='left')
df_surv['weeks_since_first'] = df_surv['week'] - df_surv['first_week']
total_users = df_surv['person_id'].nunique()

survival_weeks = []
for w in range(0, 13):
    active = df_surv[df_surv['weeks_since_first'] >= w]['person_id'].nunique()
    survival_weeks.append({
        'week': w,
        'active': active,
        'pct': round(active / total_users * 100, 1)
    })
survival_df = pd.DataFrame(survival_weeks)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=survival_df['week'], y=survival_df['pct'],
    mode='lines+markers', line=dict(color='#AB63FA', width=3),
    fill='tozeroy', fillcolor='rgba(171,99,250,0.1)'
))
fig.add_hline(y=50, line_dash='dash', line_color='orange', annotation_text='50% survival')
fig.add_hline(y=10, line_dash='dash', line_color='red',    annotation_text='10% survival')
fig.update_layout(
    title='Platform Survival Curve -- When Does Each Pct of Users Drop Off?',
    xaxis_title='Weeks Active Since Signup',
    yaxis_title='% of Users Still Active', height=430)
fig.show()

print('\nSurvival summary:')
for _, row in survival_df.iterrows():
    bar = '#' * int(row['pct'] / 5)
    print(f'  Week {int(row["week"]):2d}: {row["pct"]:5.1f}%  {bar}')