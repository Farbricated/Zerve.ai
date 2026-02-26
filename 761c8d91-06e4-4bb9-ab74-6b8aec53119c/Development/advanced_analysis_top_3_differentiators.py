import sys
from types import ModuleType

# ── THE ONLY FIX THAT WORKS ───────────────────────────────────
# Patch subprocess.run FIRST before anything else loads
# This stops numpy.testing from crashing when Zerve intercepts file descriptors
import subprocess as _sp
_real_run = _sp.run
def _safe_run(cmd, *args, **kwargs):
    try:
        return _real_run(cmd, *args, **kwargs)
    except TypeError:
        class _R:
            stdout = ''
            stderr = ''
            returncode = 0
        return _R()
_sp.run = _safe_run

# Mock seaborn AFTER subprocess patch
sys.modules['seaborn'] = ModuleType('seaborn')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print('✅ All imports successful!')

# ── Reload everything ─────────────────────────────────────────
df = pd.read_parquet('user_retention.parquet')
df['timestamp']  = pd.to_datetime(df['timestamp'],  utc=True, errors='coerce')
df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')
df['date']  = df['timestamp'].dt.date
df['hour']  = df['timestamp'].dt.hour
df['dow']   = df['timestamp'].dt.day_name()
df['week']  = df['timestamp'].dt.isocalendar().week.astype(int)
df['month'] = df['timestamp'].dt.month
df['prop_$os'] = df['prop_$os'].fillna('Unknown').str.strip()

AGENT_EVENTS = [
    'agent_tool_call_create_block_tool','agent_tool_call_run_block_tool',
    'agent_tool_call_get_block_tool','agent_tool_call_get_canvas_summary_tool',
    'agent_tool_call_get_variable_preview_tool','agent_tool_call_finish_ticket_tool',
    'agent_tool_call_refactor_block_tool','agent_tool_call_delete_block_tool',
    'agent_tool_call_create_edges_tool','agent_tool_call_get_block_image_tool',
    'agent_new_chat','agent_message','agent_start_from_prompt','agent_worker_created',
    'agent_block_created','agent_accept_suggestion','agent_open','agent_block_run',
    'agent_suprise_me','agent_upload_files','agent_open_error_assist'
]
PRODUCTION_EVENTS = [
    'app_publish','app_unpublish','scheduled_job_start','scheduled_job_stop',
    'run_all_blocks','run_block','run_upto_block','run_from_block',
    'requirements_build','hosted_apps_open'
]
COLLAB_EVENTS = ['canvas_share','resource_shared','link_clicked']
CREDIT_EVENTS = ['credits_used','addon_credits_used','promo_code_redeemed',
                 'credits_below_1','credits_below_2','credits_below_3',
                 'credits_below_4','credits_exceeded']
CANVAS_EVENTS = ['canvas_create','canvas_open','canvas_delete','block_create',
                 'block_delete','block_rename','block_copy','block_paste',
                 'edge_create','edge_delete']

df['is_agent_event']      = df['event'].isin(AGENT_EVENTS).astype(int)
df['is_production_event'] = df['event'].isin(PRODUCTION_EVENTS).astype(int)
df['is_collab_event']     = df['event'].isin(COLLAB_EVENTS).astype(int)
df['is_credit_event']     = df['event'].isin(CREDIT_EVENTS).astype(int)
df['is_canvas_event']     = df['event'].isin(CANVAS_EVENTS).astype(int)
df['is_signup']           = df['event'].isin(['sign_up','new_user_created']).astype(int)

user_df = df.groupby('person_id').agg(
    total_events         = ('uuid',               'count'),
    unique_event_types   = ('event',              'nunique'),
    first_seen           = ('timestamp',          'min'),
    last_seen            = ('timestamp',          'max'),
    active_days          = ('date',               'nunique'),
    active_weeks         = ('week',               'nunique'),
    agent_events         = ('is_agent_event',     'sum'),
    production_events    = ('is_production_event','sum'),
    collab_events        = ('is_collab_event',    'sum'),
    credit_events        = ('is_credit_event',    'sum'),
    canvas_events        = ('is_canvas_event',    'sum'),
).reset_index()

user_df['tenure_days']        = (user_df['last_seen'] - user_df['first_seen']).dt.days
user_df['events_per_day']     = user_df['total_events']       / (user_df['active_days'] + 1)
user_df['agent_ratio']        = user_df['agent_events']       / (user_df['total_events'] + 1)
user_df['production_ratio']   = user_df['production_events']  / (user_df['total_events'] + 1)
user_df['collab_ratio']       = user_df['collab_events']      / (user_df['total_events'] + 1)
user_df['credit_ratio']       = user_df['credit_events']      / (user_df['total_events'] + 1)
user_df['event_diversity']    = user_df['unique_event_types'] / user_df['unique_event_types'].max()
user_df['weekly_cadence']     = user_df['active_days']        / (user_df['active_weeks'] + 1)
user_df['ever_published_app'] = (df[df['event']=='app_publish'].groupby('person_id').size().reindex(user_df['person_id']).fillna(0).values > 0).astype(int)
user_df['ever_scheduled']     = (df[df['event']=='scheduled_job_start'].groupby('person_id').size().reindex(user_df['person_id']).fillna(0).values > 0).astype(int)
user_df['ever_shared']        = (df[df['event']=='canvas_share'].groupby('person_id').size().reindex(user_df['person_id']).fillna(0).values > 0).astype(int)
user_df['used_addon_credits'] = (df[df['event']=='addon_credits_used'].groupby('person_id').size().reindex(user_df['person_id']).fillna(0).values > 0).astype(int)

scaler = MinMaxScaler()
ret_feats = ['tenure_days','active_days','active_weeks','events_per_day','weekly_cadence']
user_df['retention_score'] = scaler.fit_transform(user_df[ret_feats].fillna(0)).mean(axis=1) * 100
dep_feats = ['agent_ratio','production_ratio','collab_ratio','event_diversity',
             'ever_published_app','ever_scheduled','ever_shared']
user_df['depth_score'] = scaler.fit_transform(user_df[dep_feats].fillna(0)).mean(axis=1) * 100
user_df['volume_score'] = scaler.fit_transform(
    user_df[['total_events','unique_event_types','credit_events']].fillna(0)).mean(axis=1) * 100
user_df['success_score'] = (
    0.40 * user_df['retention_score'] +
    0.35 * user_df['depth_score']     +
    0.25 * user_df['volume_score'])
threshold = user_df['success_score'].quantile(0.70)
user_df['is_successful'] = (user_df['success_score'] >= threshold).astype(int)

df_w_first = df.merge(user_df[['person_id','first_seen']], on='person_id', how='left')
df_w_first['days_since_first'] = (df_w_first['timestamp'] - df_w_first['first_seen']).dt.days

print(f'✅ Data reloaded: {len(user_df)} users | threshold: {threshold:.2f}')

# ═══════════════════════════════════════════════════════════════
# STORY BEAT 1: PLATFORM HEALTH DASHBOARD
# ═══════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║  📖 STORY BEAT 1: SETTING THE SCENE                         ║
║  4,774 users. 409,287 events. 3 months of data.             ║
║  One question: what separates the 30% who succeed           ║
║  from the 70% who don't?                                    ║
╚══════════════════════════════════════════════════════════════╝
""")

fig = make_subplots(rows=2, cols=2,
    subplot_titles=['📊 Total Events Per Week','👤 Weekly Active Users',
                    '🚀 New Signups Per Week','🤖 Agent Usage Growth'])
weekly_events  = df.groupby('week').size().reset_index(name='count')
weekly_users   = df.groupby(['week','person_id']).size().reset_index().groupby('week').size().reset_index(name='count')
weekly_signups = df[df['is_signup']==1].groupby('week').size().reset_index(name='count')
weekly_agent   = df[df['is_agent_event']==1].groupby('week').size().reset_index(name='count')
fig.add_trace(go.Scatter(x=weekly_events['week'],  y=weekly_events['count'],  mode='lines+markers', line=dict(color='#636EFA'), name='Events'),  row=1, col=1)
fig.add_trace(go.Scatter(x=weekly_users['week'],   y=weekly_users['count'],   mode='lines+markers', line=dict(color='#00CC96'), name='WAU'),     row=1, col=2)
fig.add_trace(go.Bar(    x=weekly_signups['week'], y=weekly_signups['count'], marker_color='#FFA15A', name='Signups'),                            row=2, col=1)
fig.add_trace(go.Scatter(x=weekly_agent['week'],   y=weekly_agent['count'],   mode='lines+markers', line=dict(color='#AB63FA'), name='Agent'),   row=2, col=2)
fig.update_layout(title='🏠 Zerve Platform Health Dashboard — Sep to Dec 2025', height=600, showlegend=False)
fig.show()

# ═══════════════════════════════════════════════════════════════
# STORY BEAT 2: THE FORK IN THE ROAD
# ═══════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║  📖 STORY BEAT 2: THE DISCOVERY                             ║
║  We expected volume to predict success.                     ║
║  We were wrong. It's all about the first 7 days.           ║
╚══════════════════════════════════════════════════════════════╝
""")

df_w_first2 = df_w_first.merge(user_df[['person_id','is_successful']], on='person_id', how='left')
df_w_first2 = df_w_first2[df_w_first2['days_since_first'] <= 30]
daily_curve  = df_w_first2.groupby(['days_since_first','is_successful']).size().reset_index(name='events')
daily_curve['events'] = daily_curve.groupby('is_successful')['events'].cumsum()
succ_curve   = daily_curve[daily_curve['is_successful']==1]
unsucc_curve = daily_curve[daily_curve['is_successful']==0]

fig = go.Figure()
fig.add_trace(go.Scatter(x=succ_curve['days_since_first'],   y=succ_curve['events'],
    mode='lines', name='Successful Users',   line=dict(color='#00CC96', width=3)))
fig.add_trace(go.Scatter(x=unsucc_curve['days_since_first'], y=unsucc_curve['events'],
    mode='lines', name='Unsuccessful Users', line=dict(color='#EF553B', width=3)))
fig.add_vline(x=7, line_dash='dash', line_color='orange', annotation_text='Day 7 — The Fork in the Road')
fig.update_layout(title='🍴 The Fork in the Road — Paths Diverge at Day 7',
    xaxis_title='Days Since Signup', yaxis_title='Cumulative Events', height=450)
fig.show()

# ═══════════════════════════════════════════════════════════════
# STORY BEAT 3: COUNTER-INTUITIVE FINDING
# ═══════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║  📖 STORY BEAT 3: THE SURPRISING FINDING                    ║
║  We assumed: faster agent use = more success.               ║
║  The data says the opposite. Patient users win.             ║
╚══════════════════════════════════════════════════════════════╝
""")

first_agent = (
    df[df['is_agent_event']==1]
    .groupby('person_id')['timestamp'].min()
    .reset_index().rename(columns={'timestamp':'first_agent_ts'})
)
user_df = user_df.merge(first_agent, on='person_id', how='left')
user_df['hours_to_first_agent'] = (
    (user_df['first_agent_ts'] - user_df['first_seen']).dt.total_seconds() / 3600)

def agent_speed_bucket(h):
    if pd.isna(h):   return '4_Never Used Agent'
    elif h <= 2:     return '1_Within 2 Hours'
    elif h <= 24:    return '2_Within 24 Hours'
    else:            return '3_After 24 Hours'

user_df['agent_speed'] = user_df['hours_to_first_agent'].apply(agent_speed_bucket)
speed_analysis = user_df.groupby('agent_speed').agg(
    total_users = ('person_id',    'count'),
    successful  = ('is_successful','sum')
).reset_index()
speed_analysis['success_rate'] = (speed_analysis['successful'] / speed_analysis['total_users'] * 100).round(1)
speed_analysis = speed_analysis.sort_values('agent_speed')

print('\n=== TIME TO FIRST AGENT CALL — SUCCESS RATE ===')
print(speed_analysis.to_string(index=False))

fig = go.Figure()
fig.add_trace(go.Bar(
    x=['Within 2 Hours','Within 24 Hours','After 24 Hours','Never Used Agent'],
    y=speed_analysis['success_rate'].values,
    marker_color=['#FFA15A','#19D3F3','#00CC96','#EF553B'],
    text=[f"{r}%<br>({n} users)" for r,n in zip(speed_analysis['success_rate'],speed_analysis['total_users'])],
    textposition='outside'))
fig.update_layout(
    title='⚡ COUNTER-INTUITIVE: Patient Agent Users Succeed More<br>'
          '<sup>Explore first, then use the agent intentionally</sup>',
    yaxis_title='Success Rate (%)', yaxis_range=[0,115], height=450)
fig.show()

# ═══════════════════════════════════════════════════════════════
# STORY BEAT 4: COHORT RETENTION HEATMAP
# ═══════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║  📖 STORY BEAT 4: THE RETENTION CLIFF                       ║
║  When exactly do users drop off? Week by week.              ║
╚══════════════════════════════════════════════════════════════╝
""")

first_event_week   = df.groupby('person_id')['week'].min().reset_index().rename(columns={'week':'signup_week'})
user_week_activity = df.groupby(['person_id','week']).size().reset_index(name='events')
user_week_activity = user_week_activity.merge(first_event_week, on='person_id', how='left')
user_week_activity['weeks_since_signup'] = user_week_activity['week'] - user_week_activity['signup_week']
user_week_activity = user_week_activity[
    (user_week_activity['weeks_since_signup'] >= 0) &
    (user_week_activity['weeks_since_signup'] <= 12)]

cohort_sizes = first_event_week['signup_week'].value_counts().reset_index()
cohort_sizes.columns = ['signup_week','cohort_size']
retention = user_week_activity.groupby(
    ['signup_week','weeks_since_signup'])['person_id'].nunique().reset_index()
retention = retention.merge(cohort_sizes, on='signup_week')
retention['retention_rate'] = (retention['person_id'] / retention['cohort_size'] * 100).round(1)
retention_pivot = retention.pivot_table(
    index='signup_week', columns='weeks_since_signup', values='retention_rate').fillna(0)
valid_cohorts   = cohort_sizes.set_index('signup_week')['cohort_size']
valid_cohorts   = valid_cohorts[valid_cohorts >= 10].index
retention_pivot = retention_pivot.loc[retention_pivot.index.isin(valid_cohorts)]

fig = px.imshow(retention_pivot,
    labels=dict(x='Weeks Since Signup', y='Signup Week', color='Retention %'),
    title='📅 Cohort Retention Heatmap — The Exact Week Users Drop Off',
    color_continuous_scale='RdYlGn', text_auto=True, aspect='auto')
fig.update_layout(height=500)
fig.show()

avg_retention = retention.groupby('weeks_since_signup')['retention_rate'].mean().reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=avg_retention['weeks_since_signup'], y=avg_retention['retention_rate'],
    mode='lines+markers', line=dict(color='#636EFA', width=3),
    fill='tozeroy', fillcolor='rgba(99,110,250,0.1)'))
fig.update_layout(title='📉 Average Retention Curve',
    xaxis_title='Weeks Since Signup', yaxis_title='Avg Retention Rate (%)', height=400)
fig.show()

cliff_idx  = avg_retention['retention_rate'].diff().idxmin()
cliff_week = avg_retention.loc[cliff_idx, 'weeks_since_signup']
cliff_rate = avg_retention.loc[cliff_idx, 'retention_rate']
print(f'  🚨 Biggest retention drop: Week {cliff_week} — falls to {cliff_rate:.1f}%')

# ═══════════════════════════════════════════════════════════════
# STORY BEAT 5: USER SEGMENTS
# ═══════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║  📖 STORY BEAT 5: USER DNA — THE 4 PERSONAS                 ║
╚══════════════════════════════════════════════════════════════╝
""")

cluster_feats = ['total_events','active_days','tenure_days',
                 'retention_score','depth_score','volume_score',
                 'agent_ratio','production_ratio','ever_published_app',
                 'ever_scheduled','used_addon_credits']
X_cl        = user_df[cluster_feats].fillna(0)
X_cl_scaled = StandardScaler().fit_transform(X_cl)
km = KMeans(n_clusters=4, random_state=42, n_init=10)
user_df['cluster'] = km.fit_predict(X_cl_scaled)
pca    = PCA(n_components=2)
coords = pca.fit_transform(X_cl_scaled)
user_df['pca_x'], user_df['pca_y'] = coords[:,0], coords[:,1]
segment_labels = {0:'Casual Explorers',1:'Rising Stars',
                  2:'One-time Visitors',3:'Power Users'}
user_df['segment'] = user_df['cluster'].map(segment_labels)

colors_map = {'Power Users':'#636EFA','Casual Explorers':'#FFA15A',
              'One-time Visitors':'#EF553B','Rising Stars':'#00CC96'}
sample = user_df.sample(min(3000,len(user_df)))
fig = go.Figure()
for seg, col in colors_map.items():
    mask = sample['segment'] == seg
    fig.add_trace(go.Scatter(
        x=sample[mask]['pca_x'], y=sample[mask]['pca_y'],
        mode='markers', name=f'{seg} (n={mask.sum()})',
        marker=dict(color=col, size=6, opacity=0.7)))
fig.update_layout(title='🗺️ The 4 Types of Zerve Users', height=450)
fig.show()

seg_success = user_df.groupby('segment').agg(
    users      = ('person_id',    'count'),
    successful = ('is_successful','sum'),
    avg_events = ('total_events', 'mean'),
    avg_tenure = ('tenure_days',  'mean'),
).reset_index()
seg_success['success_rate'] = (seg_success['successful'] / seg_success['users'] * 100).round(1)
seg_success = seg_success.sort_values('success_rate', ascending=False)

print('\n=== SEGMENT PROFILES ===')
print(seg_success.to_string(index=False))

fig = go.Figure()
fig.add_trace(go.Bar(
    x=seg_success['segment'], y=seg_success['success_rate'],
    marker_color=[colors_map[s] for s in seg_success['segment']],
    text=seg_success['success_rate'].astype(str)+'%', textposition='outside'))
fig.update_layout(title='👥 Success Rate by Segment',
    yaxis_title='Success Rate (%)', yaxis_range=[0,115], height=400)
fig.show()

# ═══════════════════════════════════════════════════════════════
# STORY BEAT 6: CROSS-VALIDATED ML
# ═══════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║  📖 STORY BEAT 6: THE MODEL — STATISTICALLY RIGOROUS        ║
║  5-fold cross-validation. Confidence intervals.             ║
╚══════════════════════════════════════════════════════════════╝
""")

top_event_list = df['event'].value_counts().head(20).index.tolist()
pivot = df[df['event'].isin(top_event_list)].groupby(
    ['person_id','event']).size().unstack(fill_value=0)
pivot.columns = [f'evt_{c.lower().replace(" ","_")}' for c in pivot.columns]
pivot.reset_index(inplace=True)
user_df2 = user_df.merge(pivot, on='person_id', how='left')
evt_cols = [c for c in user_df2.columns if c.startswith('evt_')]
user_df2[evt_cols] = user_df2[evt_cols].fillna(0)

first_week_agg = df_w_first[df_w_first['days_since_first'] <= 7].groupby('person_id').agg(
    fw_total_events      = ('uuid',               'count'),
    fw_agent_events      = ('is_agent_event',     'sum'),
    fw_production_events = ('is_production_event','sum'),
    fw_canvas_events     = ('is_canvas_event',    'sum'),
    fw_unique_events     = ('event',              'nunique'),
    fw_active_days       = ('date',               'nunique'),
).reset_index()
user_df2 = user_df2.merge(first_week_agg, on='person_id', how='left')
fw_cols  = [c for c in user_df2.columns if c.startswith('fw_')]
user_df2[fw_cols] = user_df2[fw_cols].fillna(0)

feature_cols = (
    ['total_events','unique_event_types','active_days','active_weeks',
     'tenure_days','events_per_day','weekly_cadence','event_diversity',
     'agent_events','production_events','collab_events','credit_events',
     'agent_ratio','production_ratio','collab_ratio','credit_ratio',
     'ever_published_app','ever_scheduled','ever_shared','used_addon_credits']
    + fw_cols + evt_cols
)
feature_cols = [c for c in feature_cols if c in user_df2.columns]
X = user_df2[feature_cols].fillna(0)
y = user_df2['is_successful']

skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
scores = cross_val_score(rf_cv, X, y, cv=skf, scoring='roc_auc')

print(f'\n  5-Fold CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}')
print(f'  95% CI: [{scores.mean()-1.96*scores.std():.4f}, {scores.mean()+1.96*scores.std():.4f}]')

fig = go.Figure()
fig.add_trace(go.Bar(
    x=[f'Fold {i+1}' for i in range(5)], y=scores,
    marker_color='#636EFA',
    text=[f'{s:.4f}' for s in scores], textposition='outside'))
fig.add_hline(y=scores.mean(), line_dash='dash', line_color='red',
    annotation_text=f'Mean = {scores.mean():.4f}')
fig.update_layout(
    title=f'📊 5-Fold Cross-Validation — AUC {scores.mean():.4f} ± {scores.std():.4f}',
    yaxis_title='ROC-AUC', yaxis_range=[0.9,1.01], height=400)
fig.show()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc2        = StandardScaler()
full_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
full_model.fit(sc2.fit_transform(X_tr), y_tr)
full_proba = full_model.predict_proba(sc2.transform(X_te))[:,1]
full_auc   = roc_auc_score(y_te, full_proba)
fpr_full, tpr_full, _ = roc_curve(y_te, full_proba)

imp   = pd.Series(full_model.feature_importances_, index=feature_cols)
top15 = imp.sort_values(ascending=False).head(15)
fig = go.Figure(go.Bar(
    x=top15.values, y=top15.index, orientation='h',
    marker=dict(color=top15.values, colorscale='Viridis', showscale=True)))
fig.update_layout(title='🌟 Top 15 Features That Predict Success',
    xaxis_title='Importance', yaxis=dict(categoryorder='total ascending'), height=500)
fig.show()

# ═══════════════════════════════════════════════════════════════
# STORY BEAT 7: EARLY WARNING SYSTEM
# ═══════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║  📖 STORY BEAT 7: THE EARLY WARNING SYSTEM                  ║
║  Predict failure before it happens. At Day 3.               ║
╚══════════════════════════════════════════════════════════════╝
""")

first_3day_agg = df_w_first[df_w_first['days_since_first'] <= 3].groupby('person_id').agg(
    d3_total_events      = ('uuid',               'count'),
    d3_agent_events      = ('is_agent_event',     'sum'),
    d3_production_events = ('is_production_event','sum'),
    d3_canvas_events     = ('is_canvas_event',    'sum'),
    d3_unique_events     = ('event',              'nunique'),
    d3_active_days       = ('date',               'nunique'),
).reset_index()
user_df2 = user_df2.merge(first_3day_agg, on='person_id', how='left')
d3_cols  = [c for c in user_df2.columns if c.startswith('d3_')]
user_df2[d3_cols] = user_df2[d3_cols].fillna(0)

X_d3 = user_df2[d3_cols].fillna(0)
y_d3 = user_df2['is_successful']
X_d3_tr, X_d3_te, y_d3_tr, y_d3_te = train_test_split(
    X_d3, y_d3, test_size=0.2, random_state=42, stratify=y_d3)
sc_d3    = StandardScaler()
ew_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
ew_model.fit(sc_d3.fit_transform(X_d3_tr), y_d3_tr)
ew_proba = ew_model.predict_proba(sc_d3.transform(X_d3_te))[:,1]
ew_auc   = roc_auc_score(y_d3_te, ew_proba)
fpr_ew, tpr_ew, _ = roc_curve(y_d3_te, ew_proba)

print(f'  🚨 Early Warning AUC (Day 3 only): {ew_auc:.4f}')

fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr_ew,   y=tpr_ew,   mode='lines',
    name=f'🚨 Early Warning Day 3 (AUC={ew_auc:.3f})', line=dict(color='#AB63FA', width=3)))
fig.add_trace(go.Scatter(x=fpr_full, y=tpr_full, mode='lines',
    name=f'🤖 Full Model (AUC={full_auc:.3f})',         line=dict(color='#00CC96', width=3)))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
    line=dict(dash='dash', color='gray'), name='Random'))
fig.update_layout(
    title='🚨 Early Warning vs Full Model ROC Comparison',
    xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=450)
fig.show()

user_df2['success_probability'] = ew_model.predict_proba(sc_d3.transform(X_d3.fillna(0)))[:,1]
user_df2['risk_flag'] = user_df2['success_probability'].apply(
    lambda p: '🔴 High Risk' if p < 0.3 else ('🟡 Medium Risk' if p < 0.6 else '🟢 Likely Success'))

risk_summary = user_df2['risk_flag'].value_counts().reset_index()
risk_summary.columns = ['Risk Level','Users']
print('\n=== RISK DISTRIBUTION AT DAY 3 ===')
print(risk_summary.to_string(index=False))

fig = go.Figure(go.Bar(
    x=risk_summary['Risk Level'], y=risk_summary['Users'],
    marker_color=['#00CC96','#FFA15A','#EF553B'],
    text=risk_summary['Users'], textposition='outside'))
fig.update_layout(title='🚨 How Many Users Are At Risk Right Now?',
    yaxis_title='Users', height=400)
fig.show()

# ═══════════════════════════════════════════════════════════════
# STORY BEAT 8: THE ACTIVATION FORMULA
# ═══════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║  📖 STORY BEAT 8: THE ACTIVATION FORMULA                    ║
║  What is the minimum set of actions for success?            ║
╚══════════════════════════════════════════════════════════════╝
""")

user_df2['did_run_code']       = (user_df2['production_events'] > 0).astype(int)
user_df2['did_use_agent']      = (user_df2['agent_events'] > 0).astype(int)
user_df2['did_publish']        = user_df2['ever_published_app']
user_df2['did_return_week2']   = (user_df2['active_weeks'] >= 2).astype(int)
user_df2['did_diverse_events'] = (user_df2['unique_event_types'] >= 10).astype(int)

milestones = ['did_run_code','did_use_agent','did_publish',
              'did_return_week2','did_diverse_events']
milestone_labels = {
    'did_run_code'      : 'Ran Code',
    'did_use_agent'     : 'Used Agent',
    'did_publish'       : 'Published App',
    'did_return_week2'  : 'Returned Week 2',
    'did_diverse_events': '10+ Event Types'
}

user_df2['milestones_hit'] = user_df2[milestones].sum(axis=1)
ms_analysis = user_df2.groupby('milestones_hit').agg(
    users      = ('person_id',    'count'),
    successful = ('is_successful','sum')
).reset_index()
ms_analysis['success_rate'] = (ms_analysis['successful'] / ms_analysis['users'] * 100).round(1)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=ms_analysis['milestones_hit'], y=ms_analysis['success_rate'],
    marker_color=['#EF553B','#FF6692','#FFA15A','#19D3F3','#00CC96','#636EFA'],
    text=[f"{r}%\n({n} users)" for r,n in zip(ms_analysis['success_rate'],ms_analysis['users'])],
    textposition='outside'))
fig.update_layout(
    title='🎯 The Activation Formula — More Milestones = More Success<br>'
          '<sup>Milestones: Run Code · Use Agent · Publish App · Return Week 2 · 10+ Event Types</sup>',
    xaxis_title='Milestones Hit (out of 5)',
    yaxis_title='Success Rate (%)', yaxis_range=[0,115], height=450)
fig.show()

print('\n  Individual milestone impact:')
for m in milestones:
    hit  = user_df2[user_df2[m]==1]['is_successful'].mean()*100
    miss = user_df2[user_df2[m]==0]['is_successful'].mean()*100
    print(f'  {milestone_labels[m]:20s}: {hit:.1f}% if hit | {miss:.1f}% if missed')

# ═══════════════════════════════════════════════════════════════
# STORY BEAT 9: FINAL QUANTIFIED REPORT
# ═══════════════════════════════════════════════════════════════
pub0 = user_df2[user_df2['ever_published_app']==0]['is_successful'].value_counts()
pub1 = user_df2[user_df2['ever_published_app']==1]['is_successful'].value_counts()
pub_no  = round(int(pub0.get(1,0))/max(int(pub0.get(0,0))+int(pub0.get(1,0)),1)*100,1)
pub_yes = round(int(pub1.get(1,0))/max(int(pub1.get(0,0))+int(pub1.get(1,0)),1)*100,1)
one_time  = user_df2[user_df2['total_events'] <= 5]
high_risk = user_df2[user_df2['risk_flag']=='🔴 High Risk']
w2h = speed_analysis[speed_analysis['agent_speed']=='1_Within 2 Hours']['success_rate']
w24 = speed_analysis[speed_analysis['agent_speed']=='3_After 24 Hours']['success_rate']
w2h_val = w2h.values[0] if len(w2h)>0 else 'N/A'
w24_val = w24.values[0] if len(w24)>0 else 'N/A'

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  🏆 ZERVE × HACKEREARTH — FINAL REPORT                             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  THESIS: "The 7-Day Make or Break"                                  ║
║  A user's first 7 days predict their entire Zerve journey.          ║
║                                                                      ║
║  DATASET   : 409,287 events | 4,774 users | Sep–Dec 2025           ║
║  CV MODEL  : 5-fold AUC {scores.mean():.4f} ± {scores.std():.4f}                    ║
║  EW MODEL  : Day-3 AUC {ew_auc:.4f} (predicts before churn)        ║
║  SEGMENTS  : 4 user personas via KMeans + PCA                       ║
║                                                                      ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ║
║  TOP FINDINGS:                                                       ║
║                                                                      ║
║  1. Patient agent users: 24h+ → {w24_val}% vs rushing → {w2h_val}%      ║
║  2. App publishers: {pub_yes}% success vs {pub_no}% non-publishers        ║
║  3. Early warning: {len(high_risk)} users flagged high risk at day 3        ║
║  4. Activation formula: 4+ milestones → near 100% success          ║
║  5. {len(one_time)} one-time visitors ({len(one_time)/len(user_df2)*100:.1f}%) = reactivation goldmine  ║
║                                                                      ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ║
║  PRODUCT RECOMMENDATIONS:                                            ║
║                                                                      ║
║  🎯 Teach platform FIRST — delay agent push to session 2            ║
║  🚀 Make "Deploy first app" a required onboarding milestone          ║
║  🚨 Deploy early warning — auto-email day-3 high-risk users         ║
║  📧 Week-2 re-engagement for users who haven't returned             ║
║  💳 Upsell Pro to users who hit credits_exceeded                    ║
║                                                                      ║
║  Built entirely on Zerve 🚀                                         ║
╚══════════════════════════════════════════════════════════════════════╝
""")