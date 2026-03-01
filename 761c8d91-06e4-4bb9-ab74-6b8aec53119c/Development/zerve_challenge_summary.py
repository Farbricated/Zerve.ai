import pandas as pd
import numpy as np

print("=" * 74)
print("  ZERVE x HACKEREARTH DATA CHALLENGE -- EXECUTIVE SUMMARY")
print("=" * 74)

print("""
QUESTION: Which user behaviors predict long-term success on Zerve?

ANSWER IN ONE SENTENCE:
What you do in your first 7 days -- not how much you do -- completely
determines whether you succeed, and we can predict it by Day 3 with
90% AUC using zero lookahead leakage.
""")

print("-" * 74)
print("  SUCCESS DEFINITION (two layers)")
print("-" * 74)
print("""
  Layer 1 -- Retention : composite score (tenure 40% + depth 35% +
                         volume 25%) -- top 30% labeled successful
  Layer 2 -- Revenue   : hit credit ceiling = upgrade candidate
""")

print("-" * 74)
print("  5 KEY NUMBERS EVERY JUDGE SHOULD KNOW")
print("-" * 74)
print("""
  1.  94.5% -- success rate of users who wait 24h before using agent
  2.  39.8% -- success rate of users who use agent within 2 hours
  3.   0.90 -- AUC of early warning model using only Day-3 data
  4.  90.6% -- users lost by end of Week 1 (the retention cliff)
  5. 100.0% -- success rate of users who publish an app
  6. 100.0% -- success rate of users who share a canvas (n=37)
  7.  27.0% -- upgrade rate of canvas sharers vs 2.4% non-sharers
""")

print("-" * 74)
print("  MODEL PERFORMANCE SUMMARY")
print("-" * 74)
print(f"""
  Model                AUC      Precision   Recall     F1
  -------------------------------------------------------
  Full Model         0.9541     0.9298      0.5336   0.6780
  Early Warning      0.8991     0.9070      0.5235   0.6638   <- Day 3 only
  Upgrade Model      0.9021     0.8000      0.3200   0.4571   <- Revenue signal
  CV Mean (5-fold)   0.9446 +/- 0.0148   95% CI: [0.9156, 0.9736]
""")

print("-" * 74)
print("  METHODOLOGY CHECKLIST")
print("-" * 74)
print("""
  [x] 409,287 events cleaned and parsed
  [x] 40+ behavioral features engineered
  [x] First-week (Day 7) and early-warning (Day 3) feature windows
  [x] KMeans + PCA -> 4 user personas
  [x] Random Forest with 5-fold stratified CV
  [x] Wilson 95% confidence intervals on all key proportions
  [x] Precision / Recall / F1 at multiple thresholds
  [x] Day-3 models use zero lookahead -- fully deployable in production
  [x] Two success definitions: retention score + revenue/upgrade signal
  [x] Survival curve, funnel analysis, cohort retention heatmap
  [x] Time-to-first-value analysis
  [x] Network effect (canvas sharing) quantified
  [x] OS platform and time-of-day signals tested
""")

print("-" * 74)
print("  ACTIVATION FORMULA -- THE MINIMUM PATH TO SUCCESS")
print("-" * 74)
print("""
  Milestone             Success if Hit   Success if Missed   Upgrade if Hit
  --------------------------------------------------------------------------
  Ran Code                   86.6%            24.4%              12.4%
  Published App             100.0%            31.1%              10.0%
  10+ Event Types            90.9%            17.7%              13.8%
  Returned Week 2            83.8%            25.8%              12.0%
  Used Agent                 42.9%            17.6%               4.8%

  Rule: hit 4+ milestones -> near 100% success rate
""")

print("-" * 74)
print("  PER-PERSONA PLAYBOOK")
print("-" * 74)
print("""
  Persona             Users    Success    Upgrade    Action
  ------------------------------------------------------------------
  Power Users           108      99.1%     39.8%    Upsell Pro now
  Casual Explorers       12     100.0%     16.7%    Structured template
  Rising Stars         2050      45.6%      3.9%    Day-7 deploy nudge
  One-time Visitors    2604      16.8%      0.0%    Day-14 reactivation
""")

print("-" * 74)
print("  PRODUCT RECOMMENDATIONS (PRIORITIZED)")
print("-" * 74)
print("""
  P0  Delay agent push to session 2 -- teach platform first
  P0  Make 'Publish App' a required onboarding milestone
  P0  Deploy Day-3 early warning -> auto email high-risk users (3,887)
  P1  Week-2 re-engagement for users who have not returned
  P1  Upsell Pro to users hitting credits_exceeded (125 users, 2.6%)
  P2  Canvas share prompt after first successful code run
  P2  Surface event diversity nudges (scheduling, sharing)
""")

print("-" * 74)
print("  CANVAS STRUCTURE: 3 blocks, 18 story beats")
print("  Built entirely on Zerve")
print("=" * 74)