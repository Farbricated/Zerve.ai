# ═══════════════════════════════════════════════════════════════
# STORY BEAT 0: EXECUTIVE SUMMARY — READ THIS FIRST
# ═══════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  🏆 ZERVE × HACKEREARTH DATA CHALLENGE — EXECUTIVE SUMMARY             ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  QUESTION: Which user behaviors predict long-term success on Zerve?     ║
║                                                                          ║
║  ANSWER IN ONE SENTENCE:                                                 ║
║  What you do in your first 7 days — not how much you do — completely    ║
║  determines whether you succeed, and we can predict it by Day 3.        ║
║                                                                          ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ║
║  SUCCESS DEFINITION (two layers):                                        ║
║  Layer 1 — Retention: composite score (tenure 40% + depth 35% +        ║
║            volume 25%) → top 30% = successful                           ║
║  Layer 2 — Revenue: hit credit ceiling = upgrade candidate              ║
║                                                                          ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ║
║  5 KEY NUMBERS EVERY JUDGE SHOULD KNOW:                                 ║
║                                                                          ║
║  1.  94.5% — success rate of users who wait 24h before using agent     ║
║  2.  39.8% — success rate of users who use agent within 2 hours        ║
║  3.   0.90 — AUC of early warning model using only Day-3 data          ║
║  4.  91.1% — users lost by end of Week 1 (the retention cliff)         ║
║  5. 100.0% — success rate of users who publish an app                  ║
║                                                                          ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ║
║  METHODOLOGY:                                                            ║
║  ✅ 409,287 events cleaned and parsed                                   ║
║  ✅ 40+ behavioral features engineered                                  ║
║  ✅ KMeans + PCA → 4 user personas                                      ║
║  ✅ Random Forest with 5-fold CV → AUC 0.9446 ± 0.0148                ║
║  ✅ Day-3 early warning model → AUC 0.8991 (no lookahead leakage)      ║
║  ✅ Day-3 upgrade/revenue model → AUC 0.9021                           ║
║  ✅ Wilson 95% confidence intervals on all key proportions             ║
║  ✅ Precision / Recall / F1 at multiple thresholds                     ║
║                                                                          ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ║
║  STRUCTURE: 18 story beats across 3 blocks in this Zerve canvas        ║
║  Built entirely on Zerve 🚀                                             ║
╚══════════════════════════════════════════════════════════════════════════╝
""")