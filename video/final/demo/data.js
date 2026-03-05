window.DEMO_DATA = {
  "generated_at_utc": "2026-03-03T20:34:07+00:00",
  "sources": {
    "gate": "video/assets/data/xlf_sync_gate.csv",
    "sg_budget": "video/assets/data/sg_controlled_budget.csv",
    "with_no": "video/assets/data/xlf_with_vs_no.csv",
    "main_conclusion": "video/assets/data/xlf_main_conclusion.csv",
    "headline": "video/assets/data/xlf_headline.csv",
    "xlf_trend": "yfinance:XLF (2019-01 to 2026-01, monthly normalized close)"
  },
  "gate_rows": [
    {
      "sequence": 1,
      "algo": "q_learning",
      "algo_label": "Q-learning",
      "reward_delta": 0.4562,
      "return_delta": 0.6199,
      "cvar_delta": 0.0183,
      "mdd_delta": -0.1709,
      "check_primary": true,
      "check_secondary": true,
      "check_cvar": false,
      "check_mdd": false,
      "check_seed_consistency": true,
      "tuned_lambda_cvar": 0.02,
      "promoted": false,
      "decision": "REJECTED",
      "decision_reason": "Largest utility uplift, but CVaR and MDD checks fail."
    },
    {
      "sequence": 2,
      "algo": "sg_sarsa",
      "algo_label": "SG-Sarsa",
      "reward_delta": 0.0374,
      "return_delta": 0.065,
      "cvar_delta": 0.0053,
      "mdd_delta": -0.0646,
      "check_primary": true,
      "check_secondary": true,
      "check_cvar": false,
      "check_mdd": false,
      "check_seed_consistency": true,
      "tuned_lambda_cvar": 0.03,
      "promoted": false,
      "decision": "REJECTED",
      "decision_reason": "Small utility gain, but still fails CVaR and MDD checks."
    },
    {
      "sequence": 3,
      "algo": "nstep_sarsa",
      "algo_label": "n-step Sarsa",
      "reward_delta": 0.0217,
      "return_delta": 0.0236,
      "cvar_delta": 0.0019,
      "mdd_delta": -0.031,
      "check_primary": true,
      "check_secondary": true,
      "check_cvar": true,
      "check_mdd": true,
      "check_seed_consistency": true,
      "tuned_lambda_cvar": 0.0025,
      "promoted": true,
      "decision": "PROMOTED",
      "decision_reason": "Mild utility gain with all risk checks passing under the synchronized gate."
    }
  ],
  "sg_budget_rows": [
    {
      "experiment": "sg_ctrl_lrn",
      "promoted": true,
      "episodes": 120,
      "alpha": 0.015,
      "epsilon_decay": 0.95,
      "epsilon_min": 0.01,
      "return_delta": 0.015,
      "reward_delta": 0.0368,
      "cvar_delta": 0.0001,
      "mdd_delta": 0.0001,
      "train_seconds": 55.0258
    },
    {
      "experiment": "sg_ctrl_lrn_e200",
      "promoted": false,
      "episodes": 200,
      "alpha": 0.015,
      "epsilon_decay": 0.95,
      "epsilon_min": 0.01,
      "return_delta": -0.2969,
      "reward_delta": -0.2468,
      "cvar_delta": -0.0084,
      "mdd_delta": 0.0623,
      "train_seconds": 96.4861
    },
    {
      "experiment": "sg_ctrl_final_guard_e200",
      "promoted": false,
      "episodes": 200,
      "alpha": 0.03,
      "epsilon_decay": 0.98,
      "epsilon_min": 0.12,
      "return_delta": -0.0516,
      "reward_delta": -0.0378,
      "cvar_delta": -0.0027,
      "mdd_delta": 0.0219,
      "train_seconds": 48.4963
    }
  ],
  "with_no_rows": [
    {
      "algo": "q_learning",
      "algo_label": "Q-learning",
      "reward_delta": -0.022,
      "return_delta": -0.0167,
      "cvar_delta": -0.005,
      "mdd_delta": 0.0285
    },
    {
      "algo": "sg_sarsa",
      "algo_label": "SG-Sarsa",
      "reward_delta": -0.4013,
      "return_delta": -0.4591,
      "cvar_delta": -0.0129,
      "mdd_delta": 0.1228
    },
    {
      "algo": "nstep_sarsa",
      "algo_label": "n-step Sarsa",
      "reward_delta": -0.2075,
      "return_delta": -0.1879,
      "cvar_delta": -0.0069,
      "mdd_delta": 0.0819
    }
  ],
  "main_conclusion_rows": [
    {
      "model": "Baseline(with_cvar)",
      "reward": 0.8321,
      "return": 1.1428,
      "cvar": 0.0266,
      "mdd": -0.2581,
      "train_time_s": null
    },
    {
      "model": "q_learning (with_cvar, baseline cfg)",
      "reward": 0.1336,
      "return": 0.3058,
      "cvar": 0.0072,
      "mdd": -0.0777,
      "train_time_s": 34.5765
    },
    {
      "model": "nstep_sarsa V2 (with_cvar, tuned)",
      "reward": 0.521,
      "return": 0.7303,
      "cvar": 0.0205,
      "mdd": -0.1907,
      "train_time_s": 33.1049
    }
  ],
  "headline_rows": [
    {
      "algo": "q_learning",
      "algo_label": "Q-learning",
      "promoted": false,
      "reward": 0.5898,
      "return": 0.9257,
      "cvar": 0.0255,
      "mdd": -0.2486,
      "train_time_s": 32.4743
    },
    {
      "algo": "sg_sarsa",
      "algo_label": "SG-Sarsa",
      "promoted": false,
      "reward": 0.1686,
      "return": 0.3961,
      "cvar": 0.0147,
      "mdd": -0.1631,
      "train_time_s": 29.0985
    },
    {
      "algo": "nstep_sarsa",
      "algo_label": "n-step Sarsa",
      "promoted": true,
      "reward": 0.521,
      "return": 0.7303,
      "cvar": 0.0205,
      "mdd": -0.1907,
      "train_time_s": 33.1049
    }
  ],
  "xlf_trend": [
    {
      "date": "2019-01",
      "norm_close": 1.0
    },
    {
      "date": "2019-02",
      "norm_close": 1.0224
    },
    {
      "date": "2019-03",
      "norm_close": 0.9962
    },
    {
      "date": "2019-04",
      "norm_close": 1.0857
    },
    {
      "date": "2019-05",
      "norm_close": 1.0078
    },
    {
      "date": "2019-06",
      "norm_close": 1.0749
    },
    {
      "date": "2019-07",
      "norm_close": 1.1002
    },
    {
      "date": "2019-08",
      "norm_close": 1.0484
    },
    {
      "date": "2019-09",
      "norm_close": 1.096
    },
    {
      "date": "2019-10",
      "norm_close": 1.1234
    },
    {
      "date": "2019-11",
      "norm_close": 1.1802
    },
    {
      "date": "2019-12",
      "norm_close": 1.211
    },
    {
      "date": "2020-01",
      "norm_close": 1.1787
    },
    {
      "date": "2020-02",
      "norm_close": 1.0461
    },
    {
      "date": "2020-03",
      "norm_close": 0.8261
    },
    {
      "date": "2020-04",
      "norm_close": 0.9042
    },
    {
      "date": "2020-05",
      "norm_close": 0.9288
    },
    {
      "date": "2020-06",
      "norm_close": 0.924
    },
    {
      "date": "2020-07",
      "norm_close": 0.9595
    },
    {
      "date": "2020-08",
      "norm_close": 1.0006
    },
    {
      "date": "2020-09",
      "norm_close": 0.9664
    },
    {
      "date": "2020-10",
      "norm_close": 0.958
    },
    {
      "date": "2020-11",
      "norm_close": 1.1193
    },
    {
      "date": "2020-12",
      "norm_close": 1.1899
    },
    {
      "date": "2021-01",
      "norm_close": 1.1685
    },
    {
      "date": "2021-02",
      "norm_close": 1.3042
    },
    {
      "date": "2021-03",
      "norm_close": 1.3805
    },
    {
      "date": "2021-04",
      "norm_close": 1.4701
    },
    {
      "date": "2021-05",
      "norm_close": 1.5402
    },
    {
      "date": "2021-06",
      "norm_close": 1.4934
    },
    {
      "date": "2021-07",
      "norm_close": 1.4865
    },
    {
      "date": "2021-08",
      "norm_close": 1.563
    },
    {
      "date": "2021-09",
      "norm_close": 1.5342
    },
    {
      "date": "2021-10",
      "norm_close": 1.6458
    },
    {
      "date": "2021-11",
      "norm_close": 1.5518
    },
    {
      "date": "2021-12",
      "norm_close": 1.604
    },
    {
      "date": "2022-01",
      "norm_close": 1.6044
    },
    {
      "date": "2022-02",
      "norm_close": 1.5822
    },
    {
      "date": "2022-03",
      "norm_close": 1.5802
    },
    {
      "date": "2022-04",
      "norm_close": 1.4231
    },
    {
      "date": "2022-05",
      "norm_close": 1.4627
    },
    {
      "date": "2022-06",
      "norm_close": 1.3038
    },
    {
      "date": "2022-07",
      "norm_close": 1.3975
    },
    {
      "date": "2022-08",
      "norm_close": 1.3701
    },
    {
      "date": "2022-09",
      "norm_close": 1.2652
    },
    {
      "date": "2022-10",
      "norm_close": 1.4161
    },
    {
      "date": "2022-11",
      "norm_close": 1.5132
    },
    {
      "date": "2022-12",
      "norm_close": 1.4342
    },
    {
      "date": "2023-01",
      "norm_close": 1.5332
    },
    {
      "date": "2023-02",
      "norm_close": 1.4979
    },
    {
      "date": "2023-03",
      "norm_close": 1.3549
    },
    {
      "date": "2023-04",
      "norm_close": 1.3979
    },
    {
      "date": "2023-05",
      "norm_close": 1.3385
    },
    {
      "date": "2023-06",
      "norm_close": 1.427
    },
    {
      "date": "2023-07",
      "norm_close": 1.4956
    },
    {
      "date": "2023-08",
      "norm_close": 1.4554
    },
    {
      "date": "2023-09",
      "norm_close": 1.4105
    },
    {
      "date": "2023-10",
      "norm_close": 1.376
    },
    {
      "date": "2023-11",
      "norm_close": 1.5265
    },
    {
      "date": "2023-12",
      "norm_close": 1.6067
    },
    {
      "date": "2024-01",
      "norm_close": 1.6563
    },
    {
      "date": "2024-02",
      "norm_close": 1.7238
    },
    {
      "date": "2024-03",
      "norm_close": 1.8067
    },
    {
      "date": "2024-04",
      "norm_close": 1.7312
    },
    {
      "date": "2024-05",
      "norm_close": 1.7861
    },
    {
      "date": "2024-06",
      "norm_close": 1.7703
    },
    {
      "date": "2024-07",
      "norm_close": 1.8836
    },
    {
      "date": "2024-08",
      "norm_close": 1.9697
    },
    {
      "date": "2024-09",
      "norm_close": 1.9587
    },
    {
      "date": "2024-10",
      "norm_close": 2.0088
    },
    {
      "date": "2024-11",
      "norm_close": 2.2189
    },
    {
      "date": "2024-12",
      "norm_close": 2.0977
    },
    {
      "date": "2025-01",
      "norm_close": 2.234
    },
    {
      "date": "2025-02",
      "norm_close": 2.2648
    },
    {
      "date": "2025-03",
      "norm_close": 2.1698
    },
    {
      "date": "2025-04",
      "norm_close": 2.1241
    },
    {
      "date": "2025-05",
      "norm_close": 2.2199
    },
    {
      "date": "2025-06",
      "norm_close": 2.2891
    },
    {
      "date": "2025-07",
      "norm_close": 2.2891
    },
    {
      "date": "2025-08",
      "norm_close": 2.3599
    },
    {
      "date": "2025-09",
      "norm_close": 2.3625
    },
    {
      "date": "2025-10",
      "norm_close": 2.2967
    },
    {
      "date": "2025-11",
      "norm_close": 2.3388
    },
    {
      "date": "2025-12",
      "norm_close": 2.4103
    },
    {
      "date": "2026-01",
      "norm_close": 2.3518
    }
  ]
};
