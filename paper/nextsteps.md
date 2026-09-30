                    S&P 500 RETURNS
                           │
                           ▼
                 Rolling correlation matrix
                           │
                           ▼
                  Eigenvalue decomposition
                           │
                           ▼
                    Market modes
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Mode-level features        Market-level features
              │                         │
              │                         ▼
              │                  Regime Detection
              │                         │
              │                  ┌──────┴──────┐
              │                  │             │
              │                Regime 1      Regime 2 ...
              │                  │             │
              └────────────┬─────┴─────────────┘
                           ▼
                  Predict mode quality
                           │
                           ▼
                 Portfolio construction
                           │
                           ▼
                    Walk-forward test
