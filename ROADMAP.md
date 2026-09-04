# Roadmap

## 1. Finish the FI-2010 fidelity audit

- run `author_tf1` at k=10/20/50 on five training seeds;
- quantify MC-dropout test noise separately from training-seed noise;
- run the Ablation Ledger (normalization, inference dropout, padding, channel widths);
- publish paper-vs-reproduction gap figure and hardware/time manifest.

## 2. Honest economic evaluation on raw LOB data

FI-2010 is not a defensible execution backtest dataset, so PnL is deferred rather than improvised.
A LOBSTER or crypto-L2 extension should use:

- raw timestamped L2 updates with bid/ask prices and sizes;
- strictly causal normalization fitted only from past information;
- signal at event `t`, execution no earlier than the next executable book state;
- explicit market/limit-order convention and fill assumptions;
- taker/maker fees plus half-spread/slippage/latency sensitivity;
- position and inventory limits;
- turnover, gross/net PnL, drawdown and risk-adjusted metrics;
- walk-forward splits with no event-window leakage.

The classification benchmark and economic backtest should remain separate outputs: a higher FI-2010 F1 is not itself evidence of a tradable edge.

## 3. Modern benchmark

Compare DeepLOB with the MLPLOB-style baseline and, optionally, TLOB on the same raw-data protocol. Revisit the target label itself using the horizon-bias discussion in Berti & Kasneci (2025).
