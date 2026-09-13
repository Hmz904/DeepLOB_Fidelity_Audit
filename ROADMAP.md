# Roadmap

## 1. Finish the FI-2010 fidelity audit

- [x] run `author_tf1` at k=10/20/50 — five seeds at k=10, three at k=20, one at k=50;
- [x] quantify MC-dropout test noise separately from training-seed noise — MC sd is near 0.0002 at
      every horizon against training-seed sd of 0.006, two orders of magnitude apart;
- [x] run the Ablation Ledger — padding, channel width and both dropout splits are measured; the
      z-score row is blocked on raw data rather than pending;
- [x] publish paper-vs-reproduction gap figure and hardware/time manifest.

Remaining and optional: four more seeds at k=50, about 13.8 GPU-hours, which is the only thing
keeping that horizon on a single run.

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
