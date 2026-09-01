"""Report charts for sweep results. Requires the `plot` extra (matplotlib)."""

from __future__ import annotations

from pathlib import Path

from .simulation import SweepResult


def four_panel(sweeps: list[SweepResult], out_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    books = [s.book for s in sweeps]
    x = range(len(books))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Tax-aware long-short leverage tradeoffs (synthetic Monte Carlo)")

    ax = axes[0][0]
    gross = [s.median("gross_losses_realized") for s in sweeps]
    net = [s.median("net_loss_pre_liquidation") for s in sweeps]
    width = 0.38
    ax.bar(
        [i - width / 2 for i in x], [g / 1e6 for g in gross], width, label="Gross losses realized"
    )
    ax.bar([i + width / 2 for i in x], [n / 1e6 for n in net], width, label="Net capital losses")
    ax.set_ylabel("$M over horizon (median)")
    ax.set_title("Loss realization grows faster than usable net losses")
    ax.set_xticks(list(x), books)
    ax.legend()

    ax = axes[0][1]
    med = [s.median("ending_after_tax_wealth") / 1e6 for s in sweeps]
    p10 = [s.percentile("ending_after_tax_wealth", 10) / 1e6 for s in sweeps]
    p90 = [s.percentile("ending_after_tax_wealth", 90) / 1e6 for s in sweeps]
    ax.plot(list(x), med, "o-", color="black", label="Median")
    ax.fill_between(list(x), p10, p90, alpha=0.2, label="10th-90th percentile")
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_ylabel("Terminal after-tax wealth, $M")
    ax.set_title("After-tax wealth vs leverage, zero alpha")
    ax.set_xticks(list(x), books)
    ax.legend()

    ax = axes[1][0]
    fees = [s.median("management_fees") / 1e3 for s in sweeps]
    borrow = [s.median("borrow_costs") / 1e3 for s in sweeps]
    txn = [s.median("transaction_costs") / 1e3 for s in sweeps]
    pil = [s.median("payments_in_lieu") / 1e3 for s in sweeps]
    bottom = [0.0] * len(books)
    for series, label in [
        (fees, "Management fee"),
        (borrow, "Short borrow"),
        (txn, "Trading"),
        (pil, "Payments in lieu"),
    ]:
        ax.bar(list(x), series, bottom=bottom, label=label)
        bottom = [b + s for b, s in zip(bottom, series, strict=True)]
    ax.set_ylabel("Cumulative cost, $k (median)")
    ax.set_title("Observable costs compound with gross exposure")
    ax.set_xticks(list(x), books)
    ax.legend()

    ax = axes[1][1]
    te = [s.median("tracking_error") * 100 for s in sweeps]
    dd = [-s.median("max_drawdown") * 100 for s in sweeps]
    mc = [s.margin_call_probability() * 100 for s in sweeps]
    ax.plot(list(x), te, "o-", label="Tracking error, %")
    ax.plot(list(x), dd, "s-", label="Max drawdown, %")
    ax.plot(list(x), mc, "^-", label="Margin-breach paths, %")
    ax.set_ylabel("Percent")
    ax.set_title("Risk and forced-sale exposure rise with leverage")
    ax.set_xticks(list(x), books)
    ax.legend()

    for row in axes:
        for ax in row:
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
