#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a case's target time.

For every trophy, work in logit space.  The history reads repeat identical
values between Exophase refreshes, so consecutive identical reads are
collapsed to their distinct refresh points first and the trend is fitted on
those.  The median is a least-squares line of logit(rate) against
ln(1 + hours since 2026-11-19T00:00:00Z), shrunk toward the flat (last-rate)
forecast, read at the target.  The spread is a random-walk term that grows
with the horizon, plus the trophy's own observed refresh-jump size and the
fit's residual scatter, so a refresh-sized step stays inside the 5%-95%
band.  Standard library only, deterministic, no network, no randomness.
"""

import json
import math
import os

# Standard normal quantiles for the forecast levels 0.05, 0.25, 0.5, 0.75, 0.95.
Z_VALUES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

# Launch night, the origin of the log-time axis.  Rates rise fast just after
# launch and then saturate, so log time keeps the extrapolation honest.

# Random-walk spread of the house baseline: 0.5 * sqrt(horizon in days).
RW_COEF = 0.5


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into hours since the epoch origin."""
    year = int(text[0:4])
    month = int(text[5:7])
    day = int(text[8:10])
    hour = int(text[11:13])
    minute = int(text[14:16])
    second = int(text[17:19])
    # Days since 2026-11-19, computed without the datetime module so the
    # program stays trivially portable across Python versions.
    days_in_month = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    days = 0
    for m in range(1, month):
        days += days_in_month[m - 1]
    days += day - 1
    # 2026-11-19 is day 322 of 2026 (31+28+31+30+31+30+31+31+30+31+18).
    days -= 322
    return days * 24.0 + hour + minute / 60.0 + second / 3600.0


def clamp_rate(p):
    """Clamp a percent into (0, 100) where the logit is defined."""
    if p < 0.01:
        return 0.01
    if p > 99.99:
        return 99.99
    return p


def logit(p):
    p = clamp_rate(p) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lo):
    if lo >= 0.0:
        return 100.0 / (1.0 + math.exp(-lo))
    e = math.exp(lo)
    return 100.0 * e / (1.0 + e)


def distinct_reads(history):
    """Collapse consecutive identical reads to one point each.

    A read is identified by its players-tracked count and its whole rates
    vector.  Between Exophase refreshes every read repeats, so the flat
    stretches carry no information and must not dominate the fit.
    """
    points = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates", [])))
        if key == prev_key:
            continue
        prev_key = key
        points.append(read)
    return points


def fit_line(xs, ys):
    """Least-squares line y = a + b*x; returns (a, b, resid_sd, mx, my)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) * (x - mx) for x in xs)
    if sxx <= 0.0:
        return my, 0.0, 0.0, mx, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid, mx, my


def forecast_trophy(rates, slots, x_target, horizon_days):
    """Forecast one trophy's five quantiles at the target.

    rates/slots are the distinct refresh points, oldest first.
    """
    last_rate = rates[-1] if rates else 1.0
    last_logit = logit(last_rate)

    xs = [math.log1p(max(0.0, parse_time(s))) for s in slots]
    ys = [logit(r) for r in rates]
    n = len(xs)

    if n >= 2:
        a, b, resid, mx, my = fit_line(xs, ys)
        # Shrink the slope toward the flat forecast: with few refresh points
        # the slope is mostly noise, and a runaway line is the worst error.
        # The shrunk line still passes through the mean point (mx, my).
        b *= n / (n + 2.0)
        median_logit = my + b * (x_target - mx)
        # Dampen the extrapolation: never let the median run more than 1.5
        # logit units away from the last observed rate.  A story trophy can
        # climb fast, but a line fitted on launch-night noise must not
        # invent a tenfold rise (or fall) by the target.
        delta = median_logit - last_logit
        if delta > 1.5:
            median_logit = last_logit + 1.5
        elif delta < -1.5:
            median_logit = last_logit - 1.5
    else:
        b = 0.0
        resid = 0.0
        median_logit = last_logit

    # How much this trophy jumps at a refresh, after removing the trend:
    # the step sizes between consecutive distinct reads, detrended.
    step_sq = []
    for j in range(1, n):
        dx = xs[j] - xs[j - 1]
        step = (ys[j] - ys[j - 1]) - b * dx
        step_sq.append(step * step)
    step_rms = math.sqrt(sum(step_sq) / len(step_sq)) if step_sq else 0.0

    # Spread on the logit scale: the random-walk term, the refresh-jump
    # term (growing with the horizon, since more refreshes can land in it)
    # and the fit's residual scatter, combined in quadrature.  The flat
    # term covers the refresh risk the history never showed: with no
    # distinct refresh observed at all (the practice cases), a step jump
    # is a real possibility the data cannot price, so the band is wide;
    # once refreshes have been seen, a smaller floor remains for a
    # regime the history has not shown yet.
    var = (RW_COEF * RW_COEF) * horizon_days
    var += step_rms * step_rms * (horizon_days / 2.0)
    var += resid * resid
    if n <= 1:
        var += 0.9 * 0.9
    else:
        var += 0.3 * 0.3
    sd = math.sqrt(var)
    if sd < 0.3:
        sd = 0.3
    if sd > 2.5:
        sd = 2.5

    quantiles = []
    prev = -1.0
    for z in Z_VALUES:
        q = inv_logit(median_logit + z * sd)
        q = round(q, 4)
        if q < 0.0:
            q = 0.0
        if q > 100.0:
            q = 100.0
        if q < prev:
            q = prev
        prev = q
        quantiles.append(q)
    return quantiles


def forecast_case(case):
    trophies = case["trophies"]
    history = sorted(case.get("history", []), key=lambda r: r.get("slot", ""))
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p(max(0.0, parse_time(case["target"])))

    distinct = distinct_reads(history)
    slots = [r["slot"] for r in distinct]

    rows = []
    for i, trophy in enumerate(trophies):
        rates = [r["rates"][i] for r in distinct] if distinct else []
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(rates, slots, x_target, horizon_days)})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, case["id"] + ".out")
        text = json.dumps(result, sort_keys=True, separators=(",", ":"))
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
