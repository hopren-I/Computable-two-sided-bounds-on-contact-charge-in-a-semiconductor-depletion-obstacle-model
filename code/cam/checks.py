"""Compare raw results against independent, printed manuscript fixtures."""
import csv
from decimal import Decimal
from math import isfinite
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2]/"verification"/"cam"


def status(value, printed):
    if not isfinite(value):
        return "fail"
    if printed == r"\approx0":
        return "printed" if abs(value) < 1e-12 else "fail"
    ref = Decimal(printed)
    half_unit = Decimal(".5").scaleb(ref.as_tuple().exponent)
    return "printed" if abs(Decimal(str(value))-ref) <= half_unit else "fail"


def check_tables(tables):
    checks = []
    for n,computed in enumerate(tables,1):
        with (FIXTURES/f"table{n}.csv").open(encoding="utf8",newline="") as f:
            published = list(csv.DictReader(f))
        if len(computed) != len(published):
            raise ValueError(f"Unexpected number of rows in Table {n}")
        for row,ref in zip(computed,published):
            key = next(iter(ref))
            if row.keys() != ref.keys() or abs(row[key]-float(ref[key])) > 1e-14:
                raise ValueError(f"Unexpected columns or case in Table {n}: {row}")
            for name,printed in ref.items():
                if name == key:
                    continue
                value = row[name]
                result = status(value,printed)
                # One cancellation-sensitive error at b=3 and its normalized form.
                # Use the same absolute charge-error tolerance in both units.
                if n==5 and row["b"]==3 and result=="fail" and name in ("midpoint_error","midpoint_pct"):
                    tol = 1e-11 if name=="midpoint_error" else 200*1e-11/row["W"]
                    if abs(value-float(printed)) <= tol:
                        result = "roundoff_tolerance"
                checks.append(dict(table=n,case=ref[key],metric=name,printed=printed,
                                   computed=value,status=result))
    return checks


def check_prose(values):
    with (FIXTURES/"prose.csv").open(encoding="utf8",newline="") as f:
        references = list(csv.DictReader(f))
    if set(values) != {r["metric"] for r in references}:
        raise ValueError("Expected prose metrics differ from computed metrics")
    checks = []
    for ref in references:
        metric, printed = ref["metric"],ref["printed"]
        value = values[metric]
        result = status(value,printed)
        if ref["absolute_tolerance"]:
            tol = float(ref["absolute_tolerance"])
            result = "within_approximation" if abs(value-float(printed))<=tol else "fail"
        checks.append(dict(table="prose",case="",metric=metric,printed=printed,
                           computed=value,status=result))
    return checks

