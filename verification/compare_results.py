"""Compare recomputed CSVs with the distributed scientific records."""
from pathlib import Path
import argparse
import csv
import math
import json

def compare(reference, computed):
    with reference.open() as a, computed.open() as b:
        x, y = csv.DictReader(a), csv.DictReader(b)
        assert x.fieldnames == y.fieldnames, reference
        old, new = list(x), list(y)
    assert len(old) == len(new) and old, reference
    maximum = 0
    float_differences = 0
    timing = [name for name in x.fieldnames if "seconds" in name]
    for i, (r, s) in enumerate(zip(old, new)):
        for k in r:
            if k in timing or r[k] == s[k]:
                continue
            u, v = float(r[k]), float(s[k])
            assert math.isclose(u, v, rel_tol=0, abs_tol=1e-9), (reference, i+2, k, u, v)
            maximum = max(maximum, abs(u-v))
            float_differences += 1
    return dict(rows=len(old), timing_columns=timing, numeric_differences=float_differences,
                maximum_absolute_difference=maximum, byte_identical=reference.read_bytes()==computed.read_bytes())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path, help="isolated project root with completed calculations")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = {}
    for folder, target in [("data", "data"),("result", "result")]:
        for ref in sorted((root/folder).glob("*.csv")):
            report[str(ref.relative_to(root))]=compare(ref, args.work/target/ref.name)
    print(json.dumps(report, indent=2))
