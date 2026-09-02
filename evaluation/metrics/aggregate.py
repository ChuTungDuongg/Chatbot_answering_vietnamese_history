import math
import statistics

from evaluation.metrics.specs import GROUPS


def _number(value):
    return isinstance(value, (int, float, bool)) and math.isfinite(value)


def _gate(record, key):
    if key == "viewpoint_negative":
        return record.annotation is not None and record.annotation.viewpoint_should_flag is False
    if key == "warm_start":
        return record.signals.get("cold_start") is False
    return record.signals.get(key) is True


def calculate_metrics(records):
    groups = {}
    for group, specs in GROUPS.items():
        groups[group] = {}
        for spec in specs:
            values, denominators = [], []
            for record in records:
                if spec.gate and not _gate(record, spec.gate):
                    continue
                value = (getattr(record.annotation, spec.signal, None) if spec.annotation
                         else record.signals.get(spec.signal))
                if not _number(value):
                    continue
                if spec.denominator:
                    denominator = record.signals.get(spec.denominator)
                    if not _number(denominator) or denominator <= 0:
                        continue
                    if value < 0 or value > denominator:
                        raise ValueError(f"invalid counts for {spec.name}: {value}/{denominator}")
                    denominators.append(denominator)
                values.append(float(value))
            denominator = sum(denominators) if spec.denominator else len(values)
            value = None
            if values:
                if spec.denominator:
                    value = sum(values) / denominator
                elif spec.statistic == "p95":
                    value = sorted(values)[math.ceil(len(values) * .95) - 1]
                elif spec.statistic == "median":
                    value = statistics.median(values)
                elif spec.statistic == "sum":
                    value = sum(values)
                else:
                    value = statistics.mean(values)
            groups[group][spec.name] = {"value": value, "observed_records": len(values),
                "total_records": len(records), "denominator": denominator,
                "direction": spec.direction, "definition": spec.description}
    return groups
