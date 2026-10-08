#!/usr/bin/env python
"""This account's Kaggle accelerator quota (weekly GPU hours used and left), from the kernels API.

    KAGGLE_API_TOKEN=... .venv/bin/python scripts/kaggle_quota.py

The RTX PRO 6000 draws on the weekly GPU quota (30 h, reset Saturdays 00:00 UTC); a competition rerun does not.
"""
from __future__ import annotations

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        stats = kaggle.kernels.kernels_api_client.get_accelerator_quota_statistics(
            ApiGetAcceleratorQuotaStatisticsRequest())
    for name in ("gpu_quota", "tpu_quota"):
        quota = getattr(stats, name, None)
        if quota is None:
            continue
        used, allowed = quota.time_used.total_seconds() / 3600, quota.total_time_allowed.total_seconds() / 3600
        print(f"{name}: {used:.2f} h used of {allowed:.0f} h, {allowed - used:.2f} h left "
              f"(reserved {quota.time_reserved.total_seconds() / 3600:.2f} h)")


if __name__ == "__main__":
    main()
