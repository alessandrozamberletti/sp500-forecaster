from datapackage import Package
import random


__DP_URL = 'https://datahub.io/core/s-and-p-500-companies/datapackage.json'


def retrieve(limit=0, split_ratio=0):
    package = Package(__DP_URL)
    for resource in package.resources:
        if resource.descriptor['datahub']['type'] == 'derived/csv':
            sp500 = [s[0].encode('utf-8') for s in resource.read()]
    if limit != 0:
        sp500 = random.sample(sp500, limit)
    return sp500 if split_ratio == 0 else _split(sp500, split_ratio)


def _split(sp500, ratio):
    split_idx = int(len(sp500) * ratio)
    return sp500[:split_idx], sp500[split_idx:]
