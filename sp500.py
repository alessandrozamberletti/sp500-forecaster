from datapackage import Package


class SP500:
    __DP_URL = 'https://datahub.io/core/s-and-p-500-companies/datapackage.json'

    def __init__(self, limit=0):
        package = Package(SP500.__DP_URL)
        for resource in package.resources:
            if resource.descriptor['datahub']['type'] == 'derived/csv':
                self.sp500 = [s[0].encode('utf-8') for s in resource.read()]
        if limit != 0:
            self.sp500 = self.sp500[:limit]

    def split(self, ratio):
        split_idx = int(len(self.sp500) * ratio)
        return self.sp500[:split_idx], self.sp500[split_idx:]
