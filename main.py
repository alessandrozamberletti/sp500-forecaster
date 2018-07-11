import pandas as pd
# https://stackoverflow.com/questions/50394873/import-pandas-datareader-gives-importerror-cannot-import-name-is-list-like
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
# https://github.com/pydata/pandas-datareader/issues/487
yf.pdr_override()

data = pdr.get_data_yahoo('AMZN')
data = data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]

print(data.head)
