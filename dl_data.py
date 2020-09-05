import jqdatasdk as jq
import pandas as pd
import datetime

jq.auth('18028042641', '042641')
print(jq.get_query_count())
code = 'M8888.XDCE'
freq = '5m'
NUM_YEARS = 4
end = 2009
price = jq.get_price(code, frequency=freq, start_date=datetime.date(end-NUM_YEARS, 1, 1), end_date=datetime.date(end+1, 1, 1))
price.to_csv(f'data/{code}_5m_test.csv')