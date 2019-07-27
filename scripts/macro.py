import pandas as pd
import wbdata.wbdata as wb


print(pd.DataFrame(wb.get_source()))