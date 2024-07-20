import pandas as pd
from typing import List
from .Sequence import Sequence
# from tqdm import tqdm

class DataSetProcessor:

    def get5MinSequences(self, data: pd.DataFrame, scale=False) -> List[Sequence]:

        tickers = data['ticker'].unique()
        sequences = []
        # for ticker in tqdm(tickers, desc=f"Processing {len(tickers)} tickers"):
        for ticker in tickers:
            tickerData = data[data['ticker'] == ticker]
            if scale:
                tickerData = self.normalize(tickerData)
                
            dates = tickerData['date'].unique()
            for date in dates: # slow but will always work if the data is not sorted
                dateData = tickerData[tickerData['date'] == date]
                dateData = dateData.sort_values(by='timeMinute')
                dateData = dateData.reset_index(drop=True)
                # for i in range(0, len(dateData), 5):
                #     if i + 5 > len(dateData):
                #         break
                #     sequence = Sequence(ticker, date, dateData[i:i+5])
                #     sequences.append(sequence)
                points = [(row['volume'], row['volume_weighted_average']) for index, row in dateData.iterrows()]
                sequence = Sequence(
                    ticker,
                    date,
                    points
                )

                sequences.append(sequence)

        # for each ticker
            # for each day
                # create a sequence of 5 minute intervals
        
        return sequences

    def addDateAndMinutes(self, rawData: pd.DataFrame) -> pd.DataFrame:
        data = rawData.copy()
        dates = pd.to_datetime(data['time'], unit='ms')
        dates = dates.dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
        dates = dates.dt.tz_localize(None)
        data["date"] = dates.dt.date
        data["timeTime"] = dates.dt.time
        data["timeMinute"] = dates.dt.hour * 100 + dates.dt.minute
        return data
    
    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data['volume'] = data['volume'] / data['volume'].max()
        data['volume_weighted_average'] = data['volume_weighted_average'] / data['volume_weighted_average'].max()
        return data
    

    
