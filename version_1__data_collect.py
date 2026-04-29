import yfinance as yf, numpy as np, pandas as pd, pandas_market_calendars as mcal
from datetime import *
from zoneinfo import ZoneInfo

qqqm=yf.Ticker('QQQM')
data=qqqm.history(period='max')

print(data)

#check if the stock market will open today
nyse=mcal.get_calendar('NYSE')
today=datetime.now(ZoneInfo('America/New_York'))
calendar=np.array(nyse.schedule(start_date=f'{today.year}-{today.month}-01', end_date=f'{today.year}-{today.month+2}-01').index.date) #turn the calendar from panda data frame into numpy array
next_open_day_index=np.argmax(data.index.date[-1]==calendar)+1

# calculate time interval of trading days (skipped the first trade day)
time=(np.array(np.array(data.index)[1:]-np.array(data.index)[0:-1])/timedelta(days=1)).reshape(len(data)-1,1)[1:]
time=np.concatenate((time, [[(calendar[next_open_day_index]-data.index.date[-1])/timedelta(days=1)]]))
time=time/10

data=data.to_numpy().T

new_data=[0,0,0,0,0,0]
new_data[0]=-1*(np.log(data[0,:-1])-np.log(data[0, 1:]))*100 #log returns (skipped the first trade day)
new_data[1]=-1*(np.log(data[1,:-1])-np.log(data[1, 1:]))*100 #log returns (skipped the first trade day)
new_data[2]=-1*(np.log(data[2,:-1])-np.log(data[2, 1:]))*100 #log returns (skipped the first trade day)
new_data[3]=-1*(np.log(data[3,:-1])-np.log(data[3, 1:]))*100 #log returns (skipped the first trade day)
new_data[4]=np.log(1+data[4,1:])/15                          #log(1+x) for trade volumn (skipped the first trade day)
new_data[5]=time.reshape(len(time))

new_data=np.array(new_data).T

train_data=new_data[:-1] #skip the latest trade day
train_answer=new_data[1:,0:5]
time_interval=new_data[:-1,-1].reshape(len(new_data)-1,1)

test_data=new_data[-1] # the latest trade day

print(train_data)
print(np.shape(train_data))
print(time)
print()
solve=data[:5,-1]
print(solve)
print(test_data)
