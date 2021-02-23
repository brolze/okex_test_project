# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 22:21:37 2021

@author: xujianqiao
"""

import pandas as pd
import datetime as dt
import pytz
pst = pytz.timezone('Asia/Shanghai')


if __name__=="__main__":
    # 读数据，转化一下日期
    # data = pd.read_csv("data/bitstampUSD_1-min_data_2012-01-01_to_2020-12-31.csv")
    # data['time'] = data['Timestamp'].apply(lambda x:pst.localize(dt.datetime.utcfromtimestamp(x)))
    # data['str_time'] = data['time'].apply(lambda x:dt.datetime.strftime(x,"%Y-%m-%d %H:%M:%S"))
    # data_2020 = data[data['str_time'].apply(lambda x:x[0:4])=='2020']
    # data_2020.columns
    # data_2020 = data_2020[['Timestamp','str_time', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)',
    #    'Volume_(Currency)', 'Weighted_Price']]
    # data_2020.to_excel("history_btcdata_2020.xlsx")
    
    
    
    # 策略，对于任何一个candle，应该返回买入，卖出或不做操作
    data_2020['Close']
    
    tmp = data_2020.head(10000)
    
    layout = dict(title_x=0.5, # title居中
              font=dict(family="Times New Roman",size=20,color="RebeccaPurple"),
              # 字体设置
              coloraxis_colorbar=dict(xanchor="left",x=0.75,ticks="outside"),
              # 颜色条设置
              margin=dict(b= 40,l=40, r=40, t= 40)
              # 大小设置
             )
    
    # 价格和均线
    fig = px.line(tmp, x="str_time", y="Close",title='btc分钟').update_layout(layout)
    plotly.offline.plot(fig)

    
    
    
    
    
    
    
    
    
    