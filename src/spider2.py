#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:44:01 2021

@author: xujianqiao

$ unset all_proxy && unset ALL_PROXY # 取消所有 socks 代理
$ pip install pysocks

https://www.sitstars.com/archives/94/
"""

import requests
import sys,os
import okex.account_api as account
import okex.futures_api as future
import okex.lever_api as lever
import okex.spot_api as spot
import okex.swap_api as swap
import okex.index_api as index
import okex.option_api as option
import okex.system_api as system
import okex.information_api as information
import json
import datetime as dt
import time
# import websocket_example as wbe
import asyncio
# import websockets
import dateutil.parser as dp
import hmac
import base64
import zlib
import datetime as dt
import pandas as pd
import pytz #py timezone

# 跳转代码
# from goto import with_goto

# import nest_asyncio
# nest_asyncio.apply()

# 作图
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly
#plotly.offline.init_notebook_mode(connected=True)

nextday_str = (dt.datetime.now()+dt.timedelta(days=1)).strftime("%Y%m%d")


def utc_to_local(utc_time_str):
    pst = pytz.timezone('Asia/Shanghai')
    utc_tz = pytz.timezone('UTC')
    time_str = dt.datetime.strptime(utc_time_str[:-5],# 忽略了毫秒
                                    "%Y-%m-%dT%H:%M:%S")
    utc_time_dt = utc_tz.localize(time_str)
    local_time_dt = utc_time_dt.astimezone(pst)
    return local_time_dt

def local_to_time(timestr):
    pst = pytz.timezone('Asia/Shanghai')
    time_dt = pst.localize(dt.datetime.strptime(timestr,"%Y%m%d"))
    return time_dt

def time_to_utc(time_dt):
    utc_dt = time_dt.astimezone(pytz.utc)
    time_str = utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]+"Z"
    return time_str

# @with_goto
def request_candle(candle_delta,start_dt_str,end_dt_str):
    start_dt = local_to_time(start_dt_str)
    end_dt = local_to_time(end_dt_str)
    time_delta = candle_delta*200
    
    circle_end = end_dt # 从后往前循环
    circle_start = circle_end-dt.timedelta(seconds=time_delta)
    
    list_result = []
    while True:
        try:
            result = spotAPI.get_kline(instrument_id='BTC-USDT', start=time_to_utc(circle_start),
                                        end=time_to_utc(circle_end), granularity=candle_delta)
            list_result.extend(result)
            circle_end = circle_start
            circle_start -=dt.timedelta(seconds=time_delta) #成功后减去一个delta

        except Exception as e:
            time.sleep(5) # 不成功5秒后重试

        if circle_start<start_dt:
            break
        
    content_pd = pd.DataFrame(list_result)
    content_pd.columns = ['开始时间','开盘价格','最高价格','最低价格','收盘价格','交易量']
    content_pd['candle_delta'] = candle_delta
    content_pd['开始时间'] = content_pd['开始时间'].apply(lambda x:utc_to_local(x))
    content_pd.drop_duplicates(inplace=True)
    float_col = content_pd.columns.to_list() # 将数据改为float类型
    float_col.remove('开始时间')
    content_pd[float_col] = content_pd[float_col].astype(float)
    
    return content_pd

# 验证数据正确性
def check_candle_data(content_pd):
    time_shift1 = content_pd['开始时间'].shift(1)
    shift_delta = time_shift1 - content_pd['开始时间']
    # content_pd['shift_delta'] = shift_delta
    check_series = shift_delta[~shift_delta.isnull()]
    fault_records = check_series[~(check_series==dt.timedelta(seconds=60))] # 取出不符合check的记录
    print("##############################################")
    print("错误记录数 %i"%len(fault_records))
    print(fault_records.value_counts())
    print("##############################################")
    



if __name__=="__main__":
    
    # 从文件读取账户信息
    with open('my_account.txt','r',encoding = 'utf-8') as f:
        content = f.read()
    account_info = json.loads(content)
    api_key = account_info['apikey']
    secret_key = account_info['secretkey']
    passphrase = account_info['passphrase']
    
    spotAPI = spot.SpotAPI(api_key, secret_key, passphrase, False)
#     swapAPI = swap.SwapAPI(api_key, secret_key, passphrase, False)
    
# #    accountAPI = account.AccountAPI(api_key, secret_key, passphrase, False)
# #    indexAPI = index.IndexAPI(api_key, secret_key, passphrase, False)
    
#     # 获取数据
    candle_delta = 180
    start_str = (dt.datetime.strptime(nextday_str,'%Y%m%d') \
        - dt.timedelta(seconds=candle_delta*2000)).strftime("%Y%m%d")
    content_pd = request_candle(candle_delta,start_str,nextday_str)
    check_candle_data(content_pd)


    # 存一下数据
    old_dates = content_pd['开始时间']
    content_pd['开始时间'] = content_pd['开始时间'].apply(lambda a: pd.to_datetime(a).strftime("%Y%m%d %H:%M")) 
    
    if os.path.exists("data/candle_%i.xlsx"%candle_delta):
        #读取之前的数据用于组合
        last_pd = pd.read_excel("data/candle_%i.xlsx"%candle_delta,index_col=0) 
        last_pd_max_time = last_pd['开始时间'].max()
        if last_pd_max_time in content_pd['开始时间'].to_list():
            content_pd = content_pd[content_pd['开始时间']>last_pd_max_time]
        content_pd = pd.concat([content_pd,last_pd],axis=0)
        
    content_pd.to_excel("data/candle_%i.xlsx"%candle_delta)
    


    content_pd = pd.read_excel("data/candle_%i.xlsx"%candle_delta,index_col=0) 
    # 作图
    content_pd = content_pd.sort_values(by='开始时间')
    layout = dict(title_x=0.5, # title居中
              font=dict(family="Times New Roman",size=20,color="RebeccaPurple"),
              # 字体设置
              coloraxis_colorbar=dict(xanchor="left",x=0.75,ticks="outside"),
              # 颜色条设置
              margin=dict(b= 40,l=40, r=40, t= 40)
              # 大小设置
             )
    
    # 价格和均线
    fig = px.line(content_pd, x="开始时间", y="收盘价格",title='btc分钟').update_layout(layout)
    plotly.offline.plot(fig)

    for window in [10,30,60]:
        content_pd[str(window)+'_mv'] = content_pd['收盘价格'].rolling(window).mean()
    df = content_pd.melt(id_vars='开始时间',value_vars=['收盘价格', '10_mv','30_mv','60_mv'])
    fig = px.line(df, x="开始时间", y="value", color="variable",labels={"variable": "指标"})\
        .update_layout(layout)
    fig.update_yaxes(visible=False, showticklabels=False)
    plotly.offline.plot(fig)
    
    
    # 蜡烛图
    content_pd['ma_5'] = content_pd['收盘价格'].rolling(5).mean() # 计算5*蜡烛均线
    df = content_pd
    fig = go.Figure(data=[go.Candlestick(x=df['开始时间'],
                    open=df['开盘价格'],high=df['最高价格'],
                    low=df['最低价格'],close=df['收盘价格'],
                    increasing_line_color= 'green', # 上升是红色
                    decreasing_line_color= 'red' # 下降是绿色
                    ,name = 'K线图')]).update_layout(layout)
    fig.update_layout(xaxis_rangeslider_visible=False) # 底部时间拖动条是否可见
    fig.add_trace(go.Scatter(x=df['开始时间'], y=df['ma_5'],name="5日均线"))
    plotly.offline.plot(fig)



    # def price(): # 价格与均线
    #     df = get_date()
    #     for window in [30,60,90]:
    #         df[str(window)+' mv'] = df['close'].rolling(window).mean()
    #     df = df.melt(id_vars='time',value_vars=['close', '30 mv','60 mv','90 mv'])
    #     fig = px.line(df, x="time", y="value", color="variable",labels={"variable": "指标"})\
    #         .update_layout(layout)
    #     fig.update_yaxes(visible=False, showticklabels=False)
    #     plotly.offline.plot(fig)



    # # wesockets 测试
    # url = 'wss://real.okex.com:8443/ws/v3'
    # channels = ["spot/candle60s:BTC-USDT"]
    
    # loop = asyncio.get_event_loop()

    # # 公共数据 不需要登录（行情，K线，交易数据，资金费率，限价范围，深度数据，标记价格等频道）
    # loop.run_until_complete(wbe.subscribe_without_login(url, channels))
    
    # loop.close()
    
    
     
#    # 获得K线数据
#    url = "https://www.okex.com/api/spot/v3/instruments/{instrument_id}/candles?" \
#        "granularity={granularity}&start={start}&end={end}".format(
#                                            instrument_id="BTC-USDT",
#                                            granularity="60", #间隔时间
#                                            start="2021-02-09T00:00:00.000Z",
#                                            end="2021-02-09T23:59:59.000Z",
#                                            )
#    print(url)
#    
#    # res = requests.get(url)
#    res = requests.get(url, proxies = proxies)
#    print(res.text)
#    content = json.loads(str(res.content,encoding='utf8'))
#    
#    
#    
#    content_pd = pd.DataFrame(content)
#    content_pd.columns = ['开始时间','开盘价格','最高价格','最低价格','收盘价格','交易量']
    

    
    
    
    

    