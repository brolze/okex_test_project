#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:44:01 2021

@author: xujianqiao

$ unset all_proxy && unset ALL_PROXY # 取消所有 socks 代理
$ pip install pysocks
"""

import requests
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
import datetime
import websocket_example as wbe
import asyncio
import websockets
import dateutil.parser as dp
import hmac
import base64
import zlib
import datetime as dt

import nest_asyncio
nest_asyncio.apply()



proxies={
    'http': 'http://127.0.0.1:1087',
    'https': 'https://127.0.0.1:1087'
}


res = requests.get("https://www.google.com", proxies = proxies)
print(res)

# # 获得K线数据
# url = "https://www.okex.com/api/spot/v3/instruments/{instrument_id}/candles?" \
#     "granularity={granularity}&start={start}&end={end}".format(
#                                         instrument_id="BTC-USDT",
#                                         granularity="86400", #间隔时间
#                                         start="2019-03-19T16:00:00.000Z",
#                                         end="2019-03-20T16:00:00.000Z",
#                                         )
# print(url)

# # res = requests.get(url)
# res = requests.get(url, proxies = proxies)
# print(res.text)





if __name__=="__main__":
    
    # 从文件读取账户信息
    with open('my_account.txt','r') as f:
        content = f.read()
    account_info = json.loads(content)
    api_key = account_info['apikey']
    secret_key = account_info['secretkey']
    passphrase = account_info['passphrase']
    
    proxies = {
        "http": "socks5://127.0.0.1:1086",
        "https": "socks5://127.0.0.1:1086",
    }

    # accountAPI = account.AccountAPI(api_key, secret_key, passphrase, False)    
    # indexAPI = index.IndexAPI(api_key, secret_key, passphrase, False)

    spotAPI = spot.SpotAPI(api_key, secret_key, passphrase, False)
    result = spotAPI.get_kline(instrument_id='BTC-USDT', start='2019-03-19T16:00:00.000Z',
                               end='2019-03-20T16:00:00.000Z', granularity='86400')


    # # wesockets 测试
    # url = 'wss://real.okex.com:8443/ws/v3'
    # channels = ["spot/candle60s:BTC-USDT"]
    
    # loop = asyncio.get_event_loop()

    # # 公共数据 不需要登录（行情，K线，交易数据，资金费率，限价范围，深度数据，标记价格等频道）
    # loop.run_until_complete(wbe.subscribe_without_login(url, channels))
    
    # loop.close()

    
    

    