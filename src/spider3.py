# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 22:10:21 2021

@author: xujianqiao
https://min-api.cryptocompare.com/documentation?key=Historical&cat=DailyHistoMinute
"""
   
import requests
import pandas as pd
import datetime as dt
    
if __name__ == "__main__":
    
    api_key = "66410cd39f42f664a9ef17ef857af0be8f7702148cfb15f17e83647886334a40"
    # 获得K线数据
    url = "https://min-api.cryptocompare.com/data/histo/minute/daily?" \
        "fsym={fsym}&tsym={tsym}&date={date}&api_key={api_key}"

    

    
    
    
    start_dt = "2020-01-01"
    end_dt = "2020-01-31"
    
    day = start_dt
    pd_list = []
    
    while True:
        print(day)
        req_url = url.format(
                            fsym="BTC",
                            tsym="USD",
                            date=day, #间隔时间
                            api_key = api_key
                            )
        print(req_url)
        res = requests.get(req_url)
        text_list = res.text.split("\n")
        text_list = [row.split(",") for row in text_list]
        text_pd = pd.DataFrame(text_list[1:],columns = text_list[0])
        pd_list.append(text_pd)
        
        tmp_dt = dt.datetime.strptime(day,"%Y-%m-%d") + dt.timedelta(days=1)
        day = tmp_dt.strftime("%Y-%m-%d")
        break
        if day>end_dt:
            break
        
        
    data = pd.concat(pd_list,axis=1)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        