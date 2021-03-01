# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 22:21:37 2021

@author: xujianqiao
"""

import numpy as np
import pandas as pd
import datetime as dt
import pytz
pst = pytz.timezone('Asia/Shanghai')

# 作图
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly
#plotly.offline.init_notebook_mode(connected=True)



    
    
# 币、股波动模拟器
class Simulator():
    def __init__(self,stock_data,hand_money,time_col,open_col,close_col,high,low,volume):
        self.stock_data = stock_data
        self.hand_money = hand_money #手上的钱
        self.hold_share = 0 # 手上的股
        
        # col_names
        self.time_col = time_col # 时间
        self.open_col = open_col # 开盘价
        self.close_col = close_col # 收盘价
        self.high = high # 最高价
        self.low = low #最低价
        self.volume = volume #成交量
        
        self.price_col = self.close_col # 收盘价作为主价格
        
        self.col_map = {self.time_col :'time',
                        self.open_col :'open',
                        self.close_col :'close',
                        self.high :'high',
                        self.low :'low',
                        self.volume :'volume',
            }
        
        
        # 数据初始化
        self._data_fill_na() #空数据填充
        self._records_merge()
        self._add_mv(self.stock_data,self.price_col,periods=[3,5,10])
        
    # 缺失值填充，尽量先用前面的数据填，但缺失太多则往中间靠拢
    def _data_fill_na(self):
        reverse = 1
        while True:
            if reverse==1: # 先从前往后填
                self.stock_data.fillna(method='ffill',limit=1,inplace=True)
                reverse=0
            else: # 再从后往前填
                self.stock_data.fillna(method='bfill',limit=1,inplace=True)
                reverse=1
            if self.stock_data[self.price_col].isnull().sum()==0:
                break
        
    # 合并数据，降低数据的记录时间间隔，比如1分钟蜡烛图降低为3分钟蜡烛度
    def _records_merge(self,merge_length = 3):
        '''
        Parameters
        ----------
        merge_length : int, optional 
            将 N 条记录合并为1条. The default is 3.
            
        Returns
        -------
        None.

        '''
        p = np.ones([merge_length], int)
        groups = np.vstack([n*p for n in range(1,int(np.floor(self.stock_data.shape[0]/merge_length))+1)])#竖直拼接
        groups = groups.flatten()
        self.stock_data = self.stock_data.iloc[0:len(groups),:]
        self.stock_data = self.stock_data.groupby(groups).agg({self.time_col:'first',
                                 self.open_col:'first',
                                 self.close_col:'last',
                                 self.high:'max',
                                 self.low:'min',
                                 self.volume:'sum',
                                 #'Volume_(Currency)':'sum',
                                })
        
    # mean value
    @staticmethod
    def _add_mv(df,col,periods=[5]):
        for p in periods:
            df['mv_%s'%p] = df[col].rolling(p).mean()
    
    # 根据各类指标做一个决策，买、卖或持有
    def ask_decision(self,strategy):
        str_instance = strategy()
        
        for i in range(self.stock_data.shape[0]):
            row = self.stock_data.iloc[i,:]
            row.index = row.index.to_series().apply(lambda x: self.col_map[x] if x in self.col_map.keys() else x)
            row = row.to_dict()
            evt = Event(row)
            str_instance(evt)
            
            
            if i%1000==0:
                print(i)
            
        


# 咨询事件，咨询策略得到结果
class Event():
    def __init__(self,init_json):
        self.data = init_json


# 一个策略
class Stragegy():
    def __init__(self):
        pass
    
    # 根据获取的数据执行一个操作（买、卖或什么都不做）
    def action(self,inputs):
        print("no strategy init")
        pass
    
    
    
# Dow 道式策略
class DowStrategy(Stragegy):
    
    def __init__(self):
        Stragegy.__init__()
        self.last_price_window = np.full([5],np.nan)
        self.state_window = []
    
        # last_price_window = np.append(last_price_window,1)
        # last_price_window = last_price_window[1:]
    
    def action(self):
        pass



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
    data_2020 = pd.read_excel("data/kaggle/history_btcdata_2020.xlsx",engine="openpyxl",index_col=0)
    
    
    tmp = data_2020.head(100000)
    stragety = Stragegy()
    
    tmp.reset_index(inplace=True)
    sim = Simulator(tmp,5000,'str_time','Open','Close','High','Low','Volume_(BTC)')
    sim.ask_decision(stragety)
    
    tmp = sim.stock_data
    
    last_price_window = np.full([5],np.nan)

    

    
    
    # tmp_melt = pd.melt(tmp,id_vars='str_time',value_vars=['Close',price_col])
    
    
    

    
    layout = dict(title_x=0.5, # title居中
              font=dict(family="Times New Roman",size=20,color="RebeccaPurple"),
              # 字体设置
              coloraxis_colorbar=dict(xanchor="left",x=0.75,ticks="outside"),
              # 颜色条设置
              margin=dict(b= 40,l=40, r=40, t= 40)
              # 大小设置
             )
    

    
    i=0 #读取到的price位置
    last_price = np.nan # 上一个价格
    curr_price = np.nan # 当前价格
    curr_state = None # 对上一个价格是涨是跌
    last_state = None # 前一个点的趋势是什么
    judge_state = None # 当前整体判断是涨是跌
    is_judge_point = None # 当经历一个波峰或波谷后，进行一次判断
    judge_last = None # 刚经历的是波峰还是波谷（peak，bottom）
    peaks = [] # 所有的顶点（当前价格比前后价格都高的点）
    bottoms = [] # 前后价格比当前都低的点
    buy_points = [] # 所有买入的点
    sell_points = [] # 所有卖的点
    equal_points = []
    
    hand_money = 5000 #手上的钱
    keep_stock = 0 # 持仓的股
    equal = hand_money # 相当价值
    
    

    
    
    while True:
        if i%1000==0:
            print(i)
        curr_price = tmp.iloc[i,:][price_col]
        if curr_price-last_price>0:
            curr_state = 'increase'
        elif curr_price-last_price<0:
            curr_state = 'decrease'
        else:
            curr_state = 'flat'
            
        # 判断是否为波峰波谷
        if last_state == 'increase' and curr_state == 'decrease':
            peaks.append(tmp.iloc[i-1][['str_time',price_col]].to_list())
            is_judge_point=1
            judge_last='peak'
            
        elif last_state == 'decrease' and curr_state == 'increase':
            bottoms.append(tmp.iloc[i-1][['str_time',price_col]].to_list())
            is_judge_point=1
            judge_last='bottom'
            
        try:
            # 判断是否要买
            if is_judge_point:
                if judge_last=='bottom':
                    if peaks[-1][1] > peaks[-2][1] and bottoms[-1][1] > bottoms[-2][1]:
                        if hand_money>0:
                            buy_points.append(tmp.iloc[i][['str_time',price_col]].to_list()) #买
                            keep_stock += hand_money/tmp.iloc[i][price_col] # 钱焕成股票
                            hand_money = 0 # 手上没钱了
                        
            if is_judge_point:
                if judge_last=='peak':
                    if peaks[-1][1] < peaks[-2][1] and bottoms[-1][1] < bottoms[-2][1]:
                        if keep_stock>0:
                            sell_points.append(tmp.iloc[i][['str_time',price_col]].to_list()) #卖
                            hand_money += keep_stock*tmp.iloc[i][price_col] # 股票换成
                            keep_stock = 0 # 手上没股了   
                                           
                            
        except Exception as e:
            print(e)
            
                    
        equal = hand_money + keep_stock*tmp.iloc[i][price_col]
        equal_points.append([tmp.iloc[i]['str_time'],equal])
        
            
        # 循环结束
        i+=1
        last_price = curr_price
        last_state = curr_state
        is_judge_point=0 
        
        if i>tmp.shape[0]-1:
            break
    
    
    peaks = pd.DataFrame(peaks,columns = ['str_time',price_col])
    bottoms = pd.DataFrame(bottoms,columns = ['str_time',price_col])
    buy_points = pd.DataFrame(buy_points,columns = ['str_time',price_col])
    sell_points = pd.DataFrame(sell_points,columns = ['str_time',price_col])
    buy_sell_pd = pd.concat([buy_points,sell_points],axis=1)
    buy_sell_pd.columns = ['buy_time','buy_price','sell_time','sell_price']
    buy_sell_pd['spread'] = buy_sell_pd['sell_price'] - buy_sell_pd['buy_price'] 
    equal_points = pd.DataFrame(equal_points,columns = ['str_time','value'])
    



    # 价格和均线
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(x=tmp["str_time"], y=tmp[price_col],mode='lines'))

        
    fig.add_trace(go.Scatter(x=peaks['str_time'],y=peaks[price_col],mode='markers',
        marker=dict(
        color='red',
        size=5,)
        ))
    
    fig.add_trace(go.Scatter(x=bottoms['str_time'],y=bottoms[price_col],mode='markers',
        marker=dict(
        color='blue',
        size=5,)
        ))
    
    fig.add_trace(go.Scatter(x=buy_points['str_time'],y=buy_points[price_col],mode='markers',
        marker=dict(
        color='green',
        size=8,)
        ))
    fig.add_trace(go.Scatter(x=sell_points['str_time'],y=sell_points[price_col],mode='markers',
        marker=dict(
        color='brown',
        size=8,)
        ))
    fig.add_trace(go.Scatter(x=equal_points['str_time'],y=equal_points['value'],mode='lines',
        ))
    
    
    # 总收益率、最大回撤
    print(equal_points.iloc[-1]['value']/5000)
    
    
    # mv5 2.331
    # mv10 1.494
    # mv20 1.21
    
    
    for i in range(buy_sell_pd.shape[0]):
        if i%100==0:
            print(i)
        fig.add_annotation(
            x = buy_sell_pd.iloc[i,:]['sell_time'],  # arrows' head
            y = buy_sell_pd.iloc[i,:]['sell_price'],  # arrows' head
            ax = buy_sell_pd.iloc[i,:]['buy_time'],  # arrows' tail
            ay = buy_sell_pd.iloc[i,:]['buy_price'],  # arrows' tail
            xref = "x", yref = "y",
            axref = "x", ayref = "y",
            text='',  # if you want only the arrow
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor='black'
          )
        

    fig.update_layout(layout)
    plotly.offline.plot(fig)