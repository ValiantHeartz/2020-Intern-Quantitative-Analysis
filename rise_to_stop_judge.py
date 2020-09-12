# encoding: utf-8

import sys;
reload(sys);
sys.setdefaultencoding('utf8')

import tushare as ts
import time
import pandas as pd
import numpy as np
import xlsxwriter
import datetime
import os

# 1、设置TUSHARE接口TOKEN
ts.set_token('cecf5814ed5b3708c7ba44fa1419fa250c5167bb2b37044cddb02292')  # 设置token，只需设置一次
pro = ts.pro_api()

# 2、自动获取上一交易日期
today = datetime.date.today()
my_date2 = today - datetime.timedelta(days = 1)
while ts.get_tick_data('000001', date=str(my_date2), src='tt') is None:  #get_tick_data为空
    my_date2 = my_date2 - datetime.timedelta(days = 1)
my_date2 = str(my_date2)
#print my_date2
#my_date2 = '2020-08-07' #str(my_date2) #上一交易日日期2020-08-07
my_date = my_date2[0:4] + my_date2[5:7] + my_date2[8:10]  # 变换格式后的上一交易日日期
'''
# 2.5、设置定时运行
print 'Now sleeping until 9:25:20...'
time_stamp_now = time.time()
time_stamp_092520 = time.mktime(time.strptime(str(today)+' 09:25:20','%Y-%m-%d %H:%M:%S'))  # 9.25.20对应的时间戳
time_delta = time_stamp_092520 - time_stamp_now
print 'Total ' + str(int(time_delta)) + 's left.'
for i in range(int(time_delta/60)):
    time.sleep(60)
    print 'Now ' + str(int(time_delta - (i+1)*60)) + 's left.'
time.sleep(time_delta - int(time_delta/60)*60)
print 'Now start to calculate...'
'''
# 3、条件一：筛选昨日涨停且开板次数大于1或昨日最终封板时间在14.30之后
to_rise_stock = pro.limit_list(trade_date=my_date, limit_type='U',
                               fields='ts_code,name,close,first_time,last_time,open_times,pct_chg')  # 获取昨日涨停的股票
time_stamp_1430 = time.mktime(time.strptime(my_date+' 14:30:00','%Y%m%d %H:%M:%S'))  # 14.30对应的时间戳，用于比较最终封板时间
#to_rise_stock.to_csv('C:/1.txt')
for i in range(len(to_rise_stock)):
    #if len(to_rise_stock.loc[i,'last_time'].encode('unicode-escape').decode('string_escape')) == 0:  #舍弃无
        #to_rise_stock = to_rise_stock.drop(i)
       #continue
    if to_rise_stock.loc[i,'pct_chg'] < 6 or \
            to_rise_stock.loc[i,'pct_chg'] > 11:  # 对涨跌幅进行限制，用以去掉ST股票（异常股票）和新股
        to_rise_stock = to_rise_stock.drop(i)
        continue
    if to_rise_stock.loc[i,'open_times'] < 1:# and \
            #time.mktime(time.strptime(my_date+to_rise_stock.loc[i,'last_time'],'%Y%m%d%H:%M:%S')) < time_stamp_1430:  #去掉下午2：30条件
        to_rise_stock = to_rise_stock.drop(i)
print '(1/3) Filtering price limit stock completed.' + '\n' \
      'Now filtering 0.85-condition-satisfied stock.'
to_rise_stock.to_csv('C:/0812.txt')

# 4、条件二：今日9:25成交量大于等于昨日最大单笔成交量的0.85
# 将条件一中获得的股票整理好放入stock_list
to_rise_stock = to_rise_stock.reset_index(drop=True)
length = to_rise_stock.iloc[:,0].size
stock_list = []
for i in range(length):
    stock_list.append(str(to_rise_stock.loc[i,'ts_code'])[0:6])
selected_stock_df = pd.DataFrame()#{"stock_code":"","rise_amount":""},index=["0"])
for stock in stock_list:
    #stock = '600139'
    tick_data = ts.get_tick_data(stock, date=my_date2, src='tt')
    #tick_data.to_csv('C:/5.txt')
    for i in range(5):
        tick_data = tick_data.drop(i)  #去掉前五笔数据，一般前几笔必输比较大，不作为判断依据
    today_tick_data = ts.get_realtime_quotes(stock)
    #print '委买一:'+ today_tick_data.loc[0,'b1_v']
    #print tick_data[tick_data['type']  == '买盘']
    yesterday_max_shou = int(tick_data.loc[tick_data[tick_data['type']  == '买盘']['volume'].idxmax(axis=0),'volume'])
    today_925_shou = int(today_tick_data.loc[0, 'volume'])/100
    #print stock + ' ' + str(today_925_volume)
    weimai_1_length = len(today_tick_data.loc[0, 'a1_v'].encode('unicode-escape').decode('string_escape'))
    today_open = float(today_tick_data.loc[0,'open'])
    yesterday_close = float(today_tick_data.loc[0,'pre_close'])
    rise_amount = (today_open-yesterday_close)*100/yesterday_close
    today_yesterday = round(float(today_925_shou)/yesterday_max_shou,2)
    if  today_yesterday >= 0.85 and rise_amount >= -1.5 and weimai_1_length != 0:
        selected_stock_df = selected_stock_df.append({"股票代码":stock,"股票名称":today_tick_data.loc[0,'name'],
                                "涨幅%":rise_amount,"今日/昨日":today_yesterday},ignore_index=True)
    #break
print '(2/3) Filtering 0.85-condition-satisfied stock completed.'

# 5、满足条件二的股票信息存放在selected_stock_df中，将其写入EXCEL
if selected_stock_df.empty:
    print '(3/3) Mission Completed.' + '\n' \
          'No stock matches 0.85 condition.'
    time.sleep(3)
else:
    print 'Now writing the stock data to Excel...'
    selected_stock_df = selected_stock_df.sort('涨幅%',ascending=False)
    selected_stock_df = selected_stock_df.reset_index(drop=True)
# 写入EXCEL：新建EXCEL文件，新建SHEET，对SHEET进行WRITE
    dir_path = 'price limit judge/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    workbook = xlsxwriter.Workbook(dir_path + 'price_limit ' + my_date + '.xlsx')
    worksheet = workbook.add_worksheet()
    workfomat = workbook.add_format({
    'bold' : False,                 #字体加粗
    'border' : 1,                    #单元格边框宽度
    'align' : 'left',          #对齐方式
    'valign' : 'vleft',         #字体对齐方式
    #'fg_color' : '#F4B084',         #单元格背景颜色
    })
    name_list = ['股票代码','股票名称','涨幅%','今日/昨日']
    worksheet.write(0, 0, name_list[0], workfomat)
    worksheet.write(0, 1, name_list[1], workfomat)
    worksheet.write(0, 2, name_list[2], workfomat)
    worksheet.write(0, 3, name_list[3], workfomat)
    for i in range(selected_stock_df.shape[0]):
        for j in range(len(name_list)):
            if name_list[j] == '涨幅%':
                worksheet.write(i + 1, j, round(selected_stock_df.loc[i, name_list[j]], 2), workfomat)
            else:
                worksheet.write(i+1, j, selected_stock_df.loc[i, name_list[j]], workfomat)
    workbook.close()
    print '(3/3) Mission complete!' + '\n' \
        'The Excel named ' + 'price_limit ' + my_date2 + '.xlsx' + ' has saved.'
    time.sleep(3)


