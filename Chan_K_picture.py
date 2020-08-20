# encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')

#### 图标签设置为中文
import matplotlib.pyplot as plt
from matplotlib.pylab import style
style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import my_chan as chan
import matplotlib as mat
import numpy as np
import datetime
import time
import pandas as pd

#### macd
import talib
import tushare as ts

ts.set_token('cecf5814ed5b3708c7ba44fa1419fa250c5167bb2b37044cddb02292')  # 设置token，只需设置一次
pro = ts.pro_api()

from numpy.core.multiarray import array
from numpy import nan
import os

####jqdatasdk
from jqdatasdk import *
auth('13696773255','773255') #账号是申请时所填写的手机号；密码为聚宽官网登录密码，新申请用户默认为手机号后6位

#### 从stock_code_list.txt中读取股票池
stock_code_list = []
with open("stock_code_list.txt", "r") as f:  # 打开文件
    stock_code_list = [i[:-1].split(',') for i in f.readlines()][0]  # 读取文件

#### 主要功能函数封装为draw_chan_picture这一个函数
def draw_chan_picture(stock_code,end_date,stock_frequency,dir):
    # ================需要修改的参数==============
    #stock_code = '300750' # 股票代码 XSHG 上海证券交易所 XSHE 深圳证券交易所
    #start_date = '2020-02-03'
    #end_date = '2020-07-23' # 最后生成k线日期
    stock_days =60 # 看几天/分钟前的k线　
    x_jizhun = 20 #x轴展示的时间距离  5：日，40:30分钟， 48： 5分钟
    # stock_frequency = '5m' # 1d日线， 30m 30分钟， 5m 5分钟，1m 1分钟
    #stock_frequency = '30m' # 1d日线， 30m 30分钟， 5m 5分钟，1m 1分钟
    chanK_flag = True # True 看缠论K线， False 看k线
    #dir = "C:/Users/lenovo056/Desktop/"
    # ============结束==================


    initial_trend = "down"
    cur_ji = 1 if stock_frequency=='1d' else 2 if stock_frequency=='30m' else 3 if stock_frequency=='5m' else 4

    '''
    #该接口没有权限
    quotes = ts.pro_bar(ts_code='000001.SZ', start_date='20200501', end_date='20200731')   #可以获取任意数据，是通用接口adj='qfq', , freq='30min'
    #print quotes,len(quotes)
    quotes['money']='0'
    del quotes['amount']
    del quotes['pct_chg']
    del quotes['change']
    del quotes['pre_close']
    del quotes['ts_code']
    quotes.rename(columns={'vol':'volume'}, inplace = True)
    quotes.rename(columns={'trade_date': 'date'}, inplace=True)
    quotes.set_index(['date'], inplace=True)
    print quotes
    '''
    '''
    #  老接口tushare
    #quotes = ts.get_h_data('300750', start='2020-01-01', end='2020-03-16', autype='qfq')  #此接口只能获取空值
    quotes = ts.get_k_data(code=stock_code, ktype='30', autype='hfq', start=start_date,end=end_date)  # 只能获取到请求日期为止前30天数据
    #print quotes
    quotes['money'] = '0'
    del quotes['amount']
    del quotes['code']
    del quotes['turnoverratio']
    quotes.set_index(['date'], inplace=True)
    # dt.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")-dt.timedelta(days=5)
    '''
    if stock_code[0] == '6':
        quotes = get_price(stock_code + '.XSHG', start_date, end_date, frequency=stock_frequency, skip_paused=False,
                           fq='pre')
    else:
        quotes = get_price(stock_code + '.XSHE', start_date, end_date, frequency=stock_frequency, skip_paused=False,
                           fq='pre')

    quotes = quotes.reset_index()
    quotes = quotes.rename(columns={'index':'date'})
    quotes[['date']] = quotes[['date']].astype(str)
    for i in range(len(quotes.index)):
        #print str(quotes.loc[i,'date'])[0:16]
        quotes.loc[i,'date'] = quotes.loc[i,'date'][0:10] + ' ' + quotes.loc[i,'date'][11:16]
    quotes.set_index(['date'], inplace=True)
    #quotes = get_price('000300.XSHG')[:2]
    #print quotes
    # 缠论k线
    quotes = chan.parse2ChanK(quotes, quotes.values) if chanK_flag else quotes
    # print quotes[quotes['volume'] == 0]
    quotes[quotes['volume']==0]=np.nan
    #print len(quotes)
    quotes= quotes.dropna()
    #print(len(quotes))
    Close=quotes['close']
    Open=quotes['open']
    High=quotes['high']
    Low=quotes['low']
    T0 = quotes.index.values
    #print T0
    length=len(Close)

    fig = plt.figure(figsize=(16, 16))


    ax1 = plt.subplot2grid((21,4),(0,0),rowspan=10,colspan=4)#,axisbg='#000000')

    X=np.array(range(0, length))
    pad_nan=X+nan

        #计算上 下影线
    max_clop=Close.copy()
    max_clop[Close<Open]=Open[Close<Open]
    min_clop=Close.copy()
    min_clop[Close>Open]=Open[Close>Open]

        #上影线
    line_up=np.array([High,max_clop,pad_nan])
    line_up=np.ravel(line_up,'F')
        #下影线
    line_down=np.array([Low,min_clop,pad_nan])
    line_down=np.ravel(line_down,'F')

        #计算上下影线对应的X坐标
    pad_nan=nan+X
    pad_X=np.array([X,X,X])
    pad_X=np.ravel(pad_X,'F')

        #画出实体部分,先画收盘价在上的部分
    up_cl=Close.copy()
    up_cl[Close<=Open]=nan

    up_op=Open.copy()
    up_op[Close<=Open]=nan

    down_cl=Close.copy()
    down_cl[Open<=Close]=nan
    down_op=Open.copy()
    down_op[Open<=Close]=nan

    even=Close.copy()
    even[Close!=Open]=nan

    #画出收红的实体部分
    pad_box_up=np.array([up_op,up_op,up_cl,up_cl,pad_nan])
    pad_box_up=np.ravel(pad_box_up,'F')
    pad_box_down=np.array([down_cl,down_cl,down_op,down_op,pad_nan])
    pad_box_down=np.ravel(pad_box_down,'F')
    pad_box_even=np.array([even,even,even,even,pad_nan])
    pad_box_even=np.ravel(pad_box_even,'F')

    #X的nan可以不用与y一一对应
    X_left=X-0.25
    X_right=X+0.25
    box_X=np.array([X_left,X_right,X_right,X_left,pad_nan])
    # print box_X
    box_X=np.ravel(box_X,'F')
    # print box_X
    #Close_handle=plt.plot(pad_X,line_up,color='k')

    vertices_up=array([box_X,pad_box_up]).T

    vertices_down=array([box_X,pad_box_down]).T
    vertices_even=array([box_X,pad_box_even]).T

    handle_box_up=mat.patches.Polygon(vertices_up,color='r',zorder=1)
    handle_box_down=mat.patches.Polygon(vertices_down,color='g',zorder=1)
    handle_box_even=mat.patches.Polygon(vertices_even,color='k',zorder=1)

    ax1.add_patch(handle_box_up)
    ax1.add_patch(handle_box_down)
    ax1.add_patch(handle_box_even)

    handle_line_up=mat.lines.Line2D(pad_X,line_up,color='k',linestyle='solid',zorder=0)
    handle_line_down=mat.lines.Line2D(pad_X,line_down,color='k',linestyle='solid',zorder=0)

    ax1.add_line(handle_line_up)
    ax1.add_line(handle_line_down)

    v=[0,length,Open.min()-0.5,Open.max()+0.5]
    plt.axis(v)
    '''
    T1 = T0[-len(T0):].astype(dt.date)/1000000000
    # print T0,T1
    Ti=[]
    for i in range(len(T0)/x_jizhun):
        a=i*x_jizhun
        d = dt.date.fromtimestamp(T1[a])
    #     print d
        T2=d.strftime('$%Y-%m-%d$')
        Ti.append(T2)
        #print tab
    d1= dt.date.fromtimestamp(T1[len(T0)-1])
    d2=(d1+datetime.timedelta(days=1)).strftime('$%Y-%m-%d$')
    Ti.append(d2)
    '''

    #ax1.set_xticks(np.linspace(-1,len(Close),10))#len(Ti)))
    x_display_list = range(-2,len(quotes),len(quotes)/10)
    x_display_list.append(len(quotes) + 1)
    ax1.set_xticks(x_display_list)
    #print x_display_list
    ll=Low.min()*0.97
    hh=High.max()*1.03
    ax1.set_ylim(ll,hh)

    #print quotes.index.values
    x_date_list = quotes.index.values.tolist()
    T_label_list = []
    for i in range(0, len(quotes) - 1, len(quotes)/10):
        #print i,len(quotes),len(quotes)/10
        T_label_list.append(x_date_list[i])
    T_label_list.append(x_date_list[len(quotes) - 1])
    #print x_date_list
    ax1.set_xticklabels(T_label_list)

    dat = pro.query('stock_basic', fields='symbol,name')
    company_name = list(dat.loc[dat['symbol'] == stock_code].name)[0]

    plt.title('股票名称:' + company_name + ' 股票代码:' + stock_code)
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right') #横坐标倾斜

    k_data = quotes
    k_values = k_data.values
    # 缠论k线
    chanK = quotes if chanK_flag else chan.parse2ChanK(k_data, k_values)
    fenTypes, fenIdx = chan.parse2ChanFen(chanK)
    biIdx, frsBiType, has_bi_error = chan.parse2ChanBi(fenTypes, fenIdx, chanK)
    #print  has_bi_error
    #print "笔", biIdx, frsBiType
    #print '股票代码', stock_code #get_security_info(stock_code).display_name
    #  3.得到分笔结果，计算坐标显示
    x_fenbi_seq = []
    y_fenbi_seq = []
    fenType_seq = []
    for i in range(len(biIdx)):
        if biIdx[i]:
            fenType = -frsBiType if i%2==0 else frsBiType
    #         dt = chanK['enddate'][biIdx[i]]
              # 缠论k线
            dt = chanK.index[biIdx[i]] if chanK_flag else chanK['enddate'][biIdx[i]]
            #print dt,biIdx[i]
    #         print k_data['high'][dt], k_data['low'][dt]
            #time_long = long(time.mktime((dt+datetime.timedelta(hours=8)).timetuple())*1000000000)
            time_long = chanK.index[biIdx[i]]
    #         print x_date_list.index(time_long) if time_long in x_date_list else 0
            if fenType == 1:
                plt.text(x_date_list.index(time_long), k_data['high'][dt],str(k_data['high'][dt]), ha='left', fontsize=12)
                x_fenbi_seq.append(x_date_list.index(time_long))
                y_fenbi_seq.append(k_data['high'][dt])
                fenType_seq.append(1)
            if fenType == -1:
                plt.text(x_date_list.index(time_long), k_data['low'][dt], str(k_data['low'][dt]), va='bottom', fontsize=12)
                x_fenbi_seq.append(x_date_list.index(time_long))
                y_fenbi_seq.append(k_data['low'][dt])
                fenType_seq.append(-1)

    #  在原图基础上添加分笔蓝线
    plt.plot(x_fenbi_seq,y_fenbi_seq)
    #print x_fenbi_seq
    #线段画到笔上
    xdIdxs,xfenTypes = chan.parse2ChanXD(frsBiType,biIdx,chanK)
    # print '线段', xdIdxs,xfenTypes
    x_xd_seq = []
    y_xd_seq = []
    for i in range(len(xdIdxs)):
        if xdIdxs[i]:
            fenType = xfenTypes[i]
    #         dt = chanK['enddate'][biIdx[i]]
              # 缠论k线
            dt = chanK.index[xdIdxs[i]] if chanK_flag else chanK['enddate'][xdIdxs[i]]
    #         print k_data['high'][dt], k_data['low'][dt]
            #time_long = long(time.mktime((dt+datetime.timedelta(hours=8)).timetuple())*1000000000)
            time_long = chanK.index[xdIdxs[i]]
    #         print x_date_list.index(time_long) if time_long in x_date_list else 0
            if fenType == 1:
                x_xd_seq.append(x_date_list.index(time_long))
                y_xd_seq.append(k_data['high'][dt])
            if fenType == -1:
                x_xd_seq.append(x_date_list.index(time_long))
                y_xd_seq.append(k_data['low'][dt])
    #             bottom_time = None
    #             for k_line_dto in m_line_dto.member_list[::-1]:
    #                 if k_line_dto.low == m_line_dto.low:
    #                     # get_price返回的日期，默认时间是08:00:00
    #                     bottom_time = k_line_dto.begin_time.strftime('%Y-%m-%d') +' 08:00:00'
    #                     break
    #             x_fenbi_seq.append(x_date_list.index(long(time.mktime(datetime.strptime(bottom_time, "%Y-%m-%d %H:%M:%S").timetuple())*1000000000)))
    #             y_fenbi_seq.append(m_line_dto.low)

    #  在原图基础上添加分笔蓝线
    plt.plot(x_xd_seq,y_xd_seq)

    #### 一买判断
    dif,dea,macd= talib.MACD(Close.values, fastperiod=12, slowperiod=26, signalperiod=9)
    def biaoji(name, start,end):
        plt.annotate(name,(start,end), weight = 'extra bold',
                color = 'black', xytext=(start, end*0.98),
                arrowprops=dict(facecolor='black', shrink=0.05))

    first_buy_point = 0
    if len(x_xd_seq) < 2:
        print 'XianDuan nums < 2, can not judge FirstPointToBuy'
    else:
        judge1 = (Close[x_xd_seq[len(x_xd_seq) - 2]] - Close[x_xd_seq[len(x_xd_seq) - 1]]) / (x_xd_seq[len(x_xd_seq) - 2] - x_xd_seq[len(x_xd_seq) - 1])
        judge2 = dif[x_xd_seq[len(x_xd_seq)-1]] != dif[x_xd_seq[len(x_xd_seq) - 2]:x_xd_seq[len(x_xd_seq) - 1]+1].min()
        date_maidian = (len(Close) - x_xd_seq[len(x_xd_seq) - 1])/4
        if judge1 < 0 and judge2 and date_maidian <= 5:
            print 'FirstPointToBuy discovered.'
            biaoji('FirstPointToBuy',x_xd_seq[len(x_xd_seq)-1],Low[x_xd_seq[len(x_xd_seq) - 1]])
            first_buy_point = 1

    ####中枢判断
    #### 中枢绘制
    # 获取中枢位置
    def get_pivot(fx_plot, fx_offset, fx_observe):
        #
        # 计算最近的中枢
        # 注意：一个中枢至少有三笔
        # fx_plot 笔的节点股价
        # fx_offset 笔的节点时间点（偏移）
        # fx_observe 所观测的分型点

        if fx_observe < 1:
            # 处理边界
            right_bound = 0
            left_bount = 0
            min_high = 0
            max_low = 0
            pivot_x_interval = [left_bount,right_bound]
            pivot_price_interval = [max_low, min_high]
            return pivot_x_interval, pivot_price_interval

    #    print len(fx_offset), fx_observe
        right_bound = (fx_offset[fx_observe]+fx_offset[fx_observe-1])/2
        # 右边界是所观察分型的上一笔中位
        left_bount = 0
        min_high = 0
        max_low = 0

        if fx_plot[fx_observe] >= fx_plot[fx_observe-1]:
            # 所观察分型的上一笔是往上的一笔
            min_high = fx_plot[fx_observe]
            max_low = fx_plot[fx_observe-1]
        else: # 所观察分型的上一笔是往下的一笔
            max_low = fx_plot[fx_observe]
            min_high = fx_plot[fx_observe-1]

        i = fx_observe - 1
        cover = 0 # 记录走势的重叠区，至少为3才能画中枢
        while (i >= 1):
            if fx_plot[i] >= fx_plot[i-1]:
                # 往上的一笔
                if fx_plot[i] < max_low or fx_plot[i-1] > min_high:
                    # 已经没有重叠区域了
                    left_bount = (fx_offset[i] + fx_offset[i+1])/2
                    break
                else:
                    # 有重叠区域
                    # 计算更窄的中枢价格区间
                    cover += 1
                    min_high = min(fx_plot[i], min_high)
                    max_low = max(fx_plot[i-1], max_low)

            elif fx_plot[i] < fx_plot[i-1]:
                # 往下的一笔
                if fx_plot[i] > min_high or fx_plot[i-1] < max_low:
                    # 已经没有重叠区域了
                    left_bount = (fx_offset[i] + fx_offset[i+1])/2
                    break
                else:
                    # 有重叠区域
                    # 计算更窄的中枢价格区间
                    cover += 1
                    min_high = min(fx_plot[i-1], min_high)
                    max_low = max(fx_plot[i], max_low)
            i -= 1


        if cover < 3:
            # 不满足中枢定义
            right_bound = 0
            left_bount = 0
            min_high = 0
            max_low = 0

        pivot_x_interval = [left_bount,right_bound]
        pivot_price_interval = [max_low, min_high]
        return pivot_x_interval, pivot_price_interval,i
    # 绘制中枢方框
    def plot_pivot(ax, pivot_date_interval, pivot_price_interval):
        start_point = (pivot_date_interval[0], pivot_price_interval[0])
        width = pivot_date_interval[1] - pivot_date_interval[0]
        height =  pivot_price_interval[1] - pivot_price_interval[0]
        ax.add_patch(
            plt.Rectangle(start_point,    # (x,y)
                width,          # width
                height,         # height
                linewidth=2,
                edgecolor='y',
                facecolor='none')
        )
        return

    y_fenbi_seq = []
    for i in range(len(x_fenbi_seq)):
        if fenType_seq[i] == -1:
            y_fenbi_seq.append(Low[x_fenbi_seq[i]])
        else:
            y_fenbi_seq.append(High[x_fenbi_seq[i]])

    i = len(x_fenbi_seq) - 1
    while i > 1:
        pivot_x_interval, pivot_price_interval,temp = get_pivot(y_fenbi_seq, x_fenbi_seq, i)
        i = temp
        i -= 1
        plot_pivot(ax1, pivot_x_interval, pivot_price_interval)


    ####MACD画图
    ax2 = plt.subplot2grid((21,4),(12,0),rowspan=10,colspan=4)#,axisbg='#000000')
    plt.axis(v) #设定坐标范围
    #print dif,dea,macd
    macd_ll=min([min(pd.Series(dif).dropna()),min(pd.Series(dea).dropna()),2*min(pd.Series(macd).dropna())])*1.03
    macd_hh=max([max(pd.Series(dif).dropna()),max(pd.Series(dea).dropna()),2*max(pd.Series(macd).dropna())])*1.03
    # print max(pd.Series(dea).dropna())
    ax2.set_ylim(macd_ll,macd_hh)
    plt.grid(True) #设定网格线
    ax2.set_xticks(x_display_list)
    ax2.set_xticklabels(T_label_list)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    #ax2.set_xticklabels(Ti)

    for i in range(0,len(x_date_list)):
        if macd[i] > 0:
            plt.bar(i,macd[i]*2,color='r')
        else:
            plt.bar(i,macd[i]*2,color='g')
    #plt.bar(range(0,len(x_date_list)),macd*2)
    plt.plot(range(0,len(x_date_list)),dea,'y')
    plt.plot(range(0,len(x_date_list)),dif,'b')
    #plt.show()
    def save_plt(stock_code, quotes):
        dir_path = 'Chan_K_picture/' + str(end_date) + ' ChanK_Picture/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if has_bi_error:
            s = dir_path + 'Has_bi_error ' + stock_code + ' ' + quotes.index[len(quotes) - 1][5:10] + '.png'
            plt.savefig(s)
        elif first_buy_point:
            s = dir_path + 'FirstBuy ' + stock_code + ' ' + quotes.index[len(quotes) - 1][5:10] + '.png'
            plt.savefig(s)
        else:
            s = dir_path + stock_code + ' ' + quotes.index[len(quotes) - 1][5:10] + '.png'
            plt.savefig(s)

    save_plt(stock_code,quotes)
    plt.cla()
    plt.close("all")
    #print 'success'

#### 对股票池进行遍历的函数，每次保存一只股票的缠论K线图
def draw_stock_list(stock_code_list,end_date,stock_frequency,dir):
    num = 1
    for stock_code in stock_code_list:
        print 'status:' + str(num*100/len(stock_code_list)) + '%', time.strftime('%H:%M:%S',time.localtime(time.time())) + ' stock_code:' + stock_code
        draw_chan_picture(stock_code, end_date, stock_frequency, dir)
        num += 1

# ================需要修改的参数==============
end_date = datetime.date.today()#'2020-07-23' 今天的日期
day_length = 60  # 看多少天的股票数据
start_date = end_date - datetime.timedelta(days = day_length)
stock_frequency = '30m'  # 1d日线， 30m 30分钟， 5m 5分钟，1m 1分钟
#dir = 'C:/'  # 存储路径，实际上会在C盘下新建一个股票缠论图文件夹
# ================结束=======================

#### 画出股票列表里所有股票的缠论图
draw_stock_list(stock_code_list,end_date,stock_frequency,dir)
#stock_code = '000001'
#draw_chan_picture('300463',end_date,stock_frequency,dir)
print 'Success!'