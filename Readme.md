量化分析项目归纳总结：  
1、不同版本的python使用pip时要加python27 -m pip install ...  
2、移植代码时要注意模块的版本python27 -m pip install matplotlib==2.0.2  
3、quotes经过合并处理，会损失许多K线，部分股票日期可能相差一两天，属于正常现象  
4、为了美观展示，横坐标日期与上方K线有2天的漂移，不影响趋势判断  
5、时间戳：timeArray = time.strptime(a, "%Y%m%d %H:%M:%S") # 先将时间转换为时间矩阵，注意原时间格式  
timeStamp = int(time.mktime(timeArray))   #此时间戳精确到秒
6、dataframe：  
（1）取值：  
a) iloc，即index locate 用index索引进行定位，所以参数是整型，如：df.iloc[10:20, 3:5]  
b) loc，则可以使用column名和index名进行定位，如：df.loc[‘image1’:‘image10’, ‘age’:‘score’]  
（2）删除：  
a) pd.drop(0) # 0为index,要想批量删除，可以使用for循环，此法只删除视图，pd=pd.drop(0)相当于真正删除  
b) del pd...... #此法直接真正删除  
7、pycharm插件库：http://search.gipoco.com/cached/1541090/ （在设置里添加）  
8、一次性换源：-i https://pypi.tuna.tsinghua.edu.cn/simple  
9、tushare==1.2.60，pandas==0.16.2，numpy==1.13.3，slsxwriter==1.3.0,datetime==4.3,pyinstaller==3.6