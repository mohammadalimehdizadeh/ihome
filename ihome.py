import requests as rq
import re
from bs4 import BeautifulSoup
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import datasets, linear_model, neural_network, gaussian_process, svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import datetime
import sqlite3
from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot
from matplotlib import pyplot as plt

def main(number):
# Initialization
    now = datetime.datetime.now()
    Dates = list()
    Prices = list()
    prop = list()
    Locations = list()
    Props = list()
# Process
    print('***Page*** <<<%i>>>'%number)
    print('connecting the page')
    url = re.sub(r'\d+/$',str(number),\
        'https://www.ihome.ir/%D8%AE%D8%B1%DB%8C%D8%AF-%D9%81%D8%B1%D9%88%D8%B4/%D8%A2%D9%BE%D8%A7%D8%B1%D8%AA%D9%85%D8%A7%D9%86/%D8%AA%D9%87%D8%B1%D8%A7%D9%86/2/',)
    while(True):
        try:
            r = rq.get(url)
            break
        except:
            pass
    r = r.text
    page = BeautifulSoup(r, 'html.parser')
    prices = page.find_all('div',attrs={'class':'price'})
    for home in prices[1:]:
        if re.findall(r'[\d\,]+',home.text) != []:
            Prices.append(int(re.sub(r'\,', '', (re.findall(r'[\d\,]+',home.text)[0]))))
        else:
            Prices.append(-1)

    locations = page.findAll('div' , attrs={'class':'location'})
    for home in locations[:]:
        Locations.append(home.span.text)

    areas = page.find_all('ul' , attrs={'class':"left slider_pinfo"})
    for home in areas:
        prop.append(re.findall(r'[\d\,]+' , home.text))
        if len(prop[-1]) == 2:
            Props.append((int(prop[-1][0]) ,int(re.sub(r'\,', '', prop[-1][1])) ,-1))
        elif len(prop[-1]) == 3:
            Props.append((int(prop[-1][0]) ,int(re.sub(r'\,', '', prop[-1][1])) ,int(prop[-1][2])))
        elif len(prop[-1]) == 1:
            Props.append((-1, int(re.sub(r'\,', '', prop[-1][0])) ,-1))
        else:
            pass

    dates = page.find_all('span' , attrs={'class':'date left'})
    for home in dates:
        date = re.findall(r'([\d\,]+) (\w+)' ,home.text)
        if date[0][1] == 'ثانیه':
            seconds = int(date[0][0])
            Dates.append(now-datetime.timedelta(seconds = seconds))
        elif date[0][1] == 'دقیقه':
            minutes = int(date[0][0])
            Dates.append(now-datetime.timedelta(minutes = minutes))
        elif date[0][1] == 'ساعت':
            hours = int(date[0][0])
            Dates.append(now-datetime.timedelta(hours = hours))
        elif date[0][1] == 'روز':
            days = int(date[0][0])
            Dates.append(now-datetime.timedelta(days = days))
        elif date[0][1] == 'هفته':
            weeks = int(date[0][0])
            Dates.append(now-datetime.timedelta(weeks = weeks))
        elif date[0][1] == 'ماه':
            days = 30*int(date[0][0])
            Dates.append(now-datetime.timedelta(days = days))
        else:
            print('date frame is not defined !')

    solds = page.find_all('span' , attrs={'class':'ribbon sold'}) # 1 = sold , 0 = not sold
    solds = ['not sold']*(Prices.__len__()-solds.__len__()) + ['sold']*(solds.__len__())
    print('data are collected successfully for page <<<%s>>> :)'%number)
    return Prices, Props, Dates, Locations, solds

def database(prices, num_rooms, areas, ages, dates, locations, solds):
    sqlite3.register_adapter(np.int64, lambda val: int(val))
    sqlite3.register_adapter(np.int32, lambda val: int(val))
    print('connecting to database')
    conn = sqlite3.connect('ihome.sqlite')
    cur = conn.cursor()

    cur.execute('''
    CREATE TABLE IF NOT EXISTS ihome (price INTEGER, area INTEGER, num_rooms INTEGER, age INTEGER,\
        location TEXT, sold TEXT, date TEXT)''')

    for price, num_room, area, age, date, location, sold in zip(prices, num_rooms, areas, ages, dates, locations, solds):
        cur.execute('SELECT * FROM ihome WHERE price = ? AND num_rooms = ? AND area = ? AND location = ?'\
            , (price,num_room, area, location))
        row = cur.fetchone()
        if row is None:
            print('this home is new.')
            cur.execute('''INSERT INTO ihome (price, area, num_rooms, age, location, sold, date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)''', (price, area, num_room, age, location, sold, str(date)))
        else:
            print('this home was inserted previously!')
            pass
        conn.commit()


    cur.close()
    print('data are written in database successfully. :)')

def execute(pages):
    # data collection
    prices, props, dates, locations, solds = list(), list(), list(), list(), list()
    for page in range(pages[0],pages[1]+1):
        price, prop, date, location, sold = main(page)
        prices = prices + price
        props = props + prop
        dates = dates + date
        locations = locations + location
        solds = solds + sold

    props = np.array(props)
    dates = np.array(dates)
    prices = np.array(prices)

    num_rooms = props[:,0]
    areas = props[:,1]
    ages = props[:,2]

    database(prices, num_rooms, areas, ages, dates, locations, solds)

def ML():
    print('connecting to database.')
    con = sqlite3.connect('ihome.sqlite')
    cur = con.cursor()
    cur.execute('select price , area , num_rooms from ihome where price > 100e6 and price < 200e9 and num_rooms < 13 and num_rooms > 1 and area > 50 and area < 5000')
    print('reading data...')
    data = cur.fetchall()
    print('data are reed successfully :)')
    x= np.array(list(map(lambda x: x[1:3], data)))
    y = np.array(list(map(lambda x: x[0], data)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:,0], x[:,1], y)
    ax.set_xlabel('area (m2)')
    ax.set_ylabel('number of rooms')
    ax.set_zlabel('price (tooman)')
    plt.show()
    offset = int(x.shape[0]*0.9)
    x_train = x[:offset]
    x_test = x[offset:]
    y_train = y[:offset]
    y_test = y[offset:]

    lr = LinearRegression()
    print('fitting Linear Regression...')
    lr.fit(x,y)
    print('Linear Regression fit successfully :)')

    dt = DecisionTreeRegressor()
    print('fitting Decision Tree...')
    dt.fit(x,y)
    print('Decision Tree fit successfully :)')

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    print('fitting SVR(RBF)...')
    svrrbf = svr_rbf.fit(x, y)
    print('SVR(RBF) fit successfully :)')
    print('fitting SVR(LIN)...')
    svrlin = svr_lin.fit(x, y)
    print('SVR(LIN) fit successfully :)')
    y_rbf = svrrbf.predict(x)
    y_lin = svrlin.predict(x)
    fig, (ax)
    plt.plot(x[:,0],y,'o',color = 'red',label = 'data')
    plt.plot(x[:,0],lr.predict(x), '-b', label = 'linear regression')
    plt.plot(x[:,0],dt.predict(x), '-y', label = 'decision tree')
    lw = 2
    plt.plot(x[:,0], y_rbf, color='navy', lw=lw, label='RBF model')
    plt.plot(x[:,0], y_lin, color='c', lw=lw, label='Linear model')
    plt.legend()
    plt.xlabel('Area in (m^2)')
    plt.ylabel('Price in (toman)')
    plt.show()
    1
    fig, ax1 = plt.subplots(nrows = 1, ncols = 1)
    xsample = np.array([[60,1],[90,1],[120,2],[150,2],[200,3],[300,3],[400,4],[600,4],[1000,5],[2000,7]])
    ax1.plot(xsample[:,0],lr.predict(xsample), '-b', label = 'linear regression')
    ax1.plot(xsample[:,0],dt.predict(xsample), '-y', label = 'decision tree')
    lw = 2
    ax1.plot(xsample[:,0], svrrbf.predict(xsample), color='navy', lw=lw, label='RBF model')
    ax1.plot(xsample[:,0], svrlin.predict(xsample), color='c', lw=lw, label='Linear model')
    #plt.plot(x, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.legend()
    plt.xlabel('A0rea in (m^2)')
    plt.ylabel('Price in (toman)')
    plt.show()
    1
    while(True):
        perd = input('Do you want to perdict your home price?(y/n)')
        if perd=='y':
            area0 = float(input('please enter area in (m2): '))
            num_rooms0 = int(input('please enter number of rooms: '))
            sample = np.array([[area0,num_rooms0]])
            plt.figure()
            plt.bar(['Linear Regression','Decision Tree','SVR(kernel=Linear)','SVR(kernel=RBF)']\
                ,[lr.predict(sample)[0],dt.predict(sample)[0],svr_lin.predict(sample)[0],svr_rbf.predict(sample)[0]])
            plt.show()
        elif perd=='n':
            break
print('''

    This code read data from <<<ihome.ir>>> and save them in a database
    by sqlite3 and analyze data by different Machine Learning methods include:
    Linear Regrission, Decision Tree and Support Vector Regression by 
    Linear and Radial Basis Function (RBF) and ask user to price his/her
    home by it's area in (m2) and number of rooms in tooman.
    
                *** Thanks for your attention ***                       \n\n''')
while (True):
    dbupdate = input('Do you want update database?(y/n)')
    if dbupdate=='y':
        page0 = int(input('please enter page number that you want to start collecting data: '))
        page1 = int(input('please enter page number that you want to end collecting data: '))
        execute([page0,page1])
    elif dbupdate=='n':
        break
ML()
print('\n<<<the end>>>\n')