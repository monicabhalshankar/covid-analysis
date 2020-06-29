import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import os
import math
import random
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
data3 = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv',header=0, delimiter=',', parse_dates=['Date'])
#data3['Active'] = data3['Confirmed'] - data3['Deaths'] - data3['Recovered']
data3[['Province/State']] = data3[['Province/State']].fillna('')
#data3[['Confirmed', 'Deaths', 'Recovered', 'Active']] = data3[['Confirmed', 'Deaths', 'Recovered', 'Active']].fillna(0)
#data3['Recovered'] = data3['Recovered'].astype(int)
data3.sample(6)
data2 = data3.groupby(['Country/Region'])['Confirmed'].agg('count')
data3 = data3.groupby(['Country/Region', 'Date', ])['Confirmed']
#data3['Country/Region'].value_counts()
tot = data2['Afghanistan']

data3
temp = data3.sum().diff().reset_index()

mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = 0
#temp.loc[mask, 'Deaths'] = np.nan
countries = temp['Country/Region'].unique()

x_axis = range(tot)

plt.figure( figsize=[20,800])

#print(countries)
#countries = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola']
#countries = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola',
#  'Argentina', 'Armenia', 'Australia', 'Austria',
# 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus',
# 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina',
# 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burma', 'Burundi']

#countries = ['Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic',
# 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo (Brazzaville)',
# 'Congo (Kinshasa)', 'Costa Rica', "Cote d'Ivoire", 'Croatia', 'Cuba', 'Cyprus',
# 'Czechia', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador',
# 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini',
# 'Ethiopia', 'Fiji', 'Finland', 'France']


#countries = ['Gabon', 'Gambia', 'Georgia', 'Germany',
# 'Ghana', 'Greece', 'Greenland', 'Grenada', 'Guatemala', 'Guinea',
#  'Guyana', 'Haiti', 'Holy See', 'Honduras', 'Hungary',
# 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy',
# 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya',  'Kuwait',
#  'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya',
# 'Liechtenstein', 'Lithuania', 'Luxembourg']

#countries = ['Madagascar', 'Malawi', 'Malaysia',
# 'Maldives', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova',
# 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Namibia' ,'Nepal',
# 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria',
# 'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Panama', 'Papua New Guinea',
# 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar']

#countries = ['Romania', 'Russia', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia',
# 'Saint Vincent and the Grenadines', 'San Marino', 'Sao Tome and Principe',
# 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore',
# 'Slovakia', 'Slovenia', 'Somalia', 'South Africa', 'South Korea',
# 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden',
# 'Switzerland', 'Syria']

#countries = ['Taiwan*', 'Tajikistan', 'Tanzania', 'Thailand',
# 'Timor-Leste', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'US',
# 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'Uruguay',
# 'Uzbekistan', 'Venezuela', 'Vietnam',  'Western Sahara',
# 'Yemen', 'Zambia', 'Zimbabwe']

#countries = ['Taiwan*']
#countries = [ 'Antigua and Barbuda', 'Guinea-Bissau', 'Kosovo', 'Kyrgyzstan', 'West Bank and Gaza']
#countries = ['Afghanistan', 'Australia', 'India', 'China', 'Pakistan', 'Nepal']
#plt.subplot(2,3,1)
#plt.plot(x_axis , temp.loc[temp['Country/Region']=='Australia', 'Confirmed'], label='country')
#plt.title('Australia')

#plt.subplot(2,3,2)
#plt.plot(x_axis , temp.loc[temp['Country/Region']=='India', 'Confirmed'], label='country')

#plt.subplot(2,3,3)
#plt.plot(x_axis , temp.loc[temp['Country/Region']=='Afghanistan', 'Confirmed'], label='country')
#countries = ['Afghanistan', 'Australia', 'India']
#for country in countries:
#    plt.plot(x_axis , temp.loc[temp['Country/Region']==country, 'Confirmed'], label=country)
    
#temp['Confirmed'] = temp['Confirmed'].apply(lambda x : np.nan if x==0 else x)
#temp['Confirmed'] = temp['Confirmed'].dropna()
cols = 1
#temp['Confirmed'] = temp['Confirmed'].apply(lambda x : np.nan if x==0 else x)
#temp['Confirmed'].dropna(inplace=True)
temp= temp[temp['Confirmed'] != 0]
rows = math.ceil(len(countries)/cols)
#print(temp)


def lorentzian(x, amp, cen, sig):
    return (amp/np.pi) * (sig/(x-cen)**2 + sig**2)

def sigmoid(x, A, M, P):
    return A / (1 + M*np.exp(-P*x))

def parabola(x, y, v, a):
    return y + v*x + 0.5*a*x**2

for idx, country in enumerate(countries):
    #test = temp.loc[temp['Country/Region']==country, 'Confirmed']
    
    #print(test.max())
    #print(int(idx))
    #color1 = (r+1, g+1, b+1)
    
    y_val = temp.loc[temp['Country/Region']==country, 'Confirmed']
    x_range = range(len(y_val))
    #print(y_val)
    s1 =0
    s2 =0 
    #print(x_val)
    print(y_val.max())
    x_val = []
    for i in x_range:
        #s1 = s1 + (i*j)
        #print(s1)
        #s2 = s2 + j
        x_val.append(i)
             
    #print(s1/s2)
    
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    t = np.linspace(0, len(y_val), num=len(y_val))
    #color1 = (r+1, g+1, b+1)
    y_val = temp.loc[temp['Country/Region']==country, 'Confirmed']
    #print(y_val)
    
    #p_par, p_cov = curve_fit(sigmoid, range(len(y_val)) , y_val, p0 = [0,2,0], maxfev=100000 )
    #pguess = [117, 41, 1.2]
    p_guess = [17,41,-30]
    
    # Fit the data
    #p_par, p_cov = curve_fit(lorentzian, range(len(y_val)) , y_val, p0 = pguess)
    
    p_par, p_cov = curve_fit(parabola,  t , y_val, p0 = p_guess)
    
    #y_fit = sigmoid(range(len(y_val)+10), p_par[0], p_par[1], p_par[2])
    
    #y_fit = lorentzian(range(len(y_val)+10), p_par[0], p_par[1], p_par[2])
    y_fit = parabola(t, p_par[0], p_par[1], p_par[2])
    plt.subplot(rows,cols,int(idx)+1)
    
    plt.plot(range(len(y_val)) , y_val, '-o', c = color)
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    plt.plot(range(len(y_val)) , y_fit,'k--', c = color)
    plt.title(country)

