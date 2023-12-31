#!/usr/bin/env python
# coding: utf-8

# In[1]:
import json

import streamlit as st
import pyupbit
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs


# In[2]:


# Set page configuration
st.set_page_config(
    page_title="HumanIndex",
    page_icon=":rocket:",
    layout="wide",
)


# In[2]:


def get_target_price(ticker):
    df = pyupbit.get_ohlcv(ticker)
    yesterday = df.iloc[-2]
    today_open = yesterday['close']
    yesterday_high = yesterday['high']
    yesterday_low = yesterday['low']
    target = today_open + (yesterday_high - yesterday_low) * 0.5
    return target


# In[3]:


def get_yesterday_ma5(ticker):
    # Get historical OHLCV data for the specified ticker
    df = pyupbit.get_ohlcv(ticker, interval='day', count=6)

    # Check if the DataFrame is empty or has insufficient data
    if df is None or len(df) < 6:
        return None  # Return None if data is not available

    # Calculate the 5-day moving average (MA5)
    ma5 = df['close'].rolling(window=5).mean()

    # Get yesterday's MA5 value
    yesterday_ma5 = ma5.iloc[-2]

    return yesterday_ma5


# In[4]:


def bullish_or_bearish():
 
    # Button to trigger the action
    
    button_clicked = st.button("Check Bullish or Bearish Right Now")

    if button_clicked:
        now = datetime.datetime.now()
        ma5 = get_yesterday_ma5("KRW-BTC")
        target_price = get_target_price("KRW-BTC")
        current_price = pyupbit.get_current_price("KRW-BTC")

        if (current_price > target_price) and (current_price > ma5):
            st.text("Continue Trading")
            st.text(f"ma5: {ma5}")
            st.text(f"target_price: {target_price}")
            st.text(f"current_price: {current_price}")
            st.text(f"Click Time: {now}")
        else:
            st.text("Stop Trading")
            st.text(f"Click Time: {now}")


# In[5]:


# Your trading strategy function
def ma5_above_and_range_above_strategy(df): 
    df['ma5'] = df['close'].rolling(window=5).mean().shift(1)
    df['range'] = (df['high'] - df['low']) * 0.5
    df['target'] = df['open'] + df['range'].shift(1)
    df['bull'] = df['open'] > df['ma5']
    fee = 0.001   # 0.05% *2(매수, 매도시)
    df['ror'] = np.where((df['high'] > df['target']) & df['bull'],
                          df['close'] / df['target'] - fee,
                          1)
    return df  # Add this line to return the modified DataFrame


# In[6]:


# MDD(최대가격 대비 낙폭), HPR(기간수익률) 계산 함수
def calculate_metrics(df):
    df['hpr'] = df['ror'].cumprod()
    df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100
    mdd = df['dd'].max()
    hpr = df['hpr'].iloc[-1]
    return mdd, hpr


# In[7]:


# 그래프 생성 함수
def generate_plot(df, ticker):
    close_prices = df["close"]
    hpr = df["hpr"]
    mdd = df["dd"]

    # 시계열 그래프 그리기
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(close_prices.index, close_prices, label='Close Price', marker='o', linestyle='-')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price (KRW)', color='tab:blue')
    ax1.tick_params('y', colors='tab:blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(close_prices.index, hpr, label='hpr', marker='o', linestyle='--', color='tab:red')
    ax2.set_ylabel('hpr (KRW)', color='tab:red')
    ax2.tick_params('y', colors='tab:red')
    ax2.legend(loc='upper right')

    ax3 = ax1.twinx()
    ax3.plot(close_prices.index, mdd, label='mdd', marker='o', linestyle='-.', color='tab:green')
    ax3.set_ylabel('mdd', color='tab:green')
    ax3.spines['right'].set_position(('outward', 60))
    ax3.tick_params('y', colors='tab:green')
    ax3.legend(loc='lower right')

    return fig


# In[8]:


#fetch_data
def fetch_data(selected_ticker, start_date, end_date):
    df = pyupbit.get_ohlcv(selected_ticker, interval="day", to=end_date)
    df = df[start_date.strftime('%Y-%m-%d'):]
    return df


# In[9]:


def authenticate_google_sheets():
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        st.markdown("### Authenticate with Google")
        st.markdown("Click the link below to authenticate with Google:")

        flow = InstalledAppFlow.from_client_secrets_file(
            "client_secret_998298439835-uv6ts5ta7agdj0fch4rtr6pf96su0mef.apps.googleusercontent.com.json",
            scopes=SCOPES,
            redirect_uri="https://perfecthumanindex.streamlit.app/"
        )
        auth_url, _ = flow.authorization_url(prompt='consent')
        st.markdown(f"[Click here to authenticate with Google]({auth_url})")

        # creds = flow.fetch_token(authorization_response=response)
        response_url = st.text_input("https://perfecthumanindex.streamlit.app/")
        parsed_url = urlparse(response_url)
        code = parse_qs(parsed_url.query)['code'][0]
        creds = flow.fetch_token(code=code)


        # with open("token.json", "w") as token:
        #     token.write(creds.to_json())
        # JSON 형식으로 변환하여 파일에 저장
        with open("token.json", "w") as token_file:
            json.dump(creds, token_file)

    return creds


# In[10]:


def get_bitcoin_price_difference():
    # Upbit API로부터 Bitcoin(BTC)의 최근 2일 동안의 종가 가져오기
    btc_prices = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=2)

    # 최근 2일의 종가에서 어제와 오늘의 종가 가져오기
    yesterday_price = btc_prices["close"].iloc[0]  # 첫 번째 행이 어제
    today_price = btc_prices["close"].iloc[1]  # 두 번째 행이 오늘

    # 어제와 오늘의 종가를 뺀 값을 계산
    price_difference = today_price - yesterday_price

    # 종가 차이가 양수이면 1, 음수이면 0 출력
    result = 1 if price_difference > 0 else 0
    
    # 날짜 정보 가져오기
    yesterday_date = btc_prices.index[0].strftime("%Y-%m-%d")
    today_date = btc_prices.index[1].strftime("%Y-%m-%d")
    
    # 출력
    #print(f"{yesterday_date} 종가: {yesterday_price} KRW")
    #print(f"{today_date} 종가: {today_price} KRW")
    #print(f"{yesterday_date}에서 {today_date} 종가 차이: {price_difference} KRW")
    #print(f"출력 결과: {result}")
    return result


# In[11]:


spreadsheet_id = "1CetVCZ2-iII39NUZj5AIFZiTYxX9Tw3nH2Ws7HR178M"
range_name = "sheet1"
creds = authenticate_google_sheets()
service = build("sheets", "v4", credentials=creds)

# Call the Sheets API
sheet = service.spreadsheets()
result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()

values = result.get("values", [])
df = pd.DataFrame(values)
print(df)


# In[12]:


# Streamlit 애플리케이션
def main():
    # 코인 리스트 가져오기
    tickers = pyupbit.get_tickers()

    # 탭 선택
    st.sidebar.title("Insights")
    selected_tab = st.sidebar.radio("Select Tab", ["Crypto Strategy Right Now", "Human Index"])

    if selected_tab == "Crypto Strategy Right Now":
        st.title("Crypto Strategy Now")
        # 코인 선택
        selected_ticker = st.selectbox("Select a cryptocurrency", tickers)

        # 기간 설정
        today = datetime.datetime.now().date()
        default_start_date = today.replace(day=1)  # Set the default start date to the first day of the current month
        start_date = st.date_input("Start Date", value=default_start_date)
        end_date = today
        # OHLCV 데이터 가져오기
        df = fetch_data(selected_ticker, start_date, end_date)
        # 전략 적용
        df = ma5_above_and_range_above_strategy(df)

        # MDD, HPR 계산
        mdd, hpr = calculate_metrics(df)

        # 결과 출력
        st.write(f"Selected Coin: {selected_ticker}")
        st.write(f"MDD (Maximum Drawdown): {mdd:.2f}%")
        st.write(f"HPR (Holding Period Return): {hpr:.2f}")
        # 그래프 결과 출력
        fig = generate_plot(df, selected_ticker)
        st.pyplot(fig)
        plt.close(fig)
        bullish_or_bearish()
    elif selected_tab == "Human Index":
        st.title("Human Index")
        spreadsheet_id = "1CetVCZ2-iII39NUZj5AIFZiTYxX9Tw3nH2Ws7HR178M"
        range_name = "sheet1"
        creds = authenticate_google_sheets()
        service = build("sheets", "v4", credentials=creds)
        
        # Call the Sheets API
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    
        values = result.get("values", [])
        df = pd.DataFrame(values[1:], columns=values[0])
        print(df)
        print(df.info())
        #st.write("Poll Data:")
        #st.write(df)  # Use the correct DataFrame name here
        # Convert "타임스탬프" column to datetime format if not already

        btc_price_today = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=1)
        today_close_time = btc_price_today.index[0]
#        print(df.columns)
        df['타임스탬프'] = pd.to_datetime(df['타임스탬프'].str.replace('오전', 'AM').str.replace('오후', 'PM'), format='%Y. %m. %d %p %I:%M:%S')
        # Extract date part from "타임스탬프" column
        df['타임스탬프_date'] = df['타임스탬프'].dt.date

        # Assuming you want to compare only the date part
        mask = df["타임스탬프_date"] == today_close_time.date()

        df["mask"] = mask
        print(df)
        df['UpDown_digit'] = df['예측값'].apply(lambda x: 1 if x == '상승' else 0)
        print(df)

        result = get_bitcoin_price_difference()
        result = int(result)
        #오늘 종가 날짜를 받는다
        btc_price_today = pyupbit.get_ohlcv("KRW-BTC",interval = "day", count = 1)
        today_close_time = btc_price_today.index[0]
        print(today_close_time)

        ##타임스탬프와 오늘 종가가 같은 데이터를 masking한다
        # Convert "타임스탬프" column to datetime format if not already
        df['타임스탬프'] = pd.to_datetime(df['타임스탬프'])
        # Extract date part from "타임스탬프" column
        df['타임스탬프_date'] = df['타임스탬프'].dt.date
        # Assuming you want to compare only the date part
        mask = df["타임스탬프_date"] == today_close_time.date()

        #오늘 상승하락 실적을 해당 날짜 target에 입력
        df.loc[mask, "target"] = result
        # Replace NaN values in the "target" column with the string "null"
        df["target"] = df["target"].fillna("null")
        # Drop the intermediate column if needed
        df = df.drop(columns=['타임스탬프_date'])
                ## 오늘 스코어 column을 만들어 데이터를 입력한다 
        if "score" not in df.columns:
            df["score"] = None  # or initialize it with your default value
        # Update the "score" column based on the condition
        df.loc[df["target"] == df["UpDown_digit"], 'score'] = 1
        df.loc[df["target"] != df["UpDown_digit"], 'score'] = np.nan
        print(df)
        #df.info()

        ## 누적 score 확인
        result_df = df.groupby("이메일 주소").agg({'score': 'sum'}).reset_index()
        result_df["ac_score"] = result_df["score"]
        result_df = result_df.drop(columns="score")
        #누적 score 상위부터 정렬
        rated_result_df = result_df.sort_values(by="ac_score", ascending=False)
        #상위 10개
        df_top_10 = rated_result_df.head(10)

        #결과 출력
        #print(df)
        #print(result_df)
        print(rated_result_df)
        #마스킹된 당일 필터 후 인덱스 계산
        # Convert "타임스탬프" column to datetime format if not already
        df['타임스탬프'] = pd.to_datetime(df['타임스탬프'])
        # Extract date part from "타임스탬프" column
        df['타임스탬프_date'] = df['타임스탬프'].dt.date
        # Assuming you want to compare only the date part
        mask = df["타임스탬프_date"] == today_close_time.date()
        df["mask"] = mask

        #마스크 된 애들만 필터랑(종가랑 같은 애들)
        df_filtered = df[df["mask"]]
        print(df_filtered)

        #필터링 된 오늘 투표데이터를 누적스코어 데이터랑 결합
        merged_df = pd.merge(df_filtered, result_df, on="이메일 주소", how="left")
        #print(merged_df)
        #print(merged_df["UpDown_digit"])
        #print(merged_df["score"])
        #print(merged_df["ac_score"])

        # 일반, 가중 인간지표 산출
        sum_per_email = np.sum(merged_df["UpDown_digit"] * merged_df["score"])
        weighted_sum_per_email = np.sum(merged_df["UpDown_digit"] * merged_df["ac_score"])
        total_forecast_num = merged_df["UpDown_digit"].count()

        #print(sum_per_email)
        #print(weighted_sum_per_email)
        #print(total_forecast_num)

        general_human_index = sum_per_email/total_forecast_num
        perfect_human_index = weighted_sum_per_email/total_forecast_num

        general_human_index_rounded = round(general_human_index, 2)
        perfect_human_index_rounded = round(perfect_human_index, 2)

        print("일반인간지표(0:하락, 1:상승):",general_human_index_rounded)
        print("가중인간지표(0:하락, 1:상승):",perfect_human_index_rounded)
        
        st.text(f"General Human Index (0: Bearish, 1: Bullish): {general_human_index_rounded}")
        st.text(f"Perfect Human Index (0: Bearish, 1: Bullish): {perfect_human_index_rounded}")
        st.text(f"Number of Participants: {total_forecast_num}")
if __name__ == '__main__':
    main()


# In[ ]:


# # 매수 : 5일이평선위 & 전일 변동성/2 이상, 매도 : 당일 자정

# def ma5_above_and_range_above_strategy(df): 
#     df['ma5'] = df['close'].rolling(window=5).mean().shift(1)
#     df['range'] = (df['high'] - df['low']) * 0.5
#     df['target'] = df['open'] + df['range'].shift(1)
#     df['bull'] = df['open'] > df['ma5']
#     fee = 0.001   # 0.05% *2(매수, 매도시)
#     df['ror'] = np.where((df['high'] > df['target']) & df['bull'],
#                           df['close'] / df['target'] - fee,
#                           1)
#     return df
# # MDD(최대가격 대비 낙폭), HPR(기간수익률) 계산 함수
# def calculate_metrics(df):
#     df['hpr'] = df['ror'].cumprod()
#     df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100
#     mdd = df['dd'].max()
#     hpr = df['hpr'][-2]
#     return mdd, hpr

# # 그래프 생성 함수
# def generate_plot(df, ticker):
#     close_prices = df["close"]
#     hpr = df["hpr"]
#     mdd = df["dd"]

#     # 시계열 그래프 그리기
#     fig, ax1 = plt.subplots(figsize=(10, 5))
#     ax1.plot(close_prices.index, close_prices, label='Close Price', marker='o', linestyle='-')
#     ax1.set_xlabel('Date')
#     ax1.set_ylabel('Close Price (KRW)', color='tab:blue')
#     ax1.tick_params('y', colors='tab:blue')
#     ax1.legend(loc='upper left')

#     ax2 = ax1.twinx()
#     ax2.plot(close_prices.index, hpr, label='hpr', marker='o', linestyle='--', color='tab:red')
#     ax2.set_ylabel('hpr (KRW)', color='tab:red')
#     ax2.tick_params('y', colors='tab:red')
#     ax2.legend(loc='upper right')

#     ax3 = ax1.twinx()
#     ax3.plot(close_prices.index, mdd, label='mdd', marker='o', linestyle='-.', color='tab:green')
#     ax3.set_ylabel('mdd', color='tab:green')
#     ax3.spines['right'].set_position(('outward', 60))
#     ax3.tick_params('y', colors='tab:green')
#     ax3.legend(loc='lower right')

#     # 그래프를 이미지로 변환
#     image_stream = BytesIO()
#     plt.savefig(image_stream, format='png')
#     image_stream.seek(0)
#     plt.close()

#     # 이미지를 base64로 인코딩하여 반환
#     return base64.b64encode(image_stream.read()).decode('utf-8')

# # Streamlit 애플리케이션
# def main():
#     # 페이지 제목
#     st.title("Crypto Analysis with Streamlit")

#     # 코인 리스트 가져오기
#     tickers = pyupbit.get_tickers()

#     # 코인 선택
#     selected_ticker = st.selectbox("Select a cryptocurrency", tickers)

#     # 기간 설정
#     start_date = st.date_input("Start Date", datetime.datetime(2021, 1, 1))
#     end_date = st.date_input("End Date", datetime.datetime.now())

#     # OHLCV 데이터 가져오기
#     df = pyupbit.get_ohlcv(selected_ticker, interval="day", to=end_date)
#     df = df[start_date.strftime('%Y-%m-%d'):]

#     # 전략 적용
#     ma5_above_and_range_above_strategy(df)

#     # MDD, HPR 계산
#     mdd, hpr = calculate_metrics(df)

#     # 결과 출력
#     st.write(f"Selected Coin: {selected_ticker}")
#     st.write(f"MDD (Maximum Drawdown): {mdd}%")
#     st.write(f"HPR (Holding Period Return): {hpr}")

#     # 그래프 출력
#     plot_data = generate_plot(df, selected_ticker)
#     st.image(plot_data, use_column_width=True)

# if __name__ == '__main__':
#     main()


# In[27]:


# # '22.7월 이후 알트 저점 대비 HPR 상위 10개 선정 후 그래프 그리기

# def get_hpr(ticker):
#     # 가져올 기간 설정
#     start_date = datetime.datetime(2022, 7, 1)
#     now = datetime.datetime.now() 
#     # OHLCV 데이터 가져오기
#     df = pyupbit.get_ohlcv(ticker, interval="day", count=20000, to=now)  # end date로도 변경 가능
#     df = df[start_date.strftime('%Y-%m-%d'):]
#     min_price = df['close'].min()
#     #print("'22.7월 이후 저점 :", min_price)
#     current_price = pyupbit.get_current_price(ticker)
#     #print(f"현재가 {ticker}: {current_price}")
#     hpr = current_price/min_price
#     #print("hpr:", hpr)
#     return hpr

# tickers = pyupbit.get_tickers()  
# hprs = []
# for ticker in tickers:
#     hpr = get_hpr(ticker)
#     hprs.append((ticker, hpr))

# sorted_hprs = sorted(hprs, key=lambda x: x[1], reverse=True)

# print(sorted_hprs[-5:])
# #     df['range'] = (df['high'] - df['low']) * 0.5
# #     df['target'] = df['open'] + df['range'].shift(1)
#     df['bull'] = df['open'] > df['ma5']

#     fee = 0.0032
#     df['ror'] = np.where((df['high'] > df['target']) & df['bull'],
#                           df['close'] / df['target'] - fee,
#                           1)

#     df['hpr'] = df['ror'].cumprod()
#     df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100
#     return df['hpr'][-2]


# tickers = pybithumb.get_tickers()

# hprs = []
# for ticker in tickers:
#     hpr = get_hpr(ticker)
#     hprs.append((ticker, hpr))

# sorted_hprs = sorted(hprs, key=lambda x:x[1])
# print(sorted_hprs[-5:])


# In[28]:



# #MDD(최대가격 대비 낙폭), HPR(기간수익률) 계산


# # ticker 전 코인 다 그래프화 
# tickers = pyupbit.get_tickers()  
# for ticker in tickers:

#     # Ticker 설정
#     #ticker = "KRW-SAND"

#     # 가져올 기간 설정
#     start_date = datetime.datetime(2021, 1, 1)
#     #end_date = datetime.datetime(2022, 1, 10)
#     now = datetime.datetime.now()
#     # OHLCV 데이터 가져오기
#     df = pyupbit.get_ohlcv(ticker, interval="day", count=20000, to=now) # end date로도 변경 가능
#     df = df[start_date.strftime('%Y-%m-%d'):]

#     # 전략 입력
#     ma5_above_and_range_above_strategy(df)

#     # HPR, MDD 출력
#     df['hpr'] = df['ror'].cumprod()
#     df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100
#     print("MDD: ", df['dd'].max())
#     print("HPR: ", df['hpr'][-2])
#     df.to_excel("larry_ma1.xlsx", encoding='utf-8')

#     # 그래프화 
#     close_prices = df["close"]
#     hpr = df["hpr"]
#     mdd = df["dd"]
#     # 시계열 그래프 그리기
#     fig, ax1 = plt.subplots(figsize=(10, 5))

#     # 왼쪽 Y 축 (종가)
#     ax1.plot(close_prices.index, close_prices, label='Close Price', marker='o', linestyle='-')
#     ax1.set_xlabel('Date')
#     ax1.set_ylabel('Close Price (KRW)', color='tab:blue')
#     ax1.tick_params('y', colors='tab:blue')
#     ax1.legend(loc='upper left')

#     # 오른쪽 Y 축 (hpr)
#     ax2 = ax1.twinx()
#     ax2.plot(close_prices.index, hpr, label='hpr', marker='o', linestyle='--', color='tab:red')
#     ax2.set_ylabel('hpr (KRW)', color='tab:red')
#     ax2.tick_params('y', colors='tab:red')
#     ax2.legend(loc='upper right')

#     # 오른쪽 Y 축 (mdd)
#     ax3 = ax1.twinx()
#     ax3.plot(close_prices.index,mdd, label='High Price', marker='o', linestyle='-.', color='tab:green')
#     ax3.set_ylabel('mdd', color='tab:green')
#     ax3.spines['right'].set_position(('outward', 60))  # 오른쪽 축 이동
#     ax3.tick_params('y', colors='tab:green')
#     ax3.legend(loc='lower right')

#     plt.title(f'{ticker} Open and Close Prices Over Time')
#     plt.grid(True)
#     plt.show()


# In[24]:




# def get_hpr(ticker):
#     df = pybithumb.get_ohlcv(ticker)
#     df.info()
#     df['ma5'] = df['close'].rolling(window=5).mean().shift(1)
#     df['range'] = (df['high'] - df['low']) * 0.5
#     df['target'] = df['open'] + df['range'].shift(1)
#     df['bull'] = df['open'] > df['ma5']

#     fee = 0.0032
#     df['ror'] = np.where((df['high'] > df['target']) & df['bull'],
#                           df['close'] / df['target'] - fee,
#                           1)

#     df['hpr'] = df['ror'].cumprod()
#     df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100
#     return df['hpr'][-2]


# tickers = pybithumb.get_tickers()

# hprs = []
# for ticker in tickers:
#     hpr = get_hpr(ticker)
#     hprs.append((ticker, hpr))

# sorted_hprs = sorted(hprs, key=lambda x:x[1])
# print(sorted_hprs[-5:])


# In[ ]:




