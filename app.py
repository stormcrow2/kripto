from flask import Flask, render_template, jsonify
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import yfinance as yf
from newsapi import NewsApiClient
from scipy.signal import find_peaks
from finta import TA  # Ek teknik analiz göstergeleri için
import mplfinance as mpf  # Grafik analizi için
from scipy import stats  # İstatistiksel analiz için
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class AdvancedCryptoAnalyzer:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.newsapi = NewsApiClient(api_key='51349dc6bf92489bbfa326ee4921ec0d')
        self.scaler = StandardScaler()
        self.model = self._load_ml_model()
        
    def _load_ml_model(self):
        return None  # Basitleştirilmiş versiyon için ML modelini devre dışı bırakıyoruz
    
    def _create_ml_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 6)),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def get_market_data(self, symbol, timeframe='1h', limit=1000):
        try:
            print(f"Veri alınıyor: {symbol}")  # Debug için
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                raise Exception("Veri alınamadı")
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Veri alma hatası: {str(e)}")
            return pd.DataFrame()  # Boş DataFrame döndür

    def calculate_advanced_indicators(self, df):
        try:
            # Temel göstergeler
            df['RSI'] = talib.RSI(df['close'])
            df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
            df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
            df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)
            df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
            df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
            df['EMA_100'] = talib.EMA(df['close'], timeperiod=100)
            
            # MACD
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['close'])
            
            # Bollinger Bands
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'])
            
            # Momentum göstergeleri
            df['MOM'] = talib.MOM(df['close'], timeperiod=14)
            df['ADX'] = talib.ADX(df['high'], df['low'], df['close'])
            
            # Volatilite
            df['ATR'] = talib.ATR(df['high'], df['low'], df['close'])
            
            # Özel Hesaplamalar
            df['Volatility'] = df['close'].pct_change().rolling(window=14).std() * np.sqrt(365)
            df['Risk_Ratio'] = df['ATR'] / df['close'] * 100
            
            # Trend Gücü
            df['Trend_Strength'] = abs(df['EMA_20'] - df['EMA_50']) / df['EMA_50'] * 100
            
            # Hacim Trendi
            df['Volume_MA'] = df['volume'].rolling(window=20).mean()
            df['Volume_Trend'] = df['volume'] / df['Volume_MA']
            
            # VWAP hesaplama (manuel)
            df['Typical_Price'] = (df['high'] + df['low'] + df['close']) / 3
            df['VWAP'] = (df['Typical_Price'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            return df
        
        except Exception as e:
            print(f"Gösterge hesaplama hatası: {str(e)}")
            return df

    def get_news_sentiment(self, coin):
        try:
            # Kripto para haberleri
            news = self.newsapi.get_everything(
                q=f'{coin} cryptocurrency',
                language='en',
                sort_by='publishedAt',
                page_size=10,
                from_param=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            )
            
            if not news['articles']:
                # Alternatif arama dene
                news = self.newsapi.get_everything(
                    q=coin,
                    language='en',
                    sort_by='publishedAt',
                    page_size=10,
                    from_param=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                )
            
            sentiment_scores = []
            articles_data = []
            
            for article in news['articles']:
                if article['title'] and article['description']:
                    analysis = TextBlob(article['title'] + ' ' + article['description'])
                    sentiment_score = analysis.sentiment.polarity
                    sentiment_scores.append(sentiment_score)
                    
                    articles_data.append({
                        'title': article['title'],
                        'sentiment': sentiment_score,
                        'url': article['url']
                    })
            
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                sentiment_label = 'Pozitif' if avg_sentiment > 0.1 else 'Negatif' if avg_sentiment < -0.1 else 'Nötr'
            else:
                avg_sentiment = 0
                sentiment_label = 'Nötr'
            
            return {
                'sentiment_score': avg_sentiment,
                'sentiment_label': sentiment_label,
                'news_count': len(sentiment_scores),
                'articles': articles_data[:5]
            }
        except Exception as e:
            print(f"Haber analizi hatası: {str(e)}")
            return {
                'sentiment_score': 0,
                'sentiment_label': 'Veri alınamadı',
                'news_count': 0,
                'articles': [],
                'error': str(e)
            }

    def predict_price_movement(self, df):
        try:
            # Basit bir tahmin mekanizması kullanalım
            last_prices = df['close'].tail(14)  # Son 14 günlük veri
            price_change = (last_prices.iloc[-1] - last_prices.iloc[0]) / last_prices.iloc[0] * 100
            momentum = last_prices.pct_change().mean() * 100
            
            predicted_change = price_change * 0.7 + momentum * 0.3
            
            return {
                'predicted_change': predicted_change,
                'confidence_score': 0.6  # Sabit güven skoru
            }
        except:
            return {'predicted_change': 0, 'confidence_score': 0}

    def find_support_resistance(self, df):
        prices = df['close'].values
        peaks, _ = find_peaks(prices, distance=20)
        valleys, _ = find_peaks(-prices, distance=20)
        
        resistance_levels = sorted(prices[peaks][-3:])  # Son 3 direnç
        support_levels = sorted(prices[valleys][-3:])   # Son 3 destek
        
        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels
        }

    def analyze_market_structure(self, df):
        try:
            # Piyasa Yapısı Analizi
            last_price = df['close'].iloc[-1]
            sma_200 = df['SMA_200'].iloc[-1]
            vwap = df['VWAP'].iloc[-1] if 'VWAP' in df.columns else df['close'].rolling(20).mean().iloc[-1]
            
            # Trend Analizi
            long_term_trend = "YÜKSELEN" if last_price > sma_200 else "DÜŞEN"
            medium_term_trend = "YÜKSELEN" if df['EMA_50'].iloc[-1] > df['EMA_100'].iloc[-1] else "DÜŞEN"
            short_term_trend = "YÜKSELEN" if df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1] else "DÜŞEN"
            
            # Trend Gücü
            trend_strength = df['Trend_Strength'].iloc[-1]
            trend_strength_label = "GÜÇLÜ" if trend_strength > 5 else "ZAYIF"
            
            # Volatilite Analizi
            current_volatility = df['Volatility'].iloc[-1]
            avg_volatility = df['Volatility'].mean()
            volatility_state = "YÜKSEK" if current_volatility > avg_volatility else "DÜŞÜK"
            
            # Hacim Analizi
            volume_trend = "ARTAN" if df['Volume_Trend'].iloc[-1] > 1 else "AZALAN"
            
            # Risk Analizi
            risk_level = df['Risk_Ratio'].iloc[-1]
            risk_state = "YÜKSEK" if risk_level > 3 else "ORTA" if risk_level > 1.5 else "DÜŞÜK"
            
            return {
                'market_structure': {
                    'long_term_trend': long_term_trend,
                    'medium_term_trend': medium_term_trend,
                    'short_term_trend': short_term_trend,
                    'trend_strength': trend_strength_label,
                    'volatility_state': volatility_state,
                    'volume_trend': volume_trend,
                    'risk_level': risk_state
                },
                'key_levels': {
                    'current_price': last_price,
                    'vwap': vwap,
                    'sma_200': sma_200,
                    'volatility': f"{current_volatility:.2f}%",
                    'trend_strength': f"{trend_strength:.2f}%"
                }
            }
        except Exception as e:
            print(f"Piyasa yapısı analizi hatası: {str(e)}")
            return None

    def generate_trading_signals(self, df, sentiment_data, prediction_data, levels, market_structure):
        try:
            last_row = df.iloc[-1]
            signals = []
            confidence = 0
            risk_score = 0
            
            # Market structure kontrolü
            if market_structure and 'market_structure' in market_structure:
                # Piyasa Yapısı Analizi
                if market_structure['market_structure'].get('long_term_trend') == "YÜKSELEN":
                    signals.append({"signal": "AL", "reason": "Uzun vadeli yükselen trend", "weight": 2})
                    confidence += 0.2
                    risk_score -= 0.1
                else:
                    signals.append({"signal": "SAT", "reason": "Uzun vadeli düşen trend", "weight": 2})
                    confidence += 0.2
                    risk_score += 0.1

                # Volatilite Bazlı Risk Analizi
                if market_structure['market_structure'].get('volatility_state') == "YÜKSEK":
                    risk_score += 0.2
                    signals.append({"signal": "DİKKAT", "reason": "Yüksek volatilite - risk yüksek", "weight": 1})
                
                # Hacim Analizi
                if market_structure['market_structure'].get('volume_trend') == "ARTAN":
                    confidence += 0.1
                    signals.append({"signal": "GÜÇLÜ", "reason": "Artan işlem hacmi - trend güçlü", "weight": 1})

            # Temel teknik analiz sinyalleri
            # RSI Analizi
            if 'RSI' in last_row:
                if last_row['RSI'] < 30:
                    signals.append({"signal": "AL", "reason": "Aşırı satım (RSI < 30)", "weight": 2})
                    confidence += 0.2
                elif last_row['RSI'] > 70:
                    signals.append({"signal": "SAT", "reason": "Aşırı alım (RSI > 70)", "weight": 2})
                    confidence += 0.2

            # MACD Analizi
            if 'MACD' in last_row and 'MACD_Signal' in last_row:
                if last_row['MACD'] > last_row['MACD_Signal']:
                    signals.append({"signal": "AL", "reason": "MACD sinyal çizgisini yukarı kesti", "weight": 1})
                    confidence += 0.15
                elif last_row['MACD'] < last_row['MACD_Signal']:
                    signals.append({"signal": "SAT", "reason": "MACD sinyal çizgisini aşağı kesti", "weight": 1})
                    confidence += 0.15
            
            # Stop Loss ve Take Profit Önerileri
            if 'ATR' in last_row:
                atr = last_row['ATR']
                suggested_stop_loss = last_row['close'] - (2 * atr)
                suggested_take_profit = last_row['close'] + (3 * atr)
                
                # Risk/Ödül Oranı
                risk_reward_ratio = (suggested_take_profit - last_row['close']) / (last_row['close'] - suggested_stop_loss)
                
                # Pozisyon Büyüklüğü Önerisi
                max_risk_percentage = 0.02  # Maksimum %2 risk
                suggested_position_size = (max_risk_percentage * 100000) / (last_row['close'] - suggested_stop_loss)
            else:
                suggested_stop_loss = last_row['close'] * 0.95  # Varsayılan %5 stop loss
                suggested_take_profit = last_row['close'] * 1.1  # Varsayılan %10 take profit
                risk_reward_ratio = 2
                suggested_position_size = 1
                max_risk_percentage = 0.02

            # Sentiment analizi entegrasyonu
            if sentiment_data and sentiment_data.get('sentiment_score', 0) > 0.2:
                signals.append({"signal": "AL", "reason": "Pozitif piyasa duyarlılığı", "weight": 1})
                confidence += 0.1
            elif sentiment_data and sentiment_data.get('sentiment_score', 0) < -0.2:
                signals.append({"signal": "SAT", "reason": "Negatif piyasa duyarlılığı", "weight": 1})
                confidence += 0.1

            # Genel sinyal oluşturma
            buy_weight = sum(s['weight'] for s in signals if s['signal'] in ['AL', 'GÜÇLÜ AL'])
            sell_weight = sum(s['weight'] for s in signals if s['signal'] in ['SAT', 'GÜÇLÜ SAT'])
            
            return {
                'signals': signals,
                'position': 'GÜÇLÜ AL' if buy_weight > sell_weight + 2 else 'AL' if buy_weight > sell_weight else 
                           'GÜÇLÜ SAT' if sell_weight > buy_weight + 2 else 'SAT' if sell_weight > buy_weight else 'NÖTR',
                'risk_analysis': {
                    'risk_score': risk_score,
                    'stop_loss': suggested_stop_loss,
                    'take_profit': suggested_take_profit,
                    'risk_reward_ratio': risk_reward_ratio,
                    'suggested_position_size': suggested_position_size,
                    'max_loss_amount': max_risk_percentage * 100000
                },
                'confidence_score': min(confidence * 100, 100),
                'detailed_signals': signals
            }
        except Exception as e:
            print(f"Sinyal üretme hatası: {str(e)}")
            return {
                'signals': [],
                'position': 'NÖTR',
                'risk_analysis': {
                    'risk_score': 0,
                    'stop_loss': 0,
                    'take_profit': 0,
                    'risk_reward_ratio': 0,
                    'suggested_position_size': 0,
                    'max_loss_amount': 0
                },
                'confidence_score': 0,
                'detailed_signals': [],
                'error': str(e)
            }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze/<symbol>')
def analyze(symbol):
    try:
        analyzer = AdvancedCryptoAnalyzer()
        
        # Market verilerini al
        df = analyzer.get_market_data(f"{symbol}/USDT")
        
        # DataFrame boş ise hata döndür
        if df.empty:
            return jsonify({
                'error': 'Veri alınamadı',
                'message': f'{symbol} için veri bulunamadı'
            }), 400
        
        # Teknik analiz
        df = analyzer.calculate_advanced_indicators(df)
        
        # Piyasa yapısı analizi
        market_structure = analyzer.analyze_market_structure(df)
        
        # Haber ve sentiment analizi
        sentiment_data = analyzer.get_news_sentiment(symbol)
        
        # Fiyat tahmini
        prediction_data = analyzer.predict_price_movement(df)
        
        # Destek ve direnç seviyeleri
        levels = analyzer.find_support_resistance(df)
        
        # Trading sinyalleri
        signals = analyzer.generate_trading_signals(df, sentiment_data, prediction_data, levels, market_structure)
        
        # Son fiyat ve değişim hesaplama
        last_price = float(df['close'].iloc[-1])
        price_change = float(((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100))
        
        return jsonify({
            'market_structure': market_structure,
            'trading_signals': signals,
            'technical_data': df.tail().to_dict('records'),
            'last_price': last_price,
            'price_change_24h': price_change,
            'sentiment_analysis': sentiment_data,
            'price_prediction': prediction_data,
            'support_resistance': levels,
            'risk_analysis': signals.get('risk_analysis', {})
        })
        
    except Exception as e:
        print(f"Analiz hatası: {str(e)}")  # Sunucu logları için
        return jsonify({
            'error': 'Analiz hatası',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
