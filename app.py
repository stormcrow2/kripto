from flask import Flask, render_template, jsonify, Response
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
import time
import json
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

    def get_realtime_data(self, symbol, interval='1m', limit=100):
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=f"{symbol}/USDT",
                timeframe=interval,
                limit=limit
            )
            return [{
                'time': candle[0],
                'open': candle[1],
                'high': candle[2],
                'low': candle[3],
                'close': candle[4],
                'volume': candle[5]
            } for candle in ohlcv]
        except Exception as e:
            print(f"Realtime veri hatası: {str(e)}")
            return []

    def get_technical_indicators(self, symbol, interval='1m'):
        try:
            data = self.get_realtime_data(symbol, interval)
            if not data:
                return []
            
            df = pd.DataFrame(data)
            df['MA7'] = df['close'].rolling(window=7).mean()
            df['MA25'] = df['close'].rolling(window=25).mean()
            df['MA99'] = df['close'].rolling(window=99).mean()
            
            # RSI hesaplama
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            return df.dropna().to_dict('records')
        except Exception as e:
            print(f"Teknik gösterge hatası: {str(e)}")
            return []

    def analyze_long_short_positions(self, df, sentiment_data=None):
        try:
            last_row = df.iloc[-1]
            last_price = last_row['close']
            
            # Risk ve ödül hesaplama faktörleri
            risk_factors = {
                'trend': 0,
                'momentum': 0,
                'volatility': 0,
                'sentiment': 0,
                'technical': 0
            }
            
            # Trend analizi
            if last_row['EMA_20'] > last_row['EMA_50']:
                risk_factors['trend'] += 1
            else:
                risk_factors['trend'] -= 1
                
            if last_row['SMA_200'] < last_price:
                risk_factors['trend'] += 0.5
            else:
                risk_factors['trend'] -= 0.5
                
            # Momentum analizi
            if last_row['RSI'] < 30:
                risk_factors['momentum'] += 1.5  # Aşırı satım
            elif last_row['RSI'] > 70:
                risk_factors['momentum'] -= 1.5  # Aşırı alım
                
            if last_row['MACD'] > last_row['MACD_Signal']:
                risk_factors['momentum'] += 1
            else:
                risk_factors['momentum'] -= 1
                
            # Volatilite analizi
            volatility = df['close'].pct_change().std() * np.sqrt(365)
            if volatility > 0.03:  # Yüksek volatilite
                risk_factors['volatility'] -= 0.5
            else:
                risk_factors['volatility'] += 0.5
                
            # Sentiment analizi
            if sentiment_data and 'sentiment_score' in sentiment_data:
                risk_factors['sentiment'] = sentiment_data['sentiment_score']
                
            # Teknik gösterge analizi
            if last_row['close'] > last_row['BB_upper']:
                risk_factors['technical'] -= 1
            elif last_row['close'] < last_row['BB_lower']:
                risk_factors['technical'] += 1
                
            # Toplam skor hesaplama
            total_score = sum(risk_factors.values())
            
            # Long/Short pozisyon önerileri
            if total_score > 1.5:
                position = "LONG"
                confidence = min(abs(total_score) / 5 * 100, 100)
                risk_ratio = 1.5
                reward_ratio = 3.0
            elif total_score < -1.5:
                position = "SHORT"
                confidence = min(abs(total_score) / 5 * 100, 100)
                risk_ratio = 1.5
                reward_ratio = 2.5
            else:
                position = "NÖTR"
                confidence = 50
                risk_ratio = 2
                reward_ratio = 2
                
            # Stop loss ve take profit seviyeleri
            atr = last_row['ATR']
            if position == "LONG":
                stop_loss = last_price - (atr * risk_ratio)
                take_profit = last_price + (atr * reward_ratio)
            elif position == "SHORT":
                stop_loss = last_price + (atr * risk_ratio)
                take_profit = last_price - (atr * reward_ratio)
            else:
                stop_loss = take_profit = last_price
                
            return {
                'position': position,
                'confidence': confidence,
                'risk_reward_ratio': reward_ratio / risk_ratio,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'potential_profit': abs(take_profit - last_price) / last_price * 100,
                'potential_loss': abs(stop_loss - last_price) / last_price * 100,
                'risk_factors': risk_factors,
                'suggested_leverage': min(5, 20 / (volatility * 100)),  # Volatiliteye göre kaldıraç önerisi
                'position_size': {
                    'conservative': 0.1,  # Portföyün %10'u
                    'moderate': 0.2,      # Portföyün %20'si
                    'aggressive': 0.3     # Portföyün %30'u
                }
            }
        except Exception as e:
            print(f"Long/Short analiz hatası: {str(e)}")
            return None

    def calculate_position_management(self, capital, risk_per_trade, entry_price, stop_loss, take_profit):
        try:
            # Pozisyon büyüklüğü hesaplama
            risk_amount = capital * (risk_per_trade / 100)
            stop_loss_pct = abs(entry_price - stop_loss) / entry_price
            position_size = risk_amount / stop_loss_pct
            
            # Kademeli kar alma seviyeleri
            take_profit_levels = [
                entry_price + (entry_price - stop_loss) * multiplier 
                for multiplier in [1.0, 1.5, 2.0]
            ]
            
            # Trailing stop seviyeleri
            trailing_stops = [
                entry_price + (take_profit - entry_price) * multiplier 
                for multiplier in [0.3, 0.5, 0.7]
            ]
            
            return {
                'position_size': position_size,
                'max_position_size': min(position_size, capital * 0.2),  # Maksimum %20 risk
                'take_profit_levels': take_profit_levels,
                'trailing_stops': trailing_stops,
                'suggested_exits': {
                    'conservative': {
                        'tp1_size': 0.5,  # Pozisyonun %50'si
                        'tp2_size': 0.3,  # Pozisyonun %30'u
                        'tp3_size': 0.2   # Pozisyonun %20'si
                    },
                    'aggressive': {
                        'tp1_size': 0.3,
                        'tp2_size': 0.3,
                        'tp3_size': 0.4
                    }
                }
            }
        except Exception as e:
            print(f"Pozisyon yönetimi hatası: {str(e)}")
            return None

    def calculate_portfolio_risk(self, positions, correlations=None):
        try:
            total_risk = 0
            total_exposure = 0
            risk_contributions = {}
            
            for symbol, position in positions.items():
                # Volatilite hesaplama
                volatility = position['df']['close'].pct_change().std() * np.sqrt(365)
                position_value = position['size'] * position['entry_price']
                position_risk = position_value * volatility
                
                total_risk += position_risk
                total_exposure += position_value
                risk_contributions[symbol] = position_risk
            
            # Risk metrikler
            return {
                'total_portfolio_risk': total_risk,
                'risk_contributions': risk_contributions,
                'risk_concentration': {
                    symbol: (risk / total_risk) * 100 
                    for symbol, risk in risk_contributions.items()
                },
                'portfolio_diversification_score': 1 - (max(risk_contributions.values()) / total_risk),
                'suggested_adjustments': {
                    symbol: ('AZALT' if (risk / total_risk) > 0.2 else 'ARTIR')
                    for symbol, risk in risk_contributions.items()
                }
            }
        except Exception as e:
            print(f"Portföy risk hesaplama hatası: {str(e)}")
            return None

    def multi_timeframe_analysis(self, symbol, timeframes=['1m', '5m', '15m', '1h', '4h', '1d']):
        try:
            analyses = {}
            for tf in timeframes:
                df = self.get_market_data(f"{symbol}/USDT", timeframe=tf)
                if not df.empty:
                    df = self.calculate_advanced_indicators(df)
                    
                    # Her zaman dilimi için trend analizi
                    last_row = df.iloc[-1]
                    trend = {
                        'ema_trend': 'YÜKSELEN' if last_row['EMA_20'] > last_row['EMA_50'] else 'DÜŞEN',
                        'macd_trend': 'YÜKSELEN' if last_row['MACD'] > last_row['MACD_Signal'] else 'DÜŞEN',
                        'rsi_state': 'AŞIRI ALIM' if last_row['RSI'] > 70 else 'AŞIRI SATIM' if last_row['RSI'] < 30 else 'NÖTR'
                    }
                    
                    analyses[tf] = trend
            
            # Trend uyumu analizi
            trend_agreement = {
                'bullish_count': sum(1 for tf in analyses.values() if tf['ema_trend'] == 'YÜKSELEN'),
                'bearish_count': sum(1 for tf in analyses.values() if tf['ema_trend'] == 'DÜŞEN')
            }
            
            trend_strength = (max(trend_agreement['bullish_count'], trend_agreement['bearish_count']) / len(timeframes)) * 100
            
            return {
                'timeframe_analyses': analyses,
                'trend_agreement': trend_agreement,
                'trend_strength': trend_strength,
                'overall_trend': 'GÜÇLÜ YÜKSELEN' if trend_strength > 80 and trend_agreement['bullish_count'] > trend_agreement['bearish_count']
                               else 'GÜÇLÜ DÜŞEN' if trend_strength > 80 and trend_agreement['bullish_count'] < trend_agreement['bearish_count']
                               else 'ZAYIF YÜKSELEN' if trend_agreement['bullish_count'] > trend_agreement['bearish_count']
                               else 'ZAYIF DÜŞEN' if trend_agreement['bullish_count'] < trend_agreement['bearish_count']
                               else 'KARARSIZ'
            }
        except Exception as e:
            print(f"Çoklu zaman dilimi analiz hatası: {str(e)}")
            return None

    def simulate_position(self, symbol, position_type, entry_price, stop_loss, take_profit, position_size, timeframe='5m', periods=100):
        try:
            df = self.get_market_data(f"{symbol}/USDT", timeframe=timeframe, limit=periods)
            if df.empty:
                return None
            
            simulation = {
                'trades': [],
                'pnl': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0
            }
            
            in_position = True
            entry_index = 0
            
            for i in range(1, len(df)):
                if in_position:
                    current_price = df['close'].iloc[i]
                    
                    if position_type == 'LONG':
                        if current_price <= stop_loss:
                            pnl = (stop_loss - entry_price) / entry_price * position_size
                            simulation['trades'].append({
                                'exit_price': stop_loss,
                                'exit_time': df['timestamp'].iloc[i],
                                'pnl': pnl,
                                'type': 'stop_loss'
                            })
                            in_position = False
                        elif current_price >= take_profit:
                            pnl = (take_profit - entry_price) / entry_price * position_size
                            simulation['trades'].append({
                                'exit_price': take_profit,
                                'exit_time': df['timestamp'].iloc[i],
                                'pnl': pnl,
                                'type': 'take_profit'
                            })
                            in_position = False
                    else:  # SHORT
                        if current_price >= stop_loss:
                            pnl = (entry_price - stop_loss) / entry_price * position_size
                            simulation['trades'].append({
                                'exit_price': stop_loss,
                                'exit_time': df['timestamp'].iloc[i],
                                'pnl': pnl,
                                'type': 'stop_loss'
                            })
                            in_position = False
                        elif current_price <= take_profit:
                            pnl = (entry_price - take_profit) / entry_price * position_size
                            simulation['trades'].append({
                                'exit_price': take_profit,
                                'exit_time': df['timestamp'].iloc[i],
                                'pnl': pnl,
                                'type': 'take_profit'
                            })
                            in_position = False
            
            if len(simulation['trades']) > 0:
                wins = [t for t in simulation['trades'] if t['pnl'] > 0]
                losses = [t for t in simulation['trades'] if t['pnl'] <= 0]
                
                simulation['pnl'] = sum(t['pnl'] for t in simulation['trades'])
                simulation['win_rate'] = len(wins) / len(simulation['trades']) * 100
                simulation['avg_win'] = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
                simulation['avg_loss'] = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
                
                # Maksimum drawdown hesaplama
                cumulative_pnl = np.cumsum([t['pnl'] for t in simulation['trades']])
                peak = np.maximum.accumulate(cumulative_pnl)
                drawdown = (peak - cumulative_pnl) / peak * 100
                simulation['max_drawdown'] = np.max(drawdown)
            
            return simulation
            
        except Exception as e:
            print(f"Pozisyon simülasyonu hatası: {str(e)}")
            return None

    def analyze_historical_performance(self, symbol, timeframe='1d', periods=365):
        try:
            df = self.get_market_data(f"{symbol}/USDT", timeframe=timeframe, limit=periods)
            if df.empty:
                return None
            
            # Temel metrikler
            returns = df['close'].pct_change()
            
            analysis = {
                'total_return': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100,
                'annual_return': returns.mean() * 252 * 100,
                'volatility': returns.std() * np.sqrt(252) * 100,
                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
                'max_drawdown': 0,
                'best_day': returns.max() * 100,
                'worst_day': returns.min() * 100,
                'positive_days': (returns > 0).sum() / len(returns) * 100,
                'monthly_returns': []
            }
            
            # Maksimum drawdown hesaplama
            peak = df['close'].expanding(min_periods=1).max()
            drawdown = ((peak - df['close']) / peak * 100)
            analysis['max_drawdown'] = drawdown.max()
            
            # Aylık getiriler
            df['month'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m')
            monthly = df.groupby('month')['close'].agg(['first', 'last'])
            analysis['monthly_returns'] = ((monthly['last'] / monthly['first'] - 1) * 100).to_dict()
            
            return analysis
            
        except Exception as e:
            print(f"Geçmiş performans analizi hatası: {str(e)}")
            return None

    def optimize_portfolio(self, symbols, capital, risk_tolerance='moderate'):
        try:
            # Risk tolerans seviyeleri
            risk_levels = {
                'conservative': {'max_allocation': 0.2, 'min_symbols': 5},
                'moderate': {'max_allocation': 0.3, 'min_symbols': 4},
                'aggressive': {'max_allocation': 0.4, 'min_symbols': 3}
            }
            
            risk_params = risk_levels[risk_tolerance]
            returns_data = {}
            volatility_data = {}
            correlation_matrix = pd.DataFrame()
            
            # Her sembol için veri toplama
            for symbol in symbols:
                df = self.get_market_data(f"{symbol}/USDT", timeframe='1d', limit=365)
                if not df.empty:
                    returns = df['close'].pct_change().dropna()
                    returns_data[symbol] = returns.mean() * 252
                    volatility_data[symbol] = returns.std() * np.sqrt(252)
                    correlation_matrix[symbol] = returns
            
            correlation_matrix = correlation_matrix.corr()
            
            # Portföy optimizasyonu
            optimal_weights = {}
            remaining_capital = capital
            sorted_symbols = sorted(symbols, key=lambda x: returns_data.get(x, 0) / volatility_data.get(x, float('inf')), reverse=True)
            
            for symbol in sorted_symbols:
                if len(optimal_weights) >= risk_params['min_symbols']:
                    break
                
                max_allocation = min(remaining_capital, capital * risk_params['max_allocation'])
                optimal_weights[symbol] = max_allocation
                remaining_capital -= max_allocation
            
            # Kalan sermayeyi dağıt
            if remaining_capital > 0:
                for symbol in optimal_weights:
                    optimal_weights[symbol] += remaining_capital / len(optimal_weights)
            
            # Portföy metrikleri
            portfolio_return = sum(optimal_weights[s] * returns_data[s] for s in optimal_weights)
            portfolio_volatility = np.sqrt(sum(sum(optimal_weights[s1] * optimal_weights[s2] * 
                                                 correlation_matrix.loc[s1, s2] * 
                                                 volatility_data[s1] * volatility_data[s2]
                                                 for s2 in optimal_weights)
                                             for s1 in optimal_weights))
            
            return {
                'allocations': optimal_weights,
                'metrics': {
                    'expected_return': portfolio_return * 100,
                    'volatility': portfolio_volatility * 100,
                    'sharpe_ratio': portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0,
                    'diversification_score': 1 - max(optimal_weights.values()) / capital
                },
                'risk_metrics': {
                    symbol: {
                        'volatility': volatility_data[symbol] * 100,
                        'correlation': correlation_matrix[symbol].mean()
                    } for symbol in optimal_weights
                }
            }
            
        except Exception as e:
            print(f"Portföy optimizasyonu hatası: {str(e)}")
            return None

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
        
        # Long/Short analizi
        long_short_analysis = analyzer.analyze_long_short_positions(df, sentiment_data)
        
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
            'risk_analysis': signals.get('risk_analysis', {}),
            'long_short_analysis': long_short_analysis
        })
        
    except Exception as e:
        print(f"Analiz hatası: {str(e)}")  # Sunucu logları için
        return jsonify({
            'error': 'Analiz hatası',
            'message': str(e)
        }), 500

@app.route('/realtime-data/<symbol>')
def get_realtime_data(symbol):
    def generate():
        analyzer = AdvancedCryptoAnalyzer()
        while True:
            data = analyzer.get_realtime_data(symbol)
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(5)  # 5 saniyede bir güncelle

    return Response(generate(), mimetype='text/event-stream')

@app.route('/technical-data/<symbol>/<interval>')
def get_technical_data(symbol, interval):
    analyzer = AdvancedCryptoAnalyzer()
    data = analyzer.get_technical_indicators(symbol, interval)
    return jsonify(data)

@app.route('/position-management/<symbol>')
def get_position_management(symbol):
    try:
        analyzer = AdvancedCryptoAnalyzer()
        df = analyzer.get_market_data(f"{symbol}/USDT")
        if df.empty:
            return jsonify({'error': 'Veri alınamadı'}), 400
            
        last_price = float(df['close'].iloc[-1])
        analysis = analyzer.analyze_long_short_positions(df)
        
        if not analysis:
            return jsonify({'error': 'Analiz yapılamadı'}), 400
            
        position_management = analyzer.calculate_position_management(
            capital=10000,  # Varsayılan değer
            risk_per_trade=1,  # %1 risk
            entry_price=last_price,
            stop_loss=analysis['stop_loss'],
            take_profit=analysis['take_profit']
        )
        
        return jsonify(position_management)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/portfolio-analysis', methods=['POST'])
def analyze_portfolio():
    try:
        data = request.json
        analyzer = AdvancedCryptoAnalyzer()
        positions = {}
        
        for position in data['positions']:
            df = analyzer.get_market_data(f"{position['symbol']}/USDT")
            if not df.empty:
                positions[position['symbol']] = {
                    'df': df,
                    'size': position['size'],
                    'entry_price': position['entry_price']
                }
        
        risk_analysis = analyzer.calculate_portfolio_risk(positions)
        return jsonify(risk_analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/multi-timeframe/<symbol>')
def get_multi_timeframe_analysis(symbol):
    try:
        analyzer = AdvancedCryptoAnalyzer()
        analysis = analyzer.multi_timeframe_analysis(symbol)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/simulate-position/<symbol>', methods=['POST'])
def simulate_position(symbol):
    try:
        data = request.json
        analyzer = AdvancedCryptoAnalyzer()
        
        simulation = analyzer.simulate_position(
            symbol=symbol,
            position_type=data['position_type'],
            entry_price=data['entry_price'],
            stop_loss=data['stop_loss'],
            take_profit=data['take_profit'],
            position_size=data['position_size']
        )
        
        return jsonify(simulation)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/historical-performance/<symbol>')
def get_historical_performance(symbol):
    try:
        analyzer = AdvancedCryptoAnalyzer()
        analysis = analyzer.analyze_historical_performance(symbol)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/optimize-portfolio', methods=['POST'])
def optimize_portfolio():
    try:
        data = request.json
        analyzer = AdvancedCryptoAnalyzer()
        
        optimization = analyzer.optimize_portfolio(
            symbols=data['symbols'],
            capital=data['capital'],
            risk_tolerance=data.get('risk_tolerance', 'moderate')
        )
        
        return jsonify(optimization)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
