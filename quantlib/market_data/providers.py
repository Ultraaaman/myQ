"""
具体的数据提供者实现
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from .base import BaseDataProvider, DataProviderFactory, YFINANCE_AVAILABLE, AKSHARE_AVAILABLE

try:
    import yfinance as yf
except ImportError:
    pass

try:
    import akshare as ak
except ImportError:
    pass


class YahooFinanceProvider(BaseDataProvider):
    """Yahoo Finance 数据提供者（美股）"""
    
    def __init__(self, symbol: str):
        super().__init__(symbol, 'US')
        
    def get_historical_data(self, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """获取美股历史数据"""
        if not YFINANCE_AVAILABLE:
            raise ImportError("需要安装yfinance: pip install yfinance")
            
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"警告: {self.symbol} 未获取到数据")
                return None
                
            # 重置索引，将Date从索引移到列
            data = data.reset_index()
            
            # 标准化数据格式
            data = self.standardize_data(data)
            
            self.data = data
            print(f"✓ {self.symbol}: 获取 {len(data)} 条美股记录")
            return data
            
        except Exception as e:
            print(f"× {self.symbol}: 获取美股数据失败 - {e}")
            return None
    
    def get_realtime_data(self) -> Optional[Dict[str, Any]]:
        """获取美股实时数据"""
        if not YFINANCE_AVAILABLE:
            return None
            
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            
            return {
                'symbol': self.symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'change': info.get('regularMarketChange'),
                'change_percent': info.get('regularMarketChangePercent'),
                'volume': info.get('regularMarketVolume'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'timestamp': pd.Timestamp.now()
            }
        except Exception as e:
            print(f"获取 {self.symbol} 实时数据失败: {e}")
            return None
    
    def get_company_info(self) -> Optional[Dict[str, Any]]:
        """获取公司基本信息"""
        if not YFINANCE_AVAILABLE:
            return None
            
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            
            return {
                'symbol': self.symbol,
                'company_name': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'employee_count': info.get('fullTimeEmployees'),
                'website': info.get('website'),
                'description': info.get('longBusinessSummary')
            }
        except Exception as e:
            print(f"获取 {self.symbol} 公司信息失败: {e}")
            return None


class AkshareProvider(BaseDataProvider):
    """Akshare 数据提供者（A股）"""
    
    def __init__(self, symbol: str):
        super().__init__(symbol, 'CN')

    def _format_symbol_for_minute(self, symbol):
        """格式化股票代码用于分钟级数据API

        Args:
            symbol: 原始股票代码 (如 '000001', 'sz000001', '600519', 'sh600519')

        Returns:
            str: 格式化后的代码 (如 'sz000001', 'sh600519')
        """
        # 如果已经有前缀，直接返回
        if symbol.startswith(('sz', 'sh')):
            return symbol

        # 根据代码规则添加前缀
        if symbol.startswith(('000', '001', '002', '003', '300')):
            # 深圳交易所：主板000、001，中小板002，创业板300
            return f"sz{symbol}"
        elif symbol.startswith(('600', '601', '603', '605', '688')):
            # 上海交易所：主板600、601、603、605，科创板688
            return f"sh{symbol}"
        else:
            # 默认深圳交易所
            return f"sz{symbol}"
        
    def get_historical_data(self, period: str = "1y", interval: str = "daily") -> Optional[pd.DataFrame]:
        """获取A股历史数据"""
        if not AKSHARE_AVAILABLE:
            raise ImportError("需要安装akshare: pip install akshare")

        try:
            # 判断是否为分钟级数据
            minute_intervals = ["1min", "5min", "15min", "30min", "60min", "1m", "5m", "15m", "30m", "60m"]
            is_minute_data = interval in minute_intervals

            if is_minute_data:
                # 分钟级数据处理
                period_map = {
                    "1min": "1", "1m": "1",
                    "5min": "5", "5m": "5",
                    "15min": "15", "15m": "15",
                    "30min": "30", "30m": "30",
                    "60min": "60", "60m": "60"
                }

                minute_period = period_map.get(interval, "1")

                # 确保股票代码有交易所前缀
                formatted_symbol = self._format_symbol_for_minute(self.symbol)

                # 获取分钟级数据（近5个交易日）
                data = ak.stock_zh_a_minute(
                    symbol=formatted_symbol,
                    period=minute_period,
                    adjust="qfq"  # 前复权
                )
            else:
                # 日线及以上数据
                # 获取历史数据，使用前复权
                data = ak.stock_zh_a_hist(
                    symbol=self.symbol,
                    period=interval,
                    adjust="qfq"  # 前复权
                )
            
            if data.empty:
                print(f"警告: {self.symbol} 未获取到数据")
                return None
            
            # 根据period筛选数据（只对日线数据处理）
            if not is_minute_data and period and period != "max":
                end_date = pd.Timestamp.now()
                if period == "1y":
                    start_date = end_date - pd.DateOffset(years=1)
                elif period == "6mo":
                    start_date = end_date - pd.DateOffset(months=6)
                elif period == "3mo":
                    start_date = end_date - pd.DateOffset(months=3)
                elif period == "1mo":
                    start_date = end_date - pd.DateOffset(months=1)
                elif period == "5y":
                    start_date = end_date - pd.DateOffset(years=5)
                else:
                    start_date = None

                if start_date:
                    data['日期'] = pd.to_datetime(data['日期'])
                    data = data[data['日期'] >= start_date]
            
            # 标准化数据格式
            data = self.standardize_data(data)
            
            self.data = data
            print(f"✓ {self.symbol}: 获取 {len(data)} 条A股记录")
            return data
            
        except Exception as e:
            print(f"× {self.symbol}: 获取A股数据失败 - {e}")
            return None
    
    def get_realtime_data(self) -> Optional[Dict[str, Any]]:
        """获取A股实时数据"""
        if not AKSHARE_AVAILABLE:
            return None
            
        try:
            # 获取实时行情数据
            data = ak.stock_zh_a_spot_em()
            
            # 查找指定股票
            stock_data = data[data['代码'] == self.symbol]
            if stock_data.empty:
                return None
            
            stock_info = stock_data.iloc[0]
            
            return {
                'symbol': self.symbol,
                'price': stock_info.get('最新价'),
                'change': stock_info.get('涨跌额'),
                'change_percent': stock_info.get('涨跌幅'),
                'volume': stock_info.get('成交量'),
                'market_cap': stock_info.get('总市值'),
                'pe_ratio': stock_info.get('市盈率-动态'),
                'timestamp': pd.Timestamp.now()
            }
        except Exception as e:
            print(f"获取 {self.symbol} 实时数据失败: {e}")
            return None
    
    def get_company_info(self) -> Optional[Dict[str, Any]]:
        """获取A股公司基本信息"""
        if not AKSHARE_AVAILABLE:
            return None
            
        try:
            # 获取公司概况
            info = ak.stock_individual_info_em(symbol=self.symbol)
            
            if info.empty:
                return None
            
            # 转换为字典格式
            info_dict = {}
            for _, row in info.iterrows():
                info_dict[row['item']] = row['value']
            
            return {
                'symbol': self.symbol,
                'company_name': info_dict.get('公司名称'),
                'industry': info_dict.get('行业'),
                'main_business': info_dict.get('主营业务'),
                'market_cap': info_dict.get('总市值'),
                'employee_count': info_dict.get('员工人数'),
                'website': info_dict.get('公司网址'),
                'listing_date': info_dict.get('上市时间')
            }
        except Exception as e:
            print(f"获取 {self.symbol} 公司信息失败: {e}")
            return None
    
    def get_csi300_data(self, period: str = "1y") -> Optional[pd.DataFrame]:
        """获取沪深300指数数据"""
        try:
            # 获取沪深300指数历史数据
            data = ak.stock_zh_index_daily(symbol="sh000300")
            
            if data.empty:
                print("警告: 沪深300指数未获取到数据")
                return None
            
            # 根据period筛选数据
            if period and period != "max":
                end_date = pd.Timestamp.now()
                if period == "1y":
                    start_date = end_date - pd.DateOffset(years=1)
                elif period == "6mo":
                    start_date = end_date - pd.DateOffset(months=6)
                elif period == "3mo":
                    start_date = end_date - pd.DateOffset(months=3)
                elif period == "1mo":
                    start_date = end_date - pd.DateOffset(months=1)
                elif period == "5y":
                    start_date = end_date - pd.DateOffset(years=5)
                else:
                    start_date = None
                    
                if start_date:
                    data['date'] = pd.to_datetime(data['date'])
                    data = data[data['date'] >= start_date]
            
            # 标准化数据格式
            data = self.standardize_data(data)
            
            print(f"✓ 沪深300指数: 获取 {len(data)} 条记录")
            return data
            
        except Exception as e:
            print(f"× 沪深300指数: 获取数据失败 - {e}")
            return None

    def get_order_book(self, depth: int = 5) -> Optional[Dict[str, Any]]:
        """获取订单簿数据（买卖盘口）

        Args:
            depth: 档位深度，通常为5档（akshare限制）

        Returns:
            Dict包含买卖盘口数据
        """
        if not AKSHARE_AVAILABLE:
            return None

        try:
            # 尝试使用多种方式获取盘口数据
            bid_ask_data = None

            # 方法1: 尝试使用stock_bid_ask_em
            try:
                if hasattr(ak, 'stock_bid_ask_em'):
                    bid_ask_data = ak.stock_bid_ask_em(symbol=self.symbol)
            except Exception as e:
                print(f"stock_bid_ask_em失败: {e}")

            # 方法2: 尝试使用实时行情数据提取盘口信息
            if bid_ask_data is None or bid_ask_data.empty:
                try:
                    if hasattr(ak, 'stock_zh_a_spot_em'):
                        spot_data = ak.stock_zh_a_spot_em()
                        if not spot_data.empty:
                            # 查找指定股票
                            stock_data = spot_data[spot_data['代码'] == self.symbol]
                            if not stock_data.empty:
                                bid_ask_data = stock_data
                except Exception as e:
                    print(f"从实时行情提取盘口失败: {e}")

            if bid_ask_data is None or bid_ask_data.empty:
                print(f"警告: {self.symbol} 未获取到盘口数据")
                return None

            # 解析盘口数据
            latest_data = bid_ask_data.iloc[-1]

            order_book = {
                'symbol': self.symbol,
                'timestamp': pd.Timestamp.now(),
                'asks': [],  # 卖盘（价格从低到高）
                'bids': [],  # 买盘（价格从高到低）
                'spread': 0,
                'mid_price': 0
            }

            # 提取买卖盘口数据 - 尝试多种可能的列名格式
            available_columns = list(latest_data.index) if hasattr(latest_data, 'index') else list(latest_data.keys())
            print(f"可用列: {available_columns}")  # 调试信息

            # 尝试提取基本买卖价格信息
            basic_bid_ask_patterns = [
                ('最新价', 'current_price'), ('现价', 'current_price'), ('price', 'current_price'),
                ('买一价', 'bid_1'), ('卖一价', 'ask_1'),
                ('买1', 'bid_1'), ('卖1', 'ask_1'),
                ('买价', 'bid_price'), ('卖价', 'ask_price')
            ]

            # 如果找不到详细盘口数据，至少尝试获取基本买卖价格
            current_price = 0
            bid_price = 0
            ask_price = 0

            for pattern, key in basic_bid_ask_patterns:
                if pattern in available_columns:
                    value = latest_data[pattern]
                    if pd.notna(value) and value != 0:
                        if key == 'current_price':
                            current_price = float(value)
                        elif key in ['bid_1', 'bid_price']:
                            bid_price = float(value)
                        elif key in ['ask_1', 'ask_price']:
                            ask_price = float(value)

            # 如果有基本价格信息，创建简化的订单簿
            if current_price > 0:
                # 估算买卖价（如果没有具体的买卖价）
                if bid_price == 0:
                    bid_price = current_price * 0.999  # 估算买价
                if ask_price == 0:
                    ask_price = current_price * 1.001  # 估算卖价

                order_book['bids'].append({'price': bid_price, 'volume': 100, 'level': 1})
                order_book['asks'].append({'price': ask_price, 'volume': 100, 'level': 1})

            # 尝试提取详细五档数据
            for i in range(1, min(depth + 1, 6)):  # 最多5档
                found_bid = False
                found_ask = False

                # 可能的列名格式
                bid_price_patterns = [f'买{i}', f'买{i}价', f'bid{i}', f'bid_{i}', f'买一价' if i == 1 else None]
                bid_vol_patterns = [f'买{i}量', f'买{i}手', f'bid{i}_vol', f'买一量' if i == 1 else None]
                ask_price_patterns = [f'卖{i}', f'卖{i}价', f'ask{i}', f'ask_{i}', f'卖一价' if i == 1 else None]
                ask_vol_patterns = [f'卖{i}量', f'卖{i}手', f'ask{i}_vol', f'卖一量' if i == 1 else None]

                # 查找买盘数据
                for bid_price_col in bid_price_patterns:
                    if bid_price_col and bid_price_col in available_columns:
                        for bid_vol_col in bid_vol_patterns:
                            if bid_vol_col and bid_vol_col in available_columns:
                                try:
                                    bid_price_val = float(latest_data[bid_price_col]) if pd.notna(latest_data[bid_price_col]) else 0
                                    bid_vol_val = int(latest_data[bid_vol_col]) if pd.notna(latest_data[bid_vol_col]) else 0
                                    if bid_price_val > 0:
                                        order_book['bids'].append({'price': bid_price_val, 'volume': bid_vol_val, 'level': i})
                                        found_bid = True
                                        break
                                except:
                                    continue
                        if found_bid:
                            break

                # 查找卖盘数据
                for ask_price_col in ask_price_patterns:
                    if ask_price_col and ask_price_col in available_columns:
                        for ask_vol_col in ask_vol_patterns:
                            if ask_vol_col and ask_vol_col in available_columns:
                                try:
                                    ask_price_val = float(latest_data[ask_price_col]) if pd.notna(latest_data[ask_price_col]) else 0
                                    ask_vol_val = int(latest_data[ask_vol_col]) if pd.notna(latest_data[ask_vol_col]) else 0
                                    if ask_price_val > 0:
                                        order_book['asks'].append({'price': ask_price_val, 'volume': ask_vol_val, 'level': i})
                                        found_ask = True
                                        break
                                except:
                                    continue
                        if found_ask:
                            break

            # 计算买卖价差和中间价
            if order_book['bids'] and order_book['asks']:
                best_bid = max(order_book['bids'], key=lambda x: x['price'])['price']
                best_ask = min(order_book['asks'], key=lambda x: x['price'])['price']
                order_book['spread'] = best_ask - best_bid
                order_book['mid_price'] = (best_bid + best_ask) / 2

            return order_book

        except Exception as e:
            print(f"获取 {self.symbol} 订单簿数据失败: {e}")
            return None

    def get_tick_data(self, trade_date: str = None) -> Optional[pd.DataFrame]:
        """获取逐笔交易数据

        Args:
            trade_date: 交易日期，格式YYYYMMDD，默认为今天

        Returns:
            包含逐笔交易数据的DataFrame
        """
        if not AKSHARE_AVAILABLE:
            return None

        try:
            # 尝试使用多个可能的API获取逐笔数据
            tick_data = None

            # 方法1: 尝试使用stock_zh_a_tick_163
            try:
                if hasattr(ak, 'stock_zh_a_tick_163'):
                    tick_data = ak.stock_zh_a_tick_163(symbol=self.symbol, trade_date=trade_date)
            except:
                pass

            # 方法2: 尝试使用stock_intraday_sina (当日数据)
            if tick_data is None or tick_data.empty:
                try:
                    if hasattr(ak, 'stock_intraday_sina'):
                        tick_data = ak.stock_intraday_sina(symbol=self.symbol)
                except:
                    pass

            # 方法3: 尝试使用stock_intraday_em
            if tick_data is None or tick_data.empty:
                try:
                    if hasattr(ak, 'stock_intraday_em'):
                        tick_data = ak.stock_intraday_em(symbol=self.symbol)
                except:
                    pass

            # 方法4: 使用分钟级数据作为替代
            if tick_data is None or tick_data.empty:
                try:
                    # 获取1分钟数据作为替代
                    formatted_symbol = self._format_symbol_for_minute(self.symbol)
                    tick_data = ak.stock_zh_a_minute(
                        symbol=formatted_symbol,
                        period="1",
                        adjust="qfq"
                    )
                    print(f"注意: {self.symbol} 使用1分钟数据替代逐笔数据")
                except:
                    pass

            if tick_data is None or tick_data.empty:
                print(f"警告: {self.symbol} 未获取到逐笔数据")
                return None

            # 标准化数据格式
            tick_data = self._standardize_tick_data(tick_data)

            print(f"✓ {self.symbol}: 获取 {len(tick_data)} 条逐笔交易记录")
            return tick_data

        except Exception as e:
            print(f"× {self.symbol}: 获取逐笔数据失败 - {e}")
            return None

    def get_intraday_data(self) -> Optional[pd.DataFrame]:
        """获取盘中交易明细数据

        Returns:
            包含盘中交易明细的DataFrame
        """
        if not AKSHARE_AVAILABLE:
            return None

        try:
            # 获取盘中交易明细
            intraday_data = ak.stock_intraday_em(symbol=self.symbol)

            if intraday_data.empty:
                print(f"警告: {self.symbol} 未获取到盘中交易明细")
                return None

            # 标准化数据格式
            intraday_data = self._standardize_intraday_data(intraday_data)

            print(f"✓ {self.symbol}: 获取 {len(intraday_data)} 条盘中交易明细")
            return intraday_data

        except Exception as e:
            print(f"× {self.symbol}: 获取盘中数据失败 - {e}")
            return None

    def _standardize_tick_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化逐笔数据格式"""
        if data.empty:
            return data

        data = data.copy()

        # 打印原始列名进行调试
        print(f"原始数据列名: {list(data.columns)}")

        # 标准化列名 - 扩展更多可能的列名
        column_mapping = {
            # 时间相关
            '时间': 'time', 'timestamp': 'time', 'day': 'time', 'datetime': 'time',
            # 价格相关
            '价格': 'price', 'Price': 'price', 'price': 'price', 'open': 'price',
            'close': 'price', 'high': 'price', 'low': 'price',
            # 成交量相关
            '成交量': 'volume', 'Volume': 'volume', 'volume': 'volume', '手数': 'volume',
            # 买卖方向
            '买卖盘性质': 'side', '买卖': 'side', 'side': 'side', 'direction': 'side',
            # 成交额
            '成交额': 'amount', '金额': 'amount', 'amount': 'amount'
        }

        for old_name, new_name in column_mapping.items():
            if old_name in data.columns:
                data = data.rename(columns={old_name: new_name})

        # 如果没有price列，尝试使用其他价格列
        if 'price' not in data.columns:
            price_candidates = ['close', 'open', 'high', 'low']
            for candidate in price_candidates:
                if candidate in data.columns:
                    data['price'] = data[candidate]
                    print(f"使用 {candidate} 作为 price 列")
                    break

        # 如果没有amount列，尝试计算
        if 'amount' not in data.columns and 'price' in data.columns and 'volume' in data.columns:
            try:
                data['amount'] = pd.to_numeric(data['price'], errors='coerce') * pd.to_numeric(data['volume'], errors='coerce')
                print("计算生成 amount 列")
            except:
                pass

        # 如果仍然没有关键列，创建默认值
        if 'price' not in data.columns:
            print("警告: 缺少价格数据，使用默认值")
            data['price'] = 0

        if 'volume' not in data.columns:
            print("警告: 缺少成交量数据，使用默认值")
            data['volume'] = 0

        if 'amount' not in data.columns:
            print("警告: 缺少成交额数据，使用默认值")
            data['amount'] = 0

        # 转换时间格式
        if 'time' in data.columns:
            try:
                data['time'] = pd.to_datetime(data['time'], errors='coerce')
            except:
                pass

        # 标准化买卖方向
        if 'side' in data.columns:
            data['side'] = data['side'].map({
                '买盘': 'buy', '卖盘': 'sell', '中性盘': 'neutral',
                'buy': 'buy', 'sell': 'sell', 'neutral': 'neutral'
            }).fillna('neutral')
        else:
            # 如果没有方向信息，设为neutral
            data['side'] = 'neutral'

        # 确保数值类型
        numeric_columns = ['price', 'volume', 'amount']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

        print(f"标准化后列名: {list(data.columns)}")
        return data

    def _standardize_intraday_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化盘中交易明细格式"""
        if data.empty:
            return data

        data = data.copy()

        # 标准化列名（类似逐笔数据处理）
        column_mapping = {
            '时间': 'time', 'timestamp': 'time',
            '价格': 'price', 'Price': 'price',
            '成交量': 'volume', 'Volume': 'volume', '手数': 'volume',
            '买卖盘性质': 'side', '买卖': 'side', 'direction': 'side',
            '成交额': 'amount', '金额': 'amount'
        }

        for old_name, new_name in column_mapping.items():
            if old_name in data.columns:
                data = data.rename(columns={old_name: new_name})

        return data


# 注册数据提供者
DataProviderFactory.register_provider('US', YahooFinanceProvider)
DataProviderFactory.register_provider('CN', AkshareProvider)
DataProviderFactory.register_provider('A股', AkshareProvider)
DataProviderFactory.register_provider('CHINA', AkshareProvider)