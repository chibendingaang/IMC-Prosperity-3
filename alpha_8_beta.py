from datamodel import OrderDepth, Order, TradingState
from typing import List, Dict
import numpy as np

class Trader:
    def __init__(self):
        self.mid_price_history = {}  # Dictionary to track mid prices for each product
        self.ema_prices = {}  # Dictionary to track EMAs for each product
        self.positions = {}  # Dictionary to track current positions

    def run(self, state: TradingState):
        # Trial 7 >>> 
        # introducing MACD(8,21) in place of (5,13): this yielded 0 trades through MACD
        # setting in_long/in_short_position back to abs(10) from abs(15)

        # Trial 8 >>>
        # switching back to  MACD(5,13)
        # keeping central_price history lookback window back to 5, EMA instead of SMA
        result = {}
        conversions = 0
        traderData = ""

        # Initialize tracking for new products
        for product in state.order_depths:
            if product not in self.mid_price_history:
                self.mid_price_history[product] = []
            if product not in self.ema_prices:
                self.ema_prices[product] = None

        # Update positions
        self.positions = state.position

        # Update mid prices and apply strategies
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # Update mid price if we have both buy and sell orders
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders)
                best_ask = min(order_depth.sell_orders)
                mid_price = (best_bid + best_ask) / 2
                self.mid_price_history[product].append(mid_price)
            
            # Limit history length
            if len(self.mid_price_history[product]) > 20:
                self.mid_price_history[product] = self.mid_price_history[product][-20:]
            
            # Skip if we don't have enough history
            if len(self.mid_price_history[product]) < 3:
                result[product] = []
                continue

            # Apply strategy based on product
            if product == "RAINFOREST_RESIN":
                orders = self.strategy_resin(product, state)
            elif product == "KELP":
                orders = self.strategy_reversion(product, state,0.01,0.05,lookback=10)
            elif product == "SQUID_INK":
                orders = self.strategy_reversion(product, state,0.01,0.05,lookback=10)
            elif product == "JAMS":
                orders = self.strategy_reversion(product, state,0.01,0.05,lookback=10)
            elif product == "VOLCANIC_ROCK":
                orders = self.strategy_reversion(product, state,0.01,0.05,lookback=10)
            '''
            elif product == "DJEMBES":
                orders = self.strategy_reversion(product, state,0.01,0.05,lookback=10)
            elif product == "CROISSANTS":
                orders = self.strategy_reversion(product, state,0.01,0.05,lookback=10)
            elif product == "PICNIC_BASKET1":
                orders = self.strategy_reversion(product, state,0.1,0.1,10)
            elif product == "PICNIC_BASKET2":
                orders = self.strategy_reversion(product, state,0.1,0.1,10)
            '''
            result[product] = orders

        return result, conversions, traderData
    
    def enforce_limit(self, product, qty, is_buy):
        # Define position limits based on product
        position_limits = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
        "SQUID_INK": 50,
        "CROISSANTS": 250,
        "JAMS": 350,
        "DJEMBES": 60,
        "PICNIC_BASKET1": 60,
        "PICNIC_BASKET2": 100,
        "VOLCANIC_ROCK": 400,
        "VOLCANIC_ROCK_VOUCHER": 250
        }
    
        # Get position limit for the product or use a default value of 50
        position_limit = position_limits.get(product, 50)
    
        # Get current position  
        current_pos = self.positions.get(product, 0)
    
        # Enforce limit
        if is_buy:
            return max(0, min(qty, position_limit - current_pos))
        else:
            return max(0, min(qty, current_pos + position_limit))
    
    def place_orders(self, product, prices, qty, is_buy):
        """Place orders for a product at given prices."""
        orders = []
        for price in prices:
            allowed = self.enforce_limit(product, abs(qty), is_buy)
            if allowed > 0:
                final_qty = allowed if is_buy else -allowed
                orders.append(Order(product, price, final_qty))
        return orders
    
    def strategy_resin(self, product, state):
        """Strategy for RAINFOREST_RESIN based on price momentum."""
        orders = []
        
        # Use fixed central price for RESIN
        central_price = 10000
        
        # Analyze price movements
        delta = np.array(self.mid_price_history[product][-3:]) - central_price
        slope_signs = [delta[i+1] - delta[i] for i in range(2)]
        
        sign_product = np.sign(slope_signs).prod()
        prvs_price2 = delta[-2]
        prvs_price1 = delta[-1]
        
        # Order quantities
        full_qty = 6
        half_qty = 5
        skimp_qty = 3
        
        # Price bands
        buy_prices2 = [central_price - x for x in [3, 4, 5]]
        sell_prices2 = [central_price + x for x in [3, 4, 5]]
        momentum_buy = [central_price - x for x in [0]]
        momentum_sell = [central_price + x for x in [0]]
        sell_prices1 = [central_price + x for x in [1, 2]]
        buy_prices1 = [central_price - x for x in [1, 2]]
        
        # Strategy Logic
        if prvs_price1 > 0:
            orders.extend(self.place_orders(product, sell_prices1, full_qty, is_buy=False))
            if prvs_price2 > 0 and sign_product > 0:
                orders.extend(self.place_orders(product, sell_prices2, half_qty, is_buy=False))
                orders.extend(self.place_orders(product, momentum_buy, skimp_qty, is_buy=True))
            elif sign_product == 0:
                orders.extend(self.place_orders(product, momentum_sell, skimp_qty, is_buy=False))
            elif prvs_price2 < 0:
                orders.extend(self.place_orders(product, buy_prices1, half_qty, is_buy=True))
        elif prvs_price1 == 0:
            orders.extend(self.place_orders(product, sell_prices1, half_qty, is_buy=False))
            orders.extend(self.place_orders(product, buy_prices1, half_qty, is_buy=True))
        else:  # prvs_price1 < 0
            orders.extend(self.place_orders(product, buy_prices1, full_qty, is_buy=True))
            if prvs_price2 < 0 and sign_product > 0:
                orders.extend(self.place_orders(product, buy_prices2, half_qty, is_buy=True))
                orders.extend(self.place_orders(product, momentum_sell, skimp_qty, is_buy=False))
            elif sign_product == 0:
                orders.extend(self.place_orders(product, momentum_buy, skimp_qty, is_buy=True))
            elif prvs_price2 > 0:
                orders.extend(self.place_orders(product, sell_prices1, half_qty, is_buy=False))
        
        return orders
    
    def strategy_macd(self, product, state):
        """Strategy for JAMS based on MACD-like signals."""
        orders = []
        
        # Calculate EMA values
        history = self.mid_price_history[product]
        

        # Simple implementation of EMA for short and long periods
        #0. which MACD to choose? (5,13), (8,21), (9,22), (13,28)
        
        #1. Need at least 13 data points for the longer EMA

        if len(history) >= 5:  
            # Calculate EMA-5
            alpha_5 = 2 / (5 + 1)
            ema_5 = [history[0]]  # Initialize with first price
            for i in range(1, len(history)):
                ema_5.append(alpha_5 * history[i] + (1 - alpha_5) * ema_5[-1])
        
        if len(history) >= 8:  
            # Calculate EMA-5
            alpha_8 = 2 / (8 + 1)
            ema_8 = [history[0]]  # Initialize with first price
            for i in range(1, len(history)):
                ema_8.append(alpha_8 * history[i] + (1 - alpha_8) * ema_8[-1])


        if len(history) >= 13:   
            # Calculate EMA-13
            alpha_13 = 2 / (13 + 1)
            ema_13 = [history[0]]  # Initialize with first price
            for i in range(1, len(history)):
                ema_13.append(alpha_13 * history[i] + (1 - alpha_13) * ema_13[-1])

            if len(history) >= 21:   
                # Calculate EMA-13
                alpha_21 = 2 / (21 + 1)
                ema_21 = [history[0]]  # Initialize with first price
                for i in range(1, len(history)):
                    ema_21.append(alpha_21 * history[i] + (1 - alpha_21) * ema_21[-1])
        
            
            # Get last few values for analysis
            ema_5_last3 = ema_5[-3:]
            #ema_8_last3 = ema_8[-3:]
            ema_13_last = ema_13[-2:]
            #ema_21_last = ema_21[-2:]
            
            
            # Check for crossover
            bullish_cross = ema_5_last3[-2] < ema_13_last[-2] and ema_5_last3[-1] > ema_13_last[-1]
            bearish_cross = ema_5_last3[-2] > ema_13_last[-2] and ema_5_last3[-1] < ema_13_last[-1]
            #bullish_cross = ema_8_last3[-2] < ema_21_last[-2] and ema_8_last3[-1] > ema_21_last[-1]
            #bearish_cross = ema_8_last3[-2] > ema_21_last[-2] and ema_8_last3[-1] < ema_21_last[-1]
            
            # stays bullish/bearish
            bullish_hold = ema_5_last3[-1] > ema_13_last[-1]
            bearish_hold = ema_5_last3[-1] < ema_13_last[-1]
            #bullish_hold = ema_8_last3[-1] > ema_21_last[-1]
            #bearish_hold = ema_8_last3[-1] < ema_21_last[-1]

            # Check momentum
            #ema_diff = [ema_5_last3[i+1] - ema_5_last3[i] for i in range(len(ema_5_last3)-1)]
            ema_diff = [ema_5_last3[i+1] - ema_5_last3[i] for i in range(len(ema_5_last3)-1)]

            momentum_signs = np.sign(ema_diff)
            momentum = np.prod(momentum_signs) if len(momentum_signs) > 0 else 0
            
            # Central price for orders
            # this was the issue for low volume in trades: np.mean(history) ignored recent pricetrend 
            # is this the SMA instead of EMA?
            central_price = round(np.mean(ema_5[:-5])) #round(np.mean(history[:-5])) 
            #2. check recent 5, 8, 10, 12, 13, 15
            # currently this exponential smoothing is causing no-trade 
            
            # Current position to determine if we're in a position
            current_pos = self.positions.get(product, 0)
            #3. Following thresholds for claiming long/short positions can be tweaked
            in_long_position = current_pos > 10  # Arbitrary threshold; set it based on each product's position limit
            in_short_position = current_pos < -10  
            
            # Trading logic
            if bullish_hold:
                if momentum >= 0:
                    orders.extend(self.place_orders(product, [central_price], 2, is_buy=True))
                orders.extend(self.place_orders(product, [central_price, central_price-1], 1, is_buy=True))
                #[central_price+1, central_price, central_price-1]

            if bullish_cross: #not elif, trade in all scenarios
                if momentum >= 0:
                    # Strong uptrend
                    orders.extend(self.place_orders(product, [central_price, central_price-1], 3, is_buy=True))
                    #[central_price+1, central_price, central_price-1]
                else:
                    # Weak trend
                    orders.extend(self.place_orders(product, [central_price, central_price-1], 1, is_buy=True))
                    #[central_price, central_price-1, central_price-2]
                if bullish_hold:
                    orders.extend(self.place_orders(product, [central_price, central_price-1], 2, is_buy=True))

            if bearish_hold:
                if momentum >= 0:
                    orders.extend(self.place_orders(product, [central_price], -2, is_buy=False))
                orders.extend(self.place_orders(product, [central_price, central_price+1], -1, is_buy=False))
                #[central_price-1, central_price, central_price+1]

            if bearish_cross: #not elif
                if momentum >= 0:
                    # Strong downtrend
                    orders.extend(self.place_orders(product, [central_price,central_price+1], -3, is_buy=False))
                    #[central_price-1, central_price, central_price+1]
                else:
                    # Weak trend
                    orders.extend(self.place_orders(product, [central_price, central_price+1], -1, is_buy=False))
                    #[central_price, central_price+1, central_price+2]
                
                if bearish_hold:
                    orders.extend(self.place_orders(product, [central_price, central_price+1], -2, is_buy=False))
            
            # Exit conditions 
            if in_long_position and momentum < 0:
                # Close long position (sell) as momentum has reversed
                orders.extend(self.place_orders(product, [central_price-1], -abs(current_pos), is_buy=False))
                #try to exit 
                # see if Market ORders are possible
                #4. is abs(current_pos) correct or the negative of it?
            
            if in_short_position and momentum < 0: 
                # Close short position (buy), since momentum reversed
                orders.extend(self.place_orders(product, [central_price+1], abs(current_pos), is_buy=True))
            # think of SL condition next
        return orders
    
    def strategy_reversion(self, product: str, state, lag1_thresh, lag2_thresh,lookback=20):
        """AR(3) strategy: Forecast next log-return and place directional orders."""
        order_depth = state.order_depths[product]
        orders = []

        # Check if we have enough data points
        if len(self.mid_price_history[product]) < 6:  # Need at least 6 data points
            return []

        # Get prices and compute log returns
        prices = np.array(self.mid_price_history[product])
        log_prices = np.log(prices)
        returns = np.diff(log_prices)

        # Need at least 3 returns for AR(3)
        if len(returns) < 4:  # Need at least 1 more than AR order
            return []

        # Create X matrix with lags
        X = np.zeros((len(returns) - 3, 3))
        for i in range(3):
            X[:, i] = returns[2-i:len(returns)-1-i]
    
        # Create y vector (the values we're trying to predict)
        y = returns[3:]
    
        # Solve for coefficients using least squares
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
        # Extract coefficients for analysis
        lag1_coef = beta[0]  # First lag coefficient
        lag2_coef = beta[1]  # Second lag coefficient
    
        # Predict next return using the last 3 returns
        last_three_returns = returns[-3:]
        predicted_return = np.dot(beta, last_three_returns[::-1])
    
        # Calculate central price similar to strategy_macd
        # But incorporate the effect of the first lag coefficient
        recent_prices = prices[-lookback:]  #0. Tune: (like a Moving average central_price) Use last 5 prices
    
        # Adjust the weighting based on lag1 coefficient
        # 1. Tune: If lag1 is positive, recent prices matter more,  a threshold helps filter out more
        if lag1_coef > lag1_thresh:  # More stringent threshold
            central_price = round(np.mean(recent_prices[-3:]))  # Very recent-focused
        #elif lag1_coef > 0.05:  # Moderate threshold
        #    central_price = round(np.mean(recent_prices[-8:]))  # Moderately recent-focused
        else:
            central_price = round(np.mean(recent_prices[max(-lookback,-20):]))  # Use full history
        # 1_1. Tune: can also use exponential smoothing instead

        # Current position to determine if we're in a position
        current_pos = self.positions.get(product, 0)
        in_long_position = current_pos > 10
        in_short_position = current_pos < -10
    
        # Exit conditions based on second lag significance
        # If lag2 coefficient becomes significant, exit the position
        lag2_significant = abs(lag2_coef) > lag2_thresh  #1. Tune: Threshold can be tuned: 0.5, 0.25, 0.15, 0.1, ...
    
        if lag2_significant:
            if in_long_position:
                #2. Tune: Exit long position
                sell_price = central_price - 1  # Slightly below central to ensure execution
                orders.extend(self.place_orders(product, [sell_price], -abs(current_pos), is_buy=False))
                return orders  # Exit early after placing exit orders
            
            if in_short_position:
                # Exit short position
                buy_price = central_price + 1  # Slightly above central to ensure execution
                orders.extend(self.place_orders(product, [buy_price], abs(current_pos), is_buy=True))
                return orders  # Exit early after placing exit orders
    
        # Trading logic based on predicted return
        qty = 3  # base quantity for each trade
        half_qty = 2

        # MOST tunable parameter as of now
        if predicted_return > 0.002:  #3. Tune: Add a threshold to avoid noise # 0.001, 0.002, 0.005
            # Positive prediction: Buy below central price
            buy_prices = [central_price, central_price - 1]
            momentum_buy = [central_price+1, central_price]
            orders.extend(self.place_orders(product, buy_prices, qty, is_buy=True))
            orders.extend(self.place_orders(product, momentum_buy, half_qty, is_buy=True))
        
        elif predicted_return < -0.002:  # Add a threshold to avoid noise
            # Negative prediction: Sell above central price
            sell_prices = [central_price, central_price + 1]
            momentum_sell = [central_price-1, central_price]
            orders.extend(self.place_orders(product, sell_prices, -qty, is_buy=False))
            orders.extend(self.place_orders(product, momentum_sell, -half_qty, is_buy=False))

        return orders

    def strategy_reversion_00(self, product: str, state):
        """AR(3) strategy: Forecast next log-return and place directional orders."""
        order_depth = state.order_depths[product]
        orders = []

        # Check if we have enough data points
        if len(self.mid_price_history[product]) < 4:
            return []

        # Get recent mid-prices and compute log returns
        prices = np.array(self.mid_price_history[product])
        log_prices = np.log(prices)
        returns = np.diff(log_prices)  # Now returns has len(prices)-1 elements

        # Need at least 3 returns for AR(3)
        if len(returns) < 3:
            return []

        # For AR(3), we need at least 6 returns (3 for X and 3 for y)
        if len(returns) >= 6:
            # Create X matrix with lags
            X = np.zeros((len(returns) - 3, 3))
            for i in range(3):
                X[:, i] = returns[2-i:len(returns)-1-i]
        
            # Create y vector (the values we're trying to predict)
            y = returns[3:]
        
            # Solve for coefficients using least squares
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
            # Predict next return using the last 3 returns
            last_three_returns = returns[-3:]
            predicted_return = np.dot(beta, last_three_returns[::-1])
        else:
            # Simplified model if we don't have enough data
            #6. Tune: Mean reversion assumption: negative correlation with last return
            predicted_return = -0.5 * returns[-1]

        # Get current best bid and ask
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        # Trading logic based on predicted return
        qty = 5  # base quantity for each trade
        if predicted_return > 0 and best_ask is not None:
            # Forecast is positive: Buy signal
            # but don't buy at best_ask that would be a market order!
            orders += self.place_orders(product, [best_ask], qty, is_buy=True)
        elif predicted_return < 0 and best_bid is not None:
            # Forecast is negative: Sell signal
            orders += self.place_orders(product, [best_bid], qty, is_buy=False)

        return orders
