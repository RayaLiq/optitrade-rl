from logging import info
import random
import numpy as np
import collections

# ------------------------------------------------ Financial Parameters --------------------------------------------------- #

ANNUAL_VOLAT = 0.12                                # Annual volatility in stock price
BID_ASK_SP = 1 / 8                                 # Bid-ask spread
DAILY_TRADE_VOL = 5e6                              # Average Daily trading volume  
TRAD_DAYS = 250                                    # Number of trading days in a year
DAILY_VOLAT = ANNUAL_VOLAT / np.sqrt(TRAD_DAYS)    # Daily volatility in stock price


# ----------------------------- Parameters for the Almgren and Chriss Optimal Execution Model ----------------------------- #

TOTAL_SHARES = 1000000                                               # Total number of shares to sell
STARTING_PRICE = 50                                                  # Starting price per share
LLAMBDA = 1e-6                                                       # Trader's risk aversion
LIQUIDATION_TIME = 120                                                # How many days to sell all the shares. 
NUM_N = 120                                                           # Number of trades
EPSILON = BID_ASK_SP / 2                                             # Fixed Cost of Selling.
SINGLE_STEP_VARIANCE = (DAILY_VOLAT  * STARTING_PRICE) ** 2          # Calculate single step variance
ETA = BID_ASK_SP / (0.01 * DAILY_TRADE_VOL)                          # Price Impact for Each 1% of Daily Volume Traded
GAMMA = BID_ASK_SP / (0.1 * DAILY_TRADE_VOL)                         # Permanent Impact Constant

# ----------------------------------------------------------------------------------------------------------------------- #

RISK_FREE_RATE = 0.02   # 2% annual risk-free return
MARKET_RETURN = 0.08    # 8% annual market return
BETA = 1.1              # Assumed stock beta vs market

# ------------------ GBM Parameters ------------------- #
DELTA_T = 1 / TRAD_DAYS  # Time step in years
EXPECTED_RETURN = RISK_FREE_RATE + BETA * (MARKET_RETURN - RISK_FREE_RATE)


# --------------- Heston & Merton Parameters ---------------- #
HESTON_KAPPA = 3.0        # Volatility mean-reversion speed
HESTON_THETA = 0.12**2    # Long-term variance (0.12^2)
HESTON_SIGMA_V = 0.1      # Volatility of volatility
HESTON_RHO = -0.7         # Price/vol correlation
HESTON_V0 = 0.12**2       # Initial variance

MERTON_LAMBDA = 0.5       # Jump intensity (jumps/year)
MERTON_MU_J = -0.05       # Mean jump size (log)
MERTON_SIGMA_J = 0.1      # Jump size volatility

# ----------------------- Trading Fee Parameters ----------------------- #
FIXED_FEE_PER_TRADE = 10.0      # $10 fixed fee per transaction
PROPORTIONAL_FEE_RATE = 0.001   # 0.1% fee on trade value

# Simulation Environment

class HestonMertonFeesEnvironment():
    
    def __init__(self, randomSeed = 0,
                 lqd_time = LIQUIDATION_TIME,
                 num_tr = NUM_N,
                 lambd = LLAMBDA,
                 fee_config=None):
        
        # Set the random seed
        random.seed(randomSeed)
        
        # Initialize the financial parameters so we can access them later
        self.anv = ANNUAL_VOLAT
        self.basp = BID_ASK_SP
        self.dtv = DAILY_TRADE_VOL
        self.dpv = DAILY_VOLAT
        
        # Initialize the Almgren-Chriss parameters so we can access them later
        self.total_shares = TOTAL_SHARES
        self.startingPrice = STARTING_PRICE
        self.llambda = lambd
        self.liquidation_time = lqd_time
        self.num_n = num_tr
        self.epsilon = EPSILON
        self.singleStepVariance = SINGLE_STEP_VARIANCE
        self.eta = ETA
        self.gamma = GAMMA

        # Fee parameters (default if not provided)
        fee_config = fee_config or {}
        self.fee_fixed = fee_config.get("fixed", 10.0)
        self.fee_prop = fee_config.get("prop", 0.001)


        # Initialize the GBM parameters
        self.delta_t = DELTA_T
        self.mu = EXPECTED_RETURN
        self.sigma = self.anv  # Annual volatility (sigma)
        
        # Calculate some Almgren-Chriss parameters
        self.tau = self.liquidation_time / self.num_n 
        self.eta_hat = self.eta - (0.5 * self.gamma * self.tau)
        self.kappa_hat = np.sqrt((self.llambda * self.singleStepVariance) / self.eta_hat)
        self.kappa = np.arccosh((((self.kappa_hat ** 2) * (self.tau ** 2)) / 2) + 1) / self.tau

        # Set the variables for the initial state
        self.shares_remaining = self.total_shares
        self.timeHorizon = self.num_n
        self.logReturns = collections.deque(np.zeros(6))
        
        # Set the initial impacted price to the starting price
        self.prevImpactedPrice = self.startingPrice

        # Set the initial transaction state to False
        self.transacting = False
        
        # Set a variable to keep trak of the trade number
        self.k = 0

        # Initialize Heston parameters
        self.heston_kappa = HESTON_KAPPA
        self.heston_theta = HESTON_THETA
        self.heston_sigma_v = HESTON_SIGMA_V
        self.heston_rho = HESTON_RHO
        self.current_variance = HESTON_V0  # Track current variance

        # Initialize Merton parameters
        self.jump_lambda = MERTON_LAMBDA
        self.jump_mu = MERTON_MU_J
        self.jump_sigma = MERTON_SIGMA_J

        self.current_variance = HESTON_V0  # Reset variance
        
        
    def reset(self, seed = 0, liquid_time = LIQUIDATION_TIME, num_trades = NUM_N, lamb = LLAMBDA):
        
        
        # Initialize the environment with the given parameters
        self.__init__(randomSeed = seed, lqd_time = liquid_time, num_tr = num_trades, lambd = lamb)
        
        # Set the initial state to [0,0,0,0,0,0,1,1]

        self.initial_state = np.array(list(self.logReturns) + [self.timeHorizon / self.num_n, self.shares_remaining / self.total_shares, 
                                                               np.log(self.current_variance) if self.current_variance > 0 else -10,  # Log variance 
                                                               1 if (self.jump_lambda > 0 and random.random() < 0.05) else 0  # Jump indicator 
                                                               ])

        return self.initial_state

    
    def start_transactions(self):
        
        # Set transactions on
        self.transacting = True
        
        # Set the minimum number of stocks one can sell
        self.tolerance = 1
        
        # Set the initial capture to zero
        self.totalCapture = 0
        
        # Set the initial previous price to the starting price
        self.prevPrice = self.startingPrice
        
        # Set the initial square of the shares to sell to zero
        self.totalSSSQ = 0
        
        # Set the initial square of the remaing shares to sell to zero
        self.totalSRSQ = 0
        
        # Set the initial AC utility
        self.prevUtility = self.compute_AC_utility(self.total_shares)
        

    def step(self, action, reward_function='default'):
        
        # Create a class that will be used to keep track of information about the transaction
        class Info(object):
            pass        
        info = Info()
        
        # Set the done flag to False. This indicates that we haven't sold all the shares yet.
        info.done = False
                
        # During training, if the DDPG fails to sell all the stocks before the given 
        # number of trades or if the total number shares remaining is less than 1, then stop transacting,
        # set the done Flag to True, return the current implementation shortfall, and give a negative reward.
        # The negative reward is given in the else statement below.
        if self.transacting and (self.timeHorizon == 0 or abs(self.shares_remaining) < self.tolerance):
            self.transacting = False
            info.done = True
            info.implementation_shortfall = self.total_shares * self.startingPrice - self.totalCapture

            # Calculate total fees paid
            info.total_fixed_fees = self.fee_fixed * (self.num_n - self.timeHorizon)
            gross_trade_value = self.total_shares * self.startingPrice
            info.total_proportional_fees = self.fee_prop * (gross_trade_value - self.totalCapture)
            info.total_fees = info.total_fixed_fees + info.total_proportional_fees



            info.expected_shortfall = self.get_expected_shortfall(self.total_shares)
            info.expected_variance = self.singleStepVariance * self.tau * self.totalSRSQ
            info.utility = info.expected_shortfall + self.llambda * info.expected_variance
            
        # We don't add noise before the first trade    
        if self.k == 0:
            info.price = self.prevImpactedPrice

        else:
            # Convert time step to years
            tau_in_years = self.tau / TRAD_DAYS
    
            # 1. Generate correlated Brownian motions
            Z1 = random.normalvariate(0, 1)
            Z2 = random.normalvariate(0, 1)
            dW1 = Z1 * np.sqrt(tau_in_years)
            dW2 = self.heston_rho * dW1 + np.sqrt(1 - self.heston_rho**2) * Z2 * np.sqrt(tau_in_years)
    
            # 2. Update Heston variance (with full truncation scheme)
            v_old = self.current_variance
            v_plus = max(v_old, 0)  # Ensure non-negative
            v_new = v_old + self.heston_kappa * (self.heston_theta - v_plus) * tau_in_years + \
                 self.heston_sigma_v * np.sqrt(v_plus) * dW2
            self.current_variance = max(v_new, 0)  # Truncate at 0
    
            # 3. Calculate log return from Heston model
            log_return = (self.mu - 0.5 * v_plus) * tau_in_years + np.sqrt(v_plus) * dW1
    
            # 4. Add Merton jumps
            if self.jump_lambda > 0:
                jump_prob = 1 - np.exp(-self.jump_lambda * tau_in_years)
                if random.random() < jump_prob:
                    jump_size = np.exp(self.jump_mu + self.jump_sigma * random.normalvariate(0, 1))
                    log_return += np.log(jump_size)
    
            # 5. Compute new fundamental price
            fundamental_price = self.prevImpactedPrice * np.exp(log_return)
            info.price = fundamental_price

        # If we are transacting, the stock price is affected by the number of shares we sell. The price evolves 
        # according to the Almgren and Chriss price dynamics model. 
        if self.transacting:
            
            # If action is an ndarray then extract the number from the array
            if isinstance(action, np.ndarray):
                action = action.item()            

            # Convert the action to the number of shares to sell in the current step
            sharesToSellNow = self.shares_remaining * action
            #sharesToSellNow = min(self.shares_remaining * action, self.shares_remaining)
    
            if self.timeHorizon < 2:
                sharesToSellNow = self.shares_remaining

            # Since we are not selling fractions of shares, round up the total number of shares to sell to the nearest integer. 
            info.share_to_sell_now = np.around(sharesToSellNow)

            # Calculate the permanent and temporary impact on the stock price according the AC price dynamics model
            info.currentPermanentImpact = self.permanentImpact(info.share_to_sell_now)
            info.currentTemporaryImpact = self.temporaryImpact(info.share_to_sell_now)
                
            # Apply the temporary impact on the current stock price    
            info.exec_price = info.price - info.currentTemporaryImpact
            
            # Calculate the current total capture
            # Calculate trade value before fees
            trade_value = info.share_to_sell_now * info.exec_price

            # Calculate trading fees
            fixed_fee, proportional_fee = self.compute_fees(info.exec_price, info.share_to_sell_now)
            total_fees = fixed_fee + proportional_fee

            # Calculate net proceeds after fees
            net_proceeds = trade_value - total_fees

            # Update total capture with net proceeds
            self.totalCapture += net_proceeds

            # Store fee information in the info object
            info.fixed_fee = fixed_fee
            info.proportional_fee = proportional_fee
            info.total_fees = total_fees

            # Calculate the log return for the current step and save it in the logReturn deque
            self.logReturns.append(np.log(info.price/self.prevPrice))
            self.logReturns.popleft()
            
            # Update the number of shares remaining
            self.shares_remaining -= info.share_to_sell_now
            
            # Calculate the runnig total of the squares of shares sold and shares remaining
            self.totalSSSQ += info.share_to_sell_now ** 2
            self.totalSRSQ += self.shares_remaining ** 2
                                        
            # Update the variables required for the next step
            self.timeHorizon -= 1
            self.prevPrice = info.price
            self.prevImpactedPrice = info.price - info.currentPermanentImpact
            
            # Calculate incremental implementation shortfall improvement
            current_value = self.shares_remaining * self.startingPrice
            new_value = self.shares_remaining * info.price
            if reward_function == "final_shortfall" and info.done:
                reward = -info.implementation_shortfall / (self.total_shares * self.startingPrice)
            elif reward_function == "stepwise_profit":
                reward = (current_value - new_value) / self.startingPrice
            elif reward_function == "fees_penalty":
                reward = -info.total_fees / self.total_shares
            elif reward_function == "hybrid":
                reward = (current_value - new_value) / self.startingPrice - 0.00001 * info.total_fees
            else:
                reward = (current_value - new_value) / self.startingPrice



        else:
            reward = 0.0
        
        self.k += 1
            
        # Set the new state
        # Replace existing state with:
        state = np.array(list(self.logReturns) + [self.timeHorizon / self.num_n, self.shares_remaining / self.total_shares,
                                                  np.log(self.current_variance) if self.current_variance > 0 else -10,
                                                  1 if (self.jump_lambda > 0 and random.random() < 0.05) else 0
                                                  ])

        return (state, np.array([reward]), info.done, info)

   
    def permanentImpact(self, sharesToSell):
        # Calculate the permanent impact according to equations (6) and (1) of the AC paper
        pi = self.gamma * sharesToSell
        return pi

    
    def temporaryImpact(self, sharesToSell):
        # Calculate the temporary impact according to equation (7) of the AC paper
        ti = (self.epsilon * np.sign(sharesToSell)) + ((self.eta / self.tau) * sharesToSell)
        return ti
    
    def compute_fees(self, exec_price, shares):
        value = exec_price * shares
        fixed = FIXED_FEE_PER_TRADE if shares > 0 else 0
        proportional = PROPORTIONAL_FEE_RATE * value
        return fixed, proportional

    
    def get_expected_shortfall(self, sharesToSell):
        # Calculate the expected shortfall according to equation (8) of the AC paper
        ft = 0.5 * self.gamma * (sharesToSell ** 2)        
        st = self.epsilon * sharesToSell
        tt = (self.eta_hat / self.tau) * self.totalSSSQ

        # Add expected fees (fixed fee per trade + proportional fee)
        expected_trades = self.num_n
        expected_fixed_fees = self.fee_fixed * expected_trades
        expected_prop_fees = self.fee_prop * (sharesToSell * self.startingPrice)

    
        return ft + st + tt + expected_fixed_fees + expected_prop_fees

    
    def get_AC_expected_shortfall(self, sharesToSell):
        # Calculate the expected shortfall for the optimal strategy according to equation (20) of the AC paper
        ft = 0.5 * self.gamma * (sharesToSell ** 2)        
        st = self.epsilon * sharesToSell        
        tt = self.eta_hat * (sharesToSell ** 2)

        nft = np.tanh(0.5 * self.kappa * self.tau) * (self.tau * np.sinh(2 * self.kappa * self.liquidation_time) \
                                                      + 2 * self.liquidation_time * np.sinh(self.kappa * self.tau))       
        dft = 2 * (self.tau ** 2) * (np.sinh(self.kappa * self.liquidation_time) ** 2)   
        fot = nft / dft  
        ac_shortfall = ft + st + (tt * fot)  

        # Add expected fees
        expected_trades = self.num_n
        expected_fixed_fees = FIXED_FEE_PER_TRADE * expected_trades
        expected_prop_fees = PROPORTIONAL_FEE_RATE * (sharesToSell * self.startingPrice)
   
        return ac_shortfall + expected_fixed_fees + expected_prop_fees  
        
    
    def get_AC_variance(self, sharesToSell):
        # Calculate the variance for the optimal strategy according to equation (20) of the AC paper
        ft = 0.5 * (self.singleStepVariance) * (sharesToSell ** 2)                        
        nst  = self.tau * np.sinh(self.kappa * self.liquidation_time) * np.cosh(self.kappa * (self.liquidation_time - self.tau)) \
               - self.liquidation_time * np.sinh(self.kappa * self.tau)        
        dst = (np.sinh(self.kappa * self.liquidation_time) ** 2) * np.sinh(self.kappa * self.tau)        
        st = nst / dst
        return ft * st
        
        
    def compute_AC_utility(self, sharesToSell):    
        # Calculate the AC Utility according to pg. 13 of the AC paper
        if self.liquidation_time == 0:
            return 0        
        E = self.get_AC_expected_shortfall(sharesToSell)
        V = self.get_AC_variance(sharesToSell)
        return E + self.llambda * V
    
    
    def get_trade_list(self):
        # Calculate the trade list for the optimal strategy according to equation (18) of the AC paper
        trade_list = np.zeros(self.num_n)
        ftn = 2 * np.sinh(0.5 * self.kappa * self.tau)
        ftd = np.sinh(self.kappa * self.liquidation_time)
        ft = (ftn / ftd) * self.total_shares
        for i in range(1, self.num_n + 1):       
            st = np.cosh(self.kappa * (self.liquidation_time - (i - 0.5) * self.tau))
            trade_list[i - 1] = st
        trade_list *= ft
        return trade_list
     
        
    def observation_space_dimension(self):

        return 10  # Was previously 8
    
    
    def action_space_dimension(self):
        # Return the dimension of the action
        return 1
    
    
    def stop_transactions(self):
        # Stop transacting
        self.transacting = False            
            
           