import random
import numpy as np
import collections

# Financial parameters
ANNUAL_VOLAT = 0.12
BID_ASK_SP = 1 / 8
DAILY_TRADE_VOL = 5e6
TRAD_DAYS = 250
DAILY_VOLAT = ANNUAL_VOLAT / np.sqrt(TRAD_DAYS)

# Almgren-Chriss parameters
TOTAL_SHARES = 1000000
STARTING_PRICE = 50
LLAMBDA = 1e-6
LIQUIDATION_TIME = 60
NUM_N = 60
EPSILON = BID_ASK_SP / 2
SINGLE_STEP_VARIANCE = (DAILY_VOLAT * STARTING_PRICE) ** 2
ETA = BID_ASK_SP / (0.01 * DAILY_TRADE_VOL)
GAMMA = BID_ASK_SP / (0.1 * DAILY_TRADE_VOL)

class MarketEnvironment():
    def __init__(self, randomSeed=0, lqd_time=LIQUIDATION_TIME, num_tr=NUM_N, lambd=LLAMBDA):
        random.seed(randomSeed)
        self.anv = ANNUAL_VOLAT
        self.basp = BID_ASK_SP
        self.dtv = DAILY_TRADE_VOL
        self.dpv = DAILY_VOLAT
        self.total_shares = TOTAL_SHARES
        self.startingPrice = STARTING_PRICE
        self.llambda = lambd
        self.liquidation_time = lqd_time
        self.num_n = num_tr
        self.epsilon = EPSILON
        self.singleStepVariance = SINGLE_STEP_VARIANCE
        self.eta = ETA
        self.gamma = GAMMA
        self.tau = self.liquidation_time / self.num_n 
        self.eta_hat = self.eta - (0.5 * self.gamma * self.tau)
        self.kappa_hat = np.sqrt((self.llambda * self.singleStepVariance) / self.eta_hat)
        self.kappa = np.arccosh((((self.kappa_hat ** 2) * (self.tau ** 2)) / 2) + 1) / self.tau
        self.shares_remaining = self.total_shares
        self.timeHorizon = self.num_n
        self.logReturns = collections.deque(np.zeros(6))
        self.prevImpactedPrice = self.startingPrice
        self.transacting = False
        self.k = 0

    def reset(self, seed=0, liquid_time=LIQUIDATION_TIME, num_trades=NUM_N, lamb=LLAMBDA):
        self.__init__(randomSeed=seed, lqd_time=liquid_time, num_tr=num_trades, lambd=lamb)
        self.initial_state = np.array(list(self.logReturns) + [self.timeHorizon / self.num_n,
                                                               self.shares_remaining / self.total_shares])
        return self.initial_state

    def start_transactions(self):
        self.transacting = True
        self.tolerance = 1
        self.totalCapture = 0
        self.prevPrice = self.startingPrice
        self.totalSSSQ = 0
        self.totalSRSQ = 0
        self.prevUtility = self.compute_AC_utility(self.total_shares)
        self.last_trade = 0

    def step(self, action, reward_function='ac_utility'):
        class Info: pass
        info = Info()
        info.done = False

        if self.transacting and (self.timeHorizon == 0 or abs(self.shares_remaining) < self.tolerance):
            self.transacting = False
            info.done = True
            info.implementation_shortfall = self.total_shares * self.startingPrice - self.totalCapture
            info.expected_shortfall = self.get_expected_shortfall(self.total_shares)
            info.expected_variance = self.singleStepVariance * self.tau * self.totalSRSQ
            info.utility = info.expected_shortfall + self.llambda * info.expected_variance

        if self.k == 0:
            info.price = self.prevImpactedPrice
        else:
            info.price = self.prevImpactedPrice + np.sqrt(self.singleStepVariance * self.tau) * random.normalvariate(0, 1)

        if self.transacting:
            if isinstance(action, np.ndarray):
                action = action.item()
            sharesToSellNow = self.shares_remaining * action
            if self.timeHorizon < 2:
                sharesToSellNow = self.shares_remaining
            info.share_to_sell_now = np.around(sharesToSellNow)
            info.currentPermanentImpact = self.permanentImpact(info.share_to_sell_now)
            info.currentTemporaryImpact = self.temporaryImpact(info.share_to_sell_now)
            info.exec_price = info.price - info.currentTemporaryImpact
            self.totalCapture += info.share_to_sell_now * info.exec_price
            self.logReturns.append(np.log(info.price / self.prevPrice))
            self.logReturns.popleft()
            self.shares_remaining -= info.share_to_sell_now
            self.totalSSSQ += info.share_to_sell_now ** 2
            self.totalSRSQ += self.shares_remaining ** 2
            self.timeHorizon -= 1
            self.prevPrice = info.price
            self.prevImpactedPrice = info.price - info.currentPermanentImpact
            currentUtility = self.compute_AC_utility(self.shares_remaining)

            # ----------------------------- Reward Logic ----------------------------- #
            if reward_function == 'ac_utility':
                reward = (abs(self.prevUtility) - abs(currentUtility)) / abs(self.prevUtility)
            elif reward_function == 'capture':
                reward = info.share_to_sell_now * info.exec_price / (self.total_shares * self.startingPrice)
            elif reward_function == 'final_shortfall':
                reward = 0.0
            elif reward_function == 'custom_penalty':
                penalty = 0.00001 * (info.share_to_sell_now ** 2)
                reward = (info.share_to_sell_now * info.exec_price - penalty) / (self.total_shares * self.startingPrice)
            elif reward_function == 'stepwise_shortfall':
                reward = - (info.exec_price - self.startingPrice) * info.share_to_sell_now
            elif reward_function == 'hybrid_shortfall_risk':
                risk_penalty = 1e-7 * (self.shares_remaining ** 2)
                reward = - (info.exec_price - self.startingPrice) * info.share_to_sell_now - risk_penalty
            elif reward_function == 'smoothness_penalty':
                if self.k > 1:
                    trade_diff = info.share_to_sell_now - self.last_trade
                    smooth_penalty = 0.00001 * (trade_diff ** 2)
                else:
                    smooth_penalty = 0.0
                reward = - (info.exec_price - self.startingPrice) * info.share_to_sell_now - smooth_penalty
                self.last_trade = info.share_to_sell_now
            elif reward_function == 'baseline_relative':
                reward = 0.0
            elif reward_function == "inv_time_penalty":
                # cost-like term (negative capture) – want to minimise
                step_shortfall = (self.startingPrice - info.exec_price) * info.share_to_sell_now
                step_shortfall /= (self.total_shares * self.startingPrice)

                inv_frac  = self.shares_remaining / self.total_shares
                time_frac = self.timeHorizon / self.num_n

                alpha, beta = 0.2, 0.05      # tune later
                reward = -step_shortfall - alpha * (inv_frac ** 2) - beta * (time_frac ** 2)

            elif reward_function == "risk_adjusted_utility":
                delta_u = (abs(self.prevUtility) - abs(currentUtility)) / abs(self.prevUtility)
                inv_frac = self.shares_remaining / self.total_shares
                gamma = 0.5 * self.llambda / self.tau
                reward = delta_u - gamma * self.singleStepVariance * inv_frac
            else:
                raise ValueError(f"Unknown reward function: {reward_function}")

            self.prevUtility = currentUtility

            if self.shares_remaining <= 0:
                info.implementation_shortfall = self.total_shares * self.startingPrice - self.totalCapture
                info.done = True
        else:
            reward = 0.0

        self.k += 1
        state = np.array(list(self.logReturns) + [self.timeHorizon / self.num_n, self.shares_remaining / self.total_shares])
        return (state, np.array([reward]), info.done, info)

    def permanentImpact(self, sharesToSell):
        return self.gamma * sharesToSell

    def temporaryImpact(self, sharesToSell):
        return (self.epsilon * np.sign(sharesToSell)) + ((self.eta / self.tau) * sharesToSell)

    def get_expected_shortfall(self, sharesToSell):
        ft = 0.5 * self.gamma * (sharesToSell ** 2)
        st = self.epsilon * sharesToSell
        tt = (self.eta_hat / self.tau) * self.totalSSSQ
        return ft + st + tt

    def get_AC_expected_shortfall(self, sharesToSell):
        ft = 0.5 * self.gamma * (sharesToSell ** 2)
        st = self.epsilon * sharesToSell
        tt = self.eta_hat * (sharesToSell ** 2)
        nft = np.tanh(0.5 * self.kappa * self.tau) * (self.tau * np.sinh(2 * self.kappa * self.liquidation_time)
                                                      + 2 * self.liquidation_time * np.sinh(self.kappa * self.tau))
        dft = 2 * (self.tau ** 2) * (np.sinh(self.kappa * self.liquidation_time) ** 2)
        fot = nft / dft
        return ft + st + (tt * fot)

    def get_AC_variance(self, sharesToSell):
        ft = 0.5 * self.singleStepVariance * (sharesToSell ** 2)
        nst = self.tau * np.sinh(self.kappa * self.liquidation_time) * np.cosh(self.kappa * (self.liquidation_time - self.tau)) \
              - self.liquidation_time * np.sinh(self.kappa * self.tau)
        dst = (np.sinh(self.kappa * self.liquidation_time) ** 2) * np.sinh(self.kappa * self.tau)
        st = nst / dst
        return ft * st

    def compute_AC_utility(self, sharesToSell):
        if self.liquidation_time == 0:
            return 0
        E = self.get_AC_expected_shortfall(sharesToSell)
        V = self.get_AC_variance(sharesToSell)
        return E + self.llambda * V

    def observation_space_dimension(self):
        return 8

    def action_space_dimension(self):
        return 1

    def stop_transactions(self):
        self.transacting = False
