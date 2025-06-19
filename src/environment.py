import numpy as np
import pandas as pd

class MarketEnvironment:
    def __init__(self, price_series, initial_cash=10000):
        self.price_series = price_series.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.t = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.history = []
        return self._get_state()

    def _get_state(self):
        return {
            't': self.t,
            'price': self.price_series[self.t],
            'cash': self.cash,
            'shares': self.shares
        }

    def step(self, action):
        price = self.price_series[self.t]
        if action == 1 and self.cash >= price:
            self.shares += 1
            self.cash -= price
        elif action == 2 and self.shares > 0:
            self.shares -= 1
            self.cash += price
        self.t += 1
        done = self.t >= len(self.price_series) - 1
        next_state = self._get_state()
        portfolio_value = self.cash + self.shares * price
        self.history.append((self.t, price, self.cash, self.shares, portfolio_value))
        return next_state, 0, done

    def get_history(self):
        return pd.DataFrame(self.history, columns=['t', 'price', 'cash', 'shares', 'portfolio_value'])
