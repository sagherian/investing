import numpy as np

class HyperbolicActiveInferenceAgent:
    def __init__(self, k=0.1, planning_horizon=5):
        self.k = k
        self.planning_horizon = planning_horizon

    def hyperbolic_discount(self, reward, t):
        return reward / (1 + self.k * t)

    def plan(self, env, predicted_prices):
        current_state = env._get_state()
        price = current_state['price']
        cash = current_state['cash']
        shares = current_state['shares']
        t = current_state['t']

        actions = [0, 1, 2]  # hold, buy, sell
        values = []
        for action in actions:
            sim_cash, sim_shares = cash, shares
            sim_price = price
            if action == 1 and sim_cash >= sim_price:
                sim_cash -= sim_price
                sim_shares += 1
            elif action == 2 and sim_shares > 0:
                sim_cash += sim_price
                sim_shares -= 1
            # Use predicted price for next step
            if t + 1 < len(predicted_prices):
                next_price = predicted_prices[t + 1]
            else:
                next_price = sim_price
            future_value = sim_cash + sim_shares * next_price
            discounted_value = self.hyperbolic_discount(future_value, 1)
            values.append(discounted_value)
        best_action = np.argmax(values)
        return actions[best_action]
