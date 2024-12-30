# auction.py
import math

class Auction:
    def __init__(self, config, buyers, sellers):
        self.config = config
        self.buyers = buyers
        self.sellers = sellers

        self.num_rounds = config["num_rounds"]
        self.num_periods = config["num_periods"]
        self.num_steps = config["num_steps"]
        self.min_price = config["min_price"]
        self.max_price = config["max_price"]
        self.pmin = -math.inf
        self.pmax = math.inf

        self.logs = []
        self.current_bid = None  # (price, Trader)
        self.current_ask = None  # (price, Trader)

        # We'll track indices for logging
        self.trader_index = {}
        for i, b in enumerate(buyers):
            self.trader_index[b] = f"B{i}"
        for j, s in enumerate(sellers):
            self.trader_index[s] = f"S{j}"

    def run_auction(self):
        for r in range(self.num_rounds):
            self.run_round(r)

    def run_round(self, r):
        for p in range(self.num_periods):
            for t in self.buyers + self.sellers:
                t.reset_for_new_period(self.num_steps, r, p)

            if p == 0:
                self.pmin = -math.inf
                self.pmax = math.inf

            self.current_bid = None
            self.current_ask = None
            period_trades = []

            for st in range(self.num_steps):
                price = self.run_step(r, p, st)
                if price is not None:
                    period_trades.append(price)

            if period_trades:
                self.pmin = min(period_trades)
                self.pmax = max(period_trades)
            else:
                self.pmin = -math.inf
                self.pmax = math.inf

    def run_step(self, r, p, st):
        # Mark the step
        for t in self.buyers + self.sellers:
            t.current_step = st

        # Collect buyer bids
        all_bids = []
        new_bids = []
        for b in self.buyers:
            cbid = self.current_bid[0] if self.current_bid else None
            cask = self.current_ask[0] if self.current_ask else None
            offer = b.make_bid_or_ask(cbid, cask, self.pmin, self.pmax,
                                      self.min_price, self.max_price)
            if offer is not None:
                all_bids.append(round(offer[0], 2))
                new_bids.append(offer)
            else:
                all_bids.append(None)

        # Collect seller asks
        all_asks = []
        new_asks = []
        for s in self.sellers:
            cbid = self.current_bid[0] if self.current_bid else None
            cask = self.current_ask[0] if self.current_ask else None
            offer = s.make_bid_or_ask(cbid, cask, self.pmin, self.pmax,
                                      self.min_price, self.max_price)
            if offer is not None:
                all_asks.append(round(offer[0], 2))
                new_asks.append(offer)
            else:
                all_asks.append(None)

        # Update best bid
        if new_bids:
            best_new_bid = max(new_bids, key=lambda x: x[0])
            if (self.current_bid is None) or (best_new_bid[0] > self.current_bid[0]):
                self.current_bid = best_new_bid

        # Update best ask
        if new_asks:
            best_new_ask = min(new_asks, key=lambda x: x[0])
            if (self.current_ask is None) or (best_new_ask[0] < self.current_ask[0]):
                self.current_ask = best_new_ask

        # Attempt trade
        trade = False
        price = None
        bprofit = None
        sprofit = None
        bind_ = None
        sind_ = None

        if self.current_bid and self.current_ask:
            bid_val, bid_trader = self.current_bid
            ask_val, ask_trader = self.current_ask
            if bid_trader.decide_to_buy(ask_val) and ask_trader.decide_to_sell(bid_val):
                trade = True
                price = 0.5*(bid_val + ask_val)
                bprofit = round(bid_trader.transact(price), 2)
                sprofit = round(ask_trader.transact(price), 2)
                bind_ = self.trader_index[bid_trader]
                sind_ = self.trader_index[ask_trader]
                self.current_bid = None
                self.current_ask = None

        # Log
        cbid_ = round(self.current_bid[0], 2) if self.current_bid else None
        cask_ = round(self.current_ask[0], 2) if self.current_ask else None
        price_ = round(price, 2) if price else None

        self.logs.append({
            "r": r,
            "p": p,
            "st": st,
            "bids": all_bids,
            "asks": all_asks,
            "cbid": cbid_,
            "cask": cask_,
            "trade": trade,
            "price": price_,
            "bprofit": bprofit,
            "sprofit": sprofit,
            "bind": bind_,
            "sind": sind_
        })

        return price
