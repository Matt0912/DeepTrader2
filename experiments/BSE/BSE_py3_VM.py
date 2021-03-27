# -*- coding: utf-8 -*-
#
# BSE: The Bristol Stock Exchange
#
# Version 1.3; July 21st, 2018.
# Version 1.2; November 17th, 2012. 
#
# Copyright (c) 2012-2018, Dave Cliff
#
#
# ------------------------
#
# MIT Open-Source License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ------------------------
#
#
#
# BSE is a very simple simulation of automated execution traders
# operating on a very simple model of a limit order book (LOB) exchange
#
# major simplifications in this version:
#       (a) only one financial instrument being traded
#       (b) traders can only trade contracts of size 1 (will add variable quantities later)
#       (c) each trader can have max of one order per single orderbook.
#       (d) traders can replace/overwrite earlier orders, and/or can cancel
#       (d) simply processes each order in sequence and republishes LOB to all traders
#           => no issues with exchange processing latency/delays or simultaneously issued orders.
#
# NB this code has been written to be readable/intelligible, not efficient!

# could import pylab here for graphing etc

import sys
import os
import math
import random

import numpy as np
from numpy import newaxis
import more_itertools as miter
import tensorflow as tf
from tensorflow import keras


bse_sys_minprice = 1  # minimum price in the system, in cents/pennies
bse_sys_maxprice = 1000  # maximum price in the system, in cents/pennies
ticksize = 1  # minimum change in price, in cents/pennies

# an Order/quote has a trader id, a type (buy/sell) price, quantity, timestamp, and unique i.d.
class Order:

        def __init__(self, tid, otype, price, qty, time, qid):
                self.tid = tid      # trader i.d.
                self.otype = otype  # order type
                self.price = price  # price
                self.qty = qty      # quantity
                self.time = time    # timestamp
                self.qid = qid      # quote i.d. (unique to each quote)

        def __str__(self):
                return '[%s %s P=%03d Q=%s T=%5.2f QID:%d]' % \
                       (self.tid, self.otype, self.price, self.qty, self.time, self.qid)



# Orderbook_half is one side of the book: a list of bids or a list of asks, each sorted best-first

class Orderbook_half:

        def __init__(self, booktype, worstprice):
                # booktype: bids or asks?
                self.booktype = booktype
                # dictionary of orders received, indexed by Trader ID
                self.orders = {}
                # limit order book, dictionary indexed by price, with order info
                self.lob = {}
                # anonymized LOB, lists, with only price/qty info
                self.lob_anon = []
                # summary stats
                self.best_price = None
                self.best_tid = None
                self.worstprice = worstprice
                self.n_orders = 0  # how many orders?
                self.lob_depth = 0  # how many different prices on lob?


        def anonymize_lob(self):
                # anonymize a lob, strip out order details, format as a sorted list
                # NB for asks, the sorting should be reversed
                self.lob_anon = []
                for price in sorted(self.lob):
                        qty = self.lob[price][0]
                        self.lob_anon.append([price, qty])


        def build_lob(self):
                lob_verbose = False
                # take a list of orders and build a limit-order-book (lob) from it
                # NB the exchange needs to know arrival times and trader-id associated with each order
                # returns lob as a dictionary (i.e., unsorted)
                # also builds anonymized version (just price/quantity, sorted, as a list) for publishing to traders
                self.lob = {}
                for tid in self.orders:
                        order = self.orders.get(tid)
                        price = order.price
                        if price in self.lob:
                                # update existing entry
                                qty = self.lob[price][0]
                                orderlist = self.lob[price][1]
                                orderlist.append([order.time, order.qty, order.tid, order.qid])
                                self.lob[price] = [qty + order.qty, orderlist]
                        else:
                                # create a new dictionary entry
                                self.lob[price] = [order.qty, [[order.time, order.qty, order.tid, order.qid]]]
                # create anonymized version
                self.anonymize_lob()
                # record best price and associated trader-id
                if len(self.lob) > 0 :
                        if self.booktype == 'Bid':
                                self.best_price = self.lob_anon[-1][0]
                        else :
                                self.best_price = self.lob_anon[0][0]
                        self.best_tid = self.lob[self.best_price][1][0][2]
                else :
                        self.best_price = None
                        self.best_tid = None

                if lob_verbose : print(self.lob)


        def book_add(self, order):
                # add order to the dictionary holding the list of orders
                # either overwrites old order from this trader
                # or dynamically creates new entry in the dictionary
                # so, max of one order per trader per list
                # checks whether length or order list has changed, to distinguish addition/overwrite
                #print('book_add > %s %s' % (order, self.orders))
                n_orders = self.n_orders
                self.orders[order.tid] = order
                self.n_orders = len(self.orders)
                self.build_lob()
                #print('book_add < %s %s' % (order, self.orders))
                if n_orders != self.n_orders :
                    return('Addition')
                else:
                    return('Overwrite')



        def book_del(self, order):
                # delete order from the dictionary holding the orders
                # assumes max of one order per trader per list
                # checks that the Trader ID does actually exist in the dict before deletion
                # print('book_del %s',self.orders)
                if self.orders.get(order.tid) != None :
                        del(self.orders[order.tid])
                        self.n_orders = len(self.orders)
                        self.build_lob()
                # print('book_del %s', self.orders)


        def delete_best(self):
                # delete order: when the best bid/ask has been hit, delete it from the book
                # the TraderID of the deleted order is return-value, as counterparty to the trade
                best_price_orders = self.lob[self.best_price]
                best_price_qty = best_price_orders[0]
                best_price_counterparty = best_price_orders[1][0][2]
                if best_price_qty == 1:
                        # here the order deletes the best price
                        del(self.lob[self.best_price])
                        del(self.orders[best_price_counterparty])
                        self.n_orders = self.n_orders - 1
                        if self.n_orders > 0:
                                if self.booktype == 'Bid':
                                        self.best_price = max(self.lob.keys())
                                else:
                                        self.best_price = min(self.lob.keys())
                                self.lob_depth = len(self.lob.keys())
                        else:
                                self.best_price = self.worstprice
                                self.lob_depth = 0
                else:
                        # best_bid_qty>1 so the order decrements the quantity of the best bid
                        # update the lob with the decremented order data
                        self.lob[self.best_price] = [best_price_qty - 1, best_price_orders[1][1:]]

                        # update the bid list: counterparty's bid has been deleted
                        del(self.orders[best_price_counterparty])
                        self.n_orders = self.n_orders - 1
                self.build_lob()
                return best_price_counterparty



# Orderbook for a single instrument: list of bids and list of asks

class Orderbook(Orderbook_half):

        def __init__(self):
                self.bids = Orderbook_half('Bid', bse_sys_minprice)
                self.asks = Orderbook_half('Ask', bse_sys_maxprice)
                self.tape = []
                self.quote_id = 0  #unique ID code for each quote accepted onto the book



# Exchange's internal orderbook

class Exchange(Orderbook):

        def add_order(self, order, verbose):
                # add a quote/order to the exchange and update all internal records; return unique i.d.
                order.qid = self.quote_id
                self.quote_id = order.qid + 1
                # if verbose : print('QUID: order.quid=%d self.quote.id=%d' % (order.qid, self.quote_id))
                tid = order.tid
                if order.otype == 'Bid':
                        response=self.bids.book_add(order)
                        best_price = self.bids.lob_anon[-1][0]
                        self.bids.best_price = best_price
                        self.bids.best_tid = self.bids.lob[best_price][1][0][2]
                else:
                        response=self.asks.book_add(order)
                        best_price = self.asks.lob_anon[0][0]
                        self.asks.best_price = best_price
                        self.asks.best_tid = self.asks.lob[best_price][1][0][2]
                return [order.qid, response]


        def del_order(self, time, order, verbose):
                # delete a trader's quot/order from the exchange, update all internal records
                tid = order.tid
                if order.otype == 'Bid':
                        self.bids.book_del(order)
                        if self.bids.n_orders > 0 :
                                best_price = self.bids.lob_anon[-1][0]
                                self.bids.best_price = best_price
                                self.bids.best_tid = self.bids.lob[best_price][1][0][2]
                        else: # this side of book is empty
                                self.bids.best_price = None
                                self.bids.best_tid = None
                        cancel_record = { 'type': 'Cancel', 'time': time, 'order': order }
                        self.tape.append(cancel_record)

                elif order.otype == 'Ask':
                        self.asks.book_del(order)
                        if self.asks.n_orders > 0 :
                                best_price = self.asks.lob_anon[0][0]
                                self.asks.best_price = best_price
                                self.asks.best_tid = self.asks.lob[best_price][1][0][2]
                        else: # this side of book is empty
                                self.asks.best_price = None
                                self.asks.best_tid = None
                        cancel_record = { 'type': 'Cancel', 'time': time, 'order': order }
                        self.tape.append(cancel_record)
                else:
                        # neither bid nor ask?
                        sys.exit('bad order type in del_quote()')



        def process_order2(self, time, order, verbose):
                # receive an order and either add it to the relevant LOB (ie treat as limit order)
                # or if it crosses the best counterparty offer, execute it (treat as a market order)
                oprice = order.price
                counterparty = None
                [qid, response] = self.add_order(order, verbose)  # add it to the order lists -- overwriting any previous order
                order.qid = qid
                if verbose :
                        print('QUID: order.quid=%d' % order.qid)
                        print('RESPONSE: %s' % response)
                best_ask = self.asks.best_price
                best_ask_tid = self.asks.best_tid
                best_bid = self.bids.best_price
                best_bid_tid = self.bids.best_tid
                if order.otype == 'Bid':
                        if self.asks.n_orders > 0 and best_bid >= best_ask:
                                # bid lifts the best ask
                                if verbose: print("Bid $%s lifts best ask" % oprice)
                                counterparty = best_ask_tid
                                price = best_ask  # bid crossed ask, so use ask price
                                if verbose: print('counterparty, price', counterparty, price)
                                # delete the ask just crossed
                                self.asks.delete_best()
                                # delete the bid that was the latest order
                                self.bids.delete_best()
                elif order.otype == 'Ask':
                        if self.bids.n_orders > 0 and best_ask <= best_bid:
                                # ask hits the best bid
                                if verbose: print("Ask $%s hits best bid" % oprice)
                                # remove the best bid
                                counterparty = best_bid_tid
                                price = best_bid  # ask crossed bid, so use bid price
                                if verbose: print('counterparty, price', counterparty, price)
                                # delete the bid just crossed, from the exchange's records
                                self.bids.delete_best()
                                # delete the ask that was the latest order, from the exchange's records
                                self.asks.delete_best()
                else:
                        # we should never get here
                        sys.exit('process_order() given neither Bid nor Ask')
                # NB at this point we have deleted the order from the exchange's records
                # but the two traders concerned still have to be notified
                if verbose: print('counterparty %s' % counterparty)
                if counterparty != None:
                        # process the trade
                        if verbose: print('>>>>>>>>>>>>>>>>>TRADE t=%5.2f $%d %s %s' % (time, price, counterparty, order.tid))
                        transaction_record = { 'type': 'Trade',
                                               'time': time,
                                               'price': price,
                                               'party1':counterparty,
                                               'party2':order.tid,
                                               'qty': order.qty
                                              }
                        self.tape.append(transaction_record)
                        return transaction_record
                else:
                        return None



        def tape_dump(self, fname, fmode, tmode):
                dumpfile = open(fname, fmode)
                for tapeitem in self.tape:
                        if tapeitem['type'] == 'Trade' :
                                dumpfile.write('%s, %s\n' % (tapeitem['time'], tapeitem['price']))
                dumpfile.close()
                if tmode == 'wipe':
                        self.tape = []


        # this returns the LOB data "published" by the exchange,
        # i.e., what is accessible to the traders
        def publish_lob(self, time, verbose):
                public_data = {}
                public_data['time'] = time
                public_data['bids'] = {'best':self.bids.best_price,
                                     'worst':self.bids.worstprice,
                                     'n': self.bids.n_orders,
                                     'lob':self.bids.lob_anon}
                public_data['asks'] = {'best':self.asks.best_price,
                                     'worst':self.asks.worstprice,
                                     'n': self.asks.n_orders,
                                     'lob':self.asks.lob_anon}
                public_data['QID'] = self.quote_id
                public_data['tape'] = self.tape
                if verbose:
                        print('publish_lob: t=%d' % time)
                        print('BID_lob=%s' % public_data['bids']['lob'])
                        # print('best=%s; worst=%s; n=%s ' % (self.bids.best_price, self.bids.worstprice, self.bids.n_orders))
                        print('ASK_lob=%s' % public_data['asks']['lob'])
                        # print('qid=%d' % self.quote_id)

                return public_data






##################--Traders below here--#############


# Trader superclass
# all Traders have a trader id, bank balance, blotter, and list of orders to execute
class Trader:

        def __init__(self, ttype, tid, balance, time):
                self.ttype = ttype      # what type / strategy this trader is
                self.tid = tid          # trader unique ID code
                self.balance = balance  # money in the bank
                self.blotter = []       # record of trades executed
                self.orders = []        # customer orders currently being worked (fixed at 1)
                self.n_quotes = 0       # number of quotes live on LOB
                self.willing = 1        # used in ZIP etc
                self.able = 1           # used in ZIP etc
                self.birthtime = time   # used when calculating age of a trader/strategy
                self.profitpertime = 0  # profit per unit time
                self.n_trades = 0       # how many trades has this trader done?
                self.lastquote = None   # record of what its last quote was

                ## Matt variables
                self.trade_details = [] # updated using recordTrade


        def __str__(self):
                return '[TID %s type %s balance %s blotter %s orders %s n_trades %s profitpertime %s]' \
                       % (self.tid, self.ttype, self.balance, self.blotter, self.orders, self.n_trades, self.profitpertime)


        def add_order(self, order, verbose):
                # in this version, trader has at most one order,
                # if allow more than one, this needs to be self.orders.append(order)
                if self.n_quotes > 0 :
                    # this trader has a live quote on the LOB, from a previous customer order
                    # need response to signal cancellation/withdrawal of that quote
                    response = 'LOB_Cancel'
                else:
                    response = 'Proceed'
                self.orders = [order]
                if verbose : print('add_order < response=%s' % response)
                return response


        def del_order(self, order):
                # this is lazy: assumes each trader has only one customer order with quantity=1, so deleting sole order
                # CHANGE TO DELETE THE HEAD OF THE LIST AND KEEP THE TAIL
                self.orders = []

        def bookkeep(self, trade, order, verbose, time):

                outstr=""
                for order in self.orders: outstr = outstr + str(order)

                self.blotter.append(trade)  # add trade record to trader's blotter
                # NB What follows is **LAZY** -- assumes all orders are quantity=1
                transactionprice = trade['price']
                if self.orders[0].otype == 'Bid':
                        profit = self.orders[0].price - transactionprice
                else:
                        profit = transactionprice - self.orders[0].price
                self.balance += profit
                self.n_trades += 1
                self.profitpertime = self.balance/(time - self.birthtime)

                if profit < 0 :
                        print(profit)
                        print(trade)
                        print(order)
                        sys.exit()

                if verbose: print('%s profit=%d balance=%d profit/time=%d' % (outstr, profit, self.balance, self.profitpertime))
                self.del_order(order)  # delete the order

        # trade = (type, time, price, party1, party2, qty)
        # order = original customer order = (trader id = tid, order type = otype, quote_price, qty, time, quote id (qid))
        def recordTrade(self, trade, order, lob):
                ## Record best bid and ask at time of trade
                best_bid = 0
                best_ask = 0
                best_bid_q = 0
                best_ask_q = 0

                if (lob != None):
                        best_bid = lob['bids']['best']
                        best_ask = lob['asks']['best']

                        ## Calcuate quantity of best bid/ask for microprice calc
                        num_diff_bids = len(lob['bids']['lob'])
                        num_diff_asks = len(lob['asks']['lob'])
                        
                        # best bid at end of orderbook, best ask at beginning
                        if (num_diff_bids > 0):
                                best_bid_q = lob['bids']['lob'][num_diff_bids-1][1]
                        if (num_diff_asks > 0):
                                best_ask_q = lob['asks']['lob'][0][1]

                ## Use the tape to get the time of the previous trade
                tape = lob['tape']
                prev_trade_time = 0
                if (len(tape) >= 2):
                        prev_trade_time = tape[len(tape)-2]['time']

                ## Store all details in market_details dict and append to trade_details
                market_details = { 'trade_time': trade['time'],
                                   'order_type': order.otype,
                                   'cust_order': order.price,
                                   'best_bid': best_bid,
                                   'best_ask': best_ask,
                                   'best_bid_q': best_bid_q,
                                   'best_ask_q': best_ask_q, 
                                   'prev_trade_time': prev_trade_time,
                                   'n_bids': lob['bids']['n'],
                                   'n_asks': lob['asks']['n'],
                                   'trade_price': trade['price']
                                }

                self.trade_details.append(market_details)


        # specify how trader responds to events in the market
        # this is a null action, expect it to be overloaded by specific algos
        def respond(self, time, lob, trade, verbose):
                return None

        # specify how trader mutates its parameter values
        # this is a null action, expect it to be overloaded by specific algos
        def mutate(self, time, lob, trade, verbose):
                return None



# Trader subclass Giveaway
# even dumber than a ZI-U: just give the deal away
# (but never makes a loss)
class Trader_Giveaway(Trader):

        def getorder(self, time, countdown, lob):
                if len(self.orders) < 1:
                        order = None
                else:
                        quoteprice = self.orders[0].price
                        order = Order(self.tid,
                                    self.orders[0].otype,
                                    quoteprice,
                                    self.orders[0].qty,
                                    time, lob['QID'])
                        self.lastquote=order
                return order



# Trader subclass ZI-C
# After Gode & Sunder 1993
class Trader_ZIC(Trader):

        def getorder(self, time, countdown, lob):
                if len(self.orders) < 1:
                        # no orders: return NULL
                        order = None
                else:
                        minprice = lob['bids']['worst']
                        maxprice = lob['asks']['worst']
                        qid = lob['QID']
                        limit = self.orders[0].price
                        otype = self.orders[0].otype
                        if otype == 'Bid':
                                quoteprice = random.randint(minprice, limit)
                        else:
                                quoteprice = random.randint(limit, maxprice)
                                # NB should check it == 'Ask' and barf if not
                        order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, qid)
                        self.lastquote = order
                return order


# Trader subclass Shaver
# shaves a penny off the best price
# if there is no best price, creates "stub quote" at system max/min
class Trader_Shaver(Trader):

        def getorder(self, time, countdown, lob):
                if len(self.orders) < 1:
                        order = None
                else:
                        limitprice = self.orders[0].price
                        otype = self.orders[0].otype
                        if otype == 'Bid':
                                if lob['bids']['n'] > 0:
                                        quoteprice = lob['bids']['best'] + 1
                                        if quoteprice > limitprice :
                                                quoteprice = limitprice
                                else:
                                        quoteprice = lob['bids']['worst']
                        else:
                                if lob['asks']['n'] > 0:
                                        quoteprice = lob['asks']['best'] - 1
                                        if quoteprice < limitprice:
                                                quoteprice = limitprice
                                else:
                                        quoteprice = lob['asks']['worst']
                        order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])
                        self.lastquote = order
                return order


# Trader subclass Sniper
# Based on Shaver,
# "lurks" until time remaining < threshold% of the trading session
# then gets increasing aggressive, increasing "shave thickness" as time runs out
class Trader_Sniper(Trader):

        def getorder(self, time, countdown, lob):
                lurk_threshold = 0.2
                shavegrowthrate = 3
                shave = int(1.0 / (0.01 + countdown / (shavegrowthrate * lurk_threshold)))
                if (len(self.orders) < 1) or (countdown > lurk_threshold):
                        order = None
                else:
                        limitprice = self.orders[0].price
                        otype = self.orders[0].otype

                        if otype == 'Bid':
                                if lob['bids']['n'] > 0:
                                        quoteprice = lob['bids']['best'] + shave
                                        if quoteprice > limitprice :
                                                quoteprice = limitprice
                                else:
                                        quoteprice = lob['bids']['worst']
                        else:
                                if lob['asks']['n'] > 0:
                                        quoteprice = lob['asks']['best'] - shave
                                        if quoteprice < limitprice:
                                                quoteprice = limitprice
                                else:
                                        quoteprice = lob['asks']['worst']
                        order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])
                        self.lastquote = order
                return order




# Trader subclass ZIP
# After Cliff 1997
class Trader_ZIP(Trader):

        # ZIP init key param-values are those used in Cliff's 1997 original HP Labs tech report
        # NB this implementation keeps separate margin values for buying & selling,
        #    so a single trader can both buy AND sell
        #    -- in the original, traders were either buyers OR sellers

        def __init__(self, ttype, tid, balance, time):
                self.ttype = ttype
                self.tid = tid
                self.balance = balance
                self.birthtime = time
                self.profitpertime = 0
                self.n_trades = 0
                self.blotter = []
                self.orders = []
                self.n_quotes = 0
                self.lastquote = None

                ## Matt variables
                self.trade_details = [] # updated using recordTrade

                self.job = None  # this gets switched to 'Bid' or 'Ask' depending on order-type
                self.active = False  # gets switched to True while actively working an order
                self.prev_change = 0  # this was called last_d in Cliff'97
                self.beta = 0.1 + 0.4 * random.random()
                self.momntm = 0.1 * random.random()
                self.ca = 0.05  # self.ca & .cr were hard-coded in '97 but parameterised later
                self.cr = 0.05
                self.margin = None  # this was called profit in Cliff'97
                self.margin_buy = -1.0 * (0.05 + 0.3 * random.random())
                self.margin_sell = 0.05 + 0.3 * random.random()
                self.price = None
                self.limit = None
                # memory of best price & quantity of best bid and ask, on LOB on previous update
                self.prev_best_bid_p = None
                self.prev_best_bid_q = None
                self.prev_best_ask_p = None
                self.prev_best_ask_q = None


        def getorder(self, time, countdown, lob):
                if len(self.orders) < 1:
                        self.active = False
                        order = None
                else:
                        self.active = True
                        self.limit = self.orders[0].price
                        self.job = self.orders[0].otype
                        if self.job == 'Bid':
                                # currently a buyer (working a bid order)
                                self.margin = self.margin_buy
                        else:
                                # currently a seller (working a sell order)
                                self.margin = self.margin_sell
                        quoteprice = int(self.limit * (1 + self.margin))
                        self.price = quoteprice

                        order = Order(self.tid, self.job, quoteprice, self.orders[0].qty, time, lob['QID'])
                        self.lastquote = order
                return order


        # update margin on basis of what happened in market
        def respond(self, time, lob, trade, verbose):
                # ZIP trader responds to market events, altering its margin
                # does this whether it currently has an order to work or not

                def target_up(price):
                        # generate a higher target price by randomly perturbing given price
                        ptrb_abs = self.ca * random.random()  # absolute shift
                        ptrb_rel = price * (1.0 + (self.cr * random.random()))  # relative shift
                        target = int(round(ptrb_rel + ptrb_abs, 0))
# #                        print('TargetUp: %d %d\n' % (price,target))
                        return(target)


                def target_down(price):
                        # generate a lower target price by randomly perturbing given price
                        ptrb_abs = self.ca * random.random()  # absolute shift
                        ptrb_rel = price * (1.0 - (self.cr * random.random()))  # relative shift
                        target = int(round(ptrb_rel - ptrb_abs, 0))
# #                        print('TargetDn: %d %d\n' % (price,target))
                        return(target)


                def willing_to_trade(price):
                        # am I willing to trade at this price?
                        willing = False
                        if self.job == 'Bid' and self.active and self.price >= price:
                                willing = True
                        if self.job == 'Ask' and self.active and self.price <= price:
                                willing = True
                        return willing


                def profit_alter(price):
                        oldprice = self.price
                        diff = price - oldprice
                        change = ((1.0 - self.momntm) * (self.beta * diff)) + (self.momntm * self.prev_change)
                        self.prev_change = change
                        newmargin = ((self.price + change) / self.limit) - 1.0

                        if self.job == 'Bid':
                                if newmargin < 0.0 :
                                        self.margin_buy = newmargin
                                        self.margin = newmargin
                        else :
                                if newmargin > 0.0 :
                                        self.margin_sell = newmargin
                                        self.margin = newmargin

                        # set the price from limit and profit-margin
                        self.price = int(round(self.limit * (1.0 + self.margin), 0))
# #                        print('old=%d diff=%d change=%d price = %d\n' % (oldprice, diff, change, self.price))


                # what, if anything, has happened on the bid LOB?
                bid_improved = False
                bid_hit = False
                lob_best_bid_p = lob['bids']['best']
                lob_best_bid_q = None
                if lob_best_bid_p != None:
                        # non-empty bid LOB
                        lob_best_bid_q = lob['bids']['lob'][-1][1]
                        if self.prev_best_bid_p == None or self.prev_best_bid_p < lob_best_bid_p:
                                # best bid has improved
                                # NB doesn't check if the improvement was by self
                                bid_improved = True
                        elif trade != None and ((self.prev_best_bid_p > lob_best_bid_p) or ((self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                                # previous best bid was hit
                                bid_hit = True
                elif self.prev_best_bid_p != None:
                        # the bid LOB has been emptied: was it cancelled or hit?
                        last_tape_item = lob['tape'][-1]
                        if last_tape_item['type'] == 'Cancel' :
                                bid_hit = False
                        else:
                                bid_hit = True

                # what, if anything, has happened on the ask LOB?
                ask_improved = False
                ask_lifted = False
                lob_best_ask_p = lob['asks']['best']
                lob_best_ask_q = None
                if lob_best_ask_p != None:
                        # non-empty ask LOB
                        lob_best_ask_q = lob['asks']['lob'][0][1]
                        if self.prev_best_ask_p != None and self.prev_best_ask_p > lob_best_ask_p:
                                # best ask has improved -- NB doesn't check if the improvement was by self
                                ask_improved = True
                        elif trade != None and ((self.prev_best_ask_p < lob_best_ask_p) or ((self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                                # trade happened and best ask price has got worse, or stayed same but quantity reduced -- assume previous best ask was lifted
                                ask_lifted = True
                elif self.prev_best_ask_p != None:
                        # the ask LOB is empty now but was not previously: canceled or lifted?
                        last_tape_item = lob['tape'][-1]
                        if last_tape_item['type'] == 'Cancel' :
                                ask_lifted = False
                        else:
                                ask_lifted = True


                if verbose and (bid_improved or bid_hit or ask_improved or ask_lifted):
                        print ('B_improved', bid_improved, 'B_hit', bid_hit, 'A_improved', ask_improved, 'A_lifted', ask_lifted)


                deal = bid_hit or ask_lifted

                if self.job == 'Ask':
                        # seller
                        if deal :
                                tradeprice = trade['price']
                                if self.price <= tradeprice:
                                        # could sell for more? raise margin
                                        target_price = target_up(tradeprice)
                                        profit_alter(target_price)
                                elif ask_lifted and self.active and not willing_to_trade(tradeprice):
                                        # wouldnt have got this deal, still working order, so reduce margin
                                        target_price = target_down(tradeprice)
                                        profit_alter(target_price)
                        else:
                                # no deal: aim for a target price higher than best bid
                                if ask_improved and self.price > lob_best_ask_p:
                                        if lob_best_bid_p != None:
                                                target_price = target_up(lob_best_bid_p)
                                        else:
                                                target_price = lob['asks']['worst']  # stub quote
                                        profit_alter(target_price)

                if self.job == 'Bid':
                        # buyer
                        if deal :
                                tradeprice = trade['price']
                                if self.price >= tradeprice:
                                        # could buy for less? raise margin (i.e. cut the price)
                                        target_price = target_down(tradeprice)
                                        profit_alter(target_price)
                                elif bid_hit and self.active and not willing_to_trade(tradeprice):
                                        # wouldnt have got this deal, still working order, so reduce margin
                                        target_price = target_up(tradeprice)
                                        profit_alter(target_price)
                        else:
                                # no deal: aim for target price lower than best ask
                                if bid_improved and self.price < lob_best_bid_p:
                                        if lob_best_ask_p != None:
                                                target_price = target_down(lob_best_ask_p)
                                        else:
                                                target_price = lob['bids']['worst']  # stub quote
                                        profit_alter(target_price)


                # remember the best LOB data ready for next response
                self.prev_best_bid_p = lob_best_bid_p
                self.prev_best_bid_q = lob_best_bid_q
                self.prev_best_ask_p = lob_best_ask_p
                self.prev_best_ask_q = lob_best_ask_q


######## GDX & AA both taken from snashall2019 code ##############
# Trader subclass AA
class Trader_AA(Trader):

        def __init__(self, ttype, tid, balance, time):
                # Stuff about trader
                self.ttype = ttype
                self.tid = tid
                self.balance = balance
                self.birthtime = time
                self.profitpertime = 0
                self.n_trades = 0
                self.blotter = []
                self.orders = []
                self.n_quotes = 0
                self.lastquote = None

                ## Matt variables
                self.trade_details = [] # updated using recordTrade

                self.limit = None
                self.job = None

                # learning variables
                self.r_shout_change_relative = 0.05
                self.r_shout_change_absolute = 0.05
                self.short_term_learning_rate = random.uniform(0.1, 0.5)
                self.long_term_learning_rate = random.uniform(0.1, 0.5)
                self.moving_average_weight_decay = 0.95 # how fast weight decays with time, lower is quicker, 0.9 in vytelingum
                self.moving_average_window_size = 5
                self.offer_change_rate = 3.0
                self.theta = -2.0
                self.theta_max = 2.0
                self.theta_min = -8.0
                self.marketMax = bse_sys_maxprice

                # Variables to describe the market
                self.previous_transactions = []
                self.moving_average_weights = []
                for i in range(self.moving_average_window_size):
                        self.moving_average_weights.append(self.moving_average_weight_decay**i)
                self.estimated_equilibrium = []
                self.smiths_alpha = []
                self.prev_best_bid_p = None
                self.prev_best_bid_q = None
                self.prev_best_ask_p = None
                self.prev_best_ask_q = None

                # Trading Variables
                self.r_shout = None
                self.buy_target = None
                self.sell_target = None
                self.buy_r = -1.0 * (0.3 * random.random())
                self.sell_r = -1.0 * (0.3 * random.random())



        def calcEq(self):
                # Slightly modified from paper, it is unclear inpaper
                # N previous transactions * weights / N in vytelingum, swap N denominator for sum of weights to be correct?
                if len(self.previous_transactions) == 0:
                        return
                elif len(self.previous_transactions) < self.moving_average_window_size:
                        # Not enough transactions
                        eq = float(sum(self.previous_transactions)) / max(len(self.previous_transactions), 1)
                        self.estimated_equilibrium.append(eq)
                else:
                        N_previous_transactions = self.previous_transactions[-self.moving_average_window_size:]
                        thing = [N_previous_transactions[i]*self.moving_average_weights[i] for i in range(self.moving_average_window_size)]
                        eq = sum( thing ) / sum(self.moving_average_weights)
                        self.estimated_equilibrium.append(eq)

        def calcAlpha(self):
                alpha = 0.0
                for p in self.estimated_equilibrium:
                        alpha += (p - self.estimated_equilibrium[-1])**2
                alpha = math.sqrt(alpha/len(self.estimated_equilibrium))
                self.smiths_alpha.append( alpha/self.estimated_equilibrium[-1])

        def calcTheta(self):
                gamma = 2.0 #not sensitive apparently so choose to be whatever
                # necessary for intialisation, div by 0
                if min(self.smiths_alpha) == max(self.smiths_alpha):
                        alpha_range = 0.4 #starting value i guess
                else:
                        alpha_range = (self.smiths_alpha[-1] - min(self.smiths_alpha)) / (max(self.smiths_alpha) - min(self.smiths_alpha))
                theta_range = self.theta_max - self.theta_min
                desired_theta = self.theta_min + (theta_range) * (1 - (alpha_range * math.exp(gamma * (alpha_range - 1))))
                self.theta = self.theta + self.long_term_learning_rate * (desired_theta - self.theta)

        def calcRshout(self):
                p = self.estimated_equilibrium[-1]
                l = self.limit
                theta = self.theta
                if self.job == 'Bid':
                        # Currently a buyer
                        if l <= p: #extramarginal!
                                self.r_shout = 0.0
                        else: #intramarginal :(
                                if self.buy_target > self.estimated_equilibrium[-1]:
                                        #r[0,1]
                                        self.r_shout = math.log(((self.buy_target - p) * (math.exp(theta) - 1) / (l - p)) + 1) / theta
                                else:
                                        #r[-1,0]
                                        self.r_shout = math.log((1 - (self.buy_target/p)) * (math.exp(theta) - 1) + 1) / theta


                if self.job == 'Ask':
                        # Currently a seller
                        if l >= p: #extramarginal!
                                self.r_shout = 0
                        else: #intramarginal :(
                                if self.sell_target > self.estimated_equilibrium[-1]:
                                        # r[-1,0]
                                        self.r_shout = math.log((self.sell_target - p) * (math.exp(theta) - 1) / (self.marketMax - p) + 1) / theta
                                else:
                                        # r[0,1]
                                        a = (self.sell_target-l)/(p-l)
                                        self.r_shout = (math.log((1 - a) * (math.exp(theta) - 1) + 1)) / theta

        def calcAgg(self):
                delta = 0
                if self.job == 'Bid':
                        # BUYER
                        if self.buy_target >= self.previous_transactions[-1] :
                                # must be more aggressive
                                delta = (1+self.r_shout_change_relative)*self.r_shout + self.r_shout_change_absolute
                        else :
                                delta = (1-self.r_shout_change_relative)*self.r_shout - self.r_shout_change_absolute

                        self.buy_r = self.buy_r + self.short_term_learning_rate * (delta - self.buy_r)

                if self.job == 'Ask':
                        # SELLER
                        if self.sell_target > self.previous_transactions[-1] :
                                delta = (1+self.r_shout_change_relative)*self.r_shout + self.r_shout_change_absolute
                        else :
                                delta = (1-self.r_shout_change_relative)*self.r_shout - self.r_shout_change_absolute

                        self.sell_r = self.sell_r + self.short_term_learning_rate * (delta - self.sell_r)

        def calcTarget(self):
                if len(self.estimated_equilibrium) > 0:
                        p = self.estimated_equilibrium[-1]
                        if self.limit == p:
                                p = p * 1.000001 # to prevent theta_bar = 0
                elif self.job == 'Bid':
                        p = self.limit - self.limit * 0.2  ## Initial guess for eq if no deals yet!!....
                elif self.job == 'Ask':
                        p = self.limit + self.limit * 0.2
                l = self.limit
                theta = self.theta
                if self.job == 'Bid':
                        #BUYER
                        minus_thing = (math.exp(-self.buy_r * theta) - 1) / (math.exp(theta) - 1)
                        plus_thing = (math.exp(self.buy_r * theta) - 1) / (math.exp(theta) - 1)
                        theta_bar = (theta * l - theta * p) / p
                        if theta_bar == 0:
                                theta_bar = 0.0001
                        if math.exp(theta_bar) - 1 == 0:
                                theta_bar = 0.0001
                        bar_thing = (math.exp(-self.buy_r * theta_bar) - 1) / (math.exp(theta_bar) - 1)
                        if l <= p: #Extramarginal
                                if self.buy_r >= 0:
                                        self.buy_target = l
                                else:
                                        self.buy_target = l * (1 - minus_thing)
                        else: #intramarginal
                                if self.buy_r >= 0:
                                        self.buy_target = p + (l-p)*plus_thing
                                else:
                                        self.buy_target = p*(1-bar_thing)
                        if self.buy_target > l:
                                self.buy_target = l

                if self.job == 'Ask':
                        #SELLER
                        minus_thing = (math.exp(-self.sell_r * theta) - 1) / (math.exp(theta) - 1)
                        plus_thing = (math.exp(self.sell_r * theta) - 1) / (math.exp(theta) - 1)
                        theta_bar = (theta * l - theta * p) / p
                        if theta_bar == 0:
                                theta_bar = 0.0001
                        if math.exp(theta_bar) - 1 == 0:
                                theta_bar = 0.0001
                        bar_thing = (math.exp(-self.sell_r * theta_bar) - 1) / (math.exp(theta_bar) - 1) #div 0 sometimes what!?
                        if l <= p: #Extramarginal
                                if self.buy_r >= 0:
                                        self.buy_target = l
                                else:
                                        self.buy_target = l + (self.marketMax - l)*(minus_thing)
                        else: #intramarginal
                                if self.buy_r >= 0:
                                        self.buy_target = l + (p-l)*(1-plus_thing)
                                else:
                                        self.buy_target = p + (self.marketMax - p)*(bar_thing)
                        if self.sell_target == None or self.sell_target < l:
                                self.sell_target = l

        def getorder(self, time, countdown, lob):
                if len(self.orders) < 1:
                        self.active = False
                        return None
                else:
                        self.active = True
                        self.limit = self.orders[0].price
                        self.job = self.orders[0].otype
                        self.calcTarget()

                        if self.prev_best_bid_p == None:
                                o_bid = 0
                        else:
                                o_bid = self.prev_best_bid_p
                        if self.prev_best_ask_p == None:
                                o_ask = self.marketMax
                        else:
                                o_ask = self.prev_best_ask_p

                        if self.job == 'Bid': #BUYER
                                if self.limit <= o_bid:
                                        return None
                                else:
                                        if len(self.previous_transactions) > 0: ## has been at least one transaction
                                                o_ask_plus = (1+self.r_shout_change_relative)*o_ask + self.r_shout_change_absolute
                                                quoteprice = o_bid + ((min(self.limit, o_ask_plus) - o_bid) / self.offer_change_rate)
                                        else:
                                                if o_ask <= self.buy_target:
                                                        quoteprice = o_ask
                                                else:
                                                        quoteprice = o_bid + ((self.buy_target - o_bid) / self.offer_change_rate)
                        if self.job == 'Ask':
                                if self.limit >= o_ask:
                                        return None
                                else:
                                        if len(self.previous_transactions) > 0: ## has been at least one transaction
                                                o_bid_minus = (1-self.r_shout_change_relative) * o_bid - self.r_shout_change_absolute
                                                quoteprice = o_ask - ((o_ask - max(self.limit, o_bid_minus)) / self.offer_change_rate)
                                        else:
                                                if o_bid >= self.sell_target:
                                                        quoteprice = o_bid
                                                else:
                                                        quoteprice = o_ask - ((o_ask - self.sell_target) / self.offer_change_rate)


                        order = Order(self.tid,
                                    self.orders[0].otype,
                                    quoteprice,
                                    self.orders[0].qty,
                                    time, lob['QID'])
                        self.lastquote=order
                return order

        def respond(self, time, lob, trade, verbose):
            ## Begin nicked from ZIP

            # what, if anything, has happened on the bid LOB? Nicked from ZIP..
            bid_improved = False
            bid_hit = False
            lob_best_bid_p = lob['bids']['best']
            lob_best_bid_q = None
            if lob_best_bid_p != None:
                    # non-empty bid LOB
                    lob_best_bid_q = lob['bids']['lob'][-1][1]
                    if self.prev_best_bid_p == None or self.prev_best_bid_p < lob_best_bid_p :
                            # best bid has improved
                            # NB doesn't check if the improvement was by self
                            bid_improved = True
                    elif trade != None and ((self.prev_best_bid_p > lob_best_bid_p) or ((self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                            # previous best bid was hit
                            bid_hit = True
            elif self.prev_best_bid_p != None:
                    # the bid LOB has been emptied: was it cancelled or hit?
                    last_tape_item = lob['tape'][-1]
                    if last_tape_item['type'] == 'Cancel' :
                            bid_hit = False
                    else:
                            bid_hit = True

            # what, if anything, has happened on the ask LOB?
            ask_improved = False
            ask_lifted = False
            lob_best_ask_p = lob['asks']['best']
            lob_best_ask_q = None
            if lob_best_ask_p != None:
                    # non-empty ask LOB
                    lob_best_ask_q = lob['asks']['lob'][0][1]
                    if self.prev_best_ask_p != None and self.prev_best_ask_p > lob_best_ask_p:
                            # best ask has improved -- NB doesn't check if the improvement was by self
                            ask_improved = True
                    elif trade != None and ((self.prev_best_ask_p < lob_best_ask_p) or ((self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                            # trade happened and best ask price has got worse, or stayed same but quantity reduced -- assume previous best ask was lifted
                            ask_lifted = True
            elif self.prev_best_ask_p != None:
                    # the ask LOB is empty now but was not previously: canceled or lifted?
                    last_tape_item = lob['tape'][-1]
                    if last_tape_item['type'] == 'Cancel' :
                            ask_lifted = False
                    else:
                            ask_lifted = True

            self.prev_best_bid_p = lob_best_bid_p
            self.prev_best_bid_q = lob_best_bid_q
            self.prev_best_ask_p = lob_best_ask_p
            self.prev_best_ask_q = lob_best_ask_q

            deal = bid_hit or ask_lifted

            ## End nicked from ZIP

            if deal:
                    self.previous_transactions.append(trade['price'])
                    if self.sell_target == None:
                            self.sell_target = trade['price']
                    if self.buy_target == None:
                            self.buy_target = trade['price']
                    self.calcEq()
                    self.calcAlpha()
                    self.calcTheta()
                    self.calcRshout()
                    self.calcAgg()
                    self.calcTarget()
                #     print("AA p* latest value: " + str(self.estimated_equilibrium))
                #     print("AA alpha latest value: " + str(self.smiths_alpha))
                #     print()
                    #print 'sell: ', self.sell_target, 'buy: ', self.buy_target, 'limit:', self.limit, 'eq: ',  self.estimated_equilibrium[-1], 'sell_r: ', self.sell_r, 'buy_r: ', self.buy_r, '\n'


class Trader_GDX(Trader):

        def __init__(self, ttype, tid, balance, time):
                self.ttype = ttype
                self.tid = tid
                self.balance = balance
                self.birthtime = time
                self.profitpertime = 0
                self.n_trades = 0
                self.blotter = []
                self.orders = []
                self.prev_orders = []
                self.n_quotes = 0
                self.lastquote = None

                ## Matt variables
                self.trade_details = [] # updated using recordTrade

                self.job = None  # this gets switched to 'Bid' or 'Ask' depending on order-type
                self.active = False  # gets switched to True while actively working an order

                #memory of all bids and asks and accepted bids and asks
                self.outstanding_bids = []
                self.outstanding_asks = []
                self.accepted_asks = []
                self.accepted_bids = []

                self.price = -1

                # memory of best price & quantity of best bid and ask, on LOB on previous update
                self.prev_best_bid_p = None
                self.prev_best_bid_q = None
                self.prev_best_ask_p = None
                self.prev_best_ask_q = None

                self.first_turn = True

                self.gamma = 0.1

                self.holdings = 10
                self.remaining_offer_ops = 10
                self.values = [[0 for n in range(self.remaining_offer_ops)] for m in range(self.holdings)]


        def getorder(self, time, countdown, lob):
                if len(self.orders) < 1:
                        self.active = False
                        order = None
                else:
                        self.active = True
                        self.limit = self.orders[0].price
                        self.job = self.orders[0].otype

                        #calculate price
                        if self.job == 'Bid':
                                self.price = self.calc_p_bid(self.holdings - 1, self.remaining_offer_ops - 1)
                        if self.job == 'Ask':
                                self.price = self.calc_p_ask(self.holdings - 1, self.remaining_offer_ops - 1)

                        order = Order(self.tid, self.job, self.price, self.orders[0].qty, time, lob['QID'])
                        self.lastquote = order

                if self.first_turn or self.price == -1:
                        return None
                return order

        def calc_p_bid(self, m, n):
                best_return = 0
                best_bid = 0
                second_best_return = 0
                second_best_bid = 0

                #first step size of 1 get best and 2nd best
                for i in [x*2 for x in range(int(self.limit/2))]:
                        thing = self.belief_buy(i) * ((self.limit - i) + self.gamma*self.values[m-1][n-1]) + (1-self.belief_buy(i) * self.gamma * self.values[m][n-1])
                        if thing > best_return:
                                second_best_bid = best_bid
                                second_best_return = best_return
                                best_return = thing
                                best_bid = i

                #always best bid largest one
                if second_best_bid > best_bid:
                        a = second_best_bid
                        second_best_bid = best_bid
                        best_bid = a

                #then step size 0.05
                for i in [x*0.05 for x in range(int(second_best_bid), int(best_bid))]:
                        thing = self.belief_buy(i + second_best_bid) * ((self.limit - (i + second_best_bid)) + self.gamma*self.values[m-1][n-1]) + (1-self.belief_buy(i + second_best_bid) * self.gamma * self.values[m][n-1])
                        if thing > best_return:
                                best_return = thing
                                best_bid = i + second_best_bid

                return best_bid

        def calc_p_ask(self, m, n):
                best_return = 0
                best_ask = self.limit
                second_best_return = 0
                second_best_ask = self.limit

                #first step size of 1 get best and 2nd best
                for i in [x*2 for x in range(int(self.limit/2))]:
                        j = i + self.limit
                        thing =  self.belief_sell(j) * ((j - self.limit) + self.gamma*self.values[m-1][n-1]) + (1-self.belief_sell(j) * self.gamma * self.values[m][n-1])
                        if thing > best_return:
                                second_best_ask = best_ask
                                second_best_return = best_return
                                best_return = thing
                                best_ask = j
                #always best ask largest one
                if second_best_ask > best_ask:
                        a = second_best_ask
                        second_best_ask = best_ask
                        best_ask = a

                #then step size 0.05
                for i in [x*0.05 for x in range(int(second_best_ask), int(best_ask))]:
                        thing = self.belief_sell(i + second_best_ask) * (((i + second_best_ask) - self.limit) + self.gamma*self.values[m-1][n-1]) + (1-self.belief_sell(i + second_best_ask) * self.gamma * self.values[m][n-1])
                        if thing > best_return:
                                best_return = thing
                                best_ask = i + second_best_ask

                return best_ask

        def belief_sell(self, price):
                accepted_asks_greater = 0
                bids_greater = 0
                unaccepted_asks_lower = 0
                for p in self.accepted_asks:
                        if p >= price:
                                accepted_asks_greater += 1
                for p in [thing[0] for thing in self.outstanding_bids]:
                        if p >= price:
                                bids_greater += 1
                for p in [thing[0] for thing in self.outstanding_asks]:
                        if p <= price:
                                unaccepted_asks_lower += 1

                if accepted_asks_greater + bids_greater + unaccepted_asks_lower == 0:
                        return 0
                return (accepted_asks_greater + bids_greater) / (accepted_asks_greater + bids_greater + unaccepted_asks_lower)

        def belief_buy(self, price):
                accepted_bids_lower = 0
                asks_lower = 0
                unaccepted_bids_greater = 0
                for p in self.accepted_bids:
                        if p <= price:
                                accepted_bids_lower += 1
                for p in [thing[0] for thing in self.outstanding_asks]:
                        if p <= price:
                                asks_lower += 1
                for p in [thing[0] for thing in self.outstanding_bids]:
                        if p >= price:
                                unaccepted_bids_greater += 1
                if accepted_bids_lower + asks_lower + unaccepted_bids_greater == 0:
                        return 0
                return (accepted_bids_lower + asks_lower) / (accepted_bids_lower + asks_lower + unaccepted_bids_greater)

        def respond(self, time, lob, trade, verbose):
                # what, if anything, has happened on the bid LOB?
                self.outstanding_bids = lob['bids']['lob']
                bid_improved = False
                bid_hit = False
                lob_best_bid_p = lob['bids']['best']
                lob_best_bid_q = None
                if lob_best_bid_p != None:
                        # non-empty bid LOB
                        lob_best_bid_q = lob['bids']['lob'][-1][1]
                        if self.prev_best_bid_p == None or self.prev_best_bid_p < lob_best_bid_p:
                                # best bid has improved
                                # NB doesn't check if the improvement was by self
                                bid_improved = True
                        elif trade != None and ((self.prev_best_bid_p > lob_best_bid_p) or ((self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                                # previous best bid was hit
                                self.accepted_bids.append(self.prev_best_bid_p)
                                bid_hit = True
                elif self.prev_best_bid_p != None:
                        # the bid LOB has been emptied: was it cancelled or hit?
                        last_tape_item = lob['tape'][-1]
                        if last_tape_item['type'] == 'Cancel' :
                                bid_hit = False
                        else:
                                bid_hit = True

                # what, if anything, has happened on the ask LOB?
                self.outstanding_asks = lob['asks']['lob']
                ask_improved = False
                ask_lifted = False
                lob_best_ask_p = lob['asks']['best']
                lob_best_ask_q = None
                if lob_best_ask_p != None:
                        # non-empty ask LOB
                        lob_best_ask_q = lob['asks']['lob'][0][1]
                        if self.prev_best_ask_p != None and self.prev_best_ask_p > lob_best_ask_p:
                                # best ask has improved -- NB doesn't check if the improvement was by self
                                ask_improved = True
                        elif trade != None and ((self.prev_best_ask_p < lob_best_ask_p) or ((self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                                # trade happened and best ask price has got worse, or stayed same but quantity reduced -- assume previous best ask was lifted
                                self.accepted_asks.append(self.prev_best_ask_p)
                                ask_lifted = True
                elif self.prev_best_ask_p != None:
                        # the ask LOB is empty now but was not previously: canceled or lifted?
                        last_tape_item = lob['tape'][-1]
                        if last_tape_item['type'] == 'Cancel' :
                                ask_lifted = False
                        else:
                                ask_lifted = True


                #populate expected values
                if self.first_turn:
                        self.first_turn = False
                        for n in range(1, self.remaining_offer_ops):
                                for m in range(1, self.holdings):
                                            if self.job == 'Bid':
                                                    #BUYER
                                                    self.values[m][n] = self.calc_p_bid(m, n)

                                            if self.job == 'Ask':
                                                    #BUYER
                                                    self.values[m][n] = self.calc_p_ask(m, n)


                deal = bid_hit or ask_lifted


                # remember the best LOB data ready for next response
                self.prev_best_bid_p = lob_best_bid_p
                self.prev_best_bid_q = lob_best_bid_q
                self.prev_best_ask_p = lob_best_ask_p
                self.prev_best_ask_q = lob_best_ask_q


class DeepTrader(Trader):

        def __init__(self, ttype, tid, balance, time):
                self.ttype = ttype
                self.tid = tid
                self.balance = balance
                self.birthtime = time
                self.profitpertime = 0
                self.n_trades = 0
                self.blotter = []
                self.orders = []
                self.n_quotes = 0
                self.lastquote = None

                ## Matt variables
                self.trade_details = [] # updated using recordTrade

                ## Model parameters
                self.model_data = []

                self.limit = None
                self.job = None

                ## Variables taken from AA for the calcEq() and calcAlpha() functions

                # Variables to describe the market
                self.moving_average_weight_decay = 0.9
                self.moving_average_window_size = 5

                self.previous_transactions = []
                self.moving_average_weights = []
                for i in range(self.moving_average_window_size):
                        self.moving_average_weights.append(self.moving_average_weight_decay**i)
                self.estimated_equilibrium = [0]
                self.smiths_alpha = [0]
                self.prev_best_bid_p = None
                self.prev_best_bid_q = None
                self.prev_best_ask_p = None
                self.prev_best_ask_q = None

                ## DeepTrader variables

                ## Aaron's model
                # name = "DeepTrader1_5"
                # # load json and create model
                # json_file = open(name + '.json', 'r')
                # loaded_model_json = json_file.read()
                # json_file.close()
                # loaded_model = model_from_json(loaded_model_json)

                # # load weights into new model
                # loaded_model.load_weights(name + ".h5")
                # self.model = loaded_model

                ## My model
                self.model = tf.keras.models.load_model('DeepTraderv3_trader_tuning_noDropout_10epochs_rank.h5')

                ## Values are: time, flag, customer order, spread, midprice, microprice, 
                ## best bid, best ask, time since last trade, LOB imbalance, total num of quotes on LOB, 
                ## p*, Smith's alpha, and output value (trade price)

                ## MIX MAX VALUES FOR combined_filtered_decay_10epochs_TOP40
                # self.min_vals = np.asarray([0.11, 0, 3, -151.0, 5, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0, 14])
                # self.max_vals = np.asarray([600, 2, 224.0, 950, 575.0, 770.0, 221.0, 1000.0, 25, 1.0, 65.0, 206, 13, 221])

                ## MIX MAX VALUES FOR combined_filtered_decay_10epochs_TOP20
                # self.min_vals = np.asarray([0.15, 0, 3, -150.0, 5, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0, 14])
                # self.max_vals = np.asarray([600, 2, 224.0, 950, 575.0, 770.0, 218.0, 1000.0, 30, 1.0, 65.0, 206, 13, 218])

                ## MIX MAX VALUES FOR combined_filtered_decay_10epochs_TOP10
                # self.min_vals = np.asarray([0.2125, 0, 3, -150.0, 5, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0, 14])
                # self.max_vals = np.asarray([600, 2, 224.0, 947, 574.0, 770.0, 218.0, 1000.0, 26.7, 10.0, 65.0, 200, 13, 218])

                ## OLD MIN MAX VALUES FOR DATA V3
                self.min_vals = np.asarray([0.25, 0, 50.0, -150.0, 25, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0, 50])
                self.max_vals = np.asarray([600, 1, 150.0, 950, 575.0, 770.0, 150.0, 1000.0, 30, 1.0, 65.0, 150.0, 0.8724786, 150])
                self.n_features = len(self.min_vals) - 1


        def get_current_market_state(self, lob, time):
                best_bid_q = 0
                best_ask_q = 0
                spread = 0
                midprice = 0
                microprice = 0
                n_quotes = 0
                LOB_imbalance = 0

                bid_or_ask = 0
                if (self.orders[0].otype == "Bid"):
                        bid_or_ask = 1

                cust_order_price = self.orders[0].price

                if (lob != None):
                        best_bid = lob['bids']['best']
                        best_ask = lob['asks']['best']

                        ## Calcuate quantity of best bid/ask for microprice calc
                        num_diff_bids = len(lob['bids']['lob'])
                        num_diff_asks = len(lob['asks']['lob'])
                        
                        # best bid at end of orderbook, best ask at beginning
                        if (num_diff_bids > 0):
                                best_bid_q = lob['bids']['lob'][num_diff_bids-1][1]
                        if (num_diff_asks > 0):
                                best_ask_q = lob['asks']['lob'][0][1]

                        n_quotes = lob['bids']['n'] + lob['asks']['n']

                if (best_bid != None and best_ask != None):
                        spread = best_ask - best_bid
                        midprice = (best_bid + best_ask)/2
                        microprice = ((best_bid * best_ask_q) + (best_ask * best_bid_q))/(best_bid_q + best_ask_q)
                else:
                        best_bid = 0
                        best_ask = 0

                ## Get time of last trade, minus from current time to get delta t (time since last trade)
                tape = lob['tape']
                tape = list(filter(lambda d: d['type'] == "Trade", tape))
                deltat = time
                if (len(tape) > 1):
                        prev_trade_time = tape[len(tape)-2]['time']
                        deltat = tape[len(tape)-1]['time'] - prev_trade_time

                # Calculate LOB imbalance - if n_quotes != 0, otherwise potential /0 error
                if (n_quotes > 0):
                        LOB_imbalance = (lob['bids']['n'] - lob['asks']['n'])/n_quotes

                current_state = np.asarray([time, bid_or_ask, cust_order_price, spread, midprice, microprice, best_bid, best_ask, 
                                           deltat, LOB_imbalance, n_quotes, self.estimated_equilibrium[-1], self.smiths_alpha[-1]])

                normalized_state = (current_state - self.min_vals[:self.n_features]) / (self.max_vals[:self.n_features] - self.min_vals[:self.n_features])

                return normalized_state
                #self.model_data.append(normalized_state)


                

        ## p* and alpha calculations taken from AA traders
        def calcEq(self):
                # Slightly modified from paper, it is unclear inpaper
                # N previous transactions * weights / N in vytelingum, swap N denominator for sum of weights to be correct?
                if len(self.previous_transactions) == 0:
                        return
                elif len(self.previous_transactions) < self.moving_average_window_size:
                        # Not enough transactions
                        eq = float(sum(self.previous_transactions)) / max(len(self.previous_transactions), 1)
                        self.estimated_equilibrium.append(eq)
                else:
                        N_previous_transactions = self.previous_transactions[-self.moving_average_window_size:]
                        thing = [N_previous_transactions[i]*self.moving_average_weights[i] for i in range(self.moving_average_window_size)]
                        eq = sum( thing ) / sum(self.moving_average_weights)
                        self.estimated_equilibrium.append(eq)

        def calcAlpha(self):
                alpha = 0.0
                for p in self.estimated_equilibrium:
                        alpha += (p - self.estimated_equilibrium[-1])**2
                alpha = math.sqrt(alpha/len(self.estimated_equilibrium))
                self.smiths_alpha.append( alpha/self.estimated_equilibrium[-1])

        def getorder (self , time , countdown , lob):
                if len ( self.orders ) < 1:
                        # no orders : return NULL
                        order = None
                else :
                        qid = lob['QID']
                        tape = lob ['tape']
                        otype = self.orders[0].otype
                        limit = self.orders[0].price
                        
                        # Gets the current market state and adds it to end of self.model_data
                        #self.get_current_market_state(lob, time)
                        
                        #model_input = np.asarray(self.model_data)
                        model_input = self.get_current_market_state(lob, time)
                        #model_input[2] = 0
                        model_input = model_input[newaxis, :, newaxis] # normal input

                        if (self.model.layers[0].input_shape[1] == 7):
                                model_input = np.delete(model_input,np.s_[1:7],axis=1) # worst 7
                        elif (self.model.layers[0].input_shape[1] == 6):
                                model_input = model_input[:, 1:7, :] # best 6
                        
                        # retrieving the networks output
                        # normalized_output = self.model.predict(model_input)[0][-1] ## WORKED WELL WITH THIS BUT WRONG??
                        model_input = tf.convert_to_tensor(model_input, dtype=tf.float32)
                        normalized_output = self.model.predict(model_input)[0][0]

                        # denormalizing the output
                        ## Could possibly scale this by multiplying by the range of customer orders for this session?
                        denormalized_output = (( normalized_output ) * (self.max_vals[self.n_features]
                                - self.min_vals[self.n_features])) + self.min_vals[self.n_features]
                        model_price = int(round(denormalized_output , 0))
                        if otype == "Ask":
                                if model_price < limit:
                                        #self.count[1] += 1
                                        model_price = limit
                        else :
                                if model_price > limit:
                                        #self.count[0] += 1
                                        model_price = limit

                        order = Order(self.tid, otype, model_price, self.orders[0].qty, time, qid)
                        self.lastquote = order

                return order
        
        ## Respond taken from AA, removed extra calculations about theta/R-shout etc.
        def respond(self, time, lob, trade, verbose):

            # what, if anything, has happened on the bid LOB? Nicked from ZIP..
            bid_improved = False
            bid_hit = False
            lob_best_bid_p = lob['bids']['best']
            lob_best_bid_q = None
            if lob_best_bid_p != None:
                    # non-empty bid LOB
                    lob_best_bid_q = lob['bids']['lob'][-1][1]
                    if self.prev_best_bid_p == None or self.prev_best_bid_p < lob_best_bid_p:
                            # best bid has improved
                            # NB doesn't check if the improvement was by self
                            bid_improved = True
                    elif trade != None and ((self.prev_best_bid_p > lob_best_bid_p) or ((self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                            # previous best bid was hit
                            bid_hit = True
            elif self.prev_best_bid_p != None:
                    # the bid LOB has been emptied: was it cancelled or hit?
                    last_tape_item = lob['tape'][-1]
                    if last_tape_item['type'] == 'Cancel':
                            bid_hit = False
                    else:
                            bid_hit = True

            # what, if anything, has happened on the ask LOB?
            ask_improved = False
            ask_lifted = False
            lob_best_ask_p = lob['asks']['best']
            lob_best_ask_q = None
            if lob_best_ask_p != None:
                    # non-empty ask LOB
                    lob_best_ask_q = lob['asks']['lob'][0][1]
                    if self.prev_best_ask_p != None and self.prev_best_ask_p > lob_best_ask_p:
                            # best ask has improved -- NB doesn't check if the improvement was by self
                            ask_improved = True
                    elif trade != None and ((self.prev_best_ask_p < lob_best_ask_p) or ((self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                            # trade happened and best ask price has got worse, or stayed same but quantity reduced -- assume previous best ask was lifted
                            ask_lifted = True
            elif self.prev_best_ask_p != None:
                    # the ask LOB is empty now but was not previously: canceled or lifted?
                    last_tape_item = lob['tape'][-1]
                    if last_tape_item['type'] == 'Cancel' :
                            ask_lifted = False
                    else:
                            ask_lifted = True

            self.prev_best_bid_p = lob_best_bid_p
            self.prev_best_bid_q = lob_best_bid_q
            self.prev_best_ask_p = lob_best_ask_p
            self.prev_best_ask_q = lob_best_ask_q

            deal = bid_hit or ask_lifted
            
            ## Record if the original customer order was a bid or ask
            # If the bid is hit, it was an ask (0), if the ask is lifted, it was a bid (1)
            bid_or_ask = 0
            if ask_lifted: 
                    bid_or_ask = 1
        
            ## End nicked from ZIP

            if deal:
                    self.previous_transactions.append(trade['price'])
                    self.calcEq()
                    self.calcAlpha()
                    #self.add_transaction_data(lob, time, bid_or_ask)




##########################---trader-types have all been defined now--################




##########################---Below lies the experiment/test-rig---##################



# trade_stats()
# dump CSV statistics on exchange data and trader population to file for later analysis
# this makes no assumptions about the number of types of traders, or
# the number of traders of any one type -- allows either/both to change
# between successive calls, but that does make it inefficient as it has to
# re-analyse the entire set of traders on each call

## dumpfile output headings: trial id, market session time;
## for n traders: trader name, sum profit, number of this trader, average profit per trader (APPT);
## lob best bid, lob best ask

## Matt dumpfile - details every trade of the best seller & buyer in the market
## time of trade, customer order that triggered trade, price of trade
def trade_stats(expid, traders, dumpfile, dumpfile2, time, lob):
        trader_types = {}
        n_traders = len(traders)

        ## Find buyers and sellers with highest balance
        # highest_buyer_balance = 0
        # highest_seller_balance = 0
        # tcount = 0

        ranked_traders = []
        for t in traders:
                ttype = traders[t].ttype
                if ttype in trader_types.keys():
                        t_balance = trader_types[ttype]['balance_sum'] + traders[t].balance
                        n = trader_types[ttype]['n'] + 1
                else:
                        t_balance = traders[t].balance
                        n = 1

                ranked_traders.append([traders[t], traders[t].balance])

                trader_types[ttype] = {'n':n, 'balance_sum':t_balance}

                # if traders[t].tid[0] == 'B':
                #         if traders[t].balance > highest_buyer_balance:
                #                 best_buyer = traders[t]
                #                 highest_buyer_balance = traders[t].balance
                # elif traders[t].tid[0] == 'S':
                #         if traders[t].balance > highest_seller_balance:
                #                 best_seller = traders[t]
                #                 highest_seller_balance = traders[t].balance
                
                # if ttype == "AA":
                #         if tcount < 2:
                #                 tcount += 1
                #                 print(ttype)
                #                 print(traders[t].previous_transactions)
                #                 print("AA P* & ALPHA")
                #                 print(traders[t].estimated_equilibrium)
                #                 print(traders[t].smiths_alpha)
                #                 print("%d, %d, %d" % (len(traders[t].estimated_equilibrium), len(traders[t].smiths_alpha), len(traders[t].previous_transactions)))
                #                 print()
                
                
        ranked_traders.sort(key=lambda x: x[1], reverse=True)

        ## Calculate p* and alpha at every trade which occured (just for one trader)
        transactions = []
        p_star = []
        alpha = []

        ## Get all trades from market session, store in transactions array
        tape = lob['tape']
        for tapeitem in tape:
                if tapeitem['type'] == 'Trade':
                        transactions.append(tapeitem['price'])


        def calcEqMatt(transactions, p_star):
                # Variables to calculate p*
                moving_average_weight_decay = 0.9 # how fast weight decays with time - 0.9 in vytelingum
                moving_average_window_size = 5
                moving_average_weights = []
                for i in range(moving_average_window_size):
                        moving_average_weights.append(moving_average_weight_decay**i)

                # Slightly modified from paper, it is unclear inpaper
                # N previous transactions * weights / N in vytelingum, swap N denominator for sum of weights to be correct?
                for i in range(1, len(transactions)):
                        if (i < moving_average_window_size):
                                eq = float(sum(transactions[:i])) / max(i, 1)
                                p_star.append(eq)
                        else:
                                N_previous_transactions = transactions[i-moving_average_window_size:]
                                thing = [N_previous_transactions[j]*moving_average_weights[j] for j in range(moving_average_window_size)]
                                eq = sum( thing ) / sum(moving_average_weights)
                                p_star.append(eq)


        def calcAlphaMatt(p_star, alpha):
                for i in range(1, len(transactions)):
                        a = 0.0
                        short_p_star = p_star[:i]
                        for p in short_p_star:
                                a += (p - short_p_star[-1])**2
                        a = math.sqrt(a/len(short_p_star))
                        if (i == 1):
                                alpha.append(0)
                        else:
                                alpha.append( a/short_p_star[-1])

        
        ## Calculate p* and alpha - add 0 to start
        calcEqMatt(transactions, p_star)
        calcAlphaMatt(p_star, alpha)
        p_star.insert(0,0)
        alpha.insert(0,0)

        # print("P*:")
        # print(p_star)
        
        # print("ALPHA:")
        # print(alpha)

        # Previous code that was already here - find appt for each trader

        dumpfile2.write('%s, %06d, ' % (expid, time))
        trader_ranks = []
        for ttype in sorted(list(trader_types.keys())):
                n = trader_types[ttype]['n']
                s = trader_types[ttype]['balance_sum']
                APPT = s / float(n)
                trader_ranks.append([ttype, APPT])
                dumpfile2.write('%s, %d, %d, %f, ' % (ttype, s, n, APPT))
                # if (APPT > highest_APPT):
                #         best_trader = ttype
                #         highest_APPT = APPT

        trader_ranks.sort(key=lambda x: x[1], reverse=True)
        print(trader_ranks)

        for trader in trader_ranks:
                dumpfile2.write('\t , %s, %f' % (trader[0], trader[1]))
        dumpfile2.write('\n')

        # if lob['bids']['best'] != None :
        #         dumpfile2.write('%d, ' % (lob['bids']['best']))
        # else:
        #         dumpfile2.write('N, ')
        # if lob['asks']['best'] != None :
        #         dumpfile2.write('%d, ' % (lob['asks']['best']))
        # else:
        #         dumpfile2.write('N, ')

        ## Get parameters 1-14 for Aaron Wray's training model

        output_array = []
        rank = 0

        for trader in ranked_traders:
                rank = rank + 1
                for i in range(0, len(trader[0].trade_details)):
                        best_bid = trader[0].trade_details[i]['best_bid']
                        best_ask = trader[0].trade_details[i]['best_ask']
                        best_bid_qty = trader[0].trade_details[i]['best_bid_q']
                        best_ask_qty = trader[0].trade_details[i]['best_ask_q']
                        trade_price = trader[0].trade_details[i]['trade_price']

                        if (best_ask == None):
                                best_ask = 0
                        if (best_bid == None):
                                best_bid = 0


                        ## Calculate different metrics used to train neural network
                        spread = 0
                        midprice = 0
                        microprice = 0
                        
                        spread = best_ask - best_bid
                        midprice = (best_bid + best_ask)/2
                        microprice = ((best_bid * best_ask_qty) + (best_ask * best_bid_qty))/(best_bid_qty + best_ask_qty)
                        
                        ## Flag: was the initial order a bid or an ask?
                        bid_or_ask = 0
                        if (trader[0].trade_details[i]['order_type'] == 'Bid'):
                                bid_or_ask = 1

                        # Total number of quotes on exchange
                        n_quotes = trader[0].trade_details[i]['n_bids'] + trader[0].trade_details[i]['n_asks']

                        # Calculate LOB imbalance - if n_quotes != 0, otherwise potential /0 error
                        LOB_imbalance = 0
                        if (n_quotes > 0):
                                LOB_imbalance = (trader[0].trade_details[i]['n_bids'] - trader[0].trade_details[i]['n_asks'])/n_quotes

                        # Calculate time since last trade
                        deltaT = trader[0].trade_details[i]['trade_time'] - trader[0].trade_details[i]['prev_trade_time']
                        

                        ## Write values to results file

                        output_row = [rank, trader[0].ttype, trader[0].trade_details[i]['trade_time'], bid_or_ask, trader[0].trade_details[i]['cust_order'],
                                      spread, midprice, microprice, best_bid, best_ask, deltaT, LOB_imbalance, n_quotes, trade_price]
                        output_array.append(output_row)
                        # # trial id number
                        # dumpfile.write('%s, %s, ' % (expid, trader.ttype))
                        # # trade time, (1 = bid 0 = ask), (1 = hit best 0 = didn't hit best), customer order which triggered the trade
                        # dumpfile.write('%f, %d, %f, ' % (trader.trade_details[i]['trade_time'], 
                        #                           bid_or_ask, trader.trade_details[i]['cust_order']))
                        # # bid-ask spread, midprice, microprice, current best bid, current best ask
                        # dumpfile.write('%f, %f, %f, %f, %f, ' %(spread, midprice, microprice, best_bid, best_ask))
                        # # time since last trade, LOB_imbalance, num_quotes on LOB, 
                        # dumpfile.write('%f, %f, %d, ' %(deltaT, LOB_imbalance, n_quotes))
                        # # p* metric from Vytelingum, Smith's alpha metric, trade price
                        # dumpfile.write('%d, %f, %f' % (trader.trade_details[i]['p_star'], trader.trade_details[i]['alpha'], trade_price))
                        # dumpfile.write('\n')
                        
        # Sort output_array by TIME of trade
        output_array.sort(key=lambda x: x[2])

        i = 0
        count = 0
        ## Remove every second row because all transactions are duplicated
        for row in output_array:
                if count % 2 == 0:
                        if (transactions[i] != row[-1]):
                                print("ERROR: P* AND ALPHA NOT LINED UP CORRECTLY")
                                sys.exit()
                        row.insert(len(row)-1, p_star[i])
                        row.insert(len(row)-1, alpha[i])
                        row_string = str(row)[1:len(str(row))-1]
                        dumpfile.write("%s\n" % row_string)
                        i += 1
                count += 1 
        





# create a bunch of traders from traders_spec
# returns tuple (n_buyers, n_sellers)
# optionally shuffles the pack of buyers and the pack of sellers
def populate_market(traders_spec, traders, shuffle, verbose):

        def trader_type(robottype, name):
                if robottype == 'GVWY':
                        return Trader_Giveaway('GVWY', name, 0.00, 0)
                elif robottype == 'AA':
                        return Trader_AA('AA', name, 0.00, 0)
                elif robottype == 'ZIC':
                        return Trader_ZIC('ZIC', name, 0.00, 0)
                elif robottype == 'GDX':
                        return Trader_GDX('GDX', name, 0.00, 0)
                elif robottype == 'SHVR':
                        return Trader_Shaver('SHVR', name, 0.00, 0)
                elif robottype == 'SNPR':
                        return Trader_Sniper('SNPR', name, 0.00, 0)
                elif robottype == 'ZIP':
                        return Trader_ZIP('ZIP', name, 0.00, 0)
                elif robottype == 'DT':
                        return DeepTrader('DT', name, 0.00, 0)
                elif robottype == 'ADT':
                        return AaronDeepTrader('ADT', name, 0.00, 0, 'DeepTrader1_5')
                else:
                        sys.exit('FATAL: don\'t know robot type %s\n' % robottype)


        def shuffle_traders(ttype_char, n, traders):
                for swap in range(n):
                        t1 = (n - 1) - swap
                        t2 = random.randint(0, t1)
                        t1name = '%c%02d' % (ttype_char, t1)
                        t2name = '%c%02d' % (ttype_char, t2)
                        traders[t1name].tid = t2name
                        traders[t2name].tid = t1name
                        temp = traders[t1name]
                        traders[t1name] = traders[t2name]
                        traders[t2name] = temp


        n_buyers = 0
        for bs in traders_spec['buyers']:
                ttype = bs[0]
                for b in range(bs[1]):
                        tname = 'B%02d' % n_buyers  # buyer i.d. string
                        traders[tname] = trader_type(ttype, tname)
                        n_buyers = n_buyers + 1

        if n_buyers < 1:
                sys.exit('FATAL: no buyers specified\n')

        if shuffle: shuffle_traders('B', n_buyers, traders)


        n_sellers = 0
        for ss in traders_spec['sellers']:
                ttype = ss[0]
                for s in range(ss[1]):
                        tname = 'S%02d' % n_sellers  # buyer i.d. string
                        traders[tname] = trader_type(ttype, tname)
                        n_sellers = n_sellers + 1

        if n_sellers < 1:
                sys.exit('FATAL: no sellers specified\n')

        if shuffle: shuffle_traders('S', n_sellers, traders)

        if verbose :
                for t in range(n_buyers):
                        bname = 'B%02d' % t
                        print(traders[bname])
                for t in range(n_sellers):
                        bname = 'S%02d' % t
                        print(traders[bname])


        return {'n_buyers':n_buyers, 'n_sellers':n_sellers}



# customer_orders(): allocate orders to traders
# parameter "os" is order schedule
# os['timemode'] is either 'periodic', 'drip-fixed', 'drip-jitter', or 'drip-poisson'
# os['interval'] is number of seconds for a full cycle of replenishment
# drip-poisson sequences will be normalised to ensure time of last replenishment <= interval
# parameter "pending" is the list of future orders (if this is empty, generates a new one from os)
# revised "pending" is the returned value
#
# also returns a list of "cancellations": trader-ids for those traders who are now working a new order and hence
# need to kill quotes already on LOB from working previous order
#
#
# if a supply or demand schedule mode is "random" and more than one range is supplied in ranges[],
# then each time a price is generated one of the ranges is chosen equiprobably and
# the price is then generated uniform-randomly from that range
#
# if len(range)==2, interpreted as min and max values on the schedule, specifying linear supply/demand curve
# if len(range)==3, first two vals are min & max, third value should be a function that generates a dynamic price offset
#                   -- the offset value applies equally to the min & max, so gradient of linear sup/dem curve doesn't vary
# if len(range)==4, the third value is function that gives dynamic offset for schedule min,
#                   and fourth is a function giving dynamic offset for schedule max, so gradient of sup/dem linear curve can vary
#
# the interface on this is a bit of a mess... could do with refactoring


def customer_orders(time, last_update, traders, trader_stats, os, pending, verbose):


        def sysmin_check(price):
                if price < bse_sys_minprice:
                        print('WARNING: price < bse_sys_min -- clipped')
                        price = bse_sys_minprice
                return price


        def sysmax_check(price):
                if price > bse_sys_maxprice:
                        print('WARNING: price > bse_sys_max -- clipped')
                        price = bse_sys_maxprice
                return price

        

        def getorderprice(i, sched, n, mode, issuetime):
                # does the first schedule range include optional dynamic offset function(s)?
                if len(sched[0]) > 2:
                        offsetfn = sched[0][2]
                        if callable(offsetfn):
                                # same offset for min and max
                                offset_min = offsetfn(issuetime)
                                offset_max = offset_min
                        else:
                                sys.exit('FAIL: 3rd argument of sched in getorderprice() not callable')
                        if len(sched[0]) > 3:
                                # if second offset function is specfied, that applies only to the max value
                                offsetfn = sched[0][3]
                                if callable(offsetfn):
                                        # this function applies to max
                                        offset_max = offsetfn(issuetime)
                                else:
                                        sys.exit('FAIL: 4th argument of sched in getorderprice() not callable')
                else:
                        offset_min = 0.0
                        offset_max = 0.0

                pmin = sysmin_check(offset_min + min(sched[0][0], sched[0][1]))
                pmax = sysmax_check(offset_max + max(sched[0][0], sched[0][1]))
                prange = pmax - pmin
                stepsize = prange / (n - 1)
                halfstep = round(stepsize / 2.0)

                if mode == 'fixed':
                        orderprice = pmin + int(i * stepsize) 
                elif mode == 'jittered':
                        orderprice = pmin + int(i * stepsize) + random.randint(-halfstep, halfstep)
                elif mode == 'random':
                        if len(sched) > 1:
                                # more than one schedule: choose one equiprobably
                                s = random.randint(0, len(sched) - 1)
                                pmin = sysmin_check(min(sched[s][0], sched[s][1]))
                                pmax = sysmax_check(max(sched[s][0], sched[s][1]))
                        orderprice = random.randint(pmin, pmax)
                else:
                        sys.exit('FAIL: Unknown mode in schedule')
                orderprice = sysmin_check(sysmax_check(orderprice))
                return orderprice



        def getissuetimes(n_traders, mode, interval, shuffle, fittointerval):
                interval = float(interval)
                if n_traders < 1:
                        sys.exit('FAIL: n_traders < 1 in getissuetime()')
                elif n_traders == 1:
                        tstep = interval
                else:
                        tstep = interval / (n_traders - 1)
                arrtime = 0
                issuetimes = []
                for t in range(n_traders):
                        if mode == 'periodic':
                                arrtime = interval
                        elif mode == 'drip-fixed':

                                arrtime = t * tstep
                        elif mode == 'drip-jitter':
                                arrtime = t * tstep + tstep * random.random()
                        elif mode == 'drip-poisson':
                                # poisson requires a bit of extra work
                                interarrivaltime = random.expovariate(n_traders / interval)
                                arrtime += interarrivaltime
                        else:
                                sys.exit('FAIL: unknown time-mode in getissuetimes()')
                        issuetimes.append(arrtime) 
                        
                # at this point, arrtime is the last arrival time
                if fittointerval and ((arrtime > interval) or (arrtime < interval)):
                        # generated sum of interarrival times longer than the interval
                        # squish them back so that last arrival falls at t=interval
                        for t in range(n_traders):
                                issuetimes[t] = interval * (issuetimes[t] / arrtime)
                # optionally randomly shuffle the times
                if shuffle:
                        for t in range(n_traders):
                                i = (n_traders - 1) - t
                                j = random.randint(0, i)
                                tmp = issuetimes[i]
                                issuetimes[i] = issuetimes[j]
                                issuetimes[j] = tmp
                return issuetimes
        

        def getschedmode(time, os):
                got_one = False
                for sched in os:
                        if (sched['from'] <= time) and (time < sched['to']) :
                                # within the timezone for this schedule
                                schedrange = sched['ranges']
                                mode = sched['stepmode']
                                got_one = True
                                exit  # jump out the loop -- so the first matching timezone has priority over any others
                if not got_one:
                        sys.exit('Fail: time=%5.2f not within any timezone in os=%s' % (time, os))
                return (schedrange, mode)
        

        n_buyers = trader_stats['n_buyers']
        n_sellers = trader_stats['n_sellers']

        shuffle_times = True

        cancellations = []

        if len(pending) < 1:
                # list of pending (to-be-issued) customer orders is empty, so generate a new one
                new_pending = []

                # demand side (buyers)
                issuetimes = getissuetimes(n_buyers, os['timemode'], os['interval'], shuffle_times, True)
                
                ordertype = 'Bid'
                (sched, mode) = getschedmode(time, os['dem'])             
                for t in range(n_buyers):
                        issuetime = time + issuetimes[t]
                        tname = 'B%02d' % t
                        orderprice = getorderprice(t, sched, n_buyers, mode, issuetime)
                        order = Order(tname, ordertype, orderprice, 1, issuetime, -3.14)
                        new_pending.append(order)
                        
                # supply side (sellers)
                issuetimes = getissuetimes(n_sellers, os['timemode'], os['interval'], shuffle_times, True)
                ordertype = 'Ask'
                (sched, mode) = getschedmode(time, os['sup'])
                for t in range(n_sellers):
                        issuetime = time + issuetimes[t]
                        tname = 'S%02d' % t
                        orderprice = getorderprice(t, sched, n_sellers, mode, issuetime)
                        order = Order(tname, ordertype, orderprice, 1, issuetime, -3.14)
                        new_pending.append(order)
        else:
                # there are pending future orders: issue any whose timestamp is in the past
                new_pending = []
                for order in pending:
                        if order.time < time:
                                # this order should have been issued by now
                                # issue it to the trader
                                tname = order.tid
                                response = traders[tname].add_order(order, verbose)
                                if verbose: print('Customer order: %s %s' % (response, order) )
                                if response == 'LOB_Cancel' :
                                    cancellations.append(tname)
                                    if verbose: print('Cancellations: %s' % (cancellations))
                                # and then don't add it to new_pending (i.e., delete it)
                        else:
                                # this order stays on the pending list
                                new_pending.append(order)
        return [new_pending, cancellations]



# one session in the market
def market_session(sess_id, starttime, endtime, trader_spec, order_schedule, dumpfile, dumpfile2, dump_each_trade, verbose):


        # initialise the exchange
        exchange = Exchange()


        # create a bunch of traders
        traders = {}
        trader_stats = populate_market(trader_spec, traders, True, verbose)


        # timestep set so that can process all traders in one second
        # NB minimum interarrival time of customer orders may be much less than this!! 
        timestep = 1.0 / float(trader_stats['n_buyers'] + trader_stats['n_sellers'])
        
        duration = float(endtime - starttime)

        last_update = -1.0

        time = starttime

        orders_verbose = False
        lob_verbose = False
        process_verbose = False
        respond_verbose = False
        bookkeep_verbose = False
        lob = None

        pending_cust_orders = []

        if verbose: print('\n%s;  ' % (sess_id))

        while time < endtime:

                # how much time left, as a percentage?
                time_left = (endtime - time) / duration

                # if verbose: print('\n\n%s; t=%08.2f (%4.1f/100) ' % (sess_id, time, time_left*100))

                trade = None

                [pending_cust_orders, kills] = customer_orders(time, last_update, traders, trader_stats,
                                                 order_schedule, pending_cust_orders, orders_verbose)

                # if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them
                if len(kills) > 0 :
                        # if verbose : print('Kills: %s' % (kills))
                        for kill in kills :
                                # if verbose : print('lastquote=%s' % traders[kill].lastquote)
                                if traders[kill].lastquote != None :
                                        # if verbose : print('Killing order %s' % (str(traders[kill].lastquote)))
                                        exchange.del_order(time, traders[kill].lastquote, verbose)


                # get a limit-order quote (or None) from a randomly chosen trader
                tid = list(traders.keys())[random.randint(0, len(traders) - 1)]
                order = traders[tid].getorder(time, time_left, exchange.publish_lob(time, lob_verbose))

                # if verbose: print('Trader Quote: %s' % (order))

                if order != None:
                        #print("Trader = " + traders[tid].ttype + ", Order price = " + str(order.price) + ", Order[0] price = " + str(traders[tid].orders[0]))
                        #print("\n")
                        if order.otype == 'Ask' and order.price < traders[tid].orders[0].price: sys.exit('Bad ask')
                        if order.otype == 'Bid' and order.price > traders[tid].orders[0].price: sys.exit('Bad bid')
                        # send order to exchange
                        traders[tid].n_quotes = 1
                        trade = exchange.process_order2(time, order, process_verbose)
                        if trade != None:
                                # trade occurred,
                                # so the counterparties update order lists and blotters
                                traders[trade['party1']].bookkeep(trade, order, bookkeep_verbose, time)
                                traders[trade['party1']].recordTrade(trade, order, lob)
                                traders[trade['party2']].bookkeep(trade, order, bookkeep_verbose, time)
                                traders[trade['party2']].recordTrade(trade, order, lob)
                                if dump_each_trade: trade_stats(sess_id, traders, dumpfile, dumpfile2, time, exchange.publish_lob(time, lob_verbose))

                        # traders respond to whatever happened
                        lob = exchange.publish_lob(time, lob_verbose)
                        for t in traders:
                                # NB respond just updates trader's internal variables
                                # doesn't alter the LOB, so processing each trader in
                                # sequence (rather than random/shuffle) isn't a problem
                                traders[t].respond(time, lob, trade, respond_verbose)

                time = time + timestep

        # end of an experiment -- dump the tape
        exchange.tape_dump('transactions.csv', 'w', 'keep')


        # write trade_stats for this experiment NB end-of-session summary only
        trade_stats(sess_id, traders, dumpfile, dumpfile2, time, exchange.publish_lob(time, lob_verbose))



#############################

## Recreate Aaron Wray experimental setup

def runExperiment(trader1, trader2, trader3, trader4):
        prop_groups = [[20, 10, 5, 5], [10, 10, 10, 10], [15,10, 10, 5], [15, 15, 5, 5], [25, 5, 5, 5]]
        prop_combinations = []

        print(("Running Experiment with: {0}, {1}, {2}, {3}").format(trader1, trader2, trader3, trader4))

        for group in prop_groups:
                l = list(miter.distinct_permutations(group))
                for item in l:
                        prop_combinations.append(item)
        
        fname = ('dataset_%s_%s_%s_%s.csv' % (trader1 , trader2 , trader3 , trader4))
        tdump = open(fname, 'w')

        fname2 = ('APPT_%s_%s_%s_%s.csv' % (trader1 , trader2 , trader3 , trader4))
        tdump2 = open(fname2, 'w')

        count = 1

        for item in prop_combinations:
                print("combination %d of %d" % (count, len(prop_combinations)))
                buyers_spec = [(trader1, item[0]),(trader2, item[1]),(trader3, item[2]),(trader4, item[3])]
                sellers_spec = buyers_spec
                traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}

                n_trials = 32
                trialnumber = 1
                count = count + 1

                while trialnumber <= n_trials:
                        trial_id = 'trial%04d' % trialnumber
                        market_session(trial_id, start_time, end_time, traders_spec,
                                        order_sched, tdump, tdump2, False, False)
                        tdump.flush()
                        trialnumber = trialnumber + 1

        
def run_balanced_test(trader1, trader2, DTversion):
        
        proportion = [20, 20]

        print(("Running balanced group test with {0}: {1} {2}").format(DTversion, trader1, trader2))
        
        fname = ('Balanced_test_%s_%s_%s_all.csv' % (trader1 , trader2, DTversion))
        tdump = open(fname, 'w')

        fname2 = ('Balanced_APPT_%s_%s_%s_all.csv' % (trader1 , trader2, DTversion))
        tdump2 = open(fname2, 'w')

        num_sessions = 10

        buyers_spec = [(trader1, proportion[0]),(trader2, proportion[1])]
        sellers_spec = buyers_spec
        traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}

        session_number = 1

        while session_number <= num_sessions:
                        trial_id = 'trial%04d' % session_number
                        print(trial_id)
                        market_session(trial_id, start_time, end_time, traders_spec,
                                        order_sched, tdump, tdump2, False, False)
                        tdump.flush()
                        tdump2.flush()
                        session_number = session_number + 1

def run_one_in_many_test(trader1, trader2, DTversion):

        proportion = [1, 39]

        print(("Running one in many test with: {0}, {1}").format(trader1, trader2))
        
        fname = ('One_in_many_%s_%s_%s.csv' % (trader1 , trader2, DTversion))
        tdump = open(fname, 'w')

        fname2 = ('One_in_many_APPT_%s_%s_%s.csv' % (trader1 , trader2, DTversion))
        tdump2 = open(fname2, 'w')

        num_sessions = 60

        buyers_spec = [(trader1, proportion[0]),(trader2, proportion[1])]
        sellers_spec = buyers_spec
        traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}

        session_number = 1

        while session_number <= num_sessions:
                        trial_id = 'trial%04d' % session_number
                        print(trial_id)
                        market_session(trial_id, start_time, end_time, traders_spec,
                                        order_sched, tdump, tdump2, False, False)
                        tdump.flush()
                        tdump2.flush()
                        session_number = session_number + 1

def run_one_in_many_test_tuning(trader1, trader2, condition, w, x, y, z):

        proportion = [1, 39]

        print(("Running balanced test with: {0}, {1} for trader {2}: {3}, {4}, {5}, {6}").format(trader1, trader2, condition, w, x, y, z))
        
        fname = ('Balanced_%s_%s_%s_%d_%d_%d_%d.csv' % (trader1 , trader2, condition, w, x, y, z))
        tdump = open(fname, 'w')

        fname2 = ('Balanced_APPT_%s_%s_%s_%d_%d_%d_%d.csv' % (trader1 , trader2, condition, w, x, y, z))
        tdump2 = open(fname2, 'w')

        num_sessions = 30

        buyers_spec = [(trader1, proportion[0]),(trader2, proportion[1])]
        sellers_spec = buyers_spec
        traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}

        session_number = 1

        while session_number <= num_sessions:
                        trial_id = 'trial%04d' % session_number
                        print(trial_id)
                        market_session(trial_id, start_time, end_time, traders_spec,
                                        order_sched, tdump, tdump2, False, False)
                        tdump.flush()
                        tdump2.flush()
                        session_number = session_number + 1

def feature_experiments(traders, proportion, condition):

        buyers_spec = []
        string = "_{0}".format(condition)

        for trader in traders:
                string = string + "_" + trader
                buyers_spec.append((trader, proportion))

        print(("Running feature experiment, {0} features, {1} each of: {2}").format(condition, str(proportion), string))

        fname = ('Feature_Experiment_Trades%s.csv' % (string))
        tdump = open(fname, 'w')

        fname2 = ('Feature_Experiment_APPT%s.csv' % (string))
        tdump2 = open(fname2, 'w')

        num_sessions = 20
        
        sellers_spec = buyers_spec
        traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}

        session_number = 1

        while session_number <= num_sessions:
                        trial_id = 'trial%04d' % session_number
                        print(trial_id)
                        market_session(trial_id, start_time, end_time, traders_spec,
                                        order_sched, tdump, tdump2, False, False)
                        tdump.flush()
                        session_number = session_number + 1


# # Below here is where we set up and run a series of experiments


if __name__ == "__main__":

        combination_no = int(sys.argv[1])
        # trader_no = int(sys.argv[2])
        # combination_no = 0

        # set up parameters for the session

        start_time = 0.0
        end_time = 600.0
        duration = end_time - start_time


        # schedule_offsetfn returns time-dependent offset on schedule prices
        def schedule_offsetfn(t):
                pi2 = math.pi * 2
                c = math.pi * 3000
                wavelength = t / c
                gradient = 100 * t / (c / pi2)
                amplitude = 100 * t / (c / pi2)
                offset = gradient + amplitude * math.sin(wavelength * t)
                return int(round(offset, 0))
                

# #        range1 = (10, 190, schedule_offsetfn)
# #        range2 = (200,300, schedule_offsetfn)

# #        supply_schedule = [ {'from':start_time, 'to':duration/3, 'ranges':[range1], 'stepmode':'fixed'},
# #                            {'from':duration/3, 'to':2*duration/3, 'ranges':[range2], 'stepmode':'fixed'},
# #                            {'from':2*duration/3, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
# #                          ]
           #demand_schedule = [ {'from':start_time, 'to':end_time, 'ranges':[range2], 'stepmode':'fixed'}
                        #]


        # w = random.randint(75, 100)
        # x = random.randint(w, 125)

        # if (x - w < 5):
        #         w = w + 5

        # y = random.randint(100, 125)
        # z = random.randint(y, 150)

        # if (z - y < 5):
        #         z = z + 5
        trader_list = ["ZIC", "GVWY", "SNPR", "SHVR", "ZIP","GDX", "AA"]

        w = 75
        x = 95
        y = 100
        z = 120

        range1 = (w, x, schedule_offsetfn)
        supply_schedule = [ {'from':start_time, 'to':end_time, 'ranges':[range1], 'stepmode':'random'}]

        range1 = (y, z, schedule_offsetfn)
        demand_schedule = [ {'from':start_time, 'to':end_time, 'ranges':[range1], 'stepmode':'random'}]

        order_sched = {'sup':supply_schedule, 'dem':demand_schedule,
                        'interval':30, 'timemode':'drip-poisson'}

        # run_balanced_test("DT", trader_list[combination_no], "combined_filtered_decay_top20_10epochs_all")

        if (combination_no == 0):
                # run_one_in_many_test("DT", trader_list[2], "version3_nodropout_decay_10epochs")
                for i in range(0,4):
                        run_one_in_many_test("DT", trader_list[i], "version3_no_dropout_decay_10epochs_ranked")
        elif (combination_no == 1):
                # run_one_in_many_test("DT", trader_list[6], "version3_nodropout_decay_10epochs")
                for i in range(4, len(trader_list)):
                        run_one_in_many_test("DT", trader_list[i], "version3_no_dropout_decay_10epochs_ranked")

        
        

        # conditions = [[75, 95, 100, 120], [50, 150, 50, 150], [90, 110, 100, 120], [110, 130, 100, 120], [100, 120, 90, 130], [90, 130, 100, 120], [100, 120, 100, 120]]
        # # conditions = [[50, 150, 50, 150]]
        # # cond = conditions[combination_no]
        # for cond in conditions:

        #         w = cond[0]
        #         x = cond[1]
        #         y = cond[2]
        #         z = cond[3]

        #         range1 = (w, x, schedule_offsetfn)
        #         supply_schedule = [ {'from':start_time, 'to':end_time, 'ranges':[range1], 'stepmode':'random'}]

        #         range1 = (y, z, schedule_offsetfn)
        #         demand_schedule = [ {'from':start_time, 'to':end_time, 'ranges':[range1], 'stepmode':'random'}]

        #         ## Original range & schedules

        #         # range1 = (50, 150, schedule_offsetfn)

        #         # supply_schedule = [ {'from':start_time, 'to':end_time, 'ranges':[range1], 'stepmode':'random'}]
        #         # demand_schedule = supply_schedule

        #         # order schedule can be have timemodes: periodic, drip-fixed, drip-jitter, drip-poisson
        #         order_sched = {'sup':supply_schedule, 'dem':demand_schedule,
        #                 'interval':30, 'timemode':'drip-poisson'}

        #         # combinations = ["AA GDX ZIC ZIP", "AA GDX ZIC SNPR", "AA GDX ZIC GVWY", "AA GDX ZIC SHVR",
        #         #                 "AA GDX ZIP SNPR", "AA GDX ZIP GVWY", "AA GDX ZIP SHVR", "AA GDX SNPR GVWY",
        #         #                 "AA GDX SNPR SHVR", "AA GDX GVWY SHVR", "AA ZIC ZIP SNPR", "AA ZIC ZIP GVWY",
        #         #                 "AA ZIC ZIP SHVR", "AA ZIC SNPR GVWY", "AA ZIC SNPR SHVR", "AA ZIC GVWY SHVR",
        #         #                 "AA ZIP SNPR GVWY", "AA ZIP SNPR SHVR", "AA ZIP GVWY SHVR", "AA SNPR GVWY SHVR",
        #         #                 "GDX ZIC ZIP SNPR", "GDX ZIC ZIP GVWY", "GDX ZIC ZIP SHVR", "GDX ZIC SNPR GVWY",
        #         #                 "GDX ZIC SNPR SHVR", "GDX ZIC GVWY SHVR", "GDX ZIP SNPR GVWY", "GDX ZIP SNPR SHVR",
        #         #                 "GDX ZIP GVWY SHVR", "GDX SNPR GVWY SHVR", "ZIC ZIP SNPR GVWY", "ZIC ZIP SNPR SHVR",
        #         #                 "ZIC ZIP GVWY SHVR", "ZIC SNPR GVWY SHVR", "ZIP SNPR GVWY SHVR"]

        #         # assert(len(combinations) == 35)

        #         #combinations = combinations_string.split()
                
        #         # chosen_combination = combinations[combination_no].split()
        #         # runExperiment(chosen_combination[0], chosen_combination[1], chosen_combination[2], chosen_combination[3])
        #         #runExperiment("DT", "ZIC", "GVWY", "SNPR")

                
        #         #run_one_in_many_test("DT", trader_list[4])
        #         # run_one_in_many_test("DT", trader_list[3])
        #         # run_balanced_test("DT", trader_list[combination_no])


        #         # run a sequence of trials, one session per trial

        #         # traders = ["ADT", "ZIC", "GVWY", "SNPR", "SHVR", "ZIP","GDX", "AA"]
        #         # combinations = ["ALL", "BOTTOM7", "TOP6"]

        #         # feature_experiments(traders, 5, "ALL_AARON")

        #         # for i in range(1,11):
        #         #         string = "ALL" + str(i)
        #         #         feature_experiments(traders, 5, string)

        #         # for i in range(0, len(trader_list)):
        #         #         run_one_in_many_test("DT", trader_list[i])

        #         # run_one_in_many_test_tuning("DT", trader_list[trader_no], "combined_filtered_top20_decay_10epochs_all", w, x, y, z)

        #         if (combination_no == 0):
        #                 # run_one_in_many_test_tuning("DT", trader_list[5], "combined_filtered_top20_decay_10epochs_worst7", w, x, y, z)
        #                 for i in range(0,2):
        #                         run_one_in_many_test_tuning("DT", trader_list[i], "combined_filtered_top20_decay_10epochs_all", w, x, y, z)
        #         elif (combination_no == 1):
        #                 run_one_in_many_test_tuning("DT", trader_list[2], "combined_filtered_top20_decay_10epochs_all", w, x, y, z)
        #                 # for i in range(5, len(trader_list)):
        #                 #         run_one_in_many_test_tuning("DT", trader_list[i], "combined_filtered_top20_decay_10epochs_all", w, x, y, z)
 

        # n_trials = 1
        # tdump=open('avg_balance.csv','w')
        # trial = 1
        # if n_trials > 1:
        #         dump_all = False
        # else:
        #         dump_all = True

        #         market_session(trial_id, start_time, end_time, traders_spec,
        # #                                                        order_sched, tdump, False, True)
                
        # while (trial<(n_trials+1)):
        #         trial_id = 'trial%04d' % trial
        #         market_session(trial_id, start_time, end_time, traders_spec, order_sched, tdump, dump_all)
        #         tdump.flush()
        #         trial = trial + 1
        # tdump.close()

        # sys.exit('Done Now')


        

        # run a sequence of trials that exhaustively varies the ratio of four trader types
        # NB this has weakness of symmetric proportions on buyers/sellers -- combinatorics of varying that are quite nasty
        

        # n_trader_types = 4
        # equal_ratio_n = 4
        # n_trials_per_ratio = 50

        # n_traders = n_trader_types * equal_ratio_n

        # fname = 'balances_%03d.csv' % equal_ratio_n

        # tdump = open(fname, 'w')

        # min_n = 1

        # trialnumber = 1
        # trdr_1_n = min_n
        # while trdr_1_n <= n_traders:
        #         trdr_2_n = min_n 
        #         while trdr_2_n <= n_traders - trdr_1_n:
        #                 trdr_3_n = min_n
        #                 while trdr_3_n <= n_traders - (trdr_1_n + trdr_2_n):
        #                         trdr_4_n = n_traders - (trdr_1_n + trdr_2_n + trdr_3_n)
        #                         if trdr_4_n >= min_n:
        #                                 buyers_spec = [('GVWY', trdr_1_n), ('SHVR', trdr_2_n),
        #                                                ('ZIC', trdr_3_n), ('ZIP', trdr_4_n)]
        #                                 sellers_spec = buyers_spec
        #                                 traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}
        #                                 # print buyers_spec
        #                                 trial = 1
        #                                 while trial <= n_trials_per_ratio:
        #                                         trial_id = 'trial%07d' % trialnumber
        #                                         market_session(trial_id, start_time, end_time, traders_spec,
        #                                                        order_sched, tdump, False, True)
        #                                         tdump.flush()
        #                                         trial = trial + 1
        #                                         trialnumber = trialnumber + 1
        #                         trdr_3_n += 1
        #                 trdr_2_n += 1
        #         trdr_1_n += 1
        # tdump.close()
        
        # print trialnumber


