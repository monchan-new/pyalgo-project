#
# Python Script to Simulate a
# Financial Tick Data Server
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import zmq
import math
import time
import random

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind('tcp://0.0.0.0:5555')


class InstrumentPrice:
    def __init__(self):
        self.symbol = 'SYMBOL'
        self.t = time.time()
        self.value = 100.
        self.sigma = 0.4
        self.r = 0.01

    def simulate_value(self):
        ''' Generates a new, random stock price.
        '''
        t = time.time()

        raw_dt = t - self.t
        # --- 時刻ジャンプ対策（最重要） ---
        if raw_dt < 0:
            raw_dt = 0
        if raw_dt > 1:  # 1秒以上飛んだら異常値として制限
            raw_dt = 1
        dt = (raw_dt) / (252 * 8 * 60 * 60)

        dt *= 500
        self.t = t
        self.value *= math.exp((self.r - 0.5 * self.sigma ** 2) * dt +
                               self.sigma * math.sqrt(dt) * random.gauss(0, 1))
        return self.value
    
ip = InstrumentPrice()

while True:
    msg = '{} {:.2f}'.format(ip.symbol, ip. simulate_value())
    print(msg)
    socket.send_string(msg)
    time.sleep(random.random() * 2)