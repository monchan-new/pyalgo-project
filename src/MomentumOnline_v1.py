#
# Python Script
# with Online Trading Algorithm
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import zmq
import datetime
import numpy as np
import pandas as pd

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect('tcp://0.0.0.0:5555')
socket.setsockopt_string(zmq.SUBSCRIBE, 'SYMBOL')

df = pd.DataFrame()
mom = 3
min_length = mom + 2 # excludes undefiend bar & start action after getting 3 end of bar values

while True:
    data = socket.recv_string()
    t = datetime.datetime.now()
    sym, value = data.split()
    df = pd.concat([df, pd.DataFrame({sym: float(value)}, index=[t])])

    dr = df.resample('5s', label='right').last() # new bar & undefined next bar created when enters new time-section (and modified by each tick during the same time-section)
    dr['returns'] = np.log(dr / dr.shift(1))

    dr['momentum'] = np.sign(dr['returns'].rolling(mom).mean()) # calculate by every tick

    if len(dr) > min_length: # excludes next-prepared undefined dr-bar
        min_length += 1 # to take action when only a new bar appeares
        
        print('\n' + '=' * 51)
        print(f'NEW SIGNAL | {datetime.datetime.now()}')
        print('=' * 51)
        print(dr.iloc[:-2].tail()) # automatically exludes undefined next dr-bar + refer to one last bar to see end of bar value

        if dr['momentum'].iloc[-3] == 1.0: # manually exludes undefined next dr-bar + refer to one last bar to see end of bar value
            print('\nLong market position.')
            # take some action (e.g., place buy order)
        elif dr['momentum'].iloc[-3] == -1.0:
            print('\nShort market position.')
            # take some action (e.g., place sell order)