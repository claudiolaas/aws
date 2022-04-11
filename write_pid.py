#%%
import os
import argparse
from time import sleep
parser = argparse.ArgumentParser()
print("This process has the PID", os.getpid())

parser.add_argument('--pid')
args = parser.parse_args()

while True:
    print(args)
    sleep(60)
# config = vars(args)
# print(config)

# import json

# with open('running_bots.json', 'r') as data_file:
#     data = json.load(data_file)
#     print(data)

# data.pop('123', None)

# with open('running_bots.json', 'w') as data_file:
#     data = json.dump(data, data_file)

#%%

# Sending signal 0 to a pid will raise an OSError exception if the pid is not running, and do nothing otherwise.

# import os

# def check_pid(pid):        
#     """ Check For the existence of a unix pid. """
#     try:
#         os.kill(pid, 0)
#     except OSError:
#         return False
#     else:
#         return True


# open json

# loop over entries

# check if pid of params are running

# if true, 