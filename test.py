'''Check if a python process with a given parameter string is running'''
param_string = '123'

import psutil
for process in psutil.process_iter():    
    if process.name()=='python.exe' and param_string in process.cmdline():

            print(process.cmdline())

