import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, \
    median_absolute_error
from sklearn.model_selection import train_test_split
from tkinter import *
import time

import random


print("input number")
numR = int(input())
x1_array = np.array(range(numR))
x2_array = np.array(range(numR+1))
x3_array = np.array(range(numR+1))
y1_array = np.array([random.random()*50 for i in range(numR)])
y2_array = np.array([random.random()*50 for i in range(numR+1)])
y3_array = np.array([random.random()*50 for i in range(numR+1)])

fig=plt.figure()
# real apperent
ax1=fig.add_subplot(111)
ax1.plot(x1_array, y1_array, color='grey')
ax1.set_xlabel('steps')
ax1.set_ylabel('apparent temperature')
ax1.grid(True, color='grey')
pylab.ylim(0, 50)

# predict
ax2=ax1.twinx()
ax2.plot(x2_array, y2_array, color='red')
ax2.set_title(u'Graphs of predicted weather')
pylab.ylim(0, 50)

# real
ax3=ax1.twinx()
ax3.plot(x3_array, y3_array, color='black')
ax3.set_title(u'Graphs of predicted weather')
pylab.ylim(0, 50)

# fig.stackplot(x3_array, y1_array + y2_array + y3_array, labels=['apparent', 'predict', 'real'])
# fig.legend(loc='upper right')

# plt.show()
import sys
from tkinter import *
import time

root = Tk()

statusbar = Frame(root)
statusbar.pack(side="bottom", fill="x", expand=False)

time1 = ''
clock = Label(root, font=('times', 20, 'bold'), bg='green')

def tick():
    global time1
    # get the current local time from the PC
    time2 = time.strftime('%H:%M:%S')
    # if time string has changed, update it
    if time2 != time1:
        time1 = time2
        clock.config(text=time2)
        # calls itself every 200 milliseconds
        # to update the time display as needed
        # could use >200 ms, but display gets jerky
    clock.after(200, tick)

tick()

status = Label(root, text="v1.0", bd=1, relief=SUNKEN, anchor=W)
status.pack(in_=statusbar, side=LEFT, fill=BOTH, expand=True)
clock.pack(in_=statusbar, side=RIGHT, fill=Y, expand=False)

root.mainloop(  )