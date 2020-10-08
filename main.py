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

# read in the csv data into a pandas data frame and set the date as the index
from tensorflow.contrib.distributions.python.ops.bijectors import inline

df = pd.read_csv('weather.csv', sep=';').set_index('nomer')

df = df.drop(['daily_summary'], axis=1)

# execute the describe() function and transpose the output so that it doesn't overflow the width of the screen
df.describe().T
# information about dataset
df.info()

# X will be a pandas dataframe of all columns except meantempm
X = df[
    ['date', 'month', 'day', 'time', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_bearing', 'visibility']]

# y will be a pandas series of the meantempm
y = df['apparent_temperature']

# split data into training set and a temporary set using sklearn.model_selection.traing_test_split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)

# take the remaining 20% of data in X_tmp, y_tmp and split them evenly
X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23) # X_val


X_train.shape, X_test.shape, X_val.shape
print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))
print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))
print("Testing instances    {}, Testing features    {}".format(X_test.shape[0], X_test.shape[1]))

feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]

global_step = tf.Variable(0)
tf.train.global_step = tf.Variable(0)

# parametrs of neural network
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[50, 50],
                                      model_dir='tf_wx_model')


def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=400):
    return tf.estimator.inputs.pandas_input_fn(x=X,
                                               y=y,
                                               num_epochs=num_epochs,
                                               shuffle=shuffle,
                                               batch_size=batch_size)


# network is learning
evaluations = []
STEPS = 100
for i in range(2):
    # tf.train.global_step()
    regressor.train(input_fn=wx_input_fn(X_train, y=y_train), steps=STEPS)
    evaluation = regressor.evaluate(input_fn=wx_input_fn(X_val, y_val,
                                                         num_epochs=1,
                                                         shuffle=False),
                                    steps=1)
    evaluations.append(regressor.evaluate(input_fn=wx_input_fn(X_val,
                                                               y_val,
                                                               num_epochs=1,
                                                               shuffle=False)))
    print('step number', i)

exitFlag=True
#while(exitFlag):

    # prepearing for test demonstration
print('choose number of rows')
numR = int(input())
testm = range(96453-numR, 96453)
x_value = df.loc[df.index.isin(testm)]
y_value=df.loc[df.index.isin(testm)]
x_value = x_value.drop(['apparent_temperature'], axis=1)
y_value = y_value.drop(['date', 'month', 'day', 'time', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_bearing', 'visibility'],axis=1)
x_value1 = pd.DataFrame(x_value)
y3_value = pd.DataFrame(x_value)
# input variables
print('input date, month, day, time, temperature, pressure, humidity, wind speed, wind bearing, visibility')
date=int(input())
month=int(input())
day=int(input())
time=int(input())
temperature=float(input())
pressure=float(input())
humidity=float(input())
wind_speed=float(input())
wind_bearing=float(input())
visibility=float(input())

y3_value = y3_value.drop(['date', 'month', 'day', 'time', 'pressure', 'humidity', 'wind_speed', 'wind_bearing', 'visibility'], axis=1)
    # add a new row, without answer
x_value1=pd.DataFrame([[date, month, day, time, temperature, pressure, humidity, wind_speed, wind_bearing, visibility]], columns=['date', 'month', 'day', 'time', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_bearing', 'visibility'])
x_value=x_value.append(x_value1)
y3_value1 = pd.DataFrame(
        [ temperature],
        columns=[ 'temperature'])
y3_value = y3_value.append( y3_value1)
    # print(x_value)
    # predict x_value1
pred1 = regressor.predict(input_fn=wx_input_fn(x_value,
                                                  num_epochs=1,
                                                  shuffle=False))
    # print(pred1)
predictions1 = np.array([p['predictions'][0] for p in pred1])
x1_array = np.array(range(numR))
y1_array = np.array(y_value)
x2_array = np.array(range(numR+1))
y2_array = predictions1
x3_array = np.array(range(numR+1))
y3_array = np.array(y3_value)
mapp = np.array(range(numR+1))

for i in range (numR+1):
    mapp[i-1]=(y2_array[i-1]-y3_array[i-1])*100



    # print(predictions1)
    # print ('y1=', y1_array, 'y2= ', y2_array)
    # plots settings
    # plt.plot(x=x1_array, y=y1_array)
    # plt.xlabel('time')
    # plt.ylabel('apparent temperature')
    # plt.show()

fig=plt.figure()
# real apperent
ax1=fig.add_subplot(111)
ax1.plot(x1_array, y1_array, color='green')
ax1.set_xlabel('steps')
ax1.set_ylabel('apparent temperature')
ax1.grid(True, color='grey')
pylab.ylim(0, 50)

# real
ax3=ax1.twinx()
ax3.plot(x3_array, y3_array, color='black')
ax3.set_title(u'Graphs of predicted weather')
pylab.ylim(0, 50)

# predict
ax2=ax1.twinx()
sctr = ax2.scatter(x=x2_array, y=y2_array, c=mapp, cmap='seismic')
plt.colorbar(sctr, ax=ax2, format='%d per cent')
# x2.plot(x2_array, y2_array, color='red')
ax2.set_title(u'Graphs of predicted weather')
pylab.ylim(0, 50)
plt.legend('apparent temperature', 'real temperature', 'predicted apparent temperature',loc='upper left')
plt.show()

for i in range(numR):
    root = Tk()
    root.title("Температурные условия")
    root.geometry("400x300")
    text = "real temperature = " + str(y3_array[i]) + "\n apparent temperature = " + str(y2_array[i])
    label = Label(text=text, fg='black')
    label.pack()
    #root.mainloop()
    if (y2_array[i]>y3_array[i]):
        root.configure(background='red')
    if (y2_array[i] == y3_array[i]):
        root.configure(background='green')
    if (y2_array[i] < y3_array[i]):
        root.configure(background='blue')
    plt.show()
    time.sleep(5)

# 2016 9 10 5 19 1017.68 0.75 10.09842 60 15.57

    #print('Do you want to exit? (True or False')
    #exitFlag=bool(input())

# %matplotlib inline

# manually set the parameters of the figure to and appropriate size
plt.rcParams['figure.figsize'] = [14, 10]

loss_values = [ev['loss'] for ev in evaluations]
training_steps = [ev['global_step'] for ev in evaluations]

plt.scatter(x=training_steps, y=loss_values)
plt.xlabel('Training steps (Epochs = steps / 2)')
plt.ylabel('Loss (SSE)')
plt.show()

pred = regressor.predict(input_fn=wx_input_fn(X_test,
                                              num_epochs=1,
                                              shuffle=False))
predictions = np.array([p['predictions'][0] for p in pred])
print(predictions)

print("The Explained Variance: %.2f" % explained_variance_score(
                                            y_test, predictions))
print("The Mean Absolute Error: %.2f degrees Celcius" % mean_absolute_error(
                                            y_test, predictions))
print("The Median Absolute Error: %.2f degrees Celcius" % median_absolute_error(
                                            y_test, predictions))
print("Predictions = ", predictions)


