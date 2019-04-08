# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:10:51 2019

@author: Alex
"""

#%% Markove Chain Transition Matrix

import numpy as np

def weather_forecast(days, init):
    weatherToday = init
    weatherList = [weatherToday]
    k = 0
    prob = 1.0
    while k < days:
        if weatherToday == "0":
            event = np.random.choice(possibleEvents[0], replace=True, p=transitionMatrix[0])
            if event == "00":
                prob *= transitionMatrix[0][0]
            elif event == "01":
                prob *= transitionMatrix[0][1]
                weatherToday = "1"
            elif event == "02":
                prob *= transitionMatrix[0][2]
                weatherToday = "2"
            else:
                prob *= transitionMatrix[0][3]
                weatherToday = "3"
            weatherList.append(weatherToday)
        elif weatherToday == "1":
            event = np.random.choice(possibleEvents[1], replace=True, p=transitionMatrix[1])
            if event == "11":
                prob *= transitionMatrix[1][1]
            elif event == "10":
                prob *= transitionMatrix[1][0]
                weatherToday = "0"
            elif event == "12":
                prob *= transitionMatrix[1][2]
                weatherToday = "2"
            else:
                prob *= transitionMatrix[1][3]
                weatherToday = "3"
            weatherList.append(weatherToday)
        elif weatherToday == "2":
            event = np.random.choice(possibleEvents[2], replace=True, p=transitionMatrix[2])
            if event == "22":
                prob *= transitionMatrix[2][2]
            elif event == "20":
                prob *= transitionMatrix[2][0]
                weatherToday = "0"
            elif event == "21":
                prob *= transitionMatrix[2][1]
                weatherToday = "1"
            else:
                prob *= transitionMatrix[2][3]
                weatherToday = "3"
            weatherList.append(weatherToday)
        elif weatherToday == "3":
            event = np.random.choice(possibleEvents[3], replace=True, p=transitionMatrix[3])
            if event == "33":
                prob *= transitionMatrix[3][3]
            elif event == "30":
                prob *= transitionMatrix[3][0]
                weatherToday = "0"
            elif event == "31":
                prob *= transitionMatrix[3][1]
                weatherToday = "1"
            else:
                prob *= transitionMatrix[3][2]
                weatherToday = "2"
            weatherList.append(weatherToday)

        k += 1

    return [weatherList, prob]


# The state space
states = ["0", "1", "2", "3"]

# Possible sequence of events
possibleEvents = [["00", "01", "02", "03"], ["10", "11", "12", "13"],
                  ["20", "21", "22", "23"], ["30", "31", "32", "33"]]

# Transition probability matrix
transitionMatrix = [[0.7, 0.0, 0.3, 0.0], [0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.4, 0.0, 0.6], [0.0, 0.2, 0.0, 0.8]]

days = 2
init = states[0]
#init = "3"
[x, y] = weather_forecast(days, init)

print("Start state: %s" % x[0])
print("Possible states: %s" % str(x))
print("Probability of the possible sequence of states: %s" % str(y))

#%% Random Walk

import random
import matplotlib.pylab as plt
import numpy as np


def random_walk(n):
	x = 0
	c = [x]
	for i in range(n):
		prob = random.random()
		if prob > 0.5:
			x = x + 1
		elif prob < 0.5:
			x = x - 1
		c.append(x)
	return c


numberOfSteps = 100
walk = random_walk(numberOfSteps)

n = np.arange(numberOfSteps+1)

plt.plot(n, walk, 'r-')
plt.plot(n, walk, 'bo')
plt.xlabel('Time (n)')
plt.ylabel('Position (x)')
plt.show()
