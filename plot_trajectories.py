import numpy as np
from matplotlib import pyplot as plt
from ast import literal_eval
import math
from scipy.signal import lfilter


ROUTE = 'roundabout_route'
QUALITY = 'low'
WEATHER = 'sunny'

n = 20  
b = [1.0 / n] * n
a = 1

outF1 = open("%s/%s/real_coordinates_%s_q.txt"  %(ROUTE, WEATHER, QUALITY), "r")

outF2 = open("%s/%s/calculated_coordinates_%s_q_surf.txt"  %(ROUTE, WEATHER, QUALITY), "r")
outF3 = open("%s/%s/calculated_coordinates_%s_q_sift.txt"  %(ROUTE, WEATHER, QUALITY), "r")
outF4 = open("%s/%s/calculated_coordinates_%s_q_fast.txt"  %(ROUTE, WEATHER, QUALITY), "r")
outF5 = open("%s/%s/calculated_coordinates_%s_q_orb.txt"  %(ROUTE, WEATHER, QUALITY), "r")

real_cors = np.array(literal_eval(outF1.read()))
x = [c1[0] for c1 in real_cors]
y = [c2[1] for c2 in real_cors]
plt.plot(x, y)

calc_cors_surf = np.array(literal_eval(outF2.read()))
x = [c1[0] for c1 in calc_cors_surf]
y = [c2[1] for c2 in calc_cors_surf]
x = lfilter(b,a,x)
y = lfilter(b,a,y)
plt.plot(x[19:], y[19:])

calc_cors_sift = np.array(literal_eval(outF3.read()))
x = [c1[0] for c1 in calc_cors_sift]
y = [c2[1] for c2 in calc_cors_sift]
x = lfilter(b,a,x)
y = lfilter(b,a,y)
plt.plot(x[19:], y[19:])

calc_cors_fast = np.array(literal_eval(outF4.read()))
x = [c1[0] for c1 in calc_cors_fast]
y = [c2[1] for c2 in calc_cors_fast]
x = lfilter(b,a,x)
y = lfilter(b,a,y)
plt.plot(x[19:], y[19:])

calc_cors_orb = np.array(literal_eval(outF5.read()))
x = [c1[0] for c1 in calc_cors_orb]
y = [c2[1] for c2 in calc_cors_orb]
x = lfilter(b,a,x)
y = lfilter(b,a,y)
plt.plot(x[19:], y[19:])


plt.legend(['ground truth', 'surf', 'sift', 'fast', 'orb'])
plt.show()

E = []
for i in range(0, len(calc_cors_surf)):
    x_real = real_cors[i][0]
    y_real = real_cors[i][1]
    x_calc = calc_cors_surf[i][0]
    y_calc = calc_cors_surf[i][1]
    e  = 0.05 * math.sqrt((x_real - x_calc)**2 + (y_real - y_calc)**2)
    E.append(e)

E = lfilter(b,a,E)
arg_e = np.linspace(0, calc_cors_surf[-1][0], len(calc_cors_surf))
plt.plot(arg_e, E)

E = []
for i in range(0, len(calc_cors_sift)):
    x_real = real_cors[i][0]
    y_real = real_cors[i][1]
    x_calc = calc_cors_sift[i][0]
    y_calc = calc_cors_sift[i][1]
    e  = 0.05 * math.sqrt((x_real - x_calc)**2 + (y_real - y_calc)**2)
    E.append(e)

E = lfilter(b,a,E)
arg_e = np.linspace(0, calc_cors_sift[-1][0], len(calc_cors_sift))
plt.plot(arg_e, E)

E = []
for i in range(0, len(calc_cors_fast)):
    x_real = real_cors[i][0]
    y_real = real_cors[i][1]
    x_calc = calc_cors_fast[i][0]
    y_calc = calc_cors_fast[i][1]
    e  = 0.05 * math.sqrt((x_real - x_calc)**2 + (y_real - y_calc)**2)
    E.append(e)

E = lfilter(b,a,E)
arg_e = np.linspace(0, calc_cors_fast[-1][0], len(calc_cors_fast))
plt.plot(arg_e, E)

E = []
for i in range(0, len(calc_cors_orb)):
    x_real = real_cors[i][0]
    y_real = real_cors[i][1]
    x_calc = calc_cors_orb[i][0]
    y_calc = calc_cors_orb[i][1]
    e  = 0.05 * math.sqrt((x_real - x_calc)**2 + (y_real - y_calc)**2)
    E.append(e)

E = lfilter(b,a,E)
arg_e = np.linspace(0, calc_cors_orb[-1][0], len(calc_cors_orb))
plt.plot(arg_e, E)

plt.legend(['surf', 'sift', 'fast', 'orb'])
plt.show()


outF1.close()
outF2.close()
outF3.close()
outF4.close()
outF5.close()

