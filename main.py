import math
from sympy import Symbol, cos
import matplotlib.pyplot as plt


def iteration_method(func, y_sym, y_0, e):

    y_last = y_0 - 2 * e
    y_n = y_0

    while abs(y_last - y_n) > e:
        y_last = y_n
        y_n = func.evalf(subs={y_sym: y_last})
    return y_n


def euler_method(y_diff, x_sym, y_sym, y0, a, b, h):
    x_arr = [a]
    y_arr = [y0]

    value = a + h
    while value <= b:
        y = y_arr[-1] + h/2 * \
            (y_diff.evalf(subs={x_sym: x_arr[-1], y_sym: y_arr[-1]}) +
             y_diff.evalf(subs={x_sym: value}))

        y_arr.append(y.evalf(subs={y_sym: iteration_method(y, y_sym, y_arr[-1], 0.01)}))
        x_arr.append(value)
        value += h
    return x_arr, y_arr


def runge_kutta_4th_order(y_diff, x_sym, y_sym, y0, a, b, h):
    x_arr = [a]
    y_arr = [y0]

    value = a + h
    while value <= b:
        k1 = h * y_diff.evalf(subs={x_sym: x_arr[-1], y_sym: y_arr[-1]})
        k2 = h * y_diff.evalf(subs={x_sym: x_arr[-1] + h/2, y_sym: y_arr[-1] + k1/2})
        k3 = h * y_diff.evalf(subs={x_sym: x_arr[-1] + h/2, y_sym: y_arr[-1] + k2/2})
        k4 = h * y_diff.evalf(subs={x_sym: x_arr[-1] + h, y_sym: y_arr[-1] + k3})
        y_arr.append(y_arr[-1] + 1/6 * (k1 + 2*k2 + 2*k3 + k4))
        x_arr.append(value)
        value += h

    return x_arr, y_arr


def Gauss_method(func, sym, a, b, e, t_arr=[-0.861136, -0.339981, 0.339981, 0.861136], c_arr=[0.347855, 0.652145, 0.652145, 0.347855]):
    if (b - a) / 5 > e:
        n = 5
        delta = (b - a) / n
        return sum(list([Gauss_method(func, sym, a + delta*i, a + delta*(i+1), e, t_arr, c_arr) for i in range(n)]))

    else:
        if len(t_arr) == len(c_arr):
            t_arr = [t_arr[i] + abs(a - b)/2 for i in range(len(t_arr))]

            arr = []
            for i in range(len(t_arr)):
                xi = (b + a) / 2 + (b - a) * t_arr[i] / 2
                arr.append(c_arr[i] * func.evalf(subs={sym: xi}))

            return (b - a) / 2 * sum(arr)


def find_a(s):
    if s == 0:
        return 1

    s_sym = Symbol("s")
    mult = s_sym

    for i in range(1, s - 1):
        mult *= s_sym + i

    integral = Gauss_method(mult, s_sym, 0, 1, 0.01)
    return 1/math.factorial(s) * integral


def get_subtract(y_arr):
    subtracts = [y_arr[:]]
    for i in range(len(y_arr) - 1):
        subtracts.append([subtracts[i][j + 1] - subtracts[i][j] for j in range(len(subtracts[i]) - 1)])
    return subtracts[-1][0]


def adams_extrapolation_method(y_diff, x_sym, y_sym, x_arr, y_arr, a, b, h):
    value = x_arr[-1]
    m = len(y_arr)

    while value <= b:
        suma = 0

        for s in range(m):
            suma += find_a(s) * get_subtract([y_diff.evalf(subs={x_sym: x_arr[-i], y_sym: y_arr[-i]}) for i in range(s+1, 0, -1)])

        y_arr.append(y_arr[-1] + h * suma)
        x_arr.append(value)
        value += h

    return x_arr, y_arr


def adams_interpolation_method(y_diff, x_sym, y_sym, x_arr, y_arr, a, b, h):
    value = x_arr[-1]
    m = len(y_arr)

    while value <= b:
        suma = 0

        for s in range(m):
            x_arr.append(value + h)
            y_arr.append(y_arr[-1])
            suma += find_a(s) * get_subtract(
                [y_diff.evalf(subs={x_sym: x_arr[-i], y_sym: y_arr[-i]}) for i in range(s + 2, 0, -1)])
            x_arr.pop()
            y_arr.pop()

        delta = y_arr[-1] + h * suma
        y = h * y_diff.evalf(subs={x_sym: value}) + delta
        y_arr.append(y.evalf(subs={y_sym: iteration_method(y, y_sym, y_arr[-1], 0.01)}))

        x_arr.append(value)
        value += h

    return x_arr, y_arr


x_sym = Symbol("x")
y_sym = Symbol("y")
# f = 0.418 * (x_sym**2 + sin(1.2 * x_sym))  # + 1344 * y_sym
f = cos(x_sym)
y0 = 0
a = 0
b = 6.28
#b = 3.15
e = 0.01

x_arr_1, y_arr_1 = euler_method(f, x_sym, y_sym, y0, a, b, e)
x_arr_2, y_arr_2 = runge_kutta_4th_order(f, x_sym, y_sym, y0, a, b, e)

x_arr, y_arr = runge_kutta_4th_order(f, x_sym, y_sym, y0, a, a + e * 5, e)
x_arr_3, y_arr_3 = adams_extrapolation_method(f, x_sym, y_sym, x_arr, y_arr, a, b, e)
x_arr, y_arr = runge_kutta_4th_order(f, x_sym, y_sym, y0, a, a + e * 5, e)
x_arr_4, y_arr_4 = adams_interpolation_method(f, x_sym, y_sym, x_arr, y_arr, a, b, e)


fig, ax = plt.subplots()

ax.plot(x_arr_1, y_arr_1, "g-.", label="метод Ейлера")
ax.plot(x_arr_2, y_arr_2, "b--", label="метод Рунге-Кутта")
ax.plot(x_arr_3, y_arr_3, "y--", label="екстраполяційний метод Адамса")
ax.plot(x_arr_4, y_arr_4, "r-.", label="інтерполяційний метод Адамса")


ax.grid()

plt.legend()
plt.show()


h = [1, 3, 2, 4, 7, 3, 10, 1, 7, 43, 76, 12, 65, 12]