import math


def g(x, y):
    return math.sin(x + y) + math.cos(x + 2*y)


def dgdx(x, y):
    return math.cos(x + y) - math.sin(x + 2*y)


def dgdy(x, y):
    return math.cos(x + y) - 2 * math.sin(x + 2*y)


def descent(x, y):
    para = 0.0001
    for _ in range(100000):
        grad = math.sqrt(dgdx(x, y)**2 + dgdy(x, y)**2)
        x = x - para * dgdx(x, y) / grad
        y = y - para * dgdy(x, y) / grad
        if grad < 1.0E-14:
            break
    return x, y


def main():
    x, y = 0, 0
    x, y = descent(x, y)
    print("x is %f, y is %f, minimum of g is %f" % (x, y, g(x, y)))


if __name__ == "__main__":
    main()
    input()
