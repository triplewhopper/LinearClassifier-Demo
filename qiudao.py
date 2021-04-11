import math


def add(x,y):
    ...
def sub(x,y):
    ...
def mul(x,y):
    ...
def div(x,y):
    ...
def D(f):
    if f is add:

r={
    '+':lambda f:lambda g: D(f) + D(g),
    '-':lambda f:lambda g:D(f)-D(g),
    '*':lambda f:lambda g:D(f)*g+f*D(g),
    '/':lambda f:lambda g:(D(f)*g-f*D(g))/D(g)**2
}