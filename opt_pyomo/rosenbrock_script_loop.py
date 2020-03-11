# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:52:42 2020

@author: Alex
"""

from pyomo.environ import *

model = ConcreteModel()
model.x = Var()
model.y = Var()

def rosenbrock(m):
    return (1.0-m.x)**2 + 100.0*(m.y - m.x**2)**2

model.obj = Objective(rule=rosenbrock, sense=minimize)

print('iter | x_init | y_init | x_soln | y_soln')

y_init = 5.0; it = 0;

for x_init in range(0, 6):
    model.x = x_init
    model.y = 5.0
    solver = SolverFactory('ipopt')
    solver.solve(model)
    print("{0}   {1:6.2f}   {2:6.2f}   {3:6.2f}   {4:6.2f}".format(it, x_init, 
          y_init, value(model.x), value(model.y)))
    print("--- {} ---".format(value(model.obj)))
    it += 1;