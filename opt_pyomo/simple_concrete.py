# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:19:53 2020

@author: Alex
https://pyomo.readthedocs.io/en/stable/pyomo_overview/simple_examples.html
"""

from __future__ import division
from pyomo.environ import *

model = ConcreteModel()

model.x = Var([1,2], domain=NonNegativeReals)

model.OBJ = Objective(expr = 2*model.x[1] + 3*model.x[2])

model.Constraint1 = Constraint(expr = model.x[1] >= 1)

#model.Constraint1 = Constraint(expr = 3*model.x[1] + 4*model.x[2] >= 1)


#%% Solve
from pyomo.opt import SolverFactory

# Create a solver
opt = pyomo.environ.SolverFactory('cplex')
opt.solve(model)
model.display()

