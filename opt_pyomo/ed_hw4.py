# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:31:20 2020

@author: Alex
"""

from pyomo.environ import *
import numpy as np

model = AbstractModel()
model.Units = Set()
model.a = Param(model.Units)

data = DataPortal()
data.load(filename='hw4_data.xlsx', range='data', 
                    param=model.a, index=model.Units)

instance = model.create_instance(data)
instance.pprint()

#%%

from pyomo.environ import *
import numpy as np

model = AbstractModel()

model.Units = Set()
model.Data = Param(model.Units)

def bounds_rule(model, u):    
    return (model.Data[u,'Pmin'],model.Data[u,'Pmax'])

model.P = Var(model.Units, bounds=bounds_rule, domain=NonNegativeReals)


data = DataPortal()
data.load(filename="ed_hw4.dat", model=model)

instance = model.create_instance(data)

instance.pprint()

#%% SOLVES!

from pyomo.environ import *
import numpy as np

model = AbstractModel()

model.Units = Set()

model.pMin = Param(model.Units)
model.pMax = Param(model.Units)
model.a = Param(model.Units)
model.b = Param(model.Units)
model.c = Param(model.Units)

def bounds_rule(model, u):    
    return (model.pMin[u],model.pMax[u])

model.P = Var(model.Units, bounds=bounds_rule, domain=NonNegativeReals)

# declare objective
def obj_rule(model):
    return (model.a['G1'] + model.a['G2'] + model.a['G3'] + 
            summation(model.b, model.P) + 
            summation(model.c, model.P, model.P))

model.Obj = Objective(rule=obj_rule, sense=minimize)

def load_balance(model,u):
    return summation(model.P) == 835

model.Balance = Constraint(model.Units, rule=load_balance) 

data = DataPortal()
data.load(filename="ed.dat", model=model)

instance = model.create_instance(data)
instance.pprint()

pyomo.environ.SolverFactory('cplex').solve(instance).write()
instance.display()
