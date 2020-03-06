# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:06:10 2020

@author: Alex
"""

from pyomo.environ import *
# Create a solver
opt = pyomo.environ.SolverFactory('glpk')

model = AbstractModel()

model.N = Param( within=PositiveIntegers )
model.P = Param( within=RangeSet( model.N ) )
model.M = Param( within=PositiveIntegers )

model.Locations = RangeSet( model.N )
model.Customers = RangeSet( model.M )

model.d = Param( model.Locations, model.Customers )

model.x = Var( model.Locations, model.Customers, bounds=(0.0, 1.0) )
model.y = Var( model.Locations, within=Binary )

def obj_rule(model):
    return sum( model.d[n,m]*model.x[n,m]
        for n in model.Locations for m in model.Customers )

model.obj = Objective( rule=obj_rule )

def single_x_rule(model, m):
    return sum( model.x[n,m] for n in model.Locations ) == 1.0

model.single_x = Constraint( model.Customers, rule=single_x_rule )

def bound_y_rule(model, n,m):
    return model.x[n,m] - model.y[n] <= 0.0

model.bound_y = Constraint( model.Locations, model.Customers,
                           rule=bound_y_rule )

def num_facilities_rule(model):
    return sum( model.y[n] for n in model.Locations ) == model.P

model.num_facilities = Constraint( rule=num_facilities_rule )

# Load Data
data = DataPortal()
data.load(filename="opt_pyomo\pmedian.dat", model=model)

# Create a model instance and optimize
instance = model.create_instance(data)
results = opt.solve(instance)
instance.display()
