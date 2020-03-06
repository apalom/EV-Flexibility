# -*- coding: utf-8 -*-
"""
Ex Code Credit: 
    (1) https://www.osti.gov/servlets/purl/1376827
    (2) https://pyomo.readthedocs.io/en/stable/working_models.html
"""

from pyomo.environ import *

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

#%% Import Data

data = DataPortal()
data.load(filename="pmedian.dat", model=model)

#%% Solve
from pyomo.opt import SolverFactory

# Create a solver
opt = pyomo.environ.SolverFactory('glpk')
# Add data to abstract model
instance = model.create_instance(data)

# Solve instance
opt.solve(instance)
# Display results
instance.display()




