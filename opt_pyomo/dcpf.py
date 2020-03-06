# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:15:40 2020

@author: Alex
"""

def dc_line_rule(line, i):
    line.B = Param()
    line.Limit = Param()
    line.Angle_in = Var()
    line.Angle_out = Var()
    line.Power = Var( bounds= ( -line.Limit, line.Limit ) )
    line.power_flow = Constraint( expr=
                                 line.Power == line.B*(line.Angle_in - line.Angle_out) )
    line.IN = Connector( initialize=
                     { "Power": -line.Power, "Angle": line.Angle_in } )
    line.OUT = Connector( initialize=
                      { "Power": line.Power, "Angle": line.Angle_out } )

def dc_bus_rule(bus, i):
    bus.D = Param()
    bus.Angle = Var()
    bus.Power = VarList()
    
    def _power_balance(bus, P):
        return sum(P) == bus.D
 
    bus.BUS = Connector( initialize={ "Angle": bus.Angle })
    bus.BUS.add( bus.Power, "Power", aggregate=_power_balance )

def dc_generator_rule(bus, i):
    bus.D = Param()
    bus.Angle = Var()
    bus.Power = VarList()
    
    def _power_balance(bus, P):
        return summation(P) == bus.D
 
    bus.BUS = Connector( initialize={ "Angle": bus.Angle })
    bus.BUS.add( bus.Power, "Power", aggregate=_power_balance )
