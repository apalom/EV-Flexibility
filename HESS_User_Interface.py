# HESS User Interface
# Palomino, Oct 20, 2017
# This function provides the user an opportunity to define Grid 20/20 HESS Data Download

import timeit


# start timer 
timeMain = timeit.default_timer()


#PackSize
var_utility = str(1063)
var_xfid = str(1003)

'''
#Bus DCFC
var_utility = str(1063)
var_xfid = str(1001)
'''

'''
# Define user inputs
var_startdate = input('Enter the data collection start date [yyyy-mm-dd]: ')
var_startclock = input('Enter the data collection start time [24hr hh:mm]: ')
var_enddate = input('Enter the data collection end date [yyyy-mm-dd]: ')
var_endclock = input('Enter the data collection start time [24hr hh:mm]: ')
utc_offset = input('Enter UTC Offset [MST = 0600]: ')
'''

# Define user inputs
var_startdate = '2018-06-20'
var_startclock = '00:00'
var_enddate = '2018-12-10'
var_endclock = '00:00'
utc_offset = '0800'


var_starttime = str(var_startdate)+"T"+str(var_startclock)+"-"+str(utc_offset)
var_endtime = var_enddate+"T"+var_endclock+"-"+str(utc_offset)

var_enhanced = "ENHANCED"

var_user = "uouapiuser"
var_password = "e5P2kgnv"

from HESS_Web_API_Call import HESSwebAPIcall

HESSwebAPIcall(var_utility,var_xfid,var_startdate,var_starttime,var_enddate,var_endtime,var_enhanced,var_user,var_password)


# timeit statement
elapsedMain = timeit.default_timer() - timeMain
print('Main time: {0:.4f} sec'.format(elapsedMain))
