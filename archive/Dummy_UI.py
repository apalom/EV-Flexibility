
varUtility = str(1063)
varXfid = str(1000)

# Define user inputs
varStartdate = '2017-10-20'
varStartclock = '00:01'
varEnddate = '2017-10-21'
varEndclock = '23:59'
utcOffset = '0600'
varStarttime = str(varStartdate)+"T"+str(varStartclock)+"-"+str(utcOffset)
varEndtime = varEnddate+"T"+varEndclock+"-"+str(utcOffset)

varEnhanced = "ENHANCED"

varUser = "uouapiuser"
varPassword = "e5P2kgnv"

# print(var_starttime)
# print(var_endtime)

from HESS_Web_API_Call import HESSwebAPIcall

HESSwebAPIcall(varUtility, varXfid, varStartdate, varStarttime, varEnddate, varEndtime, varEnhanced, varUser, varPassword)
