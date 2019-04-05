# HESS Web API Call
# Palomino, Oct 19, 2017

"""
Grid 20/20 HESS Web API Call
https://hess.grid2020.com/HESS/getXFcsv.jsp?UTILITY=xxxx&XFID=*&STARTTIME=yyyy-mm-ddThh:mm-hhmm&ENDTIME=yyyy-mm-ddThh:mm-hhmm&DIRECTION=FORWARD&DATATYPE=ENHANCED&USER=user&PASSWORD=password
This function downloads Grid20/20 data from the HESS Web API Server.
Below are the function parameters that define the data download.
UTILITY = ?
XFID = *
STARTTIME = yyyy-mm-ddThh:mm-hhmm
ENDTIME = yyyy-mm-ddThh:mm-hhmm
ENHANCED = Instructs HESS to return calculated values.
USER = uouapiuser
PASSWORD = e5P2kgnv
"""


def HESSwebAPIcall(varUtility, varXfid, varStartdate, varStarttime, varEnddate, varEndtime, varEnhanced, varUser, varPassword):
#def HESSwebAPIcall(varUtility, varXfid, varStarttime, varEndtime, varEnhanced, varUser, varPassword):

    # Print out Web API Request
    print("\n")
    print("User Web API Request:")
    csvUrl = ("https://hess.grid2020.com/HESS/getXFcsv.jsp?UTILITY={0}&XFID={1}&STARTTIME={2}&ENDTIME={3}&DIRECTION=FORWARD&DATATYPE={4}&USER={5}&PASSWORD={6}".format(varUtility, varXfid, varStarttime, varEndtime, varEnhanced, varUser, varPassword))
    print(csvUrl)

    # Import request module
    import requests
    import urllib

    # Pass function parameters into API call

    parameters = {"UTILITY": varUtility, "XFID": varXfid, "STARTTIME": varStarttime, "ENDTIME": varEndtime, "ENHANCED": varEnhanced, "USER": varUser, "PASSWORD": varPassword}
    # https://hess.grid2020.com/HESS/getXFcsv.jsp?UTILITY=xxxx&XFID=*&STARTTIME=yyyy-mm-ddThh:mm-hhmm&ENDTIME=yyyy-mm-ddThh:mm-hhmm&DIRECTION=FORWARD&DATATYPE=ENHANCED&USER=user&PASSWORD=password
    # response = requests.get("https://hess.grid2020.com/HESS/getXFcsv.jsp?UTILITY=1063&XFID=*&STARTTIME=2017-10-15T06:00-0600&ENDTIME=2017-10-15T18:00-0600&DIRECTION=FORWARD&DATATYPE=ENHANCED&USER=uouapiuser&PASSWORD=e5P2kgnv")

    # Make get request to HESS web API
    response = requests.get("https://hess.grid2020.com/HESS/getXFcsv.jsp", params=parameters)

    # Print the status of the server response 
    # https://en.wikipedia.org/wiki/List_of_HTTP_status_codes
    # 200 OK
    print("Server Response Status Code: {0}".format(response.status_code))

    # Print the content of the server response
    print("Server Response Content: {0}".format(response.content))

    # Download .csv data
    newFileName = 'data_XF'+str(varXfid)+'-'+str(varStartdate)+'to'+str(varEnddate)+'.csv'
    print(newFileName)
    urllib.request.urlretrieve(csvUrl, newFileName)

    # Moves downloaded .CSV file to new directory
    from HESS_File_Move import HESSfilemove

    HESSfilemove(newFileName)

    # print("\n")
    # print("README:")
    # print(__doc__)
