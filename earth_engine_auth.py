import ee

def authenticate():
    try:
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()
    print("EE Ready")
