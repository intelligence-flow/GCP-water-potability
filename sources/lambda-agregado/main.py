import requests 
from sqlalchemy import create_engine
import psycopg2
def agregar(external_request):

  headers = {} 
  request_json = external_request.get_json(silent=True)
  ph = request_json["ph"]
  Hardness = request_json["Hardness"]
  Solids = request_json["Solids"]
  Chloramines = request_json["Chloramines"]
  Sulfate = request_json["Sulfate"]
  Conductivity = request_json["Conductivity"]
  Organic_carbon = request_json["Organic_carbon"]
  Trihalomethanes = request_json["Trihalomethanes"]
  Turbidity = request_json["Turbidity"]
  Potability = request_json["Potability"]  
  headers = {
  'Content-Type': 'application/json',
  } 

  #Guardo los datos en la BD
  try:
    engine = create_engine("postgresql+psycopg2://postgres:postgres:XXX@35.xxx.xxx.xx:xxxx/postgres")
    connection = engine.connect()
    connection.execute(f"INSERT INTO water.water_potability VALUES ({ph},{Hardness},{Solids},{Chloramines},{Sulfate},{Conductivity},{Organic_carbon},{Trihalomethanes},{Turbidity},{Potability})")
    return "OK"
  except:
    return "Error en guardado de datos"
  
  return "OK"


