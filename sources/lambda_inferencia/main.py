import requests 
from sqlalchemy import create_engine
import psycopg2

def inferencia_water(external_request):

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
  ID_sensor = request_json["ID_sensor"]

  headers = {
  'Content-Type': 'application/json',
  } 

  data = [[ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity]]
  json_data =  {"columns": ["ph","Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity"],"data": data}
  response = requests.post('http://35.xxx.xxx.xx:xxxx/invocations', headers=headers, json=json_data) 

  inferencia = str(response.text)
  if len(inferencia) > 0:
    inferencia = inferencia[1]

  #Guardo la inferencia en la BD
  #try:
    #engine = create_engine("postgresql+psycopg2://postgres:Pass4Bml2@35.202.217.75:5432/postgres%3Ddbo,water")
  engine = create_engine("postgresql+psycopg2://postgres:XXX@35.xxx.xxx.xx:xxxx/postgres")
  connection = engine.connect()
  connection.execute(f"INSERT INTO water.water_infereneces VALUES ({ph},{Hardness},{Solids},{Chloramines},{Sulfate},{Conductivity},{Organic_carbon},{Trihalomethanes},{Turbidity},{inferencia},'{ID_sensor}',NOW())")
  #except:
  #  print("Error en conexion")

  return inferencia


