PGPASSWORD=XXX psql -h 35.xxx.xxx.xx:xxxx -d postgres -U postgres -c "\copy water.water_potability TO 'water.csv' WITH (FORMAT csv) "
python train_data.py
rm water.csv    
