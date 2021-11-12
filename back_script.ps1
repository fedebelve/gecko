py -m venv gecko
cd .\gecko\Scripts\
.\activate
cd ..
cd ..
cd .\backend\
pip install -r .\requirements.txt
py .\manage.py migrate
py .\manage.py runserver
