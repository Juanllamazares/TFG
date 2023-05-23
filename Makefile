
all:
	run

run:
	python3 backend/manage.py runserver

migrations:
	python3 backend/manage.py migrate


generate_models:
	python3 backend/main.py