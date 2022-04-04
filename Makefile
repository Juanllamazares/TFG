
all:
	run

run:
	python3 backend/manage.py runserver

migrations:
	python3 backend/manage.py runserver