
all:
	run

run:
	python3 backend/manage.py runserver

migrate:
	python3 backend/manage.py migrate


generate_models:
	python3 backend/main.py§

heroku:
	git subtree push --prefix backend heroku main