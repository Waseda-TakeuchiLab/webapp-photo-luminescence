run:
	python webapp_photo_luminescence

test: mypy unittest

mypy:
	mypy .

unittest:
	coverage run -m unittest
	coverage html -i
	coverage report -i