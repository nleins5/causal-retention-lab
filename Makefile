.PHONY: install run clean

install:
	python -m pip install -e .

run:
	python -m retention_lab

clean:
	rm -rf artifacts data/retention_experiment.csv

