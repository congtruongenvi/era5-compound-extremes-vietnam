.PHONY: help venv synthetic real clean

help:
	@echo "make venv       - create the environment and install requirements"
	@echo "make synthetic  - run the full pipeline on the validation field (no download)"
	@echo "make real       - run the full pipeline on ERA5 (requires CDS API key)"
	@echo "make clean      - remove generated figures and processed data"

venv:
	uv venv --python 3.12 .venv
	uv pip install --python .venv/Scripts/python.exe -r requirements.txt

synthetic:
	.venv/Scripts/python.exe src/pipeline.py --synthetic --start 1991-01-01 --end 2024-12-31

real:
	.venv/Scripts/python.exe src/pipeline.py --start 1991-01-01 --end 2024-12-31

clean:
	rm -f figures/*.png data/processed/*
