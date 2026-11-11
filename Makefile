.PHONY: install lint test gate meta bench baseline
install:
	pip install -e ".[dev]"
lint:
	ruff check . && ruff format --check . && mypy
test:
	pytest
gate:
	citegate gate --html citegate-report.html
meta:
	citegate meta
bench:
	citegate bench --services 300
baseline:  ## only after a reviewed, intentional improvement
	citegate baseline
