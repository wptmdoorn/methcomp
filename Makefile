export SHELL := /bin/bash

pytest:
	pytest methcomp --mpl-generate-path=methcomp/tests/baseline
	pytest methcomp --mpl
