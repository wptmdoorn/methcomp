export SHELL := /bin/bash

pytest:
	pytest methcomp --mpl-generate-path=methcomp/tests/baseline -p no:warnings
	pytest methcomp --mpl -p no:warnings
