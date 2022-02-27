#!/bin/bash

set -e

for i in {0..1}
do
	python -m core.train_munit --config core/munit/configs/k49dillt-2.0.yaml --k49_train_version $i
	python -m core.train_munit --config core/munit/configs/k49bglt-2.0.yaml --k49_train_version $i
done
