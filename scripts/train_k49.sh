#!/bin/bash

set -e

# Train with and without MUNIT
for i in {0..29}
do
	python -m core.train_baselines --config core/ldam_drw/configs/k49rotlt-2.0.yaml --k49_train_version $i
	python -m core.train_baselines --config core/ldam_drw/configs/k49bglt-2.0.yaml --k49_train_version $i
	python -m core.train_baselines --config core/ldam_drw/configs/k49dillt-2.0.yaml --k49_train_version $i

	python -m core.train_baselines --config core/ldam_drw/configs/k49rotlt-2.0.yaml --k49_train_version $i --train_rule DRS_Simple
	python -m core.train_baselines --config core/ldam_drw/configs/k49bglt-2.0.yaml --k49_train_version $i --train_rule DRS_Simple
	python -m core.train_baselines --config core/ldam_drw/configs/k49dillt-2.0.yaml --k49_train_version $i --train_rule DRS_Simple

	python -m core.train_baselines --config core/ldam_drw/configs/k49bglt-2.0-munit.yaml --k49_train_version $i --munit_ckpt outputs/k49bglt-2.0_vers$i/checkpoints/gen_00010000.pt --munit_config outputs/k49bglt-2.0_vers$i/config.yaml
	python -m core.train_baselines --config core/ldam_drw/configs/k49dillt-2.0-munit.yaml --k49_train_version $i --munit_ckpt outputs/k49dillt-2.0_vers$i/checkpoints/gen_00010000.pt --munit_config outputs/k49dillt-2.0_vers$i/config.yaml

	# Oracle augmentations.
	python -m core.train_baselines --config core/ldam_drw/configs/k49rotlt-2.0.yaml --k49_train_version $i --train_rule None --loss_type CE --oracle
	python -m core.train_baselines --config core/ldam_drw/configs/k49bglt-2.0.yaml --k49_train_version $i --train_rule None --loss_type CE --oracle
	python -m core.train_baselines --config core/ldam_drw/configs/k49dillt-2.0.yaml --k49_train_version $i --train_rule None --loss_type CE --oracle
done
