database_bl := "$STREAMS_DIR/examples/supersonic_sbli/database_bl.dat"

nv:
	mkdir -p $APPTAINER_TMPDIR

	rm -f nv.sandbox

	taskset -c 0-15 sudo apptainer build --sandbox \
		nv.sandbox \
		"docker://nvcr.io/nvidia/nvhpc:24.7-devel-cuda_multi-ubuntu22.04"
# "docker://nvcr.io/nvidia/nvhpc:25.3-devel-cuda_multi-ubuntu22.04"

base:
	mkdir -p $APPTAINER_TMPDIR

	rm -f base.sandbox 
	echo $APPTAINER_TMPDIR
	time taskset -c 0-15 sudo -E apptainer build --sandbox --nv base.sandbox base.apptainer
	sudo du -sh base.sandbox

build:
	rm -f streams.sif
	echo $APPTAINER_TMPDIR
	test -e libstreams*.so && rm libstreams*.so || true
	# f2py -m libstreamsMin -h ./libstreamsMin.pyf --overwrite-signature ${STREAMS_DIR}/src/min_api.F90
	# f2py -m libstreamsMod -h ./libstreamsMod.pyf --overwrite-signature ${STREAMS_DIR}/src/mod_api.F90
	# python3 patch_pyf.py
	time taskset -c 0-15 sudo -E apptainer build --nv streams.sif build.apptainer
	du -sh streams.sif

# build a config json file as input to the solver
config_output := "./output/input.json"
#streams_flow_type := "shock-boundary-layer"
streams_flow_type := "boundary-layer"

config:
	echo {{config_output}}

	# 600, 208, 100
	#--fixed-dt 0.0008439 \

	cargo r -- config-generator {{config_output}} \
		{{streams_flow_type}} \
		--steps 2500 \
		--reynolds-number 250 \
		--mach-number 2.28 \
		--x-divisions 600 \
		--y-divisions 208 \
		--z-divisions 100 \
		--json \
		--x-length 27.0 \
		--y-length 6.0 \
		--z-length 3.8 \
		--rly-wr 2.5 \
		--mpi-x-split 4 \
		--span-average-io-steps 5 \
		--probe-io-steps 0 \
		--python-flowfield-steps 500 \
		--use-python \
		--nymax-wr 201 \
		--sensor-threshold 0.1 \
		DMDc \
			--amplitude 1.0 \
			--slot-start 100 \
			--slot-end 149

	cat {{config_output}}

jet_validation_base_path := "./distribute/jet_validation/"

jet_validation_number := "20"
jet_validation_batch_name := "jet_validation_" + jet_validation_number
jet_valiation_output_folder := jet_validation_base_path + jet_validation_batch_name

jet_validation:
	echo {{jet_valiation_output_folder}}

	# 600, 208, 100
	#--steps 10000 \

	cargo r -- cases jet-validation \
		{{jet_valiation_output_folder}} \
		--batch-name {{jet_validation_batch_name}} \
		--solver-sif ./streams.sif \
		--steps 50000 \
		--database-bl {{database_bl}} \
		--matrix @karlik:matrix.org

variable_dt_base_path := "./distribute/variable_dt/"

variable_dt_case_number := "01"
variable_dt_batch_name := "variable_dt_" + variable_dt_case_number
variable_dt_output_folder := variable_dt_base_path + variable_dt_batch_name

variable_dt:
	echo {{jet_valiation_output_folder}}

	# 600, 208, 100
	#--steps 10000 \

	cargo r -- cases variable-dt \
		{{variable_dt_output_folder}} \
		--batch-name {{variable_dt_batch_name}} \
		--solver-sif ./streams.sif \
		--steps 20000 \
		--database-bl {{database_bl}} \
		--matrix @karlik:matrix.org

ai_institute_base_path := "./distribute/ai_institute/"
ai_institute_case_number := "01"
ai_institute_batch_name := "ai_institute_" + ai_institute_case_number
ai_institute_output_folder := ai_institute_base_path + ai_institute_batch_name

ai_institute:
	echo {{ai_institute_output_folder}}

	cargo r -- cases ai-institute \
		{{ai_institute_output_folder}} \
		--batch-name {{ai_institute_batch_name}} \
		--solver-sif ./streams.sif \
		--steps 50000 \
		--database-bl {{database_bl}} \
		--matrix @karlik:matrix.org

run:
	cargo r -- run-local \
		--workdir ./output/ \
		--config ./output/input.json \
		--database {{database_bl}} \
		--python-mount $STREAMS_DIR/streamspy \
		16

# get a shell inside the container
# requires the ./output directory (with its associated folders) to be created, 
# and a ./streams.sif file to be made
shell:
	apptainer shell --nv --bind ./output/distribute_save:/distribute_save,./output/input:/input ./streams.sif

# get a shell inside the container
# and bind your $STREAMS_DIR environment variable to the folder
# /streams
local:
	apptainer shell --nv --bind $STREAMS_DIR:/streams ./base.sif

vtk:
	cargo r --release -- hdf5-to-vtk ./output/distribute_save

base_animate_folder := "./distribute/ai_institute/ai_institute_01/ai_institute_01/"

animate:
	cargo r --release -- animate \
		"{{base_animate_folder}}/no_actuator" \
		--decimate 5

	#cargo r --release -- animate \
	#	"{{base_animate_folder}}/sinusoidal" \
	#	--decimate 5
