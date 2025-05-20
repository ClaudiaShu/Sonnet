#!/bin/bash

# This script will generate all the individual experiment run scripts

# Create a directory for experiment runs
mkdir -p scripts/runs

# Function to create a run script for each experiment
create_run_script() {
    local dataset_short=$1
    local dataset=$2
    local horizon=$3
    local learning_rate=$4
    local d_model=$5
    
    # Calculate n_atoms as 512/d_model
    local n_atoms=$((512 / d_model))
    
    # Create a unique filename for this experiment
    local filename="scripts/runs/run_${dataset_short}_h${horizon}.sh"
    
    # Create the run script
    cat > $filename <<EOF
#!/bin/bash
python scripts/run_experiment.py \\
    model=sonnet \\
    dataset=${dataset} \\
    exp=weatherbench \\
    exp.batch_size=64 \\
    exp.epochs=100 \\
    exp.learning_rate=${learning_rate} \\
    exp.pred_length=${horizon} \\
    exp.seq_length=28 \\
    model.model_params.d_model=${d_model} \\
    model.model_params.n_atoms=${n_atoms}
EOF
    
    # Make the script executable
    chmod +x $filename
    
    # Add this run to the master script
    echo "echo \"Running experiment: ${dataset_short} (horizon=${horizon}, d_model=${d_model})\"" >> $master_script
    echo "$filename > logs/${dataset_short}_h${horizon}_d${d_model}.log 2>&1" >> $master_script
}

# Generate scripts for each experiment in the table
create_run_script "wea_hk_2018" "weatherbench/hongkong/hongkong_2018" 4 0.002 8
create_run_script "wea_hk_2018" "weatherbench/hongkong/hongkong_2018" 12 0.005 16
create_run_script "wea_hk_2018" "weatherbench/hongkong/hongkong_2018" 28 0.0002 32
create_run_script "wea_hk_2018" "weatherbench/hongkong/hongkong_2018" 120 0.001 8

create_run_script "wea_ny_2018" "weatherbench/newyork/newyork_2018" 4 0.0002 16
create_run_script "wea_ny_2018" "weatherbench/newyork/newyork_2018" 12 0.0005 32
create_run_script "wea_ny_2018" "weatherbench/newyork/newyork_2018" 28 0.001 16
create_run_script "wea_ny_2018" "weatherbench/newyork/newyork_2018" 120 0.0005 16

create_run_script "wea_ct_2018" "weatherbench/capetown/capetown_2018" 4 0.0005 8
create_run_script "wea_ct_2018" "weatherbench/capetown/capetown_2018" 12 0.005 16
create_run_script "wea_ct_2018" "weatherbench/capetown/capetown_2018" 28 0.002 16
create_run_script "wea_ct_2018" "weatherbench/capetown/capetown_2018" 120 0.001 8

create_run_script "wea_ld_2018" "weatherbench/london/london_2018" 4 0.0005 32
create_run_script "wea_ld_2018" "weatherbench/london/london_2018" 12 0.0002 32
create_run_script "wea_ld_2018" "weatherbench/london/london_2018" 28 0.005 16
create_run_script "wea_ld_2018" "weatherbench/london/london_2018" 120 0.0002 32

create_run_script "wea_sg_2018" "weatherbench/singapore/singapore_2018" 4 0.0002 32
create_run_script "wea_sg_2018" "weatherbench/singapore/singapore_2018" 12 0.001 16
create_run_script "wea_sg_2018" "weatherbench/singapore/singapore_2018" 28 0.002 16
create_run_script "wea_sg_2018" "weatherbench/singapore/singapore_2018" 120 0.0002 8


create_run_script "wea_hk_2017" "weatherbench/hongkong/hongkong_2017" 4 0.0002 8
create_run_script "wea_hk_2017" "weatherbench/hongkong/hongkong_2017" 12 0.002 16
create_run_script "wea_hk_2017" "weatherbench/hongkong/hongkong_2017" 28 0.005 16
create_run_script "wea_hk_2017" "weatherbench/hongkong/hongkong_2017" 120 0.005 32

create_run_script "wea_ny_2017" "weatherbench/newyork/newyork_2017" 4 0.005 32
create_run_script "wea_ny_2017" "weatherbench/newyork/newyork_2017" 12 0.002 32
create_run_script "wea_ny_2017" "weatherbench/newyork/newyork_2017" 28 0.0005 32
create_run_script "wea_ny_2017" "weatherbench/newyork/newyork_2017" 120 0.0002 16

create_run_script "wea_ct_2017" "weatherbench/capetown/capetown_2017" 4 0.0002 8
create_run_script "wea_ct_2017" "weatherbench/capetown/capetown_2017" 12 0.001 32
create_run_script "wea_ct_2017" "weatherbench/capetown/capetown_2017" 28 0.005 8
create_run_script "wea_ct_2017" "weatherbench/capetown/capetown_2017" 120 0.002 8

create_run_script "wea_sg_2017" "weatherbench/singapore/singapore_2017" 4 0.002 8
create_run_script "wea_sg_2017" "weatherbench/singapore/singapore_2017" 12 0.0005 32
create_run_script "wea_sg_2017" "weatherbench/singapore/singapore_2017" 28 0.002 16
create_run_script "wea_sg_2017" "weatherbench/singapore/singapore_2017" 120 0.002 8

create_run_script "wea_ld_2017" "weatherbench/london/london_2017" 4 0.0005 32
create_run_script "wea_ld_2017" "weatherbench/london/london_2017" 12 0.002 8
create_run_script "wea_ld_2017" "weatherbench/london/london_2017" 28 0.005 16
create_run_script "wea_ld_2017" "weatherbench/london/london_2017" 120 0.002 16


create_run_script "wea_hk_2016" "weatherbench/hongkong/hongkong_2016" 4 0.0005 16
create_run_script "wea_hk_2016" "weatherbench/hongkong/hongkong_2016" 12 0.002 8
create_run_script "wea_hk_2016" "weatherbench/hongkong/hongkong_2016" 28 0.0002 8
create_run_script "wea_hk_2016" "weatherbench/hongkong/hongkong_2016" 120 0.005 8

create_run_script "wea_sg_2016" "weatherbench/singapore/singapore_2016" 4 0.001 8
create_run_script "wea_sg_2016" "weatherbench/singapore/singapore_2016" 12 0.005 32
create_run_script "wea_sg_2016" "weatherbench/singapore/singapore_2016" 28 0.0002 32
create_run_script "wea_sg_2016" "weatherbench/singapore/singapore_2016" 120 0.0005 16

create_run_script "wea_ny_2016" "weatherbench/newyork/newyork_2016" 4 0.0002 8
create_run_script "wea_ny_2016" "weatherbench/newyork/newyork_2016" 12 0.0005 8
create_run_script "wea_ny_2016" "weatherbench/newyork/newyork_2016" 28 0.0002 32
create_run_script "wea_ny_2016" "weatherbench/newyork/newyork_2016" 120 0.0005 32

create_run_script "wea_ct_2016" "weatherbench/capetown/capetown_2016" 4 0.002 32
create_run_script "wea_ct_2016" "weatherbench/capetown/capetown_2016" 12 0.005 16
create_run_script "wea_ct_2016" "weatherbench/capetown/capetown_2016" 28 0.005 8
create_run_script "wea_ct_2016" "weatherbench/capetown/capetown_2016" 120 0.005 32

create_run_script "wea_ld_2016" "weatherbench/london/london_2016" 4 0.005 32
create_run_script "wea_ld_2016" "weatherbench/london/london_2016" 12 0.0005 8
create_run_script "wea_ld_2016" "weatherbench/london/london_2016" 28 0.002 8
create_run_script "wea_ld_2016" "weatherbench/london/london_2016" 120 0.005 8



echo "All experiment scripts have been generated."
echo "To run all experiments, use: bash"
