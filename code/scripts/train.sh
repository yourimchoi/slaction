#!/bin/bash
export WANDB_MODE='disabled'

# Define cleanup function
cleanup() {
    echo -e "\nKeyboard interrupt received. Cleaning up and exiting..."
    exit 1
}

# Set up trap for SIGINT (Ctrl+C)
trap cleanup SIGINT

# Function to run the training script
run_training() {
    local gpu=$1
    local yaml_index=$2
    local num_workers=$3
    local yaml_path="../exp_config/${yaml_index}.yaml"
    
    CUDA_VISIBLE_DEVICES=$gpu python ../src/core/Train.py --config $yaml_path --num_workers $num_workers
    if [ $? -ne 0 ]; then
        echo "Training failed. Exiting."
        exit 1
    fi
}

# Function to display usage information
usage() {
    echo "Usage: $0 <gpu> <exp> <workers>"
    echo "or: $0 --gpus <gpu> --exp <exp> --num_workers <workers>"
    exit 1
}

# Main execution block
main() {
    # Parse command-line arguments
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --gpus) gpu="$2"; shift ;;
            --exp) yaml_index="$2"; shift ;;
            --num_workers) num_workers="$2"; shift ;;
            --testmode) export WANDB_MODE='disabled' ;;
            *) # if no flags, assume positional arguments
                if [ -z "$gpu" ]; then
                    gpu="$1"
                elif [ -z "$yaml_index" ]; then
                    yaml_index="$1"
                elif [ -z "$num_workers" ]; then
                    num_workers="$1"
                else
                    echo "Unknown parameter passed: $1"
                    usage
                fi
                ;;
        esac
        shift
    done

    # Check if all required parameters are provided
    if [ -z "$gpu" ] || [ -z "$yaml_index" ] || [ -z "$num_workers" ]; then
        usage
    fi

    # Execute pipeline steps
    if ! run_training $gpu $yaml_index $num_workers; then
        echo "Training failed or was interrupted"
        exit 1
    fi
}

# Call main function
main "$@"