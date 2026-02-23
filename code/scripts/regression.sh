#!/bin/bash
export WANDB_MODE='disabled'  

# Define cleanup function
cleanup() {
    echo -e "\nKeyboard interrupt received. Cleaning up and exiting..."
    exit 1
}
trap cleanup SIGINT


run_regression() {
    local exp_number=$1
    python ../src/postprocess/Regression_yr.py --exp_number $exp_number
    if [ $? -ne 0 ]; then
        echo "Regression failed. Exiting."
        exit 1
    fi
}


usage() {
    echo "Usage: $0 <exp_number>"
    echo "or: $0 --exp_number <exp_number>"
    exit 1
}


main() {
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --exp_number) exp_number="$2"; shift ;;
            *) 
                if [ -z "$exp_number" ]; then
                    exp_number="$1"
                else
                    echo "Unknown parameter passed: $1"
                    usage
                fi
                ;;
        esac
        shift
    done

    if [ -z "$exp_number" ]; then
        usage
    fi

    run_regression "$exp_number"
}

main "$@"