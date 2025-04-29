mkdir -p experiment_logs

./platform/experiment_runner_internal.sh $1 $2 $3 $4 | tee experiment_logs/"$1-$2-$3-$(date +"%Y_%m_%d").log" 2>&1

# There are 3 parameters to define. Below are the values to choose from:
# 1) webapp name
# - pythonwebapp
# 2) workload name
# - zigzag
# - ladder
# - spiky
# 3) max number of instances to run
# - ZigZag - 10
# - IncreasingLadder - 14
# - Spiky - 15

#Example of a run:
#./experiment_runner.sh pythonwebapp cpu spiky 15
#./experiment_runner.sh pythonwebapp rps zigzag 10
#./experiment_runner.sh pythonwebapp rps ladder 14
