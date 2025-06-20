partition=cnu
code=$PROJ_CODE

gpu_type="A100"
num_gpus=1
mem=124

NUM_CPU=16
TIME=48

algos=("grpo" "dr_grpo" "dapo" "optimal")

for algo in ${algos[@]}; do
    lbatch -c $NUM_CPU -g $num_gpus --gputype $gpu_type -m $mem -t $TIME -a $code -q $partition -n $algo --conda-env grpo \
        --cmd "python train.py --algo $algo"
done
