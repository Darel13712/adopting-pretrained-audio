#!/bin/bash

# Default values
time_limit_hours=14
#cpus_per_task=40
memory_gb=20
dataset="m4a"
model="musicnn"


# Parsing command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --time) time_limit_hours="$2"; shift ;;
#    --cpus-per-task) cpus_per_task="$2"; shift ;;
    --memory) memory_gb="$2"; shift ;;
    --model) model="$2"; shift ;;
    --maxseqlen) maxseqlen="$2"; shift ;;
    --batch) batch="$2"; shift ;;
    --epochs) epochs="$2"; shift ;;
    --ifreeze) item_freeze="$2"; shift ;;
    --logdir) logdir="$2"; shift ;;
    --lastepoch) last_epoch="$2"; shift ;;
    --comment) comment="$2"; shift ;;
    --hiddim) hidden_dim="$2"; shift ;;
    --shuffle) shuffle="$2"; shift ;;
#    --k) k="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# Construct job name
job_name="b_${model}_${dataset}_${hidden_dim}_${maxseqlen}_${comment}"

# Format time limit as a string
time_limit="${time_limit_hours}:00:00"

cmd="python ./train_bert.py"
[ -n "$dataset" ] && cmd+=" --dataset $dataset"
[ -n "$model" ] && cmd+=" --model_name $model"
[ -n "$maxseqlen" ] && cmd+=" --max_seq_len $maxseqlen"
[ -n "$batch" ] && cmd+=" --batch_size $batch"
[ -n "$epochs" ] && cmd+=" --num_epochs $epochs"
[ -n "$item_freeze" ] && cmd+=" --item_freeze $item_freeze"
[ -n "$comment" ] && cmd+=" --comment \"$comment\""
[ -n "$hidden_dim" ] && cmd+=" --hidden_dim $hidden_dim"
[ -n "$logdir" ] && cmd+=" --logdir $logdir"
[ -n "$last_epoch" ] && cmd+=" --last_epoch $last_epoch"
[ -n "$shuffle" ] && cmd+=" --shuffle $shuffle"

# Generate Slurm script as a string
slurm_script=$(cat << EOM
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --job-name="$job_name"
#SBATCH --time=$time_limit
#SBATCH --mem=${memory_gb}G
#SBATCH --output=logs/${job_name}.out


source ~/.e/bin/activate
module load cudnn

$cmd
EOM
)

# Submit the Slurm job by piping the script to sbatch
echo "$slurm_script" | sbatch
