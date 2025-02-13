#!/bin/bash

# Default values
time_limit_hours=48
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
    --dataset) dataset="$2"; shift ;;
    --sample) sample_type="$2"; shift ;;
    --model) model="$2"; shift ;;
    --numneg) numneg="$2"; shift ;;
    --batch) batch="$2"; shift ;;
    --epochs) epochs="$2"; shift ;;
    --ifreeze) item_freeze="$2"; shift ;;
    --ufreeze) user_freeze="$2"; shift ;;
    --logdir) logdir="$2"; shift ;;
    --lastepoch) last_epoch="$2"; shift ;;
    --comment) comment="$2"; shift ;;
    --uinit) user_init="$2"; shift ;;
    --iunfreeze) item_dynamic_unfreeze="$2"; shift ;;
    --hiddim) hidden_dim="$2"; shift ;;
    --useconf) use_confidence="$2"; shift ;;
    --l2) l2="$2"; shift ;;
    --shuffle) shuffle="$2"; shift ;;
    --gpu) gpu="$2"; shift ;;
#    --k) k="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# Construct job name
job_name="${model}_${dataset}"
[ -n "$comment" ] && job_name="${job_name}_${comment}"

# Format time limit as a string
time_limit="${time_limit_hours}:00:00"


cmd="python ./train.py"
[ -n "$dataset" ] && cmd+=" --dataset $dataset"
[ -n "$model" ] && cmd+=" --model_name $model"
[ -n "$sample_type" ] && cmd+=" --sample_type $sample_type"
[ -n "$numneg" ] && cmd+=" --neg_samples $numneg"
[ -n "$batch" ] && cmd+=" --batch_size $batch"
[ -n "$epochs" ] && cmd+=" --num_epochs $epochs"
[ -n "$item_freeze" ] && cmd+=" --item_freeze $item_freeze"
[ -n "$user_freeze" ] && cmd+=" --user_freeze $user_freeze"
[ -n "$comment" ] && cmd+=" --comment \"$comment\""
[ -n "$user_init" ] && cmd+=" --user_init $user_init"
[ -n "$item_dynamic_unfreeze" ] && cmd+=" --dynamic_item_freeze $item_dynamic_unfreeze"
[ -n "$hidden_dim" ] && cmd+=" --hidden_dim $hidden_dim"
[ -n "$use_confidence" ] && cmd+=" --use_confidence $use_confidence"
[ -n "$l2" ] && cmd+=" --l2 $l2"
[ -n "$logdir" ] && cmd+=" --logdir $logdir"
[ -n "$last_epoch" ] && cmd+=" --last_epoch $last_epoch"
[ -n "$shuffle" ] && cmd+=" --shuffle $shuffle"

g="gpu"
[ -n "$gpu" ] && g+=":$gpu"

# Generate Slurm script as a string
slurm_script=$(cat << EOM
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=$g:1
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
