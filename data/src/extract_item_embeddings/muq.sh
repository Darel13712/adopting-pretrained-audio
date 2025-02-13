#!/bin/bash


checkpoint="msd"
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --checkpoint) checkpoint="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

job_name="muq_${checkpoint}"


cmd="python ./muq_emb.py"
[ -n "$checkpoint" ] && cmd+=" --checkpoint $checkpoint"

# Generate Slurm script as a string
slurm_script=$(cat << EOM
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --job-name="$job_name"
#SBATCH --time=24:00:00
#SBATCH --mem=12G
#SBATCH --output=${job_name}.out


source ~/.e/bin/activate
module load libsndfile
module load cudnn

$cmd
EOM
)

echo "$slurm_script" | sbatch
