ssh leonl@hpc 'find /ceph/scratch/leonl/results/weights -type f -mtime -1 | xargs tar -czvf /ceph/scratch/leonl/results/last_day.tar.gz'
scp leon@hpc:/ceph/scratch/leonl/results/last_day.tar.gz /Users/leonlufkin/Documents/GitHub/Localization/localization/results/weights
