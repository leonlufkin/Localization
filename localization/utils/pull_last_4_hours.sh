#!/bin/bash

# Run the SSH and SCP commands
ssh leonl@hpc 'find /ceph/scratch/leonl/results/weights -type f -mmin -240 | xargs tar -czvf /ceph/scratch/leonl/results/last_4_hours.tar.gz'
scp leonl@hpc:/ceph/scratch/leonl/results/last_4_hours.tar.gz /Users/leonlufkin/Documents/GitHub/Localization/localization/results/weights
tar -xzvf /Users/leonlufkin/Documents/GitHub/Localization/localization/results/weights/last_4_hours.tar.gz -C /Users/leonlufkin/Documents/GitHub/Localization/localization/results/weights/
mv /Users/leonlufkin/Documents/GitHub/Localization/localization/results/weights/ceph/scratch/leonl/results/weights/* /Users/leonlufkin/Documents/GitHub/Localization/localization/results/weights/
rm -r /Users/leonlufkin/Documents/GitHub/Localization/localization/results/weights/ceph
rm /Users/leonlufkin/Documents/GitHub/Localization/localization/results/weights/last_4_hours.tar.gz

