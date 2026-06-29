#!/bin/bash
NFS_ENTRYPOINT="/mnt/nlp/scratch/home/lyczek/dyn_entrypoint.sh"
if [ -f "$NFS_ENTRYPOINT" ]; then
  exec "$NFS_ENTRYPOINT" "$@"
else
  echo "No entrypoint found at $NFS_ENTRYPOINT"
  exec "$@"
fi