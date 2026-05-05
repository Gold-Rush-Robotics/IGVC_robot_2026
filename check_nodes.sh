#!/bin/bash

echo "Checking use_sim_time for all ROS 2 nodes..."
echo "-------------------------------------------"

# Timeout in seconds for each parameter query (override with first arg)
QUERY_TIMEOUT="${1:-5}"

# Get list of all nodes (dedupe names to avoid repeated checks)
nodes=$(ros2 node list 2>/dev/null | sort -u)

if [ -z "$nodes" ]; then
  echo "No nodes found."
  exit 1
fi

for node in $nodes; do
  # Some nodes can block on parameter service calls; enforce a timeout.
  value=$(timeout "$QUERY_TIMEOUT" ros2 param get "$node" use_sim_time 2>/dev/null)
  status=$?

  if [[ $status -eq 0 ]]; then
    echo "[INFO] $node : $value"
  elif [[ $status -eq 124 ]]; then
    echo "[WARN] $node : timed out after ${QUERY_TIMEOUT}s"
  else
    echo "[WARN] $node : use_sim_time not declared or parameter service unavailable"
  fi
done

echo "-------------------------------------------"
echo "Done."