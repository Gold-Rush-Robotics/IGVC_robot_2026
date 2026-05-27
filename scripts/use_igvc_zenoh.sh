#!/usr/bin/env bash

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export ROS_LOCALHOST_ONLY=0
export ZENOH_ROUTER_CONFIG_URI="$workspace_root/src/igvc_test_bringup/config/zenoh_config.json5"

echo "IGVC Zenoh environment configured:"
echo "  RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "  ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "  ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"
echo "  ZENOH_ROUTER_CONFIG_URI=$ZENOH_ROUTER_CONFIG_URI"
