# ping_location

ROS 2 node for fetching GPS data from remote phone endpoints and computing the midpoint location.

## Overview

This package provides a ROS 2 node that:
- Connects to two remote GPS endpoints via HTTP
- Retrieves GPS coordinates (latitude, longitude, and optionally direction)
- Computes the midpoint between the two locations
- Publishes all three as standard `sensor_msgs/NavSatFix` messages

## Published Topics

- `ping_location/phone1` (`sensor_msgs/NavSatFix`) - GPS data from phone 1
- `ping_location/phone2` (`sensor_msgs/NavSatFix`) - GPS data from phone 2
- `ping_location/midpoint` (`sensor_msgs/NavSatFix`) - Computed midpoint between the two phones

## Parameters

- `phone_1_ip` (string, default: `192.168.0.162`) - IP address of the first GPS endpoint
- `phone_2_ip` (string, default: `192.168.0.164`) - IP address of the second GPS endpoint
- `port` (integer, default: `8080`) - Port number for the GPS endpoints
- `timeout` (integer, default: `5`) - HTTP request timeout in seconds
- `loop_rate` (double, default: `1.0`) - Update rate in Hz

## Usage

### Build

```bash
colcon build --packages-select ping_location
```

### Run

```bash
ros2 run ping_location ping_location_node
```

### Run with Custom Parameters

```bash
ros2 run ping_location ping_location_node \
  --ros-args \
  -p phone_1_ip:=192.168.1.100 \
  -p phone_2_ip:=192.168.1.101 \
  -p loop_rate:=2.0
```

### Monitor Output

```bash
ros2 topic echo /ping_location/midpoint
```

## Requirements

The GPS endpoints must provide data in the following HTTP format:

```
GET http://<ip>:<port>/get?lat&lon&dir

Response (JSON):
{
  "buffer": {
    "lat": {"buffer": [<latitude>]},
    "lon": {"buffer": [<longitude>]},
    "dir": {"buffer": [<direction_degrees>]}
  }
}
```

The `dir` (direction) field is optional.

## Notes

- The node logs connection errors and parsing issues at the DEBUG level
- When both phones connect successfully, only the midpoint is logged at INFO level
- Altitude is set to NaN as it is not provided by the GPS endpoints
- Position covariance is set to unknown
