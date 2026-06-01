"""Geographic helpers for GPS waypoint test nodes."""

from __future__ import annotations

import math

# WGS-84 ellipsoid parameters.
_WGS84_A = 6378137.0
_WGS84_E2 = 0.00669437999014
_WGS84_B = _WGS84_A * math.sqrt(1.0 - _WGS84_E2)
_WGS84_EP2 = _WGS84_E2 / (1.0 - _WGS84_E2)


def _ecef(lat_deg: float, lon_deg: float,
          altitude_m: float = 0.0) -> tuple[float, float, float]:
    """Geodetic latitude/longitude/altitude to ECEF meters."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (n + altitude_m) * cos_lat * math.cos(lon)
    y = (n + altitude_m) * cos_lat * math.sin(lon)
    z = (n * (1.0 - _WGS84_E2) + altitude_m) * sin_lat
    return x, y, z


def _ecef_to_gps(x_coord: float, y_coord: float,
                 z_coord: float) -> tuple[float, float, float]:
    """ECEF meters to geodetic latitude/longitude/altitude (Bowring)."""
    lon = math.atan2(y_coord, x_coord)
    p = math.hypot(x_coord, y_coord)
    theta = math.atan2(z_coord * _WGS84_A, p * _WGS84_B)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    lat = math.atan2(
        z_coord + _WGS84_EP2 * _WGS84_B * sin_t * sin_t * sin_t,
        p - _WGS84_E2 * _WGS84_A * cos_t * cos_t * cos_t)
    sin_lat = math.sin(lat)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    if abs(math.cos(lat)) > 1e-12:
        altitude = p / math.cos(lat) - n
    else:
        altitude = z_coord / sin_lat - n * (1.0 - _WGS84_E2)
    return math.degrees(lat), math.degrees(lon), altitude


def gps_to_enu(lat: float, lon: float, origin_lat: float,
               origin_lon: float) -> tuple[float, float]:
    """Convert WGS-84 latitude/longitude to local east/north meters."""
    x, y, z = _ecef(lat, lon)
    x0, y0, z0 = _ecef(origin_lat, origin_lon)
    dx = x - x0
    dy = y - y0
    dz = z - z0
    lat0 = math.radians(origin_lat)
    lon0 = math.radians(origin_lon)
    sin_lat0 = math.sin(lat0)
    cos_lat0 = math.cos(lat0)
    sin_lon0 = math.sin(lon0)
    cos_lon0 = math.cos(lon0)
    east = -sin_lon0 * dx + cos_lon0 * dy
    north = (-sin_lat0 * cos_lon0 * dx
             - sin_lat0 * sin_lon0 * dy
             + cos_lat0 * dz)
    return east, north


def enu_to_gps(east_m: float, north_m: float, origin_lat: float,
               origin_lon: float, up_m: float = 0.0,
               origin_altitude_m: float = 0.0) -> tuple[float, float, float]:
    """Convert local east/north/up meters to WGS-84 lat/lon/altitude."""
    x0, y0, z0 = _ecef(origin_lat, origin_lon, origin_altitude_m)
    lat0 = math.radians(origin_lat)
    lon0 = math.radians(origin_lon)
    sin_lat0 = math.sin(lat0)
    cos_lat0 = math.cos(lat0)
    sin_lon0 = math.sin(lon0)
    cos_lon0 = math.cos(lon0)
    dx = (-sin_lon0 * east_m
          - sin_lat0 * cos_lon0 * north_m
          + cos_lat0 * cos_lon0 * up_m)
    dy = (cos_lon0 * east_m
          - sin_lat0 * sin_lon0 * north_m
          + cos_lat0 * sin_lon0 * up_m)
    dz = cos_lat0 * north_m + sin_lat0 * up_m
    return _ecef_to_gps(x0 + dx, y0 + dy, z0 + dz)


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def yaw_from_quaternion_xyzw(quat_x: float, quat_y: float, quat_z: float,
                             quat_w: float) -> float:
    """Return planar yaw from an xyzw quaternion."""
    return math.atan2(
        2.0 * (quat_w * quat_z + quat_x * quat_y),
        1.0 - 2.0 * (quat_y * quat_y + quat_z * quat_z))
