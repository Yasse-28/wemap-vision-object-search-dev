export const EARTH_RADIUS_M = 6378137;

/** Initial compass bearing (degrees, 0 = north, clockwise) from one lng/lat to another. */
export function bearingDegrees(
  from: { latitude: number; longitude: number },
  to: { latitude: number; longitude: number },
): number {
  const toRad = Math.PI / 180;
  const lat1 = from.latitude * toRad;
  const lat2 = to.latitude * toRad;
  const dLon = (to.longitude - from.longitude) * toRad;
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x =
    Math.cos(lat1) * Math.sin(lat2) -
    Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}

export function projectKeyframeToLocalFloor(
  source: {
    latitude: number;
    longitude: number;
    heading_deg: number;
  },
  destination: {
    latitude: number;
    longitude: number;
  },
): { localX: number; localZ: number; distanceM: number } {
  const latitudeRad = source.latitude * Math.PI / 180;
  const east =
    (destination.longitude - source.longitude) *
    Math.PI / 180 *
    EARTH_RADIUS_M *
    Math.cos(latitudeRad);
  const north =
    (destination.latitude - source.latitude) *
    Math.PI / 180 *
    EARTH_RADIUS_M;
  const distance = Math.hypot(east, north);
  const bearing = Math.atan2(east, north);
  const localYaw = bearing - source.heading_deg * Math.PI / 180;
  return {
    localX: distance * Math.sin(localYaw),
    localZ: -distance * Math.cos(localYaw),
    // distanceM is consumed only by the Explorer.
    distanceM: distance,
  };
}
