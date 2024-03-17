type GeoJSON = {
  type: string;
  features: Feature[];
};

type Feature = {
  type: string;
  geometry: Geometry;
  properties: Properties;
};

type Geometry = {
  type: string;
  coordinates: number[];
};

type Properties = {
  [key: string]: any;
};

type KML = {
  type: string;
  features: Placemark[];
};

type Placemark = {
  type: string;
  geometry: Geometry;
  properties: Properties;
};

type WKT = {
  type: string;
  features: Feature[];
};

type GeoPackage = {
  type: string;
  features: Feature[];
};
