var DRIVE_FOLDER = "S2_ML_Dataset_V2";
var SCALE = 10;
var START_DATE = "2023-05-01";
var END_DATE = "2023-08-28";
var locations = [
  { name: "CairoUniv", lon: 31.20909, lat: 30.02633 },
  { name: 'CairoDowntown', lon: 31.2357, lat: 30.0444},
  { name: "NaserCity", lon: 31.33, lat: 30.06 },
  { name: "IconicTower", lon: 31.7018, lat: 30.0109 },
  { name: "AlexandriaPort", lon: 29.8954, lat: 31.2001 },
  { name: "MansouraCity", lon: 31.3667, lat: 31.0333 },
  { name: "HelwanIndustrial", lon: 31.3342, lat: 29.85 },
  { name: "TantaCity", lon: 30.7987, lat: 30.7865 },
  { name: "PortSaidCity", lon: 32.2833, lat: 31.2667 },
  { name: "ZagazigCity", lon: 31.502, lat: 30.5877 },
  { name: "SuezCity", lon: 32.5498, lat: 29.9668 },
  { name: "SiwaOasis", lon: 25.5195, lat: 29.2032 },
  { name: "HawaraFayoum", lon: 30.8986, lat: 29.2747 },
  { name: "EasternDesert", lon: 32.5, lat: 27.5 },
  { name: "WhiteDesert", lon: 28.3, lat: 27.95 },
  { name: "DakhlaOasis", lon: 28.99, lat: 25.49 },
  { name: "GizaPlateau", lon: 31.1342, lat: 29.9792 },
  { name: "QattaraDepression", lon: 27.0, lat: 29.5 },
  { name: "WadiRum", lon: 33.5, lat: 25.0 },
  { name: "SinaiDesert", lon: 33.8, lat: 29.5 },
  { name: "LibyanBorder", lon: 24.5, lat: 29.0 },
  { name: "NileDelta", lon: 31.1, lat: 30.8 },
  { name: "FayoumFarms", lon: 30.84, lat: 29.31 },
  { name: "KarnakLuxor", lon: 32.6583, lat: 25.718 },
  { name: "AssiutFarms", lon: 31.18, lat: 27.18 },
  { name: "ZagazigFarms", lon: 31.5, lat: 30.58 },
  { name: "BeheiraNileDelta", lon: 30.35, lat: 30.85 },
  { name: "MiniaFarms", lon: 30.75, lat: 28.1 },
  { name: "SohagFarms", lon: 31.69, lat: 26.55 },
  { name: "QenaFarms", lon: 32.72, lat: 26.16 },
  { name: "DamietaFarms", lon: 31.8, lat: 31.42 },
  { name: "SaddElAly", lon: 32.88098 , lat: 23.95591 },
  { name: "NasserLake", lon: 32.87, lat: 23.0 },
  { name: "ManzalaLake", lon: 32.0, lat: 31.2 },
  { name: "SuezCanal", lon: 32.55, lat: 30.5 },
  { name: "NileCairo", lon: 31.224, lat: 30.06 },
  { name: "MariutLake", lon: 29.75, lat: 31.15 },
  { name: "QarunLake", lon: 30.55, lat: 29.48 },
  { name: 'FarafraOasis',lon: 27.9700,  lat: 27.0600},  
  { name: 'BaharyaOasis', lon: 28.8391,  lat: 28.410},  
  { name: 'KhargaOasis', lon: 30.5026,  lat: 25.44751},  
  { name: 'WadiElGemal', lon: 34.759845,  lat:24.5387},  
  { name: 'WadiNatrun', lon: 30.4116,  lat: 30.3992},  
  { name: 'AbuSimbelDesert',lon: 31.6257,  lat: 22.3372},  
  { name: 'RedSeaHurghada', lon: 33.8500,  lat: 27.1800},  
  { name: 'GulfAqaba', lon: 34.9350,  lat: 29.5000},  
  { name: 'GulfSuez', lon: 32.524,  lat: 29.431},     
  { name: 'TimsahLake', lon: 32.3000,  lat: 30.5800},  
  { name: 'GreatBitterLake', lon: 32.3500,  lat: 30.3300},  
  { name: 'BardawilLake', lon: 33.160,  lat: 31.1540}, 
];

function maskClouds(image) {
  var scl = image.select("SCL");
  var mask = scl.eq(4).or(scl.eq(5)).or(scl.eq(6));
  return image.updateMask(mask);
}

function getESRIMask(region) {
  return ee.ImageCollection(
      'projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS')
    .filterBounds(region)
    .filter(ee.Filter.calendarRange(2023, 2023, 'year'))  // ← keep 2023
    .first()
    .clip(region)
    .select('b1')
    .remap([1, 2, 4, 5, 7, 8, 9, 10, 11],
           [3, 1, 1, 1, 4, 2, 0, 0, 1])
    .rename('mask')
    .toByte();
}

// function getDWMask(region) {
//   return ee
//     .ImageCollection("GOOGLE/DYNAMICWORLD/V1")
//     .filterBounds(region)
//     .filterDate(START_DATE, END_DATE)
//     .mode()
//     .clip(region)
//     .select("label")
//     .remap([0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 1, 1, 1, 1, 1, 4, 2, 0])
//     .rename("mask")
//     .toByte();
// }

// MAIN LOOP
locations.forEach(function (loc) {
  var region = ee.Geometry.Point([loc.lon, loc.lat]).buffer(3000).bounds();

  var s2 = ee
    .ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(region)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    .map(maskClouds)
    .select([
      "B1",
      "B2",
      "B3",
      "B4",
      "B5",
      "B6",
      "B7",
      "B8",
      "B8A",
      "B9",
      "B11",
      "B12",
    ])
    .median()
    .clip(region);

  var rgb = s2
    .select(["B4", "B3", "B2"])
    .divide(3000)
    .multiply(255)
    .clamp(0, 255)
    .toByte();

  var esriMask = getESRIMask(region);
  // var dwMask = getDWMask(region);

  Export.image.toDrive({
    image: s2,
    description: loc.name + "_12band",
    folder: DRIVE_FOLDER,
    scale: SCALE,
    region: region,
    maxPixels: 1e9,
    fileFormat: "GeoTIFF",
  });

  Export.image.toDrive({
    image: esriMask,
    description: loc.name + "_mask_ESRI",
    folder: DRIVE_FOLDER,
    scale: SCALE,
    region: region,
    maxPixels: 1e9,
    fileFormat: "GeoTIFF",
  });

  // Export.image.toDrive({
  //   image: dwMask,
  //   description: loc.name + "_mask_DW",
  //   folder: DRIVE_FOLDER,
  //   scale: SCALE,
  //   region: region,
  //   maxPixels: 1e9,
  //   fileFormat: "GeoTIFF",
  // });

  Export.image.toDrive({
    image: rgb,
    description: loc.name + "_RGB",
    folder: DRIVE_FOLDER,
    scale: SCALE,
    region: region,
    maxPixels: 1e9,
    fileFormat: "GeoTIFF",
  });

  print("Tasks ready: " + loc.name);
});

print("All tasks created! Go to Tasks tab and click Run on each.");
