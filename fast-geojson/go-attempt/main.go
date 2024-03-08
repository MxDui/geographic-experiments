package main

import (
	"fmt"
	"github.com/valyala/fastjson"
)

func main() {
	// Replace `geojsonData` with your actual GeoJSON data
	geojsonData := []byte(`{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Point","coordinates":[1.0,2.0]},"properties":{}}]}`)

	// Parse the GeoJSON data using fastjson
	parsed, err := fastjson.ParseBytes(geojsonData)
	if err != nil {
		fmt.Println("Error parsing GeoJSON:", err)
		return
	}

	// Access the parsed GeoJSON data
	features := parsed.GetArray("features")
	for _, feature := range features {
		geometry := feature.Get("geometry")
		coordinates := geometry.GetArray("coordinates")
		lat := coordinates[0].GetFloat64()
		lng := coordinates[1].GetFloat64()

		fmt.Printf("Latitude: %f, Longitude: %f\n", lat, lng)
	}
}