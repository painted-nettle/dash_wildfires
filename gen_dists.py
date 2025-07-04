import csv
import random
import datetime
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import math
import requests
import zipfile
import os
import time
import json
from dataclasses import dataclass
from scipy import stats
import hashlib

# Alternative way to set API key in Jupyter (uncomment if needed):
os.environ["GEMINI_API_KEY"] = "xxx"

# For Gemini API integration
try:
    from google import genai
    from google.genai import types

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Gemini API not available. Install with: pip install google-genai")

SCRIPT_DIR = Path(__file__).resolve().parent


class GeoNamesSwedishGeocoder:
    """Geocoder using GeoNames data for Swedish locations"""

    def __init__(self):
        self.cities_data = {}
        self.geonames_file_path = SCRIPT_DIR / "SE.txt"
        self.geonames_filename = "SE.txt"  # For use with zip extraction and messages
        self.geonames_url = "http://download.geonames.org/export/dump/SE.zip"
        self.loaded = False

    def download_geonames_data(self) -> bool:
        """Download and extract Swedish GeoNames data"""
        if os.path.exists(self.geonames_file_path):
            print(f"  ✓ {self.geonames_filename} already exists in {SCRIPT_DIR}")
            return True

        print(f"  📥 Downloading Swedish GeoNames data to {SCRIPT_DIR}...")

        try:
            response = requests.get(self.geonames_url, stream=True)
            response.raise_for_status()

            zip_download_path = SCRIPT_DIR / "SE.zip"
            with open(zip_download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print("  📦 Extracting geographic data...")

            with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
                # Assuming SE.txt is at the root of the zip file
                zip_ref.extract(self.geonames_filename, path=SCRIPT_DIR)

            os.remove(zip_download_path)

            if os.path.exists(self.geonames_file_path):
                print(
                    f"  ✓ Downloaded and extracted {self.geonames_filename} to {SCRIPT_DIR}"
                )
                return True
            else:
                print(
                    f"  ✗ Failed to extract {self.geonames_filename} from zip file to {SCRIPT_DIR}"
                )
                return False

        except Exception as e:
            print(f"  ✗ Error downloading GeoNames data: {e}")
            return False

    def load_geonames_data(self) -> bool:
        """Load GeoNames data into memory"""
        if self.loaded:
            return True

        if not os.path.exists(self.geonames_file_path):
            print(
                f"  ℹ {self.geonames_filename} not found in {SCRIPT_DIR}. Downloading..."
            )
            if not self.download_geonames_data():
                return False

        print(f"  📊 Loading geographic data from {self.geonames_file_path}...")

        try:
            column_names = [
                "geonameid",
                "name",
                "asciiname",
                "alternatenames",
                "latitude",
                "longitude",
                "feature_class",
                "feature_code",
                "country_code",
                "cc2",
                "admin1_code",
                "admin2_code",
                "admin3_code",
                "admin4_code",
                "population",
                "elevation",
                "dem",
                "timezone",
                "modification_date",
            ]

            # Read the data
            with open(self.geonames_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 15:
                        try:
                            feature_class = parts[6]
                            if feature_class == "P":  # Populated places
                                name = parts[1]
                                asciiname = parts[2]
                                alternatenames = parts[3]
                                lat = float(parts[4])
                                lon = float(parts[5])
                                population = int(parts[14]) if parts[14] else 0

                                coords = (lat, lon)
                                data = {"coords": coords, "population": population}

                                # Store main name
                                self.cities_data[name.lower().strip()] = data

                                # Store ASCII name
                                if asciiname and asciiname != name:
                                    self.cities_data[asciiname.lower().strip()] = data

                                # Store alternate names
                                if alternatenames:
                                    for alt_name in alternatenames.split(","):
                                        alt_name = alt_name.lower().strip()
                                        if alt_name and len(alt_name) > 1:
                                            self.cities_data[alt_name] = data
                        except (ValueError, IndexError):
                            continue

            print(f"  ✓ Loaded {len(self.cities_data):,} place names")

            major_cities = sum(
                1 for data in self.cities_data.values() if data["population"] > 50000
            )
            print(f"  ✓ Including {major_cities} major cities (>50k population)")

            self.loaded = True
            return True

        except Exception as e:
            print(f"  ✗ Error loading GeoNames data: {e}")
            return False

    def add_coordinate_variation(
        self, lat: float, lon: float, radius_km: float = 3.0
    ) -> Tuple[float, float]:
        """Add random variation to coordinates within a specified radius"""
        lat_variation = radius_km / 111.0
        lon_variation = radius_km / (111.0 * math.cos(math.radians(lat)))

        lat_offset = random.uniform(-lat_variation, lat_variation)
        lon_offset = random.uniform(-lon_variation, lon_variation)

        return (lat + lat_offset, lon + lon_offset)

    def geocode_location(
        self, city: str, neighborhood: str = None
    ) -> Optional[Tuple[float, float]]:
        """Geocode a city/neighborhood combination using GeoNames data"""
        if not self.loaded:
            return None

        city_clean = city.lower().strip()

        # Try exact match first
        city_data = self.cities_data.get(city_clean)

        if not city_data:
            # Try without special characters
            city_normalized = (
                city_clean.replace("ö", "o").replace("ä", "a").replace("å", "a")
            )
            city_data = self.cities_data.get(city_normalized)

        if not city_data:
            # Try partial matching
            for name in self.cities_data.keys():
                if city_clean in name or name in city_clean:
                    city_data = self.cities_data[name]
                    break

        if city_data:
            base_coords = city_data["coords"]

            if neighborhood:
                # Consistent variation based on neighborhood
                random.seed(hash(f"{city}_{neighborhood}"))
                varied_coords = self.add_coordinate_variation(
                    base_coords[0], base_coords[1]
                )
                random.seed()  # Reset seed
                return varied_coords
            else:
                return self.add_coordinate_variation(
                    base_coords[0], base_coords[1], radius_km=1.0
                )

        return None


class EnhancedCitySelector:
    """Enhanced city selector using GeoNames data for realistic Swedish location distribution"""

    def __init__(self, geocoder):
        self.geocoder = geocoder
        self.all_cities = []
        self.city_details = {}
        self.weighted_cities = []
        self._process_geonames_data()

    def _process_geonames_data(self):
        """Process GeoNames data to create weighted city selection"""
        if not self.geocoder.cities_data:
            print("⚠️ No GeoNames data loaded")
            return

        # Group by unique locations to avoid duplicates
        unique_locations = {}

        for name, data in self.geocoder.cities_data.items():
            coords = data["coords"]
            pop = data["population"]

            # Create unique key based on coordinates (rounded to avoid near-duplicates)
            coord_key = (round(coords[0], 2), round(coords[1], 2))

            if (
                coord_key not in unique_locations
                or pop > unique_locations[coord_key]["population"]
            ):
                unique_locations[coord_key] = {
                    "name": name.title(),
                    "coords": coords,
                    "population": pop,
                    "names": set(),
                }
            unique_locations[coord_key]["names"].add(name)

        # Convert to list and calculate weights
        for location in unique_locations.values():
            # Skip locations with very low or no population data
            if location["population"] < 50:
                continue

            city_info = {
                "name": location["name"],
                "coords": location["coords"],
                "lat": location["coords"][0],
                "lon": location["coords"][1],
                "population": location["population"],
                "alternative_names": list(location["names"]),
            }

            # Classify city type based on population
            if location["population"] >= 100000:
                city_info["type"] = "major_city"
                city_info["urban"] = True
            elif location["population"] >= 50000:
                city_info["type"] = "city"
                city_info["urban"] = True
            elif location["population"] >= 20000:
                city_info["type"] = "large_town"
                city_info["urban"] = True
            elif location["population"] >= 5000:
                city_info["type"] = "town"
                city_info["urban"] = False
            elif location["population"] >= 1000:
                city_info["type"] = "small_town"
                city_info["urban"] = False
            else:
                city_info["type"] = "village"
                city_info["urban"] = False

            # Calculate fire risk based on location (simplified model)
            # Northern locations have higher forest fire risk
            if city_info["lat"] > 65:  # Far north
                city_info["fire_risk"] = 1.4
            elif city_info["lat"] > 62:  # North
                city_info["fire_risk"] = 1.2
            elif city_info["lat"] > 59:  # Central
                city_info["fire_risk"] = 1.0
            else:  # South
                city_info["fire_risk"] = 0.7

            # Adjust fire risk based on proximity to coast (simplified)
            # Coastal areas generally have lower fire risk
            if abs(city_info["lon"] - 12.5) < 1 or abs(city_info["lon"] - 18) < 1:
                city_info["fire_risk"] *= 0.8

            # Calculate base weight for selection
            # Use log scale for population to avoid too much bias toward large cities
            pop_weight = np.log10(location["population"] + 1)

            # Adjust weight based on campaign focus
            city_info["base_weight"] = pop_weight

            self.all_cities.append(city_info)
            self.city_details[city_info["name"]] = city_info

        print(
            f"✅ Processed {len(self.all_cities)} unique Swedish locations from GeoNames"
        )

        # Print distribution
        type_counts = defaultdict(int)
        for city in self.all_cities:
            type_counts[city["type"]] += 1

        print("📊 Location distribution:")
        for city_type, count in sorted(type_counts.items()):
            print(f"   • {city_type}: {count}")

    def get_weighted_cities_for_campaign(
        self, campaign_type: str, org_type: str, focus_regions: List[str] = None
    ) -> List[Tuple[Dict, float]]:
        """Get cities with weights adjusted for specific campaign"""
        weighted_cities = []

        for city in self.all_cities:
            weight = city["base_weight"]

            # Adjust weight based on campaign type
            if campaign_type == "emergency_response":
                # Prioritize high-risk areas and populated areas
                weight *= city["fire_risk"]
                weight *= 1.5 if city["population"] > 10000 else 1.0

            elif campaign_type == "intensive_prevention":
                # Balance between risk and population
                weight *= city["fire_risk"] ** 0.5
                weight *= (city["population"] / 10000) ** 0.3

            elif campaign_type == "seasonal_preparation":
                # Focus on suburban and rural areas
                if city["type"] in ["town", "small_town"]:
                    weight *= 1.5
                elif city["type"] == "village":
                    weight *= 1.2

            elif campaign_type == "community_outreach":
                # Focus on smaller communities
                if city["population"] < 5000:
                    weight *= 2.0
                elif city["population"] < 20000:
                    weight *= 1.5

            # Adjust based on organization type
            if org_type == "Municipal Fire Department":
                # Prefer local area (would need to implement distance calculation)
                pass
            elif org_type == "Environmental NGO":
                # Prefer areas near forests (higher latitude as proxy)
                if city["lat"] > 60:
                    weight *= 1.3
            elif org_type == "Community Group":
                # Strong preference for smaller communities
                if city["population"] < 10000:
                    weight *= 2.0

            # Regional focus (if specified)
            if focus_regions:
                # Simple latitude-based regions
                if "north" in focus_regions and city["lat"] > 63:
                    weight *= 2.0
                elif "central" in focus_regions and 58 < city["lat"] <= 63:
                    weight *= 2.0
                elif "south" in focus_regions and city["lat"] <= 58:
                    weight *= 2.0

            weighted_cities.append((city, weight))

        # Normalize weights
        total_weight = sum(w for _, w in weighted_cities)
        if total_weight > 0:
            weighted_cities = [(city, w / total_weight) for city, w in weighted_cities]

        return weighted_cities

    def select_city(
        self,
        campaign_type: str = "routine_awareness",
        org_type: str = "Government Agency",
        focus_regions: List[str] = None,
    ) -> Dict:
        """Select a city based on campaign parameters"""
        weighted_cities = self.get_weighted_cities_for_campaign(
            campaign_type, org_type, focus_regions
        )

        if not weighted_cities:
            # Fallback to random city if no weighted cities
            return random.choice(self.all_cities) if self.all_cities else None

        cities, weights = zip(*weighted_cities)
        selected_city = np.random.choice(cities, p=weights)

        return selected_city

    def get_nearby_cities(self, center_city: str, radius_km: float = 50) -> List[Dict]:
        """Get cities within a certain radius of a center city"""
        if center_city not in self.city_details:
            return []

        center = self.city_details[center_city]
        nearby = []

        for city in self.all_cities:
            if city["name"] == center_city:
                continue

            # Calculate distance (simplified)
            dist = self._calculate_distance_km(
                center["lat"], center["lon"], city["lat"], city["lon"]
            )

            if dist <= radius_km:
                nearby.append({**city, "distance_km": dist})

        return sorted(nearby, key=lambda x: x["distance_km"])

    def _calculate_distance_km(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in kilometers"""
        R = 6371  # Earth's radius in kilometers

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)

        a = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def get_cities_by_type(self, city_types: List[str]) -> List[Dict]:
        """Get all cities of specific types"""
        return [city for city in self.all_cities if city["type"] in city_types]

    def get_cities_by_population(
        self, min_pop: int = 0, max_pop: int = float("inf")
    ) -> List[Dict]:
        """Get cities within a population range"""
        return [
            city for city in self.all_cities if min_pop <= city["population"] <= max_pop
        ]


@dataclass
class NeighborhoodProfile:
    """Realistic neighborhood characteristics that affect demographics"""

    name: str
    city: str
    socioeconomic_class: (
        str  # 'upper', 'upper-middle', 'middle', 'lower-middle', 'lower'
    )
    family_friendliness: float  # 0-1 score
    urbanization: str  # 'urban-core', 'urban', 'suburban', 'rural'
    avg_income_multiplier: float
    avg_age: float
    single_household_rate: float
    car_ownership_modifier: float
    property_value_modifier: float
    ethnic_diversity: float  # 0-1 score
    student_population: float  # 0-1 score
    elderly_population: float  # 0-1 score
    coordinates: Tuple[float, float]


class GeminiEnhancer:
    """Use Gemini API to generate ultra-realistic content"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        if GEMINI_AVAILABLE and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self.model = "gemini-2.0-flash-exp"
            print("✅ Gemini API initialized for enhanced realism")
        else:
            print("⚠️  Gemini API not configured - using fallback methods")

    def generate_campaign_narrative(
        self, org_type: str, season: str, year: int
    ) -> Dict:
        """Generate realistic campaign narrative that affects distribution patterns"""
        if not self.client:
            return self._fallback_campaign_narrative(org_type, season, year)

        prompt = f"""Generate a realistic wildfire prevention campaign narrative for a {org_type} in Sweden.
Season: {season} {year}
Output as JSON with these fields:
- trigger_event: What prompted this campaign (news event, weather pattern, etc)
- urgency_level: 1-10 scale
- target_demographics: List of primary target groups
- geographic_focus: Specific regions or areas of concern
- distribution_strategy: How leaflets should be distributed (concentrated bursts, steady flow, etc)
- collaboration_partners: Types of organizations that would collaborate
- expected_duration_days: How long the campaign would run
- daily_volume_pattern: 'increasing', 'decreasing', 'steady', 'burst'
Be specific and realistic to Swedish context."""

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.8,
                ),
            )

            return json.loads(response.text)
        except Exception as e:
            print(f"⚠️  Gemini API error: {e}")
            return self._fallback_campaign_narrative(org_type, season, year)

    def _fallback_campaign_narrative(
        self, org_type: str, season: str, year: int
    ) -> Dict:
        """Fallback campaign narrative when Gemini is not available"""
        narratives = {
            "summer": {
                "trigger_event": "Extended dry period and heat wave forecast",
                "urgency_level": random.randint(7, 10),
                "target_demographics": ["rural_residents", "forest_owners", "campers"],
                "geographic_focus": ["forest_areas", "camping_sites"],
                "distribution_strategy": "concentrated_burst",
                "expected_duration_days": random.randint(30, 90),
                "daily_volume_pattern": "burst",
            },
            "spring": {
                "trigger_event": "Early snowmelt and dry vegetation",
                "urgency_level": random.randint(5, 8),
                "target_demographics": ["homeowners", "gardeners"],
                "geographic_focus": ["suburban_areas", "rural_communities"],
                "distribution_strategy": "steady_flow",
                "expected_duration_days": random.randint(21, 45),
                "daily_volume_pattern": "steady",
            },
        }

        base = narratives.get(season, narratives["summer"])
        base["collaboration_partners"] = ["Municipal Fire Department", "Forest Service"]
        return base


class EnhancedGeoNamesSwedishGeocoder:
    """Enhanced geocoder with neighborhood-level precision and clustering"""

    def __init__(self, base_geocoder):
        self.base_geocoder = base_geocoder
        self.neighborhood_profiles = {}
        self.coordinate_clusters = defaultdict(list)

    def create_neighborhood_profile(
        self, city: str, neighborhood: str
    ) -> NeighborhoodProfile:
        """Create a realistic neighborhood profile with clustered characteristics"""

        # Hash neighborhood name for consistent randomness
        seed = int(hashlib.md5(f"{city}_{neighborhood}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)

        # Determine neighborhood type based on name patterns
        if any(x in neighborhood.lower() for x in ["centrum", "city", "innerstan"]):
            socioeconomic_class = rng.choice(
                ["upper", "upper-middle", "middle"], p=[0.2, 0.5, 0.3]
            )
            urbanization = "urban-core"
            family_friendliness = rng.uniform(0.2, 0.5)
            single_household_rate = rng.uniform(0.5, 0.8)
        elif any(x in neighborhood.lower() for x in ["villa", "träd", "park"]):
            socioeconomic_class = rng.choice(
                ["upper-middle", "middle", "upper"], p=[0.5, 0.3, 0.2]
            )
            urbanization = "suburban"
            family_friendliness = rng.uniform(0.7, 0.95)
            single_household_rate = rng.uniform(0.1, 0.3)
        elif any(x in neighborhood.lower() for x in ["hamn", "industri"]):
            socioeconomic_class = rng.choice(
                ["lower-middle", "lower", "middle"], p=[0.5, 0.3, 0.2]
            )
            urbanization = "urban"
            family_friendliness = rng.uniform(0.3, 0.6)
            single_household_rate = rng.uniform(0.4, 0.7)
        else:
            # Default mixed neighborhood
            socioeconomic_class = rng.choice(
                ["middle", "lower-middle", "upper-middle"], p=[0.5, 0.3, 0.2]
            )
            urbanization = rng.choice(["urban", "suburban"], p=[0.6, 0.4])
            family_friendliness = rng.uniform(0.4, 0.8)
            single_household_rate = rng.uniform(0.2, 0.6)

        # Income multipliers based on socioeconomic class
        income_multipliers = {
            "upper": rng.uniform(1.8, 2.5),
            "upper-middle": rng.uniform(1.3, 1.8),
            "middle": rng.uniform(0.9, 1.3),
            "lower-middle": rng.uniform(0.7, 0.9),
            "lower": rng.uniform(0.5, 0.7),
        }

        # Age patterns based on neighborhood type
        if "student" in neighborhood.lower() or rng.random() < 0.1:
            avg_age = rng.uniform(22, 30)
            student_population = rng.uniform(0.6, 0.9)
            elderly_population = rng.uniform(0.0, 0.1)
        elif family_friendliness > 0.7:
            avg_age = rng.uniform(38, 45)
            student_population = rng.uniform(0.0, 0.1)
            elderly_population = rng.uniform(0.1, 0.2)
        elif single_household_rate > 0.6:
            avg_age = rng.uniform(32, 40)
            student_population = rng.uniform(0.1, 0.3)
            elderly_population = rng.uniform(0.1, 0.2)
        else:
            avg_age = rng.uniform(40, 55)
            student_population = rng.uniform(0.0, 0.1)
            elderly_population = rng.uniform(0.2, 0.4)

        # Get base coordinates
        coords = self.base_geocoder.geocode_location(city, neighborhood)
        if not coords:
            coords = (59.3293, 18.0686)  # Default to Stockholm

        profile = NeighborhoodProfile(
            name=neighborhood,
            city=city,
            socioeconomic_class=socioeconomic_class,
            family_friendliness=family_friendliness,
            urbanization=urbanization,
            avg_income_multiplier=income_multipliers[socioeconomic_class],
            avg_age=avg_age,
            single_household_rate=single_household_rate,
            car_ownership_modifier=0.3 if urbanization == "urban-core" else 1.0,
            property_value_modifier=income_multipliers[socioeconomic_class]
            * (1.2 if urbanization == "urban-core" else 1.0),
            ethnic_diversity=rng.uniform(0.5, 1.0)
            if urbanization in ["urban-core", "urban"]
            else rng.uniform(0.0, 0.3),
            student_population=student_population,
            elderly_population=elderly_population,
            coordinates=coords,
        )

        return profile

    def get_neighborhood_coordinates(
        self, city: str, neighborhood: str
    ) -> Tuple[float, float]:
        """Get coordinates with realistic clustering within neighborhoods"""
        profile_key = f"{city}_{neighborhood}"

        if profile_key not in self.neighborhood_profiles:
            self.neighborhood_profiles[profile_key] = self.create_neighborhood_profile(
                city, neighborhood
            )

        profile = self.neighborhood_profiles[profile_key]
        base_lat, base_lon = profile.coordinates

        # Create realistic clustering within neighborhood
        # Streets typically run in patterns
        street_angle = random.uniform(0, math.pi)
        street_number = random.randint(1, 20)
        house_number = random.randint(1, 100)

        # Calculate offset based on street grid
        street_offset = 0.001 * street_number
        house_offset = 0.00001 * house_number

        lat_offset = street_offset * math.cos(street_angle) + house_offset * math.sin(
            street_angle
        )
        lon_offset = street_offset * math.sin(street_angle) + house_offset * math.cos(
            street_angle
        )

        # Add small random variation (few meters)
        lat_offset += random.uniform(-0.00001, 0.00001)
        lon_offset += random.uniform(-0.00001, 0.00001)

        return (round(base_lat + lat_offset, 6), round(base_lon + lon_offset, 6))


class UltraRealisticDistributionGenerator:
    """Enhanced generator with ultra-realistic patterns"""

    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)

        # Initialize enhanced geocoder
        base_geocoder = GeoNamesSwedishGeocoder()
        self.geocoder = EnhancedGeoNamesSwedishGeocoder(base_geocoder)

        # Initialize Gemini enhancer
        self.gemini = GeminiEnhancer()

        # Enhanced Swedish cities with more realistic data
        self.swedish_cities = {
            "Stockholm": {
                "weight": 25,
                "fire_risk": 0.6,
                "urban": True,
                "lat": 59.3293,
                "lon": 18.0686,
                "postal_pattern": [
                    "10",
                    "11",
                    "12",
                    "13",
                    "14",
                    "15",
                    "16",
                    "17",
                    "18",
                    "19",
                ],
                "income_distribution": {"mean": 480000, "std": 250000, "skew": 1.5},
                "age_distribution": {"young": 0.3, "middle": 0.45, "elderly": 0.25},
                "neighborhood_types": {
                    "affluent": [
                        "Östermalm",
                        "Djursholm",
                        "Lidingö",
                        "Saltsjöbaden",
                        "Bromma Villastad",
                        "Danderyd",
                        "Täby Centrum",
                        "Stocksund",
                        "Näsbypark",
                    ],
                    "middle": [
                        "Vasastan",
                        "Kungsholmen",
                        "Södermalm",
                        "Bromma",
                        "Hägersten",
                        "Enskede",
                        "Älvsjö",
                        "Farsta",
                        "Vällingby",
                        "Hässelby",
                        "Spånga",
                        "Sundbyberg",
                    ],
                    "diverse": [
                        "Rinkeby",
                        "Tensta",
                        "Husby",
                        "Skärholmen",
                        "Fittja",
                        "Alby",
                        "Hallunda",
                        "Jordbro",
                        "Brandbergen",
                        "Hagsätra",
                        "Rågsved",
                        "Vårby",
                    ],
                    "student": [
                        "Frescati",
                        "Lappkärrsberget",
                        "Flemingsberg",
                        "KTH Campus",
                    ],
                    "family": [
                        "Vällingby",
                        "Blackeberg",
                        "Enskede",
                        "Stureby",
                        "Tallkrogen",
                        "Bandhagen",
                    ],
                },
            },
            "Göteborg": {
                "weight": 12,
                "fire_risk": 0.7,
                "urban": True,
                "lat": 57.7089,
                "lon": 11.9746,
                "postal_pattern": ["40", "41", "42", "43", "44"],
                "income_distribution": {"mean": 420000, "std": 180000, "skew": 1.2},
                "age_distribution": {"young": 0.28, "middle": 0.47, "elderly": 0.25},
                "neighborhood_types": {
                    "affluent": [
                        "Örgryte",
                        "Hovås",
                        "Askim",
                        "Långedrag",
                        "Särö",
                        "Kullavik",
                        "Billdal",
                    ],
                    "middle": [
                        "Majorna",
                        "Linnéstaden",
                        "Johanneberg",
                        "Vasastaden",
                        "Haga",
                        "Guldheden",
                    ],
                    "diverse": [
                        "Angered",
                        "Bergsjön",
                        "Hjällbo",
                        "Hammarkullen",
                        "Lövgärdet",
                        "Gårdsten",
                    ],
                    "student": [
                        "Chalmers",
                        "Olofshöjd",
                        "Gibraltar",
                        "Medicinareberget",
                    ],
                    "family": [
                        "Mölndal",
                        "Partille",
                        "Torslanda",
                        "Härryda",
                        "Kungsbacka",
                        "Sävedalen",
                    ],
                },
            },
            "Malmö": {
                "weight": 8,
                "fire_risk": 0.5,
                "urban": True,
                "lat": 55.6059,
                "lon": 13.0007,
                "postal_pattern": ["20", "21", "22", "23", "24"],
                "income_distribution": {"mean": 380000, "std": 160000, "skew": 1.0},
                "age_distribution": {"young": 0.32, "middle": 0.43, "elderly": 0.25},
                "neighborhood_types": {
                    "affluent": [
                        "Limhamn",
                        "Bellevue",
                        "Ribersborg",
                        "Västra Hamnen",
                        "Fridhem",
                    ],
                    "middle": [
                        "Västra Hamnen",
                        "Möllevången",
                        "Kirseberg",
                        "Rörsjöstaden",
                        "Södervärn",
                    ],
                    "diverse": [
                        "Rosengård",
                        "Seved",
                        "Södervärn",
                        "Herrgården",
                        "Lindängen",
                        "Fosie",
                    ],
                    "student": ["Rönnen", "Delphi", "Universitetsholmen", "Orkanen"],
                    "family": [
                        "Bunkeflostrand",
                        "Oxie",
                        "Husie",
                        "Tygelsjö",
                        "Klagshamn",
                        "Vintrie",
                    ],
                },
            },
            "Uppsala": {
                "weight": 6,
                "fire_risk": 0.8,
                "urban": False,
                "lat": 59.8586,
                "lon": 17.6389,
                "postal_pattern": ["75"],
                "income_distribution": {"mean": 400000, "std": 150000, "skew": 0.8},
                "age_distribution": {"young": 0.35, "middle": 0.40, "elderly": 0.25},
                "student_city": True,
                "neighborhood_types": {
                    "affluent": ["Luthagen", "Kåbo", "Norby", "Svartbäcken"],
                    "middle": ["Fålhagen", "Eriksberg", "Årsta", "Sala backe"],
                    "diverse": ["Gottsunda", "Valsätra", "Stenhagen", "Gränby"],
                    "student": [
                        "Flogsta",
                        "Studentstaden",
                        "Rackarberget",
                        "Ekonomikum",
                    ],
                    "family": ["Sunnersta", "Nåntuna", "Vänge", "Storvreta"],
                },
            },
            "Linköping": {
                "weight": 4,
                "fire_risk": 0.7,
                "urban": False,
                "lat": 58.4109,
                "lon": 15.6214,
                "postal_pattern": ["58"],
                "income_distribution": {"mean": 390000, "std": 140000, "skew": 0.7},
                "age_distribution": {"young": 0.30, "middle": 0.45, "elderly": 0.25},
                "tech_hub": True,
                "neighborhood_types": {
                    "affluent": ["Ramshäll", "Djurgården", "Wimanshäll"],
                    "middle": [
                        "Vasastaden",
                        "Innerstaden",
                        "Johannelund",
                        "Gottfridsberg",
                    ],
                    "diverse": ["Skäggetorp", "Ryd", "Berga", "Lambohov"],
                    "student": ["Ryd", "Irrblosset", "Flamman", "Valla"],
                    "family": ["Vikingstad", "Linghem", "Sturefors", "Ljungsbro"],
                },
            },
            "Örebro": {
                "weight": 5,
                "fire_risk": 0.9,
                "urban": False,
                "lat": 59.2753,
                "lon": 15.2134,
                "postal_pattern": ["70"],
                "income_distribution": {"mean": 360000, "std": 120000, "skew": 0.5},
                "age_distribution": {"young": 0.28, "middle": 0.44, "elderly": 0.28},
                "neighborhood_types": {
                    "affluent": ["Almby", "Marieberg", "Adolfsberg"],
                    "middle": ["Centrum", "Väster", "Norr", "Rosta"],
                    "diverse": ["Vivalla", "Baronbackarna", "Oxhagen", "Boglundsängen"],
                    "student": ["Studentstaden", "Campus USÖ"],
                    "family": ["Hovsta", "Lillån", "Sörbyängen", "Odensbacken"],
                },
            },
            "Västerås": {
                "weight": 4,
                "fire_risk": 0.8,
                "urban": False,
                "lat": 59.6099,
                "lon": 16.5448,
                "postal_pattern": ["72"],
                "income_distribution": {"mean": 385000, "std": 140000, "skew": 0.6},
                "age_distribution": {"young": 0.29, "middle": 0.45, "elderly": 0.26},
                "neighborhood_types": {
                    "affluent": ["Djäkneberget", "Lillhärad", "Enhagen"],
                    "middle": ["Centrum", "Källtorp", "Malmaberg", "Eriksborg"],
                    "diverse": ["Hamre", "Bäckby", "Råby", "Skallberget"],
                    "student": ["Studentstaden", "Vågholmen"],
                    "family": ["Önsta-Gryta", "Gideonsberg", "Skiljebo", "Irsta"],
                },
            },
            "Helsingborg": {
                "weight": 4,
                "fire_risk": 0.6,
                "urban": False,
                "lat": 56.0465,
                "lon": 12.6945,
                "postal_pattern": ["25"],
                "income_distribution": {"mean": 370000, "std": 150000, "skew": 0.8},
                "age_distribution": {"young": 0.28, "middle": 0.44, "elderly": 0.28},
                "neighborhood_types": {
                    "affluent": ["Tågaborg", "Olympia", "Wilson Park", "Fredriksdal"],
                    "middle": ["Centrum", "Söder", "Norr", "Eneborg"],
                    "diverse": ["Drottninghög", "Fredriksdal", "Dalhem", "Adolfsberg"],
                    "student": ["Campus Helsingborg"],
                    "family": ["Mariastaden", "Ödåkra", "Allerum", "Mörarp"],
                },
            },
            "Norrköping": {
                "weight": 3,
                "fire_risk": 0.7,
                "urban": False,
                "lat": 58.5877,
                "lon": 16.1924,
                "postal_pattern": ["60"],
                "income_distribution": {"mean": 350000, "std": 130000, "skew": 0.5},
                "age_distribution": {"young": 0.27, "middle": 0.45, "elderly": 0.28},
                "neighborhood_types": {
                    "affluent": ["Lindö", "Smedby", "Åby"],
                    "middle": ["Centrum", "Oxelbergen", "Klockaretorpet", "Ektorp"],
                    "diverse": ["Navestad", "Hageby", "Marielund", "Klockaretorpet"],
                    "student": ["Universitetet"],
                    "family": ["Ljura", "Vikbolandet", "Åby", "Kolmården"],
                },
            },
            "Jönköping": {
                "weight": 3,
                "fire_risk": 0.8,
                "urban": False,
                "lat": 57.7826,
                "lon": 14.1618,
                "postal_pattern": ["55"],
                "income_distribution": {"mean": 360000, "std": 130000, "skew": 0.6},
                "age_distribution": {"young": 0.30, "middle": 0.43, "elderly": 0.27},
                "neighborhood_types": {
                    "affluent": ["Rosenlund", "Bymarken", "Dunkehallaområdet"],
                    "middle": ["Centrum", "Väster", "Öster", "Torpa"],
                    "diverse": ["Råslätt", "Öxnehaga", "Österängen", "Huskvarna"],
                    "student": ["Råslätt", "Campus"],
                    "family": ["Barnarp", "Tenhult", "Norrahammar", "Taberg"],
                },
            },
            "Umeå": {
                "weight": 3,
                "fire_risk": 1.2,
                "urban": False,
                "lat": 63.8258,
                "lon": 20.2630,
                "postal_pattern": ["90"],
                "income_distribution": {"mean": 380000, "std": 130000, "skew": 0.6},
                "age_distribution": {"young": 0.33, "middle": 0.42, "elderly": 0.25},
                "student_city": True,
                "neighborhood_types": {
                    "affluent": ["Röbäck", "Berghem", "Sandbacka"],
                    "middle": ["Centrum", "Teg", "Haga", "Tomtebo"],
                    "diverse": ["Ålidhem", "Ersboda", "Holmsund"],
                    "student": ["Ålidhem", "Nydalahöjd", "Mariehem"],
                    "family": ["Backen", "Böleäng", "Strömpilen", "Holmsund"],
                },
            },
            "Lund": {
                "weight": 3,
                "fire_risk": 0.5,
                "urban": False,
                "lat": 55.7047,
                "lon": 13.1910,
                "postal_pattern": ["22"],
                "income_distribution": {"mean": 420000, "std": 160000, "skew": 0.9},
                "age_distribution": {"young": 0.38, "middle": 0.37, "elderly": 0.25},
                "student_city": True,
                "neighborhood_types": {
                    "affluent": ["Professorsstaden", "Villaområdet", "Sankt Lars"],
                    "middle": ["Centrum", "Järnåkra", "Kobjer", "Nilstorp"],
                    "diverse": [
                        "Norra Fäladen",
                        "Klostergården",
                        "Linero",
                        "Östra Torn",
                    ],
                    "student": ["Sparta", "Vildanden", "Ulrikedal", "Parentesen"],
                    "family": ["Gunnesbo", "Södra Sandby", "Dalby", "Genarp"],
                },
            },
            "Borås": {
                "weight": 2,
                "fire_risk": 0.9,
                "urban": False,
                "lat": 57.7210,
                "lon": 12.9401,
                "postal_pattern": ["50"],
                "income_distribution": {"mean": 340000, "std": 120000, "skew": 0.5},
                "age_distribution": {"young": 0.26, "middle": 0.46, "elderly": 0.28},
                "neighborhood_types": {
                    "affluent": ["Villastaden", "Byttorp", "Trandared"],
                    "middle": ["Centrum", "Norrby", "Brämhult", "Sjöbo"],
                    "diverse": ["Hässleholmen", "Hulta", "Norrby", "Sjöbo"],
                    "student": ["Högskolan"],
                    "family": ["Fristad", "Sandared", "Viskafors", "Dalsjöfors"],
                },
            },
            "Sundsvall": {
                "weight": 2,
                "fire_risk": 1.1,
                "urban": False,
                "lat": 62.3908,
                "lon": 17.3069,
                "postal_pattern": ["85"],
                "income_distribution": {"mean": 360000, "std": 120000, "skew": 0.5},
                "age_distribution": {"young": 0.27, "middle": 0.45, "elderly": 0.28},
                "neighborhood_types": {
                    "affluent": ["Södra Berget", "Sidsjön", "Bosvedjan"],
                    "middle": ["Centrum", "Haga", "Östermalm", "Skönsmon"],
                    "diverse": ["Skönsberg", "Nacksta", "Ljustadalen", "Granloholm"],
                    "student": ["Universitetet"],
                    "family": ["Alnö", "Njurunda", "Matfors", "Stöde"],
                },
            },
            "Gävle": {
                "weight": 2,
                "fire_risk": 1.1,
                "urban": False,
                "lat": 60.6749,
                "lon": 17.1413,
                "postal_pattern": ["80"],
                "income_distribution": {"mean": 350000, "std": 120000, "skew": 0.5},
                "age_distribution": {"young": 0.26, "middle": 0.46, "elderly": 0.28},
                "neighborhood_types": {
                    "affluent": ["Villastaden", "Bomhus", "Sörby"],
                    "middle": ["Centrum", "Brynäs", "Strömsbro", "Hemlingby"],
                    "diverse": ["Sätra", "Andersberg", "Öster", "Nordost"],
                    "student": ["Högskolan"],
                    "family": ["Valbo", "Forsbacka", "Sandviken", "Hedesunda"],
                },
            },
            "Eskilstuna": {
                "weight": 2,
                "fire_risk": 0.8,
                "urban": False,
                "lat": 59.3666,
                "lon": 16.5077,
                "postal_pattern": ["63"],
                "income_distribution": {"mean": 340000, "std": 120000, "skew": 0.5},
                "age_distribution": {"young": 0.27, "middle": 0.45, "elderly": 0.28},
                "neighborhood_types": {
                    "affluent": ["Borsökna", "Skogstorp", "Mesta"],
                    "middle": ["Centrum", "Årby", "Torshälla", "Slagsta"],
                    "diverse": ["Lagersberg", "Fröslunda", "Skiftinge", "Råbergstorp"],
                    "student": ["MDH Campus"],
                    "family": ["Hällbybrunn", "Kjula", "Ärla", "Sundbyholm"],
                },
            },
            "Södertälje": {
                "weight": 2,
                "fire_risk": 0.7,
                "urban": False,
                "lat": 59.1955,
                "lon": 17.6253,
                "postal_pattern": ["15"],
                "income_distribution": {"mean": 360000, "std": 140000, "skew": 0.6},
                "age_distribution": {"young": 0.30, "middle": 0.45, "elderly": 0.25},
                "neighborhood_types": {
                    "affluent": ["Pershagen", "Viksberg", "Östertälje"],
                    "middle": ["Centrum", "Mariekäll", "Brunnsäng", "Karlhov"],
                    "diverse": ["Ronna", "Hovsjö", "Fornhöjden", "Geneta"],
                    "student": ["KTH Södertälje"],
                    "family": ["Enhörna", "Järna", "Hölö", "Mölnbo"],
                },
            },
            "Karlstad": {
                "weight": 2,
                "fire_risk": 0.9,
                "urban": False,
                "lat": 59.3813,
                "lon": 13.5039,
                "postal_pattern": ["65"],
                "income_distribution": {"mean": 360000, "std": 130000, "skew": 0.6},
                "age_distribution": {"young": 0.29, "middle": 0.44, "elderly": 0.27},
                "neighborhood_types": {
                    "affluent": ["Herrhagen", "Marieberg", "Sundsta"],
                    "middle": ["Centrum", "Klara", "Norrstrand", "Rud"],
                    "diverse": ["Kronoparken", "Våxnäs", "Orrholmen", "Stockfallet"],
                    "student": ["Universitetet", "Kroppkärr"],
                    "family": ["Skattkärr", "Vålberg", "Edsvalla", "Molkom"],
                },
            },
            "Växjö": {
                "weight": 2,
                "fire_risk": 0.8,
                "urban": False,
                "lat": 56.8777,
                "lon": 14.8091,
                "postal_pattern": ["35"],
                "income_distribution": {"mean": 350000, "std": 120000, "skew": 0.5},
                "age_distribution": {"young": 0.31, "middle": 0.42, "elderly": 0.27},
                "neighborhood_types": {
                    "affluent": ["Teleborg", "Öjaby", "Hovshaga"],
                    "middle": ["Centrum", "Söder", "Öster", "Väster"],
                    "diverse": ["Araby", "Dalbo", "Högstorp", "Sandsbro"],
                    "student": ["Universitetet", "Campus"],
                    "family": ["Gemla", "Ingelstad", "Braås", "Rottne"],
                },
            },
            "Kristianstad": {
                "weight": 2,
                "fire_risk": 0.7,
                "urban": False,
                "lat": 56.0294,
                "lon": 14.1567,
                "postal_pattern": ["29"],
                "income_distribution": {"mean": 340000, "std": 120000, "skew": 0.5},
                "age_distribution": {"young": 0.26, "middle": 0.45, "elderly": 0.29},
                "neighborhood_types": {
                    "affluent": ["Sommarlust", "Björket", "Hammar"],
                    "middle": ["Centrum", "Österäng", "Näsby", "Vä"],
                    "diverse": ["Gamlegården", "Charlottesborg", "Långebro"],
                    "student": ["Högskolan"],
                    "family": ["Åhus", "Tollarp", "Degeberga", "Arkelstorp"],
                },
            },
            "Luleå": {
                "weight": 1,
                "fire_risk": 1.4,
                "urban": False,
                "lat": 65.5848,
                "lon": 22.1547,
                "postal_pattern": ["97"],
                "income_distribution": {"mean": 370000, "std": 110000, "skew": 0.4},
                "age_distribution": {"young": 0.26, "middle": 0.46, "elderly": 0.28},
                "tech_hub": True,
                "neighborhood_types": {
                    "affluent": ["Hertsön", "Örnäset", "Kronan"],
                    "middle": ["Centrum", "Östermalm", "Mjölkudden", "Bergviken"],
                    "diverse": ["Hertsön", "Björkskatan", "Örnäset"],
                    "student": ["Universitetet", "Porsön"],
                    "family": ["Gammelstad", "Råneå", "Antnäs", "Södra Sunderbyn"],
                },
            },
            "Kalmar": {
                "weight": 1,
                "fire_risk": 0.8,
                "urban": False,
                "lat": 56.6634,
                "lon": 16.3566,
                "postal_pattern": ["39"],
                "income_distribution": {"mean": 340000, "std": 120000, "skew": 0.5},
                "age_distribution": {"young": 0.28, "middle": 0.44, "elderly": 0.28},
                "neighborhood_types": {
                    "affluent": ["Ängö", "Djurängen", "Malmen"],
                    "middle": ["Centrum", "Norrliden", "Funkabo", "Oxhagen"],
                    "diverse": ["Norrliden", "Tallhagen", "Rinkabyholm"],
                    "student": ["Universitetet"],
                    "family": ["Lindsdal", "Rinkabyholm", "Smedby", "Ljungbyholm"],
                },
            },
            "Skellefteå": {
                "weight": 1,
                "fire_risk": 1.3,
                "urban": False,
                "lat": 64.7507,
                "lon": 20.9528,
                "postal_pattern": ["93"],
                "income_distribution": {"mean": 360000, "std": 110000, "skew": 0.4},
                "age_distribution": {"young": 0.25, "middle": 0.47, "elderly": 0.28},
                "neighborhood_types": {
                    "affluent": ["Morö Backe", "Solbacken", "Erikslid"],
                    "middle": ["Centrum", "Norrböle", "Anderstorp", "Sörböle"],
                    "diverse": ["Morö Backe", "Norrböle", "Tuböle"],
                    "student": ["Campus Skellefteå"],
                    "family": ["Ursviken", "Byske", "Boliden", "Burträsk"],
                },
            },
            "Karlskrona": {
                "weight": 1,
                "fire_risk": 0.7,
                "urban": False,
                "lat": 56.1612,
                "lon": 15.5869,
                "postal_pattern": ["37"],
                "income_distribution": {"mean": 350000, "std": 120000, "skew": 0.5},
                "age_distribution": {"young": 0.27, "middle": 0.45, "elderly": 0.28},
                "neighborhood_types": {
                    "affluent": ["Saltö", "Dragsö", "Långö"],
                    "middle": ["Centrum", "Bergåsa", "Lyckeby", "Rödeby"],
                    "diverse": ["Kungsmarken", "Marieberg", "Galgamarken"],
                    "student": ["BTH Campus"],
                    "family": ["Nättraby", "Jämjö", "Ramdala", "Sturkö"],
                },
            },
            "Halmstad": {
                "weight": 2,
                "fire_risk": 0.7,
                "urban": False,
                "lat": 56.6745,
                "lon": 12.8578,
                "postal_pattern": ["30"],
                "income_distribution": {"mean": 360000, "std": 130000, "skew": 0.6},
                "age_distribution": {"young": 0.28, "middle": 0.44, "elderly": 0.28},
                "neighborhood_types": {
                    "affluent": ["Tylösand", "Söndrum", "Frösakull"],
                    "middle": ["Centrum", "Österskans", "Vallås", "Sofieberg"],
                    "diverse": ["Andersberg", "Frennarp", "Linehed"],
                    "student": ["Högskolan"],
                    "family": ["Oskarström", "Getinge", "Haverdal", "Fyllinge"],
                },
            },
        }

        # Small towns and rural areas (add as lower weight locations)
        rural_locations = {
            "Kiruna": {
                "weight": 0.5,
                "fire_risk": 1.5,
                "urban": False,
                "lat": 67.8557,
                "lon": 20.2253,
            },
            "Östersund": {
                "weight": 1.5,
                "fire_risk": 1.2,
                "urban": False,
                "lat": 63.1792,
                "lon": 14.6357,
            },
            "Trollhättan": {
                "weight": 1.5,
                "fire_risk": 0.8,
                "urban": False,
                "lat": 58.2837,
                "lon": 12.2886,
            },
            "Örnsköldsvik": {
                "weight": 1,
                "fire_risk": 1.2,
                "urban": False,
                "lat": 63.2909,
                "lon": 18.7175,
            },
            "Varberg": {
                "weight": 1,
                "fire_risk": 0.7,
                "urban": False,
                "lat": 57.1056,
                "lon": 12.2508,
            },
            "Uddevalla": {
                "weight": 1,
                "fire_risk": 0.8,
                "urban": False,
                "lat": 58.3498,
                "lon": 11.9424,
            },
            "Falun": {
                "weight": 1,
                "fire_risk": 1.0,
                "urban": False,
                "lat": 60.6065,
                "lon": 15.6355,
            },
            "Borlänge": {
                "weight": 1,
                "fire_risk": 1.0,
                "urban": False,
                "lat": 60.4858,
                "lon": 15.4371,
            },
            "Landskrona": {
                "weight": 1,
                "fire_risk": 0.6,
                "urban": False,
                "lat": 55.8708,
                "lon": 12.8301,
            },
            "Trelleborg": {
                "weight": 0.8,
                "fire_risk": 0.5,
                "urban": False,
                "lat": 55.3754,
                "lon": 13.1569,
            },
            "Ängelholm": {
                "weight": 0.8,
                "fire_risk": 0.6,
                "urban": False,
                "lat": 56.2428,
                "lon": 12.8621,
            },
            "Motala": {
                "weight": 0.8,
                "fire_risk": 0.8,
                "urban": False,
                "lat": 58.5373,
                "lon": 15.0422,
            },
            "Nyköping": {
                "weight": 1,
                "fire_risk": 0.8,
                "urban": False,
                "lat": 58.7530,
                "lon": 17.0088,
            },
            "Piteå": {
                "weight": 0.7,
                "fire_risk": 1.3,
                "urban": False,
                "lat": 65.3172,
                "lon": 21.4797,
            },
            "Visby": {
                "weight": 0.6,
                "fire_risk": 0.6,
                "urban": False,
                "lat": 57.6348,
                "lon": 18.2948,
            },
            "Ystad": {
                "weight": 0.6,
                "fire_risk": 0.5,
                "urban": False,
                "lat": 55.4295,
                "lon": 13.8204,
            },
            "Lidköping": {
                "weight": 0.6,
                "fire_risk": 0.8,
                "urban": False,
                "lat": 58.5053,
                "lon": 13.1578,
            },
            "Sandviken": {
                "weight": 0.8,
                "fire_risk": 1.0,
                "urban": False,
                "lat": 60.6166,
                "lon": 16.7757,
            },
            "Alingsås": {
                "weight": 0.7,
                "fire_risk": 0.8,
                "urban": False,
                "lat": 57.9303,
                "lon": 12.5333,
            },
            "Hudiksvall": {
                "weight": 0.7,
                "fire_risk": 1.1,
                "urban": False,
                "lat": 61.7287,
                "lon": 17.1059,
            },
        }

        # Add income and age distributions for smaller cities
        for city, data in rural_locations.items():
            data["income_distribution"] = {"mean": 320000, "std": 100000, "skew": 0.4}
            data["age_distribution"] = {"young": 0.24, "middle": 0.46, "elderly": 0.30}
            data["postal_pattern"] = []  # Will be geocoded
            data["neighborhood_types"] = {
                "middle": ["Centrum", "Norr", "Söder", "Väster", "Öster"],
                "family": ["Villaområdet", "Industriområdet"],
            }

        self.swedish_cities.update(rural_locations)

        # Initialize city selector after loading geocoder data
        self.city_selector = None

        # Initialize campaign patterns with seasonal and event-driven variations
        self.campaign_patterns = {
            "2014": {"summer_intensity": 0.7, "special_events": []},
            "2015": {"summer_intensity": 0.8, "special_events": []},
            "2016": {"summer_intensity": 0.9, "special_events": []},
            "2017": {"summer_intensity": 1.0, "special_events": []},
            "2018": {
                "summer_intensity": 1.5,
                "special_events": ["extreme_heat_july", "major_fires_gävleborg"],
            },
            "2019": {"summer_intensity": 1.2, "special_events": ["early_spring_fires"]},
            "2020": {
                "summer_intensity": 0.8,
                "special_events": ["covid_reduced_activity"],
            },
            "2021": {"summer_intensity": 0.9, "special_events": ["covid_recovery"]},
            "2022": {
                "summer_intensity": 1.3,
                "special_events": ["dry_spring", "heat_wave_june"],
            },
            "2023": {
                "summer_intensity": 1.1,
                "special_events": ["early_season_prevention"],
            },
            "2024": {
                "summer_intensity": 1.0,
                "special_events": ["climate_adaptation_focus"],
            },
        }

        # Enhanced organization profiles
        self.org_type_profiles = {
            "Government Agency": {
                "activity_level": 0.9,
                "burst_tendency": 0.3,
                "leaflet_preference": ["safety", "regulation", "prevention"],
                "seasonal_pattern": "consistent",
                "geographic_reach": "national",
                "target_demographic": "broad",
                "working_hours": True,
                "weekend_activity": 0.1,
                "budget_cycle": "quarterly",
                "collaboration_rate": 0.7,
                "digital_adoption": 0.6,
                "response_to_events": 0.9,
                "distribution_patterns": {
                    "route_efficiency": 0.9,  # How efficiently they plan routes
                    "batch_clustering": 0.8,  # Tendency to cluster deliveries
                    "repeat_targeting": 0.3,  # Tendency to revisit same areas
                },
            },
            "Municipal Fire Department": {
                "activity_level": 0.8,
                "burst_tendency": 0.8,
                "leaflet_preference": ["emergency", "local_safety", "prevention"],
                "seasonal_pattern": "high_summer",
                "geographic_reach": "local",
                "target_demographic": "residents",
                "working_hours": False,
                "weekend_activity": 0.7,
                "budget_cycle": "annual",
                "collaboration_rate": 0.9,
                "digital_adoption": 0.4,
                "response_to_events": 1.0,
                "distribution_patterns": {
                    "route_efficiency": 0.7,
                    "batch_clustering": 0.9,
                    "repeat_targeting": 0.6,
                },
            },
            "Environmental NGO": {
                "activity_level": 0.6,
                "burst_tendency": 0.7,
                "leaflet_preference": ["environment", "conservation", "awareness"],
                "seasonal_pattern": "spring_summer",
                "geographic_reach": "regional",
                "target_demographic": "environmentally_conscious",
                "working_hours": True,
                "weekend_activity": 0.3,
                "budget_cycle": "project",
                "collaboration_rate": 0.8,
                "digital_adoption": 0.8,
                "response_to_events": 0.6,
                "distribution_patterns": {
                    "route_efficiency": 0.5,
                    "batch_clustering": 0.6,
                    "repeat_targeting": 0.4,
                },
            },
            "Community Group": {
                "activity_level": 0.4,
                "burst_tendency": 0.8,
                "leaflet_preference": ["community", "local_safety", "awareness"],
                "seasonal_pattern": "event_driven",
                "geographic_reach": "very_local",
                "target_demographic": "local_residents",
                "working_hours": False,
                "weekend_activity": 0.6,
                "budget_cycle": "donation",
                "collaboration_rate": 0.7,
                "digital_adoption": 0.5,
                "response_to_events": 0.7,
                "distribution_patterns": {
                    "route_efficiency": 0.3,
                    "batch_clustering": 0.95,  # Very localized
                    "repeat_targeting": 0.8,  # Know their community well
                },
            },
        }

        # Realistic Swedish name patterns with regional variations
        self.name_patterns = {
            "traditional_swedish": {
                "weight": 0.3,
                "first_names": {
                    "male": [
                        "Erik",
                        "Lars",
                        "Karl",
                        "Nils",
                        "Anders",
                        "Per",
                        "Olof",
                        "Sven",
                        "Gunnar",
                        "Johan",
                        "Magnus",
                        "Bengt",
                        "Ulf",
                        "Bo",
                        "Åke",
                        "Rolf",
                        "Lennart",
                        "Kjell",
                        "Göran",
                        "Hans",
                        "Arne",
                        "Leif",
                        "Torbjörn",
                        "Håkan",
                        "Mats",
                        "Stefan",
                        "Mikael",
                        "Jan",
                        "Thomas",
                        "Peter",
                        "Christer",
                        "Tommy",
                        "Kenneth",
                        "Roger",
                        "Björn",
                        "Ingvar",
                        "Rune",
                        "Bertil",
                        "Ove",
                        "Kent",
                        "Kurt",
                        "Stig",
                        "Tomas",
                        "Martin",
                        "Patrik",
                        "Jörgen",
                        "Daniel",
                        "Fredrik",
                        "Robert",
                        "Henrik",
                        "Claes",
                        "Jonas",
                        "Niklas",
                        "Marcus",
                        "Andreas",
                        "Christian",
                        "Simon",
                        "Viktor",
                        "Gösta",
                        "Evert",
                        "Folke",
                        "Sigurd",
                        "Ivar",
                        "Axel",
                        "Vilhelm",
                        "Gustaf",
                        "Albin",
                        "Hugo",
                        "Edvin",
                        "Arvid",
                        "Helge",
                        "Ragnar",
                        "Valter",
                        "Oskar",
                    ],
                    "female": [
                        "Anna",
                        "Eva",
                        "Karin",
                        "Ingrid",
                        "Margareta",
                        "Birgitta",
                        "Kristina",
                        "Maria",
                        "Elisabet",
                        "Linnéa",
                        "Kerstin",
                        "Lena",
                        "Helena",
                        "Marianne",
                        "Annika",
                        "Ulla",
                        "Gunilla",
                        "Malin",
                        "Jenny",
                        "Hanna",
                        "Linda",
                        "Susanne",
                        "Monica",
                        "Johanna",
                        "Carina",
                        "Sofia",
                        "Emma",
                        "Sara",
                        "Katarina",
                        "Marie",
                        "Yvonne",
                        "Anette",
                        "Camilla",
                        "Åsa",
                        "Ulrika",
                        "Caroline",
                        "Jessica",
                        "Therese",
                        "Gun",
                        "Britt",
                        "Inger",
                        "Maj",
                        "Astrid",
                        "Siv",
                        "Berit",
                        "Gudrun",
                        "Rut",
                        "Elin",
                        "Ida",
                        "Alice",
                        "Maja",
                        "Elsa",
                        "Julia",
                        "Ella",
                        "Ebba",
                        "Olivia",
                        "Wilma",
                        "Klara",
                        "Nellie",
                        "Agnes",
                        "Isabelle",
                        "Vera",
                        "Ellen",
                        "Stella",
                        "Clara",
                        "Emilia",
                        "Alva",
                        "Alma",
                        "Elise",
                        "Saga",
                        "Selma",
                        "Elsa",
                        "Astrid",
                        "Hedvig",
                        "Signe",
                        "Freja",
                        "Märta",
                        "Ingeborg",
                    ],
                },
                "last_names": [
                    "Andersson",
                    "Johansson",
                    "Karlsson",
                    "Nilsson",
                    "Eriksson",
                    "Larsson",
                    "Olsson",
                    "Persson",
                    "Svensson",
                    "Gustafsson",
                    "Pettersson",
                    "Jonsson",
                    "Jansson",
                    "Hansson",
                    "Bengtsson",
                    "Jönsson",
                    "Lindberg",
                    "Jakobsson",
                    "Magnusson",
                    "Olofsson",
                    "Lindström",
                    "Lindqvist",
                    "Lindgren",
                    "Berg",
                    "Axelsson",
                    "Bergström",
                    "Lundberg",
                    "Lind",
                    "Lundgren",
                    "Lundqvist",
                    "Mattsson",
                    "Berglund",
                    "Fredriksson",
                    "Sandberg",
                    "Henriksson",
                    "Forsberg",
                    "Sjöberg",
                    "Wallin",
                    "Ali",
                    "Engström",
                    "Danielsson",
                    "Håkansson",
                    "Eklund",
                    "Lundin",
                    "Gunnarsson",
                    "Björk",
                    "Bergman",
                    "Holm",
                    "Samuelsson",
                    "Fransson",
                    "Wikström",
                    "Isaksson",
                    "Bergqvist",
                    "Arvidsson",
                    "Nyström",
                    "Holmberg",
                    "Löfgren",
                    "Söderberg",
                    "Nyberg",
                    "Blomqvist",
                    "Claesson",
                    "Nordström",
                    "Lundström",
                    "Eliasson",
                    "Pålsson",
                    "Björklund",
                    "Viklund",
                    "Sandström",
                    "Lund",
                    "Nordin",
                    "Ström",
                    "Åberg",
                    "Ekström",
                    "Hermansson",
                ],
            },
            "modern_swedish": {
                "weight": 0.3,
                "first_names": {
                    "male": [
                        "Alexander",
                        "Oscar",
                        "William",
                        "Lucas",
                        "Liam",
                        "Oliver",
                        "Noah",
                        "Hugo",
                        "Elias",
                        "Leo",
                        "Viktor",
                        "Emil",
                        "Leon",
                        "Ludvig",
                        "Adrian",
                        "Axel",
                        "Alfred",
                        "Theo",
                        "Vincent",
                        "Matteo",
                        "Nils",
                        "Adam",
                        "Arvid",
                        "Filip",
                        "Elliot",
                        "Albin",
                        "Edvin",
                        "Valter",
                        "Sixten",
                        "Melvin",
                        "Isak",
                        "Love",
                        "Casper",
                        "Benjamin",
                        "Kevin",
                        "Robin",
                        "Felix",
                        "Anton",
                        "Gustav",
                        "Vilgot",
                        "Charlie",
                        "Viggo",
                        "Harry",
                        "Milton",
                        "Maximilian",
                        "Loui",
                        "Sam",
                        "Frank",
                        "Jacob",
                        "Melker",
                        "Malte",
                    ],
                    "female": [
                        "Alice",
                        "Maja",
                        "Elsa",
                        "Astrid",
                        "Wilma",
                        "Freja",
                        "Olivia",
                        "Ebba",
                        "Klara",
                        "Alma",
                        "Agnes",
                        "Ella",
                        "Stella",
                        "Vera",
                        "Ellen",
                        "Selma",
                        "Julia",
                        "Alva",
                        "Alicia",
                        "Saga",
                        "Ines",
                        "Ellie",
                        "Nova",
                        "Emilia",
                        "Nellie",
                        "Isabelle",
                        "Luna",
                        "Clara",
                        "Lily",
                        "Lea",
                        "Lilly",
                        "Molly",
                        "Mila",
                        "My",
                        "Felicia",
                        "Amanda",
                        "Elvira",
                        "Hilma",
                        "Tuva",
                        "Tilda",
                        "Matilda",
                        "Ida",
                        "Linn",
                        "Nora",
                        "Zoe",
                        "Nathalie",
                        "Tilde",
                        "Majken",
                        "Stina",
                        "Lova",
                        "Cornelia",
                        "Melissa",
                    ],
                },
                "last_names": [
                    "Berg",
                    "Lindberg",
                    "Lindgren",
                    "Lindqvist",
                    "Holm",
                    "Holmgren",
                    "Holmqvist",
                    "Strand",
                    "Ekberg",
                    "Sandell",
                    "Norberg",
                    "Ågren",
                    "Öberg",
                    "Sjögren",
                    "Engberg",
                    "Hedberg",
                    "Sundberg",
                    "Dahlberg",
                    "Hellström",
                    "Sjöström",
                    "Falk",
                    "Blom",
                    "Ek",
                    "Rosén",
                    "Åström",
                ],
            },
            "immigrant_names": {
                "weight": 0.3,
                "first_names": {
                    "male": [
                        # Middle Eastern
                        "Mohammed",
                        "Ali",
                        "Ahmed",
                        "Omar",
                        "Hassan",
                        "Ibrahim",
                        "Yusuf",
                        "Abdullah",
                        "Khalid",
                        "Mustafa",
                        "Mahmoud",
                        "Samir",
                        "Karim",
                        "Tariq",
                        # Eastern European
                        "Aleksandar",
                        "Milos",
                        "Vladimir",
                        "Nikola",
                        "Stefan",
                        "Marko",
                        "Ivan",
                        "Andrei",
                        "Pavel",
                        "Dmitri",
                        "Sergei",
                        "Boris",
                        "Mikhail",
                        "Piotr",
                        # Asian
                        "Wei",
                        "Jun",
                        "Hiroshi",
                        "Takeshi",
                        "Ravi",
                        "Amit",
                        "Suresh",
                        "Raj",
                        # African
                        "Abdi",
                        "Yassin",
                        "Osman",
                        "Daud",
                        "Ismail",
                        "Yonas",
                        "Samuel",
                        "Daniel",
                        # Latin American
                        "Carlos",
                        "Juan",
                        "Miguel",
                        "Luis",
                        "José",
                        "Pedro",
                        "Diego",
                        "Alejandro",
                    ],
                    "female": [
                        # Middle Eastern
                        "Fatima",
                        "Aisha",
                        "Maryam",
                        "Sara",
                        "Leila",
                        "Yasmin",
                        "Nour",
                        "Hala",
                        "Zahra",
                        "Amina",
                        "Khadija",
                        "Rania",
                        "Layla",
                        "Salma",
                        "Dina",
                        "Nadia",
                        # Eastern European
                        "Natasha",
                        "Olga",
                        "Elena",
                        "Irina",
                        "Katarina",
                        "Milena",
                        "Ana",
                        "Maria",
                        "Svetlana",
                        "Tatiana",
                        "Marina",
                        "Jelena",
                        "Ivana",
                        "Petra",
                        "Zuzana",
                        # Asian
                        "Mei",
                        "Li",
                        "Yuki",
                        "Sakura",
                        "Priya",
                        "Anjali",
                        "Deepa",
                        "Kavita",
                        # African
                        "Amara",
                        "Zara",
                        "Nia",
                        "Ayana",
                        "Kadija",
                        "Sahra",
                        "Muna",
                        "Hawa",
                        # Latin American
                        "Sofia",
                        "Isabella",
                        "Valentina",
                        "Camila",
                        "Daniela",
                        "Lucia",
                        "Paula",
                    ],
                },
                "last_names": [
                    # Middle Eastern
                    "Ali",
                    "Ahmed",
                    "Hassan",
                    "Mohamed",
                    "Ibrahim",
                    "Haddad",
                    "Al-Said",
                    "Khalil",
                    "Hamdi",
                    "Nasser",
                    "Rashid",
                    "Saleh",
                    "Yousef",
                    "Aziz",
                    # Eastern European
                    "Novak",
                    "Jovanovic",
                    "Petrovic",
                    "Nikolic",
                    "Stojanovic",
                    "Popovic",
                    "Kowalski",
                    "Nowak",
                    "Wojcik",
                    "Petrov",
                    "Ivanov",
                    "Volkov",
                    # Asian
                    "Wang",
                    "Li",
                    "Zhang",
                    "Liu",
                    "Chen",
                    "Tanaka",
                    "Sato",
                    "Suzuki",
                    "Patel",
                    "Sharma",
                    "Singh",
                    "Kumar",
                    "Gupta",
                    "Mehta",
                    "Rao",
                    # African
                    "Okonkwo",
                    "Diallo",
                    "Mensah",
                    "Kamara",
                    "Traore",
                    "Sesay",
                    "Jallow",
                    # Latin American
                    "Garcia",
                    "Rodriguez",
                    "Martinez",
                    "Lopez",
                    "Gonzalez",
                    "Hernandez",
                    "Silva",
                ],
            },
        }

        # Initialize state tracking
        self.organizations = []
        self.leaflets = []
        self.org_state = {}
        self.active_campaigns = []
        self.neighborhood_history = defaultdict(
            lambda: {
                "last_distribution": None,
                "total_distributions": 0,
                "saturation_level": 0.0,
            }
        )
        self.next_distribution_id = 1

        # Route planning cache
        self.distribution_routes = {}
        self.current_routes = {}

    def load_existing_data(self):
        """Load and analyze existing organizations and leaflets"""
        print("\n📚 Loading existing data...")

        # Load geocoder
        print("🗺️  Initializing enhanced geocoder...")
        if self.geocoder.base_geocoder.load_geonames_data():
            print("✅ Enhanced geocoder ready with neighborhood profiling")
            # Initialize city selector after geocoder is loaded
            self.city_selector = EnhancedCitySelector(self.geocoder.base_geocoder)

        # Load organizations
        org_file = SCRIPT_DIR / "orgs.csv"
        if org_file.exists():
            with open(org_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.organizations = list(reader)
            print(f"✅ Loaded {len(self.organizations)} organizations")
            self.enhance_organizations_realistically()

        # Load leaflets
        leaflet_file = SCRIPT_DIR / "leaflets.csv"
        if leaflet_file.exists():
            with open(leaflet_file, "r", encoding="iso-8859-1") as f:
                reader = csv.DictReader(f, delimiter="¦")
                self.leaflets = list(reader)
            print(f"✅ Loaded {len(self.leaflets)} leaflets")
            self.enhance_leaflets_realistically()

        self.initialize_org_states()
        self.find_next_distribution_id()

    def enhance_organizations_realistically(self):
        """Add ultra-realistic attributes to organizations"""
        for org in self.organizations:
            org_type = org.get("Type", "Unknown")

            if org_type in self.org_type_profiles:
                profile = self.org_type_profiles[org_type]
                org.update(profile)
            else:
                # Set default values for organizations without a profile
                org["activity_level"] = 0.5
                org["burst_tendency"] = 0.5
                org["seasonal_pattern"] = "consistent"
                org["geographic_reach"] = "regional"  # Default reach
                org["target_demographic"] = "general"
                org["working_hours"] = True
                org["weekend_activity"] = 0.2
                org["budget_cycle"] = "annual"
                org["collaboration_rate"] = 0.5
                org["digital_adoption"] = 0.5
                org["response_to_events"] = 0.7
                org["distribution_patterns"] = {
                    "route_efficiency": 0.6,
                    "batch_clustering": 0.7,
                    "repeat_targeting": 0.5,
                }

            # Assign realistic home cities based on organization type and name
            org_name = org.get("Name", "").lower()

            # Use enhanced city selector if available
            if hasattr(self, "city_selector") and self.city_selector:
                # Select city using GeoNames data
                if org.get("geographic_reach", "regional") == "national":
                    # National orgs concentrate in major cities
                    selected_city = self.city_selector.select_city(
                        campaign_type="routine_awareness",
                        org_type=org_type,
                        focus_regions=None,
                    )
                    org["home_city"] = (
                        selected_city["name"] if selected_city else "Stockholm"
                    )
                    org["city_data"] = selected_city
                elif org.get("geographic_reach", "regional") == "very_local":
                    # Community groups in smaller areas
                    small_cities = self.city_selector.get_cities_by_type(
                        ["town", "small_town", "village"]
                    )
                    if small_cities:
                        selected_city = random.choice(small_cities)
                        org["home_city"] = selected_city["name"]
                        org["city_data"] = selected_city
                    else:
                        selected_city = self.city_selector.select_city()
                        org["home_city"] = (
                            selected_city["name"] if selected_city else "Stockholm"
                        )
                        org["city_data"] = selected_city
                else:
                    # Regional orgs - balanced selection
                    selected_city = self.city_selector.select_city(
                        campaign_type="routine_awareness", org_type=org_type
                    )
                    org["home_city"] = (
                        selected_city["name"] if selected_city else "Stockholm"
                    )
                    org["city_data"] = selected_city
            else:
                # Fallback to old method if city selector not available
                # Extract city from name if present
                home_city = None
                for city in self.swedish_cities.keys():
                    if city.lower() in org_name:
                        home_city = city
                        break

                if not home_city:
                    # Assign based on realistic distribution
                    if org.get("geographic_reach", "regional") == "national":
                        # National orgs concentrate in major cities
                        home_city = random.choices(
                            ["Stockholm", "Göteborg", "Malmö"], weights=[60, 25, 15]
                        )[0]
                    elif org.get("geographic_reach", "regional") == "very_local":
                        # Community groups spread across all cities
                        cities = list(self.swedish_cities.keys())
                        weights = [self.swedish_cities[c]["weight"] for c in cities]
                        home_city = random.choices(cities, weights=weights)[0]
                    else:
                        # Regional orgs favor medium-large cities
                        cities = list(self.swedish_cities.keys())
                        weights = [
                            self.swedish_cities[c]["weight"]
                            * (
                                1.5
                                if c
                                in [
                                    "Stockholm",
                                    "Göteborg",
                                    "Malmö",
                                    "Uppsala",
                                    "Linköping",
                                ]
                                else 1.0
                            )
                            for c in cities
                        ]
                        home_city = random.choices(cities, weights=weights)[0]

                org["home_city"] = home_city

            # Add organizational history and characteristics
            org["founded_year"] = self._generate_founded_year(org_type)
            org["employee_count"] = self._generate_employee_count(
                org_type, org.get("size_category", "medium")
            )
            org["annual_budget"] = self._generate_annual_budget(
                org_type, org["employee_count"]
            )
            org["distribution_capability"] = self._calculate_distribution_capability(
                org
            )

            # Add behavioral patterns
            org["peak_activity_hours"] = self._generate_peak_hours(org_type)
            org["route_planning_skill"] = (
                random.uniform(0.3, 1.0)
                if org_type == "Community Group"
                else random.uniform(0.6, 0.95)
            )
            org["volunteer_availability"] = self._generate_volunteer_pattern(org_type)

    def _generate_founded_year(self, org_type: str) -> int:
        """Generate realistic founding years based on organization type"""
        if org_type == "Government Agency":
            # Government agencies tend to be older
            return random.choices(
                [
                    random.randint(1940, 1969),
                    random.randint(1970, 1999),
                    random.randint(2000, 2020),
                ],
                weights=[30, 50, 20],
            )[0]
        elif org_type == "Environmental NGO":
            # Environmental movement peaked in 1980s-2000s
            return random.choices(
                [
                    random.randint(1970, 1989),
                    random.randint(1990, 2009),
                    random.randint(2010, 2023),
                ],
                weights=[25, 50, 25],
            )[0]
        elif org_type == "Community Group":
            # Community groups can be any age but many are recent
            return random.choices(
                [
                    random.randint(1950, 1979),
                    random.randint(1980, 1999),
                    random.randint(2000, 2023),
                ],
                weights=[20, 30, 50],
            )[0]
        else:
            return random.randint(1960, 2020)

    def _generate_employee_count(self, org_type: str, size_category: str) -> int:
        """Generate realistic employee counts"""
        size_ranges = {
            "Government Agency": {
                "small": (10, 50),
                "medium": (50, 200),
                "large": (200, 1000),
            },
            "Municipal Fire Department": {
                "small": (20, 50),
                "medium": (50, 150),
                "large": (150, 400),
            },
            "Environmental NGO": {
                "small": (2, 10),
                "medium": (10, 30),
                "large": (30, 100),
            },
            "Community Group": {"small": (0, 2), "medium": (2, 5), "large": (5, 15)},
            "Emergency Services": {
                "small": (30, 80),
                "medium": (80, 200),
                "large": (200, 500),
            },
            "Forest Service": {
                "small": (5, 20),
                "medium": (20, 50),
                "large": (50, 150),
            },
            "Research Institution": {
                "small": (10, 30),
                "medium": (30, 100),
                "large": (100, 300),
            },
        }

        ranges = size_ranges.get(
            org_type, {"small": (1, 10), "medium": (10, 50), "large": (50, 200)}
        )
        return random.randint(*ranges[size_category])

    def _generate_annual_budget(self, org_type: str, employee_count: int) -> int:
        """Generate realistic annual budgets in SEK"""
        # Base cost per employee varies by org type
        cost_per_employee = {
            "Government Agency": random.randint(800000, 1200000),
            "Municipal Fire Department": random.randint(700000, 1000000),
            "Environmental NGO": random.randint(400000, 700000),
            "Community Group": random.randint(50000, 200000),
            "Emergency Services": random.randint(800000, 1100000),
            "Forest Service": random.randint(600000, 900000),
            "Research Institution": random.randint(900000, 1500000),
        }

        base_cost = cost_per_employee.get(org_type, 500000)

        # Add operational costs
        operational_multiplier = random.uniform(1.3, 2.0)

        return int(employee_count * base_cost * operational_multiplier)

    def _calculate_distribution_capability(self, org: Dict) -> Dict:
        """Calculate realistic distribution capabilities"""
        budget = org.get("annual_budget", 1000000)
        employees = org.get("employee_count", 10)
        org_type = org.get("Type", "Unknown")

        # Estimate portion of budget for distribution
        distribution_budget_ratio = {
            "Government Agency": 0.15,
            "Municipal Fire Department": 0.10,
            "Environmental NGO": 0.25,
            "Community Group": 0.40,
            "Emergency Services": 0.12,
            "Forest Service": 0.18,
            "Research Institution": 0.08,
        }

        ratio = distribution_budget_ratio.get(org_type, 0.15)
        distribution_budget = budget * ratio

        # Cost per leaflet (printing, handling, delivery)
        cost_per_leaflet = random.uniform(2.5, 5.0)

        # Annual capacity
        max_annual_leaflets = int(distribution_budget / cost_per_leaflet)

        # Daily capacity (working days ~250/year)
        max_daily_leaflets = int(max_annual_leaflets / 250)

        # Volunteer multiplier for community groups
        if org_type == "Community Group":
            volunteer_multiplier = random.uniform(2.0, 5.0)
            max_daily_leaflets = int(max_daily_leaflets * volunteer_multiplier)

        return {
            "max_annual": max_annual_leaflets,
            "max_daily": max_daily_leaflets,
            "typical_batch": int(max_daily_leaflets * random.uniform(0.3, 0.7)),
            "surge_capacity": int(
                max_daily_leaflets * random.uniform(1.5, 3.0)
            ),  # During campaigns
        }

    def _generate_peak_hours(self, org_type: str) -> List[int]:
        """Generate realistic peak activity hours"""
        if org_type in ["Government Agency", "Research Institution"]:
            # Standard office hours
            return list(range(9, 17))
        elif org_type == "Municipal Fire Department":
            # Two shifts
            return list(range(8, 12)) + list(range(14, 18))
        elif org_type == "Community Group":
            # Evenings and weekends
            return list(range(17, 20)) + list(range(10, 14))
        else:
            # Variable
            return list(range(8, 18))

    def _generate_volunteer_pattern(self, org_type: str) -> Dict:
        """Generate volunteer availability patterns"""
        if org_type == "Community Group":
            return {
                "weekday": random.uniform(0.3, 0.6),
                "weekend": random.uniform(0.7, 0.9),
                "summer": random.uniform(0.4, 0.7),
                "holiday": random.uniform(0.1, 0.3),
            }
        elif org_type == "Environmental NGO":
            return {
                "weekday": random.uniform(0.4, 0.7),
                "weekend": random.uniform(0.6, 0.8),
                "summer": random.uniform(0.5, 0.8),
                "holiday": random.uniform(0.2, 0.4),
            }
        else:
            # Professional organizations
            return {"weekday": 1.0, "weekend": 0.1, "summer": 0.7, "holiday": 0.0}

    def enhance_leaflets_realistically(self):
        """Enhance leaflets with ultra-realistic attributes"""

        category_keywords = {
            "emergency": [
                "akut",
                "emergency",
                "evakuering",
                "evacuation",
                "larm",
                "alarm",
                "nöd",
            ],
            "prevention": [
                "förebygg",
                "prevent",
                "säkerhet",
                "safety",
                "försiktig",
                "caution",
                "skydd",
            ],
            "forest": [
                "skog",
                "forest",
                "träd",
                "tree",
                "natur",
                "woodland",
                "vegetation",
            ],
            "community": [
                "samhälle",
                "community",
                "lokal",
                "local",
                "grann",
                "neighbor",
                "område",
            ],
            "environment": [
                "miljö",
                "environment",
                "natur",
                "nature",
                "djur",
                "wildlife",
                "ekologi",
            ],
            "regulation": [
                "lag",
                "law",
                "regel",
                "regulation",
                "förordning",
                "policy",
                "myndighet",
            ],
            "education": [
                "utbildning",
                "education",
                "lär",
                "learn",
                "kunskap",
                "knowledge",
                "information",
            ],
        }

        for leaflet in self.leaflets:
            title = leaflet.get("Title", "").lower()
            text = leaflet.get("Text", "").lower()
            content = f"{title} {text}"

            categories = []
            category_scores = {}

            for category, keywords in category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in content)
                if score > 0:
                    categories.append(category)
                    category_scores[category] = score

            if not categories:
                categories = ["general"]

            leaflet["categories"] = categories
            leaflet["primary_category"] = (
                max(categories, key=lambda c: category_scores.get(c, 0))
                if category_scores
                else "general"
            )

            leaflet["urgency_level"] = self._calculate_urgency(content, categories)
            leaflet["target_audience"] = self._determine_audience(categories, content)
            leaflet["reading_level"] = random.choices(
                ["basic", "intermediate", "advanced"], weights=[40, 50, 10]
            )[0]
            leaflet["language"] = "sv"
            leaflet["format"] = random.choices(
                ["brochure", "flyer", "poster", "booklet"], weights=[40, 35, 15, 10]
            )[0]
            leaflet["pages"] = random.choices([1, 2, 4, 8], weights=[50, 30, 15, 5])[0]

            # Add production details
            leaflet["production_cost"] = self._calculate_production_cost(leaflet)
            leaflet["effectiveness_score"] = random.uniform(0.5, 0.95)
            leaflet["requires_translation"] = (
                random.random() < 0.15
            )  # 15% need translation
            leaflet["seasonal_relevance"] = self._determine_seasonal_relevance(leaflet)
            leaflet["distribution_weight"] = random.choice(
                [20, 40, 60, 80, 100]
            )  # grams

    def _calculate_urgency(self, content: str, categories: List[str]) -> str:
        """Calculate urgency level based on content"""
        if "emergency" in categories or any(
            word in content for word in ["akut", "omedelbart", "genast"]
        ):
            return "critical"
        elif "prevention" in categories and any(
            word in content for word in ["viktig", "important", "allvarlig"]
        ):
            return "high"
        elif "education" in categories or "research" in categories:
            return "low"
        else:
            return "medium"

    def _determine_audience(self, categories: List[str], content: str) -> str:
        """Determine target audience from content"""
        if any(word in content for word in ["barn", "children", "skola", "school"]):
            return "families"
        elif "forest" in categories or any(
            word in content for word in ["skogsägare", "markägare"]
        ):
            return "landowners"
        elif "community" in categories:
            return "residents"
        elif "regulation" in categories:
            return "professionals"
        else:
            return "general"

    def _calculate_production_cost(self, leaflet: Dict) -> float:
        """Calculate realistic production cost per leaflet"""
        base_costs = {"flyer": 0.5, "brochure": 1.5, "poster": 3.0, "booklet": 5.0}

        format_type = leaflet.get("format", "flyer")
        base_cost = base_costs.get(format_type, 1.0)

        # Page multiplier
        pages = leaflet.get("pages", 1)
        page_multiplier = 1.0 + (pages - 1) * 0.3

        # Quality multiplier
        if leaflet.get("urgency_level") == "critical":
            quality_multiplier = 1.5  # Higher quality for critical messages
        else:
            quality_multiplier = 1.0

        # Color printing
        color_multiplier = 1.5 if random.random() < 0.7 else 1.0

        return round(
            base_cost * page_multiplier * quality_multiplier * color_multiplier, 2
        )

    def _determine_seasonal_relevance(self, leaflet: Dict) -> Dict:
        """Determine when leaflet is most relevant"""
        categories = leaflet.get("categories", [])
        urgency = leaflet.get("urgency_level", "medium")

        if urgency == "critical" or "emergency" in categories:
            # Always relevant but peaks in summer
            return {"spring": 0.7, "summer": 1.0, "autumn": 0.6, "winter": 0.4}
        elif "prevention" in categories:
            # Highest before fire season
            return {"spring": 1.0, "summer": 0.8, "autumn": 0.4, "winter": 0.3}
        elif "education" in categories:
            # Year-round but lower in summer vacation
            return {"spring": 0.9, "summer": 0.5, "autumn": 0.9, "winter": 0.8}
        else:
            # Default pattern
            return {"spring": 0.7, "summer": 0.9, "autumn": 0.6, "winter": 0.4}

    def generate_realistic_campaign(
        self, org: Dict, date: datetime.date, context: Dict
    ) -> Dict:
        """Generate ultra-realistic campaign based on context and triggers"""
        org_type = org.get("Type", "Unknown")
        year = str(date.year)
        month = date.month

        # Determine season
        if month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        elif month in [9, 10, 11]:
            season = "autumn"
        else:
            season = "winter"

        # Check for special events in this year
        year_context = self.campaign_patterns.get(year, {})
        special_events = year_context.get("special_events", [])
        intensity_modifier = year_context.get("summer_intensity", 1.0)

        # Use Gemini to generate campaign narrative if available
        campaign_narrative = self.gemini.generate_campaign_narrative(
            org_type, season, date.year
        )

        # Determine campaign type based on various factors
        if any(
            "extreme" in event or "major_fires" in event for event in special_events
        ):
            campaign_type = "emergency_response"
            urgency = random.randint(8, 10)
        elif season == "summer" and intensity_modifier > 1.2:
            campaign_type = "intensive_prevention"
            urgency = random.randint(6, 9)
        elif season == "spring":
            campaign_type = "seasonal_preparation"
            urgency = random.randint(4, 7)
        elif org_type == "Research Institution" and random.random() < 0.3:
            campaign_type = "research_dissemination"
            urgency = random.randint(2, 5)
        else:
            campaign_type = "routine_awareness"
            urgency = random.randint(3, 6)

        # Calculate duration based on campaign type and organization capacity
        capability = org.get("distribution_capability", {})
        max_daily = capability.get("max_daily", 100)

        duration_ranges = {
            "emergency_response": (7, 21),
            "intensive_prevention": (30, 90),
            "seasonal_preparation": (21, 60),
            "research_dissemination": (14, 30),
            "routine_awareness": (14, 45),
        }

        duration = random.randint(*duration_ranges[campaign_type])

        # Adjust duration based on budget and capacity
        if org.get("annual_budget", 0) < 1000000:
            duration = int(duration * 0.7)

        # Daily volume based on urgency and capacity
        if campaign_type == "emergency_response":
            daily_volume_min = int(max_daily * 0.8)
            daily_volume_max = capability.get("surge_capacity", max_daily * 2)
        else:
            daily_volume_min = int(max_daily * 0.3)
            daily_volume_max = int(max_daily * 0.7)

        # Geographic targeting based on campaign narrative
        geographic_targets = self._determine_geographic_targets(
            org, campaign_type, campaign_narrative.get("geographic_focus", [])
        )

        # Collaboration decisions
        collaboration_likelihood = {
            "emergency_response": 0.95,
            "intensive_prevention": 0.7,
            "seasonal_preparation": 0.5,
            "research_dissemination": 0.3,
            "routine_awareness": 0.4,
        }

        should_collaborate = random.random() < collaboration_likelihood[campaign_type]

        campaign = {
            "id": f"CAMP-{date.strftime('%Y%m%d')}-{org['ID']}-{random.randint(1000, 9999)}",
            "type": campaign_type,
            "narrative": campaign_narrative,
            "urgency": urgency,
            "lead_org": org["ID"],
            "participating_orgs": [org["ID"]],
            "start_date": date,
            "end_date": date + datetime.timedelta(days=duration),
            "daily_volume_min": daily_volume_min,
            "daily_volume_max": daily_volume_max,
            "geographic_targets": geographic_targets,
            "demographic_focus": campaign_narrative.get(
                "target_demographics", ["general"]
            ),
            "special_context": special_events,
            "budget_allocated": self._calculate_campaign_budget(
                org, duration, daily_volume_max
            ),
            "distribution_strategy": campaign_narrative.get(
                "distribution_strategy", "standard"
            ),
            "success_metrics": self._define_success_metrics(campaign_type),
        }

        # Add collaborators if needed
        if should_collaborate:
            collaborators = self._find_realistic_collaborators(
                org, campaign_type, geographic_targets
            )
            campaign["participating_orgs"].extend([c["ID"] for c in collaborators])

            # Adjust budget and capacity with collaboration
            for collaborator in collaborators:
                collab_capability = collaborator.get("distribution_capability", {})
                campaign["daily_volume_max"] += int(
                    collab_capability.get("max_daily", 50) * 0.5
                )
                campaign["budget_allocated"] += self._calculate_campaign_budget(
                    collaborator, duration, collab_capability.get("max_daily", 50) * 0.5
                )

        return campaign

    def _determine_geographic_targets(
        self, org: Dict, campaign_type: str, narrative_focus: List[str]
    ) -> List[Dict]:
        """Determine realistic geographic targets for campaign"""
        targets = []
        home_city = org.get("home_city", "Stockholm")
        reach = org.get("geographic_reach", "local")

        # Use city selector if available
        if hasattr(self, "city_selector") and self.city_selector:
            if reach == "national":
                # National campaigns focus on high-risk areas
                if campaign_type == "emergency_response":
                    # Get cities weighted for emergency response
                    weighted_cities = (
                        self.city_selector.get_weighted_cities_for_campaign(
                            campaign_type, org.get("Type"), narrative_focus
                        )
                    )
                    # Select top weighted cities
                    top_cities = sorted(
                        weighted_cities, key=lambda x: x[1], reverse=True
                    )[:10]

                    for city, weight in top_cities:
                        targets.append(
                            {
                                "city": city["name"],
                                "priority": "high" if weight > 0.05 else "medium",
                                "neighborhoods": self._get_neighborhoods_for_city(
                                    city["name"], campaign_type
                                ),
                                "population": city["population"],
                                "fire_risk": city["fire_risk"],
                            }
                        )
                else:
                    # Balanced national coverage
                    major_cities = self.city_selector.get_cities_by_type(
                        ["major_city", "city"]
                    )
                    selected_cities = random.sample(
                        major_cities, min(8, len(major_cities))
                    )

                    for city in selected_cities:
                        targets.append(
                            {
                                "city": city["name"],
                                "priority": "high"
                                if city["name"] == home_city
                                else "medium",
                                "neighborhoods": self._get_neighborhoods_for_city(
                                    city["name"], campaign_type
                                ),
                                "population": city["population"],
                                "fire_risk": city["fire_risk"],
                            }
                        )

            elif reach == "regional":
                # Regional campaigns focus on nearby areas
                home_city_data = org.get("city_data")
                if home_city_data:
                    nearby_cities = self.city_selector.get_nearby_cities(
                        home_city_data["name"], radius_km=200
                    )

                    # Add home city first
                    targets.append(
                        {
                            "city": home_city_data["name"],
                            "priority": "high",
                            "neighborhoods": self._get_neighborhoods_for_city(
                                home_city_data["name"], campaign_type
                            ),
                            "population": home_city_data["population"],
                            "fire_risk": home_city_data["fire_risk"],
                        }
                    )

                    # Add nearby cities
                    for city in nearby_cities[:5]:
                        priority = (
                            "high"
                            if city["distance_km"] < 50
                            else "medium"
                            if city["distance_km"] < 100
                            else "low"
                        )
                        targets.append(
                            {
                                "city": city["name"],
                                "priority": priority,
                                "neighborhoods": self._get_neighborhoods_for_city(
                                    city["name"], campaign_type
                                ),
                                "population": city["population"],
                                "fire_risk": city["fire_risk"],
                                "distance_km": city["distance_km"],
                            }
                        )

            else:  # local or very_local
                # Focus intensely on home city
                home_city_data = org.get("city_data", {})
                if home_city_data:
                    targets.append(
                        {
                            "city": home_city_data.get("name", home_city),
                            "priority": "critical"
                            if campaign_type == "emergency_response"
                            else "high",
                            "neighborhoods": self._get_neighborhoods_for_city(
                                home_city, campaign_type
                            ),
                            "population": home_city_data.get("population", 0),
                            "fire_risk": home_city_data.get("fire_risk", 0.7),
                        }
                    )
        else:
            # Fallback to original method
            if reach == "national":
                # National campaigns focus on high-risk areas
                if campaign_type == "emergency_response":
                    # Target areas with highest fire risk
                    risk_cities = sorted(
                        self.swedish_cities.items(),
                        key=lambda x: x[1].get("fire_risk", 0),
                        reverse=True,
                    )[:5]
                    for city, data in risk_cities:
                        targets.append(
                            {
                                "city": city,
                                "priority": "high",
                                "neighborhoods": self._select_target_neighborhoods(
                                    city, "high_risk"
                                ),
                            }
                        )
                else:
                    # Balanced national coverage
                    major_cities = ["Stockholm", "Göteborg", "Malmö", "Uppsala"]
                    for city in major_cities:
                        targets.append(
                            {
                                "city": city,
                                "priority": "high" if city == home_city else "medium",
                                "neighborhoods": self._select_target_neighborhoods(
                                    city, "mixed"
                                ),
                            }
                        )

            elif reach == "regional":
                # Regional campaigns focus on nearby areas
                base_data = self.swedish_cities.get(home_city, {})
                base_lat, base_lon = (
                    base_data.get("lat", 59.0),
                    base_data.get("lon", 18.0),
                )

                # Find cities within ~200km
                regional_cities = []
                for city, data in self.swedish_cities.items():
                    dist_km = self._calculate_distance_km(
                        base_lat, base_lon, data.get("lat", 59.0), data.get("lon", 18.0)
                    )
                    if dist_km <= 200:
                        regional_cities.append((city, dist_km))

                # Sort by distance and select closest
                regional_cities.sort(key=lambda x: x[1])
                for city, dist in regional_cities[:5]:
                    priority = (
                        "high" if dist < 50 else "medium" if dist < 100 else "low"
                    )
                    targets.append(
                        {
                            "city": city,
                            "priority": priority,
                            "neighborhoods": self._select_target_neighborhoods(
                                city, campaign_type
                            ),
                        }
                    )

            else:  # local or very_local
                # Focus intensely on home city
                city_data = self.swedish_cities.get(home_city, {})
                neighborhood_types = city_data.get("neighborhood_types", {})

                if campaign_type == "emergency_response":
                    # Target all neighborhoods
                    all_neighborhoods = []
                    for ntype, nlist in neighborhood_types.items():
                        all_neighborhoods.extend(nlist)
                    targets.append(
                        {
                            "city": home_city,
                            "priority": "critical",
                            "neighborhoods": all_neighborhoods,
                        }
                    )
                else:
                    # Target specific neighborhood types
                    if org.get("Type") == "Community Group":
                        # Community groups know their specific area
                        target_neighborhoods = random.sample(
                            sum(neighborhood_types.values(), []),
                            min(3, len(sum(neighborhood_types.values(), []))),
                        )
                    else:
                        # Other orgs target by demographics
                        target_neighborhoods = self._select_target_neighborhoods(
                            home_city, campaign_type
                        )

                    targets.append(
                        {
                            "city": home_city,
                            "priority": "high",
                            "neighborhoods": target_neighborhoods,
                        }
                    )

        return targets

    def _get_neighborhoods_for_city(
        self, city_name: str, campaign_type: str
    ) -> List[str]:
        """Get neighborhoods for a city, using stored data or generating them"""
        # Check if we have predefined neighborhoods for this city
        if city_name in self.swedish_cities:
            city_data = self.swedish_cities[city_name]
            neighborhood_types = city_data.get("neighborhood_types", {})

            if neighborhood_types:
                return self._select_target_neighborhoods(city_name, campaign_type)

        # Generate generic neighborhoods for smaller cities
        return self._generate_generic_neighborhoods()

    def _generate_generic_neighborhoods(self) -> List[str]:
        """Generate generic neighborhood names for cities without predefined ones"""
        neighborhoods = []

        # Common Swedish neighborhood patterns
        prefixes = ["Centrum", "Norra", "Södra", "Östra", "Västra", "Gamla", "Nya"]
        suffixes = ["", "staden", "området", "parken", "torget", "gården"]

        # Generate 3-5 neighborhoods
        num_neighborhoods = random.randint(3, 5)
        used_combinations = set()

        while len(neighborhoods) < num_neighborhoods:
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes)
            name = f"{prefix}{suffix}".strip()

            if name not in used_combinations:
                neighborhoods.append(name)
                used_combinations.add(name)

        return neighborhoods

    def _calculate_distance_km(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in kilometers"""
        R = 6371  # Earth's radius in kilometers

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _select_target_neighborhoods(self, city: str, focus_type: str) -> List[str]:
        """Select neighborhoods based on campaign focus"""
        city_data = self.swedish_cities.get(city, {})
        neighborhood_types = city_data.get("neighborhood_types", {})

        if not neighborhood_types:
            # Generate generic neighborhoods
            return self._generate_generic_neighborhoods()

        selected = []

        if focus_type == "high_risk":
            # Prioritize family neighborhoods and suburbs
            selected.extend(neighborhood_types.get("family", []))
            selected.extend(neighborhood_types.get("suburban", []))
        elif focus_type == "emergency_response":
            # All neighborhoods
            for nlist in neighborhood_types.values():
                selected.extend(nlist)
        elif focus_type == "targeted_education":
            # Specific demographics
            selected.extend(neighborhood_types.get("student", []))
            selected.extend(neighborhood_types.get("diverse", []))
        else:
            # Mixed selection
            for ntype, nlist in neighborhood_types.items():
                if nlist:
                    selected.extend(random.sample(nlist, min(2, len(nlist))))

        return selected[:10]  # Limit to 10 neighborhoods

    def _calculate_campaign_budget(
        self, org: Dict, duration: int, daily_volume: int
    ) -> int:
        """Calculate realistic campaign budget"""
        # Base cost per leaflet (production + distribution)
        cost_per_leaflet = 3.5

        # Total leaflets
        total_leaflets = (
            duration * daily_volume * 0.7
        )  # Assume 70% of max capacity on average

        # Direct costs
        direct_costs = total_leaflets * cost_per_leaflet

        # Overhead (planning, coordination, evaluation)
        overhead_rate = 0.25 if org.get("Type") == "Government Agency" else 0.15
        overhead = direct_costs * overhead_rate

        # Staff costs
        daily_staff_cost = org.get("employee_count", 10) * 500  # Rough estimate
        staff_costs = daily_staff_cost * duration

        total_budget = int(direct_costs + overhead + staff_costs)

        # Cap at reasonable portion of annual budget
        annual_budget = org.get("annual_budget", 1000000)
        max_campaign_budget = annual_budget * 0.15  # Max 15% on single campaign

        return min(total_budget, int(max_campaign_budget))

    def _define_success_metrics(self, campaign_type: str) -> Dict:
        """Define realistic success metrics for campaign"""
        metrics = {
            "emergency_response": {
                "primary": "areas_covered",
                "target_coverage": 0.9,
                "response_time_hours": 24,
                "follow_up_required": True,
            },
            "intensive_prevention": {
                "primary": "households_reached",
                "target_coverage": 0.7,
                "repeat_contact_rate": 0.3,
                "behavior_change_goal": 0.2,
            },
            "seasonal_preparation": {
                "primary": "at_risk_contacted",
                "target_coverage": 0.6,
                "information_retention": 0.4,
                "action_taken_rate": 0.15,
            },
            "research_dissemination": {
                "primary": "stakeholders_informed",
                "target_coverage": 0.5,
                "further_inquiries": 0.1,
                "policy_influence": 0.05,
            },
            "routine_awareness": {
                "primary": "general_reach",
                "target_coverage": 0.4,
                "brand_recognition": 0.3,
                "website_traffic_increase": 0.1,
            },
        }

        return metrics.get(campaign_type, metrics["routine_awareness"])

    def _find_realistic_collaborators(
        self, lead_org: Dict, campaign_type: str, targets: List[Dict]
    ) -> List[Dict]:
        """Find realistic collaboration partners"""
        potential_collaborators = []
        lead_type = lead_org.get("Type")
        lead_city = lead_org.get("home_city")

        # Define collaboration affinity
        collaboration_affinity = {
            "Government Agency": [
                "Municipal Fire Department",
                "Emergency Services",
                "Forest Service",
            ],
            "Municipal Fire Department": [
                "Emergency Services",
                "Government Agency",
                "Community Group",
            ],
            "Environmental NGO": [
                "Research Institution",
                "Community Group",
                "Forest Service",
            ],
            "Emergency Services": ["Municipal Fire Department", "Government Agency"],
            "Forest Service": [
                "Government Agency",
                "Environmental NGO",
                "Research Institution",
            ],
            "Research Institution": ["Government Agency", "Environmental NGO"],
            "Community Group": [
                "Municipal Fire Department",
                "Environmental NGO",
                "Community Group",
            ],
        }

        preferred_types = collaboration_affinity.get(lead_type, [])

        # Find organizations that match criteria
        for org in self.organizations:
            if org["ID"] == lead_org["ID"]:
                continue

            # Check type affinity
            if org.get("Type") not in preferred_types:
                continue

            # Check geographic overlap
            org_city = org.get("home_city")
            org_reach = org.get("geographic_reach")

            geographic_match = False
            if org_reach == "national":
                geographic_match = True
            elif org_reach == "regional":
                # Check if any target cities are within regional reach
                for target in targets:
                    if self._cities_within_region(org_city, target["city"]):
                        geographic_match = True
                        break
            elif org_city == lead_city:
                geographic_match = True

            if not geographic_match:
                continue

            # Check capacity and availability
            if random.random() < org.get("collaboration_rate", 0.5):
                potential_collaborators.append(org)

        # Select collaborators based on campaign type
        if campaign_type == "emergency_response":
            # Emergency campaigns involve many partners
            num_collaborators = min(len(potential_collaborators), random.randint(3, 8))
        elif campaign_type == "intensive_prevention":
            num_collaborators = min(len(potential_collaborators), random.randint(2, 5))
        else:
            num_collaborators = min(len(potential_collaborators), random.randint(1, 3))

        # Prioritize by relevance
        potential_collaborators.sort(
            key=lambda x: (
                x.get("Type") in ["Municipal Fire Department", "Emergency Services"],
                x.get("home_city") == lead_city,
                x.get("collaboration_rate", 0),
            ),
            reverse=True,
        )

        return potential_collaborators[:num_collaborators]

    def _cities_within_region(self, city1: str, city2: str) -> bool:
        """Check if two cities are within the same region"""
        # Use city selector if available
        if hasattr(self, "city_selector") and self.city_selector:
            city1_data = self.city_selector.city_details.get(city1)
            city2_data = self.city_selector.city_details.get(city2)

            if city1_data and city2_data:
                dist = self._calculate_distance_km(
                    city1_data["lat"],
                    city1_data["lon"],
                    city2_data["lat"],
                    city2_data["lon"],
                )
                return dist <= 200

        # Fallback to swedish_cities
        c1_data = self.swedish_cities.get(city1, {})
        c2_data = self.swedish_cities.get(city2, {})

        dist = self._calculate_distance_km(
            c1_data.get("lat", 59),
            c1_data.get("lon", 18),
            c2_data.get("lat", 59),
            c2_data.get("lon", 18),
        )

        return dist <= 200  # Within 200km

    def plan_distribution_route(
        self, org: Dict, date: datetime.date, campaign: Dict, batch_size: int
    ) -> List[Dict]:
        """Plan realistic distribution route with geographic clustering"""
        routes = []

        # Get target areas from campaign
        targets = campaign.get("geographic_targets", [])
        if not targets:
            return routes

        # Prioritize targets
        high_priority = [t for t in targets if t["priority"] in ["critical", "high"]]
        medium_priority = [t for t in targets if t["priority"] == "medium"]
        low_priority = [t for t in targets if t["priority"] == "low"]

        # Start with highest priority
        all_targets = high_priority + medium_priority + low_priority

        distribution_pattern = org.get("distribution_patterns", {})
        route_efficiency = distribution_pattern.get("route_efficiency", 0.5)
        batch_clustering = distribution_pattern.get("batch_clustering", 0.7)

        # Create route segments
        current_city = None
        current_neighborhoods = []
        distributions_planned = 0

        for target in all_targets:
            if distributions_planned >= batch_size:
                break

            city = target["city"]
            neighborhoods = target["neighborhoods"]

            if not neighborhoods:
                continue

            # Decide how many distributions per neighborhood
            if campaign["type"] == "emergency_response":
                per_neighborhood = random.randint(20, 50)
            elif target["priority"] == "critical":
                per_neighborhood = random.randint(15, 30)
            else:
                per_neighborhood = random.randint(5, 20)

            # Plan route through neighborhoods
            if route_efficiency > 0.7:
                # Efficient routing - optimize path
                neighborhoods = self._optimize_neighborhood_route(city, neighborhoods)
            else:
                # Less efficient - some randomness
                random.shuffle(neighborhoods)

            for neighborhood in neighborhoods:
                if distributions_planned >= batch_size:
                    break

                # Check saturation
                nh_key = f"{city}_{neighborhood}"
                nh_history = self.neighborhood_history[nh_key]

                # Avoid over-saturation unless emergency
                if (
                    nh_history["saturation_level"] > 0.7
                    and campaign["type"] != "emergency_response"
                    and random.random() > 0.3
                ):
                    continue

                # Determine actual number for this neighborhood
                actual_count = min(per_neighborhood, batch_size - distributions_planned)

                if batch_clustering > 0.8:
                    # High clustering - deliver to adjacent addresses
                    actual_count = int(actual_count * random.uniform(0.8, 1.0))

                # Create route segment
                segment = {
                    "city": city,
                    "neighborhood": neighborhood,
                    "count": actual_count,
                    "priority": target["priority"],
                    "sequence": len(routes),
                }

                routes.append(segment)
                distributions_planned += actual_count

                # Update saturation
                nh_history["saturation_level"] = min(
                    1.0, nh_history["saturation_level"] + actual_count / 1000
                )

        return routes

    def _optimize_neighborhood_route(
        self, city: str, neighborhoods: List[str]
    ) -> List[str]:
        """Optimize route through neighborhoods (simplified TSP)"""
        if len(neighborhoods) <= 3:
            return neighborhoods

        # Simple nearest-neighbor heuristic
        optimized = [neighborhoods[0]]
        remaining = neighborhoods[1:]

        while remaining:
            current = optimized[-1]
            # Find "nearest" neighborhood (simplified - could use actual coordinates)
            nearest = min(remaining, key=lambda x: abs(hash(current) - hash(x)))
            optimized.append(nearest)
            remaining.remove(nearest)

        return optimized

    def generate_distribution_batch(
        self, org: Dict, date: datetime.date, campaign: Dict = None
    ) -> List[Dict]:
        """Generate ultra-realistic distribution batch with route planning"""
        distributions = []

        # Determine batch size based on multiple factors
        capability = org.get("distribution_capability", {})
        base_daily = capability.get("typical_batch", 100)

        # Adjust for day of week
        weekday = date.weekday()
        if weekday >= 5:  # Weekend
            volunteer_avail = org.get("volunteer_availability", {})
            if org.get("Type") == "Community Group":
                daily_capacity = int(base_daily * volunteer_avail.get("weekend", 0.8))
            else:
                daily_capacity = int(base_daily * org.get("weekend_activity", 0.3))
        else:
            daily_capacity = base_daily

        # Adjust for campaign
        if campaign:
            if campaign["type"] == "emergency_response":
                daily_capacity = capability.get("surge_capacity", daily_capacity * 2)
            else:
                # Use campaign specified volumes
                daily_capacity = random.randint(
                    campaign["daily_volume_min"],
                    min(
                        campaign["daily_volume_max"],
                        capability.get("surge_capacity", daily_capacity),
                    ),
                )

        # Plan distribution route
        if campaign:
            route_plan = self.plan_distribution_route(
                org, date, campaign, daily_capacity
            )
        else:
            # Simple local distribution
            route_plan = [
                {
                    "city": org.get("home_city", "Stockholm"),
                    "neighborhood": self._generate_neighborhood_name(),
                    "count": daily_capacity,
                    "priority": "routine",
                    "sequence": 0,
                }
            ]

        # Generate distributions following route plan
        org_leaflets = [l for l in self.leaflets if l["Org_ID"] == org["ID"]]
        if not org_leaflets:
            return []

        # Select leaflets based on campaign and season
        leaflet_weights = self._calculate_leaflet_weights(org_leaflets, campaign, date)

        for segment in route_plan:
            city = segment["city"]
            neighborhood = segment["neighborhood"]
            count = segment["count"]

            # Get neighborhood profile
            profile_key = f"{city}_{neighborhood}"
            if profile_key not in self.geocoder.neighborhood_profiles:
                self.geocoder.neighborhood_profiles[profile_key] = (
                    self.geocoder.create_neighborhood_profile(city, neighborhood)
                )

            neighborhood_profile = self.geocoder.neighborhood_profiles[profile_key]

            # Generate recipients based on neighborhood profile
            for i in range(count):
                # Select leaflet (may vary by recipient)
                leaflet = random.choices(org_leaflets, weights=leaflet_weights)[0]

                # Generate recipient matching neighborhood profile
                recipient = self.generate_realistic_recipient(
                    city, neighborhood, neighborhood_profile, date, campaign
                )

                # Get precise coordinates
                lat, lon = self.geocoder.get_neighborhood_coordinates(
                    city, neighborhood
                )

                distribution = {
                    "Distribution_ID": self.next_distribution_id,
                    "Leaflet_ID": leaflet["Leaflet_ID"],
                    "Org_ID": org["ID"],
                    "Delivered_Date": date.strftime("%Y-%m-%d"),
                    "Delivered_City": city,
                    "Delivered_Neighbourhood": neighborhood,
                    "Addressee_Name": recipient["name"],
                    "Addressee_Gender": recipient["gender"],
                    "Addressee_Age": recipient["age"],
                    "Annual_Income_SEK": recipient["income"],
                    "Property_Value_SEK": recipient["property_value"],
                    "Household_Cars": recipient["cars"],
                    "Household_Size": recipient["household_size"],
                    "Latitude": lat,
                    "Longitude": lon,
                }

                distributions.append(distribution)
                self.next_distribution_id += 1

            # Update neighborhood history
            nh_history = self.neighborhood_history[profile_key]
            nh_history["last_distribution"] = date
            nh_history["total_distributions"] += count

        return distributions

    def _calculate_leaflet_weights(
        self, leaflets: List[Dict], campaign: Dict, date: datetime.date
    ) -> List[float]:
        """Calculate realistic weights for leaflet selection"""
        weights = []
        season = self._get_season(date)

        for leaflet in leaflets:
            weight = 1.0

            # Campaign relevance
            if campaign:
                if (
                    campaign["type"] == "emergency_response"
                    and leaflet.get("urgency_level") == "critical"
                ):
                    weight *= 5.0
                elif campaign[
                    "type"
                ] == "intensive_prevention" and "prevention" in leaflet.get(
                    "categories", []
                ):
                    weight *= 3.0
                elif campaign[
                    "type"
                ] == "research_dissemination" and "education" in leaflet.get(
                    "categories", []
                ):
                    weight *= 4.0

            # Seasonal relevance
            seasonal_relevance = leaflet.get("seasonal_relevance", {})
            weight *= seasonal_relevance.get(season, 0.5)

            # Effectiveness score
            weight *= leaflet.get("effectiveness_score", 0.7)

            # Cost considerations (cheaper leaflets might be preferred for large campaigns)
            if campaign and campaign.get("daily_volume_max", 0) > 500:
                cost = leaflet.get("production_cost", 1.0)
                weight *= 2.0 / cost  # Inverse relationship with cost

            weights.append(weight)

        return weights

    def _get_season(self, date: datetime.date) -> str:
        """Get season from date"""
        month = date.month
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "autumn"
        else:
            return "winter"

    def generate_realistic_recipient(
        self,
        city: str,
        neighborhood: str,
        profile: NeighborhoodProfile,
        date: datetime.date,
        campaign: Dict = None,
    ) -> Dict:
        """Generate recipient with characteristics matching neighborhood profile"""

        # Age distribution based on neighborhood
        if profile.student_population > 0.5:
            age = int(np.random.normal(25, 5))
            age = max(18, min(35, age))
        elif profile.elderly_population > 0.3:
            age = int(np.random.normal(70, 10))
            age = max(55, min(95, age))
        elif profile.family_friendliness > 0.7:
            # Parent age distribution
            age = int(np.random.normal(40, 8))
            age = max(28, min(55, age))
        else:
            # Use neighborhood average with realistic variation
            age = int(np.random.normal(profile.avg_age, 12))
            age = max(18, min(95, age))

        # Gender with slight demographic variations
        if profile.elderly_population > 0.3:
            # Slight female bias in elderly populations
            gender = random.choices(
                ["Female", "Male", "Other"], weights=[52, 47.8, 0.2]
            )[0]
        else:
            gender = random.choices(
                ["Female", "Male", "Other"], weights=[50.2, 49.6, 0.2]
            )[0]

        # Name generation based on neighborhood diversity
        name = self._generate_realistic_name(gender, profile.ethnic_diversity)

        # Income based on neighborhood profile and age
        income = self._generate_realistic_income(age, profile, city)

        # Household composition
        household = self._generate_household_composition(age, profile, income)

        # Property value
        property_value = self._generate_property_value(
            age, income, profile, household["size"]
        )

        # Car ownership
        cars = self._generate_car_ownership(household["size"], income, profile)

        return {
            "name": name,
            "gender": gender,
            "age": age,
            "income": income,
            "property_value": property_value,
            "household_size": household["size"],
            "cars": cars,
            "neighbourhood": neighborhood,
        }

    def _generate_realistic_name(self, gender: str, ethnic_diversity: float) -> str:
        """Generate name based on demographic patterns"""
        # Determine name pattern based on diversity
        pattern_choice = random.random()

        if pattern_choice < (1 - ethnic_diversity):
            pattern = "traditional_swedish"
        elif pattern_choice < (1 - ethnic_diversity * 0.3):
            pattern = "modern_swedish"
        else:
            pattern = "immigrant_names"

        name_data = self.name_patterns[pattern]

        if gender == "Other":
            gender_key = random.choice(["male", "female"])
        else:
            gender_key = gender.lower()

        first_name = random.choice(name_data["first_names"][gender_key])
        last_name = random.choice(name_data["last_names"])

        # Double-barreled surnames more common in certain demographics
        if pattern == "modern_swedish" and random.random() < 0.15:
            last_name = f"{last_name}-{random.choice(self.name_patterns['traditional_swedish']['last_names'])}"

        return f"{first_name} {last_name}"

    def _generate_realistic_income(
        self, age: int, profile: NeighborhoodProfile, city: str
    ) -> int:
        """Generate income with realistic distributions"""
        # Try to get city data from city selector first
        if hasattr(self, "city_selector") and self.city_selector:
            city_details = self.city_selector.city_details.get(city)
            if city_details:
                # Use population as proxy for income level
                pop = city_details["population"]
                if pop > 100000:
                    base_mean = 450000
                    base_std = 200000
                elif pop > 50000:
                    base_mean = 400000
                    base_std = 150000
                elif pop > 20000:
                    base_mean = 370000
                    base_std = 130000
                else:
                    base_mean = 340000
                    base_std = 110000

                income_dist = {"mean": base_mean, "std": base_std, "skew": 0.8}
            else:
                # Fallback
                income_dist = {"mean": 380000, "std": 140000, "skew": 0.7}
        else:
            # Use swedish_cities data
            city_data = self.swedish_cities.get(city, {})
            income_dist = city_data.get(
                "income_distribution", {"mean": 400000, "std": 150000, "skew": 1.0}
            )

        # Base income from city distribution
        base_mean = income_dist["mean"] * profile.avg_income_multiplier
        base_std = income_dist["std"]

        # Age-based adjustments
        if age < 25:
            age_multiplier = 0.4
        elif age < 30:
            age_multiplier = 0.7
        elif age < 35:
            age_multiplier = 0.85
        elif age < 45:
            age_multiplier = 1.0
        elif age < 55:
            age_multiplier = 1.1
        elif age < 65:
            age_multiplier = 1.05
        elif age < 70:
            age_multiplier = 0.6  # Retirement
        else:
            age_multiplier = 0.4

        # Student adjustment
        if profile.student_population > 0.5 and age < 30:
            age_multiplier *= 0.5

        # Generate with skewed distribution
        if income_dist["skew"] > 1.0:
            # Use log-normal for right-skewed distribution
            income = np.random.lognormal(np.log(base_mean * age_multiplier), 0.5)
        else:
            # Use normal distribution
            income = np.random.normal(base_mean * age_multiplier, base_std * 0.8)

        # Apply realistic bounds
        if age < 25:
            income = max(80000, min(income, 350000))
        elif profile.socioeconomic_class == "upper":
            income = max(400000, min(income, 5000000))
        elif profile.socioeconomic_class == "lower":
            income = max(150000, min(income, 400000))
        else:
            income = max(200000, min(income, 1500000))

        return int(income)

    def _generate_household_composition(
        self, age: int, profile: NeighborhoodProfile, income: int
    ) -> Dict:
        """Generate realistic household composition"""

        # Single household probability
        single_prob = profile.single_household_rate

        # Adjust for age
        if age < 25:
            single_prob = min(0.9, single_prob * 1.5)
        elif age > 70:
            single_prob = min(0.8, single_prob * 1.3)

        if random.random() < single_prob:
            return {"size": 1, "type": "single"}

        # Couple or family
        if profile.family_friendliness > 0.7 and 30 <= age <= 50:
            # High chance of family with children
            if random.random() < 0.7:
                # Family with children
                num_children = random.choices([1, 2, 3, 4], weights=[35, 45, 15, 5])[0]
                return {"size": 2 + num_children, "type": "family"}
            else:
                # Couple
                return {"size": 2, "type": "couple"}
        elif age < 30:
            # Young couple or roommates
            if random.random() < 0.6:
                return {"size": 2, "type": "couple"}
            else:
                # Roommates
                return {
                    "size": random.choices([2, 3, 4], weights=[50, 35, 15])[0],
                    "type": "shared",
                }
        elif age > 65:
            # Elderly couple or single
            if random.random() < 0.6:
                return {"size": 2, "type": "elderly_couple"}
            else:
                return {"size": 1, "type": "elderly_single"}
        else:
            # Mixed
            return {
                "size": random.choices([1, 2, 3, 4], weights=[30, 40, 20, 10])[0],
                "type": "mixed",
            }

    def _generate_property_value(
        self, age: int, income: int, profile: NeighborhoodProfile, household_size: int
    ) -> int:
        """Generate realistic property values"""

        # Ownership probability based on age and income
        ownership_prob = 0.0
        if age < 25:
            ownership_prob = 0.05 if income > 300000 else 0.02
        elif age < 35:
            ownership_prob = min(0.6, income / 1000000)
        elif age < 65:
            ownership_prob = min(0.85, income / 500000)
        else:
            ownership_prob = 0.7

        # Adjust for neighborhood
        if profile.socioeconomic_class in ["lower", "lower-middle"]:
            ownership_prob *= 0.7
        elif profile.socioeconomic_class in ["upper", "upper-middle"]:
            ownership_prob *= 1.2

        # Student areas have lower ownership
        if profile.student_population > 0.5:
            ownership_prob *= 0.3

        if random.random() > ownership_prob:
            return 0  # Renter

        # Calculate property value based on income and neighborhood
        value_to_income_ratio = random.uniform(4, 8)

        # Adjust for neighborhood
        value_to_income_ratio *= profile.property_value_modifier

        # Urban core is more expensive
        if profile.urbanization == "urban-core":
            value_to_income_ratio *= 1.3

        # Larger households need bigger properties
        if household_size > 3:
            value_to_income_ratio *= 1.2

        property_value = int(income * value_to_income_ratio)

        # Apply realistic bounds based on neighborhood
        if profile.socioeconomic_class == "upper":
            property_value = max(3000000, min(property_value, 20000000))
        elif profile.socioeconomic_class == "lower":
            property_value = max(800000, min(property_value, 2500000))
        else:
            property_value = max(1500000, min(property_value, 8000000))

        return property_value

    def _generate_car_ownership(
        self, household_size: int, income: int, profile: NeighborhoodProfile
    ) -> int:
        """Generate realistic car ownership"""

        # Base car ownership on urbanization
        if profile.urbanization == "urban-core":
            if income < 300000:
                car_weights = [70, 28, 2, 0]
            elif income < 500000:
                car_weights = [50, 40, 10, 0]
            else:
                car_weights = [30, 50, 18, 2]
        elif profile.urbanization == "suburban":
            if household_size == 1:
                car_weights = [15, 75, 10, 0]
            elif household_size == 2:
                car_weights = [10, 60, 28, 2]
            else:
                car_weights = [5, 40, 45, 10]
        else:  # rural
            if household_size == 1:
                car_weights = [10, 80, 10, 0]
            else:
                car_weights = [5, 35, 50, 10]

        # Income adjustment
        if income < 200000:
            car_weights[0] += 20
            car_weights[1] -= 10
            car_weights[2] -= 10
            car_weights[3] = 0
        elif income > 800000:
            car_weights[0] = max(0, car_weights[0] - 10)
            car_weights[2] += 5
            car_weights[3] += 5

        # Apply car ownership modifier
        car_weights = [w * profile.car_ownership_modifier for w in car_weights]

        # Normalize weights
        total = sum(car_weights)
        car_weights = [w / total for w in car_weights]

        return random.choices([0, 1, 2, 3], weights=car_weights)[0]

    def generate_distributions(self, num_distributions: int = 10000) -> List[Dict]:
        """Generate ultra-realistic distributions with complex patterns"""
        distributions = []
        current_date = datetime.date(2014, 1, 1)
        end_date = datetime.date(2024, 12, 31)

        print(f"\n🚀 Generating {num_distributions:,} ultra-realistic distributions...")
        print("   Creating natural patterns with:")
        print("   • Neighborhood demographic clustering")
        print("   • Realistic campaign responses to events")
        print("   • Route-optimized delivery patterns")
        print("   • Income and property correlations")
        print("   • Seasonal and temporal variations")
        if hasattr(self, "city_selector") and self.city_selector:
            print(
                f"   • {len(self.city_selector.all_cities)} real Swedish locations from GeoNames"
            )

        # Track progress
        last_progress = 0
        start_time = time.time()

        # Generate day by day
        while current_date <= end_date and len(distributions) < num_distributions:
            # Check for special events or triggers
            year_str = str(current_date.year)
            year_context = self.campaign_patterns.get(year_str, {})

            # Determine if new campaigns should start
            for org in self.organizations:
                if len(distributions) >= num_distributions:
                    break

                org_state = self.org_state[org["ID"]]

                # Skip if in cooldown
                if (
                    org_state.get("cooldown_until")
                    and current_date < org_state["cooldown_until"]
                ):
                    continue

                # Check if should start new campaign
                should_start_campaign = False

                # Event-driven campaigns
                if any(
                    "extreme" in event or "fire" in event
                    for event in year_context.get("special_events", [])
                ):
                    if org.get("response_to_events", 0) > random.random():
                        should_start_campaign = True

                # Seasonal campaigns
                elif current_date.month in [4, 5] and not org_state.get(
                    "current_campaign"
                ):
                    if random.random() < 0.05 * org.get("activity_level", 0.5):
                        should_start_campaign = True

                # Regular campaigns
                elif not org_state.get("current_campaign"):
                    days_since_last = 30
                    if org_state.get("last_activity"):
                        days_since_last = (
                            current_date - org_state["last_activity"]
                        ).days

                    if days_since_last > 30 and random.random() < 0.02 * org.get(
                        "activity_level", 0.5
                    ):
                        should_start_campaign = True

                if should_start_campaign:
                    campaign = self.generate_realistic_campaign(
                        org, current_date, year_context
                    )
                    self.active_campaigns.append(campaign)

                    # Set campaign for all participating orgs
                    for participant_id in campaign["participating_orgs"]:
                        self.org_state[participant_id]["current_campaign"] = campaign[
                            "id"
                        ]

            # Process active campaigns
            for campaign in list(self.active_campaigns):
                if current_date > campaign["end_date"]:
                    # End campaign
                    for participant_id in campaign["participating_orgs"]:
                        state = self.org_state[participant_id]
                        state["current_campaign"] = None
                        state["cooldown_until"] = current_date + datetime.timedelta(
                            days=random.randint(14, 45)
                        )
                    self.active_campaigns.remove(campaign)
                    continue

                # Generate distributions for campaign
                if current_date >= campaign["start_date"]:
                    for participant_id in campaign["participating_orgs"]:
                        if len(distributions) >= num_distributions:
                            break

                        participant = next(
                            o for o in self.organizations if o["ID"] == participant_id
                        )

                        # Check if org is active today
                        if self._should_org_work_today(participant, current_date):
                            batch = self.generate_distribution_batch(
                                participant, current_date, campaign
                            )
                            distributions.extend(batch)
                            self.org_state[participant_id]["last_activity"] = (
                                current_date
                            )

            # Non-campaign distributions
            for org in self.organizations:
                if len(distributions) >= num_distributions:
                    break

                org_state = self.org_state[org["ID"]]
                if org_state.get("current_campaign"):
                    continue  # Already handled above

                # Random non-campaign activity
                if self._should_org_work_today(org, current_date):
                    base_prob = org.get("activity_level", 0.5) * 0.01

                    # Seasonal adjustment
                    if current_date.month in [6, 7, 8]:
                        base_prob *= 2.0

                    if random.random() < base_prob:
                        batch = self.generate_distribution_batch(
                            org, current_date, None
                        )
                        distributions.extend(
                            batch[
                                : min(
                                    len(batch), num_distributions - len(distributions)
                                )
                            ]
                        )
                        org_state["last_activity"] = current_date

            # Progress update
            progress = int(len(distributions) / num_distributions * 100)
            if progress > last_progress:
                elapsed = time.time() - start_time
                rate = len(distributions) / elapsed if elapsed > 0 else 0
                eta = (num_distributions - len(distributions)) / rate if rate > 0 else 0
                print(
                    f"\r  ✓ Progress: {progress}% ({len(distributions):,} records) - "
                    f"{rate:.0f} records/sec - ETA: {eta / 60:.1f} min",
                    end="",
                )
                last_progress = progress

            current_date += datetime.timedelta(days=1)

        # Trim to exact number
        distributions = distributions[:num_distributions]

        print(f"\n✅ Generated {len(distributions):,} distribution records")

        # Print analysis
        self.print_enhanced_analysis(distributions)

        return distributions

    def _should_org_work_today(self, org: Dict, date: datetime.date) -> bool:
        """Determine if organization works on given date"""
        weekday = date.weekday()

        # Check working hours
        if org.get("working_hours", True) and weekday >= 5:
            # Weekend - check volunteer availability
            volunteer_avail = org.get("volunteer_availability", {})
            return random.random() < volunteer_avail.get("weekend", 0.3)

        # Check holidays
        if self._is_swedish_holiday(date):
            volunteer_avail = org.get("volunteer_availability", {})
            return random.random() < volunteer_avail.get("holiday", 0.1)

        # Check vacation periods
        if self._is_vacation_period(date):
            volunteer_avail = org.get("volunteer_availability", {})
            return random.random() < volunteer_avail.get("summer", 0.5)

        return True

    def _is_swedish_holiday(self, date: datetime.date) -> bool:
        """Check if date is Swedish holiday"""
        holidays = [
            (1, 1),  # New Year
            (1, 6),  # Epiphany
            (5, 1),  # Labor Day
            (6, 6),  # National Day
            (12, 24),  # Christmas Eve
            (12, 25),  # Christmas Day
            (12, 26),  # Boxing Day
            (12, 31),  # New Year's Eve
        ]

        return (date.month, date.day) in holidays

    def _is_vacation_period(self, date: datetime.date) -> bool:
        """Check if date is in typical Swedish vacation period"""
        # Summer vacation
        if (
            date.month == 7
            or (date.month == 6 and date.day > 20)
            or (date.month == 8 and date.day < 10)
        ):
            return True

        # Winter sport vacation (week 7-9)
        if date.month == 2 and 10 <= date.day <= 28:
            return True

        return False

    def print_enhanced_analysis(self, distributions: List[Dict]):
        """Print comprehensive analysis of generated distributions"""
        print("\n📊 Enhanced Distribution Analysis:")

        # Temporal patterns
        dates = [
            datetime.datetime.strptime(d["Delivered_Date"], "%Y-%m-%d").date()
            for d in distributions
        ]
        years = Counter(d.year for d in dates)
        months = Counter(d.month for d in dates)
        weekdays = Counter(d.weekday() for d in dates)

        print(f"\n  📅 Temporal Patterns:")
        print(f"    • Years: {dict(sorted(years.items()))}")
        print(
            f"    • Seasonal distribution: "
            + f"Spring: {sum(1 for d in dates if d.month in [3, 4, 5]) / len(dates) * 100:.1f}%, "
            + f"Summer: {sum(1 for d in dates if d.month in [6, 7, 8]) / len(dates) * 100:.1f}%, "
            + f"Autumn: {sum(1 for d in dates if d.month in [9, 10, 11]) / len(dates) * 100:.1f}%, "
            + f"Winter: {sum(1 for d in dates if d.month in [12, 1, 2]) / len(dates) * 100:.1f}%"
        )

        # Geographic patterns
        cities = Counter(d["Delivered_City"] for d in distributions)
        neighborhoods = Counter(
            f"{d['Delivered_City']}_{d['Delivered_Neighbourhood']}"
            for d in distributions
        )

        print(f"\n  🗺️  Geographic Distribution:")
        print(f"    • Cities covered: {len(cities)}")
        print(f"    • Top 5 cities: {cities.most_common(5)}")
        print(f"    • Unique neighborhoods: {len(neighborhoods)}")
        print(f"    • Most targeted neighborhoods: {neighborhoods.most_common(3)}")

        # Demographic analysis
        ages = [d["Addressee_Age"] for d in distributions]
        incomes = [d["Annual_Income_SEK"] for d in distributions]
        household_sizes = Counter(d["Household_Size"] for d in distributions)

        print(f"\n  👥 Demographic Patterns:")
        print(
            f"    • Age distribution: Mean: {np.mean(ages):.1f}, Std: {np.std(ages):.1f}"
        )
        print(
            f"    • Age groups: "
            + f"18-30: {sum(1 for a in ages if 18 <= a < 30) / len(ages) * 100:.1f}%, "
            + f"30-50: {sum(1 for a in ages if 30 <= a < 50) / len(ages) * 100:.1f}%, "
            + f"50-70: {sum(1 for a in ages if 50 <= a < 70) / len(ages) * 100:.1f}%, "
            + f"70+: {sum(1 for a in ages if a >= 70) / len(ages) * 100:.1f}%"
        )
        print(
            f"    • Income distribution: Median: {np.median(incomes):,.0f} SEK, "
            + f"25th percentile: {np.percentile(incomes, 25):,.0f}, "
            + f"75th percentile: {np.percentile(incomes, 75):,.0f}"
        )
        print(f"    • Household sizes: {dict(household_sizes.most_common())}")

        # Property ownership
        owners = sum(1 for d in distributions if d["Property_Value_SEK"] > 0)
        owner_pct = owners / len(distributions) * 100

        print(f"\n  🏠 Property Patterns:")
        print(f"    • Ownership rate: {owner_pct:.1f}%")
        print(
            f"    • Median property value (owners): {np.median([d['Property_Value_SEK'] for d in distributions if d['Property_Value_SEK'] > 0]):,.0f} SEK"
        )

        # Organization activity
        org_activity = Counter(d["Org_ID"] for d in distributions)
        org_types = {}
        for org in self.organizations:
            org_type = org.get("Type", "Unknown")
            if org_type not in org_types:
                org_types[org_type] = 0
            org_types[org_type] += org_activity.get(org["ID"], 0)

        print(f"\n  🏢 Organization Activity:")
        print(f"    • Active organizations: {len(org_activity)}")
        print(
            f"    • Activity by type: {dict(sorted(org_types.items(), key=lambda x: x[1], reverse=True))}"
        )

        # Coordinate coverage
        coords = [
            (d["Latitude"], d["Longitude"])
            for d in distributions
            if d["Latitude"] is not None and d["Longitude"] is not None
        ]
        unique_coords = len(set(coords))
        coord_rate = len(coords) / len(distributions) * 100

        print(f"\n  📍 Geographic Precision:")
        print(f"    • Geocoding success rate: {coord_rate:.1f}%")
        print(f"    • Unique coordinates: {unique_coords:,}")
        print(
            f"    • Average addresses per coordinate: {len(coords) / unique_coords:.1f}"
        )

    def save_distributions(
        self, distributions: List[Dict], append_to_existing: bool = True
    ):
        """Save distributions to CSV file"""
        fieldnames = [
            "Distribution_ID",
            "Leaflet_ID",
            "Org_ID",
            "Delivered_Date",
            "Delivered_City",
            "Delivered_Neighbourhood",
            "Addressee_Name",
            "Addressee_Gender",
            "Addressee_Age",
            "Annual_Income_SEK",
            "Property_Value_SEK",
            "Household_Cars",
            "Household_Size",
            "Latitude",
            "Longitude",
        ]

        dist_file = self.output_dir / "dists.csv"

        if append_to_existing and dist_file.exists():
            print(
                f"\n📝 Appending {len(distributions):,} records to existing dists.csv..."
            )
            with open(dist_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(distributions)
        else:
            print(f"\n📝 Creating new dists.csv with {len(distributions):,} records...")
            with open(dist_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(distributions)

        print(f"✅ Distribution data saved to {dist_file}")

    def generate_and_save(
        self, num_distributions: int = 10000, append_to_existing: bool = True
    ):
        """Complete workflow to generate and save ultra-realistic distributions"""

        print("🇸🇪 Ultra-Realistic Swedish Distribution Generator v4.0")
        print("=" * 70)
        print("   Enhanced with:")
        print("   • Neighborhood demographic profiling")
        print("   • Event-driven campaign generation")
        print("   • Realistic route planning")
        print("   • Complex socioeconomic correlations")
        print("   • Natural clustering patterns")

        # Optional Gemini integration message
        if self.gemini.client:
            print("   • AI-powered campaign narratives")

        # Load existing data
        self.load_existing_data()

        # Generate distributions
        start_time = time.time()
        distributions = self.generate_distributions(num_distributions)
        generation_time = time.time() - start_time

        # Save to CSV
        self.save_distributions(distributions, append_to_existing)

        print(f"\n⏱️  Total generation time: {generation_time / 60:.1f} minutes")
        print(
            f"🚀 Generation rate: {len(distributions) / generation_time:.0f} records/second"
        )

        return distributions

    def find_next_distribution_id(self):
        """Find the next available distribution ID"""
        dist_file = self.output_dir / "dists.csv"
        if dist_file.exists():
            with open(dist_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_ids = []
                for row in reader:
                    try:
                        existing_ids.append(int(row["Distribution_ID"]))
                    except (ValueError, KeyError):
                        continue
                self.next_distribution_id = max(existing_ids) + 1 if existing_ids else 1
            print(f"✅ Next distribution ID: {self.next_distribution_id}")

    def _generate_neighborhood_name(self) -> str:
        """Generate a realistic Swedish neighborhood name"""
        patterns = [
            ("Gamla", ["stan", "torget", ""]),
            ("Nya", ["hem", "staden", "torget"]),
            ("Östra", ["", "bergen", "hamnen"]),
            ("Västra", ["", "stranden", "kusten"]),
            ("Norra", ["", "delen", "gatan"]),
            ("Södra", ["", "ängen", "udden"]),
            ("Kungs", ["bergen", "holmen", "gatan"]),
            ("Drott", ["ninggatan", "torget", "holm"]),
            ("Skogs", ["backen", "lunden", "hem"]),
            ("Berg", ["", "hem", "lunden"]),
            ("Strand", ["", "vägen", "parken"]),
            ("Park", ["staden", "vägen", ""]),
            ("Centrum", ["", "staden", ""]),
            ("Hamn", ["staden", "området", ""]),
            ("Villa", ["staden", "området", "parken"]),
        ]

        prefix, suffixes = random.choice(patterns)
        suffix = random.choice(suffixes)
        return f"{prefix}{suffix}".strip()

    def initialize_org_states(self):
        """Initialize organization states"""
        for org in self.organizations:
            self.org_state[org["ID"]] = {
                "last_activity": None,
                "current_campaign": None,
                "distribution_count": 0,
                "cooldown_until": None,
                "campaign_history": [],
                "seasonal_activity": {},
            }


def main():
    """Main execution function"""

    # Initialize generator
    generator = UltraRealisticDistributionGenerator(".")

    # Generate ultra-realistic distributions
    distributions = generator.generate_and_save(
        num_distributions=500000,  # Adjust as needed
        append_to_existing=True,
    )

    print("\n🎉 Ultra-realistic generation complete!")
    print("📈 The enhanced data now includes:")
    print("  • Neighborhood-level demographic clustering")
    print("  • Event-driven campaigns responding to real patterns")
    print("  • Realistic income-age-property correlations")
    print("  • Natural distribution routes and delivery patterns")
    print("  • Complex organizational behaviors and collaborations")
    print("  • Seasonal and temporal variations matching Swedish patterns")
    print("  • Geographic precision with realistic address clustering")

    if GEMINI_AVAILABLE:
        print("  • AI-generated campaign narratives for enhanced realism")

    return distributions


if __name__ == "__main__":
    distributions = main()
