"""
Step 1: Download and Prepare SPDX License Data
- Downloads official SPDX license list
- Extracts full license texts
- Creates canonical label schema
"""

import json
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SPDX_JSON_URL = "https://raw.githubusercontent.com/spdx/license-list-data/main/json/licenses.json"
OUTPUT_DIR = Path("data/spdx")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_spdx_list():
    """Download official SPDX license list"""
    logger.info("Downloading SPDX License List...")
    
    try:
        response = requests.get(SPDX_JSON_URL, timeout=30)
        response.raise_for_status()
        
        licenses_data = response.json()
        
        # Save raw data
        with open(OUTPUT_DIR / "spdx_licenses.json", "w") as f:
            json.dump(licenses_data, f, indent=2)
        
        logger.info(f"✓ Downloaded {len(licenses_data.get('licenses', []))} SPDX licenses")
        return licenses_data
        
    except Exception as e:
        logger.error(f"Failed to download SPDX list: {e}")
        logger.info("Using local copy if available...")
        
        local_file = OUTPUT_DIR / "spdx_licenses.json"
        if local_file.exists():
            with open(local_file) as f:
                return json.load(f)
        else:
            return None

def extract_license_texts():
    """Extract texts from XML license files (already in data/raw/license-list-XML)"""
    logger.info("Extracting license texts from existing XML files...")
    
    import xml.etree.ElementTree as ET
    
    license_dir = Path("../data/raw/license-list-XML")
    if not license_dir.exists():
        logger.warning(f"License XML directory not found: {license_dir}")
        return {}
    
    texts = {}
    xml_files = list(license_dir.glob("*.xml"))
    
    for xml_file in xml_files:
        license_id = xml_file.stem
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract all text
            text_parts = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text_parts.append(elem.text.strip())
            
            texts[license_id] = " ".join(text_parts)
            
        except Exception as e:
            logger.warning(f"Error parsing {license_id}: {e}")
    
    logger.info(f"✓ Extracted texts for {len(texts)} licenses")
    
    # Save texts
    with open(OUTPUT_DIR / "license_texts.json", "w") as f:
        json.dump(texts, f, indent=2)
    
    return texts

def build_label_schema(spdx_data, license_texts):
    """Build canonical label schema"""
    logger.info("Building label schema...")
    
    schema = {
        "licenses": {},
        "metadata": {
            "total_licenses": 0,
            "with_texts": 0,
            "categories": {}
        }
    }
    
    licenses = spdx_data.get("licenses", [])
    
    for lic in licenses:
        lic_id = lic.get("licenseId", "")
        
        schema["licenses"][lic_id] = {
            "name": lic.get("name", ""),
            "isOsiApproved": lic.get("isOsiApproved", False),
            "isDeprecated": lic.get("isDeprecatedLicenseId", False),
            "hasText": lic_id in license_texts,
            "categories": [],
            "aliases": [lic_id],  # Will be updated with LicenseLynx data
        }
        
        schema["metadata"]["total_licenses"] += 1
        if lic_id in license_texts:
            schema["metadata"]["with_texts"] += 1
    
    # Categorize licenses
    for lic_id, info in schema["licenses"].items():
        if "GPL" in lic_id or "AGPL" in lic_id:
            info["categories"].append("copyleft")
        elif info["isOsiApproved"]:
            info["categories"].append("permissive")
        else:
            info["categories"].append("other")
        
        category = info["categories"][0]
        if category not in schema["metadata"]["categories"]:
            schema["metadata"]["categories"][category] = 0
        schema["metadata"]["categories"][category] += 1
    
    # Save schema
    with open(OUTPUT_DIR / "label_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    
    logger.info(f"✓ Built label schema with {schema['metadata']['total_licenses']} licenses")
    logger.info(f"  - With texts: {schema['metadata']['with_texts']}")
    logger.info(f"  - Categories: {schema['metadata']['categories']}")
    
    return schema

def main():
    logger.info("=" * 80)
    logger.info("STEP 1: PREPARE SPDX LICENSE DATA")
    logger.info("=" * 80)
    
    # Download SPDX list
    spdx_data = download_spdx_list()
    if not spdx_data:
        logger.error("Could not load SPDX data")
        return
    
    # Extract license texts
    license_texts = extract_license_texts()
    
    # Build label schema
    schema = build_label_schema(spdx_data, license_texts)
    
    logger.info("=" * 80)
    logger.info("STEP 1 COMPLETE")
    logger.info("=" * 80)
    logger.info("Output files:")
    logger.info(f"  - {OUTPUT_DIR}/spdx_licenses.json")
    logger.info(f"  - {OUTPUT_DIR}/license_texts.json")
    logger.info(f"  - {OUTPUT_DIR}/label_schema.json")
    logger.info("Next: Run step 2 to download ScanCode LicenseDB")

if __name__ == "__main__":
    main()
