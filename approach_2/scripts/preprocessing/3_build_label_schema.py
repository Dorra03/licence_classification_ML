"""
Step 3: Build Final Label Schema
- Combines SPDX and ScanCode data
- Creates canonical label mapping
- Handles license aliases and variants
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_final_label_schema():
    """Build final label schema combining SPDX + ScanCode"""
    logger.info("Building final label schema...")
    
    # Load SPDX data
    with open("data/spdx/label_schema.json") as f:
        spdx_schema = json.load(f)
    
    # Load ScanCode data
    with open("data/scancode/scancode_licenses.json") as f:
        scancode_db = json.load(f)
    
    # Load alias mapping
    with open("data/scancode/license_alias_mapping.json") as f:
        alias_mapping = json.load(f)
    
    # Build unified schema
    unified_schema = {
        "classes": {},
        "aliases": {},
        "variants": {},
        "metadata": {
            "total_classes": 0,
            "total_aliases": 0,
            "total_variants": 0,
            "sources": ["SPDX", "ScanCode"]
        }
    }
    
    # Add all SPDX licenses as classes
    for spdx_id, spdx_lic in spdx_schema["licenses"].items():
        unified_schema["classes"][spdx_id] = {
            "name": spdx_lic.get("name", ""),
            "source": "SPDX",
            "osi_approved": spdx_lic.get("isOsiApproved", False),
            "deprecated": spdx_lic.get("isDeprecated", False),
            "categories": spdx_lic.get("categories", []),
            "aliases": [],
            "variants": 0,
            "scancode_mappings": []
        }
    
    # Add ScanCode mappings and variants
    for sc_key, sc_lic in scancode_db["licenses"].items():
        spdx_id = sc_lic.get("spdx_id", "LicenseRef-unknown")
        
        # Ensure SPDX class exists
        if spdx_id not in unified_schema["classes"]:
            unified_schema["classes"][spdx_id] = {
                "name": sc_lic.get("name", ""),
                "source": "ScanCode",
                "osi_approved": False,
                "deprecated": False,
                "categories": ["other"],
                "aliases": [],
                "variants": 0,
                "scancode_mappings": []
            }
        
        # Add ScanCode mapping
        unified_schema["classes"][spdx_id]["scancode_mappings"].append(sc_key)
        
        # Add aliases
        for alias in sc_lic.get("aliases", []):
            unified_schema["aliases"][alias.lower()] = spdx_id
            unified_schema["classes"][spdx_id]["aliases"].append(alias)
        
        # Add text variants
        variants = sc_lic.get("text_samples", [])
        unified_schema["variants"][f"{spdx_id}_scancode"] = {
            "spdx_id": spdx_id,
            "source": "ScanCode",
            "scancode_key": sc_key,
            "samples": variants,
            "count": len(variants)
        }
        
        unified_schema["classes"][spdx_id]["variants"] += len(variants)
    
    # Count metadata
    unified_schema["metadata"]["total_classes"] = len(unified_schema["classes"])
    unified_schema["metadata"]["total_aliases"] = len(unified_schema["aliases"])
    unified_schema["metadata"]["total_variants"] = sum(
        unified_schema["classes"][cid]["variants"]
        for cid in unified_schema["classes"]
    )
    
    # Save
    with open(OUTPUT_DIR / "unified_label_schema.json", "w") as f:
        json.dump(unified_schema, f, indent=2)
    
    logger.info(f"✓ Built unified schema:")
    logger.info(f"  - Total classes: {unified_schema['metadata']['total_classes']}")
    logger.info(f"  - Total aliases: {unified_schema['metadata']['total_aliases']}")
    logger.info(f"  - Total variants: {unified_schema['metadata']['total_variants']}")
    
    return unified_schema

def create_class_mapping():
    """Create numeric ID mapping for classes"""
    logger.info("Creating class ID mapping...")
    
    with open(OUTPUT_DIR / "unified_label_schema.json") as f:
        schema = json.load(f)
    
    # Create mapping
    class_mapping = {
        "spdx_id_to_class_id": {},
        "class_id_to_spdx_id": {},
        "total_classes": len(schema["classes"])
    }
    
    for idx, spdx_id in enumerate(sorted(schema["classes"].keys())):
        class_mapping["spdx_id_to_class_id"][spdx_id] = idx
        class_mapping["class_id_to_spdx_id"][str(idx)] = spdx_id
    
    # Save
    with open(OUTPUT_DIR / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    
    logger.info(f"✓ Created class mapping with {class_mapping['total_classes']} classes")
    
    return class_mapping

def generate_statistics():
    """Generate dataset statistics"""
    logger.info("Generating statistics...")
    
    with open(OUTPUT_DIR / "unified_label_schema.json") as f:
        schema = json.load(f)
    
    stats = {
        "summary": {
            "total_spdx_licenses": schema["metadata"]["total_classes"],
            "total_aliases": schema["metadata"]["total_aliases"],
            "total_variant_samples": schema["metadata"]["total_variants"],
            "average_variants_per_license": schema["metadata"]["total_variants"] / schema["metadata"]["total_classes"] if schema["metadata"]["total_classes"] > 0 else 0,
        },
        "by_category": {},
        "by_source": {
            "spdx_only": 0,
            "scancode_only": 0,
            "both": 0
        }
    }
    
    # Count by category
    for spdx_id, lic in schema["classes"].items():
        for category in lic.get("categories", []):
            if category not in stats["by_category"]:
                stats["by_category"][category] = {
                    "count": 0,
                    "variants": 0
                }
            stats["by_category"][category]["count"] += 1
            stats["by_category"][category]["variants"] += lic["variants"]
        
        # Count by source
        if lic["source"] == "SPDX" and len(lic["scancode_mappings"]) > 0:
            stats["by_source"]["both"] += 1
        elif lic["source"] == "SPDX":
            stats["by_source"]["spdx_only"] += 1
        else:
            stats["by_source"]["scancode_only"] += 1
    
    # Save
    with open(OUTPUT_DIR / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"✓ Generated statistics:")
    logger.info(f"  Summary: {stats['summary']}")
    logger.info(f"  By source: {stats['by_source']}")
    logger.info(f"  By category: {stats['by_category']}")
    
    return stats

def main():
    logger.info("=" * 80)
    logger.info("STEP 3: BUILD FINAL LABEL SCHEMA")
    logger.info("=" * 80)
    
    # Build unified schema
    unified_schema = build_final_label_schema()
    
    # Create class mapping
    class_mapping = create_class_mapping()
    
    # Generate statistics
    stats = generate_statistics()
    
    logger.info("=" * 80)
    logger.info("STEP 3 COMPLETE")
    logger.info("=" * 80)
    logger.info("Output files:")
    logger.info(f"  - {OUTPUT_DIR}/unified_label_schema.json")
    logger.info(f"  - {OUTPUT_DIR}/class_mapping.json")
    logger.info(f"  - {OUTPUT_DIR}/dataset_statistics.json")
    logger.info("Next: Run step 4 to normalize samples")

if __name__ == "__main__":
    main()
