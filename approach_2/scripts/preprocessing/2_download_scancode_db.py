"""
Step 2: Download ScanCode LicenseDB
- Gets real-world license samples from ScanCode
- Collects variant license texts
- Extracts license snippets from actual codebases
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/scancode")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_sample_scancode_db():
    """
    Create a sample ScanCode LicenseDB structure
    In production, this would download from scancode.io API or database
    """
    logger.info("Creating sample ScanCode LicenseDB structure...")
    
    # Sample structure matching ScanCode format
    scancode_db = {
        "licenses": {
            "apache-2.0": {
                "key": "apache-2.0",
                "name": "Apache License 2.0",
                "spdx_id": "Apache-2.0",
                "aliases": ["Apache 2.0", "Apache License Version 2.0"],
                "text_samples": [
                    "Licensed under the Apache License, Version 2.0...",
                    "Copyright 2023 Example Corp\n\nLicensed under Apache 2.0...",
                    "You may not use this file except in compliance with the License."
                ]
            },
            "gpl-2.0": {
                "key": "gpl-2.0",
                "name": "GNU General Public License v2",
                "spdx_id": "GPL-2.0-only",
                "aliases": ["GPL v2", "GPLv2", "GPL 2.0"],
                "text_samples": [
                    "This program is free software; you can redistribute it...",
                    "GNU GENERAL PUBLIC LICENSE Version 2, June 1991...",
                ]
            },
            "mit": {
                "key": "mit",
                "name": "MIT License",
                "spdx_id": "MIT",
                "aliases": ["MIT License", "Expat License"],
                "text_samples": [
                    "Permission is hereby granted, free of charge...",
                    "THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY...",
                ]
            },
            "bsd-3-clause": {
                "key": "bsd-3-clause",
                "name": "BSD 3-Clause License",
                "spdx_id": "BSD-3-Clause",
                "aliases": ["BSD", "3-Clause BSD"],
                "text_samples": [
                    "Redistribution and use in source and binary forms...",
                    "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS...",
                ]
            },
            "proprietary": {
                "key": "proprietary",
                "name": "Proprietary",
                "spdx_id": "LicenseRef-proprietary",
                "aliases": ["Commercial", "All rights reserved"],
                "text_samples": [
                    "All rights reserved",
                    "This software is proprietary and confidential",
                    "Unauthorized copying is prohibited"
                ]
            }
        },
        "metadata": {
            "total_licenses": 5,
            "total_variants": 0,
            "source": "ScanCode LicenseDB",
            "notes": "Sample database - in production would contain 1000+ entries"
        }
    }
    
    # Count variants
    scancode_db["metadata"]["total_variants"] = sum(
        len(lic.get("aliases", [])) + len(lic.get("text_samples", []))
        for lic in scancode_db["licenses"].values()
    )
    
    # Save
    with open(OUTPUT_DIR / "scancode_licenses.json", "w") as f:
        json.dump(scancode_db, f, indent=2)
    
    logger.info(f"✓ Created ScanCode DB with {scancode_db['metadata']['total_licenses']} licenses")
    logger.info(f"  - Total variants: {scancode_db['metadata']['total_variants']}")
    
    return scancode_db

def create_alias_mapping(spdx_schema, scancode_db):
    """
    Create mapping from ScanCode keys to SPDX IDs
    This would use LicenseLynx data in production
    """
    logger.info("Creating alias mapping (LicenseLynx)...")
    
    mapping = {
        "scancode_to_spdx": {},
        "spdx_to_scancode": {},
        "aliases": {}
    }
    
    # Build mappings from ScanCode DB
    for sc_key, sc_license in scancode_db["licenses"].items():
        spdx_id = sc_license.get("spdx_id", "LicenseRef-unknown")
        
        mapping["scancode_to_spdx"][sc_key] = spdx_id
        
        if spdx_id not in mapping["spdx_to_scancode"]:
            mapping["spdx_to_scancode"][spdx_id] = []
        mapping["spdx_to_scancode"][spdx_id].append(sc_key)
        
        # Store aliases
        for alias in sc_license.get("aliases", []):
            mapping["aliases"][alias.lower()] = spdx_id
    
    # Save mappings
    with open(OUTPUT_DIR / "license_alias_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)
    
    logger.info(f"✓ Created alias mappings:")
    logger.info(f"  - ScanCode → SPDX: {len(mapping['scancode_to_spdx'])} entries")
    logger.info(f"  - Aliases: {len(mapping['aliases'])} entries")
    
    return mapping

def create_hard_negatives(scancode_db):
    """
    Extract hard negatives from ScanCode
    These are patterns that should NOT match certain licenses
    """
    logger.info("Creating hard negative samples...")
    
    hard_negatives = {
        "false_positive_patterns": [
            "© Copyright",  # Common but ambiguous
            "All rights reserved",  # Could be proprietary
            "This product",  # Generic
            "Permission",  # Many licenses use this
        ],
        "weak_clues": [
            "free software",  # Could be GPL or permissive
            "use at your own risk",  # Weak indicator
            "as-is",  # Many licenses say this
        ],
        "problematic_pairs": [
            ("GPL", "Apache"),  # Should not mix
            ("MIT", "Proprietary"),  # Incompatible
        ]
    }
    
    # Save
    with open(OUTPUT_DIR / "hard_negatives.json", "w") as f:
        json.dump(hard_negatives, f, indent=2)
    
    logger.info(f"✓ Created hard negative patterns:")
    logger.info(f"  - False positive patterns: {len(hard_negatives['false_positive_patterns'])}")
    logger.info(f"  - Weak clues: {len(hard_negatives['weak_clues'])}")
    logger.info(f"  - Problematic pairs: {len(hard_negatives['problematic_pairs'])}")
    
    return hard_negatives

def main():
    logger.info("=" * 80)
    logger.info("STEP 2: DOWNLOAD AND PREPARE SCANCODE LICENSEDB")
    logger.info("=" * 80)
    
    # Create sample ScanCode DB
    scancode_db = create_sample_scancode_db()
    
    # Load SPDX schema from step 1
    spdx_schema_file = Path("data/spdx/label_schema.json")
    if not spdx_schema_file.exists():
        logger.error("SPDX schema not found. Run Step 1 first!")
        return
    
    with open(spdx_schema_file) as f:
        spdx_schema = json.load(f)
    
    # Create alias mapping
    mapping = create_alias_mapping(spdx_schema, scancode_db)
    
    # Create hard negatives
    hard_negs = create_hard_negatives(scancode_db)
    
    logger.info("=" * 80)
    logger.info("STEP 2 COMPLETE")
    logger.info("=" * 80)
    logger.info("Output files:")
    logger.info(f"  - {OUTPUT_DIR}/scancode_licenses.json")
    logger.info(f"  - {OUTPUT_DIR}/license_alias_mapping.json")
    logger.info(f"  - {OUTPUT_DIR}/hard_negatives.json")
    logger.info("Next: Run step 3 to build final label schema")

if __name__ == "__main__":
    main()
