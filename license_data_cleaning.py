"""
License Data Cleaning Pipeline
Processes SPDX license XML files and creates a cleaned, labeled dataset
"""

import os
import pandas as pd
import re
import string
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define license categories
COPYLEFT_LICENSES = {
    'GPL', 'AGPL', 'LGPL', 'SSPL', 'EPL', 'MPL', 'CDDL', 'EUPL', 'GFDL', 'ADSL'
}

PERMISSIVE_LICENSES = {
    'Apache', 'MIT', 'BSD', 'ISC', 'Zlib', 'Boost', 'CC0', 'Unlicense', 
    'Python', 'PSF', 'WTF', 'X11', 'Artistic'
}

class LicenseDataCleaner:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.licenses = []
        
    def extract_text_from_xml(self, text_element):
        """Extract clean text from XML element"""
        if text_element is None:
            return ""
        
        def get_text(elem):
            result = []
            if elem.text:
                result.append(elem.text)
            for child in elem:
                result.append(get_text(child))
                if child.tail:
                    result.append(child.tail)
            return ''.join(result)
        
        return get_text(text_element).strip()
    
    def step1_structural_cleaning(self):
        """Step 1: Parse XML files and extract license information"""
        logger.info("Starting Step 1: Structural Cleaning (XML Parsing)")
        
        xml_files = list(self.input_dir.glob("*.xml"))
        logger.info(f"Found {len(xml_files)} XML files")
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Define namespace
                ns = {'spdx': 'http://www.spdx.org/license'}
                
                # Find license element
                license_elem = root.find('.//spdx:license', ns)
                if license_elem is None:
                    license_elem = root.find('.//license')
                
                if license_elem is not None:
                    license_id = license_elem.get('licenseId') or license_elem.get('id', 'UNKNOWN')
                    is_osi_approved = license_elem.get('isOsiApproved', 'false').lower() == 'true'
                    is_deprecated = license_elem.get('isDeprecatedLicenseId', 'false').lower() == 'true'
                    
                    # Extract license text
                    text_elem = license_elem.find('.//spdx:text', ns)
                    if text_elem is None:
                        text_elem = license_elem.find('.//text')
                    
                    license_text = self.extract_text_from_xml(text_elem) if text_elem is not None else ""
                    
                    self.licenses.append({
                        'license_id': license_id,
                        'raw_text': license_text,
                        'is_osi_approved': is_osi_approved,
                        'is_deprecated': is_deprecated
                    })
                    
            except Exception as e:
                logger.warning(f"Error parsing {xml_file.name}: {e}")
        
        logger.info(f"Extracted {len(self.licenses)} licenses")
        return self.licenses
    
    def step2_dataset_cleaning(self):
        """Step 2: Remove deprecated licenses, empty texts, and duplicates"""
        logger.info("Starting Step 2: Dataset-level Cleaning")
        
        initial_count = len(self.licenses)
        
        # Remove deprecated licenses
        self.licenses = [lic for lic in self.licenses if not lic['is_deprecated']]
        removed_deprecated = initial_count - len(self.licenses)
        logger.info(f"Removed {removed_deprecated} deprecated licenses")
        
        # Remove empty or very short texts (< 50 characters)
        initial_count = len(self.licenses)
        self.licenses = [lic for lic in self.licenses if len(lic['raw_text'].strip()) >= 50]
        removed_empty = initial_count - len(self.licenses)
        logger.info(f"Removed {removed_empty} licenses with empty/short texts")
        
        # Remove duplicates based on license_id
        initial_count = len(self.licenses)
        seen = set()
        unique_licenses = []
        for lic in self.licenses:
            if lic['license_id'] not in seen:
                unique_licenses.append(lic)
                seen.add(lic['license_id'])
        removed_dupes = initial_count - len(unique_licenses)
        self.licenses = unique_licenses
        logger.info(f"Removed {removed_dupes} duplicate licenses")
        
        logger.info(f"Remaining licenses after dataset cleaning: {len(self.licenses)}")
        return self.licenses
    
    def step3_text_cleaning(self):
        """Step 3: Text preprocessing (lowercase, remove punctuation, etc.)"""
        logger.info("Starting Step 3: Text Cleaning (NLP Preprocessing)")
        
        # English stopwords (common words to remove)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'is', 'are', 'am', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'as', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'
        }
        
        for lic in self.licenses:
            text = lic['raw_text']
            
            # Lowercase
            text = text.lower()
            
            # Remove HTML/XML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove numbers
            text = re.sub(r'\d+', '', text)
            
            # Remove punctuation (keep some important ones like hyphen)
            text = ''.join(char if char not in string.punctuation or char == '-' else ' ' for char in text)
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove stopwords
            words = text.split()
            words = [w for w in words if w not in stopwords and len(w) > 2]
            text = ' '.join(words)
            
            lic['cleaned_text'] = text
        
        logger.info("Text cleaning completed")
        return self.licenses
    
    def step4_label_creation(self):
        """Step 4: Create labels for ML classification"""
        logger.info("Starting Step 4: Label Creation")
        
        for lic in self.licenses:
            license_id = lic['license_id'].upper()
            
            # Classify as permissive or copyleft
            is_permissive = any(keyword in license_id for keyword in PERMISSIVE_LICENSES)
            is_copyleft = any(keyword in license_id for keyword in COPYLEFT_LICENSES)
            
            if is_copyleft and not is_permissive:
                license_type = 'copyleft'
            elif is_permissive:
                license_type = 'permissive'
            else:
                # Check if it contains both characteristics
                license_type = 'hybrid' if is_copyleft else 'other'
            
            # OSI approved label
            osi_label = 'osi-approved' if lic['is_osi_approved'] else 'not-osi-approved'
            
            # Active/Deprecated label (deprecated already filtered, so all are active)
            activity_label = 'active'
            
            lic['license_type'] = license_type
            lic['osi_status'] = osi_label
            lic['activity_status'] = activity_label
        
        logger.info("Label creation completed")
        return self.licenses
    
    def save_processed_data(self):
        """Save cleaned and labeled data to CSV and JSON"""
        logger.info("Saving processed data")
        
        # Create DataFrame
        df = pd.DataFrame(self.licenses)
        
        # Save to CSV
        csv_path = self.output_dir / 'licenses_cleaned.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV to {csv_path}")
        
        # Save to JSON
        json_path = self.output_dir / 'licenses_cleaned.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.licenses, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON to {json_path}")
        
        # Save summary statistics
        summary = {
            'total_licenses': len(self.licenses),
            'osi_approved_count': sum(1 for lic in self.licenses if lic['is_osi_approved']),
            'license_type_distribution': df['license_type'].value_counts().to_dict(),
            'osi_status_distribution': df['osi_status'].value_counts().to_dict(),
            'avg_text_length': df['raw_text'].str.len().mean(),
            'avg_cleaned_text_length': df['cleaned_text'].str.len().mean()
        }
        
        summary_path = self.output_dir / 'cleaning_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")
        
        # Display summary
        logger.info("\n=== DATA CLEANING SUMMARY ===")
        logger.info(f"Total licenses processed: {summary['total_licenses']}")
        logger.info(f"OSI approved: {summary['osi_approved_count']}")
        logger.info(f"Average raw text length: {summary['avg_text_length']:.0f} chars")
        logger.info(f"Average cleaned text length: {summary['avg_cleaned_text_length']:.0f} chars")
        logger.info(f"\nLicense Type Distribution:")
        for ltype, count in summary['license_type_distribution'].items():
            logger.info(f"  {ltype}: {count}")
        
        return df, summary
    
    def run_pipeline(self):
        """Run complete cleaning pipeline"""
        logger.info("=" * 60)
        logger.info("STARTING LICENSE DATA CLEANING PIPELINE")
        logger.info("=" * 60)
        
        self.step1_structural_cleaning()
        self.step2_dataset_cleaning()
        self.step3_text_cleaning()
        self.step4_label_creation()
        df, summary = self.save_processed_data()
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return df, summary


if __name__ == "__main__":
    # Define paths
    input_dir = r"c:\Users\ASUS\Desktop\ML project 2\data\raw\license-list-XML"
    output_dir = r"c:\Users\ASUS\Desktop\ML project 2\data\processed"
    
    # Run pipeline
    cleaner = LicenseDataCleaner(input_dir, output_dir)
    df, summary = cleaner.run_pipeline()
    
    # Display sample of cleaned data
    logger.info("\n=== SAMPLE OF CLEANED DATA ===")
    logger.info(df[['license_id', 'license_type', 'osi_status', 'cleaned_text']].head(10).to_string())
