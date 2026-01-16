"""
Exploratory Data Analysis (EDA) - Before & After Data Cleaning
Analyzes raw vs. processed license data to understand cleaning impact
"""

import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

class LicenseEDA:
    def __init__(self, raw_dir, processed_dir, output_dir):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_licenses = []
        self.processed_df = None
        
    def load_raw_data(self):
        """Load all raw XML license files"""
        logger.info("Loading raw license data from XML files...")
        
        xml_files = list(self.raw_dir.glob("*.xml"))
        logger.info(f"Found {len(xml_files)} XML files")
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                ns = {'spdx': 'http://www.spdx.org/license'}
                license_elem = root.find('.//spdx:license', ns)
                if license_elem is None:
                    license_elem = root.find('.//license')
                
                if license_elem is not None:
                    license_id = license_elem.get('licenseId') or license_elem.get('id', 'UNKNOWN')
                    is_osi_approved = license_elem.get('isOsiApproved', 'false').lower() == 'true'
                    is_deprecated = license_elem.get('isDeprecatedLicenseId', 'false').lower() == 'true'
                    
                    text_elem = license_elem.find('.//spdx:text', ns)
                    if text_elem is None:
                        text_elem = license_elem.find('.//text')
                    
                    raw_text = self._extract_text_from_xml(text_elem) if text_elem is not None else ""
                    
                    self.raw_licenses.append({
                        'license_id': license_id,
                        'raw_text': raw_text,
                        'text_length': len(raw_text),
                        'is_osi_approved': is_osi_approved,
                        'is_deprecated': is_deprecated
                    })
            except Exception as e:
                logger.warning(f"Error parsing {xml_file.name}: {e}")
        
        logger.info(f"Loaded {len(self.raw_licenses)} licenses")
        return self.raw_licenses
    
    def _extract_text_from_xml(self, text_element):
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
    
    def load_processed_data(self):
        """Load processed CSV data"""
        logger.info("Loading processed license data...")
        csv_path = self.processed_dir / 'licenses_cleaned.csv'
        self.processed_df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.processed_df)} processed licenses")
        return self.processed_df
    
    def analyze_raw_data(self):
        """Analyze raw dataset"""
        logger.info("=" * 60)
        logger.info("ANALYZING RAW DATASET")
        logger.info("=" * 60)
        
        raw_df = pd.DataFrame(self.raw_licenses)
        
        # Basic statistics
        print("\n📊 DATASET SIZE & STRUCTURE")
        print(f"  Total licenses: {len(raw_df)}")
        print(f"  Deprecated licenses: {raw_df['is_deprecated'].sum()}")
        print(f"  Active licenses: {(~raw_df['is_deprecated']).sum()}")
        print(f"  OSI approved: {raw_df['is_osi_approved'].sum()}")
        print(f"  Non-OSI approved: {(~raw_df['is_osi_approved']).sum()}")
        
        # Empty texts
        empty_texts = (raw_df['text_length'] == 0).sum()
        short_texts = (raw_df['text_length'] < 50).sum()
        print(f"\n⚠️ DATA QUALITY ISSUES")
        print(f"  Empty license texts: {empty_texts}")
        print(f"  Very short texts (<50 chars): {short_texts}")
        
        # Text length statistics
        print(f"\n📏 TEXT LENGTH DISTRIBUTION (chars)")
        print(f"  Mean: {raw_df['text_length'].mean():.0f}")
        print(f"  Median: {raw_df['text_length'].median():.0f}")
        print(f"  Min: {raw_df['text_length'].min()}")
        print(f"  Max: {raw_df['text_length'].max()}")
        print(f"  Std Dev: {raw_df['text_length'].std():.0f}")
        print(f"\n  Quartiles:")
        for q in [0.25, 0.5, 0.75]:
            print(f"    Q{int(q*100)}: {raw_df['text_length'].quantile(q):.0f}")
        
        # Duplicates check
        print(f"\n🔍 DUPLICATE DETECTION")
        duplicate_texts = raw_df['raw_text'].duplicated().sum()
        print(f"  Duplicate texts: {duplicate_texts}")
        
        return raw_df
    
    def analyze_processed_data(self):
        """Analyze processed dataset"""
        logger.info("=" * 60)
        logger.info("ANALYZING PROCESSED DATASET")
        logger.info("=" * 60)
        
        df = self.processed_df
        
        print("\n📊 DATASET SUMMARY")
        print(f"  Total licenses: {len(df)}")
        print(f"  Removed during cleaning: {len(self.raw_licenses) - len(df)} ({((len(self.raw_licenses) - len(df)) / len(self.raw_licenses) * 100):.1f}%)")
        
        # Text statistics
        print(f"\n📝 CLEANED TEXT STATISTICS")
        print(f"  Average raw text length: {df['raw_text'].str.len().mean():.0f} chars")
        print(f"  Average cleaned text length: {df['cleaned_text'].str.len().mean():.0f} chars")
        print(f"  Compression ratio: {(1 - df['cleaned_text'].str.len().mean() / df['raw_text'].str.len().mean()) * 100:.1f}%")
        
        # Vocabulary
        all_words = ' '.join(df['cleaned_text']).split()
        vocab_size = len(set(all_words))
        print(f"  Total tokens: {len(all_words):,}")
        print(f"  Unique tokens (vocabulary): {vocab_size:,}")
        print(f"  Average tokens per license: {len(all_words) / len(df):.0f}")
        
        # Label distribution
        print(f"\n🏷️ LABEL DISTRIBUTION")
        print(f"\n  License Type:")
        for license_type, count in df['license_type'].value_counts().items():
            pct = count / len(df) * 100
            print(f"    {license_type}: {count} ({pct:.1f}%)")
        
        print(f"\n  OSI Status:")
        for status, count in df['osi_status'].value_counts().items():
            pct = count / len(df) * 100
            print(f"    {status}: {count} ({pct:.1f}%)")
        
        # Class imbalance
        print(f"\n⚖️ CLASS BALANCE ANALYSIS")
        license_counts = df['license_id'].value_counts()
        print(f"  Unique license IDs: {len(license_counts)}")
        print(f"  License ID distribution:")
        print(f"    Most common: {license_counts.index[0]} ({license_counts.iloc[0]})")
        print(f"    Least common: {license_counts.index[-1]} ({license_counts.iloc[-1]})")
        
        # Imbalance ratio
        imbalance_ratio = license_counts.max() / license_counts.min()
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
        
        # Top 10 licenses
        print(f"\n  Top 10 Most Frequent Licenses:")
        for idx, (license_id, count) in enumerate(license_counts.head(10).items(), 1):
            print(f"    {idx}. {license_id}: {count}")
        
        return df
    
    def create_visualizations(self, raw_df):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        # 1. Text Length Distribution (Raw vs Processed)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Data Exploration: Before & After Cleaning', fontsize=16, fontweight='bold')
        
        # Raw text length distribution
        axes[0, 0].hist(raw_df['text_length'], bins=50, color='coral', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Text Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Raw License Text Length Distribution')
        axes[0, 0].axvline(raw_df['text_length'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {raw_df["text_length"].mean():.0f}')
        axes[0, 0].legend()
        
        # Cleaned text length distribution
        cleaned_lengths = self.processed_df['cleaned_text'].str.len()
        axes[0, 1].hist(cleaned_lengths, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Text Length (characters)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Cleaned License Text Length Distribution')
        axes[0, 1].axvline(cleaned_lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cleaned_lengths.mean():.0f}')
        axes[0, 1].legend()
        
        # Metadata overview - Raw
        metadata_labels = ['OSI Approved', 'Not OSI Approved', 'Active', 'Deprecated']
        metadata_counts = [
            raw_df['is_osi_approved'].sum(),
            (~raw_df['is_osi_approved']).sum(),
            (~raw_df['is_deprecated']).sum(),
            raw_df['is_deprecated'].sum()
        ]
        axes[1, 0].barh(metadata_labels, metadata_counts, color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'])
        axes[1, 0].set_xlabel('Count')
        axes[1, 0].set_title('Raw Data: Metadata Overview')
        for i, v in enumerate(metadata_counts):
            axes[1, 0].text(v + 5, i, str(v), va='center')
        
        # License Type Distribution - Processed
        license_type_counts = self.processed_df['license_type'].value_counts()
        colors_license = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']
        axes[1, 1].pie(license_type_counts, labels=license_type_counts.index, autopct='%1.1f%%',
                       colors=colors_license, startangle=90)
        axes[1, 1].set_title('Processed Data: License Type Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_overview.png', dpi=300, bbox_inches='tight')
        logger.info("Saved: eda_overview.png")
        plt.close()
        
        # 2. OSI Status & Activity Status
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('License Status Distribution (Processed Data)', fontsize=14, fontweight='bold')
        
        # OSI Status
        osi_counts = self.processed_df['osi_status'].value_counts()
        colors_osi = ['#27ae60', '#e74c3c']
        axes[0].bar(osi_counts.index, osi_counts.values, color=colors_osi)
        axes[0].set_ylabel('Count')
        axes[0].set_title('OSI Approval Status')
        for i, v in enumerate(osi_counts.values):
            axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        # License Type Distribution
        license_type_dist = self.processed_df['license_type'].value_counts()
        colors_type = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']
        axes[1].bar(range(len(license_type_dist)), license_type_dist.values, color=colors_type)
        axes[1].set_xticks(range(len(license_type_dist)))
        axes[1].set_xticklabels(license_type_dist.index, rotation=45)
        axes[1].set_ylabel('Count')
        axes[1].set_title('License Type Distribution')
        for i, v in enumerate(license_type_dist.values):
            axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'license_status_distribution.png', dpi=300, bbox_inches='tight')
        logger.info("Saved: license_status_distribution.png")
        plt.close()
        
        # 3. Top 15 Licenses (Class Imbalance)
        fig, ax = plt.subplots(figsize=(12, 8))
        top_licenses = self.processed_df['license_id'].value_counts().head(15)
        colors_imbalance = plt.cm.viridis(np.linspace(0, 1, len(top_licenses)))
        
        ax.barh(range(len(top_licenses)), top_licenses.values, color=colors_imbalance)
        ax.set_yticks(range(len(top_licenses)))
        ax.set_yticklabels(top_licenses.index)
        ax.set_xlabel('Frequency')
        ax.set_title('Top 15 Most Frequent Licenses (Class Imbalance Check)', fontweight='bold')
        ax.invert_yaxis()
        
        for i, v in enumerate(top_licenses.values):
            ax.text(v + 0.05, i, str(v), va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_licenses_imbalance.png', dpi=300, bbox_inches='tight')
        logger.info("Saved: top_licenses_imbalance.png")
        plt.close()
        
        # 4. Before/After Comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = ['Total Licenses', 'After Cleaning', 'Removed\n(Deprecated/Empty)']
        values = [len(raw_df), len(self.processed_df), len(raw_df) - len(self.processed_df)]
        colors_comparison = ['#3498db', '#27ae60', '#e74c3c']
        
        bars = ax.bar(categories, values, color=colors_comparison, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Number of Licenses', fontsize=12)
        ax.set_title('Data Cleaning Impact: Before vs After', fontsize=14, fontweight='bold')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cleaning_impact.png', dpi=300, bbox_inches='tight')
        logger.info("Saved: cleaning_impact.png")
        plt.close()
        
        # 5. Text Length Box Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_to_plot = [
            raw_df['text_length'],
            self.processed_df['raw_text'].str.len(),
            self.processed_df['cleaned_text'].str.len()
        ]
        
        bp = ax.boxplot(data_to_plot, labels=['Raw License\nText', 'Processed\nRaw Text', 'Processed\nCleaned Text'],
                        patch_artist=True)
        
        colors = ['#e74c9c', '#f39c12', '#27ae60']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Text Length (characters)', fontsize=12)
        ax.set_title('Text Length Distribution: Before and After Cleaning', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'text_length_boxplot.png', dpi=300, bbox_inches='tight')
        logger.info("Saved: text_length_boxplot.png")
        plt.close()
    
    def generate_report(self, raw_df):
        """Generate comprehensive EDA report"""
        logger.info("Generating EDA report...")
        
        report = []
        report.append("=" * 80)
        report.append("LICENSE DATA CLEANING - EXPLORATORY DATA ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Summary
        report.append("\n📊 EXECUTIVE SUMMARY")
        report.append("-" * 80)
        report.append(f"Raw Licenses: {len(raw_df)}")
        report.append(f"Processed Licenses: {len(self.processed_df)}")
        report.append(f"Licenses Removed: {len(raw_df) - len(self.processed_df)} ({((len(raw_df) - len(self.processed_df)) / len(raw_df) * 100):.1f}%)")
        
        # Before cleaning
        report.append("\n\n📈 BEFORE CLEANING (RAW DATA)")
        report.append("-" * 80)
        report.append(f"\n1️⃣ DATASET SIZE & STRUCTURE")
        report.append(f"   Total licenses: {len(raw_df)}")
        report.append(f"   Deprecated: {raw_df['is_deprecated'].sum()} ({raw_df['is_deprecated'].sum() / len(raw_df) * 100:.1f}%)")
        report.append(f"   Active: {(~raw_df['is_deprecated']).sum()} ({(~raw_df['is_deprecated']).sum() / len(raw_df) * 100:.1f}%)")
        report.append(f"   OSI Approved: {raw_df['is_osi_approved'].sum()} ({raw_df['is_osi_approved'].sum() / len(raw_df) * 100:.1f}%)")
        report.append(f"   Not OSI Approved: {(~raw_df['is_osi_approved']).sum()} ({(~raw_df['is_osi_approved']).sum() / len(raw_df) * 100:.1f}%)")
        
        report.append(f"\n2️⃣ DATA QUALITY ISSUES")
        empty_texts = (raw_df['text_length'] == 0).sum()
        short_texts = (raw_df['text_length'] < 50).sum()
        report.append(f"   Empty texts: {empty_texts}")
        report.append(f"   Very short texts (<50 chars): {short_texts}")
        report.append(f"   Total problematic: {empty_texts + short_texts}")
        
        report.append(f"\n3️⃣ LICENSE TEXT LENGTH STATISTICS")
        report.append(f"   Mean: {raw_df['text_length'].mean():.0f} characters")
        report.append(f"   Median: {raw_df['text_length'].median():.0f} characters")
        report.append(f"   Std Dev: {raw_df['text_length'].std():.0f}")
        report.append(f"   Min: {raw_df['text_length'].min()} characters")
        report.append(f"   Max: {raw_df['text_length'].max()} characters")
        report.append(f"   Q25: {raw_df['text_length'].quantile(0.25):.0f}")
        report.append(f"   Q75: {raw_df['text_length'].quantile(0.75):.0f}")
        
        report.append(f"\n4️⃣ DUPLICATE DETECTION")
        duplicates = raw_df['raw_text'].duplicated().sum()
        report.append(f"   Duplicate texts found: {duplicates}")
        
        # After cleaning
        report.append("\n\n✨ AFTER CLEANING (PROCESSED DATA)")
        report.append("-" * 80)
        report.append(f"\n1️⃣ FINAL DATASET SIZE")
        report.append(f"   Total licenses: {len(self.processed_df)}")
        report.append(f"   Licenses removed: {len(raw_df) - len(self.processed_df)}")
        report.append(f"   Removal rate: {((len(raw_df) - len(self.processed_df)) / len(raw_df) * 100):.1f}%")
        report.append(f"   Retention rate: {(len(self.processed_df) / len(raw_df) * 100):.1f}%")
        
        report.append(f"\n2️⃣ TEXT STATISTICS")
        report.append(f"   Average raw text: {self.processed_df['raw_text'].str.len().mean():.0f} characters")
        report.append(f"   Average cleaned text: {self.processed_df['cleaned_text'].str.len().mean():.0f} characters")
        report.append(f"   Text compression: {(1 - self.processed_df['cleaned_text'].str.len().mean() / self.processed_df['raw_text'].str.len().mean()) * 100:.1f}%")
        
        all_words = ' '.join(self.processed_df['cleaned_text']).split()
        vocab_size = len(set(all_words))
        report.append(f"   Total tokens: {len(all_words):,}")
        report.append(f"   Vocabulary size: {vocab_size:,}")
        report.append(f"   Avg tokens/license: {len(all_words) / len(self.processed_df):.0f}")
        
        report.append(f"\n3️⃣ LABEL DISTRIBUTION")
        report.append(f"   License Types:")
        for license_type, count in self.processed_df['license_type'].value_counts().items():
            report.append(f"      • {license_type}: {count} ({count / len(self.processed_df) * 100:.1f}%)")
        
        report.append(f"\n   OSI Status:")
        for status, count in self.processed_df['osi_status'].value_counts().items():
            report.append(f"      • {status}: {count} ({count / len(self.processed_df) * 100:.1f}%)")
        
        report.append(f"\n4️⃣ CLASS IMBALANCE ANALYSIS")
        license_counts = self.processed_df['license_id'].value_counts()
        report.append(f"   Unique licenses: {len(license_counts)}")
        report.append(f"   Most frequent: {license_counts.index[0]} ({license_counts.iloc[0]} occurrences)")
        report.append(f"   Least frequent: {license_counts.index[-1]} ({license_counts.iloc[-1]} occurrences)")
        report.append(f"   Imbalance ratio: {license_counts.max() / license_counts.min():.2f}x")
        
        # Recommendations
        report.append("\n\n💡 ML READINESS ASSESSMENT")
        report.append("-" * 80)
        report.append("✅ STRENGTHS:")
        report.append(f"   • Clean dataset: {len(self.processed_df)} high-quality samples")
        report.append(f"   • Good vocabulary: {vocab_size:,} unique tokens for feature extraction")
        report.append(f"   • Balanced OSI status: {self.processed_df['osi_status'].value_counts().min()} minimum count")
        
        report.append("\n⚠️ CONSIDERATIONS:")
        imbalance_ratio = license_counts.max() / license_counts.min()
        if imbalance_ratio > 10:
            report.append(f"   • High class imbalance ({imbalance_ratio:.1f}x) - consider stratified sampling or class weighting")
        report.append(f"   • Many unique licenses ({len(license_counts)}) - large output space for classifier")
        
        report.append("\n📋 RECOMMENDATIONS FOR NEXT STEPS:")
        report.append("   1. Use stratified train-test split for imbalanced classes")
        report.append("   2. Consider ensemble methods or class weighting in ML models")
        report.append("   3. Try TF-IDF or BERT embeddings for text features")
        report.append("   4. Focus on top-20 licenses if binary classification needed")
        report.append("   5. Evaluate both accuracy and F1-score for unbalanced classes")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / 'eda_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Saved: eda_report.txt")
        print("\n" + report_text)
        
        return report_text
    
    def run_full_analysis(self):
        """Run complete EDA pipeline"""
        logger.info("=" * 80)
        logger.info("STARTING FULL EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 80)
        
        # Load data
        self.load_raw_data()
        self.load_processed_data()
        
        # Convert raw to DataFrame for analysis
        raw_df = pd.DataFrame(self.raw_licenses)
        
        # Analyze both datasets
        self.analyze_raw_data()
        self.analyze_processed_data()
        
        # Visualizations
        self.create_visualizations(raw_df)
        
        # Generate report
        self.generate_report(raw_df)
        
        logger.info("=" * 80)
        logger.info("ANALYSIS COMPLETE!")
        logger.info(f"All outputs saved to: {self.output_dir}")
        logger.info("=" * 80)


if __name__ == "__main__":
    raw_dir = r"c:\Users\ASUS\Desktop\ML project 2\data\raw\license-list-XML"
    processed_dir = r"c:\Users\ASUS\Desktop\ML project 2\data\processed"
    output_dir = r"c:\Users\ASUS\Desktop\ML project 2\analysis"
    
    eda = LicenseEDA(raw_dir, processed_dir, output_dir)
    eda.run_full_analysis()
