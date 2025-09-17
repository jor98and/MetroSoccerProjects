#!/usr/bin/env python3
"""
Test script for enhanced date processing utilities.

This script validates the date processing enhancements by testing various
scenarios and edge cases that might occur in soccer match data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from date_utils import enhance_match_data_processing, DateProcessor


def create_test_data():
    """Create test data with various date formats and edge cases."""
    
    # Test data with various date formats
    test_matches = [
        # Good data
        {"Season": "2023-24", "Date": "Monday, September 4, 2023", "Home Team": "Team A", "Away Team": "Team B", "Home Score": 2, "Away Score": 1},
        {"Season": "2023-24", "Date": "September 8, 2023", "Home Team": "Team C", "Away Team": "Team D", "Home Score": 1, "Away Score": 3},
        {"Season": "2023-24", "Date": "09/12/2023", "Home Team": "Team A", "Away Team": "Team C", "Home Score": 0, "Away Score": 0},
        {"Season": "2023-24", "Date": "2023-09-15", "Home Team": "Team B", "Away Team": "Team D", "Home Score": 3, "Away Score": 2},
        
        # Different season
        {"Season": "2024-25", "Date": "August 30, 2024", "Home Team": "Team A", "Away Team": "Team B", "Home Score": 1, "Away Score": 2},
        {"Season": "2024-25", "Date": "Friday, September 6, 2024", "Home Team": "Team C", "Away Team": "Team D", "Home Score": 2, "Away Score": 1},
        
        # Edge cases
        {"Season": "2023-24", "Date": "Dec 1, 2023", "Home Team": "Team E", "Away Team": "Team F", "Home Score": 1, "Away Score": 0},
        {"Season": "2023-24", "Date": "12/15/2023", "Home Team": "Team E", "Away Team": "Team F", "Home Score": 2, "Away Score": 1},
        
        # Problematic data (will be caught by validation)
        {"Season": "2023-24", "Date": "", "Home Team": "Team G", "Away Team": "Team H", "Home Score": 1, "Away Score": 1},
        {"Season": "2023-24", "Date": "Invalid Date", "Home Team": "Team I", "Away Team": "Team J", "Home Score": None, "Away Score": 2},
        {"Season": "2023-24", "Date": "2023-09-10", "Home Team": "", "Away Team": "Team K", "Home Score": 2, "Away Score": 1},
    ]
    
    return pd.DataFrame(test_matches)


def test_date_parsing():
    """Test the date parsing functionality."""
    print("=== Testing Date Parsing ===")
    
    processor = DateProcessor()
    
    # Test various date formats
    test_dates = [
        "Monday, September 4, 2023",
        "September 4, 2023", 
        "Sep 4, 2023",
        "09/04/2023",
        "2023-09-04",
        "9/4/2023",
        "9-4-2023",
        "",
        None,
        "Invalid Date",
        "2023-13-45"  # Invalid date
    ]
    
    print("Testing date parsing with various formats:")
    for date_str in test_dates:
        parsed = processor.parse_date_flexible(date_str)
        status = "✓" if not pd.isna(parsed) else "✗"
        print(f"  {status} '{date_str}' → {parsed}")
    
    return True


def test_chronological_validation():
    """Test chronological order validation."""
    print("\n=== Testing Chronological Validation ===")
    
    processor = DateProcessor()
    
    # Create test data with out-of-order dates
    test_data = pd.DataFrame([
        {"Season": "2023-24", "Date": "2023-09-01", "Home Team": "A", "Away Team": "B"},
        {"Season": "2023-24", "Date": "2023-09-03", "Home Team": "C", "Away Team": "D"},
        {"Season": "2023-24", "Date": "2023-09-02", "Home Team": "E", "Away Team": "F"},  # Out of order
        {"Season": "2024-25", "Date": "2024-08-30", "Home Team": "A", "Away Team": "C"},
        {"Season": "2024-25", "Date": "2024-09-01", "Home Team": "B", "Away Team": "D"},
    ])
    
    print("Testing chronological validation:")
    validation = processor.validate_chronological_order(test_data)
    
    print(f"  Overall valid: {validation['is_valid']}")
    if validation['issues']:
        print("  Issues found:")
        for issue in validation['issues']:
            print(f"    - {issue}")
    
    print("  Season summaries:")
    for season, summary in validation['season_summaries'].items():
        print(f"    {season}: {summary}")
    
    return validation


def test_missing_data_detection():
    """Test missing data detection."""
    print("\n=== Testing Missing Data Detection ===")
    
    processor = DateProcessor()
    
    # Create test data with missing values
    test_data = pd.DataFrame([
        {"Season": "2023-24", "Date": "2023-09-01", "Home Team": "A", "Away Team": "B", "Home Score": 2, "Away Score": 1},
        {"Season": "2023-24", "Date": None, "Home Team": "C", "Away Team": "D", "Home Score": 1, "Away Score": 3},
        {"Season": "2023-24", "Date": "2023-09-03", "Home Team": "", "Away Team": "F", "Home Score": 0, "Away Score": 0},
        {"Season": None, "Date": "2023-09-04", "Home Team": "G", "Away Team": "H", "Home Score": "invalid", "Away Score": 2},
    ])
    
    print("Testing missing data detection:")
    analysis = processor.detect_missing_data(test_data)
    
    print(f"  Total rows: {analysis['total_rows']}")
    print("  Missing data by column:")
    for col, count in analysis['missing_data'].items():
        print(f"    {col}: {count} missing")
    
    if analysis['data_quality_issues']:
        print("  Data quality issues:")
        for issue in analysis['data_quality_issues']:
            print(f"    - {issue}")
    
    return analysis


def test_full_processing():
    """Test the full enhanced processing pipeline."""
    print("\n=== Testing Full Processing Pipeline ===")
    
    # Create comprehensive test data
    test_df = create_test_data()
    print(f"Created test data with {len(test_df)} matches")
    
    print("\nOriginal data preview:")
    print(test_df.head())
    
    # Apply enhanced processing
    try:
        processed_df = enhance_match_data_processing(test_df)
        print(f"\nProcessed data: {len(processed_df)} matches remaining")
        
        print("\nProcessed data columns:")
        print(processed_df.columns.tolist())
        
        print("\nSample processed data:")
        print(processed_df[['Season', 'Date', 'Date_Original', 'Matchday_Number', 'Home Team', 'Away Team']].head(10))
        
        return processed_df
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return None


def test_matchday_timeline():
    """Test matchday timeline creation."""
    print("\n=== Testing Matchday Timeline ===")
    
    processor = DateProcessor()
    
    # Create test data with multiple matches on same dates
    test_data = pd.DataFrame([
        {"Season": "2023-24", "Date": "2023-09-01", "Home Team": "A", "Away Team": "B"},
        {"Season": "2023-24", "Date": "2023-09-01", "Home Team": "C", "Away Team": "D"},  # Same date
        {"Season": "2023-24", "Date": "2023-09-03", "Home Team": "E", "Away Team": "F"},
        {"Season": "2023-24", "Date": "2023-09-05", "Home Team": "G", "Away Team": "H"},
        {"Season": "2024-25", "Date": "2024-08-30", "Home Team": "A", "Away Team": "C"},
    ])
    
    print("Creating matchday timeline:")
    timeline_df = processor.create_matchday_timeline(test_data)
    
    print("Timeline results:")
    print(timeline_df[['Season', 'Date', 'Matchday_Number', 'Home Team', 'Away Team']])
    
    return timeline_df


def main():
    """Run all tests."""
    print("MetroSoccer Enhanced Date Processing - Test Suite")
    print("=" * 60)
    
    try:
        # Run individual tests
        test_date_parsing()
        test_chronological_validation()
        test_missing_data_detection()
        test_matchday_timeline()
        
        # Run full pipeline test
        processed_data = test_full_processing()
        
        print("\n" + "=" * 60)
        if processed_data is not None:
            print("✓ All tests completed successfully!")
            print(f"✓ Final processed dataset: {len(processed_data)} matches")
            
            # Show final validation
            processor = DateProcessor() 
            final_validation = processor.validate_chronological_order(processed_data)
            if final_validation['is_valid']:
                print("✓ Final dataset passes chronological validation")
            else:
                print("✗ Final dataset has chronological issues")
                
        else:
            print("✗ Some tests failed")
            
    except Exception as e:
        print(f"✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()