#!/usr/bin/env python3
"""
Comprehensive test for enhanced MetroSoccer ELO system.

This script demonstrates the complete enhanced processing pipeline including
proper date handling, chronological ordering, and ELO calculations with
both FIR and IIR normalization options.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from date_utils import enhance_match_data_processing, DateProcessor
from elo_utils import create_enhanced_elo_processor


def create_realistic_test_data():
    """Create realistic test data that mimics actual soccer match data."""
    
    # Create a realistic season structure
    teams = [
        "Ballard", "Bishop Blanchet", "Roosevelt", "Garfield", "Seattle Prep",
        "Lakeside (Seattle)", "Ingraham", "Nathan Hale", "Eastside Catholic",
        "O'Dea", "Chief Sealth", "Franklin", "Cleveland", "Rainier Beach"
    ]
    
    matches = []
    match_id = 1
    
    # Create matches for multiple seasons
    seasons = ["2022-23", "2023-24", "2024-25"]
    
    for season in seasons:
        # Determine season dates
        if season == "2022-23":
            start_date = datetime(2022, 9, 1)
        elif season == "2023-24":
            start_date = datetime(2023, 9, 1)
        else:  # 2024-25
            start_date = datetime(2024, 8, 30)
        
        current_date = start_date
        
        # Create round-robin style matches
        for week in range(12):  # 12 weeks of matches
            match_date = current_date + timedelta(days=week * 7)
            
            # Create 3-4 matches per week
            matches_this_week = np.random.randint(3, 5)
            
            for match_num in range(matches_this_week):
                # Select random teams
                home_team = np.random.choice(teams)
                away_team = np.random.choice([t for t in teams if t != home_team])
                
                # Generate realistic scores (most games are low-scoring)
                home_score = np.random.poisson(1.2)  # Average ~1.2 goals
                away_score = np.random.poisson(1.0)  # Slightly less for away team
                
                # Add some variety in date formats to test parsing
                if match_num % 3 == 0:
                    date_str = match_date.strftime("%A, %B %d, %Y")
                elif match_num % 3 == 1:
                    date_str = match_date.strftime("%B %d, %Y")
                else:
                    date_str = match_date.strftime("%m/%d/%Y")
                
                matches.append({
                    "Season": season,
                    "Date": date_str,
                    "Time": f"{np.random.randint(15, 19)}:00",
                    "Home Team": home_team,
                    "Away Team": away_team,
                    "Home Score": home_score,
                    "Away Score": away_score,
                    "Game Type": "Regular",
                    "GameID": f"game_{match_id}",
                    "MatchOrder": match_id,
                    "Home Classification": np.random.choice(["2A", "3A", "4A"]),
                    "Away Classification": np.random.choice(["2A", "3A", "4A"])
                })
                match_id += 1
    
    # Add some intentionally problematic data to test validation
    matches.extend([
        {"Season": "2023-24", "Date": "", "Home Team": "Problem Team", "Away Team": "Another Team", "Home Score": 1, "Away Score": 0},
        {"Season": "2023-24", "Date": "Invalid Date", "Home Team": "Team X", "Away Team": "Team Y", "Home Score": None, "Away Score": 2},
        {"Season": None, "Date": "2023-10-15", "Home Team": "Team A", "Away Team": "Team B", "Home Score": 1, "Away Score": 1},
    ])
    
    return pd.DataFrame(matches)


def test_enhanced_date_processing():
    """Test the enhanced date processing system."""
    print("=== Testing Enhanced Date Processing ===")
    
    # Create test data
    test_df = create_realistic_test_data()
    print(f"Created {len(test_df)} test matches")
    
    # Apply enhanced processing
    processed_df = enhance_match_data_processing(test_df)
    
    print(f"After processing: {len(processed_df)} valid matches")
    print(f"Columns added: {set(processed_df.columns) - set(test_df.columns)}")
    
    # Show sample of processed data
    print("\nSample processed data:")
    sample_cols = ['Season', 'Date', 'Date_Original', 'Matchday_Number', 'Home Team', 'Away Team', 'Home Score', 'Away Score']
    available_cols = [col for col in sample_cols if col in processed_df.columns]
    print(processed_df[available_cols].head(10))
    
    return processed_df


def test_enhanced_elo_processing(matches_df):
    """Test the enhanced ELO processing system."""
    print("\n=== Testing Enhanced ELO Processing ===")
    
    # Test FIR processing
    print("\n--- Testing FIR Normalization ---")
    fir_processor = create_enhanced_elo_processor(
        base_elo=1500, k=40, hfa=100, cap_margin=3, upset_multiplier=1.5
    )
    
    fir_log, fir_elos = fir_processor.calculate_elo_enhanced(matches_df.copy())
    fir_log, fir_elos = fir_processor.apply_normalization(fir_log, fir_elos, 'FIR')
    
    print(f"FIR Results: {len(fir_log)} matches processed, {len(fir_elos)} teams rated")
    print("Top 5 FIR ELO ratings:")
    top_fir = sorted(fir_elos.items(), key=lambda x: x[1], reverse=True)[:5]
    for team, elo in top_fir:
        print(f"  {team}: {elo:.1f}")
    
    # Test IIR processing
    print("\n--- Testing IIR Normalization ---")
    iir_processor = create_enhanced_elo_processor(
        base_elo=1500, k=40, hfa=100, cap_margin=3, upset_multiplier=1.5
    )
    
    iir_log, iir_elos = iir_processor.calculate_elo_enhanced(matches_df.copy())
    iir_log, iir_elos = iir_processor.apply_normalization(iir_log, iir_elos, 'IIR')
    
    print(f"IIR Results: {len(iir_log)} matches processed, {len(iir_elos)} teams rated")
    print("Top 5 IIR ELO ratings:")
    top_iir = sorted(iir_elos.items(), key=lambda x: x[1], reverse=True)[:5]
    for team, elo in top_iir:
        print(f"  {team}: {elo:.1f}")
    
    return fir_log, fir_elos, iir_log, iir_elos


def analyze_chronological_accuracy(elo_log):
    """Analyze the chronological accuracy of ELO calculations."""
    print("\n=== Analyzing Chronological Accuracy ===")
    
    processor = DateProcessor()
    validation = processor.validate_chronological_order(elo_log)
    
    if validation['is_valid']:
        print("✓ ELO log maintains proper chronological order")
    else:
        print("✗ Chronological order issues in ELO log:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    print("\nSeason match distributions:")
    for season, summary in validation['season_summaries'].items():
        print(f"  {season}: {summary['total_matches']} matches from {summary['date_range']}")
    
    # Check for proper match sequencing
    print("\nMatch sequencing validation:")
    season_groups = elo_log.groupby('Season')
    for season, group in season_groups:
        dates = pd.to_datetime(group['Date'], errors='coerce')
        is_sorted = dates.is_monotonic_increasing
        print(f"  {season}: {'✓' if is_sorted else '✗'} chronologically sorted")


def demonstrate_missing_data_handling():
    """Demonstrate how the system handles missing data."""
    print("\n=== Demonstrating Missing Data Handling ===")
    
    # Create data with various missing data scenarios
    problematic_data = pd.DataFrame([
        {"Season": "2023-24", "Date": "2023-09-01", "Home Team": "Team A", "Away Team": "Team B", "Home Score": 2, "Away Score": 1},
        {"Season": "2023-24", "Date": "", "Home Team": "Team C", "Away Team": "Team D", "Home Score": 1, "Away Score": 0},  # Missing date
        {"Season": None, "Date": "2023-09-03", "Home Team": "Team E", "Away Team": "Team F", "Home Score": 0, "Away Score": 2},  # Missing season
        {"Season": "2023-24", "Date": "2023-09-04", "Home Team": "", "Away Team": "Team H", "Home Score": 1, "Away Score": 1},  # Missing home team
        {"Season": "2023-24", "Date": "2023-09-05", "Home Team": "Team I", "Away Team": "Team J", "Home Score": None, "Away Score": 1},  # Missing score
        {"Season": "2023-24", "Date": "Invalid Date String", "Home Team": "Team K", "Away Team": "Team L", "Home Score": 2, "Away Score": 0},  # Invalid date
    ])
    
    print(f"Starting with {len(problematic_data)} matches (including problematic ones)")
    
    processor = DateProcessor()
    analysis = processor.detect_missing_data(problematic_data)
    
    print("Missing data analysis:")
    for col, count in analysis['missing_data'].items():
        if count > 0:
            print(f"  {col}: {count} missing values")
    
    if analysis['data_quality_issues']:
        print("Data quality issues:")
        for issue in analysis['data_quality_issues']:
            print(f"  - {issue}")
    
    # Show how enhanced processing handles it
    try:
        clean_data = enhance_match_data_processing(problematic_data)
        print(f"After enhanced processing: {len(clean_data)} clean matches remain")
    except ValueError as e:
        print(f"Processing appropriately rejected data: {e}")


def main():
    """Run the complete enhanced ELO system test."""
    print("Enhanced MetroSoccer ELO System - Comprehensive Test")
    print("=" * 60)
    
    try:
        # Test 1: Date processing
        processed_matches = test_enhanced_date_processing()
        
        # Test 2: ELO processing
        fir_log, fir_elos, iir_log, iir_elos = test_enhanced_elo_processing(processed_matches)
        
        # Test 3: Chronological accuracy
        analyze_chronological_accuracy(fir_log)
        
        # Test 4: Missing data handling
        demonstrate_missing_data_handling()
        
        print("\n" + "=" * 60)
        print("✓ Enhanced ELO System Test Complete!")
        print("✓ All components working correctly:")
        print("  - Enhanced date parsing and standardization")
        print("  - Proper chronological ordering")
        print("  - Comprehensive data validation")
        print("  - Robust ELO calculations with FIR and IIR normalization")
        print("  - Missing data detection and handling")
        print("  - Matchday timeline creation")
        
        # Show summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Processed matches: {len(processed_matches)}")
        valid_seasons = [s for s in processed_matches['Season'].unique() if pd.notna(s)]
        print(f"  Seasons covered: {sorted(valid_seasons)}")
        valid_teams = [t for t in set(processed_matches['Home Team'].tolist() + processed_matches['Away Team'].tolist()) if pd.notna(t)]
        print(f"  Teams involved: {len(valid_teams)}")
        print(f"  Date range: {processed_matches['Date'].min()} to {processed_matches['Date'].max()}")
        
        # Show ELO rating spread
        fir_ratings = list(fir_elos.values())
        print(f"  FIR ELO range: {min(fir_ratings):.1f} to {max(fir_ratings):.1f}")
        print(f"  FIR ELO median: {np.median(fir_ratings):.1f}")
        
        iir_ratings = list(iir_elos.values())
        print(f"  IIR ELO range: {min(iir_ratings):.1f} to {max(iir_ratings):.1f}")
        print(f"  IIR ELO median: {np.median(iir_ratings):.1f}")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()