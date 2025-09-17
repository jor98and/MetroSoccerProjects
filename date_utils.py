"""
Enhanced date handling utilities for Metro Soccer ELO processing.

This module provides robust date parsing, validation, and chronological ordering
functions to ensure accurate ELO model calculations that depend on proper
temporal sequencing of matches.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
import re


class DateProcessor:
    """Enhanced date processing for soccer match data."""
    
    def __init__(self):
        # Common date formats that might appear on the website
        self.date_formats = [
            '%A, %B %d, %Y',      # Monday, January 15, 2024
            '%A, %b %d, %Y',      # Mon, Jan 15, 2024  
            '%B %d, %Y',          # January 15, 2024
            '%b %d, %Y',          # Jan 15, 2024
            '%m/%d/%Y',           # 01/15/2024
            '%m-%d-%Y',           # 01-15-2024
            '%Y-%m-%d',           # 2024-01-15
            '%d/%m/%Y',           # 15/01/2024 (alternative format)
            '%m/%d/%y',           # 01/15/24
            '%m-%d-%y',           # 01-15-24
        ]
    
    def parse_date_flexible(self, date_str: str) -> pd.Timestamp:
        """
        Parse date string using multiple format attempts.
        
        Args:
            date_str: Raw date string from web scraping
            
        Returns:
            Parsed datetime as pandas Timestamp, or NaT if parsing fails
        """
        if pd.isna(date_str) or date_str == '' or date_str is None:
            return pd.NaT
        
        # Clean the date string
        date_str = str(date_str).strip()
        
        # Try each format
        for fmt in self.date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except (ValueError, TypeError):
                continue
        
        # If specific formats fail, try pandas' flexible parsing
        try:
            return pd.to_datetime(date_str, infer_datetime_format=True)
        except (ValueError, TypeError, pd.errors.ParserError):
            warnings.warn(f"Could not parse date '{date_str}'. Setting to NaT.")
            return pd.NaT
    
    def standardize_dates(self, df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
        """
        Standardize date formats in a DataFrame.
        
        Args:
            df: DataFrame containing match data
            date_column: Name of the date column to process
            
        Returns:
            DataFrame with standardized dates and additional metadata
        """
        if df.empty or date_column not in df.columns:
            return df
        
        df = df.copy()
        
        # Parse dates flexibly
        print(f"Parsing {len(df)} date entries...")
        df['Date_Parsed'] = df[date_column].apply(self.parse_date_flexible)
        
        # Keep original date strings for reference
        df['Date_Original'] = df[date_column].copy()
        
        # Replace the original date column with standardized format (YYYY-MM-DD)
        df[date_column] = df['Date_Parsed'].dt.strftime('%Y-%m-%d')
        
        # Check for parsing failures
        invalid_dates = df['Date_Parsed'].isna()
        if invalid_dates.any():
            invalid_count = invalid_dates.sum()
            print(f"Warning: {invalid_count} rows have invalid dates")
            print("Sample invalid dates:")
            print(df[invalid_dates]['Date_Original'].head().tolist())
            
            # Remove rows with unparseable dates
            df = df[~invalid_dates].copy()
            print(f"Removed {invalid_count} rows with invalid dates")
        
        # Add date metadata
        df['Date_DayOfWeek'] = df['Date_Parsed'].dt.day_name()
        df['Date_Month'] = df['Date_Parsed'].dt.month
        df['Date_Year'] = df['Date_Parsed'].dt.year
        
        # Clean up temporary column
        df = df.drop('Date_Parsed', axis=1)
        
        return df
    
    def validate_chronological_order(self, df: pd.DataFrame, 
                                   season_col: str = 'Season',
                                   date_col: str = 'Date') -> Dict:
        """
        Validate that matches are in proper chronological order within each season.
        
        Args:
            df: DataFrame with match data
            season_col: Column name for season
            date_col: Column name for dates
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'season_summaries': {}
        }
        
        if df.empty:
            validation_results['issues'].append("DataFrame is empty")
            validation_results['is_valid'] = False
            return validation_results
        
        # Convert dates to datetime for comparison
        df_temp = df.copy()
        df_temp['Date_for_validation'] = pd.to_datetime(df_temp[date_col], errors='coerce')
        
        for season in df_temp[season_col].unique():
            season_data = df_temp[df_temp[season_col] == season].copy()
            season_data = season_data.sort_index()  # Keep original order
            
            # Check if dates are monotonic (non-decreasing)
            dates = season_data['Date_for_validation'].dropna()
            if len(dates) > 1:
                is_monotonic = dates.is_monotonic_increasing
                date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                
                validation_results['season_summaries'][season] = {
                    'total_matches': len(season_data),
                    'valid_dates': len(dates),
                    'date_range': date_range,
                    'is_chronological': is_monotonic
                }
                
                if not is_monotonic:
                    validation_results['is_valid'] = False
                    validation_results['issues'].append(
                        f"Season {season}: Dates are not in chronological order"
                    )
                    
                    # Find out-of-order dates
                    for i in range(1, len(dates)):
                        if dates.iloc[i] < dates.iloc[i-1]:
                            validation_results['issues'].append(
                                f"  Date {dates.iloc[i].strftime('%Y-%m-%d')} comes after "
                                f"{dates.iloc[i-1].strftime('%Y-%m-%d')}"
                            )
        
        return validation_results
    
    def sort_matches_chronologically(self, df: pd.DataFrame, 
                                   season_col: str = 'Season',
                                   date_col: str = 'Date') -> pd.DataFrame:
        """
        Sort matches in proper chronological order within each season.
        
        Args:
            df: DataFrame with match data
            season_col: Column name for season
            date_col: Column name for dates
            
        Returns:
            DataFrame sorted chronologically
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Convert Date to datetime for proper sorting
        df['Date_for_sorting'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Sort by Season and Date (datetime)
        df = df.sort_values(by=[season_col, 'Date_for_sorting']).reset_index(drop=True)
        
        # Drop the temporary sorting column
        df = df.drop('Date_for_sorting', axis=1)
        
        return df
    
    def detect_missing_data(self, df: pd.DataFrame) -> Dict:
        """
        Detect missing or problematic data in match records.
        
        Args:
            df: DataFrame with match data
            
        Returns:
            Dictionary with missing data analysis
        """
        missing_analysis = {
            'total_rows': len(df),
            'missing_data': {},
            'data_quality_issues': []
        }
        
        if df.empty:
            missing_analysis['data_quality_issues'].append("DataFrame is empty")
            return missing_analysis
        
        # Check for missing values in key columns
        key_columns = ['Season', 'Date', 'Home Team', 'Away Team', 'Home Score', 'Away Score']
        for col in key_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_analysis['missing_data'][col] = missing_count
                if missing_count > 0:
                    missing_analysis['data_quality_issues'].append(
                        f"{col}: {missing_count} missing values"
                    )
        
        # Check for duplicate matches
        if all(col in df.columns for col in ['Season', 'Date', 'Home Team', 'Away Team']):
            duplicates = df.duplicated(subset=['Season', 'Date', 'Home Team', 'Away Team']).sum()
            if duplicates > 0:
                missing_analysis['data_quality_issues'].append(
                    f"Found {duplicates} duplicate matches"
                )
        
        # Check for invalid scores
        if 'Home Score' in df.columns and 'Away Score' in df.columns:
            invalid_scores = 0
            try:
                home_scores = pd.to_numeric(df['Home Score'], errors='coerce')
                away_scores = pd.to_numeric(df['Away Score'], errors='coerce')
                invalid_scores = (home_scores.isna() | away_scores.isna()).sum()
            except:
                invalid_scores = len(df)
            
            if invalid_scores > 0:
                missing_analysis['data_quality_issues'].append(
                    f"Found {invalid_scores} matches with invalid scores"
                )
        
        return missing_analysis
    
    def create_matchday_timeline(self, df: pd.DataFrame,
                               season_col: str = 'Season',
                               date_col: str = 'Date') -> pd.DataFrame:
        """
        Create a timeline of matchdays with proper ordering for ELO calculations.
        
        Args:
            df: DataFrame with match data
            season_col: Column name for season
            date_col: Column name for dates
            
        Returns:
            DataFrame with matchday timeline and ordering information
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure proper date format
        df['Date_dt'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Create matchday timeline per season
        matchday_info = []
        
        for season in df[season_col].unique():
            season_data = df[df[season_col] == season].copy()
            
            # Get unique dates and sort them
            unique_dates = season_data['Date_dt'].dropna().unique()
            unique_dates = sorted(unique_dates)
            
            for i, date in enumerate(unique_dates, 1):
                date_str = date.strftime('%Y-%m-%d')
                matches_on_date = len(season_data[season_data['Date_dt'] == date])
                
                matchday_info.append({
                    'Season': season,
                    'Date': date_str,
                    'Date_dt': date,
                    'Matchday_Number': i,
                    'Matches_Count': matches_on_date,
                    'Day_of_Week': date.strftime('%A'),
                    'Month': date.strftime('%B'),
                    'Year': date.year
                })
        
        matchday_df = pd.DataFrame(matchday_info)
        
        # Merge back with original data
        df = df.merge(
            matchday_df[['Season', 'Date', 'Matchday_Number']], 
            on=['Season', 'Date'], 
            how='left'
        )
        
        # Clean up temporary column
        df = df.drop('Date_dt', axis=1)
        
        return df


def enhance_match_data_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to enhance match data processing with improved date handling.
    
    Args:
        df: Raw match data DataFrame
        
    Returns:
        Processed DataFrame with enhanced date handling and validation
    """
    processor = DateProcessor()
    
    print("=== Enhanced Match Data Processing ===")
    print(f"Starting with {len(df)} matches")
    
    # Step 1: Standardize dates
    df = processor.standardize_dates(df)
    print(f"After date standardization: {len(df)} matches")
    
    # Step 2: Validate data quality
    missing_analysis = processor.detect_missing_data(df)
    if missing_analysis['data_quality_issues']:
        print("\nData Quality Issues Found:")
        for issue in missing_analysis['data_quality_issues']:
            print(f"  - {issue}")
    
    # Step 3: Sort chronologically
    df = processor.sort_matches_chronologically(df)
    print("Matches sorted chronologically")
    
    # Step 4: Validate chronological order
    validation = processor.validate_chronological_order(df)
    if validation['is_valid']:
        print("✓ Chronological order validation passed")
    else:
        print("✗ Chronological order issues found:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    # Step 5: Create matchday timeline
    df = processor.create_matchday_timeline(df)
    print("Matchday timeline created")
    
    print(f"\nFinal dataset: {len(df)} matches")
    valid_seasons = [s for s in df['Season'].unique() if pd.notna(s)]
    print(f"Seasons covered: {sorted(valid_seasons)}")
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Date processing utilities loaded successfully")