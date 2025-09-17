"""
Enhanced ELO utilities for Metro Soccer ELO processing.

This module provides enhanced ELO calculation functions that properly handle
chronological ordering and include comprehensive validation for accurate
ELO model results.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings

from date_utils import DateProcessor


class ELOProcessor:
    """Enhanced ELO processing with proper chronological handling."""
    
    def __init__(self, base_elo: float = 1500, k: float = 40, hfa: float = 100, 
                 cap_margin: int = 3, upset_multiplier: float = 1.5):
        """
        Initialize ELO processor with parameters.
        
        Args:
            base_elo: Starting ELO rating for all teams
            k: K-factor for ELO calculations
            hfa: Home field advantage adjustment
            cap_margin: Maximum margin of victory considered
            upset_multiplier: Multiplier for upset adjustments
        """
        self.base_elo = base_elo
        self.k = k
        self.hfa = hfa
        self.cap_margin = cap_margin
        self.upset_multiplier = upset_multiplier
        self.date_processor = DateProcessor()
    
    def validate_match_data(self, matches_df: pd.DataFrame) -> Dict:
        """
        Comprehensive validation of match data before ELO processing.
        
        Args:
            matches_df: DataFrame with match data
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        if matches_df.empty:
            validation_results['is_valid'] = False
            validation_results['issues'].append("No match data provided")
            return validation_results
        
        # Required columns
        required_cols = ['Season', 'Date', 'Home Team', 'Away Team', 'Home Score', 'Away Score']
        missing_cols = [col for col in required_cols if col not in matches_df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing columns: {missing_cols}")
            return validation_results
        
        # Check for missing data
        missing_analysis = self.date_processor.detect_missing_data(matches_df)
        if missing_analysis['data_quality_issues']:
            validation_results['warnings'].extend(missing_analysis['data_quality_issues'])
        
        # Validate chronological order
        chrono_validation = self.date_processor.validate_chronological_order(matches_df)
        if not chrono_validation['is_valid']:
            validation_results['is_valid'] = False
            validation_results['issues'].extend(chrono_validation['issues'])
        
        # Check for duplicate matches
        if len(matches_df) > 0:
            duplicates = matches_df.duplicated(
                subset=['Season', 'Date', 'Home Team', 'Away Team'], 
                keep=False
            ).sum()
            if duplicates > 0:
                validation_results['warnings'].append(f"Found {duplicates} potential duplicate matches")
        
        # Validate scores
        try:
            home_scores = pd.to_numeric(matches_df['Home Score'], errors='coerce')
            away_scores = pd.to_numeric(matches_df['Away Score'], errors='coerce')
            invalid_scores = (home_scores.isna() | away_scores.isna()).sum()
            if invalid_scores > 0:
                validation_results['is_valid'] = False
                validation_results['issues'].append(f"Found {invalid_scores} matches with invalid scores")
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Score validation failed: {e}")
        
        # Statistics
        validation_results['stats'] = {
            'total_matches': len(matches_df),
            'seasons': sorted([s for s in matches_df['Season'].unique().tolist() if pd.notna(s)]),
            'teams': sorted(list(set([t for t in matches_df['Home Team'].tolist() + matches_df['Away Team'].tolist() if pd.notna(t)]))),
            'date_range': self._get_date_range(matches_df)
        }
        
        return validation_results
    
    def _get_date_range(self, df: pd.DataFrame) -> str:
        """Get the date range for the dataset."""
        try:
            dates = pd.to_datetime(df['Date'], errors='coerce').dropna()
            if len(dates) > 0:
                return f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
            else:
                return "No valid dates"
        except:
            return "Date range unavailable"
    
    def prepare_matches_for_elo(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare match data for ELO calculation with enhanced processing.
        
        Args:
            matches_df: Raw match data
            
        Returns:
            Processed and validated match data ready for ELO calculation
        """
        print("=== Preparing Matches for ELO Calculation ===")
        
        # Validate input data
        validation = self.validate_match_data(matches_df)
        if not validation['is_valid']:
            raise ValueError(f"Match data validation failed: {validation['issues']}")
        
        if validation['warnings']:
            print("Warnings found:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        # Ensure proper chronological sorting
        matches_df = self.date_processor.sort_matches_chronologically(matches_df)
        
        # Add matchday timeline
        matches_df = self.date_processor.create_matchday_timeline(matches_df)
        
        # Add match sequencing
        matches_df = self._add_match_sequencing(matches_df)
        
        print(f"Prepared {len(matches_df)} matches for ELO calculation")
        print(f"Seasons: {validation['stats']['seasons']}")
        print(f"Teams: {len(validation['stats']['teams'])}")
        print(f"Date range: {validation['stats']['date_range']}")
        
        return matches_df
    
    def _add_match_sequencing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sequential numbering for matches within seasons."""
        df = df.copy()
        
        # Add global match number within each season
        df['Season_Match_Number'] = df.groupby('Season').cumcount() + 1
        
        # Add per-team match counters
        team_match_counter = defaultdict(int)
        home_match_counts = []
        away_match_counts = []
        
        for _, row in df.iterrows():
            home_team = row["Home Team"]
            away_team = row["Away Team"]
            team_match_counter[home_team] += 1
            team_match_counter[away_team] += 1
            home_match_counts.append(team_match_counter[home_team])
            away_match_counts.append(team_match_counter[away_team])
        
        df["Home Match #"] = home_match_counts
        df["Away Match #"] = away_match_counts
        
        return df
    
    def calculate_elo_enhanced(self, matches_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Calculate ELO ratings with enhanced chronological processing.
        
        Args:
            matches_df: Prepared match data
            
        Returns:
            Tuple of (ELO log DataFrame, final ELO ratings dict)
        """
        print("=== Enhanced ELO Calculation ===")
        
        # Prepare data
        matches_df = self.prepare_matches_for_elo(matches_df)
        
        # Initialize ELO tracking
        team_elos = defaultdict(lambda: self.base_elo)
        elo_log = []
        
        print(f"Starting ELO calculation with base rating {self.base_elo}")
        print(f"Parameters: K={self.k}, HFA={self.hfa}, Cap Margin={self.cap_margin}")
        
        # Process matches in chronological order
        for match_idx, (_, row) in enumerate(matches_df.iterrows()):
            season = row["Season"]
            date = row["Date"]
            home = row["Home Team"]
            away = row["Away Team"]
            
            try:
                hs = float(row["Home Score"])
                as_ = float(row["Away Score"])
            except (ValueError, TypeError):
                print(f"Warning: Skipping match {match_idx + 1} due to invalid scores")
                continue
            
            # Calculate match result
            result_home = 1 if hs > as_ else 0 if hs < as_ else 0.5
            
            # Get current ELO ratings
            home_elo_before = team_elos[home]
            away_elo_before = team_elos[away]
            
            # Calculate expected result with home field advantage
            expected_home = 1 / (1 + 10 ** ((away_elo_before - (home_elo_before + self.hfa)) / 400))
            
            # Margin of victory factor (capped)
            margin = max(1, min(abs(hs - as_), self.cap_margin))
            
            # Dynamic K adjustment for draws and upsets
            if result_home == 0.5:  # Draw
                surprise_factor = abs(result_home - expected_home)
                k_adjust = 0.5 + (self.upset_multiplier * surprise_factor)
            else:
                k_adjust = 1.0
            
            # Calculate ELO changes
            change_home = k_adjust * self.k * margin * (result_home - expected_home)
            change_away = -change_home
            
            # Update ratings
            team_elos[home] += change_home
            team_elos[away] += change_away
            
            # Log the match
            elo_log.append({
                "Season": season,
                "Date": date,
                "Matchday_Number": row.get("Matchday_Number", None),
                "Season_Match_Number": row.get("Season_Match_Number", match_idx + 1),
                "Home Team": home,
                "Away Team": away,
                "Home Score": hs,
                "Away Score": as_,
                "Home Match #": row.get("Home Match #", None),
                "Away Match #": row.get("Away Match #", None),
                "Home ELO Before": home_elo_before,
                "Away ELO Before": away_elo_before,
                "Home ELO After": team_elos[home],
                "Away ELO After": team_elos[away],
                "Home ELO Change": change_home,
                "Away ELO Change": change_away,
                "Expected Home Win": expected_home,
                "Actual Result": result_home,
                "Margin": margin,
                "K Adjustment": k_adjust,
                "Home Classification": row.get("Home Classification", "Unknown"),
                "Away Classification": row.get("Away Classification", "Unknown")
            })
        
        elo_log_df = pd.DataFrame(elo_log)
        final_elos = dict(team_elos)
        
        print(f"✓ Processed {len(elo_log_df)} matches")
        print(f"✓ Final ELO ratings calculated for {len(final_elos)} teams")
        
        return elo_log_df, final_elos
    
    def apply_normalization(self, elo_log_df: pd.DataFrame, final_elos: Dict, 
                          normalization_type: str = 'FIR') -> Tuple[pd.DataFrame, Dict]:
        """
        Apply ELO normalization (FIR or IIR).
        
        Args:
            elo_log_df: ELO calculation log
            final_elos: Final ELO ratings
            normalization_type: 'FIR' or 'IIR'
            
        Returns:
            Tuple of (normalized ELO log, normalized final ELOs)
        """
        print(f"=== Applying {normalization_type} Normalization ===")
        
        if normalization_type.upper() == 'FIR':
            return self._apply_fir_normalization(elo_log_df, final_elos)
        elif normalization_type.upper() == 'IIR':
            return self._apply_iir_normalization(elo_log_df, final_elos)
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}")
    
    def _apply_fir_normalization(self, elo_log_df: pd.DataFrame, final_elos: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Apply FIR (Finite Impulse Response) normalization."""
        final_elos_series = pd.Series(final_elos)
        shift = self.base_elo - final_elos_series.median()
        
        print(f"FIR normalization shift: {shift:.2f}")
        
        # Apply shift to final ELOs
        normalized_final_elos = {team: elo + shift for team, elo in final_elos.items()}
        
        # Apply shift to ELO log
        elo_log_normalized = elo_log_df.copy()
        for col in ["Home ELO Before", "Away ELO Before", "Home ELO After", "Away ELO After"]:
            if col in elo_log_normalized.columns:
                elo_log_normalized[col] += shift
        
        return elo_log_normalized, normalized_final_elos
    
    def _apply_iir_normalization(self, elo_log_df: pd.DataFrame, final_elos: Dict, n: int = 3) -> Tuple[pd.DataFrame, Dict]:
        """Apply IIR (Infinite Impulse Response) normalization."""
        print(f"IIR normalization with window size: {n}")
        
        # For IIR, we apply a rolling adjustment
        # This is a simplified implementation - full IIR would be more complex
        elo_log_normalized = elo_log_df.copy()
        
        # Apply rolling median adjustment per season
        for season in elo_log_normalized['Season'].unique():
            season_mask = elo_log_normalized['Season'] == season
            season_data = elo_log_normalized[season_mask].copy()
            
            # Calculate rolling adjustment
            for col in ["Home ELO After", "Away ELO After"]:
                if col in season_data.columns:
                    rolling_median = season_data[col].rolling(window=n, min_periods=1).median()
                    adjustment = self.base_elo - rolling_median
                    season_data[col] += adjustment
            
            elo_log_normalized.loc[season_mask] = season_data
        
        # Recalculate final ELOs from the normalized log
        normalized_final_elos = {}
        for team in final_elos.keys():
            home_elos = elo_log_normalized[elo_log_normalized['Home Team'] == team]['Home ELO After']
            away_elos = elo_log_normalized[elo_log_normalized['Away Team'] == team]['Away ELO After']
            
            if len(home_elos) > 0:
                normalized_final_elos[team] = home_elos.iloc[-1]
            elif len(away_elos) > 0:
                normalized_final_elos[team] = away_elos.iloc[-1]
            else:
                normalized_final_elos[team] = self.base_elo
        
        return elo_log_normalized, normalized_final_elos


def create_enhanced_elo_processor(base_elo: float = 1500, k: float = 40, 
                                hfa: float = 100, **kwargs) -> ELOProcessor:
    """
    Factory function to create an enhanced ELO processor.
    
    Args:
        base_elo: Starting ELO rating
        k: K-factor
        hfa: Home field advantage
        **kwargs: Additional parameters
        
    Returns:
        Configured ELOProcessor instance
    """
    return ELOProcessor(base_elo=base_elo, k=k, hfa=hfa, **kwargs)


if __name__ == "__main__":
    print("Enhanced ELO utilities loaded successfully")