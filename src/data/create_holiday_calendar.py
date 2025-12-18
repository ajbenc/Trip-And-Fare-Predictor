"""
NYC 2022 Holiday Calendar Creator
==================================
Creates a comprehensive holiday calendar for 2022 including:
- Federal holidays
- NYC-specific events
- Major sporting events
- Holiday weeks and surrounding days

This is static data - no API needed!

Author: NYC Taxi ML Team
Date: 2024
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def create_2022_holidays():
    """
    Create comprehensive 2022 holiday calendar for NYC.
    """
    print("="*80)
    print("🎉 NYC 2022 HOLIDAY CALENDAR CREATOR")
    print("="*80)
    print()
    
    holidays = []
    
    # ==================== FEDERAL HOLIDAYS ====================
    print("📅 Adding Federal Holidays...")
    
    federal_holidays = [
        {'date': '2022-01-01', 'name': 'New Year\'s Day', 'type': 'federal', 'major': True},
        {'date': '2022-01-17', 'name': 'Martin Luther King Jr. Day', 'type': 'federal', 'major': False},
        {'date': '2022-02-21', 'name': 'Presidents\' Day', 'type': 'federal', 'major': False},
        {'date': '2022-05-30', 'name': 'Memorial Day', 'type': 'federal', 'major': True},
        {'date': '2022-06-20', 'name': 'Juneteenth', 'type': 'federal', 'major': False},  # Observed June 20 (19th was Sunday)
        {'date': '2022-07-04', 'name': 'Independence Day', 'type': 'federal', 'major': True},
        {'date': '2022-09-05', 'name': 'Labor Day', 'type': 'federal', 'major': True},
        {'date': '2022-10-10', 'name': 'Columbus Day', 'type': 'federal', 'major': False},
        {'date': '2022-11-11', 'name': 'Veterans Day', 'type': 'federal', 'major': False},
        {'date': '2022-11-24', 'name': 'Thanksgiving', 'type': 'federal', 'major': True},
        {'date': '2022-12-25', 'name': 'Christmas Day', 'type': 'federal', 'major': True},
        {'date': '2022-12-26', 'name': 'Christmas (Observed)', 'type': 'federal', 'major': True},  # Sunday observed Monday
    ]
    
    holidays.extend(federal_holidays)
    print(f"   ✅ Added {len(federal_holidays)} federal holidays")
    
    # ==================== NYC-SPECIFIC EVENTS ====================
    print("🗽 Adding NYC-Specific Events...")
    
    nyc_events = [
        {'date': '2022-01-31', 'name': 'Chinese New Year (Lunar)', 'type': 'cultural', 'major': False},
        {'date': '2022-03-17', 'name': 'St. Patrick\'s Day', 'type': 'cultural', 'major': True},
        {'date': '2022-03-17', 'name': 'St. Patrick\'s Day Parade', 'type': 'parade', 'major': True},
        {'date': '2022-04-15', 'name': 'Good Friday', 'type': 'religious', 'major': False},
        {'date': '2022-04-17', 'name': 'Easter Sunday', 'type': 'religious', 'major': False},
        {'date': '2022-06-12', 'name': 'Puerto Rican Day Parade', 'type': 'parade', 'major': True},
        {'date': '2022-11-06', 'name': 'NYC Marathon', 'type': 'sporting', 'major': True},
        {'date': '2022-11-24', 'name': 'Macy\'s Thanksgiving Day Parade', 'type': 'parade', 'major': True},
        {'date': '2022-12-31', 'name': 'New Year\'s Eve', 'type': 'celebration', 'major': True},
    ]
    
    holidays.extend(nyc_events)
    print(f"   ✅ Added {len(nyc_events)} NYC-specific events")
    
    # ==================== BLACK FRIDAY & CYBER MONDAY ====================
    print("🛍️ Adding Shopping Events...")
    
    shopping_events = [
        {'date': '2022-11-25', 'name': 'Black Friday', 'type': 'shopping', 'major': True},
        {'date': '2022-11-28', 'name': 'Cyber Monday', 'type': 'shopping', 'major': False},
    ]
    
    holidays.extend(shopping_events)
    print(f"   ✅ Added {len(shopping_events)} shopping events")
    
    # ==================== HOLIDAY WEEKS ====================
    print("📆 Adding Holiday Weeks...")
    
    # Christmas/New Year week (Dec 24-Jan 2)
    holiday_weeks = [
        {'date': '2022-12-24', 'name': 'Christmas Eve', 'type': 'holiday_week', 'major': True},
        {'date': '2022-12-23', 'name': 'Christmas Week', 'type': 'holiday_week', 'major': False},
        {'date': '2022-12-27', 'name': 'Christmas Week', 'type': 'holiday_week', 'major': False},
        {'date': '2022-12-28', 'name': 'Christmas Week', 'type': 'holiday_week', 'major': False},
        {'date': '2022-12-29', 'name': 'Christmas Week', 'type': 'holiday_week', 'major': False},
        {'date': '2022-12-30', 'name': 'New Year\'s Eve Week', 'type': 'holiday_week', 'major': False},
    ]
    
    holidays.extend(holiday_weeks)
    print(f"   ✅ Added {len(holiday_weeks)} holiday week days")
    
    # ==================== CREATE DATAFRAME ====================
    print()
    print("📊 Creating holiday calendar DataFrame...")
    
    holidays_df = pd.DataFrame(holidays)
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    
    # Sort by date
    holidays_df = holidays_df.sort_values('date').reset_index(drop=True)
    
    # Add day of week
    holidays_df['day_of_week'] = holidays_df['date'].dt.day_name()
    
    print(f"✅ Created calendar with {len(holidays_df)} holidays and events")
    print()
    
    # ==================== ADD SURROUNDING DAYS ====================
    print("🔄 Adding days before/after major holidays...")
    
    # For each major holiday, mark the day before and after
    major_holidays = holidays_df[holidays_df['major'] == True]['date'].tolist()
    
    surrounding_days = []
    for holiday_date in major_holidays:
        # Day before
        day_before = {
            'date': holiday_date - timedelta(days=1),
            'name': f'Day Before {holidays_df[holidays_df["date"]==holiday_date]["name"].values[0]}',
            'type': 'surrounding',
            'major': False,
            'day_of_week': (holiday_date - timedelta(days=1)).strftime('%A')
        }
        surrounding_days.append(day_before)
        
        # Day after
        day_after = {
            'date': holiday_date + timedelta(days=1),
            'name': f'Day After {holidays_df[holidays_df["date"]==holiday_date]["name"].values[0]}',
            'type': 'surrounding',
            'major': False,
            'day_of_week': (holiday_date + timedelta(days=1)).strftime('%A')
        }
        surrounding_days.append(day_after)
    
    # Add surrounding days to DataFrame
    surrounding_df = pd.DataFrame(surrounding_days)
    holidays_df = pd.concat([holidays_df, surrounding_df], ignore_index=True)
    
    # Remove duplicates (same date might be both a holiday and a surrounding day)
    holidays_df = holidays_df.drop_duplicates(subset=['date'], keep='first')
    
    # Sort again
    holidays_df = holidays_df.sort_values('date').reset_index(drop=True)
    
    print(f"✅ Added surrounding days (total now: {len(holidays_df)} dates)")
    print()
    
    # ==================== SAVE TO FILE ====================
    output_dir = Path('data/external')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'holidays_2022.csv'
    
    holidays_df.to_csv(output_file, index=False)
    
    print("="*80)
    print("💾 HOLIDAY CALENDAR SAVED!")
    print("="*80)
    print(f"📁 File: {output_file}")
    print(f"📊 Total dates: {len(holidays_df)}")
    print()
    
    # ==================== PRINT SUMMARY ====================
    _print_summary(holidays_df)
    
    print("="*80)
    print("✅ SUCCESS! Holiday calendar ready for ML training")
    print("="*80)
    print()
    print("🎯 Next steps:")
    print("   1. Feature engineering: Merge taxi + weather + holidays")
    print("   2. Create temporal split (train/val/test)")
    print("   3. Train models with enhanced features")


def _print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("📊 HOLIDAY CALENDAR SUMMARY - 2022")
    print("="*80)
    print()
    
    print(f"📅 Coverage:")
    print(f"   First date: {df['date'].min().strftime('%Y-%m-%d')}")
    print(f"   Last date: {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Total dates: {len(df)}")
    print()
    
    print(f"🎉 Breakdown by Type:")
    type_counts = df['type'].value_counts()
    for holiday_type, count in type_counts.items():
        print(f"   {holiday_type.capitalize():20} {count:3} dates")
    print()
    
    print(f"⭐ Major Holidays/Events:")
    major = df[df['major'] == True].sort_values('date')
    for _, row in major.iterrows():
        print(f"   {row['date'].strftime('%Y-%m-%d')} ({row['day_of_week']:9}) - {row['name']}")
    print()
    
    # Monthly distribution
    print(f"📆 Monthly Distribution:")
    df['month'] = df['date'].dt.month
    monthly = df.groupby('month').size()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for month in range(1, 13):
        if month in monthly.index:
            count = monthly[month]
            bar = '█' * count
            print(f"   {month_names[month-1]:>3}: {bar} ({count})")
    print()


if __name__ == '__main__':
    create_2022_holidays()
