"""
Download NYC Yellow Taxi Trip Data for May 2022
================================================

This script downloads the Parquet file from NYC TLC official website.
The dataset contains trip records for yellow taxis in May 2022.

Data Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
"""

import requests
import os
from pathlib import Path

def download_taxi_data(year=2022, month=5):
    """
    Download NYC Yellow Taxi trip data for a specific month.
    
    Args:
        year (int): Year of the data (e.g., 2022)
        month (int): Month of the data (1-12)
    """
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct URL
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
    filepath = data_dir / filename
    
    # Check if file already exists
    if filepath.exists():
        print(f"✅ File already exists: {filepath}")
        print(f"   Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        return filepath
    
    print(f"📥 Downloading NYC Taxi Data for {year}-{month:02d}...")
    print(f"   URL: {url}")
    print(f"   Destination: {filepath}")
    
    try:
        # Download with progress indication
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.2f} MB / {total_size / 1024 / 1024:.2f} MB)", end='')
        
        print(f"\n✅ Download completed successfully!")
        print(f"   File saved to: {filepath}")
        print(f"   Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        return filepath
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading file: {e}")
        if filepath.exists():
            filepath.unlink()  # Remove partial file
        return None

def verify_parquet_file(filepath):
    """
    Verify that the downloaded Parquet file is valid and readable.
    
    Args:
        filepath (Path): Path to the Parquet file
    """
    try:
        import pandas as pd
        
        print(f"\n🔍 Verifying Parquet file...")
        df = pd.read_parquet(filepath)
        
        print(f"✅ File is valid and readable!")
        print(f"\n📊 Dataset Overview:")
        print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"\n   Columns:")
        for col in df.columns:
            print(f"      • {col} ({df[col].dtype})")
        
        print(f"\n   Sample Data (first 3 rows):")
        print(df.head(3))
        
        print(f"\n   Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying file: {e}")
        return False

if __name__ == "__main__":
    print("="*70)
    print("NYC YELLOW TAXI DATA DOWNLOAD - MAY 2022")
    print("="*70)
    
    # Download May 2022 data
    filepath = download_taxi_data(year=2022, month=5)
    
    if filepath:
        # Verify the downloaded file
        verify_parquet_file(filepath)
        
        print("\n" + "="*70)
        print("✅ DATA DOWNLOAD AND VERIFICATION COMPLETE!")
        print("="*70)
        print(f"\n📁 Data location: {filepath}")
        print(f"\n🚀 Next steps:")
        print(f"   1. Run exploratory data analysis (EDA)")
        print(f"   2. Preprocess and engineer features")
        print(f"   3. Train machine learning models")
    else:
        print("\n" + "="*70)
        print("❌ DATA DOWNLOAD FAILED")
        print("="*70)
        print(f"\n📝 Manual download instructions:")
        print(f"   1. Visit: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page")
        print(f"   2. Download: yellow_tripdata_2022-05.parquet")
        print(f"   3. Place in: data/raw/ directory")
