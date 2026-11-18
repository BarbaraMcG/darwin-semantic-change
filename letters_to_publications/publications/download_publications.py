#!/usr/bin/env python3
"""
download darwin's major publications from available online sources.
uses project gutenberg, darwin online, and internet archive as sources.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# publication metadata with download sources
PUBLICATIONS = {
    "journal_of_researches_1839": {
        "title": "Journal of Researches into the Natural History and Geology",
        "year": 1839,
        "gutenberg_id": 3704,
        "gutenberg_url": "https://www.gutenberg.org/cache/epub/3704/pg3704.txt",
        "filename": "journal_of_researches_1839.txt"
    },
    "origin_1859": {
        "title": "On the Origin of Species by Means of Natural Selection",
        "year": 1859,
        "gutenberg_id": 1228,
        "gutenberg_url": "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
        "filename": "origin_of_species_1859.txt"
    },
    "orchids_1862": {
        "title": "On the Various Contrivances by Which British and Foreign Orchids are Fertilised by Insects",
        "year": 1862,
        "gutenberg_id": 2087,
        "gutenberg_url": "https://www.gutenberg.org/cache/epub/2087/pg2087.txt",
        "filename": "orchids_1862.txt"
    },
    "climbing_plants_1865": {
        "title": "On the Movements and Habits of Climbing Plants",
        "year": 1865,
        "gutenberg_id": 2084,
        "gutenberg_url": "https://www.gutenberg.org/cache/epub/2084/pg2084.txt",
        "filename": "climbing_plants_1865.txt"
    },
    "descent_1871": {
        "title": "The Descent of Man and Selection in Relation to Sex",
        "year": 1871,
        "gutenberg_id": 2300,
        "gutenberg_url": "https://www.gutenberg.org/cache/epub/2300/pg2300.txt",
        "filename": "descent_of_man_1871.txt"
    },
    "expression_1872": {
        "title": "The Expression of the Emotions in Man and Animals",
        "year": 1872,
        "gutenberg_id": 1227,
        "gutenberg_url": "https://www.gutenberg.org/cache/epub/1227/pg1227.txt",
        "filename": "expression_of_emotions_1872.txt"
    },
    "insectivorous_1875": {
        "title": "Insectivorous Plants",
        "year": 1875,
        "gutenberg_id": 5765,
        "gutenberg_url": "https://www.gutenberg.org/cache/epub/5765/pg5765.txt",
        "filename": "insectivorous_plants_1875.txt"
    },
    "fertilisation_1876": {
        "title": "The Effects of Cross and Self Fertilisation in the Vegetable Kingdom",
        "year": 1876,
        "gutenberg_id": 4346,
        "gutenberg_url": "https://www.gutenberg.org/cache/epub/4346/pg4346.txt",
        "filename": "cross_and_self_fertilisation_1876.txt"
    },
    "forms_of_flowers_1877": {
        "title": "The Different Forms of Flowers on Plants of the Same Species",
        "year": 1877,
        "gutenberg_id": 4183,
        "gutenberg_url": "https://www.gutenberg.org/cache/epub/4183/pg4183.txt",
        "filename": "forms_of_flowers_1877.txt"
    },
    "movement_in_plants_1880": {
        "title": "The Power of Movement in Plants",
        "year": 1880,
        "gutenberg_id": 5605,
        "gutenberg_url": "https://www.gutenberg.org/cache/epub/5605/pg5605.txt",
        "filename": "movement_in_plants_1880.txt"
    }
}

# note: living and fossil cirripedia (1851-4) and erasmus darwin (1897) 
# are not readily available on project gutenberg. will need manual download
# from darwin online: http://darwin-online.org.uk/

MISSING_PUBLICATIONS = [
    {
        "title": "A Monograph of the Sub-class Cirripedia (Living and Fossil)",
        "year": "1851-1854",
        "note": "Available at: http://darwin-online.org.uk/EditorialIntroductions/Freeman_BarnacleCirripedia.html",
        "manual_download": True
    },
    {
        "title": "The Life of Erasmus Darwin",
        "year": 1897,
        "note": "Available at: http://darwin-online.org.uk/converted/published/1887_LifeErasmus_F1497/1887_LifeErasmus_F1497.html",
        "manual_download": True
    }
]


def download_file(url: str, output_path: Path, timeout: int = 30) -> bool:
    """
    download a file from url to output path.
    
    args:
        url: source url
        output_path: destination file path
        timeout: request timeout in seconds
        
    returns:
        bool indicating success
    """
    try:
        print(f"downloading: {output_path.name}...")
        
        req = Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Darwin Research Project - Academic Use)'
        })
        
        with urlopen(req, timeout=timeout) as response:
            content = response.read().decode('utf-8')
        
        # write content to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✓ downloaded {len(content):,} characters")
        return True
        
    except (URLError, HTTPError) as e:
        print(f"  ✗ failed to download: {e}")
        return False


def download_all_publications(output_dir: Path) -> Dict[str, bool]:
    """
    download all available publications.
    
    args:
        output_dir: directory to save publications
        
    returns:
        dict mapping publication keys to download success status
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    for key, pub_info in PUBLICATIONS.items():
        output_path = output_dir / pub_info['filename']
        
        # skip if already exists
        if output_path.exists():
            print(f"skipping {pub_info['filename']} (already exists)")
            results[key] = True
            continue
        
        # download
        success = download_file(pub_info['gutenberg_url'], output_path)
        results[key] = success
        
        # be respectful to project gutenberg servers
        if success:
            time.sleep(2)
    
    return results


def create_metadata_file(output_dir: Path):
    """create a metadata file documenting all publications and sources."""
    metadata_path = output_dir / "publications_metadata.md"
    
    with open(metadata_path, 'w') as f:
        f.write("# Darwin Publications Metadata\n\n")
        f.write("This directory contains Darwin's major publications used for Stage 3 analysis.\n\n")
        f.write("## Downloaded Publications\n\n")
        
        for key, pub in PUBLICATIONS.items():
            f.write(f"### {pub['title']} ({pub['year']})\n\n")
            f.write(f"- **File**: `{pub['filename']}`\n")
            f.write(f"- **Source**: Project Gutenberg (ID: {pub['gutenberg_id']})\n")
            f.write(f"- **URL**: {pub['gutenberg_url']}\n")
            f.write(f"- **Year**: {pub['year']}\n\n")
        
        f.write("## Publications Requiring Manual Download\n\n")
        f.write("The following publications are not available on Project Gutenberg ")
        f.write("and must be downloaded manually from Darwin Online:\n\n")
        
        for pub in MISSING_PUBLICATIONS:
            f.write(f"### {pub['title']} ({pub['year']})\n\n")
            f.write(f"- **Note**: {pub['note']}\n")
            f.write(f"- **Manual Download Required**: Yes\n\n")
        
        f.write("\n## Sources\n\n")
        f.write("- **Project Gutenberg**: https://www.gutenberg.org/\n")
        f.write("- **Darwin Online**: http://darwin-online.org.uk/\n")
        f.write("- **Darwin Correspondence Project**: https://www.darwinproject.ac.uk/\n\n")
        
        f.write("## Usage Notes\n\n")
        f.write("- All texts are in plain text format (UTF-8 encoding)\n")
        f.write("- Project Gutenberg texts include standard header/footer boilerplate\n")
        f.write("- For embedding generation, you may want to strip PG headers/footers\n")
        f.write("- Texts are from various editions - verify edition details if needed\n")
    
    print(f"\nmetadata file created: {metadata_path}")


def main():
    """main execution function."""
    # determine output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir
    
    print("=" * 70)
    print("Darwin Publications Downloader")
    print("=" * 70)
    print(f"\noutput directory: {output_dir}")
    print(f"publications to download: {len(PUBLICATIONS)}")
    print()
    
    # download publications
    results = download_all_publications(output_dir)
    
    # summary
    successful = sum(1 for v in results.values() if v)
    print("\n" + "=" * 70)
    print(f"download complete: {successful}/{len(PUBLICATIONS)} successful")
    print("=" * 70)
    
    # create metadata
    create_metadata_file(output_dir)
    
    # note about manual downloads
    if MISSING_PUBLICATIONS:
        print(f"\n⚠ Note: {len(MISSING_PUBLICATIONS)} publications require manual download")
        print("see publications_metadata.md for details")


if __name__ == "__main__":
    main()

