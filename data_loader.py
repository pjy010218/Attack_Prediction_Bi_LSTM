#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataLoader class for loading and parsing cybersecurity data files.
Supports STIX2 ATT&CK data and NVD CVE data.
"""

import json
from typing import List, Dict, Any
from stix2 import parse


class DataLoader:
    """
    A class for loading and parsing cybersecurity data files.
    
    This class provides methods to load STIX2 ATT&CK data and NVD CVE data
    from JSON files and return structured data objects.
    """
    
    def __init__(self):
        """Initialize the DataLoader."""
        pass
    
    def load_attack_stix(self, file_path: str) -> Dict[str, List[Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                bundle = parse(file.read(), allow_custom=True)
            
            attack_patterns = []
            intrusion_sets = []
            malware = []
    
            for obj in bundle.objects:
                obj_type = obj.get('type')
                if obj_type == 'attack-pattern':
                    attack_patterns.append(obj)
                elif obj_type == 'intrusion-set':
                    intrusion_sets.append(obj)
                elif obj_type == 'malware':
                    malware.append(obj)
            
            return {
                'attack_patterns': attack_patterns,
                'intrusion_sets': intrusion_sets,
                'malware': malware
            }
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {file_path}")
        except Exception as e:
            raise ValueError(f"Error parsing STIX2 data: {str(e)}")
    
    # Load NVD CVE data from a JSON feed file.
    def load_nvd_cve(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                nvd_data = json.load(file)
            
            # Extract CVE items from the feed
            # NVD CVE feeds typically have a 'CVE_Items' key
            if 'vulnerabilities' in nvd_data:
                return nvd_data.get('vulnerabilities', [])

            elif 'CVE_Items' in nvd_data:
                return nvd_data['CVE_Items']

            else:
                raise ValueError("Unsupported NVD CVE JSON format. Expected 'vulnerabilities' or 'CVE_Items' key.")
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading NVD CVE data: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Example of how to use the DataLoader class
    loader = DataLoader()
    
    # Example for loading ATT&CK STIX data
    try:
        attack_data = loader.load_attack_stix(r"D:\Github\TK_Graph\data\enterprise-attack.json")
        print(f"Loaded {len(attack_data['attack_patterns'])} attack patterns")
        print(f"Loaded {len(attack_data['intrusion_sets'])} intrusion sets")
        print(f"Loaded {len(attack_data['malware'])} malware objects")
    except Exception as e:
        print(f"Error loading ATT&CK data: {e}")
    
    # Example for loading NVD CVE data
    try:
        nvd_files = [
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2024.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2023.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2022.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2021.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2020.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2019.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2018.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2017.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2016.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2015.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2014.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2013.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2012.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2011.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2010.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2009.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2008.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2007.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2006.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2005.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2004.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2003.json",
            r"D:\Github\TK_Graph\data\nvd-cve-feed-2002.json"
        ]

        cve_data = []
        for nvd_file in nvd_files:
            try:
                items = loader.load_nvd_cve(nvd_file)
                cve_data.extend(items)
                print(f"Loaded {len(items)} CVE items from {nvd_file}")
            except Exception as fe:
                print(f"Failed to load CVE data from {nvd_file}: {fe}")

        print(f"Total CVE items aggregated: {len(cve_data)} from {len(nvd_files)} file(s)")
    except Exception as e:
        print(f"Error loading NVD CVE data: {e}")
