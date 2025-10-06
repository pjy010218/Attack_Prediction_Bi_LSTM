import json
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase
from data_loader import DataLoader
from pathlib import Path

class ThreatGraphBuilder:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def create_ttp_nodes(self, attack_data: Dict[str, List[Any]]) -> None:
        """
        Create 'TTP' nodes from STIX 'attack-pattern' objects, including the TTP ID (TID).
        """
        attack_patterns = attack_data.get("attack_patterns", [])
        rows: List[Dict[str, Any]] = []
        
        for ap in attack_patterns:
            ap_id = ap.get("id")
            if not ap_id:
                continue

            tid = ""
            if "external_references" in ap:
                for ref in ap.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        tid = ref.get("external_id")
                        break
            
            rows.append({
                "id": ap_id,
                "name": ap.get("name", ""),
                "description": ap.get("description", ""),
                "tid": tid
            })

        if not rows:
            return

        cypher = (
            "UNWIND $rows AS row "
            "MERGE (t:TTP {id: row.id}) "
            "SET t.name = row.name, t.description = row.description, t.tid = row.tid"
        )

        with self.driver.session(database=self.database) as session:
            session.run(cypher, rows=rows)

    def create_cve_cwe_nodes(self, cve_items: list, batch_size: int = 1000) -> None:
        """
        CVE와 CWE 노드 및 관계를 배치(batch) 단위로 효율적으로 생성합니다.
        구버전 NVD JSON 파일 구조를 정확히 파싱합니다.
        """
        print("  - Preparing CVE data for ingestion...")
        all_rows = []
        for item in cve_items:
            try:
                cve_id = item['cve']['CVE_data_meta']['ID']
                
                description = ""
                for desc_data in item['cve']['description']['description_data']:
                    if desc_data['lang'] == 'en':
                        description = desc_data['value']
                        break
                
                cvss_score = None
                if 'impact' in item and 'baseMetricV3' in item['impact']:
                    cvss_score = item['impact']['baseMetricV3']['cvssV3'].get('baseScore')
                elif 'impact' in item and 'baseMetricV2' in item['impact']:
                    cvss_score = item['impact']['baseMetricV2']['cvssV2'].get('baseScore')

                cwe_ids = []
                for problem_data in item['cve']['problemtype']['problemtype_data']:
                    for desc in problem_data['description']:
                        cwe_value = desc.get('value')
                        if cwe_value and 'CWE-' in cwe_value and cwe_value != "NVD-CWE-noinfo":
                            cwe_ids.append(cwe_value)
                
                all_rows.append({
                    "cve_id": cve_id,
                    "description": description,
                    "cvss_score": cvss_score,
                    "cwes": list(set(cwe_ids)),
                })
            except (KeyError, IndexError, TypeError):
                continue

        if not all_rows:
            print("  - No valid CVE data could be processed.")
            return

        cypher = """
        UNWIND $rows AS row
        MERGE (c:CVE {id: row.cve_id})
        SET c.description = row.description, c.cvss_score = row.cvss_score
        FOREACH (cwe_id IN row.cwes |
            MERGE (w:CWE {id: cwe_id})
            MERGE (c)-[r:IS_INSTANCE_OF]->(w)
            SET r.weight = 2.0
        )
        """

        total_rows = len(all_rows)
        print(f"  - Total {total_rows} valid CVE records to process.")
        with self.driver.session(database=self.database) as session:
            for i in range(0, total_rows, batch_size):
                batch = all_rows[i:i + batch_size]
                try:
                    session.run(cypher, rows=batch)
                    print(f"  - Processed batch {i // batch_size + 1} / {(total_rows + batch_size - 1) // batch_size} ({len(batch)} items)")
                except Exception as e:
                    print(f"  - ❌ ERROR processing batch {i // batch_size + 1}: {e}")

    def create_actor_ttp_relationships(self, attack_data: Dict[str, List[Any]], stix_file_path: Path) -> None:
        """Create 'Actor' nodes and 'USES' relationships to 'TTP' with weights."""
        intrusion_sets = attack_data.get("intrusion_sets", [])
        intrusion_set_map = {iset.get("id"): iset.get("name", "") for iset in intrusion_sets if iset.get("id")}

        with open(stix_file_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        rows: List[Dict[str, Any]] = []
        for obj in bundle.get("objects", []):
            if obj.get("type") == "relationship" and obj.get("relationship_type") == "uses":
                source_ref = obj.get("source_ref", "")
                target_ref = obj.get("target_ref", "")
                if source_ref.startswith("intrusion-set--") and target_ref.startswith("attack-pattern--"):
                    rows.append({
                        "actor_id": source_ref,
                        "actor_name": intrusion_set_map.get(source_ref, ""),
                        "ttp_id": target_ref,
                    })
        if not rows: return

        cypher = (
            "UNWIND $rows AS row "
            "MERGE (a:Actor {id: row.actor_id}) SET a.name = row.actor_name "
            "MERGE (t:TTP {id: row.ttp_id}) "
            "MERGE (a)-[r:USES]->(t) SET r.weight = 1.0"
        )
        with self.driver.session(database=self.database) as session:
            session.run(cypher, rows=rows)

    def create_tactic_nodes_and_relationships(self, attack_data: Dict[str, List[Any]]) -> None:
        """Tactic 노드를 생성하고, TTP와 Tactic 사이에 BELONGS_TO 관계를 생성합니다."""
        attack_patterns = attack_data.get("attack_patterns", [])
        rows = []
        for ap in attack_patterns:
            ttp_id = ap.get("id")
            if not ttp_id or not ap.get("kill_chain_phases"):
                continue
            for phase in ap.get("kill_chain_phases", []):
                if phase.get("kill_chain_name") == "mitre-attack":
                    rows.append({
                        "ttp_id": ttp_id,
                        "tactic_name": phase.get("phase_name")
                    })
        
        if not rows: return
        cypher = """
        UNWIND $rows as row
        MATCH (t:TTP {id: row.ttp_id})
        MERGE (tac:Tactic {name: row.tactic_name})
        MERGE (t)-[:BELONGS_TO]->(tac)
        """
        with self.driver.session(database=self.database) as session:
            session.run(cypher, rows=rows)

    def create_ttp_sequence_relationships(self) -> None:
        """Tactic의 '단계(Phase)'에 따라 TTP 노드들 사이에 PRECEDES 관계를 생성합니다."""
        phase1_tactics = ["reconnaissance", "resource-development", "initial-access"]
        phase2_tactics = [
            "execution", "persistence", "privilege-escalation", "defense-evasion",
            "credential-access", "discovery", "lateral-movement", "collection"
        ]
        phase3_tactics = ["command-and-control", "exfiltration", "impact"]

        cypher_phase1_to_2 = """
        MATCH (p1_ttp:TTP)-[:BELONGS_TO]->(tac1:Tactic) WHERE tac1.name IN $phase1
        MATCH (p2_ttp:TTP)-[:BELONGS_TO]->(tac2:Tactic) WHERE tac2.name IN $phase2
        MERGE (p1_ttp)-[r:PRECEDES]->(p2_ttp)
        SET r.weight = 0.5, r.phase_transition = '1->2'
        """
        
        cypher_phase2_to_3 = """
        MATCH (p2_ttp:TTP)-[:BELONGS_TO]->(tac2:Tactic) WHERE tac2.name IN $phase2
        MATCH (p3_ttp:TTP)-[:BELONGS_TO]->(tac3:Tactic) WHERE tac3.name IN $phase3
        MERGE (p2_ttp)-[r:PRECEDES]->(p3_ttp)
        SET r.weight = 0.5, r.phase_transition = '2->3'
        """
        
        cypher_within_phase2 = """
        MATCH (p2_ttp_a:TTP)-[:BELONGS_TO]->(tac2a:Tactic), (p2_ttp_b:TTP)-[:BELONGS_TO]->(tac2b:Tactic)
        WHERE tac2a.name IN $phase2 AND tac2b.name IN $phase2 AND id(p2_ttp_a) <> id(p2_ttp_b)
        MERGE (p2_ttp_a)-[r:PRECEDES]->(p2_ttp_b)
        SET r.weight = 1.0, r.phase_transition = '2->2'
        """

        with self.driver.session(database=self.database) as session:
            print("  - Creating sequence relationships from Phase 1 to Phase 2...")
            session.run(cypher_phase1_to_2, phase1=phase1_tactics, phase2=phase2_tactics)
            
            print("  - Creating sequence relationships from Phase 2 to Phase 3...")
            session.run(cypher_phase2_to_3, phase2=phase2_tactics, phase3=phase3_tactics)

            print("  - Creating sequence relationships within Phase 2...")
            session.run(cypher_within_phase2, phase2=phase2_tactics)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    
    loader = DataLoader()
    attack_data = {}
    cve_data = []

    print("--- 1. Data Loading ---")
    try:
        attack_file = DATA_DIR / "enterprise-attack.json"
        attack_data = loader.load_attack_stix(attack_file)
        print(f"Loaded {len(attack_data['attack_patterns'])} attack patterns, {len(attack_data['intrusion_sets'])} intrusion sets, {len(attack_data['malware'])} malware objects")
    except Exception as e:
        print(f"ERROR: Failed to load ATT&CK data: {e}")
    
    try:
        nvd_files = [
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2024.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2023.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2022.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2021.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2020.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2019.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2018.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2017.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2016.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2015.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2014.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2013.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2012.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2011.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2010.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2009.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2008.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2007.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2006.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2005.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2004.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2003.json",
            "D:\\Github\\TK_Graph\\data\\nvd-cve-feed-2002.json"
        ]

        for nvd_file in nvd_files:
            items = loader.load_nvd_cve(nvd_file)
            cve_data.extend(items)
        print(f"Total CVE items aggregated: {len(cve_data)}")
    except Exception as e:
        print(f"ERROR: Failed to load NVD CVE data: {e}")
    
    print("\n--- 2. Graph Building ---")
    try:
        builder = ThreatGraphBuilder(uri="bolt://localhost:7687", user="neo4j", password="password")
        print("Successfully connected to Neo4j.")

        # 데이터베이스 초기화 (선택 사항, 새로 구축 시 권장)
        # print("Cleaning existing database...")
        # with builder.driver.session(database="neo4j") as session:
        #     session.run("MATCH (n) DETACH DELETE n")
        
        # if attack_data:
        #     print("\n[Step 1/5] Creating TTP nodes...")
        #     builder.create_ttp_nodes(attack_data)
        #     print("  -> TTP node creation process finished.")

        # if cve_data:
        #     print("\n[Step 2/5] Creating CVE and CWE nodes...")
        #     builder.create_cve_cwe_nodes(cve_data)
        #     print("  -> CVE/CWE node creation process finished.")

        # if attack_data:
        #     print("\n[Step 3/5] Creating Actor-TTP relationships...")
        #     builder.create_actor_ttp_relationships(attack_data, attack_file)
        #     print("  -> Actor-TTP relationship creation process finished.")

        # if attack_data:
        #     print("\n[Step 4/5] Creating Tactic nodes and relationships...")
        #     builder.create_tactic_nodes_and_relationships(attack_data)
        #     print("  -> Tactic node and relationship creation process finished.")

        print("\n[Step 5/5] Creating TTP sequence relationships...")
        builder.create_ttp_sequence_relationships()
        print("  -> TTP sequence relationship creation process finished.")

        builder.close()
        print("\nGraph building process complete.")
        
    except Exception as e:
        print(f"\nFATAL ERROR during graph building: {e}")