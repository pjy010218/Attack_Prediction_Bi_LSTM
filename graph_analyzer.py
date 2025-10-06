from typing import Dict, Any, Optional, List
from neo4j import GraphDatabase

class ThreatGraphAnalyzer:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.graph_name = "threat_intelligence_graph"

    def close(self):
        self.driver.close()

    def check_gds_available(self) -> bool:
        cypher = "RETURN gds.version() IS NOT NULL AS available"
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher).single()
                return result and result['available']
        except Exception as e:
            print(f"GDS library not available: {e}")
            return False

    def project_graph(self) -> bool:
        try:
            with self.driver.session(database=self.database) as session:
                session.run(f"CALL gds.graph.drop('{self.graph_name}', false)")
                print(f"Dropped existing projected graph: {self.graph_name}")
        except Exception:
            pass

        node_projection = {
            'CVE': {}, 'TTP': {}, 'CWE': {}, 'Actor': {}
        }

        relationship_projection = {
            'IS_EXPLOITED_BY': {'orientation': 'UNDIRECTED', 'properties': 'weight'},
            'USES': {'orientation': 'UNDIRECTED', 'properties': 'weight'},
            'IS_INSTANCE_OF': {'orientation': 'UNDIRECTED', 'properties': 'weight'}
        }
        cypher = "CALL gds.graph.project($graph_name, $node_spec, $rel_spec)"
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, graph_name=self.graph_name, node_spec=node_projection, rel_spec=relationship_projection).single()
                if result:
                    print(f"Successfully projected undirected weighted graph: {self.graph_name} with {result['nodeCount']} nodes and {result['relationshipCount']} relationships.")
                    return True
                # GDS 2.1 이전 버전과의 호환성을 위해 레거시 반환값도 확인
                elif session.run("CALL gds.graph.exists($graph_name) YIELD exists RETURN exists", graph_name=self.graph_name).single()['exists']:
                    print(f"Successfully projected undirected weighted graph: {self.graph_name}")
                    return True
                else:
                    print("Failed to project graph")
                    return False
        except Exception as e:
            print(f"Error projecting graph: {e}")
            return False

    def calculate_and_write_centrality(self, batch_size: int = 5000) -> bool:
        """
        GDS를 사용해 TTP 노드의 중개 중심성을 계산하고,
        그 결과를 다시 TTP 노드의 속성으로 정확하게 기록합니다.
        """
        
        # 1. GDS로 중심성 계산 및 TTP ID와 함께 결과 스트리밍
        stream_cypher = f"""
        CALL gds.betweenness.stream('{self.graph_name}')
        YIELD nodeId, score
        // [수정] GDS의 내부 ID(nodeId)를 실제 TTP 노드의 ID(tid)로 변환
        RETURN gds.util.asNode(nodeId).tid AS ttp_tid, score AS centrality
        """

        print("Calculating TTP betweenness centrality and streaming results...")
        try:
            with self.driver.session(database=self.database) as session:
                results = [record.data() for record in session.run(stream_cypher)]
            print(f" -> Calculation complete. Fetched {len(results)} TTP centrality scores.")
        except Exception as e:
            print(f"Error calculating or streaming centrality: {e}")
            return False

        if not results:
            print(" -> No centrality scores were calculated. Check GDS graph projection.")
            return True 
        
        # 2. 계산된 결과를 TTP 노드에 다시 기록
        write_cypher = """
        UNWIND $updates AS update
        // [수정] CVE가 아닌 TTP 노드를 찾아서 속성 업데이트
        MATCH (t:TTP {tid: update.ttp_tid})
        SET t.betweenness_centrality = update.centrality
        """
        print("Writing centrality scores to TTP nodes in batches...")
        try:
            with self.driver.session(database=self.database) as session:
                total_rows = len(results)
                for i in range(0, total_rows, batch_size):
                    batch = results[i:i + batch_size]
                    session.run(write_cypher, updates=batch)
                    print(f"  - Wrote batch {i // batch_size + 1} / {(total_rows + batch_size - 1) // batch_size} ({len(batch)} items)")
            print(f" -> Successfully updated {total_rows} TTP nodes.")
            return True
        except Exception as e:
            print(f"Error writing centrality scores to DB: {e}")
            return False

    def get_top_cve_by_centrality(self, limit: int = 10) -> list:
        # ... (기존 코드와 동일)
        cypher = """
        MATCH (c:CVE) WHERE c.betweenness_centrality IS NOT NULL
        RETURN c.id as cve_id, c.description as description,
               c.cvss_score as cvss_score, c.betweenness_centrality as centrality
        ORDER BY c.betweenness_centrality DESC LIMIT $limit
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, limit=limit)
                return [record.data() for record in result]
        except Exception as e:
            print(f"Error retrieving top CVEs: {e}")
            return []

    def analyze_threat_graph(self) -> bool:
        print("Starting threat graph analysis...")
        if not self.check_gds_available(): return False
        if not self.project_graph(): return False
        
        # --- Fixed part: 새로운 단일 함수 호출 ---
        if not self.calculate_and_write_centrality(): return False
        
        top_cves = self.get_top_cve_by_centrality(5)
        if top_cves:
            print("\n--- Top 5 CVEs by Betweenness Centrality (Weighted) ---")
            for i, cve in enumerate(top_cves, 1):
                print(f"{i}. {cve['cve_id']} - Score: {cve['centrality']:.4f}")
                print(f"   CVSS: {cve['cvss_score']}")
                print(f"   Description: {(cve['description'] or '')[:100]}...")
                print()
        return True

def main():
    # ... (기존 코드와 동일)
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"
    
    analyzer = None
    try:
        analyzer = ThreatGraphAnalyzer(uri, user, password)
        success = analyzer.analyze_threat_graph()
        if success: print("\nAnalysis completed successfully!")
        else: print("\nAnalysis failed!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if analyzer: analyzer.close()

if __name__ == "__main__":
    main()