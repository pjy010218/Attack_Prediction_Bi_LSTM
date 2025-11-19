import pandas as pd
import json
import glob
import os
from tqdm import tqdm

def process_uwf_zeekdata_parquet(
    input_folder: str,
    output_file: str,
    min_sequence_length: int = 2
):
    """
    UWF-ZeekData24의 Parquet 파일들을 병합하고 TTP 시퀀스를 추출합니다.
    
    Args:
        input_folder: Parquet 파일들이 들어있는 최상위 폴더 경로
        output_file: 저장할 JSON 파일 경로 (예: 'uwf_real_sequences.json')
        min_sequence_length: 시퀀스로 인정할 최소 TTP 개수
    """
    print(f"--- 1. 파일 탐색 시작: {input_folder} ---")
    # 하위 폴더까지 포함하여 모든 .parquet 파일 검색
    parquet_files = glob.glob(os.path.join(input_folder, "**", "*.parquet"), recursive=True)
    
    if not parquet_files:
        print("Error: 해당 경로에서 Parquet 파일을 찾을 수 없습니다.")
        return

    print(f"-> 총 {len(parquet_files)}개의 Parquet 파일을 찾았습니다.")

    # 2. 데이터 로드 및 병합 (메모리 최적화)
    print("--- 2. 데이터 로드 및 병합 중... ---")
    
    # UWF-ZeekData의 mission_logs에서 필요한 핵심 컬럼 (데이터셋 버전에 따라 이름 확인 필요)
    # 보통: 'mission_id' (공격 식별), 'mitre_attck_technique' (TTP), 'timestamp' (순서)
    # 실제 컬럼명이 다를 경우 수정해주세요. (예: 'id', 'ts', 'technique_id' 등)
    target_columns = [
        'mission_id',           # 공격 시나리오 ID (Group key)
        'mitre_attck_technique', # TTP ID
        'timestamp'             # 정렬 기준 시간
    ]
    
    dfs = []
    for file in tqdm(parquet_files, desc="파일 읽기"):
        try:
            # 필요한 컬럼만 읽어서 메모리 절약
            df_chunk = pd.read_parquet(file, columns=target_columns)
            dfs.append(df_chunk)
        except Exception as e:
            # 컬럼이 없는 파일(예: conn.log 등 mission_logs가 아닌 파일)은 패스
            pass
            
    if not dfs:
        print("Error: 처리할 데이터프레임이 없습니다. 컬럼명을 확인해주세요.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"-> 전체 데이터 병합 완료: {len(full_df)} 행")

    # 3. 데이터 전처리
    print("--- 3. 데이터 전처리 (결측치 제거 및 정렬) ---")
    
    # TTP가 없는 행 제거
    full_df = full_df.dropna(subset=['mitre_attck_technique'])
    
    # TTP ID 포맷 클렌징 (공백 제거 및 'T'로 시작하는지 확인)
    full_df['mitre_attck_technique'] = full_df['mitre_attck_technique'].astype(str).str.strip()
    full_df = full_df[full_df['mitre_attck_technique'].str.startswith('T')]
    
    # 시간 순서 정렬
    if 'timestamp' in full_df.columns:
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
        full_df = full_df.sort_values(by=['mission_id', 'timestamp'])
    else:
        print("Warning: timestamp 컬럼이 없어 정렬하지 못했습니다. 데이터 순서를 신뢰할 수 없습니다.")

    # 4. 시퀀스 추출 (Grouping)
    print("--- 4. TTP 시퀀스 생성 중... ---")
    sequences = []
    
    # Mission ID별로 그룹화하여 TTP 리스트 생성
    grouped = full_df.groupby('mission_id')['mitre_attck_technique'].apply(list)
    
    for mission_id, ttp_list in tqdm(grouped.items(), desc="시퀀스 변환"):
        # 중복된 연속 TTP 제거 (선택 사항: T1059 -> T1059 -> T1059 인 경우 하나로 칠지 여부)
        # 여기서는 그대로 둡니다. 필요하면 아래 주석 해제
        # ttp_list = [k for k, g in itertools.groupby(ttp_list)]
        
        if len(ttp_list) >= min_sequence_length:
            sequences.append(ttp_list)

    print(f"-> 추출된 유효 시퀀스 수: {len(sequences)}")
    if sequences:
        print(f"-> 예시 시퀀스: {sequences[0]}")

    # 5. 저장
    print(f"--- 5. 파일 저장: {output_file} ---")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sequences, f, indent=2)
    print("완료!")

# --- 실행부 ---
if __name__ == "__main__":
    # UWF-ZeekData24의 Parquet 파일이 있는 폴더 경로
    INPUT_DIR = "./data/uwf_zeekdata24/mission_logs" 
    OUTPUT_FILE = "uwf_real_sequences.json"
    
    process_uwf_zeekdata_parquet(INPUT_DIR, OUTPUT_FILE)
