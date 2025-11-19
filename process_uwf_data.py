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
    *수정사항: 제공된 26개 컬럼명을 반영 (label_technique, ts, src_ip_zeek 사용)
    """
    print(f"--- 1. 파일 탐색 시작: {input_folder} ---")
    parquet_files = glob.glob(os.path.join(input_folder, "**", "*.parquet"), recursive=True)
    
    if not parquet_files:
        print("Error: 해당 경로에서 Parquet 파일을 찾을 수 없습니다.")
        return

    print(f"-> 총 {len(parquet_files)}개의 Parquet 파일을 찾았습니다.")

    # 2. 데이터 로드 및 병합
    print("--- 2. 데이터 로드 및 병합 중... ---")
    
    # [수정] 제공해주신 컬럼명 반영
    # mission_id가 없으므로, 공격자 식별을 위해 'src_ip_zeek'를 사용합니다.
    target_columns = [
        'src_ip_zeek',      # 그룹화 기준 (공격자 IP)
        'label_technique',  # TTP ID (핵심 데이터)
        'ts'                # 정렬 기준 (타임스탬프)
    ]
    
    dfs = []
    for file in tqdm(parquet_files, desc="파일 읽기"):
        try:
            # 필요한 컬럼만 읽어서 메모리 절약
            df_chunk = pd.read_parquet(file, columns=target_columns)
            dfs.append(df_chunk)
        except Exception as e:
            # 해당 컬럼이 없는 파일은 건너뜀
            pass
            
    if not dfs:
        print("Error: 처리할 데이터가 없습니다. 파일 경로와 컬럼명을 다시 확인해주세요.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"-> 전체 데이터 병합 완료: {len(full_df)} 행")

    # 3. 데이터 전처리
    print("--- 3. 데이터 전처리 (결측치 제거 및 정렬) ---")
    
    # [수정] label_technique 컬럼 사용
    # TTP 레이블이 없는 행(일반 트래픽 등) 제거
    full_df = full_df.dropna(subset=['label_technique'])
    
    # TTP ID 포맷 클렌징 (문자열 변환, 공백 제거)
    full_df['label_technique'] = full_df['label_technique'].astype(str).str.strip()
    
    # 'T'로 시작하는 유효한 TTP ID만 남김 (예: 'T1059') - 데이터에 따라 조절 가능
    full_df = full_df[full_df['label_technique'].str.startswith('T')]
    
    # [수정] ts 컬럼 기준 정렬
    # ts가 float(Unix timestamp)일 수도 있고 datetime일 수도 있으므로 변환 시도
    if 'ts' in full_df.columns:
        full_df['ts'] = pd.to_datetime(full_df['ts'], unit='s', errors='coerce').fillna(pd.to_datetime(full_df['ts'], errors='coerce'))
        # 소스 IP별로, 시간 순서대로 정렬
        full_df = full_df.sort_values(by=['src_ip_zeek', 'ts'])
    else:
        print("Warning: 'ts' 컬럼 처리 중 문제가 발생했습니다. 정렬이 부정확할 수 있습니다.")

    # 4. 시퀀스 추출 (Grouping)
    print("--- 4. TTP 시퀀스 생성 중... ---")
    sequences = []
    
    # [수정] src_ip_zeek(공격자 IP) 기준으로 그룹화하여 TTP 리스트 생성
    # -> "동일 IP에서 발생한 일련의 공격 행위"를 하나의 시퀀스로 간주
    grouped = full_df.groupby('src_ip_zeek')['label_technique'].apply(list)
    
    for src_ip, ttp_list in tqdm(grouped.items(), desc="시퀀스 변환"):
        # (선택사항) 연속된 중복 TTP 제거가 필요하면 아래 주석 해제
        # ttp_list = [k for k, g in itertools.groupby(ttp_list)]
        
        if len(ttp_list) >= min_sequence_length:
            sequences.append(ttp_list)

    print(f"-> 추출된 유효 시퀀스 수: {len(sequences)}")
    if sequences:
        print(f"-> 예시 시퀀스 (첫 번째 IP의 공격 흐름): {sequences[0]}")

    # 5. 저장
    print(f"--- 5. 파일 저장: {output_file} ---")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sequences, f, indent=2)
    print("완료!")

# --- 실행부 ---
if __name__ == "__main__":
    # UWF-ZeekData24의 Parquet 파일이 있는 폴더 경로를 지정하세요.
    INPUT_DIR = "./data/uwf_zeekdata24" 
    OUTPUT_FILE = "uwf_real_sequences.json"
    
    process_uwf_zeekdata_parquet(INPUT_DIR, OUTPUT_FILE)
