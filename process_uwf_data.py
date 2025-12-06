import pandas as pd
import json
import glob
import os
from tqdm import tqdm

def process_uwf_smartly(
    input_folder: str,
    output_file: str,
    time_threshold_minutes: int = 30,  # 세션 분리 기준 (분)
    window_size: int = 20,             # LSTM 입력 길이 (슬라이딩 윈도우)
    stride: int = 5                    # 윈도우 이동 간격
):
    print(f"--- 1. 파일 탐색: {input_folder} ---")
    parquet_files = glob.glob(os.path.join(input_folder, "**", "*.parquet"), recursive=True)
    if not parquet_files:
        print("Error: Parquet 파일이 없습니다.")
        return

    # 1. 컬럼 자동 감지 (이전과 동일)
    sample_df = pd.read_parquet(parquet_files[0])
    cols = sample_df.columns.tolist()
    
    # 컬럼 매핑
    src_ip_col = next((c for c in ['src_ip_zeek', 'id.orig_h', 'src_ip', 'uid'] if c in cols), None)
    ttp_col = next((c for c in ['label_technique', 'mitre_attck_technique'] if c in cols), None)
    ts_col = next((c for c in ['ts', 'timestamp', 'datetime'] if c in cols), None)

    if not (src_ip_col and ttp_col and ts_col):
        print(f"Fatal Error: 필수 컬럼(IP, TTP, Time)을 찾지 못했습니다. 감지된 컬럼: {cols}")
        return
    
    print(f"-> 매핑 완료: IP[{src_ip_col}], TTP[{ttp_col}], Time[{ts_col}]")

    # 2. 데이터 로드
    print("--- 2. 데이터 로드 및 병합 ---")
    dfs = []
    for file in tqdm(parquet_files):
        try:
            df_chunk = pd.read_parquet(file, columns=[src_ip_col, ttp_col, ts_col])
            dfs.append(df_chunk)
        except: pass
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # 전처리: TTP 결측 제거 및 포맷팅
    full_df = full_df.dropna(subset=[ttp_col])
    full_df[ttp_col] = full_df[ttp_col].astype(str).str.strip()
    full_df = full_df[full_df[ttp_col].str.startswith('T')] # TTP 포맷 필터링

    # 시간 변환 및 정렬
    full_df[ts_col] = pd.to_datetime(full_df[ts_col], unit='s', errors='coerce').fillna(pd.to_datetime(full_df[ts_col], errors='coerce'))
    full_df = full_df.sort_values(by=[src_ip_col, ts_col])
    
    print(f"-> 전처리 완료 데이터: {len(full_df)} 행")

    # ---------------------------------------------------------
    # [핵심] 3. 시간 기반 세션 분리 (Time-based Sessionization)
    # ---------------------------------------------------------
    print(f"--- 3. 세션 분리 (기준: {time_threshold_minutes}분) ---")
    
    # IP별로 이전 로그와의 시간 차이 계산
    # diff()는 바로 앞 행과의 차이를 구함
    full_df['time_diff'] = full_df.groupby(src_ip_col)[ts_col].diff()
    
    # 시간 차이가 임계값(30분)보다 크면 '새로운 세션(True)'으로 마킹
    # pd.Timedelta로 분 단위 변환
    threshold = pd.Timedelta(minutes=time_threshold_minutes)
    full_df['is_new_session'] = full_df['time_diff'] > threshold
    full_df['is_new_session'] = full_df['is_new_session'].fillna(False) # 첫 행 처리

    # 누적 합(cumsum)을 이용해 세션 ID 생성
    # 예: [False, False, True, False] -> [0, 0, 1, 1]
    # IP별로 독립적인 세션 ID를 만들기 위해 IP + Session_Num 조합 사용
    full_df['session_id_local'] = full_df.groupby(src_ip_col)['is_new_session'].cumsum()
    
    # 최종 그룹화: [IP, Session_ID]
    grouped = full_df.groupby([src_ip_col, 'session_id_local'])[ttp_col].apply(list)
    
    print(f"-> 분리된 총 세션(공격) 수: {len(grouped)}")

    # ---------------------------------------------------------
    # [핵심] 4. 슬라이딩 윈도우 & 중복 제거 (Sliding Window)
    # ---------------------------------------------------------
    print(f"--- 4. 슬라이딩 윈도우 적용 (Size: {window_size}, Stride: {stride}) ---")
    
    final_sequences = []
    
    import itertools

    for _, ttp_list in tqdm(grouped.items(), desc="시퀀스 가공"):
        # 1) 연속 중복 제거 (Dedup)
        # [A, A, B, B, B, A] -> [A, B, A]
        clean_list = [k for k, g in itertools.groupby(ttp_list)]
        
        # 길이가 너무 짧으면 패스
        if len(clean_list) < 2:
            continue
            
        # 2) 슬라이딩 윈도우 (Sliding Window)
        # 긴 시퀀스를 모델 입력 크기에 맞춰 여러 개로 쪼갬
        if len(clean_list) <= window_size:
            # 윈도우보다 짧으면 그냥 통째로 추가
            final_sequences.append(clean_list)
        else:
            # 윈도우보다 길면 잘라서 추가
            # 예: 길이 100 -> 0~20, 5~25, 10~30 ...
            for i in range(0, len(clean_list) - window_size + 1, stride):
                window = clean_list[i : i + window_size]
                final_sequences.append(window)

    print(f"-> 최종 생성된 시퀀스 개수: {len(final_sequences)}")
    if final_sequences:
        print(f"-> 예시: {final_sequences[0]}")

    # 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_sequences, f, indent=2)
    print(f"--- 완료! 저장됨: {output_file} ---")

if __name__ == "__main__":
    # 설정값
    INPUT_DIR = "data/uwf_zeekdata24"
    OUTPUT_FILE = "uwf_smart_sequences.json"
    
    # 30분 이상 쉬면 다른 공격으로 간주, 윈도우 크기는 10~20 추천
    process_uwf_smartly(
        INPUT_DIR, 
        OUTPUT_FILE, 
        time_threshold_minutes=30, 
        window_size=20, 
        stride=5
    )