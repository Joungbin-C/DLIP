import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

file_path = "accel_data/accel_data_20250611_80.csv"

try:
    df = pd.read_csv(file_path, header=None)

    df.columns = ['Timestamp(ms)', 'X', 'Y', 'Z']

    # 데이터 전처리
    timestamps = df['Timestamp(ms)'].values
    x = df['X'].values.astype(float)
    y = df['Y'].values.astype(float)
    z = df['Z'].values.astype(float)

    # 가속도 벡터 크기 계산
    a_mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # 샘플링 주파수 계산
    time_diff = np.mean(np.diff(timestamps))  # 평균 샘플링 간격(ms)
    sample_rate = 1000 / time_diff  # Hz 단위 변환

    # FFT 분석
    N = len(a_mag)
    T = 1.0 / sample_rate
    yf = fft(a_mag - np.mean(a_mag))  # DC 성분 제거
    xf = fftfreq(N, T)[:N // 2]
    magnitude = 2.0 / N * np.abs(yf[:N // 2])

    # 최대 진동 주파수 출력
    main_freq = xf[np.argmax(magnitude)]
    print(f"▶ 주요 진동 주파수: {main_freq:.2f} Hz")

except FileNotFoundError:
    print(f"오류: {file_path} 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"오류 발생: {str(e)}")
