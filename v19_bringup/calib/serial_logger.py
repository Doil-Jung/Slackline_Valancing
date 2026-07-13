# -*- coding: utf-8 -*-
"""
serial_logger.py — free_swing_log 등 시리얼 CSV 로거 (PC측)
============================================================
사용:  python serial_logger.py COM3 [--baud 115200] [--out swing.csv] [--prefix L]
       (macOS/리눅스는 COM3 대신 /dev/ttyACM0 등)
'프리픽스,데이터...' 행만 골라 CSV로 저장한다 (free_swing_log 는 "L,t_ms,phi_deg").
Ctrl+C 로 종료. 필요 패키지: pip install pyserial
"""
import argparse, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", default="log.csv")
    ap.add_argument("--prefix", default="L")
    a = ap.parse_args()

    try:
        import serial
    except ImportError:
        sys.exit("pyserial 필요:  pip install pyserial")

    ser = serial.Serial(a.port, a.baud, timeout=1)
    print(f"[logger] {a.port} @ {a.baud} → {a.out}  (프리픽스 '{a.prefix},' 행만 저장, Ctrl+C 종료)")
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("t_ms,phi_deg\n")
        try:
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue
                if line.startswith(a.prefix + ","):
                    f.write(line[len(a.prefix) + 1:] + "\n")
                    n += 1
                    if n % 500 == 0:
                        print(f"  {n} samples...")
                else:
                    print("  [dev]", line)   # 장치의 안내 메시지 표시
        except KeyboardInterrupt:
            pass
    print(f"[logger] 저장 완료: {a.out} ({n} samples)")

if __name__ == "__main__":
    main()
