# -*- coding: utf-8 -*-
"""
server.py — v19 전물리 시뮬레이터 로컬 서버
============================================
실행:  cd v19_sim && python server.py     (필요: pip install mujoco aiohttp scipy numpy)
접속:  http://localhost:8199

물리·제어는 전부 파이썬(MuJoCo)에서 돌고, 브라우저는 v17식 UI(3D 뷰·슬라이더·
재생속도·1장 스텝·그래프)만 담당한다. WebSocket 으로 30Hz 프레임 전송.
"""
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v19_bringup"))
sys.path.insert(0, HERE)

import numpy as np
from aiohttp import web, WSMsgType

from sim_engine import SimEngine

PORT = 8199
TICK = 1.0 / 30.0          # 프레임 전송 주기
GRAPH_DECIM_FAST = 4       # 실속도 재생 시 그래프 데시메이션 (125Hz)

# 리빌드(모델 재컴파일·게인 재계산)가 필요한 현실화 키
REAL_REBUILD = {"f_phi", "c_phi_extra", "f_ankle", "b_ankle",
                "armature", "b_gear", "f_hip", "mj_dt"}


class SimServer:
    def __init__(self):
        self.eng = SimEngine()
        self.eng.kp_servo, self.eng.kd_servo = 25.0, 0.4   # 전물리 튜닝 기본값
        self.running = False
        self.speed = 1.0
        self.acc = 0.0             # 실시간 누적(제어스텝 단위)
        self.clients = set()
        self.pending = []          # 이번 틱에 쌓인 그래프 포인트

    # ---------- 상태 패킷 ----------
    def sample(self):
        e = self.eng
        f = e.frame()
        x, xh = f["x"], f["xh"]
        return [f["t"],
                round(np.rad2deg(x[0]), 4), round(np.rad2deg(x[1]), 4),
                round(np.rad2deg(x[2]), 4), round(np.rad2deg(x[2]-x[1]), 4),
                round(np.rad2deg(f["A"]), 4), round(np.rad2deg(f["Ae"]), 4),
                round(f["tau_act"], 4), f["cur"],
                round(np.rad2deg(xh[1]), 4), round(np.rad2deg(xh[2]), 4)]

    def view_msg(self):
        e = self.eng
        f = e.frame()
        return dict(type="frame", pts=self.pending, view=dict(
            t=f["t"], q=f["q"], fwe=f["fwe"], cyc=f["cyc"], fallen=f["fallen"],
            tau=f["tau_act"], tau_cmd=f["tau_cmd"], cur=f["cur"],
            x=[round(np.rad2deg(v), 3) for v in f["x"][:3]],
            xh=[round(np.rad2deg(v), 3) for v in f["xh"][:3]],
            A=round(np.rad2deg(f["A"]), 3), step=f["step"], running=self.running))

    def init_msg(self):
        e = self.eng
        bi = e.build_info()
        p = e.p
        return dict(type="init",
                    build=dict(w_eff=round(bi["w_eff"], 3), lam_u=round(bi["lam_u"], 3),
                               eps_star=round(bi["eps_star"], 3),
                               gamma_fold=round(bi["gamma_fold"], 2),
                               gamma_cycle=round(bi["gamma_cycle"], 2),
                               K=[round(k, 1) for k in bi["K"]],
                               I_r_calc=round(bi["I_r_calc"], 5), I_r_mj=round(bi["I_r_mj"], 5),
                               R=p["R"], DT=p["DT"]),
                    geom=dict(R=p["R"]*(1+e.mismatch.get("R", 0)/100.0),
                              L1=p["L1"]*(1+e.mismatch.get("L1", 0)/100.0),
                              L2=p["L2"]*(1+e.mismatch.get("L2", 0)/100.0)),
                    settings=dict(mode=e.ctrl_mode, estimator=e.estimator,
                                  servo_aware=e.servo_aware, gain=e.gain_scale,
                                  kp=e.kp_servo, kd=e.kd_servo, speed=self.speed,
                                  real=e.real, mismatch=e.mismatch,
                                  x0=e.x0,
                                  nominal=dict(R=p["R"], m1=p["m1"], m2=p["m2"],
                                               L1=p["L1"], L2=p["L2"],
                                               crank_m_each=p["crank_m_each"],
                                               bottom_m=p["bottom_m"],
                                               c_phi=p["c_phi"], B_THETA=p["B_THETA"],
                                               RHO=p["RHO"],
                                               A_TRIGGER_DEG=p["A_TRIGGER_DEG"],
                                               T_FOLD=p["T_FOLD"], T_WAIT=p["T_WAIT"],
                                               T_EXTEND=p["T_EXTEND"], T_REST=p["T_REST"],
                                               CUR_SAFE_UNIT=p["CUR_SAFE_UNIT"],
                                               R_COST=p["R_COST"])))

    async def broadcast(self, msg):
        dead = []
        data = json.dumps(msg)
        for ws in self.clients:
            try:
                await ws.send_str(data)
            except ConnectionError:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    # ---------- 시뮬 틱 ----------
    async def loop(self):
        while True:
            t0 = asyncio.get_event_loop().time()
            if self.running and self.clients:
                self.acc += self.speed * TICK / self.eng.p["DT"]
                n = int(self.acc)
                self.acc -= n
                n = min(n, 200)                     # 폭주 방지
                decim = GRAPH_DECIM_FAST if self.speed > 0.3 else 1
                for i in range(n):
                    self.eng.control_step()
                    if self.eng.step_count % decim == 0:
                        self.pending.append(self.sample())
                if self.eng.fallen and n > 0:
                    pass                            # 낙하해도 물리는 계속 (매달려 스윙)
            if self.clients and (self.pending or self.running):
                await self.broadcast(self.view_msg())
                self.pending = []
            dt = asyncio.get_event_loop().time() - t0
            await asyncio.sleep(max(0.001, TICK - dt))

    # ---------- 명령 처리 ----------
    async def handle(self, ws, m):
        e = self.eng
        cmd = m.get("cmd")
        if cmd == "hello":
            await ws.send_str(json.dumps(self.init_msg()))
            await ws.send_str(json.dumps(self.view_msg()))
        elif cmd == "run":
            self.running = True
        elif cmd == "pause":
            self.running = False
            await self.broadcast(self.view_msg())
        elif cmd == "reset":
            e.reset()
            self.pending = []
            await self.broadcast(dict(type="clear"))
            await self.broadcast(self.view_msg())
        elif cmd == "speed":
            self.speed = float(np.clip(m["v"], 0.01, 3.0))
        elif cmd == "step":
            self.running = False
            for _ in range(int(m.get("n", 1))):
                e.control_step()
                self.pending.append(self.sample())
            await self.broadcast(self.view_msg())
            self.pending = []
        elif cmd == "back":
            self.running = False
            for _ in range(int(m.get("n", 1))):
                if not e.step_back():
                    break
            await self.broadcast(dict(type="rewind", t=e.data.time))
            await self.broadcast(self.view_msg())
        elif cmd == "perturb":
            e.perturb(float(m["v"]))
        elif cmd == "mode":
            e.ctrl_mode = m["v"]
            e.fwe.update(phase="idle", t=0.0, armed=True, cycles=0)
        elif cmd == "estimator":
            e.estimator = m["v"]
        elif cmd == "gain":
            e.gain_scale = float(m["v"])
        elif cmd == "pd":
            e.kp_servo = float(m["kp"]); e.kd_servo = float(m["kd"])
        elif cmd == "x0":
            e.x0.update({k: float(m[k]) for k in ("phi0", "alpha0", "theta0") if k in m})
            e.reset()
            await self.broadcast(dict(type="clear"))
            await self.broadcast(self.view_msg())
        elif cmd == "real":
            k, v = m["k"], m["v"]
            e.real[k] = (bool(v) if k in ("servo_on", "sensor_on") else float(v))
            if k in REAL_REBUILD:
                await self.rebuild()
        elif cmd == "nominal":
            k, v = m["k"], m["v"]
            if k == "Q_DIAG":
                e.nominal[k] = [float(x) for x in v]
            else:
                e.nominal[k] = float(v)
            await self.rebuild()
        elif cmd == "mismatch":
            e.mismatch[m["k"]] = float(m["v"])
            await self.rebuild()
        elif cmd == "servo_aware":
            e.servo_aware = bool(m["v"])
            await self.rebuild()
        elif cmd == "fwe_preset":
            # ★원클릭 FWE 시연: 스윕 최적 시간표(params 기본) + 발목엔코더 + c_φ=0.008
            #   + γ(T)×실행갭 + 온라인 노브. 설정 후 바로 재생 시작.
            #   ★재현성: 이전에 만진 편차/현실화 슬라이더를 전부 초기화하고 시작한다
            #   (안 그러면 예: kt −10% 가 남아 생존 56s → 10s 로 떨어져도 원인을 모름)
            self.running = False
            self.pending = []
            from sim_engine import REAL_DEFAULT
            e.nominal = dict(c_phi=0.008, RHO=0.95)
            e.mismatch = dict(m1=0.0, m2=0.0, L1=0.0, L2=0.0, R=0.0, kt=0.0)
            e.real = dict(REAL_DEFAULT)
            e.gain_scale = 1.0
            e.real.update(enc_use_gyro=False, fwe_settle_s=0.0, fwe_adapt_T=False,
                          fwe_ff=True, fwe_fast_L=False, w_noload_dps=462.0,
                          # ★챔피언 설정 (2026-07-18): 위치모드(사다리꼴 420°/s·8000°/s²)
                          #   + 1-노브 RLS (Gc 고정, gc 만 학습 — 2노브는 공선성 표류로 사망)
                          #   20판 지터 분포: 중앙값 70.4s (전류모드 43.4s, 고정γ 32.6s)
                          fwe_pos_mode=True, pos_vel_dps=420.0, pos_acc_dps2=8000.0,
                          fwe_gamma_cal=22.7, fwe_Gc_cal=15.8,
                          fwe_online=True, fwe_online_1p=True, fwe_online_ff=0.95,
                          fwe_online_rej_deg=2.0)
            e.kp_servo, e.kd_servo = 80.0, 1.3
            e.estimator = "enc"
            e.ctrl_mode = "fwe3"
            # ★누를 때마다 새 추첨: 초기 기울기 ±0.02° 지터 (혼돈계라 수명이 매번 다름).
            #   '리셋' 버튼은 이 x0 를 재사용 → 같은 판 리플레이(결정론 재현).
            jit = float(np.random.default_rng().uniform(-0.02, 0.02))
            e.x0 = dict(phi0=0.0, alpha0=0.5 + jit, theta0=0.5 + jit)
            e.build()   # γ*·Gc* 는 위에서 위치모드 캘리브값(22.7/15.8)으로 직접 지정됨
            e.reset()
            await self.broadcast(dict(type="clear"))
            await self.broadcast(self.init_msg())
            await self.broadcast(self.view_msg())
            self.running = True

    async def rebuild(self):
        self.running = False
        self.pending = []
        try:
            self.eng.build()
            await self.broadcast(dict(type="clear"))
            await self.broadcast(self.init_msg())
            await self.broadcast(self.view_msg())
        except Exception as ex:                     # 파라미터 조합 불능(비양정치 등)
            await self.broadcast(dict(type="error", msg=str(ex)))


SRV = SimServer()


async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    SRV.clients.add(ws)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    await SRV.handle(ws, json.loads(msg.data))
                except Exception as ex:
                    await ws.send_str(json.dumps(dict(type="error", msg=str(ex))))
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        SRV.clients.discard(ws)
    return ws


async def index(request):
    return web.FileResponse(os.path.join(HERE, "static", "index.html"))


def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/three.min.js",
                       lambda r: web.FileResponse(os.path.join(HERE, "static", "three.min.js")))

    async def on_startup(app):
        asyncio.get_event_loop().create_task(SRV.loop())
    app.on_startup.append(on_startup)

    print("=" * 60)
    print("  v19 전물리 시뮬레이터")
    print(f"  브라우저에서 http://localhost:{PORT} 를 여세요")
    print("=" * 60)
    web.run_app(app, port=PORT, print=None)


if __name__ == "__main__":
    main()
