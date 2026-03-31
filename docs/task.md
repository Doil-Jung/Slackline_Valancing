# Slackline Balancing Simulator Improvement Plan

- [x] Analyze existing codebase (V1, V2, V3)
- [x] Identify physics model issues (하체 법선 고정 문제)
- [x] Propose 3-DOF model to user → approved
- [x] Implement 3-DOF model (V3)
- [x] Controllability analysis for LQR (rank=6, controllable!)
- [x] V4 LQR controller implementation
  - [x] Backup V3 to v3_three_dof/
  - [x] Create LQR controller (lqr.js)
  - [x] Update main.js, index.html for V4
  - [x] Fix V3 navigation (add V4 link)
  - [x] Browser testing ✓
