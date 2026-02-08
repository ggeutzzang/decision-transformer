# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소의 코드를 작업할 때 참조하는 가이드입니다.

## 📋 문서 작성 가이드라인

이 프로젝트의 문서는 **시각화**를 최우선으로 합니다. 설명을 추가할 때는 다음 순서를 따르세요:

1. **먼저 다이어그램으로**: `sequenceDiagram`, `flowchart`, `graph` 등으로 시각화
2. **핵심만 요약**: 텍스트는 다이어그램의 보조 역할로만 사용
3. **상세 내용은 `doc/`로**: 긴 설명은 `doc/`의 적절한 문서에 위임

### 권장 다이어그램 유형

| 용도 | 추천 다이어그램 | 예시 |
|------|----------------|------|
| **시간 순서 흐름** | `sequenceDiagram` | 추론 과정, 학습 루프, 함수 호출 순서 |
| **데이터 처리 파이프라인** | `flowchart TD` | 전처리, 모델 forward, 평가 과정 |
| **아키텍처 구조** | `flowchart TB` + `subgraph` | 모델 구조, 모듈 관계 |
| **상태 전환** | `stateDiagram-v2` | 에피소드 진행, RTG 업데이트 |
| **개념 비교** | `flowchart LR` (분기형) | DT vs BC, Atari vs Gym |

---

## Quick Overview

Decision Transformer는 강화학습을 **시퀀스 모델링 문제**로 재구성한 연구입니다.

```mermaid
flowchart LR
    subgraph Traditional["🔄 기존 RL"]
        Bellman["Bellman 방정식"]
        Value["Value Function"]
    end

    subgraph DT["🤖 Decision Transformer"]
        Seq["Sequence Modeling"]
        GPT["GPT Architecture"]
    end

    Traditional -.->|"패러다임 전환"| DT

    style DT fill:#e3f2fd
```

**📖 상세 설명**: [`doc/system-analysis.md`](./doc/system-analysis.md#1-개요)

---

## 📚 상세 문서 맵

```mermaid
mindmap
    root((doc/))
        Overview["README.md<br/>문서 가이드"]
        Learning["learning-plan.md<br/>Phase별 학습"]
        Architecture["architecture-flow.md<br/>아키텍처"]
        System["system-analysis.md<br/>시스템 분석"]
        Code["code-walkthrough.md<br/>코드 분석"]
```

| 문서 | 용도 | 링크 |
|------|------|------|
| **문서 가이드** | `doc/` 구조 및 학습 경로 | [`doc/README.md`](./doc/README.md) |
| **학습 계획** | Phase별 학습 로드맵 | [`doc/learning-plan.md`](./doc/learning-plan.md) |
| **아키텍처** | 전체 시스템 다이어그램 | [`doc/architecture-flow.md`](./doc/architecture-flow.md) |
| **시스템 분석** | Atari + Gym 비교 | [`doc/system-analysis.md`](./doc/system-analysis.md) |
| **코드 분석** | 구현 상세 설명 | [`doc/code-walkthrough.md`](./doc/code-walkthrough.md) |

---

## Project Structure

```mermaid
flowchart TB
    subgraph Root["decision-transformer/"]
        direction LR

        subgraph Atari["🎮 atari/"]
            A1["DQN-replay 데이터"]
            A2["minGPT 구현"]
            A3["이미지 입력"]
        end

        subgraph Gym["🤸 gym/"]
            G1["D4RL 데이터"]
            G2["HF GPT-2"]
            G3["연속 state"]
        end

        subgraph Doc["📚 doc/"]
            D1["README.md"]
            D2["learning-plan.md"]
            D3["architecture-flow.md"]
            D4["system-analysis.md"]
            D5["code-walkthrough.md"]
        end
    end

    style Atari fill:#fff3e0
    style Gym fill:#e8f5e9
    style Doc fill:#e3f2fd
```

---

## Quick Start

### Atari 환경

```mermaid
sequenceDiagram
    participant User as 👤 사용자
    participant Shell as 🖥️ 터미널
    participant Conda as Conda
    participant GCS as gsutil
    participant Script as run_dt_atari.py

    User->>Shell: cd atari
    User->>Conda: conda env create -f conda_env.yml
    Conda-->>User: 환경 생성 완료

    User->>Shell: mkdir dqn_replay
    User->>GCS: gsutil -m cp -R gs://atari-replay-datasets/dqn/Breakout dqn_replay
    GCS-->>User: 데이터 다운로드 완료

    User->>Script: python run_dt_atari.py --game Breakout
    Script-->>User: 학습 시작
```

### Gym 환경

```mermaid
sequenceDiagram
    participant User as 👤 사용자
    participant Shell as 🖥️ 터미널
    participant D4RL as D4RL
    participant Script as experiment.py

    User->>Shell: cd gym
    User->>Conda: conda env create -f conda_env.yml
    Conda-->>User: 환경 생성 완료

    User->>D4RL: python data/download_d4rl_datasets.py
    D4RL-->>User: 데이터셋 다운로드 완료

    User->>Script: python experiment.py --env hopper --dataset medium
    Script-->>User: 학습 시작
```

---

## Core Concepts

### Return-to-Go (RTG)

```mermaid
flowchart LR
    subgraph RTG["RTG 업데이트"]
        R1["RTG = 100"]
        R2["reward = 10"]
        R3["RTG = 100 - 10 = 90"]
    end

    R1 -->|"-"| R2 --> R3

    style R1 fill:#ffcdd2
    style R3 fill:#c8e6c9
```

**📖 상세**: [`doc/learning-plan.md`](./doc/learning-plan.md#22-return-to-go-개념-깊이-이해)

### Sequence Structure

```mermaid
flowchart LR
    subgraph Input["입력 시퀀스"]
        R0["R₀"] --> S0["s₀"] --> A0["a₀"]
        A0 --> R1["R₁"] --> S1["s₁"] --> A1["a₁"]
    end

    subgraph Output["예측"]
        P0["  "]
        P1["→a₀"]
        P2["  "]
        P3["  "]
        P4["→a₁"]
    end

    S0 -.-> P1
    S1 -.-> P4

    style P1 fill:#4ecdc4
    style P4 fill:#4ecdc4
```

**📖 상세**: [`doc/architecture-flow.md`](./doc/architecture-flow.md#6-모델별-시퀀스-구성-비교)

### Inference Flow

```mermaid
sequenceDiagram
    participant User as 🎯 목표
    participant Model as 🧠 DT
    participant Env as 🌍 환경

    User->>Model: target_return = 100
    Model->>Env: action₀
    Env-->>Model: reward = 10

    Note over Model: RTG = 100 - 10 = 90

    Model->>Env: action₁
    Env-->>Model: reward = 20

    Note over Model: RTG = 90 - 20 = 70

    Model->>Env: action₂
```

**📖 상세**: [`doc/architecture-flow.md`](./doc/architecture-flow.md#5-평가-추론-과정)

---

## Environment Comparison

| 항목 | Atari | Gym |
|------|-------|-----|
| **디렉토리** | `atari/` | `gym/` |
| **실행 위치** | `cd atari` | `cd gym` |
| **Context Length** | 30 | 20 |
| **State** | 이미지 (4×84×84) | 연속 벡터 |
| **Action** | 이산적 (분류) | 연속적 (회귀) |
| **모델** | minGPT (6L, 8H) | HF GPT-2 (3L, 1H) |
| **데이터셋** | DQN replay buffers | D4RL pickle |

**📖 상세**: [`doc/system-analysis.md`](./doc/system-analysis.md#84-atari-vs-gym-차이점)

---

## Common Issues

```mermaid
flowchart TD
    Start[문제 발생] --> Q1{PYTHONPATH?}
    Q1 -->|Yes| Q2{MuJoCo?}
    Q1 -->|No| A1["해당 디렉토리에서<br/>cd atari 또는 cd gym"]

    Q2 -->|문제| A2["pip install mujoco"]
    Q2 -->|OK| Q3{GPU 메모리?}

    Q3 -->|부족| A3["batch_size 줄이기"]
    Q3 -->|충분| Q4{D4RL 설치?}

    Q4 -->|문제| A4["pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl"]
    Q4 -->|OK| A5["이슈 트래커 확인"]

    style A1 fill:#c8e6c9
    style A2 fill:#c8e6c9
    style A3 fill:#c8e6c9
    style A4 fill:#c8e6c9
```

---

## References

- **논문**: [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
- **원본 코드**: [https://github.com/kzl/decision-transformer](https://github.com/kzl/decision-transformer)
- **상세 문서**: [`doc/`](./doc/) 디렉토리 참조
