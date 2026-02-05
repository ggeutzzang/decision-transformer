# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Decision Transformer는 강화학습을 시퀀스 모델링 문제로 재구성한 연구 프로젝트입니다. GPT 아키텍처를 사용하여 (return-to-go, state, action) 시퀀스를 모델링하고, 원하는 return을 조건으로 하여 행동을 예측합니다.

```mermaid
flowchart LR
    subgraph Traditional["🔄 기존 RL"]
        direction TB
        T1["State s"] --> T2["Policy π(s)"]
        T2 --> T3["Action a"]
        T3 --> T4["Reward r"]
        T4 --> T5["Bellman Update"]
        T5 -.-> T2
    end

    subgraph DT["🤖 Decision Transformer"]
        direction TB
        D1["Target Return R̂"] --> D4
        D2["State s"] --> D4["Transformer"]
        D3["Past Actions"] --> D4
        D4 --> D5["Action a"]
    end

    Traditional -.->|"패러다임 전환"| DT

    style Traditional fill:#ffebee
    style DT fill:#e3f2fd
```

이 저장소는 두 개의 독립적인 실험 환경을 포함합니다:

```mermaid
flowchart TB
    subgraph Root["📁 decision-transformer"]
        direction LR
        subgraph Atari["🎮 atari/"]
            A1["DQN-replay 데이터셋"]
            A2["minGPT 기반 구현"]
            A3["이미지 입력 (84×84×4)"]
        end

        subgraph Gym["🤸 gym/"]
            G1["D4RL 데이터셋"]
            G2["HuggingFace GPT-2 기반"]
            G3["연속 상태 벡터 입력"]
        end
    end

    style Atari fill:#fff3e0
    style Gym fill:#e8f5e9
```

## Development Commands

### Atari 환경

**환경 설정:**
```bash
cd atari
conda env create -f conda_env.yml
conda activate decision-transformer-atari
```

**데이터셋 다운로드:**
```bash
mkdir dqn_replay
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] dqn_replay
# 예: gsutil -m cp -R gs://atari-replay-datasets/dqn/Breakout dqn_replay
```

**단일 실험 실행:**
```bash
cd atari
python run_dt_atari.py --seed 123 --context_length 30 --epochs 5 \
  --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 \
  --game 'Breakout' --batch_size 128 --data_dir_prefix ./dqn_replay
```

**재현 스크립트 실행:**
```bash
cd atari
bash run.sh  # 여러 게임과 seed에 대한 전체 실험 실행
```

**모델 타입:**
- `reward_conditioned`: Decision Transformer (DT)
- `naive`: Behavior Cloning (BC) 베이스라인

### OpenAI Gym 환경

**환경 설정:**
```bash
cd gym
conda env create -f conda_env.yml
conda activate decision-transformer-gym
```

**데이터셋 다운로드:**
```bash
cd gym
# D4RL 설치 필요: https://github.com/rail-berkeley/d4rl
python data/download_d4rl_datasets.py
```

**실험 실행:**
```bash
cd gym
python experiment.py --env hopper --dataset medium --model_type dt

# Weights & Biases 로깅 활성화
python experiment.py --env hopper --dataset medium --model_type dt -w True
```

**지원되는 환경:**
- `hopper`: Hopper-v3
- `halfcheetah`: HalfCheetah-v3
- `walker2d`: Walker2d-v3
- `reacher2d`: 커스텀 Reacher2D 환경

**지원되는 데이터셋:**
- `medium`, `medium-replay`, `medium-expert`, `expert` (D4RL 데이터셋 종류)

**모델 타입:**
- `dt`: Decision Transformer
- `bc`: Behavior Cloning

## Architecture Overview

### Core Sequence Modeling Approach

Decision Transformer는 기존 RL의 벨만 방정식 대신 autoregressive sequence modeling을 사용합니다:

```mermaid
flowchart LR
    subgraph Input["📥 입력 시퀀스"]
        R1["R̂₁"] --> S1["s₁"] --> A1["a₁"]
        R2["R̂₂"] --> S2["s₂"] --> A2["a₂"]
        R3["R̂₃"] --> S3["s₃"] --> A3["?"]
    end

    subgraph Process["🧠 처리"]
        Input --> TF["GPT-2<br/>Transformer"]
        TF --> CM["Causal Masking<br/>(미래 토큰 차단)"]
    end

    subgraph Output["📤 출력"]
        CM --> Pred["State 위치에서<br/>Action 예측"]
        Pred --> A3_pred["a₃ 예측"]
    end

    style R1 fill:#ffcdd2
    style R2 fill:#ffcdd2
    style R3 fill:#ffcdd2
    style S1 fill:#c8e6c9
    style S2 fill:#c8e6c9
    style S3 fill:#c8e6c9
    style A1 fill:#bbdefb
    style A2 fill:#bbdefb
    style A3 fill:#fff9c4
```

**핵심 개념:**
- 입력: `(R_1, s_1, a_1, R_2, s_2, a_2, ...)` 형태의 시퀀스
- R은 returns-to-go (미래 누적 보상)
- GPT-2 기반 transformer가 state에서 action을 예측
- 조건부 생성: 원하는 return을 지정하여 행동 정책을 유도

### Key Components

```mermaid
flowchart TB
    subgraph Atari["🎮 Atari 구현 (atari/)"]
        direction TB
        AM["mingpt/"]
        AM --> AM1["model_atari.py<br/>GPT 모델"]
        AM --> AM2["trainer_atari.py<br/>학습 루프"]

        AR["run_dt_atari.py<br/>메인 스크립트"]
        AD["create_dataset.py<br/>데이터셋 생성"]
        AB["fixed_replay_buffer.py<br/>버퍼 로딩"]
    end

    subgraph Gym["🤸 Gym 구현 (gym/)"]
        direction TB
        subgraph Models["models/"]
            GM1["decision_transformer.py<br/>메인 DT 모델"]
            GM2["trajectory_gpt2.py<br/>커스텀 GPT-2"]
            GM3["mlp_bc.py<br/>BC 베이스라인"]
        end

        subgraph Training["training/"]
            GT1["seq_trainer.py<br/>DT 트레이너"]
            GT2["act_trainer.py<br/>BC 트레이너"]
        end

        subgraph Eval["evaluation/"]
            GE1["evaluate_episodes.py<br/>에피소드 평가"]
        end

        GX["experiment.py<br/>메인 실험 스크립트"]
    end

    style Atari fill:#fff3e0
    style Gym fill:#e8f5e9
```

### Data Processing

```mermaid
flowchart TB
    subgraph AtariData["🎮 Atari 데이터 처리"]
        AD1["DQN-replay 버퍼<br/>(50개/게임)"] --> AD2["궤적 샘플링"]
        AD2 --> AD3["프레임 스택<br/>(4×84×84)"]
        AD3 --> AD4["(s, a, rtg) 시퀀스"]
    end

    subgraph GymData["🤸 Gym 데이터 처리"]
        GD1["D4RL 데이터셋"] --> GD2["Pickle 변환<br/>(env-dataset-v2.pkl)"]
        GD2 --> GD3["State 정규화<br/>(평균/표준편차)"]
        GD3 --> GD4["RTG 계산<br/>(discount cumsum)"]
        GD4 --> GD5["Context K 추출<br/>(기본 K=20)"]
    end

    style AtariData fill:#fff3e0
    style GymData fill:#e8f5e9
```

### Model Details

**Decision Transformer 아키텍처:**

```mermaid
flowchart TB
    subgraph Inputs["📥 입력"]
        RTG["RTG<br/>(batch, K, 1)"]
        State["State<br/>(batch, K, state_dim)"]
        Action["Action<br/>(batch, K, act_dim)"]
        Time["Timestep<br/>(batch, K)"]
    end

    subgraph Embedding["1️⃣ 임베딩"]
        RTG --> |"Linear"| RE["RTG Emb"]
        State --> |"Linear"| SE["State Emb"]
        Action --> |"Linear"| AE["Action Emb"]
        Time --> |"Embedding"| TE["Time Emb"]

        RE --> |"+"| REF["R + T"]
        TE --> REF
        SE --> |"+"| SEF["S + T"]
        TE --> SEF
        AE --> |"+"| AEF["A + T"]
        TE --> AEF
    end

    subgraph Stack["2️⃣ 시퀀스 구성"]
        REF --> Interleave
        SEF --> Interleave
        AEF --> Interleave
        Interleave["Interleave<br/>[R,s,a,R,s,a,...]"] --> LN["LayerNorm"]
    end

    subgraph TF["3️⃣ Transformer"]
        LN --> GPT["GPT-2<br/>(Causal Attention)"]
        GPT --> Out["(batch, K×3, hidden)"]
    end

    subgraph Heads["4️⃣ 예측 헤드"]
        Out --> |"[:, 1::3, :]"| PA["predict_action<br/>⭐ 주요 목표"]
        Out --> |"[:, 2::3, :]"| PS["predict_state<br/>(미사용)"]
        Out --> |"[:, 2::3, :]"| PR["predict_return<br/>(미사용)"]
    end

    style Inputs fill:#e1f5fe
    style Embedding fill:#fff3e0
    style Stack fill:#f3e5f5
    style TF fill:#e8f5e9
    style Heads fill:#ffebee
    style PA fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
```

**시퀀스 구성** ([decision_transformer.py:73-78](gym/decision_transformer/models/decision_transformer.py#L73-L78)):
```python
# (R, s, a) 트리플을 스택하여 시퀀스 생성
# 최종 형태: [batch, seq_len*3, hidden_dim]
# R_1, s_1, a_1, R_2, s_2, a_2, ...
```

**예측 헤드** ([decision_transformer.py:97-99](gym/decision_transformer/models/decision_transformer.py#L97-L99)):
- `predict_action`: state 토큰에서 다음 action 예측 (주요 목표)
- `predict_state`: action 토큰에서 다음 state 예측 (논문에서 미사용)
- `predict_return`: action 토큰에서 다음 return 예측 (논문에서 미사용)

**추론 시** ([decision_transformer.py:103-140](gym/decision_transformer/models/decision_transformer.py#L103-L140)):
- `get_action()`: 현재까지의 궤적과 원하는 rtg를 받아 다음 action 반환
- Max length로 컨텍스트 윈도우 제한, 패딩 처리

### 학습 vs 추론 흐름

```mermaid
flowchart TB
    subgraph Training["📚 학습 (Offline)"]
        T1["과거 데이터셋<br/>(trajectories)"] --> T2["RTG 계산<br/>(실제 값)"]
        T2 --> T3["(R, s, a) 시퀀스 구성"]
        T3 --> T4["Transformer Forward"]
        T4 --> T5["Action 예측"]
        T5 --> T6["MSE Loss<br/>(예측 vs 실제)"]
        T6 --> T7["Backprop"]
    end

    subgraph Inference["🎯 추론 (Online)"]
        I1["목표 Return 설정<br/>(사용자 지정)"] --> I2["초기 RTG = 목표"]
        I2 --> I3["현재 State 관측"]
        I3 --> I4["Transformer로<br/>Action 예측"]
        I4 --> I5["환경에서 실행"]
        I5 --> I6["Reward 획득"]
        I6 --> I7["RTG 업데이트<br/>(RTG -= reward)"]
        I7 --> I3
    end

    Training --> |"학습된 모델"| Inference

    style Training fill:#e3f2fd
    style Inference fill:#fff8e1
```

## Important Implementation Notes

```mermaid
flowchart LR
    subgraph Notes["⚠️ 주의사항"]
        N1["PYTHONPATH<br/>각 디렉토리 추가 필요"]
        N2["실행 위치<br/>cd atari 또는 cd gym"]
        N3["Context Length<br/>Atari: 30 / Gym: 20"]
        N4["하이퍼파라미터<br/>환경별로 다름"]
    end

    style Notes fill:#fff3e0
```

- **PYTHONPATH 설정**: 각 디렉토리(`atari/`, `gym/`)를 PYTHONPATH에 추가해야 할 수 있음
- **스크립트 실행 위치**: 항상 해당 하위 디렉토리에서 실행 (`cd atari` 또는 `cd gym`)
- **모델 체크포인트**: Atari는 자동으로 체크포인트 저장, Gym은 wandb 옵션 사용 시 로깅
- **Context length**: Atari는 `context_length` (기본 30), Gym은 `K` (기본 20) 파라미터로 제어
- **하이퍼파라미터**: 각 게임/환경마다 최적 설정이 다름 - `run.sh` 또는 `experiment.py` 참조

## Known Issues

- **off-by-one 버그 수정됨**: rtg 계산 관련 버그 패치 적용됨 (최근 커밋 참조)
- **MuJoCo 라이선스**: Gym 환경은 MuJoCo 설치 및 라이선스 필요
- **GPU 메모리**: Atari 학습 시 배치 크기 조절 필요할 수 있음
