# Decision Transformer Architecture Flow

이 문서는 Decision Transformer (Atari 환경)의 전체 아키텍처와 데이터 흐름을 flowchart로 설명합니다.

## 1. 전체 시스템 개요

```mermaid
graph TB
    A[DQN Replay Buffers] --> B[Data Loading & Preprocessing]
    B --> C[StateActionReturnDataset]
    C --> D[DataLoader]
    D --> E[GPT Model]
    E --> F[Trainer]
    F --> G[Evaluation]
    G --> H[Atari Environment]

    style E fill:#ff6b6b
    style B fill:#4ecdc4
    style G fill:#45b7d1
```

## 2. 데이터 파이프라인 상세

```mermaid
flowchart TD
    Start([시작: DQN Replay Logs]) --> LoadBuffers[50개 버퍼 중 마지막 N개 선택]
    LoadBuffers --> SampleTraj[각 버퍼에서 궤적 샘플링]

    SampleTraj --> CollectData{num_steps 달성?}
    CollectData -->|No| SampleTraj
    CollectData -->|Yes| ComputeRTG[Return-to-Go 계산]

    ComputeRTG --> RTGCalc["RTG[t] = sum(rewards[t:end])"]
    RTGCalc --> CreateTimesteps[Timestep 배열 생성]
    CreateTimesteps --> Dataset[(StateActionReturnDataset)]

    Dataset --> BatchSampling[배치 샘플링]
    BatchSampling --> EpisodeBoundary{에피소드 경계 체크}
    EpisodeBoundary --> NormalizeStates[이미지 정규화: /255]
    NormalizeStates --> Output([출력: states, actions, rtgs, timesteps])

    style ComputeRTG fill:#ffd93d
    style RTGCalc fill:#fcbf49
    style Dataset fill:#6bcf7f
```

### RTG 계산 세부 과정

```mermaid
flowchart LR
    T0[t=0<br/>reward=1]
    T1[t=1<br/>reward=2]
    T2[t=2<br/>reward=3]
    T3[t=3<br/>reward=4]

    T3 --> R3[RTG_3=4]
    T2 --> R2[RTG_2=7<br/>3+4]
    T1 --> R1[RTG_1=9<br/>2+3+4]
    T0 --> R0[RTG_0=10<br/>1+2+3+4]

    style R0 fill:#ff6b6b
    style R1 fill:#ee5a6f
    style R2 fill:#c44569
    style R3 fill:#a73e5c
```

## 3. GPT 모델 아키텍처

```mermaid
flowchart TD
    subgraph Input["입력 데이터"]
        States["States<br/>(batch, K, 4×84×84)"]
        Actions["Actions<br/>(batch, K, 1)"]
        RTGs["RTGs<br/>(batch, K, 1)"]
        Timesteps["Timesteps<br/>(batch, 1, 1)"]
    end

    States --> CNN["CNN Encoder<br/>Conv2d layers"]
    Actions --> ActEmb["Action Embedding<br/>nn.Embedding"]
    RTGs --> RTGEmb["RTG Embedding<br/>Linear + Tanh"]

    CNN --> StateEmb["State Embeddings<br/>(batch, K, 128)"]
    ActEmb --> ActionEmb["Action Embeddings<br/>(batch, K, 128)"]
    RTGEmb --> RTGEmbOut["RTG Embeddings<br/>(batch, K, 128)"]

    StateEmb --> Stack
    ActionEmb --> Stack
    RTGEmbOut --> Stack

    Stack["시퀀스 스택<br/>[R₀, s₀, a₀, R₁, s₁, a₁, ...]"] --> PosEmb["위치 임베딩 추가<br/>Global + Relative"]

    Timesteps --> GlobalPos["Global Position<br/>Embedding"]
    GlobalPos --> PosEmb

    PosEmb --> Dropout[Dropout]
    Dropout --> Transformer

    subgraph Transformer["Transformer Blocks (×6)"]
        direction TB
        LN1[LayerNorm] --> Attn[Multi-Head<br/>Causal Attention]
        Attn --> Res1[+ Residual]
        Res1 --> LN2[LayerNorm]
        LN2 --> MLP["MLP<br/>(4× expansion)"]
        MLP --> Res2[+ Residual]
    end

    Transformer --> FinalLN[Final LayerNorm]
    FinalLN --> Head["Linear Head<br/>(128 → vocab_size)"]
    Head --> Extract["예측 추출<br/>state 위치만 선택<br/>(1::3)"]
    Extract --> Output["Action Logits<br/>(batch, K, vocab_size)"]

    style Stack fill:#ff6b6b
    style Transformer fill:#4ecdc4
    style Extract fill:#ffd93d
```

### CNN Encoder 세부 구조

```mermaid
flowchart LR
    Input["Input<br/>(4, 84, 84)"] --> Conv1["Conv2d<br/>kernel=8, stride=4<br/>4→32 channels"]
    Conv1 --> ReLU1[ReLU]
    ReLU1 --> Output1["(32, 20, 20)"]

    Output1 --> Conv2["Conv2d<br/>kernel=4, stride=2<br/>32→64 channels"]
    Conv2 --> ReLU2[ReLU]
    ReLU2 --> Output2["(64, 9, 9)"]

    Output2 --> Conv3["Conv2d<br/>kernel=3, stride=1<br/>64→64 channels"]
    Conv3 --> ReLU3[ReLU]
    ReLU3 --> Output3["(64, 7, 7)"]

    Output3 --> Flatten["Flatten<br/>3136"]
    Flatten --> Linear["Linear<br/>3136→128"]
    Linear --> Tanh[Tanh]
    Tanh --> FinalOutput["Output<br/>(128,)"]

    style Input fill:#e3f2fd
    style FinalOutput fill:#c8e6c9
```

## 4. 학습 루프

```mermaid
flowchart TD
    Start([학습 시작]) --> InitModel[모델 초기화<br/>GPTConfig]
    InitModel --> InitOptim[Optimizer 설정<br/>AdamW, lr=6e-4]
    InitOptim --> EpochLoop{Epoch < max_epochs?}

    EpochLoop -->|Yes| BatchLoop[배치 순회]
    BatchLoop --> LoadBatch["배치 로드<br/>(states, actions, rtgs, timesteps)"]
    LoadBatch --> Forward["Forward Pass<br/>logits, loss = model(...)"]
    Forward --> Backward[역전파<br/>loss.backward]
    Backward --> ClipGrad["Gradient Clipping<br/>max_norm=1.0"]
    ClipGrad --> OptimStep[optimizer.step]
    OptimStep --> LRDecay{LR Decay 활성화?}

    LRDecay -->|Yes| Warmup{Warmup 단계?}
    Warmup -->|Yes| LinearWarmup["Linear Warmup<br/>lr × (tokens/warmup_tokens)"]
    Warmup -->|No| CosineDecay["Cosine Decay<br/>lr × 0.5(1 + cos(π×progress))"]

    LinearWarmup --> NextBatch
    CosineDecay --> NextBatch
    LRDecay -->|No| NextBatch{더 많은 배치?}

    NextBatch -->|Yes| LoadBatch
    NextBatch -->|No| Evaluate[평가 단계]

    Evaluate --> SetTargetRTG["Target RTG 설정<br/>Breakout: 90<br/>Seaquest: 1150<br/>Qbert: 14000<br/>Pong: 20"]
    SetTargetRTG --> RunEpisodes["10 에피소드 실행<br/>get_returns()"]
    RunEpisodes --> CalcAvgReturn[평균 Return 계산]
    CalcAvgReturn --> EpochLoop

    EpochLoop -->|No| End([학습 완료])

    style Forward fill:#ff6b6b
    style Evaluate fill:#4ecdc4
    style SetTargetRTG fill:#ffd93d
```

### Learning Rate 스케줄

```mermaid
graph LR
    A[0] -->|Linear Warmup| B[warmup_tokens]
    B -->|Cosine Decay| C[final_tokens]

    subgraph "LR 변화"
        D["lr × progress"]
        E["lr × max(0.1, 0.5(1 + cos(π×progress)))"]
    end

    A -.->|사용| D
    B -.->|사용| E

    style B fill:#ffd93d
```

## 5. 평가 (추론) 과정

```mermaid
flowchart TD
    Start([평가 시작]) --> InitEnv[환경 초기화<br/>Atari Env]
    InitEnv --> SetModel[model.eval]
    SetModel --> EpisodeLoop["에피소드 반복<br/>(10회)"]

    EpisodeLoop --> Reset[state = env.reset]
    Reset --> InitRTG["rtg = target_return<br/>(예: 90 for Breakout)"]
    InitRTG --> InitTimestep[timestep = 0]
    InitTimestep --> FirstAction["첫 Action 샘플링<br/>sample(model, state, rtgs=[rtg], timesteps=0)"]

    FirstAction --> StepLoop{Done?}

    StepLoop -->|No| ExecuteAction["action 실행<br/>env.step(action)"]
    ExecuteAction --> GetReward["state, reward, done 획득"]
    GetReward --> UpdateRTG["RTG 업데이트<br/>rtg = rtg - reward"]
    UpdateRTG --> UpdateTimestep["timestep += 1"]
    UpdateTimestep --> AppendState["state를 history에 추가"]
    AppendState --> CropHistory{History > context_length?}

    CropHistory -->|Yes| Crop["마지막 K개만 유지"]
    CropHistory -->|No| SkipCrop[Skip]
    Crop --> SampleNext
    SkipCrop --> SampleNext["다음 Action 샘플링<br/>sample(model, all_states, actions, rtgs, timesteps)"]
    SampleNext --> StepLoop

    StepLoop -->|Yes| RecordReturn[episode_return 기록]
    RecordReturn --> MoreEpisodes{더 많은 에피소드?}

    MoreEpisodes -->|Yes| EpisodeLoop
    MoreEpisodes -->|No| CalcMean["평균 return 계산<br/>sum(returns) / 10"]
    CalcMean --> PrintResult["출력:<br/>Target: {target_rtg}<br/>Actual: {avg_return}"]
    PrintResult --> End([평가 완료])

    style UpdateRTG fill:#ff6b6b
    style SampleNext fill:#4ecdc4
    style CalcMean fill:#45b7d1
```

### RTG 동적 업데이트 예시

```mermaid
flowchart LR
    subgraph "에피소드 진행"
        T0["t=0<br/>RTG=90<br/>action→reward=5"]
        T1["t=1<br/>RTG=85<br/>action→reward=10"]
        T2["t=2<br/>RTG=75<br/>action→reward=3"]
        T3["t=3<br/>RTG=72<br/>..."]
    end

    T0 -->|"rtg - reward"| T1
    T1 -->|"rtg - reward"| T2
    T2 -->|"rtg - reward"| T3

    style T0 fill:#ff6b6b
    style T1 fill:#ee5a6f
    style T2 fill:#c44569
    style T3 fill:#a73e5c
```

## 6. 모델별 시퀀스 구성 비교

### Reward-Conditioned (Decision Transformer)

```mermaid
flowchart LR
    subgraph "입력 시퀀스"
        R0[R₀] --> S0[s₀] --> A0[a₀] --> R1[R₁] --> S1[s₁] --> A1[a₁] --> R2[R₂] --> S2[s₂]
    end

    subgraph "Transformer 출력"
        O0[?] --> O1[→a₀] --> O2[?] --> O3[?] --> O4[→a₁] --> O5[?] --> O6[?] --> O7[→a₂]
    end

    R0 -.-> O0
    S0 -.-> O1
    A0 -.-> O2
    R1 -.-> O3
    S1 -.-> O4
    A1 -.-> O5
    R2 -.-> O6
    S2 -.-> O7

    style O1 fill:#4ecdc4
    style O4 fill:#4ecdc4
    style O7 fill:#4ecdc4
```

**핵심:** State 위치 (1::3)에서만 action 예측 추출

### Naive (Behavior Cloning)

```mermaid
flowchart LR
    subgraph "입력 시퀀스"
        S0[s₀] --> A0[a₀] --> S1[s₁] --> A1[a₁] --> S2[s₂] --> A2[a₂]
    end

    subgraph "Transformer 출력"
        O0[→a₀] --> O1[?] --> O2[→a₁] --> O3[?] --> O4[→a₂] --> O5[?]
    end

    S0 -.-> O0
    A0 -.-> O1
    S1 -.-> O2
    A1 -.-> O3
    S2 -.-> O4
    A2 -.-> O5

    style O0 fill:#ffd93d
    style O2 fill:#ffd93d
    style O4 fill:#ffd93d
```

**핵심:** State 위치 (0::2)에서 action 예측 추출, RTG 없음

## 7. Causal Attention Masking

```mermaid
flowchart TD
    subgraph "Attention Matrix (Reward-Conditioned)"
        direction LR
        Row0["R₀: [R₀]"]
        Row1["s₀: [R₀, s₀]"]
        Row2["a₀: [R₀, s₀, a₀]"]
        Row3["R₁: [R₀, s₀, a₀, R₁]"]
        Row4["s₁: [R₀, s₀, a₀, R₁, s₁]"]
        Row5["a₁: [R₀, s₀, a₀, R₁, s₁, a₁]"]
    end

    Info["각 위치는 자신과<br/>이전 위치만 볼 수 있음<br/>(하삼각 마스크)"]

    Row0 --> Row1 --> Row2 --> Row3 --> Row4 --> Row5
    Info -.-> Row4

    style Row1 fill:#c8e6c9
    style Row4 fill:#c8e6c9
```

**예시:** s₁ 위치에서 a₁을 예측할 때, [R₀, s₀, a₀, R₁, s₁]까지 모두 참조 가능

## 8. 샘플링 프로세스

```mermaid
flowchart TD
    Start([sample 함수 호출]) --> CheckContext{context > block_size?}
    CheckContext -->|Yes| CropContext["마지막 K개만 유지<br/>(sliding window)"]
    CheckContext -->|No| KeepAll[전체 유지]

    CropContext --> Forward
    KeepAll --> Forward["Forward Pass<br/>logits = model(states, actions, rtgs, timesteps)"]

    Forward --> ExtractLast["마지막 timestep logits 추출<br/>logits[:, -1, :]"]
    ExtractLast --> Temperature["Temperature Scaling<br/>logits / temperature"]
    Temperature --> TopK{top_k 사용?}

    TopK -->|Yes| ApplyTopK["상위 k개만 유지<br/>나머지는 -inf"]
    TopK -->|No| SkipTopK[Skip]

    ApplyTopK --> Softmax
    SkipTopK --> Softmax[Softmax<br/>확률 분포 변환]

    Softmax --> SampleMode{sample=True?}
    SampleMode -->|Yes| Stochastic["확률적 샘플링<br/>torch.multinomial"]
    SampleMode -->|No| Greedy["Greedy 선택<br/>argmax"]

    Stochastic --> Return
    Greedy --> Return([Action 반환])

    style Forward fill:#ff6b6b
    style Temperature fill:#ffd93d
    style Stochastic fill:#4ecdc4
```

## 9. 전체 시스템 데이터 플로우

```mermaid
flowchart TD
    subgraph Storage["오프라인 데이터"]
        DQN["DQN Replay Buffers<br/>(50 checkpoints)"]
    end

    subgraph Preprocessing["전처리"]
        Load[create_dataset.py]
        RTG[RTG 계산]
        Dataset[StateActionReturnDataset]
    end

    subgraph Training["학습"]
        DataLoader[DataLoader]
        Model["GPT Model<br/>(6 layers, 8 heads, 128 dim)"]
        Loss["Cross-Entropy Loss"]
        Optim["AdamW Optimizer"]
    end

    subgraph Evaluation["평가"]
        Env[Atari Environment]
        Sample[sample 함수]
        UpdateRTG["RTG 업데이트<br/>rtg -= reward"]
    end

    DQN --> Load
    Load --> RTG
    RTG --> Dataset
    Dataset --> DataLoader
    DataLoader --> Model
    Model --> Loss
    Loss --> Optim
    Optim --> Model

    Model --> Sample
    Sample --> Env
    Env --> UpdateRTG
    UpdateRTG --> Sample

    style Model fill:#ff6b6b
    style RTG fill:#ffd93d
    style UpdateRTG fill:#4ecdc4
```

## 10. 핵심 개념 요약

```mermaid
mindmap
    root((Decision Transformer))
        RL as Sequence Modeling
            Bellman Equation 불필요
            Value Function 불필요
            단순 Supervised Learning
        Return-to-Go
            미래 누적 보상
            목표 성능 명시
            동적 업데이트: rtg -= reward
        Transformer Architecture
            Multi-head Attention
            Causal Masking
            Long-range Dependencies
        Offline RL
            기존 데이터셋 활용
            환경 상호작용 불필요
            안전하고 확장 가능
        시퀀스 구성
            Reward-Conditioned: R, s, a
            Naive BC: s, a
            State 위치에서 Action 예측
```

## 11. 모델 타입별 차이점

| 특성 | Reward-Conditioned (DT) | Naive (BC) |
|------|------------------------|-----------|
| **입력 시퀀스** | [R₀, s₀, a₀, R₁, s₁, a₁, ...] | [s₀, a₀, s₁, a₁, ...] |
| **시퀀스 길이** | K × 3 | K × 2 |
| **예측 위치** | 1::3 (state 위치) | 0::2 (state 위치) |
| **조건화** | RTG로 목표 return 지정 | 조건 없음 |
| **평가 RTG** | 게임별 목표값<br/>(Breakout: 90) | 0 (무시됨) |
| **추론 시 RTG** | 동적 업데이트 | 사용 안 함 |

## 12. 하이퍼파라미터 요약

```mermaid
flowchart LR
    subgraph Model["모델 구조"]
        Layers[6 Layers]
        Heads[8 Attention Heads]
        Embed[128 Embedding Dim]
        Context[30 Context Length]
    end

    subgraph Training["학습 설정"]
        LR[6e-4 Learning Rate]
        BS[128 Batch Size]
        EP[5 Epochs]
        Warmup[512×20 Warmup Tokens]
    end

    subgraph Data["데이터"]
        Buffers[50 Replay Buffers]
        Steps[500K Steps]
        TrajPB[10 Trajectories/Buffer]
    end

    style Model fill:#e3f2fd
    style Training fill:#fff3e0
    style Data fill:#f3e5f5
```

---

## 참고 자료

- **논문:** [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
- **코드:** [decision-transformer/atari/](../atari/)
- **주요 파일:**
  - [run_dt_atari.py](../atari/run_dt_atari.py): 메인 실행 스크립트
  - [model_atari.py](../atari/mingpt/model_atari.py): GPT 모델 구현
  - [trainer_atari.py](../atari/mingpt/trainer_atari.py): 학습 및 평가 루프
  - [create_dataset.py](../atari/create_dataset.py): 데이터셋 생성 및 RTG 계산
  - [utils.py](../atari/mingpt/utils.py): 샘플링 함수
