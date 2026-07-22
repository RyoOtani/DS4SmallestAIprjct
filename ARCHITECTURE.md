# tinyllm アーキテクチャ詳細

## 設計思想

### ds4 の精神
- **単一バイナリ**: 全ての機能を 1 つの実行ファイルに統合
- **最小限のコード**: コア推論エンジンは C で約 5000 行
- **省リソース**: 8GB メモリで GPT-4 級の能力
- **外部依存ゼロ**: POSIX + C 標準ライブラリのみ
- **UNIX 哲学**: 小さなツールの組み合わせ (パイプ/サブプロセス)

### モデル効率性 (DeepSeek 由来)
- **細粒度 MoE**: トークン単位でトップ 2 エキスパートのみ活性化
- **MLA**: KV キャッシュを低ランク潜在空間に圧縮
- **MTP**: マルチトークン予測による知識密度向上

## データフロー

```
ユーザー入力 (CLI/HTTP)
        │
        ▼
┌───────────────────┐
│ トークナイザー (BPE) │  ← テキスト → トークン列
│ FIM 特殊トークン対応  │
└───────┬───────────┘
        │ tokens[]
        ▼
┌───────────────────┐
│ 埋め込み層          │  ← トークン → hidden_dim ベクトル
└───────┬───────────┘
        │ hidden[hidden_dim]
        ▼
┌───────────────────────────────────────┐
│ Transformer 層 × N (デフォルト 32)       │
│                                       │
│  for each layer:                      │
│    ┌─────────────────────────────┐    │
│    │ 1. RMS Norm (pre-attn)      │    │
│    │ 2. MLA Attention            │    │
│    │    ├─ Q = W_q @ hidden      │    │
│    │    ├─ KV_latent = W_kv @ h  │ ← 圧縮!
│    │    ├─ K = W_k_up @ latent   │    │
│    │    ├─ V = W_v_up @ latent   │    │
│    │    ├─ RoPE (Q, K)           │    │
│    │    ├─ Scaled Dot-Product    │    │
│    │    │  (cached latents →     │    │
│    │    │   reconstructed K,V)   │    │
│    │    └─ W_o @ attn_out        │    │
│    ├─ Residual: h = h + attn     │    │
│    ├─ RMS Norm (pre-FFN)         │    │
│    ├─ FFN (dense or MoE)         │    │
│    │   MoE:                      │    │
│    │   ├─ Gate: top-2 experts    │    │
│    │   ├─ Expert 0: SwiGLU FFN   │    │
│    │   ├─ Expert 1: SwiGLU FFN   │    │
│    │   └─ Weighted sum           │    │
│    └─ Residual: h = h + ffn      │    │
│    └─────────────────────────────┘    │
└───────────────────────────────────────┘
        │ hidden[hidden_dim]
        ▼
┌───────────────────┐
│ 最終 RMS Norm       │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ LM Head            │  ← hidden → vocab_size logits
└───────┬───────────┘
        │ logits[vocab_size]
        ▼
┌───────────────────┐
│ サンプラー          │
│ ├─ Temperature     │
│ ├─ Top-k           │
│ ├─ Top-p (nucleus) │
│ └─ Repetition Pen. │
└───────┬───────────┘
        │ next_token
        ▼
    出力 or ループ継続
```

## MLA (Multi-head Latent Attention) の詳細

MLA は DeepSeek-V2/V3 の中核的革新です。従来の Multi-Head Attention は各ヘッドの K, V を完全な次元で保存しますが、MLA はそれらを低ランク潜在空間に圧縮します。

### 従来の MHA の KV キャッシュ
```
メモリ = n_layers × 2 × n_heads × head_dim × seq_len × sizeof(float16)

例: 32 層, 32 ヘッド, head_dim=128, seq_len=4096
  = 32 × 2 × 32 × 128 × 4096 × 2 bytes
  = 2.0 GB (KV キャッシュのみ)
```

### MLA の KV キャッシュ
```
メモリ = n_layers × 2 × latent_dim × seq_len × sizeof(float16)

例: 32 層, latent_dim=512, seq_len=4096
  = 32 × 2 × 512 × 4096 × 2 bytes
  = 256 MB (約 1/8!)
```

### 演算の流れ
```
Input: hidden ∈ R^D

1. 圧縮:    kv_latent = W_kv_compress @ hidden     ∈ R^L   (L ≪ H×d)
2. Up-proj: K = W_k_up @ kv_latent                  ∈ R^{H×d}
3. Up-proj: V = W_v_up @ kv_latent                  ∈ R^{H×d}
4. Q = W_q @ hidden                                 ∈ R^{H×d}
5. RoPE を Q, K に適用
6. Scaled Dot-Product Attention (キャッシュされた latent から K,V を復元)
7. Output = W_o @ attn_out
```

### キャッシュ時の注意点
- キャッシュに保存するのは **latent** (∈ R^L) のみ
- 推論時は `W_k_up @ cached_latent` で K を、`W_v_up @ cached_latent` で V を復元
- 復元コストが追加で発生するが、メモリ削減効果の方が大きい
- 効率化のため、`W_k_up^T @ Q` を事前計算し、潜在空間で dot product を計算する最適化も可能

## MoE (Mixture of Experts)

### ゲートルーティング
```c
// ゲート: scores = gate_w @ hidden
// Top-2 エキスパートを選択
// Softmax で重みを正規化

int expert_indices[2];
float expert_weights[2];
tl_moe_gate(&layer->gate, hidden, expert_indices, expert_weights, 2);
```

### スパース計算
- 256 エキスパート中 2 つのみ活性化
- 活性化パラメータ = 総パラメータ × (2/256) ≈ 0.8%
- 7B 活性化 / 30B 総パラメータの構成が現実的

### ロードバランシング (学習時)
```python
loss_aux = N * sum(f_i * p_i)  # f_i: expert i への割り当て頻度, p_i: 平均ゲート確率
```

## 自己修正エージェントループ

```
PLAN → THINK → ACT → OBSERVE → (REPEAT or DONE)

1. PLAN:   タスクを CoT で分解、スクラッチパッドに計画を出力
2. THINK:  現在の状況を分析、次のアクションを決定
3. ACT:    ツール呼び出し (<tool_call> XML フォーマット)
4. OBSERVE: ツール結果を観測、成功/失敗を判断
5. 失敗時:  最大 5 回まで再試行、別のアプローチを試行
```

### 利用可能ツール
| ツール | 説明 |
|--------|------|
| `run_cmd` | シェルコマンド実行 |
| `read_file` | ファイル読み取り |
| `write_file` | ファイル書き込み |
| `search_code` | grep / AST 検索 |
| `run_test` | テスト実行 |
| `web_search` | Web 検索 (キャッシュ) |
| `rag_retrieve` | ローカル文書検索 |
| `mem_store` | 長期記憶保存 |
| `mem_recall` | 長期記憶呼出 |
| `sandbox_exec` | Docker/Podman 隔離実行 |
| `browser` | ヘッドレスブラウザ |

## 量子化戦略

### 混合精度
```
レイヤー位置        量子化ビット
─────────────────────────────
最初の 3 層          6-bit (Q6_K)
中間層               4-bit (Q4_0)
最後の 3 層          6-bit (Q6_K)
埋め込み層           8-bit (Q8_0) or FP16
LM Head             8-bit (Q8_0) or FP16
```

最初と最後の層は入出力の品質に直結するため高精度を維持し、中間層は大胆に量子化します。

### Q4_0 フォーマット
```
ブロックサイズ: 32 要素
1 ブロックのレイアウト:
  [16 bytes: 4-bit values (32 × 4bit ÷ 8bit/byte)]
  [2 bytes: float16 scale]
  合計: 18 bytes / 32 要素 = 4.5 bits/要素 (実効)

値の範囲: [-7, 7] × scale (8 を引くため、対称範囲 [-8, 7] 相当)
```

## メモリ予算 (8GB ターゲット)

```
コンポーネント              サイズ (概算)
─────────────────────────────────────
モデル重み (Q4_0, 30B 換算)    ~4.0 GB
KV キャッシュ (MLA 圧縮)       ~0.2 GB
活性化バッファ                  ~1.0 GB
RAG インデックス               ~0.5 GB
長期記憶                       ~0.3 GB
OS / その他                   ~2.0 GB
─────────────────────────────────────
合計                           ~8.0 GB
```

## 拡張計画 (GPT-5 レベルへ)

1. **活性化 7B MoE** (全パラ 30B): 細粒度 MoE + MLA で 8GB 内
2. **マルチモーダル**: CLIP 視覚エンコーダ、Whisper 音声エンコーダを GGUF 拡張統合
3. **ローカル継続学習**: LoRA アダプタ (数百 MB) + 自動ロールバック
4. **完全自律エージェント**: 階層プランニング + サブエージェント + 長期デーモン
5. **進化的アルゴリズム**: コード候補生成 → テスト評価 → 交叉突然変異
6. **外在化ブレインストーミング**: ランダム概念合成 + ツール検証の反復
