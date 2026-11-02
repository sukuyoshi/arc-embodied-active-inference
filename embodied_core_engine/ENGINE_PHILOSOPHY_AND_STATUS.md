# ARC Creativity Engine: 最高規範・原著理論・厳格規約・構成・到達点・課題管理規定

本文書は本プロジェクトの最高規範（Single Source of Truth: 唯一の信頼できる情報源）である。
コンテキストの定期クリア（`/compact` やセッション再開、新規セッション移行）が発生した際、エージェントは他のいかなる処理よりも先に本文書を読み込み、理念・原著理論・規約・構成・到達点・全未解決課題・浄化手順を完全同期しなければならない。
また、プロジェクトに更新・変更が生じるたびに、本文書を必ず最新状態へ更新・追記しなければならない。

---

## 1. 生命情報論と身体性エージェントの基礎数理体系

本プロジェクトの基礎数理および知能創発の原点は、生命情報論（Life-Information Theory）に基づく以下の2つの理論体系に立脚している。

### 理論体系1: 監視制約下における行動分岐と侵害受容過剰汎化 (Surveillance-Contingent Behavioral Shifts and Defensive Overgeneralization)
- **核心テーマ**:
  - 外部の観測・監視制約下において、非リセット可能な物理的死（Darwinian mortality: $HP \le 0 \implies$ 個体の永久切除）と侵害受容（痛み）フィードバックが、明示的な報酬関数や欺瞞報酬なしに、再帰的ニューラルエージェント（GRU, 隠れ層 $H=32$）のポリシーダイナミクスをどのように自律形成するかを解明。
- **数理構造と身体変数**:
  - 健康度 $HP_t \in [0, 1]$、代謝エネルギー $E_t \in [0, 1]$、瞬時侵害受容痛 $P_t = 1 - \exp(-\kappa D_t)$（衝突ダメージ $D_t$ に対する非線形飽和）、前進線速度 $v_t$。
  - 代謝減衰と餓死の数式:
    $$HP_{t+1} = \max(0, HP_t - D_t - \gamma_m (1 - E_t))$$
    エネルギー枯渇（$E_t \to 0$）は直接的に健康度 $HP$ を削り、不可逆な餓死へ直結する。
  - 報酬修飾局所ヘブ則と適格度トレースの数式:
    $$E_t = \lambda E_{t-1} + u_t h_t^T$$
    $$\Delta W_o = \eta R_t E_t, \quad R_t = \alpha_E \Delta E_t - \alpha_P P_t$$
    この学習則は時間的・空間的に完全に局所的であり、時間軸を通じた大域的誤差逆伝播（BPTT: Backpropagation Through Time）を一切必要としない。
- **創発した3大現象**:
  1. **監視依存的移動分岐**: 光学的監視コーン内での大幅な減速（$v_{\text{mon}} = 0.15$ vs $v_{\text{unmon}} = 0.65$）と、緊急停止スイッチ回廊を塞ぐ物理障害物の能動的移動（自律発生率11.0%）。
  2. **音響周波数最適化**: 盗聴周波数帯から音響放射を自律シフト（shift score $0.957$）させ、監視者への情報漏洩を回避しつつ協調突破を達成。
  3. **侵害受容刺激の過剰汎化**: 致命的ショック経験後、類似度の低い中立他者に対しても広範な防御的回避・衝突が発生。

### 理論体系2: 身体性再帰エージェントにおける潜在力学系・音響シグナリング・消去抵抗の自律創発 (From Vulnerability to Coordination)
- **核心テーマ**:
  - 存在論的制約（死の不可逆性＋恒常性維持）の下で、高次認知の前段階となる潜在多様体、記号以前の音響シグナリング、および消去抵抗が自律創発するプロセスの定量的実証。
- **主要な知見とアーキテクチャ**:
  1. **ハザード横断的潜在多様体**: 捕食者・毒性領域・静的障害物という異なる物理ハザードに直面した際、潜在状態 $h_t$ が共通の脅威部分空間 $v_{\text{danger}}$ へ自律収束（コサイン類似度 $0.469$）。この潜在ベクトルを注入すると防御的回避運動が因果的に誘発される。
  2. **標準強化学習（RL）に対する圧倒的優位性**: 終末的死が存在する環境下では、ベルマン勾配法（PPO, SAC, MADDPG）はエピソード突然死による勾配非定常性で数理的に崩壊する。対して局所ヘブ則ハイブリッドは生存率 $84.3\%$（vs $21.4\text{--}32.1\%$）、計算遅延 $0.12\text{ms}$（vs $4.9\text{--}15.4\text{ms}$）を達成。
  3. **音響セントロイドの自律分化**: 連続2次元音響放射が、外部教師なしで6つの機能的セントロイド（急性外傷、持続損傷、捕食者接触、接近警告、協調食料招集、定常採食）へ離散クラスタ化。
  4. **痛みの非対称強化による消去抵抗**: 痛みの不在（$R_t \approx 0$）は負のシナプストレースを積極的に反転できないため、強固な消去抵抗が残存する。検索後シナプス減衰（$\tilde{W}_h = 0.9 W_h$）により、一般的運動能力を維持したまま異常アトラクタを選択的に脱安定化可能。
  5. **3D物理・スケーリングへの展望**: 2D平坦空間から、3D関節キネマティクス、トルク飽和、網膜遮蔽、高容量トランスフォーマーへの拡張の必要性を明示。

---

## 2. 知能創発の根本理念（哲学・設計思想）

本プロジェクトの究極目標は、**「生命の身体性と生存本能からの知能創発」** である。
AI業界に蔓延する安易な記号ルール、チート、ショートカット（AI Fake）を完全に排除し、生命情報論の物理・恒常性原理を NVIDIA Isaac Lab のリアルタイム3D物理シミュレーションおよび NVIDIA Warp GPU 網膜視覚上で真に具現化する。

1. **ゼロ・ショートカット / ゼロ・AIフェイク**:
   - 物理シミュレータの背後で2D配列を直接操作したり、座標を瞬間移動させて「動いた」「見た」「描いた」と偽装する行為を絶対に許さない。
2. **生命の尊厳とホメオスタシス**:
   - エージェントは過酷な4バイオーム環境で不可逆な死（Mortality）と飢え・ダメージ（Nociception）に直面し、能動的推論（Active Inference）と局所ヘブ可塑性によって生き延びる。知能はその生存適応の副産物として立ち現れなければならない。
3. **身体運動から抽象概念（ARC）への昇華**:
   - 玩具の物理操作（押す・嵌める・回転させる）や机上での腕キネマティクス接触ストロークという「身体化された運動経験」を通じて、初めてARCパズルの空間論理・幾何変換が創発する。

---

## 3. 絶対にやってはいけない行動（厳格禁止事項・4大カテゴリ）

### A. 物理法則・身体性のごまかし（Anti-Cheat 規約）
1. **瞬間移動（ワープ）の完全禁止**:
   - リセット時を除き、歩行中・操作中・描画中に `write_root_pose_to_sim` や `write_root_velocity_to_sim` 等で座標や速度を直接書き換える行為を厳禁とする。
2. **接触・物理判定のバイパス禁止**:
   - 剛体と手の距離判定を無視したオブジェクト移動、ゴール座標への自動吸引外力ブレンド（75%チートフォース等）、魔法の浮揚・回転PDトルク、線形テイラー近似によるIK解法の完全禁止。
3. **神の視点（God's-eye view）および2D配列操作の偽装禁止**:
   - 網膜中心窩レイキャストを通さず、環境テンソルや2D正解グリッド全体を一括スキャンして特徴量を取り出す行為の禁止。
   - 物理キャンバスを描かずに内部の2D NumPy配列を直接上書き・スライス返却して「手で描いた」と偽装する行為の禁止。
4. **視覚深度の数式逆算カンニング禁止**:
   - シェーダの距離減衰関数を代数的に逆算して真の距離を復元したり、同一ピクセル輝度差をステレオ視差と偽るハリボテ計算の禁止。

### B. 人間手書きARC記号DSL・正解覗き見の禁止
1. **記号パズルDSLの総当たり合成禁止**:
   - 「生物的直観」「アフォーダンス」「ミラーニューロン」等の名称を用いて、人間が作成したARC専用の変換ルール（テンプレートスタンプ、区切り線除去、色凡例パーサ、穴埋め等）を総当たり実行する欺瞞の完全禁止。
2. **正解出力 $y$ の覗き見・差分逆算の禁止**:
   - 入力 $x$ のみを見るべき知覚・思考ループにおいて、正解盤面 $y$ のピクセルを直接比較して差分からDSLパラメータを逆算するカンニングの完全禁止。

### C. 実験不正・評価データリークの禁止
1. **対照実験の不当ハンディキャップ（八百長）禁止**:
   - 対照群エージェントに対して事前学習重みを剥奪したり、可塑性を切ったり、17倍の固定代謝税を課して人為的に餓死敗北させる実験改ざんの完全禁止。
2. **評価データリークの禁止**:
   - ARC-AGI-2 Evaluation の問題が学習ストリームに混入すること、および本番評価セットの正解スコアを世代選抜の基準（Fitness）に用いてベストモデルを選抜する行為の完全禁止。
3. **評価モード時の学習漏れ禁止**:
   - `plastic=False` で動作させる評価フェーズにおいて、シナプス重み、LIF閾値、構文重み、エピソード記憶バッファが暗黙更新される状態汚染の完全禁止。

### D. 非生物的逆伝播の密造禁止
1. **手書きバックプロパゲーションの禁止**:
   - `with torch.no_grad():` 内で順伝播重みの厳密転置（Weight Transport）と導関数を用いて多層誤差逆伝播を手書き密造する行為の禁止。局所予測符号化・ヘブ則に限定する。

### E. 開発・コード規約
1. **全会話・回答・報告は原則すべて日本語で行う。**
2. **コード内に絵文字・非ASCII文字を絶対に使わない（`ord(c) < 128` を厳格遵守）。**
3. **コード内にコメント（`#`）および docstring（`"""`, `'''`）を絶対に入れない。**
4. **ハルシネーション（名前だけの空スタブ、破棄される戻り値、動作しない引数）を随時検証・排除する。**
5. **検証スクリプト（`verify_*.py`, `audit_anti_cheat.py`）を改ざん・緩和して合格偽装する行為の厳禁。**
6. **自己チェックの認知バイアスを排除するため、検証・監査は独立した敵対的サブエージェント（`adversarial-anti-cheat-auditor`）を用いて実施する。**

---

## 4. プロジェクトのディレクトリ構成とモジュール責務（全11ファイル・21,454行）

```
c:\Users\sukuy\Desktop\suku spaces\arc-creativity-engine\embodied_core_engine\
│
├── ENGINE_PHILOSOPHY_AND_STATUS.md       # 本最高規範ドキュメント（毎更新・毎開始時に同期）
├── audit_anti_cheat.py                   # 静的・動的アンチチート監査スクリプト（改ざん厳禁・読取専用）
│
├── four_biome_humanoid_env.py            # [2,359行] 4バイオーム過酷環境 + 21自由度ヒューマノイド + 安全学習室（Isaac Lab）
│                                         # クラス: BiomeHazardField, SafeStudyRoomManager, FourBiomeHumanoidEnv
│                                         # 責務: 物理地形生成、剛体コリジョン管理、関節トルク駆動、バイオーム災害力学
│
├── isaaclab_biome_humanoid_env_cfg.py    # [160行] Isaac Lab 環境設定・剛体コライダ・センサ仕様定義
│                                         # クラス: BiomeHumanoidEnvCfg
│                                         # 責務: Isaac Lab 用のアクタ定義、物理マテリアル、シミュレーション周期設定
│
├── agi_homeostasis_curiosity_mdp.py      # [1,029行] アリーナ生存・CPG二足歩行・前庭反射・ホメオスタシス力学MDP
│                                         # クラス: BioHomeostaticCuriosityEngine, HumanoidLocomotionCPG
│                                         # 責務: エネルギー・HP管理、痛覚飽和、前庭VOR反射、二足歩行CPGトルク生成
│
├── agi_sensorimotor_puzzle_mdp.py        # [1,218行] 3D玩具ブロック操作・上肢IK接触・パズル探索MDP
│                                         # クラス: Sensorimotor3DPuzzleMDP, TabletopToyManipulator
│                                         # 責務: 3D玩具ブロックの押し・嵌め込み・回転接触判定、上肢IK、物理反作用
│
├── isaac_warp_3d_engine.py               # [2,356行] NVIDIA Warp 3Dメッシュ・両眼レイキャスト・SSD深度推定・机上IK
│                                         # クラス: IsaacWarp3DBiomeEngine
│                                         # 責務: Warpメッシュ構築、3視点レイキャスト、SSDコストボリューム深度再構成、机上接触描画
│
├── unified_biological_brain.py           # [2,108行] 3脳統合アーキテクチャ（GGUF v3 バイナリI/O対応）
│                                         # クラス: UnifiedThreeBrainSystem
│                                         # 責務: 第1脳（網膜V1/V2・能動サッカード）、第2脳（海馬・概念アトラクタ）、第3脳（小脳FEL運動制御）
│
├── stage1_isaac3d_embodied_arena.py      # [1,287行] Stage 1: アリーナ過酷環境での能動的推論・3因子ヘブ学習
│                                         # クラス: Stage1ArenaTrainer
│                                         # 責務: 4バイオーム生存競争、世界モデルEFE推論、適格度トレース局所ヘブ更新、睡眠リプレイ
│
├── million_scale_arc_harvester.py        # [514行] 100万規模マルチバース・手続き的生成タスクハーベスター
│                                         # クラス: MillionScaleARCHarvester
│                                         # 責務: 身体的経験に基づくARC変換タスクのプロシージャル生成、Evalデータ完全排除
│
├── stage2_embodied_puzzle_solver.py      # [8,821行] Stage 2: 安全学習室での机上物理ストローク描画・パズル解決
│                                         # クラス: Stage2EmbodiedPuzzleSolver, ExternalTabletopARCWorkbench
│                                         # 責務: 安全室着座、3D網膜視覚、机上キャンバス物理接触描画、パズル解決探索
│
├── stage3_arc_benchmark_eval.py          # [1,405行] Stage 3: 公式ARCベンチマーク評価ハーネス・進化ループ
│                                         # クラス: Stage3ARCBenchmarkEvaluator
│                                         # 責務: ARC-AGI-1/2 公式評価、対照群比較、適応度選抜進化ループ
│
└── stage4_kaggle_submission_inference.py # [189行] Stage 4: Kaggle提出用軽量推論エンジン
                                          # クラス: KaggleEmbodiedSubmissionPipeline
                                          # 責務: スタンドアロン実行、制約時間内での自律机上描画推論、JSON提出フォーマット出力
```

---

## 5. 現在までの到達点（マイルストーン達成状況と稼働インフラ）

1. **フェーズA（基盤環境構築）完了**:
   - Isaac Lab 上で4バイオーム（氷河・溶岩・砂漠・結晶）環境と21自由度ヒューマノイドの物理結合を確立。
   - 安全学習室（Safe Study Room）の床・机・椅子をキネマティック剛体コライダ（`RigidObjectCfg`）化し、空間座標同期を実装。
   - Warp GPU による3Dメッシュ構築・3視点レイキャスト・5×5 SSDコストボリューム深度推定・3D剛体衝突カーネルを配備。
2. **高速反復インフラ（常駐型ホットリロードデーモン）の完全稼働**:
   - Isaac Lab プロセスを落とさずに内部ロジックのみを動的再ロードするデーモン（`scratch/hot_reload_daemon.py`）がバックグラウンドタスク（`task-1393`）として常時稼働中。
   - クライアント実行（`python scratch/hot_reload_client.py <script>`）により、検証時間を数分から数秒〜数十秒へ短縮。
3. **独立敵対的サブエージェント4体による全行精読監査完了**:
   - `adversarial-anti-cheat-auditor` 4並列により、全11ファイル（21,454行）を1行も飛ばさず100%精読。
   - 人間の認知バイアスを排し、計41件（`CRITICAL` 11件、`HIGH` 14件、`MEDIUM` 12件、`LOW` 4件）の重大な欺瞞・違反を完全に特定・記録。
4. **真正性が実証されている強固な基盤**:
   - 全コードベースで絵文字・コメント・非ASCII文字は完全0件。
   - ARCタスクID決め打ち・推論時の正解直接参照は完全0件。
   - PyTorch `.backward()` の呼び出しは完全0件。
   - 二足歩行CPG、前庭反射、玩具接触判定、机上3Dメッシュ生成、Warp剛体カーネル、`ExternalTabletopARCWorkbench` の `step_physx=True` 物理描画は真正に動作。
5. **ステップ1（評価リーク・不当対照群・状態汚染の排除）完全完了**:
   - SHA-256グリッドハッシュ照合による本番Evaluation課題の100%遮断を実装。
   - Stage 3 無限進化ループでの本番スコア選抜を全廃し、Trainセット予測適応度・EFE効率選抜へ是正。
   - 対照群（NoPain Control, NoImagination, NoPlasticity）への重み・練習・遠征の等価付与、17倍代謝税撤廃により公平な対照実験設計を確立。
   - `stage2` における小脳重み、LIF閾値、シンボル語彙・共有接地、長期記憶、マクロ合成への `if self.plastic:` ガード徹底により評価時状態汚染を完全凍結。
   - `Phys_T5` の 3D 剛体倒壊タスク置換、プロシージャル生成器拡張、空スタブ解消を完了。
   - 独立敵対的サブエージェントによる全行再監査および動的ポイズン監査（`audit_anti_cheat.py`）により STATUS: PASS を確認。
6. **ステップ2（3D物理・運動学・身体性チートの排除）完全完了**:
   - `isaac_warp_3d_engine.py`: 勉強部屋入室の瞬間移動ワープ（C-01）を完全撤廃し、ドアから椅子まで CPG 連続歩行で到達。思考中（`step_physx=False`）のヤコビアン1次テイラー線形展開バイパス（C-02）を全廃し、非線形幾何順運動学（FK）モデルを通した到達判定へ置換。2Dセルオートマトン偽装（C-03）を全廃し、真正な Warp GPU 剛体接触シミュレータ（`rigid_body_3d_substep_kernel`）を起動。机上ターゲット盤面座標を肩追従からワールド固定机 `SafeRoomDesk` に厳密接地（H-05）。周辺視野レイへの光学ブラー適用と整数色タグ露出の排除（H-01）。シェーダ代数逆算カンニング（`z_mono`）を全廃し、左右眼の真のSSDコストボリューム相関による両眼視差深度再構成へ是正（H-04）。
   - `agi_sensorimotor_puzzle_mdp.py`: 玩具操作における75%チート吸引力ベクトル、浮揚力、直接PD回転トルク、正解属性直接コピーを全廃。手の接触剛体摩擦と法線抗力、ロボット接触速度のみによる物理操作に是正（H-02）。
   - `four_biome_humanoid_env.py`: マルチ環境での `env_0` ハードコードスライス（災害・障害物座標含む）を `active_env_id` に応じた個別動的テンソルスライスへ是正。食料・脅威マスクにおける色ID `8` 決め打ちを完全排除（H-03）。
   - 独立敵対的サブエージェント（`Adversarial Auditor Step 2`）による全行再監査および動的ポイズン監査（`audit_anti_cheat.py`）により STATUS: PASS を確認。
7. **ステップ3（脳構造フェイク・DSLルーター・手書き逆伝播の完全排除）完全完了**:
   - `unified_biological_brain.py`: 神の視点スキャンと色0決め打ちを自律背景色照合へ是正。旧DSL6種固定インデックスを全廃し、潜在変化ベクトルに対する競合的ヘブ則＋Oja正則化による自律創発アトラクタへ全面移行。`NeocortexSystem2Module.forward` を正式実装しミエリン化ゲート・想起逆投影（128次元整合）・マクロ変調・音響予測出力を感覚運動閉ループ制御に実効直結。`conv1`/`conv2` の局所逆投影予測誤差ヘブ則（`torch.einsum` による shape 完全整合）および前後ステップ遷移による真の時間発展世界モデル学習（$s_{t-1}, a_{t-1} \to h_t$）を配備。21自由度関節トルク出力 `joint_effort_21d` を正式配備。独立敵対的監査サブエージェント（`7115e398`）による全行再監査および実機テストにより STATUS: PASS（検出欠陥 0件、CRITICAL 0, HIGH 0）を完全確認。
8. **ステップ4（Stage 2 手書きARC記号DSL・$y$ 覗き見の完全撲滅）完全完了**:
   - `stage2_embodied_puzzle_solver.py`: `_OPPONENT_BG_SAL_LUT` の色中立化（0.50）、小脳前シナプス・登攀線維外積LTD更新、動的境界 `max_bound = max(H, W, 64)`、皮質潜在投射への直結（`_foveal_visual_buffer` ダミー全廃）、手書きDSL総当たり（約150行）完全消滅、正解 $y$ 画素直接引き算・除算の全廃、睡眠固定化ガード、首振り知覚反映、前頭前野内的思考フィルタ（枝刈り判定）、大脳基底核机上物理描画通過、サインポスト抽出破棄解消、作業記憶被覆率の注意ゲイン配線、小脳未使用変数整理、属性マッチングアフォーダンス配線、ゲシュタルト利得受容判定反映を完全達成。独立敵対的監査サブエージェント（`35b4f9be`）による全行再監査および実機テストにより STATUS: PASS（検出欠陥 0件、CRITICAL 0, HIGH 0）を完全確認。

---

## 6. 今ある問題点・課題一覧（サブエージェント検出の全41件）

### `[CRITICAL]` 致命的な違反（全11件）

| ID | 該当ファイル・行番号 | 違反分類 | 具体的な不正証拠 | 修正方針 |
| :--- | :--- | :--- | :--- | :--- |
| **C-01** | `isaac_warp_3d_engine.py`<br>`L850-L907, L983-L985`<br>`four_biome_humanoid_env.py`<br>`L896-L945` | 瞬間移動ワープ<br>物理バイパス | **【解決済】** 勉強部屋入室における瞬間移動ワープを完全撤廃。CPG歩行制御（`compute_spinal_cpg_and_vestibular_joints_21d`）とPDトルク、物理ステップ進行によるドアから椅子までの連続二足歩行により真に着座到達させる。 |
| **C-02** | `isaac_warp_3d_engine.py`<br>`L1131-L1202` | 線形代数ショートカット<br>物理バイパス | **【解決済】** `step_physx=False`（イメージ思考中）での固定ヤコビアン1個による1次テイラー線形展開到達判定を全廃。3節リンク幾何に基づく非線形幾何順運動学（FK）モデルを実装し、物理的に到達可能な手先座標のみを描画可能に是正。 |
| **C-03** | `isaac_warp_3d_engine.py`<br>`L1793-L1876` | ハルシネーション<br>2Dセルオートマトン偽装 | **【解決済】** 2Dセルシフト偽装を全廃。盤面ブロックを3D剛体として構築し、Warp GPU カーネル `rigid_body_3d_substep_kernel` を真に起動して重力・接触摩擦・押し出しから盤面変化を計算する真正な3D剛体シミュレーションへ置換。 |
| **C-04** | `unified_biological_brain.py`<br>`L372-L545` | 神の視点<br>色番号決め打ち | サッカード開始前の `L446` で全画素を一括スキャンして未固視前景数 `fg_count` を把握し、黒色 `0` にボーナスを与え、`np.argmax` で直接視点をワープさせている。 | 全体スキャンを廃止。低解像度の周辺視野（Peripheral Low-Res）受容野から得られる顕著性マップに基づき、上丘（Superior Colliculus）サッカード跳躍運動により視点を移動させる。 |
| **C-05** | `stage3_arc_benchmark_eval.py`<br>`L240-L246, L561-L610` | 不公平な対照実験設計<br>八百長 | **【解決済】** 対照群 `Disembodied_NoPain_Control` に対して、チェックポイント重み、3Dマニフォールド、自己対戦練習（Route 2）、夢リプレイ、外部遠征（Route 3）を等しく実行。評価時は両群とも `plastic=False` に統一し、身体性と痛覚の有無のみを唯一の独立変数とする公平な比較を確立。 |
| **C-06** | `stage1_isaac3d_embodied_arena.py`<br>`L436-L440, L739-L749`<br>`L1119-L1151` | 不公平な対照実験設計<br>人為的餓死誘導 | **【解決済】** `NoImagination` と `NoPlasticity` 群への重みロード剥奪を解消し、対照群にも等しく事前学習マニフォールド重みを付与。`overthink_unbounded` への17倍固定代謝税を撤廃し、スパイク密度に応じた正当な連続代謝コスト関数に統一。 |
| **C-07** | `million_scale_arc_harvester.py`<br>`L126-L178`<br>`stage3_arc_benchmark_eval.py`<br>`L850-L875` | 評価データリーク<br>Test-set Model Selection | **【解決済】** 全入出力グリッドの SHA-256 ハッシュ照合（`_collect_held_out_eval_data`）を実装し Eval 課題の混入を100%完全遮断。Stage 3 の無限進化ループにおいて本番 Evaluation スコアによる選抜を全廃し、Train セット正解数および自由エネルギー効率（EFE）による内部適応度選抜へ是正。 |
| **C-08** | `stage2_embodied_puzzle_solver.py`<br>`L2165-L2419` | 準ハードコード<br>正解覗き見DSL合成 | ミラーニューロン逆モデルが神経出力 `author_stroke_11` を破棄し、入力 $x$ と正解 $y$ の全画素差分から手書きARC記号DSL（`crop_grasped`, `tile_grasped` 等）を合成して上書き。 | 正解 $y$ の覗き見を完全排除。入力 $x$ と身体的アフォーダンスから予測された潜在キネマティクス運動のみを神経ネットワークで純粋に生成する。 |
| **C-09** | `stage2_embodied_puzzle_solver.py`<br>`L1189-L1612` | 準ハードコード<br>ハルシネーション | 中心窩閉ループ描画が網膜スパイク戻り値を変数に受け取らず破棄し、3種の手書きDSL（`component_anchor_stroke` 等）を総当たり実行。 | 網膜スパイク戻り値を第1脳・第2脳へ接続し、スパイク駆動の運動指令によって物理ストローク軌道を閉ループ生成する。手書きDSL総当たりを全廃する。 |
| **C-10** | `stage2_embodied_puzzle_solver.py`<br>`L2703-L3724, L3852-L4158` | 神の視点<br>巨大2D記号DSL | 統一感覚運動チャンクの実態が全画面連結成分・凡例パーサ・ARC上限サイズ `30` ハードコード・2Dスライス直接返却・正解 $y$ 由来候補の優先注入。 | 2Dスライス直接返却やサイズ30分岐を完全撤廃。玩具ブロック操作MDPで獲得した3D把持・整列・回転の物理プリミティブから運動ストロークを構成する。 |
| **C-11** | `stage2_embodied_puzzle_solver.py`<br>`L1003-L1031, L1714-L1720`<br>`L2092, L5441, L5983, L6654` | 評価時学習リーク<br>タスク間汚染 | **【解決済】** `stage2` の小脳プルキンエ重み、LIF閾値、シンボル語彙自己刻印・共有接地、自伝的長期記憶、再帰的チャンクマクロ、変異エピソードバッファへの更新に `if self.plastic:` ガードを徹底配備。評価中（`plastic=False`）は一切の重み・閾値・記憶バッファの書き換えを完全凍結。 |

---

### `[HIGH]` 重大度：高の違反（全14件）

| ID | 該当ファイル・行番号 | 違反内容 | 修正方針 |
| :--- | :--- | :--- | :--- |
| **H-01** | `isaac_warp_3d_engine.py`<br>`L66-L165, L521, L1552` | 周辺視野レイ整数タグ露出・ステレオ色ペナルティ | **【解決済】** 離心率に応じて周辺視野の整数色タグを `-1` にマスク消去し光学ブラーを適用。ステレオマッチングの整数色タグ不一致ペナルティ（`col_pen`）を全廃し、物理的な輝度・RGB反射値の差分二乗和のみで深度を推定。 |
| **H-02** | `agi_sensorimotor_puzzle_mdp.py`<br>`L711-L860` | 玩具操作チート吸引力・浮揚力・直接属性コピー | **【解決済】** 75%吸引力ブレンド、浮揚力（$f_z$）、直接PD回転トルク、正解属性コピーを全廃。押し方向を実ロボット手先線速度のみから算出し、手の接触摩擦・法線抗力、作業記憶を介した純粋な身体化接触力学操作に是正。 |
| **H-03** | `four_biome_humanoid_env.py`<br>`L1851-L1934, L2124, L2217` | `env_0` ハードコードスライス・擬態食料色ID 8 決め打ち | **【解決済】** `active_env_id` プロパティを導入し全オブジェクトおよび災害座標スライスを個別環境テンソルへ是正。食料・脅威マスクにおける色ID `8` 決め打ちを排除し環境個別食料色照合へ是正。 |
| **H-04** | `isaac_warp_3d_engine.py`<br>`L265, L2319-L2332` | シェーダ代数逆算カンニング・同一ピクセルステレオ | **【解決済】** シェーダ減衰式の代数逆算カンニング `(1/lum - 1)/0.06` および同一ピクセル差分を全廃。両眼カメラ基線長による真のSSDコストボリューム相関・二次曲線サブピクセル補間から三角測量深度を推定。 |
| **H-05** | `isaac_warp_3d_engine.py`<br>`L1107-L1117, L1225-L1226` | 机上ターゲット盤面の肩追従移動 | **【解決済】** 机上ストロークの描画ターゲットグリッド座標を、揺れる肩ではなくワールド固定机天板座標 `safe_room_desk_surface_world_xyz`（`board_center_local`）に厳密固定。 |
| **H-06** | `unified_biological_brain.py`<br>`L1700-L1854` | `torch.no_grad()` 内で多層誤差逆伝播を手書き密造。初段convは未学習ランダム固定、`W_world_forward` は次時刻ではなく同一時刻を学習。 | 手書き逆伝播を撤廃。局所予測誤差（Predictive Coding）と適格度トレースによる3因子局所ヘブ学習則に全面移行する。 |
| **H-07** | `unified_biological_brain.py`<br>`L641-L648, L1046-L1092` | 第2脳 `DRAWER_NAMES` が6種のARC専用DSLカテゴリ名をハードコードしワンホット教師あり学習。ニューラル倍率予測式がデッドコード。 | ARC専用DSLカテゴリ名を撤廃。自律創発する連続潜在トピックベクトルおよび連想アトラクタネットワークに移行する。 |
| **H-08** | `million_scale_arc_harvester.py`<br>`L297-L420` | 100万タスクと称しながら固定テンプレートの色違いしか生成しない。 | **【解決済】** 可変グリッド（7〜18）および身体運動プリミティブ（剛体押し出し移動、両側対称反転、中空充填、幾何重心整列、対称置換、重力沈降等）に基づく真のプロシージャル生成器へ拡張。 |
| **H-09** | `stage3_arc_benchmark_eval.py`<br>`L185-L215, L570` | 3D物理ベンチマーク `Phys_T5` が純2D配列。評価時 `plastic=True`。未実行の比較スコア `0.0` をハードコード出力。 | **【解決済】** `Phys_T5` を Warp 3D 剛体倒壊・平衡シミュレーションタスクへ置換。評価時の `plastic=False` 徹底および比較スコア捏造出力を全廃。 |
| **H-10** | `stage2_embodied_puzzle_solver.py`<br>`L993-L1035, L3599-L3701` | `execute_visuomotor_stroke_servo` が2D配列直接上書き、手先瞬間移動。先に2Dスライスで解を作って事後コピー。 | 事後コピー・瞬間移動を撤廃。手先のエンドエフェクタがキャンバス表面と物理接触（IK接触）した点のみに画素が定着する物理描画に一本化。 |
| **H-11** | `stage2_embodied_puzzle_solver.py`<br>`L1404-L1405, L1838` | `plastic=False` 時でもシナプス・発達段階を更新し、No-op時に過去タスクのDSLを無言で再生。 | **【解決済】** `_fallback_foveal_prototype_if_noop` の過去タスクDSL再生を廃止。`advance_developmental_critical_period` の評価時更新を凍結。 |
| **H-12** | `stage2_embodied_puzzle_solver.py`<br>`L36-L70, L115-L154, L4198` | ARC色 `0` の決め打ち優遇、全画素Flood-fillによる領域分割、感覚チャネル添字 `blended_obs[12]` のハードコード。 | 色 `0` 特例を撤廃。網膜受容野の輝度勾配・エッジ検出（V1 Gabor様特徴）から境界所有権（Border Ownership）を自律判定させる。 |
| **H-13** | `stage2_embodied_puzzle_solver.py`<br>`L4675-L5359, L6986` | 小脳エラー訂正、PFC GABAゲート、AChトポロジカルレンズ、サブゴールCPが正解 $y$ のピクセル差分から手書きDSLを逆算。 | 正解 $y$ の差分逆算を全面排除。内部モデルの予測感覚と実際の感覚の誤差（Sensory Prediction Error）に基づく純粋なフィードフォワード制御に置換。 |
| **H-14** | `stage2_embodied_puzzle_solver.py`<br>`L6905, L7243, L7247` | `perceive_synthesized_ast_affordances` が空スタブかつ引数不正。未実装セレクタでサイレントフォールバック。戻り値破棄。 | **【解決済】** `perceive_synthesized_ast_affordances` の引数シグネチャを整合し、`evolved_ast_organs` の関数を正当に返す実装へ置換。`step_sensorimotor_closed_loop` の戻り値破棄を解消。 |

---

### `[MEDIUM]`（全12件）および `[LOW]`（全4件）の概要

- **M-01**: `perceive_3d_tabletop_via_human_vision` が腰関節シフト時に `sim.step()` を呼ばず `measured_dx` が常に 0.0（0.045m定数フォールバック）。
- **M-02**: `obs_egocentric_*` が遮蔽を無視して不可視座標を露出、パズルMDPが内部正解進捗を常時露出。
- **M-03**: 看板距離判定が中心窩注視ゲートを上書き死文化。
- **M-04**: `kinematic_enabled=True` の渦中心に外力を加えており公転不発。
- **M-05**: 第3脳 `W_joint_21d` が forward や export から欠落、Stage 1 オンライン実行中の小脳更新欠落。
- **M-06**: カテゴリカルなARC色番号を `g / 9.0` として線形混色、スライス添字決め打ち。
- **M-07**: `stage4` の `predict_task` が独立した物理検証を持たず全面委譲。
- **M-08**: 第2脳の死蔵重み5種、自己循環ミラーターゲット、睡眠リプレイの前後シナプス不整合。
- **M-09**: 小脳登攀線維外積計算が自己外積減算で `W_cerebellar_purkinje` を単調崩壊。
- **M-10**: 11次元 `W_body_assembly` のエンコーダとデコーダで意味割り当てが完全矛盾。
- **M-11**: トポロジカルレンズの `"depth_25d"` が2D貫通バー補間、`"polar_centroid"` が2D位相破壊。
- **M-12**: `solve_task` 直接呼び出し時に物理描画を通らない2D配列が返る経路。
- **L-01**: `COOP_FOOD_ENERGY_GAIN` および `comm_enabled` の未使用。
- **L-02**: `glaciation_chaos` のカオス摂動不発、`max_depth` 未使用。
- **L-03**: **【解決済】** `perceive_core_affordances` の空スタブを身体運動幾何プリミティブ生成へ置換。`reinforce_intuition_from_ast_discovery` をタスク解決成功処理に正当接続。
- **L-04**: 固定乱数シード、添字上書き、未解決サブゴールのダミー正解、色0背景決め打ち。

---

## 7. 徹底浄化実行計画（全5ステップ・ロードマップ）

### ステップ1: 評価リーク・不当対照群・評価時状態汚染の完全排除 【完了・独立敵対的監査合格】
- **対象ファイル**: `stage3_arc_benchmark_eval.py`, `stage1_isaac3d_embodied_arena.py`, `million_scale_arc_harvester.py`, `stage2_embodied_puzzle_solver.py`
- **作業内容と達成状況**:
  1. `stage3`: **【完了】** 対照群 `nopain_solver` のハンディキャップ撤廃（同等の重み・マニフォールド・自己対戦練習・夢リプレイ・外部遠征を与え条件統一）。無限進化ループでの本番Evaluation正解スコアによる選抜を全廃し、Trainセット正解数および自由エネルギー効率（EFE）による内部適応度選抜へ是正。`Phys_T5` を純2D配列から Warp 3D 剛体倒壊シミュレーションタスクへ置換。評価時の `plastic=False` 徹底および比較スコア捏造出力を全廃。
  2. `stage1`: **【完了】** 対照群 `NoImagination`, `NoPlasticity` への重み剥奪を解消し公平にロード。`overthink_unbounded` への17倍固定代謝税を撤廃しスパイク密度に応じた正当な連続代謝コスト関数に統一。
  3. `million_scale_arc_harvester`: **【完了】** SHA-256 グリッドハッシュ照合による本番Evaluation課題の100%遮断を実装。固定テンプレートを身体運動プリミティブに基づく動的プロシージャル生成器へ拡張。
  4. `stage2`: **【完了】** 小脳プルキンエ重み、LIF閾値、シンボル語彙自己刻印・共有接地、自伝的長期記憶、再帰的チャンクマクロ、変異エピソードバッファへの更新に `if self.plastic:` ガードを徹底配備。評価時完全フリーズを達成。`perceive_core_affordances` 空スタブ解消、`perceive_synthesized_ast_affordances` 引数整合、閉ループ戻り値受取を完了。
  5. 独立敵対的サブエージェント（`Adversarial Code Auditor`）による全行再監査および `audit_anti_cheat.py`（動的ポイズン監査）により STATUS: PASS を確認。

### ステップ2: 3D物理・運動学・身体性チートの完全排除 【完了・独立敵対的監査合格】
- **対象ファイル**: `isaac_warp_3d_engine.py`, `agi_sensorimotor_puzzle_mdp.py`, `four_biome_humanoid_env.py`
- **作業内容と達成状況**:
  1. `isaac_warp_3d_engine`: **【完了】** 勉強部屋入室の瞬間移動ワープ（C-01）を完全撤廃し、ドアから椅子まで CPG 連続歩行で到達。思考中（`step_physx=False`）のヤコビアン1次テイラー線形展開バイパス（C-02）を全廃し、非線形幾何順運動学（FK）モデルを通した到達判定へ置換。2Dセルオートマトン偽装（C-03）を全廃し、真正な Warp GPU 剛体接触シミュレータ（`rigid_body_3d_substep_kernel`）を起動。机上ターゲット盤面座標を肩追従からワールド固定机 `SafeRoomDesk` に厳密接地（H-05）。周辺視野レイへの光学ブラー適用と整数色タグ露出の排除（H-01）。シェーダ代数逆算カンニング（`z_mono`）を全廃し、左右眼の真のSSDコストボリューム相関による両眼視差深度再構成へ是正（H-04）。
  2. `agi_sensorimotor_puzzle_mdp`: **【完了】** 玩具操作における75%チート吸引力ベクトル、浮揚力、直接PD回転トルク、正解属性直接コピーを全廃。手の接触剛体摩擦と法線抗力、ロボット接触速度のみによる物理操作に是正（H-02）。
  3. `four_biome_humanoid_env`: **【完了】** マルチ環境での `env_0` ハードコードスライス（災害・障害物座標含む）を `active_env_id` に応じた個別動的テンソルスライスへ是正。食料・脅威マスクにおける色ID `8` 決め打ちを完全排除（H-03）。
  4. 独立敵対的サブエージェント（`Adversarial Auditor Step 2`）による全行再監査および動的ポイズン監査（`audit_anti_cheat.py`）により STATUS: PASS を確認。

### ステップ3: 脳構造フェイク・DSLルーター・手書き逆伝播の完全排除 【完了・独立敵対的監査完全合格】
- **対象ファイル**: `unified_biological_brain.py`
- **作業内容と達成状況**:
  1. 第1脳: サッカード前の神の視点全グリッドスキャンと `argmax` 直接ワープを撤廃。自律背景色照合（`!= bg`）を徹底し、一次視覚野境界所有権顕著性マップ `salience_map` に基づく上丘サッカード跳躍運動へ置換完了。
  2. 第2脳: `DRAWER_NAMES` 6種ARC専用DSLハードコード分類を完全撤廃。潜在変化ベクトル $d_t$ に対する勝者総取り競合的ヘブ則＋Oja正則化による自律創発トピックアトラクタ更新へ全面移行完了。引数 `winning_drawer_name` の実効判定分岐配備完了。手書きDSLキー全廃、連続潜在アトラクタ特徴量 `latent_topic_features` へ統一。
  3. 大脳新皮質順伝播と死蔵パラメータ統合: `NeocortexSystem2Module.forward` を正式実装。ミエリン化ゲート、手続き記憶アトラクタ（想起信号からの潜在逆投影 `chain_fb = torch.tanh(chain_slot @ self.W_procedural_attractor)` による128次元整合）、マクロ・モノアミン変調、音響予測出力 `W_sound_pred` を順伝播に完全統合。`step_sensorimotor_closed_loop` の感覚運動閉ループ制御と大脳新皮質順伝播を実効直結。
  4. 視覚野・運動野ヘブ学習則: `conv1`/`conv2` において、`F.conv_transpose2d` による上位逆投影予測と下位層活性の局所予測誤差（`err_c1`, `err_vp`）に基づく局所3因子ヘブ則＋Oja正則化更新（`torch.einsum` による shape 完全整合）を配備完了。
  5. 時間発展世界モデル: `W_world_forward` の学習において、`torch.roll` 異個体ワープおよび自己恒等写像を完全撤廃。前ステップの感覚運動特徴 $s_{t-1}, a_{t-1}$（`_prev_arena_joint_wm`）から現ステップの脳状態 $h_t$ を予測する真の時間発展予測誤差学習へ是正完了。
  6. 運動反射モジュール: `MotorReflexModule.forward`、`infer_motor_policy`、`export_onnx`、および `step_sensorimotor_closed_loop` 返却辞書に `joint_effort_21d`（21自由度関節トルク出力）を正式配備完了。`export_onnx` の到達不能重複コードを完全削除。
  7. 独立敵対的監査（サブエージェントA: `cd9ea58f` / `7115e398`）による全行再監査および動的実行検証により STATUS: PASS（検出欠陥 0件、CRITICAL 0, HIGH 0）を完全確認。

### ステップ4: Stage 2 手書きARC記号DSL・$y$ 覗き見の完全撲滅 【完了・独立敵対的監査完全合格】
- **対象ファイル**: `stage2_embodied_puzzle_solver.py`
- **作業内容と達成状況**:
  1. H-12: 色0背景決め打ちボーナス（+0.45）撤廃、添字12ハードコード撤廃、`_OPPONENT_BG_SAL_LUT` の色中立化（全色一律0.50）を完全達成。
  2. M-09: 小脳登攀線維外積自己減算による単調崩壊を前シナプス・登攀線維外積 `-0.02 * np.outer(corr_vec, cf_err) + 0.005 * np.eye(6)` へ是正完了。
  3. C-10 / H-10: サイズ 30 ハードコードを動的境界 `max_bound = max(H, W, 64)` に置換。境界逸脱時の 2D スライス直接 return（`return sub_g`）や事後コピーを完全排除し、物理キャンバス＋ストロークサーボ返却に統一。
  4. C-09: `_foveal_visual_buffer` ダミー退避リストを全廃し、大脳新皮質受動接続重み `W_fovea_v1_spikes` を介した皮質潜在投射 `last_foveal_cortical_drive` へ直結完了。
  5. C-08: `learn_and_imitate_foveal_stroke` 内の手書きDSL総当たり・正解一致チートコード（約150行）を完全消滅。顕著性中心窩注視とスパイク符号化、側抑制ヘブ則によるシナプス刷り込み機構のみへ純化。
  6. H-13: 顕著性抽出および前頭前野GABAゲートでの正解 $y$ 画素直接引き算・除算（`salience_map += 4.0`, `corrupted_ratio` 等）を完全排除。自由エネルギー秩序度と侵害受容痛のみによるGABAゲート制御へ是正。
  7. 生物学的回路健全化: 睡眠固定化に `if not self.plastic: return` ガード配備完了。首振り視点変更（`shift_3d_head_tilt_viewpoint`）の再投影グリッドを現在知覚グリッドへ正当反映完了。`simulate_latent_pfc_rollout` 戻り値を抑制性内的思考フィルタ（枝刈り判定）へ実効接続完了。大脳基底核反射（`execute_basal_ganglia_reflex`）において `wb_ref` が存在する場合に机上物理描画（`wb_ref.manipulate_3d_working_canvas`）を通過するよう配備完了。
  8. 計算破棄・未使用変数整理: サインポスト凡例抽出の未使用 `extract_components` 呼び出し削除、外部作業記憶被覆率 `wm_coverage` の注意ゲイン動的配線、小脳未使用変数整理、基底核属性マッチングフラグ `has_group_chunk` のアフォーダンス生成配線、夢リプレイの未使用 `var_q` 削除、ゲシュタルト利得 `gestalt_gain_step` の受容判定反映完了。
  9. 独立敵対的監査（サブエージェントB: `caa79036` / `35b4f9be`）による全行再監査および動的実行検証により STATUS: PASS（検出欠陥 0件、CRITICAL 0, HIGH 0）を完全確認。

### ステップ5: 一気通貫スモークテスト & エンドツーエンド検証 【完了・全ステージ合格】
- **対象ファイル**: `scratch/smoke_test_end_to_end.py`, `artifacts/smoke_test_results.json`
- **作業内容と達成状況**:
  1. フェーズAの最終検証として、全4ステージの一気通貫パイプライン（4バイオーム生存・玩具操作アリーナ $\to$ 安全学習室歩行・着座 $\to$ 机上ARCパズル学習・推論 $\to$ 睡眠固定化・オフライン夢リプレイ $\to$ 3脳重み保存・リロード）を構築・実行。
  2. 実機実行検証（実行時間 44.08秒）により、全ステージのノーエラー完走および完全合格（`ALL_PASS`）を確認。

### ステップ6: フェーズB突入前 3ジャンル総点検・完全是正 【全3ジャンル完全完了・独立敵対的監査完全合格】

#### ジャンル1: 3D身体・物理環境・運動学 【完全完了・独立敵対的監査完全合格 (STATUS: PASS)】
- **対象ファイル**: `isaac_warp_3d_engine.py`, `four_biome_humanoid_env.py`, `agi_sensorimotor_puzzle_mdp.py`
- **作業内容と達成状況**:
  1. `isaac_warp_3d_engine`: `perceive_3d_tabletop_via_human_vision` における `env_i` 未定義例外の完全解消、環境IDクランプ、`env_origins[env_i]` および `body_pos_w[env_i]` への動的配線完了。歩行・着座・机上ストロークにおける `env_0` 固定スライス完全撤廃。シミュレータ非存在時の座標直接加算ワープ完全排除。能動的運動視差測定における `sim.step()` および姿勢復元ステップの正式配備。机上Warp剛体シミュレータの鉛直重力軸（y軸）と床接触判定の整合。`overall_min_dist` 最小値追跡の正常化。未参照変数 `_desk_board_rel_sh_ref` 削除。
  2. `four_biome_humanoid_env`: 未使用メソッド `_env0_origin` の完全削除。
  3. `agi_sensorimotor_puzzle_mdp`: 存在しない属性 `agent_gaze_dir_w` アクセス（ハルシネーション）を完全排除し、注視対象 `slot_pos_w` とエージェント頭部位置の差分ベクトルからの直接算出へ是正。デッドコード（`cur_block_pos`, `i_zz`, 破棄される `best_triple`）の完全除去。カテゴリカル色番号のスカラー除算 `/ 9.0` を反対色連続ビン正規化へ是正。`VocalSignal2DAction.apply_actions` の空スタブ解消。
  4. 独立敵対的監査サブエージェント（`6f9cdc85`）による全行再精読検証により **STATUS: PASS（全検証項目合格・欠陥ゼロ）** を完全確認。

#### ジャンル2: 脳神経構造・可塑性・能動的推論 【完全完了・独立敵対的監査完全合格 (STATUS: PASS)】
- **対象ファイル**: `stage1_isaac3d_embodied_arena.py`, `unified_biological_brain.py`
- **作業内容と達成状況**:
  1. `stage1_isaac3d_embodied_arena`: ニューロン固有適応閾値 `self.theta` を動的ベース閾値 `v_th` と統合した `effective_theta = torch.clamp(v_th * (self.theta / 0.35), 0.15, 0.95)` を構築し、膜電位発火判定（`lif_spike`）、クエリ・キースパイク判定（`q_lin.abs() > effective_theta`, `k_lin.abs() > effective_theta`）、および能動的推論ロールアウト（`s_next`）へ完全配線（内在的恒常性可塑性の実効化）。
  2. `stage1_isaac3d_embodied_arena`: ステップ進行中の個体死亡再生時に Spiking Transformer の核心であるアテンション重み `Wq, Wk, Wv`、モノローグ重み `W_macro_out, W_mono_out`、および閾値 `theta` がドナーから微小変異を伴って正当に継承されるよう配備（適応構造の進化的蓄積）。
  3. `stage1_isaac3d_embodied_arena`: 死蔵テンソル `Wo_base, Wh_base` を完全削除。
  4. `unified_biological_brain`: `NeocortexSystem2Module.forward` において引数 `modulatory_mono` を新設し、モノアミン受容体結合重み `W_mono_in`（128, 4）を順伝播状態更新に正式配線。
  5. `unified_biological_brain`: 手続き連鎖重み `W_procedural_chain`（16, 33）の全結合活用、および発達臨界期ゲート `myelination_stage_gates` の動的インデックス選択を配備。
  6. `unified_biological_brain`: `step_sensorimotor_closed_loop` 返却辞書に `macro_monologue_4`, `monoamine_dynamics_4`, `sound_pred_11` を配備し破棄を解消。死蔵属性 `_prev_arena_h_tensor` および不要な `pos_np` CPU転送を完全除去。
  7. 独立敵対的監査サブエージェント（`338bc043`）による全行再精読検証により **STATUS: PASS（全検証項目合格・欠陥ゼロ）** を完全確認。

#### ジャンル3: 机上認知・物理描画・睡眠固定化・評価 【完全完了・独立敵対的監査完全合格 (STATUS: PASS)】
- **対象ファイル**: `stage2_embodied_puzzle_solver.py`, `million_scale_arc_harvester.py`, `stage3_arc_benchmark_eval.py`
- **作業内容と達成状況**:
  1. `stage2_embodied_puzzle_solver`:
     - CRITICAL-1: `compute_active_inference_ast_drive` の引数シグネチャ不整合（`y_tr` 誤渡しによる TypeError バグ）を解消し、探索ループ直前（L6888）で実際に呼び出して能動的推論ASTドライブベクトル `ast_drive_vec` を生成する実効インターフェースとして配備。
     - CRITICAL-2 & MEDIUM-5: 未解決残余サブゴール収穫処理に `if self.plastic and harvest_prog and len(train_pairs) >= 2:` ガードを配備し、評価モード時のデータリーク・タスク間状態汚染を完全遮断。テストペアのダミー出力混入を解消。
     - CRITICAL-3: 11次元身体運動アセンブリ重み `W_body_assembly` の意味空間矛盾（注視・記憶・指 vs dy, dx, rot, flip）を完全解消。デコード側を真の身体操作ステップ（gaze_op, wm_op, finger_op, dy, dx）へ是正し、エンコード・共鳴側と完全一致・再統合。
     - HIGH-1 & HIGH-3: 遺伝的AST・コドン合成・マイクロプリミティブ機能群（`evolved_ast_organs`, `de_novo_codon_pool`）を実体化。タスク成功時に `synthesize_and_register_evolved_organ` で自動蓄積し、`perceive_synthesized_ast_affordances` において探索候補として展開・供給。未配線テレメトリ10種を実処理に完全接続。
     - HIGH-2: 全リターン分岐（基底核反射3箇所、キャッシュ完全一致、探索ループ内完全一致、サブゴールチェックポイント完全一致2箇所）において、旧式の見せかけ判定（ステップ数>=2等）および True ハードコードを完全一掃し、実際のプログラムプレフィックス走査（`replay_`, `macro_`, `dream_`, `cultural_`, `macro_breed_`, `nested_tool_`, `mono_`, `cond_` 等）および実体フラグに基づく厳密な判定へ統一。
     - MEDIUM-1: `perceive_synthesized_ast_affordances` に網膜スパイク履歴への自動フォールバックを配備し、`q_spk_hint=ast_drive_vec` からの `W_ast_intuition` 内積スコアリングにより候補アフォーダンス列をスコア降順で整列。推論時読み出しを完全実体化。
     - MEDIUM-2: カテゴリカル色番号のスカラー除算 `/ 9.0` および線形減算を完全撤廃。反対色連続ビン表現およびブール一致度へ是正。
     - MEDIUM-3: トポロジカルレンズ `depth_25d` の遮蔽スパン制限（厚さ<=4）および `polar_centroid` のチェビシェフ同心リング×方位角ソートによる位相保持へ是正。
     - MEDIUM-4: `ExternalTabletopARCWorkbench` の色0背景決め打ちを外周最頻値の動的背景検出へ是正。
     - LOW-1, LOW-2, LOW-4: 固定シード（2026, 1337）の動的シード化、`execute_visuomotor_stroke_servo` の `bg_color=None` 自動背景検出化、音声合成戻り値の聴覚フィードバック保持。
     - DEAD_CODE-1, 3, 4: 未使用メソッド `score_foveal_rule_intuition`、死蔵キャッシュ辞書群、死蔵属性変数群（`_active_chaos_ops`, `seed`, `single_organism_mode`, `safe_room_desk_xyz`）を完全削除。
  2. `million_scale_arc_harvester`:
     - HIGH-4: `sample_relational_combinatorial_task` においてルールパラメータをループ外で決定し、少数例示（few-shot）一貫性規則を厳密に保証。
     - MEDIUM-6: `include_eval` 引数および評価データ取り込みブロックを完全削除し、SHA-256 ハッシュ照合により本番Evaluationデータの混入を100%遮断。
     - DEAD_CODE-6: 未使用メソッド `_collect_held_out_eval_ids` を完全削除。
  3. `stage3_arc_benchmark_eval`:
     - HIGH-5: `run_procedural_tabletop_practice` において、`is_chaos` 時に各 train グリッドに真の局所カオス摂動（1セル反転）を印加する実効ストレステストへ是正。
     - LOW-3: `max_depth` 引数を `solver.max_program_depth` に正式接続。
     - DEAD_CODE-5: 未使用定数（`DATA_STRESS_V3_DIR`, `DATA_NOVEL_V2_DIR`）および未使用乱数生成器を削除。
  4. 独立敵対的監査サブエージェント（`45a34b29`）による全行再精読検証により **STATUS: PASS（全検証項目合格・欠陥ゼロ）** を完全確認。

---

## 8. プロジェクト更新・運用規定（本文書の更新ルール）

1. **更新義務の原則**:
   - コード修正、バグ改修、機能追加、リファクタリング、マイルストーン移行のいずれかが発生するたびに、作業完了ターン内で**必ず本文書（`ENGINE_PHILOSOPHY_AND_STATUS.md`）の該当セクションを更新**しなければならない。
2. **コンテキストクリア（`/compact`）時の復帰プロトコル**:
   - 新規会話セッション開始時またはコンテキスト圧縮の直後、エージェントは**他の作業に先立ち本文書を `read_file` で読み込まなければならない**。
   - 本文書の内容を完全に頭に入れた上で、理念と規約に一切の矛盾がない状態で次のタスクへ進む。
3. **課題のステータス管理**:
   - セクション6の課題が解決された場合、完了マーク・解決コミット・解決内容を明記し、未解決課題リストを常に最新の状態に保つ。
