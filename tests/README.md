# tests/

离线断言套件。**不联网、零花费**，跑完约 30 秒。

```bash
python tests/run_all.py              # 全部 586 项，汇总在最后
python tests/run_all.py -k phase     # 只跑文件名含 phase 的
python tests/run_all.py --list       # 只列脚本，不执行
python tests/verify_phase.py         # 单个脚本也能直接跑
```

## 这里为什么不是 pytest

`requirements.txt` 里没有 pytest，这是故意的。500 多项断言按**流水线阶段**拆成 12 个
独立脚本，每个自带用例计数（`结果：N 通过 / M 失败`）与退出码，能单独跑、能一键汇总。
转成 pytest 只能换来一个"pytest"命令名，却要重写全部用例并引入 fixture 机制，不划算。

要加断言：复制任意一个 `verify_*.py` 的骨架（`check()` + 末尾 `raise SystemExit(main())`），
不用理解 fixture、conftest、plugin。

## 每个脚本守什么

| 脚本 | 项数 | 守的东西 |
|---|---|---|
| `verify_phase.py` | 127 | 六态阶段机、三个停点、接口契约、前端结构、重出/停止/删除 |
| `verify_judge.py` | 65 | 提示词拼装顺序、超长截断保护、场外人物预检、主体句念法 |
| `verify_pipeline.py` | 69 | S1→S6 全链路契约与状态迁移 |
| `verify_ref_desc.py` | 59 | 参考图反推角色卡：解析、覆盖规则、缓存、换图重推 |
| `verify_delivery.py` | 40 | 交付物完整性（成片、报告、账目、字幕） |
| `verify_docker.py` | 41 | Dockerfile 四条硬要求（字体、ffmpeg、不装浏览器、双卷） |
| `verify_providers.py` | 38 | 四个 provider 的构造契约与错误分类（不发真实请求） |
| `verify_state.py` | 37 | SQLite 状态机：迁移合法性、reconcile、并发 |
| `verify_web.py` | 35 | Web 接口与视图契约 |
| `verify_review.py` | 26 | 人工裁决与降级链路 |
| `verify_models.py` | 26 | Pydantic Schema 的边界（必填、枚举、语法定时） |
| `verify_audio.py` | 23 | 字幕切分、TTS 音色映射、音轨编排 |
| `check_ui.py` | — | 前端结构：孤儿 id、未定义函数、标签配对（无计数，只报问题） |

## fixtures/

`fixtures/m0_eval/` 是 **M0 判定阈值的标定数据集**，不是测试用例：

- `pairs/` 30 张对照图（同人 15 对 + 异人 15 对），用来标定 75 分那条通过线
- `raw_results*.json` 各模型的判定原始输出（qwen3-vl-plus、豆包 turbo 等）
- `PROMPTS.md` 标注口径与偏差登记
- `report.md` 标定结论（`tools/m0_calc.py` 生成）

它占 89MB，是「为什么阈值是 75 而不是 85」的唯一依据，所以跟着仓库一起发布。
只想跑断言不需要它——`verify_*.py` 里没有一条依赖这些图片。
