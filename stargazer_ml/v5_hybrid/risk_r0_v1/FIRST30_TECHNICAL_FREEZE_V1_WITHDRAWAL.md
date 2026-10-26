# 首30码入口技术冻结v1撤回（未取数）

本轮工程最终复核发现，入口main的最外层异常统一输出network_started=false，即使某些授权后落盘异常已经发生过请求，也可能误报零网络。这是错误会计语义，不能依赖使用者猜测。

修正：未知外层异常输出network_started=null与证据目录存在状态，要求核对claim/attempt账本。未授权check/fetch的正常拒绝分支仍明确network_requests=0，且不构造Batch/opener。新增反例测试只模拟IO异常，不取数。

第一技术manifest `contracts/r0_first30_fetch_manifest_v1.json`（SHA256 0649f3c3676681091eaac0c2c3b87ec03d54374ba81f0fea661ba8d443cde6f1）、草案授权v1和验收v1均保留，**不是现行执行依据**。原入口/测试/冻结器/验收器源码保存在 `_first30_entry_freeze_v1_withdrawn/`，供核对旧指纹；原代码路径已更新，不能把旧manifest与新代码混用。

当前入口仍expansion_first30_v1.py（接口/批次scope不变），但绑定新manifest_v2、草案授权_v2和验收_v2；必须单独批准新三哈希后才可取数，旧草案未签、不创建生产目录。请求清单30码/8MiB响应/12MiB落盘不变，0重试，未发生任何真实取数或批次“烧毁”。
