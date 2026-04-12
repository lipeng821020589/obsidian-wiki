# Harness 工程调研

> AI Agent 测试与执行框架

## 1. 什么是 Harness？

**Harness** 在 AI 领域指：
- 测试执行框架
- Agent 运行环境
- 状态管理与追踪

## 2. 主流 Harness 框架

### Claude CLI Harness
```bash
# 安装
brew install anthropic/cli/claude

# 运行测试
claude test run ./tests/
```

### OpenAI Codex Harness
```bash
# 安装
npm install -g @openai/codex

# 运行
codex test --suite suiteName
```

### OpenClaw ACPX Harness
```bash
# 启动 ACPX 会话
openclaw sessions spawn --runtime acp
```

## 3. 核心功能

| 功能 | 说明 |
|------|------|
| **测试执行** | 运行断言、评估结果 |
| **状态管理** | 跟踪对话上下文 |
| **会话追踪** | 记录完整交互 |
| **结果评估** | 打分、报错、报告 |

## 4. 实现原理

```
┌─────────┐
│ Harness ├─► Test Runner
├─────────┤
│ State   ├─► Context Manager  
├─────────┤
│ eval    ├─► Result Assessor
└─────────┘
```

### 核心接口
```typescript
interface Harness {
  run(tests: Test[]): Promise<TestResult>
  track(state: State): void
  evaluate(output: string): Score
}
```

## 5. 实践应用

### 本地开发
```bash
# 初始化
harness init my-project

# 运行测试
harness run

# 调试
harness debug --verbose
```

### CI/CD 集成
```yaml
# .github/workflows/test.yml
- name: Run Harness Tests
  run: harness run --reporter json > results.json
```

## 6. 参考

- https://github.com/nousresearch/hermes-agent
- https://docs.openclaw.ai
- https://github.com/Anthropic/claude-code