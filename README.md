# Fan-UI Gin Starter

这是一个基于 Gin 的成熟项目骨架，包含：
- 分层结构（router/handler/service/model）
- 统一日志（slog JSON）
- 请求追踪（Request ID）
- 健康检查与就绪检查
- 优雅关停

## 目录结构

```
cmd/server            # 入口
configs               # 配置文件
internal/config       # 配置加载
internal/handler      # HTTP 处理器
internal/middleware   # 中间件
internal/model        # 领域模型
internal/router       # 路由
internal/service      # 业务服务
pkg/logger            # 日志封装
```

## 快速开始

```bash
cp configs/config.yaml configs/local.yaml
APP_CONFIG=configs/local.yaml go run ./cmd/server
```

## 接口示例

- `GET /healthz`
- `GET /readyz`
- `GET /api/v1/users/:id`

