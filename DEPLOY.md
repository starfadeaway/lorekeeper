# Docker 部署说明

> 目标：一条命令起 API、MySQL、前端 Nginx；数据落盘可持久化；本地/内网可用，不直接暴露公网。
>
> 实测环境：Ubuntu 24.04.4 LTS、Docker 29.1.3、Docker Compose 2.40.3。

## 1. 架构

```text
浏览器
  └─ web: nginx:80（宿主机 8080）
       ├─ /       → frontend/dist
       └─ /api/   → api:8000（FastAPI，SSE 关闭 buffering）
                      ├─ MySQL（会话/轨迹/评测）
                      └─ ./data（LanceDB 索引 + 原始语料）
```

- `web`：Nginx 托管前端构建产物，反代 `/api`。
- `api`：FastAPI + 手写 ReAct + 检索，非 root 运行。
- `mysql`：MySQL 8.0，数据在命名卷 `mysql_data`。
- `data`：宿主机 `./data` 绑定到容器 `/data`，存 LanceDB/向量/评测产物。

## 2. 文件

```text
backend/Dockerfile          # Python 3.13 多阶段构建
frontend/Dockerfile         # Node 22 build + Nginx runtime
frontend/nginx.conf         # SPA + /api 反代 + SSE 配置
docker-compose.yml          # api / mysql / web
.dockerignore               # 排除 .env、data、node_modules 等
.env                        # 运行时密钥，不入库、不打进镜像
```

## 3. 前置条件

```bash
docker version
docker compose version
```

需要 Docker Engine + Compose v2。
如果 `docker pull` 报 DNS/连接错误，先解决镜像源；本项目实测使用：

```json
{
  "registry-mirrors": ["https://docker.m.daocloud.io"]
}
```

写入 `/etc/docker/daemon.json` 后：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

## 4. 准备配置

```bash
cd ~/lorekeeper
cp .env.example .env
nano .env
```

至少填：

- `DEEPSEEK_API_KEY`（主聊天）
- `SILICONFLOW_API_KEY`（embedding / rerank / judge）
- 可选 `MINIMAX_API_KEY`

收紧权限：

```bash
chmod 600 .env
```

`.env` 不会进镜像：`.dockerignore` 已排除，Dockerfile 也没有 `COPY . .`。

## 5. 构建与启动

```bash
docker compose build
docker compose up -d
docker compose ps
```

期望：

```text
lorekeeper-api-1     Up (healthy)   0.0.0.0:8000->8000/tcp
lorekeeper-mysql-1   Up (healthy)   3306/tcp
lorekeeper-web-1     Up             0.0.0.0:8080->80/tcp
```

## 6. 验证

```bash
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool | head -n 15
curl -s -o /dev/null -w "web=%{http_code}\n" http://127.0.0.1:8080/
curl -s http://127.0.0.1:8080/api/health | python3 -m json.tool | head -n 8
```

期望：

- `ok: true`、`ready: true`；
- `mysql.state: ready`；
- `games` 含 `sr` / `ys` / `zzz`；
- `web=200`；
- 通过 8080 也能拿到 health（说明 Nginx 反代正常）。

浏览器：

```text
http://127.0.0.1:8080
```

从 Windows 访问：

```bash
hostname -I
```

```text
http://<VM的IP>:8080
```

如果 NAT 下打不开，把 VM 网络改成 Bridged，或在 VMware NAT 做 8080/8000 端口转发。

## 7. 数据持久化

- 索引/语料：宿主机 `./data` → 容器 `/data`
- MySQL：命名卷 `mysql_data`

验证：

```bash
docker compose down
docker compose up -d
docker compose ps
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool | head -n 8
```

`down` 只删容器/网络，保留 `./data` 和 `mysql_data`；重启后历史不丢。

备份：

```bash
tar -czf data-backup-$(date +%F).tgz data
docker compose exec mysql sh -c 'mysqldump -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE"' > mysql-backup-$(date +%F).sql
```

## 8. 常用命令

```bash
cd ~/lorekeeper

docker compose ps                 # 状态
docker compose logs -f api        # API 日志
docker compose logs -f web        # Nginx 日志
docker compose restart api        # 只重启 API
docker compose up -d --build      # 改代码后重建
docker compose down               # 停止（保留数据）
```

清理未使用镜像/构建缓存（可选）：

```bash
docker image prune -f
docker builder prune -f
```

## 9. 排错

### 9.1 `docker pull` 失败

- `registry-1.docker.io` 解析到 `157.240.x.x` / `2a03:2880...`：DNS 污染，直接配镜像源；
- 配完看：
  ```bash
  docker info | grep -A2 'Registry Mirrors'
  ```

### 9.2 `api` 一直 unhealthy

```bash
docker compose logs --tail=100 api
```

重点看：

- MySQL 认证失败：`.env` 的 `MYSQL_PASSWORD` 与 compose 覆盖值是否一致；
- `/data` 权限：容器内 UID 10001；
  ```bash
  sudo chown -R 10001:10001 data
  ```
- 索引缺失：`docker compose exec api ls -la /data/lance`

### 9.3 页面 502

```bash
docker compose ps
docker compose logs --tail=50 api
docker compose logs --tail=50 web
```

`web` 的 `/api/` 反代目标是 `api:8000`；`api` 不健康时 Nginx 会 502。

### 9.4 SSE 不流式

`frontend/nginx.conf` 已配置：

```nginx
proxy_buffering off;
proxy_cache off;
proxy_read_timeout 3600s;
```

如果改过，确认这三项还在。

### 9.5 端口冲突

`docker compose ps` 若提示 8000/8080 被占用，改 `docker-compose.yml` 的 `ports` 左侧，例如：

```yaml
ports:
  - "18000:8000"
```

## 10. 安全边界

- `.env` 不入库、不打进镜像；`chmod 600`；
- 不要 `COPY . .`，不要删除 `.dockerignore`；
- 不要把 8000/8080 直接暴露公网；当前没有鉴权/TLS；
- Key 一旦出现在日志/聊天/截图里，视为泄露，应轮换；
- 生产化再加：TLS、鉴权、限流、日志采集、备份任务、监控告警。

## 11. 已知限制

- **单 API 副本**：LanceDB 和 BM25 是本地索引，不能直接 `--scale api=N`；要水平扩展需换共享向量库/检索服务。
- **无 Alembic**：建表靠启动时 `init_db`，生产化应补迁移。
- **无 OTel/Prometheus**：目前只有 `/api/health` 和日志。
- **镜像较大**：LanceDB/numpy 等占空间；可后续做更细的 runtime 依赖拆分。

## 12. 最短验收清单

```text
√ docker compose ps：api/mysql healthy，web Up
√ /api/health：ok=true, ready=true, mysql.ready
√ games：sr/ys/zzz 都在
√ http://127.0.0.1:8080：200
√ 页面上问一句，能流式回答、引用可点
√ docker compose down && up -d：LanceDB 索引与 health 恢复（MySQL 历史会话未逐条抽查）
√ 镜像内无 /app/.env
```

> 验收环境：Ubuntu 24.04.4 LTS / Docker 29.1.3 / Compose 2.40.3，2026-11-07 在 VMware VM 实跑；不是 Windows 本机模拟。
> 第 6 项验证了容器重建后索引仍能加载、health 仍 ready、问答仍可用；MySQL 历史会话没有逐条抽查，因此不写成「历史消息逐条核过」。
