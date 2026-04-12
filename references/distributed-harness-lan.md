# 局域网主从连接通信详解

> 如何在 Termux 环境中搭建多手机 Harness Agent 集群

## 1. 通信架构设计

### 1.1 网络拓扑

```
┌────────────────────────────────────────────────────────────┐
│                     局域网 (LAN)                         │
│                   192.168.1.0/24                         │
│                                                            │
│   ┌──────────────┐        ┌──────────────┐               │
│   │  主手机     │        │  从手机 A   │               │
│   │ Master     │◄──────►│ Worker 1   │               │
│   │ :8080     │  TCP   │  :8081    │               │
│   └──────────────┘        └──────────────┘               │
│           │                        │                       │
│           │                ┌──────────────┐              │
│           └─────────────►│  从手机 B   │              │
│                         │ Worker 2    │              │
│                         └──────────────┘              │
└────────────────────────────────────────────────────────────┘
```

---

## 2. 连接方式对比

### 方案对比

| 方式 | 延迟 | 复杂度 | 可靠性 | 推荐场景 |
|------|------|--------|--------|----------|
| **WebSocket** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 实时通信 |
| **HTTP REST** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 简单任务 |
| **MQTT** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 消息队列 |
| **SSH Tunnel** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 端口转发 |

---

## 3. 实施方案一：WebSocket (推荐)

### 3.1 主手机 (Server)

```bash
# 安装 Node.js
pkg install nodejs

# 创建 server 目录
mkdir ~/acpx-server && cd ~/acpx-server

# 初始化项目
npm init -y
```

```javascript
// server.js
const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 8080 });

// 注册的从手机
const workers = new Map();

console.log('[Master] Server started on port 8080');

wss.on('connection', (ws, req) => {
  const clientId = `worker-${Date.now()}`;
  workers.set(clientId, { ws, status: 'idle' });
  
  console.log(`[+] Worker connected: ${clientId}`);
  
  ws.on('message', (data) => {
    const msg = JSON.parse(data);
    handleMessage(clientId, msg);
  });
  
  ws.on('close', () => {
    workers.delete(clientId);
    console.log(`[-] Worker disconnected: ${clientId}`);
  });
});

function handleMessage(clientId, msg) {
  const worker = workers.get(clientId);
  
  switch (msg.type) {
    case 'register':
      worker.role = msg.role;
      worker.status = 'ready';
      console.log(`[.] Worker ${clientId} registered as ${msg.role}`);
      break;
      
    case 'result':
      console.log(`[>] Result from ${clientId}:`, msg.data);
      broadcast({ type: 'result', from: clientId, data: msg.data });
      break;
      
    case 'status':
      worker.status = msg.status;
      break;
  }
}

function broadcast(msg) {
  workers.forEach((worker) => {
    if (worker.ws.readyState === WebSocket.OPEN) {
      worker.ws.send(JSON.stringify(msg));
    }
  });
}

// 分发任务给空闲从手机
function dispatchTask(task) {
  for (const [id, worker] of workers) {
    if (worker.status === 'idle') {
      worker.status = 'busy';
      worker.ws.send(JSON.stringify({ type: 'task', task }));
      return true;
    }
  }
  return false;
}

setInterval(() => {
  console.log(`[i] Active workers: ${workers.size}`);
}, 30000);
```

```bash
# 安装 ws
npm install ws

# 后台运行
node server.js > server.log 2>&1 &

# 开机自启 (Termux)
echo "cd ~/acpx-server && node server.js" >> ~/.bashrc
```

### 3.2 从手机 (Client)

```bash
# 安装 Node.js
pkg install nodejs

# 创建 client 目录
mkdir ~/acpx-client && cd ~/acpx-client

npm init -y
npm install ws
```

```javascript
// client.js
const WebSocket = require('ws');

// 配置主手机 IP
const MASTER_IP = process.env.MASTER_IP || '192.168.1.100';
const MASTER_PORT = process.env.MASTER_PORT || '8080';
const DEVICE_ID = process.env.DEVICE_ID || 'phone-' + Date.now();

const ws = new WebSocket(`ws://${MASTER_IP}:${MASTER_PORT}`);

console.log(`[Worker] Connecting to ${MASTER_IP}:${MASTER_PORT}...`);

ws.on('open', () => {
  console.log('[+] Connected to master');
  ws.send(JSON.stringify({
    type: 'register',
    role: process.env.ROLE || 'worker',
    deviceId: DEVICE_ID
  }));
});

ws.on('message', (data) => {
  const msg = JSON.parse(data);
  handleMessage(msg);
});

ws.on('close', () => {
  console.log('[-] Disconnected, reconnecting...');
  setTimeout(connect, 5000);
});

function handleMessage(msg) {
  switch (msg.type) {
    case 'task':
      console.log('[>] Received task:', msg.task);
      executeTask(msg.task);
      break;
  }
}

async function executeTask(task) {
  const { spawn } = require('child_process');
  
  console.log('[i] Executing task...');
  
  // 执行 Hermes/ACPX 命令
  const proc = spawn('hermes', [task.prompt], {
    cwd: process.env.HERMES_DIR || '/storage/shared/hermes'
  });
  
  let output = '';
  proc.stdout.on('data', (d) => output += d);
  proc.stderr.on('data', (d) => output += d);
  
  proc.on('close', (code) => {
    ws.send(JSON.stringify({
      type: 'result',
      deviceId: DEVICE_ID,
      task: task.prompt,
      output,
      exitCode: code
    }));
  });
}
```

```bash
# 配置并运行
MASTER_IP=192.168.1.100 ROLE=coder node client.js
```

---

## 4. 实施方案二：MQTT (进阶)

### 4.1 MQTT Broker (主手机)

```bash
# 安装 mosquitto (Android 需要 Termux)
pkg install mosquitto

# 配置
mkdir ~/.mosquitto && cd ~/.mosquitto

cat > mosquitto.conf << 'EOF'
listener 1883
allow_anonymous true
persistence false
EOF

# 运行 broker
mosquitto -c mosquitto.conf
```

### 4.2 MQTT Client

```javascript
// mqtt-client.js
const mqtt = require('mqtt');

const broker = process.env.MQTT_BROKER || 'mqtt://192.168.1.100:1883';
const deviceId = process.env.DEVICE_ID || 'phone-' + Date.now();

const client = mqtt.connect(broker);

const tasksTopic = 'hermes/tasks/' + deviceId;
const resultsTopic = 'hermes/results';
const statusTopic = 'hermes/status';

client.on('connect', () => {
  console.log('[+] MQTT connected');
  
  // 订阅自己任务
  client.subscribe(tasksTopic);
  client.subscribe('hermes/broadcast');
  
  // 状态在线
  client.publish(statusTopic, JSON.stringify({
    deviceId,
    status: 'online',
    role: process.env.ROLE || 'worker'
  }));
});

client.on('message', (topic, payload) => {
  const msg = JSON.parse(payload);
  
  if (topic === tasksTopic || topic === 'hermes/broadcast') {
    executeTask(msg);
  }
});

async function executeTask(task) {
  // 执行 Hermes
  const { exec } = require('child_process');
  
  exec(`hermes "${task.prompt}"`, (err, stdout, stderr) => {
    client.publish(resultsTopic, JSON.stringify({
      deviceId,
      task: task.prompt,
      result: stdout || stderr,
      timestamp: Date.now()
    }));
  });
}
```

```bash
npm install mqtt
MQTT_BROKER=mqtt://192.168.1.100 ROLE=coder node mqtt-client.js
```

---

## 5. 方案三：简单 HTTP

### 5.1 主手机 (REST API)

```javascript
// rest-server.js
const express = require('express');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

const workers = new Map();
const taskQueue = [];

// 报告 worker 在线
app.post('/register', (req, res) => {
  const { deviceId, role, ip } = req.body;
  workers.set(deviceId, { ip, role, status: 'idle' });
  res.json({ success: true });
});

// 提交任务
app.post('/task', (req, res) => {
  const { prompt, role } = req.body;
  
  // 找空闲 worker
  for (const [id, w] of workers) {
    if (role && w.role !== role) continue;
    if (w.status === 'idle') {
      w.status = 'busy';
      // 调用 worker
      // ...
      return res.json({ assignedTo: id });
    }
  }
  
  taskQueue.push({ prompt, role, time: Date.now() });
  res.json({ queued: true, position: taskQueue.length });
});

// 查询结果
app.get('/result/:taskId', (req, res) => {
  // 返回结果
});

app.listen(3000, () => console.log('[REST] Port 3000'));
```

```bash
npm install express
node rest-server.js
```

---

## 6. 网络配置

### 6.1 查找手机 IP

```bash
# 在 Termux 中
ifconfig  # 或
ip addr show
```

通常输出:
```
wlan0: inet 192.168.1.xxx
```

### 6.2 确保同一局域网

```bash
# 主手机 IP
192.168.1.100  (设为静态)

# 从手机 A
192.168.1.101

# 从手机 B  
192.168.1.102
```

### 6.3 防火墙

```bash
# 如果需要打开端口
# Android 可能需要 Termux:boot 权限
```

---

## 7. 安全考虑

### 7.1 认证

```javascript
// 简单 token 认证
const AUTH_TOKEN = 'your-secret-token';

function authenticate(msg) {
  return msg.token === AUTH_TOKEN;
}
```

### 7.2 内网隔离

- 不暴露到公网
- 仅局域网访问
- 考虑 VLAN

---

## 8. 启动脚本

### 8.1 主手机启动

```bash
#!/data/data/com.termux/files/home/scripts/start-master.sh
#!/bin/bash

cd ~/acpx-server
node server.js > ../logs/master.log 2>&1 &
echo $! > ../master.pid

echo "Master started PID: $(cat ../master.pid)"
```

### 8.2 从手机启动

```bash
#!/data/data/com.termux/files/home/scripts/start-worker.sh
#!/bin/bash

export MASTER_IP="${1:-192.168.1.100}"
export ROLE="${2:-worker}"

cd ~/acpx-client
node client.js > ../logs/worker.log 2>&1 &
echo $! > ../worker.pid

echo "Worker started connecting to $MASTER_IP"
```

### 8.3 Termux 开机自启

```bash
# 创建执行脚本
mkdir -p ~/.termux/boot
cat > ~/.termux/boot/start-cluster.sh << 'EOF'
#!/bin/bash
echo "[Boot] Starting cluster services..."

# 检查主从角色
if [ -f ~/.acpx-role ]; then
  ROLE=$(cat ~/.acpx-role)
  if [ "$ROLE" = "master" ]; then
    ~/scripts/start-master.sh
  else
    ~/scripts/start-worker.sh
  fi
fi
EOF

chmod +x ~/.termux/boot/start-cluster.sh
```

---

## 9. 测试

### 9.1 连接测试

```bash
# 主手机
nc -l 8080  # 监听

# 从手机
nc 192.168.1.100 8080  # 连接
```

### 9.2 消息测试

```bash
# 发送 JSON
echo '{"type":"test","msg":"hello"}' | nc -w 3 192.168.1.100 8080
```

---

## 10. 监控

### 10.1 状态检查

```bash
# 定期检查连接
while true; do
  curl -s http://localhost:3000/status
  sleep 30
done
```

### 10.2 日志

```bash
# 实时日志
tail -f ~/logs/worker.log
```

---

## 11. 常见问题

| 问题 | 解决 |
|------|------|
| 连接不上 | 检查 IP + 防火墙 |
| 频繁断线 | 心跳重连 |
| 任务卡住 | 超时杀进程 |
| 编码问题 | UTF-8 |