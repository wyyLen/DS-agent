# MetaGPT Backend 启动脚本
# 激活 MetaGPT 虚拟环境并启动后端

Write-Host "激活 MetaGPT 虚拟环境..." -ForegroundColor Cyan
.\venv_metagpt\Scripts\Activate.ps1

Write-Host "设置框架为 MetaGPT..." -ForegroundColor Cyan
$env:AGENT_FRAMEWORK='metagpt'

Write-Host "启动 MetaGPT 后端..." -ForegroundColor Green
python examples\ds_agent\agent_service\start_backend.py
