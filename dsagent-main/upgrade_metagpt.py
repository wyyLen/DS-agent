"""
DSAgent升级MetaGPT到最新版本的脚本

此脚本会：
1. 备份当前的metagpt目录
2. 从GitHub克隆最新的MetaGPT
3. 保留DSAgent特有的修改
4. 整合新旧代码
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
METAGPT_DIR = PROJECT_ROOT / "metagpt"
BACKUP_DIR = PROJECT_ROOT / f"metagpt_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TEMP_DIR = PROJECT_ROOT / "metagpt_new_temp"

# MetaGPT GitHub仓库
METAGPT_REPO = "https://github.com/geekan/MetaGPT.git"

# DSAgent特有的文件和目录（需要保留）
DSAGENT_SPECIFIC = [
    "metagpt/roles/ds_agent/",
    "metagpt/actions/ds_agent/",
    "metagpt/rag/engines/customMixture.py",
    "metagpt/rag/engines/customWorkflowGM.py",
    "metagpt/rag/engines/customSolutionSamplesGenerate.py",
    "metagpt/rag/engines/customEmbeddingComparisonEngine.py",
    "metagpt/rag/engines/GraphMatching/",
    "metagpt/rag/engines/graphUtils.py",
    "metagpt/strategy/ds_planner.py",
    "metagpt/strategy/ds_task_type.py",
    "metagpt/strategy/lats_react.py",
    "metagpt/tools/tool_recommend.py",
]


def print_step(step_num, message):
    """打印步骤信息"""
    print(f"\n{'='*80}")
    print(f"步骤 {step_num}: {message}")
    print('='*80)


def backup_current_metagpt():
    """备份当前的MetaGPT目录"""
    print_step(1, "备份当前MetaGPT目录")
    
    if METAGPT_DIR.exists():
        print(f"正在备份 {METAGPT_DIR} 到 {BACKUP_DIR}...")
        shutil.copytree(METAGPT_DIR, BACKUP_DIR)
        print(f"✓ 备份完成: {BACKUP_DIR}")
    else:
        print("⚠️ MetaGPT目录不存在，跳过备份")


def clone_latest_metagpt():
    """克隆最新的MetaGPT代码"""
    print_step(2, "从GitHub克隆最新MetaGPT")
    
    # 删除临时目录（如果存在）
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    
    print(f"正在克隆 {METAGPT_REPO}...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", METAGPT_REPO, str(TEMP_DIR)],
            check=True,
            cwd=PROJECT_ROOT
        )
        print("✓ 克隆完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 克隆失败: {e}")
        sys.exit(1)


def extract_dsagent_files():
    """提取DSAgent特有的文件"""
    print_step(3, "提取DSAgent特有的文件")
    
    dsagent_backup = PROJECT_ROOT / "dsagent_files_backup"
    if dsagent_backup.exists():
        shutil.rmtree(dsagent_backup)
    dsagent_backup.mkdir()
    
    extracted_files = []
    
    for path_str in DSAGENT_SPECIFIC:
        source_path = BACKUP_DIR / path_str
        
        if not source_path.exists():
            print(f"  ⚠️  未找到: {path_str}")
            continue
        
        # 计算相对路径
        rel_path = Path(path_str).relative_to("metagpt")
        dest_path = dsagent_backup / rel_path
        
        if source_path.is_dir():
            print(f"  复制目录: {path_str}")
            shutil.copytree(source_path, dest_path)
        else:
            print(f"  复制文件: {path_str}")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
        
        extracted_files.append(path_str)
    
    print(f"\n✓ 提取了 {len(extracted_files)} 个DSAgent特有文件/目录")
    return dsagent_backup


def replace_metagpt():
    """替换旧的MetaGPT为新版本"""
    print_step(4, "替换MetaGPT目录")
    
    # 删除旧的metagpt目录
    if METAGPT_DIR.exists():
        print(f"删除旧版本: {METAGPT_DIR}")
        shutil.rmtree(METAGPT_DIR)
    
    # 复制新的metagpt目录（只复制metagpt子目录）
    new_metagpt_source = TEMP_DIR / "metagpt"
    if new_metagpt_source.exists():
        print(f"复制新版本: {new_metagpt_source} -> {METAGPT_DIR}")
        shutil.copytree(new_metagpt_source, METAGPT_DIR)
        print("✓ 替换完成")
    else:
        print("❌ 新版本的metagpt目录不存在")
        sys.exit(1)


def merge_dsagent_files(dsagent_backup):
    """将DSAgent特有文件合并到新的MetaGPT"""
    print_step(5, "合并DSAgent特有文件")
    
    # 遍历备份的DSAgent文件
    for item in dsagent_backup.rglob("*"):
        if item.is_file():
            # 计算相对路径
            rel_path = item.relative_to(dsagent_backup)
            dest_path = METAGPT_DIR / rel_path
            
            # 创建目标目录
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            print(f"  合并: {rel_path}")
            shutil.copy2(item, dest_path)
    
    print("✓ 合并完成")


def update_const_file():
    """更新const.py文件，添加DSAgent特有的常量"""
    print_step(6, "更新常量定义")
    
    const_file = METAGPT_DIR / "const.py"
    
    if not const_file.exists():
        print("⚠️ const.py不存在，跳过更新")
        return
    
    # 读取当前内容
    with open(const_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经有DSAgent相关常量
    if "EXP_PLAN" in content:
        print("  const.py已包含DSAgent常量，跳过")
        return
    
    # 添加DSAgent常量
    dsagent_constants = '''

# DSAgent specific constants
EXP_PLAN = EXAMPLE_DATA_PATH / "exp_bank/plan_exp.json"
WORKFLOW_EXP = EXAMPLE_DATA_PATH / "exp_bank/workflow_exp2_clean.json"
AGENT_SERVICE_FILE = EXAMPLE_PATH / "ds_agent" / "agent_service" / "uploads"
'''
    
    with open(const_file, 'a', encoding='utf-8') as f:
        f.write(dsagent_constants)
    
    print("✓ 已添加DSAgent常量到const.py")


def cleanup():
    """清理临时文件"""
    print_step(7, "清理临时文件")
    
    if TEMP_DIR.exists():
        print(f"删除临时目录: {TEMP_DIR}")
        shutil.rmtree(TEMP_DIR)
    
    print("✓ 清理完成")


def create_upgrade_report():
    """创建升级报告"""
    print_step(8, "生成升级报告")
    
    report_file = PROJECT_ROOT / "METAGPT_UPGRADE_REPORT.md"
    
    report_content = f"""# MetaGPT升级报告

**升级时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 升级内容

### 1. 备份位置
- 旧版本备份: `{BACKUP_DIR.name}`
- DSAgent文件备份: `dsagent_files_backup/`

### 2. 更新内容
- ✓ 从GitHub克隆最新MetaGPT代码
- ✓ 保留DSAgent特有的文件和目录
- ✓ 合并新旧代码

### 3. DSAgent特有文件（已保留）

{chr(10).join(f'- {path}' for path in DSAGENT_SPECIFIC)}

### 4. 新增/更新的常量

在 `metagpt/const.py` 中添加了DSAgent特有常量：
- `EXP_PLAN`: 文本经验库路径
- `WORKFLOW_EXP`: 工作流经验库路径  
- `AGENT_SERVICE_FILE`: Agent服务上传目录

## 验证步骤

请执行以下步骤验证升级：

1. **检查导入**
   ```bash
   python -c "from dsagent_core.roles.ds_agent_stream import DSAgentStream; print('✓ DSAgent导入成功')"
   ```

2. **运行集成测试**
   ```bash
   python test_metagpt_integration.py
   ```

3. **启动完整系统**
   ```bash
   python start_dsagent_system.py
   ```

## 回滚方案

如果升级后出现问题，可以回滚到旧版本：

```bash
# 删除新版本
rm -rf metagpt

# 恢复备份
cp -r {BACKUP_DIR.name} metagpt
```

## 注意事项

1. **DSAgent Core适配器**: 新的MetaGPT版本应该与dsagent_core包完全兼容
2. **API变化**: 如果MetaGPT有破坏性更新，可能需要调整DSAgent代码
3. **依赖检查**: 运行 `pip install -r requirements.txt` 确保依赖完整

## 后续步骤

- [ ] 测试文本经验检索功能
- [ ] 测试工作流经验检索功能  
- [ ] 测试树搜索功能
- [ ] 测试前端UI交互
- [ ] 更新文档

---
*此报告由升级脚本自动生成*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✓ 升级报告已生成: {report_file}")


def main():
    """主函数"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              DSAgent - MetaGPT 升级工具                                     ║
║              将内嵌的MetaGPT更新到GitHub最新版本                             ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # 确认操作
    print("⚠️  警告: 此操作将替换当前的MetaGPT代码")
    print(f"   旧版本将备份到: {BACKUP_DIR.name}")
    print()
    
    confirm = input("是否继续? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("操作已取消")
        return
    
    try:
        # 执行升级步骤
        backup_current_metagpt()
        clone_latest_metagpt()
        dsagent_backup = extract_dsagent_files()
        replace_metagpt()
        merge_dsagent_files(dsagent_backup)
        update_const_file()
        cleanup()
        create_upgrade_report()
        
        print("\n" + "="*80)
        print("🎉 MetaGPT升级完成！")
        print("="*80)
        print()
        print("下一步:")
        print("  1. 查看升级报告: METAGPT_UPGRADE_REPORT.md")
        print("  2. 运行测试: python test_metagpt_integration.py")
        print("  3. 启动系统: python start_dsagent_system.py")
        print()
        
    except Exception as e:
        print(f"\n❌ 升级过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n尝试回滚...")
        if BACKUP_DIR.exists() and METAGPT_DIR.exists():
            shutil.rmtree(METAGPT_DIR)
            shutil.copytree(BACKUP_DIR, METAGPT_DIR)
            print("✓ 已回滚到旧版本")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
